#!/usr/bin/env python3
"""
视频+文本多模态动态融合分类模型（改进版 v2）
在原始版本基础上集成六大核心改进：
  FR-1: TriModalTextEncoder - 三模态文本编码
  FR-3: MambaVideoEncoder   - Mamba 视频编码
  FR-4: SparseMoEConfidNet  - 稀疏 MoE 置信度网络
  FR-5: HierarchicalMoEFusion - 层次化 MoE 融合
  FR-6: ProcessRewardModel  - 过程奖励模型

兼容原始接口，通过 args 标志控制是否启用新模块。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.models.bert import BertClf
from src.models.video import VideoClf

# ---------- 新增模块（按需启用）----------
# 使用小写变量名避免 linter 将其视作常量
_trimodal_ok: bool = False
_mamba_ok: bool = False
_moe_confid_ok: bool = False
_hier_fusion_ok: bool = False
_prm_ok: bool = False

# 使用 Optional 类型声明，避免"可能未绑定"错误
_TriModalTextEncoder: Optional[type] = None
_MambaVideoEncoder: Optional[type] = None
_SparseMoEConfidNet: Optional[type] = None
_HierarchicalMoEFusion: Optional[type] = None
_ProcessRewardModel: Optional[type] = None

try:
    from src.models.trimodal_encoder import TriModalTextEncoder as _TriModalTextEncoder  # type: ignore[assignment]
    _trimodal_ok = True
except Exception:
    pass

try:
    from src.models.mamba_encoder import MambaVideoEncoder as _MambaVideoEncoder  # type: ignore[assignment]
    _mamba_ok = True
except Exception:
    pass

try:
    from src.models.sparse_moe import SparseMoEConfidNet as _SparseMoEConfidNet  # type: ignore[assignment]
    _moe_confid_ok = True
except Exception:
    pass

try:
    from src.models.hierarchical_fusion import HierarchicalMoEFusion as _HierarchicalMoEFusion  # type: ignore[assignment]
    _hier_fusion_ok = True
except Exception:
    pass

try:
    from src.models.prm import ProcessRewardModel as _ProcessRewardModel  # type: ignore[assignment]
    _prm_ok = True
except Exception:
    pass


# ==========================================================================
# 原有模块（保留，向后兼容）
# ==========================================================================

class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""

    def __init__(self, text_dim, video_dim, hidden_dim=256):
        super().__init__()
        self.text_to_video_query = nn.Linear(text_dim, hidden_dim)
        self.text_to_video_key = nn.Linear(video_dim, hidden_dim)
        self.text_to_video_value = nn.Linear(video_dim, hidden_dim)

        self.video_to_text_query = nn.Linear(video_dim, hidden_dim)
        self.video_to_text_key = nn.Linear(text_dim, hidden_dim)
        self.video_to_text_value = nn.Linear(text_dim, hidden_dim)

        self.text_output_proj = nn.Linear(hidden_dim, text_dim)
        self.video_output_proj = nn.Linear(hidden_dim, video_dim)

        self.dropout = nn.Dropout(0.1)
        self.layer_norm_text = nn.LayerNorm(text_dim)
        self.layer_norm_video = nn.LayerNorm(video_dim)

    def forward(self, text_features, video_features):
        # 文本→视频
        tq = self.text_to_video_query(text_features)
        vk = self.text_to_video_key(video_features)
        vv = self.text_to_video_value(video_features)
        scale = tq.size(-1) ** -0.5
        # squeeze(-1) 替代 squeeze()，避免 B=1 时 batch 维被消除；移除跨 batch 的错误 softmax
        t_att = torch.matmul(tq.unsqueeze(1), vk.unsqueeze(2)).squeeze(-1) * scale  # [B, 1]
        enhanced_t = self.layer_norm_text(
            text_features + self.dropout(self.text_output_proj(t_att * vv))
        )

        # 视频→文本
        vq = self.video_to_text_query(video_features)
        tk = self.video_to_text_key(text_features)
        tv = self.video_to_text_value(text_features)
        v_att = torch.matmul(vq.unsqueeze(1), tk.unsqueeze(2)).squeeze(-1) * scale  # [B, 1]
        enhanced_v = self.layer_norm_video(
            video_features + self.dropout(self.video_output_proj(v_att * tv))
        )

        return enhanced_t, enhanced_v


class PrototypeNetwork(nn.Module):
    """原型网络，增强小样本类别学习"""

    def __init__(self, feature_dim, n_classes, prototype_dim=128):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, prototype_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(prototype_dim, prototype_dim),
        )
        self.prototypes = nn.Parameter(torch.randn(n_classes, prototype_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, features):
        proj = F.normalize(self.feature_proj(features), p=2, dim=1)
        nproto = F.normalize(self.prototypes, p=2, dim=1)
        sims = torch.matmul(proj, nproto.t())
        return sims * 10.0, proj


# ==========================================================================
# 改进版主模型
# ==========================================================================

class MultimodalLateFusionClf_video(nn.Module):
    """
    视频+文本多模态动态融合分类器（改进版 v2）

    通过 args 标志渐进式启用新模块：
      args.use_trimodal      (bool)  启用三模态文本编码器 (FR-1)
      args.use_mamba         (bool)  启用 Mamba 视频编码器 (FR-3)
      args.use_sparse_moe    (bool)  启用 Sparse MoE ConfidNet (FR-4)
      args.use_hier_fusion   (bool)  启用层次化 MoE 融合 (FR-5)
      args.use_prm           (bool)  启用 Process Reward Model (FR-6)
      args.use_video_desc    (bool)  是否使用 video_description（默认True）
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_classes = args.n_classes
        self.use_video_desc = getattr(args, "use_video_desc", True)

        # ---- 文本编码器 ----
        use_trimodal = getattr(args, "use_trimodal", False) and _trimodal_ok
        if use_trimodal and _TriModalTextEncoder is not None:
            self.trimodal_encoder: Optional[nn.Module] = _TriModalTextEncoder(args)
        else:
            self.trimodal_encoder = None
        self.txtclf = BertClf(args)

        # ---- 视频编码器 ----
        use_mamba = getattr(args, "use_mamba", False) and _mamba_ok
        if use_mamba and _MambaVideoEncoder is not None:
            self.mamba_encoder: Optional[nn.Module] = _MambaVideoEncoder(args)
        else:
            self.mamba_encoder = None
        self.videoclf = VideoClf(args)

        # ---- 跨模态注意力（原有，保留）----
        self.cross_modal_attention = CrossModalAttention(
            text_dim=args.hidden_sz,
            video_dim=args.video_feature_dim,
            hidden_dim=256,
        )

        # ---- 原型网络（原有，保留）----
        self.text_prototype_net = PrototypeNetwork(args.hidden_sz, self.n_classes, 128)
        self.video_prototype_net = PrototypeNetwork(args.video_feature_dim, self.n_classes, 128)

        # ---- 置信度网络 ----
        use_sparse_moe = getattr(args, "use_sparse_moe", False) and _moe_confid_ok
        if use_sparse_moe and _SparseMoEConfidNet is not None:
            self.sparse_moe_confid: Optional[nn.Module] = _SparseMoEConfidNet(args)
            self.ConfidNet_txt_class: Optional[nn.ModuleList] = None
            self.ConfidNet_video_class: Optional[nn.ModuleList] = None
        else:
            self.sparse_moe_confid = None
            self.ConfidNet_txt_class = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(args.hidden_sz, args.hidden_sz),
                    nn.ReLU(), nn.Dropout(args.dropout),
                    nn.Linear(args.hidden_sz, args.hidden_sz // 2),
                    nn.ReLU(), nn.Dropout(args.dropout),
                    nn.Linear(args.hidden_sz // 2, args.hidden_sz // 4),
                    nn.ReLU(), nn.Dropout(args.dropout),
                    nn.Linear(args.hidden_sz // 4, 1),
                    nn.Sigmoid(),
                ) for _ in range(self.n_classes)
            ])
            self.ConfidNet_video_class = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(args.video_feature_dim, args.video_feature_dim),
                    nn.ReLU(), nn.Dropout(args.dropout),
                    nn.Linear(args.video_feature_dim, args.video_feature_dim // 2),
                    nn.ReLU(), nn.Dropout(args.dropout),
                    nn.Linear(args.video_feature_dim // 2, args.video_feature_dim // 4),
                    nn.ReLU(), nn.Dropout(args.dropout),
                    nn.Linear(args.video_feature_dim // 4, 1),
                    nn.Sigmoid(),
                ) for _ in range(self.n_classes)
            ])

        # ---- 融合层 ----
        use_hier_fusion = getattr(args, "use_hier_fusion", False) and _hier_fusion_ok
        if use_hier_fusion and _HierarchicalMoEFusion is not None:
            self.hier_fusion: Optional[nn.Module] = _HierarchicalMoEFusion(args)
        else:
            self.hier_fusion = None

        # ---- PRM ----
        use_prm = getattr(args, "use_prm", False) and _prm_ok
        if use_prm and _ProcessRewardModel is not None:
            self.prm: Optional[nn.Module] = _ProcessRewardModel(args)
            self.prm_adjustment_weight = getattr(args, "prm_adjustment_weight", 0.2)
        else:
            self.prm = None
            self.prm_adjustment_weight = 0.2

        # ---- 类别预测器 & 类别感知增强（原有）----
        combined = args.hidden_sz + args.video_feature_dim
        self.class_predictor = nn.Sequential(
            nn.Linear(combined, combined // 2), nn.ReLU(), nn.Dropout(args.dropout),
            nn.Linear(combined // 2, combined // 4), nn.ReLU(), nn.Dropout(args.dropout),
            nn.Linear(combined // 4, self.n_classes), nn.Softmax(dim=1),
        )
        self.class_aware_text_enhancer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(args.hidden_sz, args.hidden_sz // 2), nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_sz // 2, args.hidden_sz), nn.Tanh(),
            ) for _ in range(self.n_classes)
        ])
        self.class_aware_video_enhancer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(args.video_feature_dim, args.video_feature_dim // 2), nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.video_feature_dim // 2, args.video_feature_dim), nn.Tanh(),
            ) for _ in range(self.n_classes)
        ])

        self._init_networks()

    # ------------------------------------------------------------------
    def _init_networks(self):
        for module_list in [self.class_aware_text_enhancer,
                             self.class_aware_video_enhancer]:
            for net in module_list:
                for layer in net:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
        if self.ConfidNet_txt_class is not None:
            for net in self.ConfidNet_txt_class:
                for layer in net:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
        if self.ConfidNet_video_class is not None:
            for net in self.ConfidNet_video_class:
                for layer in net:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
        for layer in self.class_predictor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------
    def forward(self, txt, mask, segment, video_frames, choice="train",
                text_list=None, video_desc_list=None):
        """
        Args:
            txt, mask, segment: BERT 输入（当 use_trimodal=False 时使用）
            video_frames: [B, T, 3, H, W]
            choice: "train" / "test" / "static"
            text_list, video_desc_list: 原始字符串列表（use_trimodal=True 时使用）
        """
        # ---- 文本特征提取 ----
        if self.trimodal_encoder is not None and text_list is not None and self.use_video_desc:
            # 使用 TriModalTextEncoder 并包含 video_description
            txt_f = self.trimodal_encoder(text_list, video_desc_list)
            txt_out = self.txtclf.clf(txt_f)
        else:
            # 使用原始的 BertEncoder（保持向后兼容）
            txt_out, txt_f = self.txtclf(txt, mask, segment)

        # ---- 视频特征提取 ----
        if self.mamba_encoder is not None:
            video_f = self.mamba_encoder(video_frames)
            video_out = self.videoclf.clf(video_f)
        else:
            video_out, video_f = self.videoclf(video_frames)

        # ---- 跨模态注意力增强 ----
        enhanced_txt_f, enhanced_video_f = self.cross_modal_attention(txt_f, video_f)
        enhanced_txt_out = self.txtclf.clf(enhanced_txt_f)
        enhanced_video_out = self.videoclf.clf(enhanced_video_f)

        # ---- 原型输出 ----
        txt_proto_out, _ = self.text_prototype_net(enhanced_txt_f)
        video_proto_out, _ = self.video_prototype_net(enhanced_video_f)

        if self.args.use_dynamic_fusion:
            if choice == "train":
                return self._forward_train_v2(
                    enhanced_txt_out, enhanced_txt_f,
                    enhanced_video_out, enhanced_video_f,
                    txt_proto_out, video_proto_out,
                )
            elif choice == "test":
                return self._forward_test_v2(
                    enhanced_txt_out, enhanced_txt_f,
                    enhanced_video_out, enhanced_video_f,
                    txt_proto_out, video_proto_out,
                )

        # 静态融合
        return 0.5 * enhanced_txt_out + 0.5 * enhanced_video_out, enhanced_txt_out, enhanced_video_out

    # ------------------------------------------------------------------
    def _compute_confidence(self, txt_f, video_f, class_probs):
        """计算文本和视频置信度（支持 Sparse MoE 或原始类别特定网络）"""
        if self.sparse_moe_confid is not None:
            txt_conf, video_conf, routing_info = self.sparse_moe_confid(
                txt_f, video_f, return_routing=True
            )
        else:
            # 原有逻辑：类别特定置信度加权平均
            assert self.ConfidNet_txt_class is not None
            assert self.ConfidNet_video_class is not None
            txt_confs = torch.stack(
                [self.ConfidNet_txt_class[i](txt_f) for i in range(self.n_classes)],
                dim=2,
            )  # [B, 1, n_classes]
            video_confs = torch.stack(
                [self.ConfidNet_video_class[i](video_f) for i in range(self.n_classes)],
                dim=2,
            )  # [B, 1, n_classes]
            cp = class_probs.unsqueeze(1)  # [B, 1, n_classes]
            txt_conf = (txt_confs * cp).sum(dim=2)    # [B, 1]
            video_conf = (video_confs * cp).sum(dim=2) # [B, 1]
            routing_info = {}
        return txt_conf, video_conf, routing_info

    def _compute_fusion_weights(self, txt_f, video_f, txt_conf, video_conf,
                                class_logits, video_logits, routing_info, is_test=False):
        """计算最终融合权重（支持层次化 MoE 或原始全息权重）"""
        if self.hier_fusion is not None:
            w_text, w_video, fusion_routing = self.hier_fusion(
                txt_f, video_f, txt_conf, video_conf,
                class_logits, return_routing=True,
            )
            routing_info.update(fusion_routing)
        else:
            # 原有全息权重逻辑
            eps = 1e-8
            txt_holo = torch.log(video_conf + eps) / (torch.log(txt_conf * video_conf + eps) + eps)
            video_holo = torch.log(txt_conf + eps) / (torch.log(txt_conf * video_conf + eps) + eps)
            combined_txt = txt_conf + txt_holo
            combined_video = video_conf + video_holo

            if is_test:
                # 不确定性调整：分别用文本 logits 和视频 logits 计算各自不确定性
                txt_pred = F.softmax(class_logits, dim=1)
                video_pred = F.softmax(video_logits, dim=1)
                uniform = 1.0 / class_logits.size(1)
                txt_unc = (txt_pred - uniform).abs().mean(dim=1, keepdim=True)
                vid_unc = (video_pred - uniform).abs().mean(dim=1, keepdim=True)
                cond = txt_unc > vid_unc
                t_adj = torch.where(cond, torch.ones_like(txt_unc), txt_unc / vid_unc)
                v_adj = torch.where(cond, vid_unc / txt_unc, torch.ones_like(vid_unc))
                combined_txt = combined_txt * t_adj
                combined_video = combined_video * v_adj

            weights = F.softmax(torch.cat([combined_txt, combined_video], dim=-1), dim=-1)
            w_text = weights[:, 0:1]
            w_video = weights[:, 1:2]

        return w_text, w_video, routing_info

    # ------------------------------------------------------------------
    def _forward_train_v2(self, txt_out, txt_f, video_out, video_f,
                           txt_proto_out, video_proto_out):
        """
        改进版训练前向传播。
        返回：(final_fused, txt_out, video_out, txt_confs, video_confs,
                class_probs, txt_proto_out, video_proto_out)
        """
        txt_f_d = txt_f.detach()
        video_f_d = video_f.detach()

        # 类别预测 + 类别感知增强
        class_probs = self.class_predictor(torch.cat([txt_f_d, video_f_d], dim=-1))
        txt_f_enh, video_f_enh = self._class_aware_enhance(txt_f_d, video_f_d, class_probs)

        # 置信度
        txt_conf, video_conf, routing_info = self._compute_confidence(
            txt_f_enh, video_f_enh, class_probs
        )

        # 融合权重
        w_text, w_video, routing_info = self._compute_fusion_weights(
            txt_f_enh, video_f_enh, txt_conf, video_conf, txt_out, video_out, routing_info
        )

        # 主融合
        main_fused = w_text.detach() * txt_out + w_video.detach() * video_out
        proto_fused = 0.5 * txt_proto_out + 0.5 * video_proto_out
        final_fused = 0.8 * main_fused + 0.2 * proto_fused

        # PRM 调整
        if self.prm is not None:
            fusion_weights = torch.cat([w_text, w_video], dim=-1)
            adjustment, _ = self.prm(
                txt_f_enh, video_f_enh, routing_info,
                fusion_weights, txt_conf, video_conf,
            )
            final_fused = final_fused + self.prm_adjustment_weight * adjustment

        # 构造兼容原始接口的置信度张量
        txt_confs = txt_conf.unsqueeze(-1).expand(-1, 1, self.n_classes)
        video_confs = video_conf.unsqueeze(-1).expand(-1, 1, self.n_classes)

        return (final_fused, txt_out, video_out, txt_confs, video_confs,
                class_probs, txt_proto_out, video_proto_out)

    # ------------------------------------------------------------------
    def _forward_test_v2(self, txt_out, txt_f, video_out, video_f,
                          txt_proto_out, video_proto_out):
        """
        改进版测试前向传播。
        返回：(final_fused, txt_out, video_out, txt_confs, video_confs,
                w_text, w_video, class_probs, txt_proto_out, video_proto_out)
        """
        class_probs = self.class_predictor(torch.cat([txt_f, video_f], dim=-1))
        txt_f_enh, video_f_enh = self._class_aware_enhance(txt_f, video_f, class_probs)

        txt_conf, video_conf, routing_info = self._compute_confidence(
            txt_f_enh, video_f_enh, class_probs
        )

        w_text, w_video, routing_info = self._compute_fusion_weights(
            txt_f_enh, video_f_enh, txt_conf, video_conf, txt_out, video_out, routing_info,
            is_test=True,
        )

        main_fused = w_text.detach() * txt_out + w_video.detach() * video_out
        proto_fused = 0.5 * txt_proto_out + 0.5 * video_proto_out
        final_fused = 0.8 * main_fused + 0.2 * proto_fused

        if self.prm is not None:
            fusion_weights = torch.cat([w_text, w_video], dim=-1)
            adjustment, prm_scores = self.prm(
                txt_f_enh, video_f_enh, routing_info,
                fusion_weights, txt_conf, video_conf,
            )
            final_fused = final_fused + self.prm_adjustment_weight * adjustment
            # prm_scores 可由调用者从模型外访问（不放入 return tuple 保持兼容）
            self._last_prm_scores = prm_scores

        txt_confs = txt_conf.unsqueeze(-1).expand(-1, 1, self.n_classes)
        video_confs = video_conf.unsqueeze(-1).expand(-1, 1, self.n_classes)

        return (final_fused, txt_out, video_out, txt_confs, video_confs,
                w_text, w_video, class_probs, txt_proto_out, video_proto_out)

    # ------------------------------------------------------------------
    def _class_aware_enhance(self, txt_f, video_f, class_probs):
        """类别感知特征增强（原有逻辑提取为公共方法）"""
        txt_parts, vid_parts = [], []
        for i in range(self.n_classes):
            w = class_probs[:, i:i+1]
            txt_parts.append(w * self.class_aware_text_enhancer[i](txt_f))
            vid_parts.append(w * self.class_aware_video_enhancer[i](video_f))
        enh_txt = torch.stack(txt_parts, dim=0).sum(dim=0)
        enh_vid = torch.stack(vid_parts, dim=0).sum(dim=0)
        return txt_f + 0.3 * enh_txt, video_f + 0.3 * enh_vid

    # ------------------------------------------------------------------
    # 以下保留原有方法名，供旧训练脚本调用（委托到 v2 版本）
    # ------------------------------------------------------------------
    def _forward_train_enhanced(self, txt_out, txt_f, video_out, video_f,
                                 txt_proto_out, video_proto_out):
        return self._forward_train_v2(txt_out, txt_f, video_out, video_f,
                                       txt_proto_out, video_proto_out)

    def _forward_test_enhanced(self, txt_out, txt_f, video_out, video_f,
                                txt_proto_out, video_proto_out):
        return self._forward_test_v2(txt_out, txt_f, video_out, video_f,
                                      txt_proto_out, video_proto_out)
