"""
层次化 MoE 融合层（FR-5）
保留 PDF 全息权重核心思想，升级为两层可学习 MoE 路由：
  Layer 1：模态路由（3 专家：text_dominant / video_dominant / balanced）
  Layer 2：类别路由（n_classes 个专家，细化权重）
最终权重：Softmax(Layer1_out + 0.3 * Layer2_adjust)，保证和为 1。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalMoEFusion(nn.Module):
    """层次化 MoE 融合层，替代硬编码全息权重"""

    def __init__(self, args):
        super().__init__()
        hidden_sz = getattr(args, "hidden_sz", 768)
        video_dim = getattr(args, "video_feature_dim", 512)
        n_classes = getattr(args, "n_classes", 10)
        dropout = getattr(args, "dropout", 0.3)
        self.adjustment_weight = getattr(args, "fusion_adjustment_weight", 0.3)
        self.n_classes = n_classes

        # ---------- Layer 1：模态路由（3 专家）----------
        # 输入：[text_f, video_f, text_conf, video_conf, text_holo, video_holo]
        l1_input_dim = hidden_sz + video_dim + 4

        self.modality_router = nn.Sequential(
            nn.Linear(l1_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

        # 3 个模态专家，偏置用不同初始化表达偏好
        expert_bias_inits = [
            torch.tensor([0.7, 0.3]),   # text_dominant
            torch.tensor([0.3, 0.7]),   # video_dominant
            torch.tensor([0.5, 0.5]),   # balanced
        ]
        self.modality_experts = nn.ModuleList()
        for bias_init in expert_bias_inits:
            expert = nn.Sequential(
                nn.Linear(l1_input_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 2),
            )
            with torch.no_grad():
                expert[-1].bias.copy_(bias_init)
            self.modality_experts.append(expert)

        # ---------- Layer 2：类别路由（n_classes 专家）----------
        # 输入：[text_f, video_f, layer1_weights(2)]
        l2_input_dim = hidden_sz + video_dim + 2

        self.class_router = nn.Sequential(
            nn.Linear(l2_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )
        self.class_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(l2_input_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 2),  # 输出调整量 [Δw_text, Δw_video]
            )
            for _ in range(n_classes)
        ])

    # ------------------------------------------------------------------
    def forward(self, text_features, video_features, text_conf, video_conf,
                class_logits, return_routing=False):
        """
        Args:
            text_features:  [B, hidden_sz]
            video_features: [B, video_dim]
            text_conf:      [B, 1]
            video_conf:     [B, 1]
            class_logits:   [B, n_classes]
            return_routing: bool

        Returns:
            w_text:  [B, 1]
            w_video: [B, 1]
            routing_info: Dict（可选）
        """
        eps = 1e-8

        # 1. 计算 PDF 全息权重（保留核心思想，作为 Layer 1 输入特征）
        tc = text_conf.clamp(min=eps)
        vc = video_conf.clamp(min=eps)
        prod = (tc * vc).clamp(min=eps)
        text_holo = torch.log(vc) / (torch.log(prod) + eps)
        video_holo = torch.log(tc) / (torch.log(prod) + eps)

        # 2. Layer 1：模态路由
        l1_input = torch.cat(
            [text_features, video_features, text_conf, video_conf,
             text_holo, video_holo], dim=-1
        )  # [B, l1_input_dim]

        modality_weights = F.softmax(self.modality_router(l1_input), dim=-1)  # [B, 3]

        expert_outs = torch.stack(
            [exp(l1_input) for exp in self.modality_experts], dim=1
        )  # [B, 3, 2]

        # 加权聚合
        layer1_out = torch.einsum("bn,bnd->bd", modality_weights, expert_outs)  # [B, 2]

        # 3. Layer 2：类别路由
        l2_input = torch.cat([text_features, video_features, layer1_out], dim=-1)

        class_probs = F.softmax(class_logits, dim=-1)  # [B, n_classes]

        class_expert_outs = torch.stack(
            [exp(l2_input) for exp in self.class_experts], dim=1
        )  # [B, n_classes, 2]

        layer2_adjust = torch.einsum("bn,bnd->bd", class_probs, class_expert_outs)  # [B, 2]

        # 4. 最终权重
        final_weights = F.softmax(
            layer1_out + self.adjustment_weight * layer2_adjust, dim=-1
        )  # [B, 2]

        w_text = final_weights[:, 0:1]
        w_video = final_weights[:, 1:2]

        if return_routing:
            routing_info = {
                "holographic_weights": torch.cat([text_holo, video_holo], dim=-1),
                "layer1_modality_weights": modality_weights,
                "layer1_output": layer1_out,
                "layer2_class_probs": class_probs,
                "layer2_adjustments": layer2_adjust,
                "final_weights": final_weights,
            }
            return w_text, w_video, routing_info

        return w_text, w_video
