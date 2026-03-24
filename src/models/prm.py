"""
Process Reward Model（FR-6）
五步过程监督：文本质量、视频质量、对齐度、路由决策、融合效果。
每步输出 0-1 评分，聚合后对最终 logits 作微调。
"""

import torch
import torch.nn as nn


def _build_scorer(input_dim: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )


class ProcessRewardModel(nn.Module):
    """
    过程奖励模型，提供五步监督信号并生成 logits 调整量。

    使用方法：
      adjustment, scores = prm(text_f, video_f, routing_info,
                               fusion_weights, text_conf, video_conf)
      final_logits = main_logits + 0.2 * adjustment
    """

    def __init__(self, args):
        super().__init__()
        hidden_sz = getattr(args, "hidden_sz", 768)
        video_dim = getattr(args, "video_feature_dim", 512)
        n_classes = getattr(args, "n_classes", 10)
        dropout = getattr(args, "dropout", 0.3)

        # 五个评分器，输入维度各不同
        self.text_quality_scorer = _build_scorer(hidden_sz, dropout)
        self.video_quality_scorer = _build_scorer(video_dim, dropout)
        self.alignment_scorer = _build_scorer(hidden_sz + video_dim, dropout)
        # routing 输入：text_f + video_f + text_top2_w(2) + video_top2_w(2) + modality_w(3)
        self.routing_scorer = _build_scorer(hidden_sz + video_dim + 7, dropout)
        # fusion 输入：text_f + video_f + fusion_weights(2) + text_conf(1) + video_conf(1)
        self.fusion_scorer = _build_scorer(hidden_sz + video_dim + 4, dropout)

        # 聚合器：5 个评分 → n_classes 调整量
        self.score_aggregator = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    # ------------------------------------------------------------------
    def forward(self, text_features, video_features, routing_info,
                fusion_weights, text_conf, video_conf, return_scores=True):
        """
        Args:
            text_features:   [B, hidden_sz]
            video_features:  [B, video_dim]
            routing_info:    Dict（来自 SparseMoEConfidNet / HierarchicalMoEFusion）
            fusion_weights:  [B, 2]  [w_text, w_video]
            text_conf:       [B, 1]
            video_conf:      [B, 1]
            return_scores:   bool

        Returns:
            adjustment:      [B, n_classes]
            scores:          Dict（return_scores=True）
        """
        B = text_features.size(0)
        device = text_features.device

        # 1. 文本质量
        text_quality = self.text_quality_scorer(text_features)       # [B, 1]

        # 2. 视频质量
        video_quality = self.video_quality_scorer(video_features)    # [B, 1]

        # 3. 对齐度
        align_input = torch.cat([text_features, video_features], dim=-1)
        alignment = self.alignment_scorer(align_input)               # [B, 1]

        # 4. 路由决策评分
        txt_routing_w = routing_info.get(
            "text_top_k_weights", torch.zeros(B, 2, device=device))
        vid_routing_w = routing_info.get(
            "video_top_k_weights", torch.zeros(B, 2, device=device))
        modal_routing_w = routing_info.get(
            "layer1_modality_weights", torch.zeros(B, 3, device=device))

        routing_input = torch.cat(
            [text_features, video_features,
             txt_routing_w, vid_routing_w, modal_routing_w], dim=-1
        )
        routing_score = self.routing_scorer(routing_input)           # [B, 1]

        # 5. 融合效果评分
        fusion_input = torch.cat(
            [text_features, video_features,
             fusion_weights, text_conf, video_conf], dim=-1
        )
        fusion_score = self.fusion_scorer(fusion_input)              # [B, 1]

        # 6. 聚合
        all_scores = torch.cat(
            [text_quality, video_quality, alignment, routing_score, fusion_score],
            dim=-1
        )  # [B, 5]
        adjustment = self.score_aggregator(all_scores)               # [B, n_classes]

        if return_scores:
            scores = {
                "text_quality": text_quality,
                "video_quality": video_quality,
                "alignment": alignment,
                "routing": routing_score,
                "fusion": fusion_score,
                "overall": all_scores.mean(dim=-1, keepdim=True),
            }
            return adjustment, scores

        return adjustment
