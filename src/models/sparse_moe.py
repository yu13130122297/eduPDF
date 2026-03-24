"""
Sparse MoE ConfidNet（FR-4）
用 8 专家稀疏混合专家模型替换 20 个独立置信度网络。
- 8 个专家：2 共享 + 2 模态 + 4 场景
- Top-2 稀疏路由，权重和为 1
- 负载均衡辅助损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


EXPERT_TYPES = [
    "shared_1",        # 0: 共享专家
    "shared_2",        # 1: 共享专家
    "text_specialist",  # 2: 文本专家
    "video_specialist", # 3: 视频专家
    "lecture",         # 4: 讲授场景
    "discussion",      # 5: 讨论场景
    "writing",         # 6: 板书场景
    "silence",         # 7: 沉寂场景
]


def _build_expert(input_dim: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 1),
        nn.Sigmoid(),
    )


class SparseMoEConfidNet(nn.Module):
    """
    稀疏混合专家置信度网络。
    输入文本特征 + 视频特征，输出各模态的置信度标量。
    """

    def __init__(self, args):
        super().__init__()
        self.n_experts = 8
        self.top_k = 2
        self.expert_types = EXPERT_TYPES

        hidden_sz = getattr(args, "hidden_sz", 768)
        video_dim = getattr(args, "video_feature_dim", 512)
        dropout = getattr(args, "dropout", 0.3)
        self.load_balance_weight = getattr(args, "moe_load_balance_weight", 0.01)

        combined_dim = hidden_sz + video_dim
        self.experts = nn.ModuleList([
            _build_expert(combined_dim, dropout) for _ in range(self.n_experts)
        ])

        # 各模态独立路由器
        self.text_router = nn.Sequential(
            nn.Linear(hidden_sz, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.n_experts),
        )

        self.video_router = nn.Sequential(
            nn.Linear(video_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.n_experts),
        )

    # ------------------------------------------------------------------
    def forward(self, text_features, video_features, return_routing=False):
        """
        Returns:
            text_confidence:  [B, 1]
            video_confidence: [B, 1]
            routing_info:     Dict（return_routing=True 时返回）
        """
        B = text_features.size(0)
        combined = torch.cat([text_features, video_features], dim=-1)  # [B, D]

        # --- 路由 logits ---
        text_logits = self.text_router(text_features)   # [B, 8]
        video_logits = self.video_router(video_features) # [B, 8]

        # --- Top-2 稀疏选择 ---
        tk_txt_val, tk_txt_idx = torch.topk(text_logits, self.top_k, dim=-1)
        tk_vid_val, tk_vid_idx = torch.topk(video_logits, self.top_k, dim=-1)

        # Softmax 归一化（仅 Top-2）
        txt_weights = F.softmax(tk_txt_val, dim=-1)   # [B, 2]
        vid_weights = F.softmax(tk_vid_val, dim=-1)   # [B, 2]

        # --- 所有专家输出 ---
        expert_outs = torch.stack(
            [exp(combined) for exp in self.experts], dim=1
        )  # [B, 8, 1]

        # --- 加权聚合 ---
        text_conf = self._aggregate(B, tk_txt_idx, txt_weights, expert_outs,
                                    text_features.device)
        video_conf = self._aggregate(B, tk_vid_idx, vid_weights, expert_outs,
                                     video_features.device)

        # --- 负载均衡损失 ---
        if self.training:
            self.load_balance_loss = self._load_balance_loss(
                text_logits, video_logits
            )
        else:
            self.load_balance_loss = torch.zeros(1, device=text_features.device)

        if return_routing:
            routing_info = {
                "text_top_k_indices": tk_txt_idx,
                "text_top_k_weights": txt_weights,
                "video_top_k_indices": tk_vid_idx,
                "video_top_k_weights": vid_weights,
                "expert_types": self.expert_types,
            }
            return text_conf, video_conf, routing_info

        return text_conf, video_conf

    # ------------------------------------------------------------------
    def _aggregate(self, B, top_k_indices, top_k_weights, expert_outs, device):
        """用 Top-2 专家加权求和"""
        conf = torch.zeros(B, 1, device=device)
        for k in range(self.top_k):
            idx = top_k_indices[:, k]            # [B]
            w = top_k_weights[:, k:k+1]          # [B, 1]
            # 逐样本取对应专家输出
            chosen = expert_outs[torch.arange(B), idx, :]  # [B, 1]
            conf = conf + w * chosen
        return conf

    # ------------------------------------------------------------------
    def _load_balance_loss(self, text_logits, video_logits):
        """鼓励专家均匀使用的辅助损失"""
        target = 1.0 / self.n_experts
        txt_avg = F.softmax(text_logits, dim=-1).mean(dim=0)   # [8]
        vid_avg = F.softmax(video_logits, dim=-1).mean(dim=0)  # [8]
        loss = ((txt_avg - target) ** 2).sum() + ((vid_avg - target) ** 2).sum()
        return self.load_balance_weight * loss
