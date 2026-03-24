"""
Mamba 视频编码器（FR-3）
用状态空间模型 Mamba 替换 LSTM，支持 16-100 帧长视频，线性复杂度。

依赖：pip install mamba-ssm
若环境不支持 mamba-ssm，自动回退到内置 SimpleMamba（纯 PyTorch 实现）。
"""

import torch
import torch.nn as nn
import torchvision

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


# ------------------------------------------------------------------
# 纯 PyTorch 的轻量 SSM 回退实现（当 mamba-ssm 不可用时使用）
# ------------------------------------------------------------------
class SimpleMamba(nn.Module):
    """轻量 SSM 替代实现，使用 GRU 近似 Mamba 行为"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv,
                                 padding=d_conv - 1, groups=d_inner)
        self.act = nn.SiLU()
        self.gru = nn.GRU(d_inner, d_state, batch_first=True)
        self.out_proj = nn.Linear(d_state, d_model)

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)                          # [B, L, 2*d_inner]
        x_part, z = xz.chunk(2, dim=-1)               # each [B, L, d_inner]

        # Conv1d 沿时间维
        x_conv = self.conv1d(x_part.transpose(1, 2))  # [B, d_inner, L+pad]
        x_conv = x_conv[:, :, :L].transpose(1, 2)     # [B, L, d_inner]
        x_conv = self.act(x_conv)

        out, _ = self.gru(x_conv)                      # [B, L, d_state]
        out = self.out_proj(out)                       # [B, L, D]
        out = out * torch.sigmoid(z[..., :out.size(-1)])
        return out


def _build_mamba_layer(d_model, d_state, d_conv, expand):
    if MAMBA_AVAILABLE:
        return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    return SimpleMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)


# ------------------------------------------------------------------
# MambaVideoEncoder
# ------------------------------------------------------------------
class MambaVideoEncoder(nn.Module):
    """
    ResNet50 提取帧特征 + 多层 Mamba 时序编码。
    保持与原 VideoEncoder 相同的输出接口：[batch, video_feature_dim]
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        d_model = getattr(args, "mamba_d_model", 512)
        d_state = getattr(args, "mamba_d_state", 16)
        d_conv = getattr(args, "mamba_d_conv", 4)
        expand = getattr(args, "mamba_expand", 2)
        n_layers = getattr(args, "mamba_n_layers", 4)
        self.temporal_pooling = getattr(args, "video_pooling_type", "last")
        video_feature_dim = getattr(args, "video_feature_dim", 512)

        # ResNet50 backbone（去掉 avgpool + fc）
        resnet = torchvision.models.resnet50(pretrained=True)
        self.frame_encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        if getattr(args, "freeze_video_cnn", False):
            for param in self.frame_encoder.parameters():
                param.requires_grad = False

        # 特征投影：ResNet50 输出 2048 → d_model
        self.frame_projection = nn.Linear(2048, d_model)

        # 多层 Mamba + 残差 + LayerNorm
        self.mamba_layers = nn.ModuleList([
            _build_mamba_layer(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

        # 输出投影到 video_feature_dim
        self.output_projection = nn.Linear(d_model, video_feature_dim)

    def forward(self, video_frames):
        """
        Args:
            video_frames: [B, T, 3, H, W]
        Returns:
            video_features: [B, video_feature_dim]
        """
        B, T, C, H, W = video_frames.shape

        # 1. 逐帧提取 ResNet50 特征
        frames_flat = video_frames.view(B * T, C, H, W)
        freeze = getattr(self.args, "freeze_video_cnn", False)
        with torch.set_grad_enabled(not freeze):
            feat = self.frame_encoder(frames_flat)       # [B*T, 2048, h, w]
        feat = self.spatial_pool(feat).view(B, T, -1)   # [B, T, 2048]

        # 2. 投影到 Mamba 维度
        x = self.frame_projection(feat)                  # [B, T, d_model]

        # 3. 多层 Mamba 编码（残差连接）
        for mamba, norm in zip(self.mamba_layers, self.layer_norms):
            residual = x
            x = mamba(x)
            x = norm(x + residual)

        # 4. 时序池化
        if self.temporal_pooling == "mean":
            x = x.mean(dim=1)
        elif self.temporal_pooling == "max":
            x = x.max(dim=1)[0]
        else:  # "last"（默认）
            x = x[:, -1, :]

        # 5. 输出投影
        return self.output_projection(x)                 # [B, video_feature_dim]
