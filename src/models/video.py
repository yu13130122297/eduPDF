#!/usr/bin/env python3
#
# 视频编码器和分类器实现
# 基于ImageEncoder的设计思路，适配视频数据的时序特征提取
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np


class VideoEncoder(nn.Module):
    """
    视频编码器：使用ResNet提取帧特征，LSTM处理时序信息
    支持特征缓存以加速后续训练
    """
    def __init__(self, args):
        super(VideoEncoder, self).__init__()
        self.args = args

        # 特征缓存配置
        self.use_cache = getattr(args, 'use_video_cache', True)
        self.cache_dir = getattr(args, 'video_cache_dir', 'checkpoints/video_features')

        # 使用更轻量的ResNet50作为帧特征提取器，减少内存使用
        resnet = torchvision.models.resnet50(pretrained=True)
        # 移除最后的全连接层和平均池化层，保留到conv5_x
        modules = list(resnet.children())[:-2]
        self.frame_encoder = nn.Sequential(*modules)
        
        # 空间池化层，将7x7特征图池化为固定大小
        pool_func = (
            nn.AdaptiveAvgPool2d
            if args.video_pooling_type == "avg"
            else nn.AdaptiveMaxPool2d
        )
        self.spatial_pool = pool_func((1, 1))  # 池化为1x1
        
        # LSTM处理时序信息
        self.temporal_lstm = nn.LSTM(
            input_size=2048,  # ResNet50的输出特征维度
            hidden_size=args.video_hidden_sz,
            num_layers=args.video_lstm_layers,
            batch_first=True,
            bidirectional=args.video_bidirectional,
            dropout=args.dropout if args.video_lstm_layers > 1 else 0
        )
        
        # 计算LSTM输出维度
        lstm_output_dim = args.video_hidden_sz * (2 if args.video_bidirectional else 1)
        
        # 特征投影层，将LSTM输出投影到指定维度
        self.feature_projection = nn.Linear(lstm_output_dim, args.video_feature_dim)
        
        # 是否冻结CNN参数
        if args.freeze_video_cnn:
            for param in self.frame_encoder.parameters():
                param.requires_grad = False
    
    def forward(self, video_frames):
        """
        前向传播
        Args:
            video_frames: [batch_size, num_frames, 3, height, width]
        Returns:
            video_features: [batch_size, video_feature_dim]
        """
        batch_size, num_frames, channels, height, width = video_frames.shape
        
        # 重塑为 [batch_size * num_frames, 3, height, width] 以便批量处理
        frames_flat = video_frames.view(-1, channels, height, width)
        
        # 提取每帧的CNN特征
        with torch.set_grad_enabled(not self.args.freeze_video_cnn):
            # 使用梯度检查点减少内存使用
            if self.training and not self.args.freeze_video_cnn:
                frame_features = torch.utils.checkpoint.checkpoint(self.frame_encoder, frames_flat)
            else:
                frame_features = self.frame_encoder(frames_flat)  # [batch_size * num_frames, 2048, 7, 7]
        
        # 空间池化
        frame_features = self.spatial_pool(frame_features)  # [batch_size * num_frames, 2048, 1, 1]
        frame_features = frame_features.view(batch_size * num_frames, -1)  # [batch_size * num_frames, 2048]
        
        # 重塑为时序数据 [batch_size, num_frames, 2048]
        frame_features = frame_features.view(batch_size, num_frames, -1)
        
        # LSTM处理时序信息
        lstm_out, _ = self.temporal_lstm(frame_features)
        
        # 根据池化策略选择最终特征
        if self.args.video_pooling_type == "last":
            # 使用最后一个时间步的输出
            video_features = lstm_out[:, -1, :]
        elif self.args.video_pooling_type == "mean":
            # 使用所有时间步输出的平均值
            video_features = torch.mean(lstm_out, dim=1)
        elif self.args.video_pooling_type == "max":
            # 使用所有时间步输出的最大值
            video_features, _ = torch.max(lstm_out, dim=1)
        else:
            video_features = lstm_out[:, -1, :]  # 默认使用最后一个时间步
        
        # 特征投影
        video_features = self.feature_projection(video_features)
        
        return video_features  # [batch_size, video_feature_dim]


class VideoClf(nn.Module):
    """
    视频分类器：包含视频编码器和分类头
    """
    def __init__(self, args):
        super(VideoClf, self).__init__()
        self.args = args
        self.video_encoder = VideoEncoder(args)
        self.clf = nn.Linear(args.video_feature_dim, args.n_classes)
        
        # 初始化分类层权重
        nn.init.xavier_uniform_(self.clf.weight)
        nn.init.zeros_(self.clf.bias)
    
    def forward(self, video_frames):
        """
        前向传播
        Args:
            video_frames: [batch_size, num_frames, 3, height, width]
        Returns:
            logits: [batch_size, n_classes] - 分类logits
            features: [batch_size, video_feature_dim] - 视频特征
        """
        video_features = self.video_encoder(video_frames)
        logits = self.clf(video_features)
        return logits, video_features


def load_video_frames(video_path, num_frames=16, frame_size=(224, 224), sampling_strategy="uniform"):
    """
    从视频文件中加载指定数量的帧
    Args:
        video_path: 视频文件路径
        num_frames: 需要采样的帧数
        frame_size: 帧的目标尺寸 (width, height)
        sampling_strategy: 采样策略 ("uniform", "random", "center")
    Returns:
        frames: [num_frames, 3, height, width] 的tensor
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        # 如果视频帧数不足，重复最后一帧
        frame_indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
    else:
        # 根据采样策略选择帧索引
        if sampling_strategy == "uniform":
            # 均匀采样
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        elif sampling_strategy == "random":
            # 随机采样
            frame_indices = sorted(np.random.choice(total_frames, num_frames, replace=False))
        elif sampling_strategy == "center":
            # 中心采样
            start_idx = (total_frames - num_frames) // 2
            frame_indices = list(range(start_idx, start_idx + num_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # BGR转RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 调整尺寸
            frame = cv2.resize(frame, frame_size)
            # 转换为tensor并归一化到[0,1]
            frame = torch.from_numpy(frame).float() / 255.0
            # 调整维度顺序为 [3, height, width]
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
        else:
            # 如果读取失败，使用零填充
            frames.append(torch.zeros(3, frame_size[1], frame_size[0]))
    
    cap.release()
    
    # 堆叠为 [num_frames, 3, height, width]
    frames_tensor = torch.stack(frames)
    return frames_tensor


def get_video_transforms(is_training=True):
    """
    获取视频数据的变换
    Args:
        is_training: 是否为训练模式
    Returns:
        transform: 变换函数
    """
    if is_training:
        # 训练时的数据增强
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # 测试时只进行标准化
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform
