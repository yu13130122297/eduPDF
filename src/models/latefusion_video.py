#!/usr/bin/env python3
#
# 视频+文本多模态动态融合分类模型（增强版）
# 参考latefusion_pdf.py的动态融合机制，适配视频+文本的多模态学习
# 新增功能：
# 1. 跨模态注意力机制
# 2. 类别感知的特征增强
# 3. 焦点损失处理类别不平衡
# 4. 原型网络增强小样本学习
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.models.bert import BertClf
from src.models.video import VideoClf


class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""

    def __init__(self, text_dim, video_dim, hidden_dim=256):
        super(CrossModalAttention, self).__init__()
        self.text_dim = text_dim
        self.video_dim = video_dim
        self.hidden_dim = hidden_dim

        # 文本到视频的注意力
        self.text_to_video_query = nn.Linear(text_dim, hidden_dim)
        self.text_to_video_key = nn.Linear(video_dim, hidden_dim)
        self.text_to_video_value = nn.Linear(video_dim, hidden_dim)

        # 视频到文本的注意力
        self.video_to_text_query = nn.Linear(video_dim, hidden_dim)
        self.video_to_text_key = nn.Linear(text_dim, hidden_dim)
        self.video_to_text_value = nn.Linear(text_dim, hidden_dim)

        # 输出投影
        self.text_output_proj = nn.Linear(hidden_dim, text_dim)
        self.video_output_proj = nn.Linear(hidden_dim, video_dim)

        self.dropout = nn.Dropout(0.1)
        self.layer_norm_text = nn.LayerNorm(text_dim)
        self.layer_norm_video = nn.LayerNorm(video_dim)

    def forward(self, text_features, video_features):
        """
        Args:
            text_features: [batch_size, text_dim]
            video_features: [batch_size, video_dim]
        Returns:
            enhanced_text: [batch_size, text_dim]
            enhanced_video: [batch_size, video_dim]
        """
        batch_size = text_features.shape[0]

        # 文本查询视频信息
        text_query = self.text_to_video_query(text_features)  # [batch_size, hidden_dim]
        video_key = self.text_to_video_key(video_features)    # [batch_size, hidden_dim]
        video_value = self.text_to_video_value(video_features) # [batch_size, hidden_dim]

        # 计算注意力权重
        text_attention_scores = torch.matmul(text_query.unsqueeze(1), video_key.unsqueeze(2)).squeeze()  # [batch_size]
        text_attention_weights = F.softmax(text_attention_scores.unsqueeze(1), dim=1)  # [batch_size, 1]

        # 应用注意力
        text_attended = text_attention_weights * video_value  # [batch_size, hidden_dim]
        enhanced_text_features = self.text_output_proj(text_attended)  # [batch_size, text_dim]
        enhanced_text_features = self.layer_norm_text(text_features + self.dropout(enhanced_text_features))

        # 视频查询文本信息
        video_query = self.video_to_text_query(video_features)  # [batch_size, hidden_dim]
        text_key = self.video_to_text_key(text_features)        # [batch_size, hidden_dim]
        text_value = self.video_to_text_value(text_features)    # [batch_size, hidden_dim]

        # 计算注意力权重
        video_attention_scores = torch.matmul(video_query.unsqueeze(1), text_key.unsqueeze(2)).squeeze()  # [batch_size]
        video_attention_weights = F.softmax(video_attention_scores.unsqueeze(1), dim=1)  # [batch_size, 1]

        # 应用注意力
        video_attended = video_attention_weights * text_value  # [batch_size, hidden_dim]
        enhanced_video_features = self.video_output_proj(video_attended)  # [batch_size, video_dim]
        enhanced_video_features = self.layer_norm_video(video_features + self.dropout(enhanced_video_features))

        return enhanced_text_features, enhanced_video_features


class PrototypeNetwork(nn.Module):
    """原型网络，用于增强小样本类别的学习"""

    def __init__(self, feature_dim, n_classes, prototype_dim=128):
        super(PrototypeNetwork, self).__init__()
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.prototype_dim = prototype_dim

        # 特征投影到原型空间
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, prototype_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(prototype_dim, prototype_dim)
        )

        # 可学习的类别原型
        self.prototypes = nn.Parameter(torch.randn(n_classes, prototype_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, features):
        """
        Args:
            features: [batch_size, feature_dim]
        Returns:
            prototype_logits: [batch_size, n_classes]
            projected_features: [batch_size, prototype_dim]
        """
        # 投影特征到原型空间
        projected_features = self.feature_proj(features)  # [batch_size, prototype_dim]
        projected_features = F.normalize(projected_features, p=2, dim=1)

        # 计算与原型的相似度
        normalized_prototypes = F.normalize(self.prototypes, p=2, dim=1)  # [n_classes, prototype_dim]

        # 计算余弦相似度
        similarities = torch.matmul(projected_features, normalized_prototypes.t())  # [batch_size, n_classes]

        # 使用温度参数缩放
        temperature = 10.0
        prototype_logits = similarities * temperature

        return prototype_logits, projected_features


class MultimodalLateFusionClf_video(nn.Module):
    """
    视频+文本多模态动态融合分类器（增强版）

    核心思想：
    1. 分别使用BERT和VideoEncoder提取文本和视频特征
    2. 使用跨模态注意力机制增强特征表示
    3. 为每个类别学习独立的置信度网络和融合权重
    4. 使用原型网络增强小样本类别的学习
    5. 使用类别特定的全息权重计算动态融合权重
    6. 在训练和测试阶段采用不同的融合策略
    """

    def __init__(self, args):
        super(MultimodalLateFusionClf_video, self).__init__()
        self.args = args
        self.n_classes = args.n_classes

        # 文本分类器（BERT）
        self.txtclf = BertClf(args)

        # 视频分类器
        self.videoclf = VideoClf(args)

        # 跨模态注意力机制
        self.cross_modal_attention = CrossModalAttention(
            text_dim=args.hidden_sz,
            video_dim=args.video_feature_dim,
            hidden_dim=256
        )

        # 原型网络（用于小样本学习）
        self.text_prototype_net = PrototypeNetwork(
            feature_dim=args.hidden_sz,
            n_classes=self.n_classes,
            prototype_dim=128
        )

        self.video_prototype_net = PrototypeNetwork(
            feature_dim=args.video_feature_dim,
            n_classes=self.n_classes,
            prototype_dim=128
        )

        # 类别特定的文本置信度网络（增强版）
        # 为每个类别创建独立的置信度网络，增加深度和注意力
        self.ConfidNet_txt_class = nn.ModuleList([
            nn.Sequential(
                nn.Linear(args.hidden_sz, args.hidden_sz),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_sz, args.hidden_sz // 2),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_sz // 2, args.hidden_sz // 4),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_sz // 4, 1),
                nn.Sigmoid()
            ) for _ in range(self.n_classes)
        ])

        # 类别特定的视频置信度网络（增强版）
        self.ConfidNet_video_class = nn.ModuleList([
            nn.Sequential(
                nn.Linear(args.video_feature_dim, args.video_feature_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.video_feature_dim, args.video_feature_dim // 2),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.video_feature_dim // 2, args.video_feature_dim // 4),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.video_feature_dim // 4, 1),
                nn.Sigmoid()
            ) for _ in range(self.n_classes)
        ])

        # 类别预测网络（增强版，用于确定使用哪个类别的权重）
        self.class_predictor = nn.Sequential(
            nn.Linear(args.hidden_sz + args.video_feature_dim, (args.hidden_sz + args.video_feature_dim) // 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear((args.hidden_sz + args.video_feature_dim) // 2, (args.hidden_sz + args.video_feature_dim) // 4),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear((args.hidden_sz + args.video_feature_dim) // 4, self.n_classes),
            nn.Softmax(dim=1)
        )

        # 类别感知的特征增强网络
        self.class_aware_text_enhancer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(args.hidden_sz, args.hidden_sz // 2),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_sz // 2, args.hidden_sz),
                nn.Tanh()
            ) for _ in range(self.n_classes)
        ])

        self.class_aware_video_enhancer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(args.video_feature_dim, args.video_feature_dim // 2),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.video_feature_dim // 2, args.video_feature_dim),
                nn.Tanh()
            ) for _ in range(self.n_classes)
        ])

        # 初始化网络权重
        self._init_networks()
    
    def _init_networks(self):
        """初始化所有网络的权重"""
        # 初始化类别特定的置信度网络
        for class_nets in [self.ConfidNet_txt_class, self.ConfidNet_video_class,
                          self.class_aware_text_enhancer, self.class_aware_video_enhancer]:
            for net in class_nets:
                for layer in net:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)

        # 初始化类别预测网络
        for layer in self.class_predictor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, txt, mask, segment, video_frames, choice='train'):
        """
        前向传播（增强版）

        Args:
            txt: 文本token序列 [batch_size, seq_len]
            mask: 注意力掩码 [batch_size, seq_len]
            segment: 段落标识 [batch_size, seq_len]
            video_frames: 视频帧 [batch_size, num_frames, 3, height, width]
            choice: 模式选择 ('train', 'test', 'static')

        Returns:
            根据choice返回不同的输出组合
        """
        # 获取文本和视频的分类结果及特征
        txt_out, txt_f = self.txtclf(txt, mask, segment)  # txt_f: [batch_size, hidden_sz]
        video_out, video_f = self.videoclf(video_frames)  # video_f: [batch_size, video_feature_dim]

        # 应用跨模态注意力机制增强特征
        enhanced_txt_f, enhanced_video_f = self.cross_modal_attention(txt_f, video_f)

        # 使用增强后的特征重新计算分类结果
        enhanced_txt_out = self.txtclf.clf(enhanced_txt_f)
        enhanced_video_out = self.videoclf.clf(enhanced_video_f)

        # 计算原型网络的输出（用于小样本学习）
        txt_proto_out, _ = self.text_prototype_net(enhanced_txt_f)
        video_proto_out, _ = self.video_prototype_net(enhanced_video_f)

        # 如果使用动态融合
        if self.args.use_dynamic_fusion:
            if choice == 'train':
                return self._forward_train_enhanced(
                    enhanced_txt_out, enhanced_txt_f, enhanced_video_out, enhanced_video_f,
                    txt_proto_out, video_proto_out
                )
            elif choice == 'test':
                return self._forward_test_enhanced(
                    enhanced_txt_out, enhanced_txt_f, enhanced_video_out, enhanced_video_f,
                    txt_proto_out, video_proto_out
                )

        # 静态融合（简单平均）
        else:
            fused_out = 0.5 * enhanced_txt_out + 0.5 * enhanced_video_out
            return fused_out, enhanced_txt_out, enhanced_video_out
    
    def _forward_train(self, txt_out, txt_f, video_out, video_f):
        """
        训练阶段的动态融合
        
        Args:
            txt_out: 文本分类logits [batch_size, n_classes]
            txt_f: 文本特征 [batch_size, hidden_sz]
            video_out: 视频分类logits [batch_size, n_classes]
            video_f: 视频特征 [batch_size, video_feature_dim]
        
        Returns:
            融合logits, 文本logits, 视频logits, 文本置信度, 视频置信度
        """
        # 计算置信度（使用detach避免梯度传播到特征提取器）
        txt_f_detached = txt_f.clone().detach()
        video_f_detached = video_f.clone().detach()
        
        txt_confidence = self.ConfidNet_txt(txt_f_detached)  # [batch_size, 1]
        video_confidence = self.ConfidNet_video(video_f_detached)  # [batch_size, 1]
        
        # 计算全息权重（holographic weights）
        # 这是PDF论文中的核心创新：通过对数比值计算互补权重
        eps = 1e-8  # 防止除零
        txt_holo = torch.log(video_confidence + eps) / (torch.log(txt_confidence * video_confidence + eps) + eps)
        video_holo = torch.log(txt_confidence + eps) / (torch.log(txt_confidence * video_confidence + eps) + eps)
        
        # 组合置信度：原始置信度 + 全息权重
        combined_txt = txt_confidence.detach() + txt_holo.detach()
        combined_video = video_confidence.detach() + video_holo.detach()
        
        # 计算归一化权重
        weights = torch.stack([combined_txt, combined_video], dim=1)  # [batch_size, 2, 1]
        weights = F.softmax(weights, dim=1)  # 在模态维度上进行softmax
        
        w_txt = weights[:, 0]  # [batch_size, 1]
        w_video = weights[:, 1]  # [batch_size, 1]
        
        # 动态加权融合
        fused_out = w_txt.detach() * txt_out + w_video.detach() * video_out
        
        return fused_out, txt_out, video_out, txt_confidence, video_confidence

    def _forward_train_class_specific(self, txt_out, txt_f, video_out, video_f):
        """
        训练阶段的类别特定动态融合

        为每个类别学习独立的融合权重，让模型能够为不同类别分配不同的模态重要性

        Args:
            txt_out: 文本分类logits [batch_size, n_classes]
            txt_f: 文本特征 [batch_size, hidden_sz]
            video_out: 视频分类logits [batch_size, n_classes]
            video_f: 视频特征 [batch_size, video_feature_dim]

        Returns:
            融合logits, 文本logits, 视频logits, 类别特定文本置信度, 类别特定视频置信度, 类别概率
        """
        batch_size = txt_f.shape[0]

        # 使用detach的特征进行置信度计算
        txt_f_detached = txt_f.clone().detach()
        video_f_detached = video_f.clone().detach()

        # 预测类别分布（用于加权不同类别的置信度）
        combined_features = torch.cat([txt_f_detached, video_f_detached], dim=1)
        class_probs = self.class_predictor(combined_features)  # [batch_size, n_classes]

        # 为每个类别计算置信度
        txt_confidences = []  # 存储每个类别的文本置信度
        video_confidences = []  # 存储每个类别的视频置信度

        for class_idx in range(self.n_classes):
            txt_conf_class = self.ConfidNet_txt_class[class_idx](txt_f_detached)  # [batch_size, 1]
            video_conf_class = self.ConfidNet_video_class[class_idx](video_f_detached)  # [batch_size, 1]

            txt_confidences.append(txt_conf_class)
            video_confidences.append(video_conf_class)

        # 将置信度堆叠为张量
        txt_confidences = torch.stack(txt_confidences, dim=2)  # [batch_size, 1, n_classes]
        video_confidences = torch.stack(video_confidences, dim=2)  # [batch_size, 1, n_classes]

        # 使用类别概率加权置信度
        class_probs_expanded = class_probs.unsqueeze(1)  # [batch_size, 1, n_classes]

        # 加权平均得到最终置信度
        txt_confidence = torch.sum(txt_confidences * class_probs_expanded, dim=2)  # [batch_size, 1]
        video_confidence = torch.sum(video_confidences * class_probs_expanded, dim=2)  # [batch_size, 1]

        # 计算类别特定的全息权重
        eps = 1e-8
        txt_holo = torch.log(video_confidence + eps) / (torch.log(txt_confidence * video_confidence + eps) + eps)
        video_holo = torch.log(txt_confidence + eps) / (torch.log(txt_confidence * video_confidence + eps) + eps)

        # 组合置信度
        combined_txt = txt_confidence.detach() + txt_holo.detach()
        combined_video = video_confidence.detach() + video_holo.detach()

        # 计算归一化权重
        weights = torch.stack([combined_txt, combined_video], dim=1)  # [batch_size, 2, 1]
        weights = F.softmax(weights, dim=1)

        w_txt = weights[:, 0]  # [batch_size, 1]
        w_video = weights[:, 1]  # [batch_size, 1]

        # 动态加权融合
        fused_out = w_txt.detach() * txt_out + w_video.detach() * video_out

        # 返回类别特定的置信度信息（用于分析）
        return fused_out, txt_out, video_out, txt_confidences, video_confidences, class_probs

    def _forward_train_enhanced(self, txt_out, txt_f, video_out, video_f, txt_proto_out, video_proto_out):
        """
        增强版训练阶段的动态融合

        结合跨模态注意力、原型网络和类别感知增强

        Args:
            txt_out: 增强后的文本分类logits [batch_size, n_classes]
            txt_f: 增强后的文本特征 [batch_size, hidden_sz]
            video_out: 增强后的视频分类logits [batch_size, n_classes]
            video_f: 增强后的视频特征 [batch_size, video_feature_dim]
            txt_proto_out: 文本原型网络输出 [batch_size, n_classes]
            video_proto_out: 视频原型网络输出 [batch_size, n_classes]

        Returns:
            融合logits, 文本logits, 视频logits, 类别特定文本置信度, 类别特定视频置信度, 类别概率, 原型输出
        """
        # 使用detach的特征进行置信度计算
        txt_f_detached = txt_f.clone().detach()
        video_f_detached = video_f.clone().detach()

        # 预测类别分布（用于加权不同类别的置信度）
        combined_features = torch.cat([txt_f_detached, video_f_detached], dim=1)
        class_probs = self.class_predictor(combined_features)  # [batch_size, n_classes]

        # 类别感知的特征增强
        enhanced_txt_features = []
        enhanced_video_features = []

        for class_idx in range(self.n_classes):
            # 获取该类别的权重
            class_weight = class_probs[:, class_idx:class_idx+1]  # [batch_size, 1]

            # 类别特定的特征增强
            class_enhanced_txt = self.class_aware_text_enhancer[class_idx](txt_f_detached)
            class_enhanced_video = self.class_aware_video_enhancer[class_idx](video_f_detached)

            # 加权累积
            enhanced_txt_features.append(class_weight * class_enhanced_txt)
            enhanced_video_features.append(class_weight * class_enhanced_video)

        # 聚合所有类别的增强特征
        final_enhanced_txt = torch.stack(enhanced_txt_features, dim=0).sum(dim=0)  # [batch_size, hidden_sz]
        final_enhanced_video = torch.stack(enhanced_video_features, dim=0).sum(dim=0)  # [batch_size, video_feature_dim]

        # 与原始特征结合
        combined_txt_f = txt_f_detached + 0.3 * final_enhanced_txt
        combined_video_f = video_f_detached + 0.3 * final_enhanced_video

        # 为每个类别计算置信度
        txt_confidences = []
        video_confidences = []

        for class_idx in range(self.n_classes):
            txt_conf_class = self.ConfidNet_txt_class[class_idx](combined_txt_f)  # [batch_size, 1]
            video_conf_class = self.ConfidNet_video_class[class_idx](combined_video_f)  # [batch_size, 1]

            txt_confidences.append(txt_conf_class)
            video_confidences.append(video_conf_class)

        # 将置信度堆叠为张量
        txt_confidences = torch.stack(txt_confidences, dim=2)  # [batch_size, 1, n_classes]
        video_confidences = torch.stack(video_confidences, dim=2)  # [batch_size, 1, n_classes]

        # 使用类别概率加权置信度
        class_probs_expanded = class_probs.unsqueeze(1)  # [batch_size, 1, n_classes]

        # 加权平均得到最终置信度
        txt_confidence = torch.sum(txt_confidences * class_probs_expanded, dim=2)  # [batch_size, 1]
        video_confidence = torch.sum(video_confidences * class_probs_expanded, dim=2)  # [batch_size, 1]

        # 计算类别特定的全息权重
        eps = 1e-8
        txt_holo = torch.log(video_confidence + eps) / (torch.log(txt_confidence * video_confidence + eps) + eps)
        video_holo = torch.log(txt_confidence + eps) / (torch.log(txt_confidence * video_confidence + eps) + eps)

        # 组合置信度
        combined_txt = txt_confidence.detach() + txt_holo.detach()
        combined_video = video_confidence.detach() + video_holo.detach()

        # 计算归一化权重
        weights = torch.stack([combined_txt, combined_video], dim=1)  # [batch_size, 2, 1]
        weights = F.softmax(weights, dim=1)

        w_txt = weights[:, 0]  # [batch_size, 1]
        w_video = weights[:, 1]  # [batch_size, 1]

        # 动态加权融合（结合原型网络输出）
        # 主要融合
        main_fused_out = w_txt.detach() * txt_out + w_video.detach() * video_out

        # 原型融合（用于增强小样本类别）
        proto_fused_out = 0.5 * txt_proto_out + 0.5 * video_proto_out

        # 最终融合（主要输出 + 原型输出）
        final_fused_out = 0.8 * main_fused_out + 0.2 * proto_fused_out

        # 返回增强版输出
        return (final_fused_out, txt_out, video_out, txt_confidences, video_confidences,
                class_probs, txt_proto_out, video_proto_out)

    def _forward_test(self, txt_out, txt_f, video_out, video_f):
        """
        测试阶段的动态融合
        
        在测试阶段，额外考虑预测的不确定性来调整权重
        
        Args:
            txt_out: 文本分类logits [batch_size, n_classes]
            txt_f: 文本特征 [batch_size, hidden_sz]
            video_out: 视频分类logits [batch_size, n_classes]
            video_f: 视频特征 [batch_size, video_feature_dim]
        
        Returns:
            融合logits, 文本logits, 视频logits, 文本置信度, 视频置信度, 文本权重, 视频权重
        """
        # 计算置信度
        txt_confidence = self.ConfidNet_txt(txt_f)
        video_confidence = self.ConfidNet_video(video_f)
        
        # 计算全息权重
        eps = 1e-8
        txt_holo = torch.log(video_confidence + eps) / (torch.log(txt_confidence * video_confidence + eps) + eps)
        video_holo = torch.log(txt_confidence + eps) / (torch.log(txt_confidence * video_confidence + eps) + eps)
        
        # 组合置信度
        combined_txt = txt_confidence + txt_holo
        combined_video = video_confidence + video_holo
        
        # 计算预测不确定性（基于预测分布的熵）
        txt_pred = F.softmax(txt_out, dim=1)
        video_pred = F.softmax(video_out, dim=1)
        
        # 计算与均匀分布的距离作为不确定性度量
        uniform_prob = 1.0 / txt_pred.shape[1]
        txt_uncertainty = torch.mean(torch.abs(txt_pred - uniform_prob), dim=1, keepdim=True)
        video_uncertainty = torch.mean(torch.abs(video_pred - uniform_prob), dim=1, keepdim=True)
        
        # 根据不确定性调整权重
        # 不确定性高的模态权重应该降低
        condition = txt_uncertainty > video_uncertainty
        txt_adjust = torch.where(condition, torch.ones_like(txt_uncertainty), txt_uncertainty / video_uncertainty)
        video_adjust = torch.where(condition, video_uncertainty / txt_uncertainty, torch.ones_like(video_uncertainty))
        
        # 应用不确定性调整
        adjusted_txt = combined_txt * txt_adjust
        adjusted_video = combined_video * video_adjust
        
        # 计算最终权重
        weights = torch.stack([adjusted_txt, adjusted_video], dim=1)
        weights = F.softmax(weights, dim=1)
        
        w_txt = weights[:, 0]
        w_video = weights[:, 1]
        
        # 动态加权融合
        fused_out = w_txt.detach() * txt_out + w_video.detach() * video_out
        
        return fused_out, txt_out, video_out, txt_confidence, video_confidence, w_txt, w_video

    def _forward_test_class_specific(self, txt_out, txt_f, video_out, video_f):
        """
        测试阶段的类别特定动态融合

        在测试阶段，使用类别特定的权重并考虑预测不确定性

        Args:
            txt_out: 文本分类logits [batch_size, n_classes]
            txt_f: 文本特征 [batch_size, hidden_sz]
            video_out: 视频分类logits [batch_size, n_classes]
            video_f: 视频特征 [batch_size, video_feature_dim]

        Returns:
            融合logits, 文本logits, 视频logits, 类别特定文本置信度, 类别特定视频置信度, 文本权重, 视频权重, 类别概率
        """
        # 预测类别分布
        combined_features = torch.cat([txt_f, video_f], dim=1)
        class_probs = self.class_predictor(combined_features)  # [batch_size, n_classes]

        # 为每个类别计算置信度
        txt_confidences = []
        video_confidences = []

        for class_idx in range(self.n_classes):
            txt_conf_class = self.ConfidNet_txt_class[class_idx](txt_f)  # [batch_size, 1]
            video_conf_class = self.ConfidNet_video_class[class_idx](video_f)  # [batch_size, 1]

            txt_confidences.append(txt_conf_class)
            video_confidences.append(video_conf_class)

        # 将置信度堆叠为张量
        txt_confidences = torch.stack(txt_confidences, dim=2)  # [batch_size, 1, n_classes]
        video_confidences = torch.stack(video_confidences, dim=2)  # [batch_size, 1, n_classes]

        # 使用类别概率加权置信度
        class_probs_expanded = class_probs.unsqueeze(1)  # [batch_size, 1, n_classes]

        # 加权平均得到最终置信度
        txt_confidence = torch.sum(txt_confidences * class_probs_expanded, dim=2)  # [batch_size, 1]
        video_confidence = torch.sum(video_confidences * class_probs_expanded, dim=2)  # [batch_size, 1]

        # 计算类别特定的全息权重
        eps = 1e-8
        txt_holo = torch.log(video_confidence + eps) / (torch.log(txt_confidence * video_confidence + eps) + eps)
        video_holo = torch.log(txt_confidence + eps) / (torch.log(txt_confidence * video_confidence + eps) + eps)

        # 组合置信度
        combined_txt = txt_confidence + txt_holo
        combined_video = video_confidence + video_holo

        # 计算预测不确定性
        txt_pred = F.softmax(txt_out, dim=1)
        video_pred = F.softmax(video_out, dim=1)

        uniform_prob = 1.0 / txt_pred.shape[1]
        txt_uncertainty = torch.mean(torch.abs(txt_pred - uniform_prob), dim=1, keepdim=True)
        video_uncertainty = torch.mean(torch.abs(video_pred - uniform_prob), dim=1, keepdim=True)

        # 根据不确定性调整权重
        condition = txt_uncertainty > video_uncertainty
        txt_adjust = torch.where(condition, torch.ones_like(txt_uncertainty), txt_uncertainty / video_uncertainty)
        video_adjust = torch.where(condition, video_uncertainty / txt_uncertainty, torch.ones_like(video_uncertainty))

        # 应用不确定性调整
        adjusted_txt = combined_txt * txt_adjust
        adjusted_video = combined_video * video_adjust

        # 计算最终权重
        weights = torch.stack([adjusted_txt, adjusted_video], dim=1)
        weights = F.softmax(weights, dim=1)

        w_txt = weights[:, 0]
        w_video = weights[:, 1]

        # 动态加权融合
        fused_out = w_txt.detach() * txt_out + w_video.detach() * video_out

        return fused_out, txt_out, video_out, txt_confidences, video_confidences, w_txt, w_video, class_probs

    def _forward_test_enhanced(self, txt_out, txt_f, video_out, video_f, txt_proto_out, video_proto_out):
        """
        增强版测试阶段的动态融合

        结合跨模态注意力、原型网络、类别感知增强和不确定性调整

        Args:
            txt_out: 增强后的文本分类logits [batch_size, n_classes]
            txt_f: 增强后的文本特征 [batch_size, hidden_sz]
            video_out: 增强后的视频分类logits [batch_size, n_classes]
            video_f: 增强后的视频特征 [batch_size, video_feature_dim]
            txt_proto_out: 文本原型网络输出 [batch_size, n_classes]
            video_proto_out: 视频原型网络输出 [batch_size, n_classes]

        Returns:
            融合logits, 文本logits, 视频logits, 类别特定文本置信度, 类别特定视频置信度,
            文本权重, 视频权重, 类别概率, 原型输出
        """
        # 预测类别分布
        combined_features = torch.cat([txt_f, video_f], dim=1)
        class_probs = self.class_predictor(combined_features)  # [batch_size, n_classes]

        # 类别感知的特征增强
        enhanced_txt_features = []
        enhanced_video_features = []

        for class_idx in range(self.n_classes):
            # 获取该类别的权重
            class_weight = class_probs[:, class_idx:class_idx+1]  # [batch_size, 1]

            # 类别特定的特征增强
            class_enhanced_txt = self.class_aware_text_enhancer[class_idx](txt_f)
            class_enhanced_video = self.class_aware_video_enhancer[class_idx](video_f)

            # 加权累积
            enhanced_txt_features.append(class_weight * class_enhanced_txt)
            enhanced_video_features.append(class_weight * class_enhanced_video)

        # 聚合所有类别的增强特征
        final_enhanced_txt = torch.stack(enhanced_txt_features, dim=0).sum(dim=0)
        final_enhanced_video = torch.stack(enhanced_video_features, dim=0).sum(dim=0)

        # 与原始特征结合
        combined_txt_f = txt_f + 0.3 * final_enhanced_txt
        combined_video_f = video_f + 0.3 * final_enhanced_video

        # 为每个类别计算置信度
        txt_confidences = []
        video_confidences = []

        for class_idx in range(self.n_classes):
            txt_conf_class = self.ConfidNet_txt_class[class_idx](combined_txt_f)
            video_conf_class = self.ConfidNet_video_class[class_idx](combined_video_f)

            txt_confidences.append(txt_conf_class)
            video_confidences.append(video_conf_class)

        # 将置信度堆叠为张量
        txt_confidences = torch.stack(txt_confidences, dim=2)  # [batch_size, 1, n_classes]
        video_confidences = torch.stack(video_confidences, dim=2)  # [batch_size, 1, n_classes]

        # 使用类别概率加权置信度
        class_probs_expanded = class_probs.unsqueeze(1)  # [batch_size, 1, n_classes]

        # 加权平均得到最终置信度
        txt_confidence = torch.sum(txt_confidences * class_probs_expanded, dim=2)  # [batch_size, 1]
        video_confidence = torch.sum(video_confidences * class_probs_expanded, dim=2)  # [batch_size, 1]

        # 计算类别特定的全息权重
        eps = 1e-8
        txt_holo = torch.log(video_confidence + eps) / (torch.log(txt_confidence * video_confidence + eps) + eps)
        video_holo = torch.log(txt_confidence + eps) / (torch.log(txt_confidence * video_confidence + eps) + eps)

        # 组合置信度
        combined_txt = txt_confidence + txt_holo
        combined_video = video_confidence + video_holo

        # 计算预测不确定性
        txt_pred = F.softmax(txt_out, dim=1)
        video_pred = F.softmax(video_out, dim=1)

        uniform_prob = 1.0 / txt_pred.shape[1]
        txt_uncertainty = torch.mean(torch.abs(txt_pred - uniform_prob), dim=1, keepdim=True)
        video_uncertainty = torch.mean(torch.abs(video_pred - uniform_prob), dim=1, keepdim=True)

        # 根据不确定性调整权重
        condition = txt_uncertainty > video_uncertainty
        txt_adjust = torch.where(condition, torch.ones_like(txt_uncertainty), txt_uncertainty / video_uncertainty)
        video_adjust = torch.where(condition, video_uncertainty / txt_uncertainty, torch.ones_like(video_uncertainty))

        # 应用不确定性调整
        adjusted_txt = combined_txt * txt_adjust
        adjusted_video = combined_video * video_adjust

        # 计算最终权重
        weights = torch.stack([adjusted_txt, adjusted_video], dim=1)
        weights = F.softmax(weights, dim=1)

        w_txt = weights[:, 0]
        w_video = weights[:, 1]

        # 动态加权融合（结合原型网络输出）
        # 主要融合
        main_fused_out = w_txt.detach() * txt_out + w_video.detach() * video_out

        # 原型融合（用于增强小样本类别）
        proto_fused_out = 0.5 * txt_proto_out + 0.5 * video_proto_out

        # 最终融合（主要输出 + 原型输出）
        final_fused_out = 0.8 * main_fused_out + 0.2 * proto_fused_out

        return (final_fused_out, txt_out, video_out, txt_confidences, video_confidences,
                w_txt, w_video, class_probs, txt_proto_out, video_proto_out)
