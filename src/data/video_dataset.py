#!/usr/bin/env python3
#
# 视频+文本数据集处理模块
# 基于现有的JsonlDataset，适配视频数据的加载和预处理
#

import json
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import random
import hashlib

from src.data.vocab import Vocab
from src.models.video import load_video_frames, get_video_transforms


def get_video_cache_path(video_path, cache_dir, num_frames):
    """生成缓存文件路径"""
    # 使用视频路径的hash作为文件名
    video_hash = hashlib.md5(video_path.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{video_hash}_f{num_frames}.pt")
    return cache_path


class VideoTextDataset(Dataset):
    """
    视频+文本数据集类
    
    支持从JSONL文件加载视频路径、文本和标签信息
    自动处理视频帧采样和文本tokenization
    """
    
    def __init__(self, data_path, tokenizer, video_transforms, vocab, args):
        """
        初始化数据集
        
        Args:
            data_path: JSONL数据文件路径
            tokenizer: 文本tokenizer
            video_transforms: 视频数据变换
            vocab: 词汇表
            args: 配置参数
        """
        # 加载JSONL数据
        self.data = [json.loads(line) for line in open(data_path, 'r', encoding='utf-8')]
        self.data_dir = os.path.dirname(data_path)
        
        self.tokenizer = tokenizer
        self.video_transforms = video_transforms
        self.vocab = vocab
        self.args = args
        
        # 类别数量
        self.n_classes = len(args.labels)

        # 视频特征缓存配置
        self.use_cache = getattr(args, 'use_video_cache', True)
        self.cache_dir = getattr(args, 'video_cache_dir', 'checkpoints/video_features')

        # 创建缓存目录
        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"创建视频特征缓存目录: {self.cache_dir}")

        # 文本处理参数
        self.text_start_token = ["[CLS]"]
        self.max_seq_len = args.max_seq_len

        print(f"加载了 {len(self.data)} 个样本从 {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """
        获取单个样本

        Returns:
            sentence: tokenized文本 [seq_len]
            segment: 段落标识 [seq_len]
            video_frames: 视频帧 [num_frames, 3, height, width]
            label: 标签索引
            sample_id: 样本ID
            original_id: 原始数据集ID
            text: 原始文本字符串（用于trimodal encoder）
            video_description: 视频描述字符串（用于trimodal encoder）
        """
        item = self.data[index]

        # 处理文本
        sentence, segment = self._process_text(item['text'])

        # 处理视频
        video_frames = self._process_video(item['video'])

        # 处理标签
        label = self.args.labels.index(item['label'])

        # 样本ID - 使用原始数据集的ID而不是索引
        original_id = item['id']
        sample_id = torch.LongTensor([index])  # 保留索引用于内部处理

        # 保存原始文本和视频描述（用于trimodal encoder）
        text = item.get('text', '')
        video_description = item.get('video_description', '')

        return sentence, segment, video_frames, label, sample_id, original_id, text, video_description
    
    def _process_text(self, text):
        """
        处理文本数据
        
        Args:
            text: 原始文本字符串
        
        Returns:
            sentence: tokenized文本tensor [seq_len]
            segment: 段落标识tensor [seq_len]
        """
        # Tokenize文本
        tokens = self.tokenizer(text)
        
        # 添加起始token
        tokens = self.text_start_token + tokens
        
        # 截断或填充到最大长度
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        
        # 转换为ID
        if hasattr(self.vocab, 'stoi'):
            # BERT tokenizer
            sentence = [self.vocab.stoi.get(token, self.vocab.stoi.get('[UNK]', 0)) for token in tokens]
        else:
            # 其他tokenizer
            sentence = [self.vocab.stoi.get(token, 0) for token in tokens]
        
        # 填充到最大长度
        while len(sentence) < self.max_seq_len:
            sentence.append(0)  # PAD token
        
        # 创建段落标识（全部为0，表示第一个句子）
        segment = [0] * len(sentence)
        
        return torch.LongTensor(sentence), torch.LongTensor(segment)
    
    def _process_video(self, video_path):
        """
        处理视频数据（支持特征缓存）

        Args:
            video_path: 视频文件路径

        Returns:
            video_frames: 视频帧tensor [num_frames, 3, height, width]
        """
        try:
            # 如果是相对路径，转换为绝对路径
            if not os.path.isabs(video_path):
                video_path = os.path.join(self.data_dir, video_path)

            # 尝试从缓存加载
            if self.use_cache:
                cache_path = get_video_cache_path(video_path, self.cache_dir, self.args.num_frames)
                if os.path.exists(cache_path):
                    return torch.load(cache_path)

            # 加载视频帧
            frames = load_video_frames(
                video_path=video_path,
                num_frames=self.args.num_frames,
                frame_size=(224, 224),  # 标准输入尺寸
                sampling_strategy=self.args.frame_sampling_strategy
            )

            # 应用数据变换
            if self.video_transforms:
                # 对每一帧应用变换
                transformed_frames = []
                for frame in frames:
                    # frame: [3, H, W]
                    transformed_frame = self.video_transforms(frame)
                    transformed_frames.append(transformed_frame)
                frames = torch.stack(transformed_frames)

            # 保存到缓存
            if self.use_cache:
                torch.save(frames, cache_path)

            return frames  # [num_frames, 3, height, width]

        except Exception as e:
            print(f"处理视频时出错 {video_path}: {e}")
            # 返回零填充的帧
            return torch.zeros(self.args.num_frames, 3, 224, 224)


def collate_video_text_fn(batch, args):
    """
    批处理整理函数

    Args:
        batch: 批次数据列表
        args: 配置参数

    Returns:
        整理后的批次数据
    """
    # 解包批次数据（包含text和video_description）
    sentences, segments, video_frames, labels, sample_ids, original_ids, texts, video_descs = zip(*batch)

    # 文本数据
    sentences = torch.stack(sentences)  # [batch_size, seq_len]
    segments = torch.stack(segments)    # [batch_size, seq_len]

    # 创建注意力掩码
    attention_mask = (sentences != 0).long()  # PAD token的ID为0

    # 视频数据
    video_frames = torch.stack(video_frames)  # [batch_size, num_frames, 3, height, width]

    # 标签
    labels = torch.LongTensor(labels)  # [batch_size]

    # 样本ID
    sample_ids = torch.stack(sample_ids)  # [batch_size, 1]

    # 原始ID列表
    original_ids = list(original_ids)  # 保持为字符串列表

    # 原始文本和视频描述列表（用于trimodal encoder）
    text_list = list(texts)
    video_desc_list = list(video_descs)

    return sentences, attention_mask, segments, video_frames, labels, sample_ids, original_ids, text_list, video_desc_list


def get_video_data_loaders(args):
    """
    获取视频+文本数据加载器

    Args:
        args: 配置参数

    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    from pytorch_pretrained_bert import BertTokenizer
    from src.data.helpers import get_labels_and_frequencies, get_vocab
    import functools
    import os  # 确保os模块在函数开始就被导入

    # 获取tokenizer
    try:
        bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        tokenizer = bert_tokenizer.tokenize
    except Exception as e:
        print(f"加载BERT tokenizer失败: {e}")
        print(f"尝试的路径: {args.bert_model}")
        # 尝试使用绝对路径
        abs_path = os.path.abspath(args.bert_model)
        print(f"绝对路径: {abs_path}")
        if os.path.exists(abs_path):
            bert_tokenizer = BertTokenizer.from_pretrained(abs_path, do_lower_case=True)
            tokenizer = bert_tokenizer.tokenize
        else:
            raise Exception(f"BERT模型路径不存在: {abs_path}")

    # 获取标签和词汇表
    train_path = os.path.join(args.data_path, "train.jsonl")
    args.labels, args.label_freqs = get_labels_and_frequencies(train_path)

    # 确保args有model属性用于get_vocab
    if not hasattr(args, 'model'):
        args.model = 'latefusion_video'

    vocab = get_vocab(args)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz
    args.n_classes = len(args.labels)
    
    print(f"数据集标签: {args.labels}")
    print(f"类别数量: {args.n_classes}")
    
    # 获取视频变换
    train_video_transforms = get_video_transforms(is_training=True)
    test_video_transforms = get_video_transforms(is_training=False)
    
    # 创建数据集
    train_dataset = VideoTextDataset(
        data_path=os.path.join(args.data_path, "train.jsonl"),
        tokenizer=tokenizer,
        video_transforms=train_video_transforms,
        vocab=vocab,
        args=args
    )
    args.train_data_len = len(train_dataset)
    
    # 验证集
    val_path = os.path.join(args.data_path, "val.jsonl")
    if os.path.exists(val_path):
        val_dataset = VideoTextDataset(
            data_path=val_path,
            tokenizer=tokenizer,
            video_transforms=test_video_transforms,
            vocab=vocab,
            args=args
        )
    else:
        print("警告: 未找到验证集文件，使用测试集作为验证集")
        val_dataset = None
    
    # 测试集
    test_dataset = VideoTextDataset(
        data_path=os.path.join(args.data_path, "test.jsonl"),
        tokenizer=tokenizer,
        video_transforms=test_video_transforms,
        vocab=vocab,
        args=args
    )
    
    # 创建整理函数
    collate_fn = functools.partial(collate_video_text_fn, args=args)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_sz,
            shuffle=False,
            num_workers=args.n_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
