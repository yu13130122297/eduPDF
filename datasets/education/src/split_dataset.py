#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
教育数据集划分脚本
将train.jsonl文件按比例划分为训练集、验证集和测试集
"""

import json
import random
import os
from collections import Counter
from dotenv import load_dotenv

def load_env_config(env_path):
    """
    加载.env配置文件，获取启用的分类
    
    Args:
        env_path (str): .env文件路径
    
    Returns:
        dict: 分类名称到启用状态的映射
    """
    # 加载.env文件
    load_dotenv(env_path)
    
    # 定义分类映射
    category_mapping = {
        '学生发言': os.getenv('STUDENT_SPEECH_ENABLED', 'true').lower() == 'true',
        '学生讨论': os.getenv('STUDENT_DISCUSSION_ENABLED', 'true').lower() == 'true',
        '技术操作': os.getenv('TECH_OPERATION_ENABLED', 'true').lower() == 'true',
        '教师反馈': os.getenv('TEACHER_FEEDBACK_ENABLED', 'true').lower() == 'true',
        '教师巡视': os.getenv('TEACHER_PATROL_ENABLED', 'true').lower() == 'true',
        '教师指令': os.getenv('TEACHER_INSTRUCTION_ENABLED', 'true').lower() == 'true',
        '教师提问': os.getenv('TEACHER_QUESTION_ENABLED', 'true').lower() == 'true',
        '教师板书': os.getenv('TEACHER_WRITING_ENABLED', 'true').lower() == 'true',
        '教师讲授': os.getenv('TEACHER_LECTURE_ENABLED', 'true').lower() == 'true',
        '课堂沉寂': os.getenv('CLASSROOM_SILENCE_ENABLED', 'true').lower() == 'true'
    }
    
    return category_mapping

def load_jsonl(file_path):
    """
    加载JSONL文件
    
    Args:
        file_path (str): JSONL文件路径
    
    Returns:
        list: 数据列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def filter_data_by_config(data, enabled_categories):
    """
    根据配置过滤数据，只保留启用的分类
    
    Args:
        data (list): 原始数据
        enabled_categories (dict): 启用的分类配置
    
    Returns:
        list: 过滤后的数据
    """
    filtered_data = []
    for item in data:
        label = item['label']
        if label in enabled_categories and enabled_categories[label]:
            filtered_data.append(item)
    
    return filtered_data

def save_jsonl(data, file_path):
    """
    保存数据为JSONL格式
    
    Args:
        data (list): 数据列表
        file_path (str): 保存路径
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def stratified_split(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    按标签分层划分数据集
    
    Args:
        data (list): 原始数据
        train_ratio (float): 训练集比例
        val_ratio (float): 验证集比例
        test_ratio (float): 测试集比例
        random_seed (int): 随机种子
    
    Returns:
        tuple: (训练集, 验证集, 测试集)
    """
    # 设置随机种子确保结果可复现
    random.seed(random_seed)
    
    # 按标签分组
    label_groups = {}
    for item in data:
        label = item['label']
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(item)
    
    train_data = []
    val_data = []
    test_data = []
    
    # 对每个标签分别进行划分
    for label, items in label_groups.items():
        # 打乱数据
        random.shuffle(items)
        
        n_total = len(items)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val  # 剩余的分配给测试集
        
        # 划分数据
        train_data.extend(items[:n_train])
        val_data.extend(items[n_train:n_train + n_val])
        test_data.extend(items[n_train + n_val:])
        
        print(f"标签 '{label}': 总数={n_total}, 训练={n_train}, 验证={n_val}, 测试={n_test}")
    
    # 再次打乱各个数据集
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data

def print_dataset_stats(data, dataset_name):
    """
    打印数据集统计信息
    
    Args:
        data (list): 数据集
        dataset_name (str): 数据集名称
    """
    print(f"\n{dataset_name} 统计信息:")
    print(f"总样本数: {len(data)}")
    
    # 统计标签分布
    label_counts = Counter([item['label'] for item in data])
    print("标签分布:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} ({count/len(data)*100:.1f}%)")

def main():
    """
    主函数
    """
    # 文件路径配置
    input_file = "/datapool/home/2005900028/yzc/PDF/datasets/education/single_train.jsonl"
    env_file = "/datapool/home/2005900028/yzc/PDF/datasets/education/src/.env"
    output_dir = "/datapool/home/2005900028/yzc/PDF/datasets/education"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在")
        return
    
    if not os.path.exists(env_file):
        print(f"错误: 配置文件 {env_file} 不存在")
        return
    
    print("开始加载配置...")
    # 加载配置文件
    enabled_categories = load_env_config(env_file)
    print("启用的分类:")
    for category, enabled in enabled_categories.items():
        status = "启用" if enabled else "禁用"
        print(f"  {category}: {status}")
    
    print("\n开始加载数据...")
    # 加载原始数据
    data = load_jsonl(input_file)
    print(f"成功加载 {len(data)} 条数据")
    
    # 打印原始数据统计
    print_dataset_stats(data, "原始数据集")
    
    # 根据配置过滤数据
    print("\n根据配置过滤数据...")
    filtered_data = filter_data_by_config(data, enabled_categories)
    print(f"过滤后剩余 {len(filtered_data)} 条数据")
    
    # 打印过滤后数据统计
    print_dataset_stats(filtered_data, "过滤后数据集")
    
    # 使用过滤后的数据进行划分
    data = filtered_data
    
    print("\n开始划分数据集...")
    # 划分数据集 (70% 训练, 15% 验证, 15% 测试)
    train_data, val_data, test_data = stratified_split(
        data, 
        train_ratio=0.7, 
        val_ratio=0.15, 
        test_ratio=0.15,
        random_seed=42
    )
    
    # 保存划分后的数据集
    print("\n保存数据集...")
    save_jsonl(train_data, os.path.join(output_dir, "train.jsonl"))
    save_jsonl(val_data, os.path.join(output_dir, "val.jsonl"))
    save_jsonl(test_data, os.path.join(output_dir, "test.jsonl"))
    
    # 打印各数据集统计信息
    print_dataset_stats(train_data, "训练集")
    print_dataset_stats(val_data, "验证集")
    print_dataset_stats(test_data, "测试集")
    
    print("\n数据集划分完成!")
    print(f"训练集: {len(train_data)} 样本 -> train.jsonl")
    print(f"验证集: {len(val_data)} 样本 -> val.jsonl")
    print(f"测试集: {len(test_data)} 样本 -> test.jsonl")

if __name__ == "__main__":
    main()