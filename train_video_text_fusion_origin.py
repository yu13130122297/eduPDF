#!/usr/bin/env python3
#
# 视频+文本多模态动态融合分类训练脚本（增强版）
# 基于原始版本，优化内存使用，增加了详细的训练过程分析和可视化
# 主要改进：
# 1. 优化内存使用策略，包括梯度累积、混合精度训练等
# 2. 增加了详细的训练过程分析和可视化功能
# 3. 提供了更完整的实验结果保存和分析工具
# 4. 新增跨模态注意力、原型网络、焦点损失等增强功能
#

import os
# 设置环境变量解决OpenCV显示问题
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['DISPLAY'] = ''

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import numpy as np
import gc
import matplotlib.pyplot as plt
import seaborn as sns

# 设置字体和解决字体警告
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
import warnings
# 过滤所有matplotlib和seaborn的字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*Glyph.*missing from current font.*')

from src.data.video_dataset import get_video_data_loaders
from src.models.latefusion_video import MultimodalLateFusionClf_video
from src.utils.utils import set_seed
from sklearn.utils.class_weight import compute_class_weight


class FocalLoss(nn.Module):
    """焦点损失，用于处理类别不平衡问题"""

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            focal_loss = at * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    """类别平衡损失"""

    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0):
        super(ClassBalancedLoss, self).__init__()
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(weights)

        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.gamma = gamma

    def forward(self, inputs, targets):
        if self.weights.device != inputs.device:
            self.weights = self.weights.to(inputs.device)

        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, weight=self.weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


def _load_yaml_config(path: str) -> dict:
    """加载 YAML 配置文件，合并 base.yaml（如果存在）"""
    try:
        import yaml
    except ImportError:
        raise ImportError("请安装 PyYAML: pip install pyyaml")

    base_path = os.path.join(os.path.dirname(path), "base.yaml")
    cfg: dict = {}
    if os.path.exists(base_path) and os.path.abspath(base_path) != os.path.abspath(path):
        with open(base_path, "r", encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
    with open(path, "r", encoding="utf-8") as f:
        cfg.update(yaml.safe_load(f) or {})
    return cfg


def get_args():
    """获取命令行参数（内存优化版本）"""
    parser = argparse.ArgumentParser(description="视频+文本多模态动态融合分类训练（内存优化）")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径（如 configs/phase1.yaml）")
    
    # 基础参数（减少内存使用）
    parser.add_argument("--batch_sz", type=int, default=16, help="批次大小")
    parser.add_argument("--bert_model", type=str, default="./bert-base-chinese", help="BERT模型路径")
    parser.add_argument("--data_path", type=str, default="datasets/education/", help="数据路径")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout率")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--max_epochs", type=int, default=100, help="最大训练轮数")
    parser.add_argument("--max_seq_len", type=int, default=128, help="最大序列长度（减小以节省内存）")
    parser.add_argument("--patience", type=int, default=3, help="早停耐心值")
    parser.add_argument("--savedir", type=str, default="checkpoints/", help="模型保存路径")
    parser.add_argument("--seed", type=int, default=123, help="随机种子")
    parser.add_argument("--task", type=str, default="education", help="任务名称")
    parser.add_argument("--task_type", type=str, default="classification", help="任务类型")
    parser.add_argument("--name", type=str, default="video_text_fusion_lite", help="实验名称")
    parser.add_argument("--n_workers", type=int, default=4, help="数据加载器工作进程数")
    
    # 模型参数（减小模型尺寸）
    parser.add_argument("--hidden_sz", type=int, default=768, help="BERT隐藏层大小")
    parser.add_argument("--embed_sz", type=int, default=300, help="词嵌入维度")
    parser.add_argument("--img_hidden_sz", type=int, default=2048, help="图像特征维度")
    parser.add_argument("--num_image_embeds", type=int, default=1, help="图像嵌入数量")
    
    # 视频相关参数（减少内存使用）
    parser.add_argument("--num_frames", type=int, default=8, help="每个视频采样的帧数")
    parser.add_argument("--frame_sampling_strategy", type=str, default="uniform", 
                       choices=["uniform", "random", "center"], help="帧采样策略")
    parser.add_argument("--video_hidden_sz", type=int, default=256, help="视频LSTM隐藏层大小（减小）")
    parser.add_argument("--video_lstm_layers", type=int, default=1, help="视频LSTM层数（减少）")
    parser.add_argument("--video_bidirectional", type=bool, default=False, help="是否使用双向LSTM（关闭以节省内存）")
    parser.add_argument("--video_feature_dim", type=int, default=512, help="视频特征维度（减小）")
    parser.add_argument("--video_pooling_type", type=str, default="mean", 
                       choices=["last", "mean", "max"], help="视频特征池化方式")
    parser.add_argument("--freeze_video_cnn", type=bool, default=True, help="是否冻结视频CNN（冻结以节省内存）")
    parser.add_argument("--video_augmentation", type=bool, default=False, help="是否使用视频数据增强（关闭以节省内存）")
    
    # 融合相关参数
    parser.add_argument("--use_dynamic_fusion", type=bool, default=True, help="是否使用动态融合")
    parser.add_argument("--fusion_loss_weight", type=float, default=1.0, help="融合损失权重")
    parser.add_argument("--text_loss_weight", type=float, default=0.3, help="文本损失权重（减小）")
    parser.add_argument("--video_loss_weight", type=float, default=0.3, help="视频损失权重（减小）")
    parser.add_argument("--prototype_loss_weight", type=float, default=0.2, help="原型损失权重")
    parser.add_argument("--confidence_reg_weight", type=float, default=0.005, help="置信度正则化权重（减小）")

    # 类别不平衡处理参数
    parser.add_argument("--use_focal_loss", type=bool, default=True, help="是否使用焦点损失")
    parser.add_argument("--focal_alpha", type=float, default=0.25, help="焦点损失alpha参数")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="焦点损失gamma参数")
    parser.add_argument("--use_class_weights", type=bool, default=True, help="是否使用类别权重")
    
    # 兼容性参数
    parser.add_argument("--model", type=str, default="latefusion_video", help="模型类型")
    
    # 内存优化参数
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--mixed_precision", type=bool, default=True, help="是否使用混合精度训练")
    
    # Mamba相关参数
    parser.add_argument("--use_mamba_attention", type=bool, default=False, help="是否使用Mamba增强的跨模态注意力")
    parser.add_argument("--mamba_d_state", type=int, default=16, help="Mamba状态空间维度")
    parser.add_argument("--mamba_num_heads", type=int, default=8, help="Mamba注意力头数")
    parser.add_argument("--use_mamba_fusion", type=bool, default=False, help="是否使用Mamba动态融合权重")
    parser.add_argument("--mamba_fusion_hidden", type=int, default=128, help="Mamba融合隐藏层维度")
    parser.add_argument("--mamba_fusion_d_state", type=int, default=8, help="Mamba融合状态空间维度")
    
    # 新模块开关
    parser.add_argument("--use_trimodal", type=bool, default=False, help="启用三模态文本编码器 (FR-1)")
    parser.add_argument("--use_mamba", type=bool, default=False, help="启用 Mamba 视频编码器 (FR-3)")
    parser.add_argument("--use_sparse_moe", type=bool, default=False, help="启用 Sparse MoE 置信度网络 (FR-4)")
    parser.add_argument("--use_hier_fusion", type=bool, default=False, help="启用层次化 MoE 融合 (FR-5)")
    parser.add_argument("--use_prm", type=bool, default=False, help="启用过程奖励模型 (FR-6)")
    parser.add_argument("--use_video_desc", type=bool, default=True, help="是否使用 video_description（默认True）")

    # 先解析已知参数，获取 --config
    args, _ = parser.parse_known_args()

    # 若指定了 YAML，将 YAML 值作为命令行默认值（命令行显式传入的参数仍会覆盖）
    if args.config:
        yaml_cfg = _load_yaml_config(args.config)
        parser.set_defaults(**yaml_cfg)
        print(f"[config] 已加载 {args.config}，共 {len(yaml_cfg)} 个参数")

    return parser.parse_args()


def compute_class_weights_and_samples(train_loader):
    """计算类别权重和每个类别的样本数（使用预设分布）"""
    print("📊 使用预设的类别分布...")

    # 根据提供的分类报告直接设置类别样本数
    # 对应类别：课堂沉寂、教师反馈、教师板书、教师提问、学生发言、教师巡视、教师讲授、教师指令、学生讨论、技术操作
    samples_per_class = [84, 21, 69, 116, 171, 105, 174, 39, 35, 6]
    class_names = ['课堂沉寂', '教师反馈', '教师板书', '教师提问', '学生发言',
                   '教师巡视', '教师讲授', '教师指令', '学生讨论', '技术操作']

    # 计算类别权重（反比于样本数量）
    total_samples = sum(samples_per_class)
    n_classes = len(samples_per_class)

    # 使用balanced权重计算方法
    class_weights = []
    for count in samples_per_class:
        if count > 0:
            weight = total_samples / (n_classes * count)
        else:
            weight = 1.0  # 对于没有样本的类别给予默认权重
        class_weights.append(weight)

    print("类别样本分布:")
    for i, (name, count, weight) in enumerate(zip(class_names, samples_per_class, class_weights)):
        print(f"  类别 {i} ({name}): {count} 样本, 权重: {weight:.4f}")

    return torch.tensor(class_weights, dtype=torch.float32), samples_per_class


def clear_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def create_analysis_directories(base_dir, epoch):
    """创建分析结果的目录结构"""
    epoch_dir = os.path.join(base_dir, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)

    return epoch_dir


def plot_confusion_matrix(y_true, y_pred, labels, epoch, phase, save_dir):
    """绘制并保存混淆矩阵"""
    # 临时抑制所有matplotlib警告
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        cm = confusion_matrix(y_true, y_pred)

        # 使用英文标签避免字体问题
        english_labels = [f'Class_{i}' for i in range(len(labels))]

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=english_labels, yticklabels=english_labels)
        plt.title(f'Confusion Matrix - Epoch {epoch} ({phase})')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

    # 添加标签映射说明（避免中文字体问题，保存到文件）
    label_mapping = '\n'.join([f'Class_{i}: {label}' for i, label in enumerate(labels)])

    # 创建目录结构
    epoch_dir = create_analysis_directories(save_dir, epoch)

    # 将标签映射保存到epoch目录
    mapping_file = os.path.join(epoch_dir, f'label_mapping_{phase}.txt')
    with open(mapping_file, 'w', encoding='utf-8') as f:
        f.write("Label Mapping:\n")
        f.write(label_mapping)

    # 在图上只显示英文说明
    plt.figtext(0.02, 0.02, f'See label_mapping_{phase}.txt for mapping',
                fontsize=8, verticalalignment='bottom')

    # 保存图片到epoch目录
    save_path = os.path.join(epoch_dir, f'confusion_matrix_{phase}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return cm


def analyze_errors(y_true, y_pred, labels, sample_ids, original_ids, epoch, phase, save_dir):
    """分析错误预测的样本"""
    errors = []

    for true_label, pred_label, sample_id, original_id in zip(y_true, y_pred, sample_ids, original_ids):
        if true_label != pred_label:
            errors.append({
                'sample_id': original_id,  # 使用原始数据集ID
                'internal_id': int(sample_id),  # 保留内部索引
                'true_label': labels[int(true_label)],
                'pred_label': labels[int(pred_label)],
                'true_idx': int(true_label),
                'pred_idx': int(pred_label)
            })

    # 创建目录结构并保存错误分析
    epoch_dir = create_analysis_directories(save_dir, epoch)
    error_file = os.path.join(epoch_dir, f'error_analysis_{phase}.json')
    with open(error_file, 'w', encoding='utf-8') as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)

    # 打印错误统计
    if errors:
        print(f"\n{phase} 错误分析 (Epoch {epoch}):")
        print(f"总错误数: {len(errors)}")

        # 按错误类型统计
        error_types = {}
        for error in errors:
            error_type = f"{error['true_label']} -> {error['pred_label']}"
            error_types[error_type] = error_types.get(error_type, 0) + 1

        print("错误类型统计:")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count}")

    return errors


def plot_fusion_weights(text_weights, video_weights, epoch, phase, save_dir):
    """绘制融合权重分布"""
    # 临时抑制所有matplotlib警告
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        plt.figure(figsize=(12, 4))

        # 文本权重分布
        plt.subplot(1, 3, 1)
        plt.hist(text_weights, bins=20, alpha=0.7, color='blue', label='Text Weight')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.title(f'Text Weight Distribution - Epoch {epoch}')
        plt.legend()

        # 视频权重分布
        plt.subplot(1, 3, 2)
        plt.hist(video_weights, bins=20, alpha=0.7, color='red', label='Video Weight')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.title(f'Video Weight Distribution - Epoch {epoch}')
        plt.legend()

        # 权重对比
        plt.subplot(1, 3, 3)
        plt.scatter(text_weights, video_weights, alpha=0.6)
        plt.xlabel('Text Weight')
        plt.ylabel('Video Weight')
        plt.title(f'Weight Comparison - Epoch {epoch}')
        plt.plot([0, 1], [1, 0], 'r--', alpha=0.5, label='Sum=1')
        plt.legend()

        plt.tight_layout()

        # 创建目录结构并保存图片
        epoch_dir = create_analysis_directories(save_dir, epoch)
        save_path = os.path.join(epoch_dir, f'fusion_weights_{phase}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    # 打印权重统计
    print(f"\n{phase} 融合权重统计 (Epoch {epoch}):")
    print(f"文本权重 - 均值: {np.mean(text_weights):.3f}, 标准差: {np.std(text_weights):.3f}")
    print(f"视频权重 - 均值: {np.mean(video_weights):.3f}, 标准差: {np.std(video_weights):.3f}")
    print(f"文本权重占优样本比例: {np.mean(np.array(text_weights) > np.array(video_weights)):.3f}")


def plot_class_specific_fusion_weights(txt_confidences, video_confidences, class_probs, labels, label_names, epoch, phase, save_dir):
    """绘制类别特定的融合权重分布"""
    # 临时抑制所有matplotlib警告
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        n_classes = len(label_names)

        # 创建更大的图形来容纳类别特定的分析
        fig, axes = plt.subplots(2, n_classes + 1, figsize=(4 * (n_classes + 1), 8))
        if n_classes == 1:
            axes = axes.reshape(2, -1)

        # 转换为numpy数组便于处理
        try:
            txt_confidences_np = txt_confidences.squeeze().cpu().numpy()  # [batch_size, n_classes]
            video_confidences_np = video_confidences.squeeze().cpu().numpy()  # [batch_size, n_classes]
            class_probs_np = class_probs.cpu().numpy()  # [batch_size, n_classes]
        except Exception as e:
            print(f"❌ 数据转换失败: {e}")
            return

        # 检查数据形状
        if len(txt_confidences_np.shape) != 2 or len(video_confidences_np.shape) != 2:
            print(f"❌ 置信度数据形状错误: txt={txt_confidences_np.shape}, video={video_confidences_np.shape}")
            return

        # 确保标签数量与置信度数据匹配
        n_samples = txt_confidences_np.shape[0]
        if n_samples == 0:
            print("⚠️ 没有置信度数据可供分析")
            return

        if len(labels) > n_samples:
            # 如果标签数量多于置信度数据，只使用前n_samples个标签
            labels_np = np.array(labels[:n_samples])
            print(f"⚠️ 标签数量({len(labels)})多于置信度数据({n_samples})，使用前{n_samples}个标签进行分析")
        else:
            labels_np = np.array(labels)

        # 检查是否有有效的标签
        if len(labels_np) == 0:
            print("⚠️ 没有标签数据可供分析")
            return

        # 为每个类别绘制权重分布
        for class_idx in range(n_classes):
            class_name = label_names[class_idx]

            # 获取该类别的样本
            class_mask = labels_np == class_idx
            if np.sum(class_mask) == 0:
                continue

            class_txt_conf = txt_confidences_np[class_mask, class_idx]
            class_video_conf = video_confidences_np[class_mask, class_idx]

            # 计算该类别的权重（简化版本，实际权重计算更复杂）
            total_conf = class_txt_conf + class_video_conf + 1e-8
            class_txt_weights = class_txt_conf / total_conf
            class_video_weights = class_video_conf / total_conf

            # 第一行：文本权重分布
            axes[0, class_idx].hist(class_txt_weights, bins=15, alpha=0.7, color='blue',
                                   label=f'Text Weight\n(n={len(class_txt_weights)})')
            axes[0, class_idx].set_xlabel('Text Weight')
            axes[0, class_idx].set_ylabel('Frequency')
            axes[0, class_idx].set_title(f'{class_name}\nText Weights')
            axes[0, class_idx].legend()
            axes[0, class_idx].grid(True, alpha=0.3)

            # 第二行：视频权重分布
            axes[1, class_idx].hist(class_video_weights, bins=15, alpha=0.7, color='red',
                                   label=f'Video Weight\n(n={len(class_video_weights)})')
            axes[1, class_idx].set_xlabel('Video Weight')
            axes[1, class_idx].set_ylabel('Frequency')
            axes[1, class_idx].set_title(f'{class_name}\nVideo Weights')
            axes[1, class_idx].legend()
            axes[1, class_idx].grid(True, alpha=0.3)

        # 最后一列：整体对比
        all_txt_weights = []
        all_video_weights = []
        all_class_labels = []

        for class_idx in range(n_classes):
            class_mask = labels_np == class_idx
            if np.sum(class_mask) == 0:
                continue

            class_txt_conf = txt_confidences_np[class_mask, class_idx]
            class_video_conf = video_confidences_np[class_mask, class_idx]

            total_conf = class_txt_conf + class_video_conf + 1e-8
            class_txt_weights = class_txt_conf / total_conf
            class_video_weights = class_video_conf / total_conf

            all_txt_weights.extend(class_txt_weights)
            all_video_weights.extend(class_video_weights)
            all_class_labels.extend([class_idx] * len(class_txt_weights))

        # 整体权重对比散点图
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for class_idx in range(n_classes):
            class_mask = np.array(all_class_labels) == class_idx
            if np.sum(class_mask) == 0:
                continue
            # 安全地获取颜色，避免索引越界
            color = colors[class_idx % len(colors)]
            axes[0, -1].scatter(np.array(all_txt_weights)[class_mask],
                              np.array(all_video_weights)[class_mask],
                              alpha=0.6, color=color,
                              label=label_names[class_idx], s=20)

        axes[0, -1].set_xlabel('Text Weight')
        axes[0, -1].set_ylabel('Video Weight')
        axes[0, -1].set_title('Class-Specific Weight Comparison')
        axes[0, -1].plot([0, 1], [1, 0], 'k--', alpha=0.5, label='Sum=1')
        axes[0, -1].legend()
        axes[0, -1].grid(True, alpha=0.3)

        # 类别预测概率分布
        for class_idx in range(n_classes):
            class_probs_for_class = class_probs_np[:, class_idx]
            # 安全地获取颜色，避免索引越界
            color = colors[class_idx % len(colors)]
            axes[1, -1].hist(class_probs_for_class, bins=15, alpha=0.6,
                           color=color, label=label_names[class_idx])

        axes[1, -1].set_xlabel('Predicted Class Probability')
        axes[1, -1].set_ylabel('Frequency')
        axes[1, -1].set_title('Class Prediction Distribution')
        axes[1, -1].legend()
        axes[1, -1].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片
        epoch_dir = create_analysis_directories(save_dir, epoch)
        save_path = os.path.join(epoch_dir, f'class_specific_fusion_weights_{phase}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        # 打印类别特定的权重统计
        print(f"\n{phase} 类别特定融合权重统计 (Epoch {epoch}):")
        for class_idx in range(n_classes):
            class_mask = labels_np == class_idx
            if np.sum(class_mask) == 0:
                continue

            class_txt_conf = txt_confidences_np[class_mask, class_idx]
            class_video_conf = video_confidences_np[class_mask, class_idx]

            total_conf = class_txt_conf + class_video_conf + 1e-8
            class_txt_weights = class_txt_conf / total_conf
            class_video_weights = class_video_conf / total_conf

            print(f"  {label_names[class_idx]}:")
            print(f"    文本权重 - 均值: {np.mean(class_txt_weights):.3f}, 标准差: {np.std(class_txt_weights):.3f}")
            print(f"    视频权重 - 均值: {np.mean(class_video_weights):.3f}, 标准差: {np.std(class_video_weights):.3f}")
            print(f"    文本权重占优比例: {np.mean(class_txt_weights > class_video_weights):.3f}")
            print(f"    样本数量: {len(class_txt_weights)}")


def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def save_detailed_results(results, save_dir):
    """保存详细的训练结果"""
    results_file = os.path.join(save_dir, 'detailed_results.json')

    # 转换numpy类型为Python原生类型
    converted_results = convert_numpy_types(results)

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(converted_results, f, indent=2, ensure_ascii=False)

    print(f"详细结果已保存到: {results_file}")


def get_next_experiment_dir(base_dir, experiment_name):
    """获取下一个可用的实验目录名"""
    # 如果基础目录不存在，创建第一个实验目录
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    # 查找现有的实验目录
    existing_dirs = []
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and item.startswith(experiment_name):
                existing_dirs.append(item)

    # 如果没有现有目录，创建第一个
    if not existing_dirs:
        new_dir = os.path.join(base_dir, experiment_name + "1")
        return new_dir, 1

    # 找到最大的编号
    max_num = 0
    for dir_name in existing_dirs:
        # 提取编号部分
        suffix = dir_name[len(experiment_name):]
        if suffix.isdigit():
            max_num = max(max_num, int(suffix))

    # 创建下一个编号的目录
    next_num = max_num + 1
    new_dir = os.path.join(base_dir, experiment_name + str(next_num))
    return new_dir, next_num


def print_epoch_summary(epoch, save_dir):
    """打印每个epoch的文件组织结构"""
    epoch_dir = os.path.join(save_dir, f'epoch_{epoch}')

    print(f"\n📁 Epoch {epoch} 文件:")
    if os.path.exists(epoch_dir):
        files = os.listdir(epoch_dir)
        if files:
            for file in sorted(files):
                print(f"  ✓ {file}")
        else:
            print("  (暂无文件)")
    else:
        print("  (目录不存在)")


def save_predictions_json(predictions, labels, sample_ids, original_ids, label_names, epoch, phase, save_dir):
    """保存预测结果为JSON文件"""
    predictions_data = []

    for pred_idx, true_idx, sample_id, original_id in zip(predictions, labels, sample_ids, original_ids):
        prediction_item = {
            'sample_id': original_id,  # 使用原始数据集ID
            'internal_id': int(sample_id),  # 保留内部索引用于调试
            'true_label_idx': int(true_idx),
            'pred_label_idx': int(pred_idx),
            'true_label_name': label_names[true_idx],
            'pred_label_name': label_names[pred_idx],
            'is_correct': bool(pred_idx == true_idx)
        }
        predictions_data.append(prediction_item)

    # 创建目录结构并保存预测结果
    epoch_dir = create_analysis_directories(save_dir, epoch)
    predictions_file = os.path.join(epoch_dir, f'predictions_{phase}.json')
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(predictions_data, f, indent=2, ensure_ascii=False)

    # 统计信息
    total_samples = len(predictions_data)
    correct_predictions = sum(1 for item in predictions_data if item['is_correct'])
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0

    print(f"预测结果已保存到: {predictions_file}")
    print(f"样本总数: {total_samples}, 正确预测: {correct_predictions}, 准确率: {accuracy:.4f}")

    return predictions_file, predictions_data


def train_epoch_memory_efficient(model, train_loader, optimizer, criterion, focal_criterion, args, epoch, scaler=None):
    """内存高效的训练函数（增强版）"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_sample_ids = []
    all_original_ids = []
    all_text_weights = []
    all_video_weights = []
    all_txt_confidences = []
    all_video_confidences = []
    all_class_probs = []

    pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch}", mininterval=2.0, maxinterval=10.0)

    for batch_idx, batch in enumerate(pbar):
        sentences, attention_mask, segments, video_frames, labels, sample_ids, original_ids, text_list, video_desc_list = batch

        # 移动到GPU
        if torch.cuda.is_available():
            sentences = sentences.cuda(non_blocking=True)
            attention_mask = attention_mask.cuda(non_blocking=True)
            segments = segments.cuda(non_blocking=True)
            video_frames = video_frames.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        
        # 使用混合精度训练
        if args.mixed_precision and scaler is not None:
            with torch.cuda.amp.autocast():
                # 前向传播
                if args.use_dynamic_fusion:
                    model_output = model(
                        sentences, attention_mask, segments, video_frames, choice='train',
                        text_list=text_list, video_desc_list=video_desc_list
                    )

                    # 处理不同版本的模型输出（增强版）
                    if len(model_output) == 5:
                        # 原始版本：全局权重
                        fused_logits, text_logits, video_logits, text_conf, video_conf = model_output
                        class_probs = None
                        txt_proto_out = video_proto_out = None
                    elif len(model_output) == 6:
                        # 类别特定版本
                        fused_logits, text_logits, video_logits, text_conf, video_conf, class_probs = model_output
                        txt_proto_out = video_proto_out = None
                    elif len(model_output) == 8:
                        # 增强版本：包含原型网络输出
                        fused_logits, text_logits, video_logits, text_conf, video_conf, class_probs, txt_proto_out, video_proto_out = model_output
                    else:
                        raise ValueError(f"Unexpected model output length: {len(model_output)}")

                    # 计算损失（使用焦点损失处理类别不平衡）
                    if args.use_focal_loss:
                        fusion_loss = focal_criterion(fused_logits, labels)
                        text_loss = focal_criterion(text_logits, labels)
                        video_loss = focal_criterion(video_logits, labels)
                    else:
                        fusion_loss = criterion(fused_logits, labels)
                        text_loss = criterion(text_logits, labels)
                        video_loss = criterion(video_logits, labels)

                    # 原型损失（如果有原型网络输出）
                    prototype_loss = 0
                    if txt_proto_out is not None and video_proto_out is not None:
                        if args.use_focal_loss:
                            txt_proto_loss = focal_criterion(txt_proto_out, labels)
                            video_proto_loss = focal_criterion(video_proto_out, labels)
                        else:
                            txt_proto_loss = criterion(txt_proto_out, labels)
                            video_proto_loss = criterion(video_proto_out, labels)
                        prototype_loss = 0.5 * (txt_proto_loss + video_proto_loss)

                    # 置信度正则化损失（处理类别特定的置信度）
                    if text_conf.dim() == 3:  # 类别特定版本 [batch_size, 1, n_classes]
                        # 对所有类别的置信度进行正则化
                        conf_reg_loss = (torch.mean(torch.abs(text_conf - 0.5)) +
                                       torch.mean(torch.abs(video_conf - 0.5)))
                    else:  # 原始版本 [batch_size, 1]
                        conf_reg_loss = (torch.mean(torch.abs(text_conf - 0.5)) +
                                       torch.mean(torch.abs(video_conf - 0.5)))

                    # 总损失（包含原型损失）
                    loss = (args.fusion_loss_weight * fusion_loss +
                           args.text_loss_weight * text_loss +
                           args.video_loss_weight * video_loss +
                           args.prototype_loss_weight * prototype_loss +
                           args.confidence_reg_weight * conf_reg_loss)
                    
                    # 梯度累积
                    loss = loss / args.gradient_accumulation_steps
                    preds = torch.argmax(fused_logits, dim=1)
                else:
                    fused_logits, text_logits, video_logits = model(
                        sentences, attention_mask, segments, video_frames, choice='static'
                    )
                    loss = criterion(fused_logits, labels) / args.gradient_accumulation_steps
                    preds = torch.argmax(fused_logits, dim=1)
            
            # 反向传播（混合精度）
            scaler.scale(loss).backward()
            
            # 梯度累积
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
        else:
            # 标准精度训练
            if args.use_dynamic_fusion:
                model_output = model(
                    sentences, attention_mask, segments, video_frames, choice='train',
                    text_list=text_list, video_desc_list=video_desc_list
                )

                # 处理不同版本的模型输出
                if len(model_output) == 5:
                    # 原始版本：全局权重
                    fused_logits, text_logits, video_logits, text_conf, video_conf = model_output
                    class_probs = None
                elif len(model_output) == 6:
                    # 类别特定版本
                    fused_logits, text_logits, video_logits, text_conf, video_conf, class_probs = model_output
                else:
                    raise ValueError(f"Unexpected model output length: {len(model_output)}")

                # 计算损失
                fusion_loss = criterion(fused_logits, labels)
                text_loss = criterion(text_logits, labels)
                video_loss = criterion(video_logits, labels)

                # 置信度正则化损失（处理类别特定的置信度）
                if text_conf.dim() == 3:  # 类别特定版本 [batch_size, 1, n_classes]
                    # 对所有类别的置信度进行正则化
                    conf_reg_loss = (torch.mean(torch.abs(text_conf - 0.5)) +
                                   torch.mean(torch.abs(video_conf - 0.5)))
                else:  # 原始版本 [batch_size, 1]
                    conf_reg_loss = (torch.mean(torch.abs(text_conf - 0.5)) +
                                   torch.mean(torch.abs(video_conf - 0.5)))

                # 总损失
                loss = (args.fusion_loss_weight * fusion_loss +
                       args.text_loss_weight * text_loss +
                       args.video_loss_weight * video_loss +
                       args.confidence_reg_weight * conf_reg_loss)
                
                # 梯度累积
                loss = loss / args.gradient_accumulation_steps
                preds = torch.argmax(fused_logits, dim=1)
            else:
                fused_logits, text_logits, video_logits = model(
                    sentences, attention_mask, segments, video_frames, choice='static'
                )
                loss = criterion(fused_logits, labels) / args.gradient_accumulation_steps
                preds = torch.argmax(fused_logits, dim=1)
            
            # 反向传播
            loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # 统计
        total_loss += loss.item() * args.gradient_accumulation_steps
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_sample_ids.extend(sample_ids.cpu().numpy().flatten())
        all_original_ids.extend(original_ids)

        # 收集权重信息（仅在动态融合时）
        if args.use_dynamic_fusion and batch_idx % 5 == 0:  # 每5个batch收集一次权重
            model.eval()
            with torch.no_grad():
                test_outputs = model(sentences, attention_mask, segments, video_frames, choice='test',
                                  text_list=text_list, video_desc_list=video_desc_list)
                # 处理不同版本的输出
                if len(test_outputs) == 7:
                    # 原始版本：全局权重
                    text_weights = test_outputs[5].cpu().numpy().flatten()
                    video_weights = test_outputs[6].cpu().numpy().flatten()
                    all_text_weights.extend(text_weights)
                    all_video_weights.extend(video_weights)
                elif len(test_outputs) == 8:
                    # 类别特定版本 - 只收集权重，置信度数据在最后统一收集
                    text_weights = test_outputs[5].cpu().numpy().flatten()
                    video_weights = test_outputs[6].cpu().numpy().flatten()
                    all_text_weights.extend(text_weights)
                    all_video_weights.extend(video_weights)
            model.train()

        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item() * args.gradient_accumulation_steps:.4f}',
            'Avg_Loss': f'{total_loss/(batch_idx+1):.4f}'
        })

        # 定期清理内存
        if batch_idx % 10 == 0:
            clear_memory()
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    avg_loss = total_loss / len(train_loader)

    # 如果是类别特定模型且需要详细分析，收集完整的置信度数据
    if args.use_dynamic_fusion and len(all_txt_confidences) == 0:
        # 重新遍历数据集收集类别特定信息
        print("📊 收集类别特定权重数据...")
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= 10:  # 只收集前10个batch的数据用于分析
                    break

                sentences, attention_mask, segments, video_frames, labels, sample_ids, original_ids, text_list, video_desc_list = batch

                if torch.cuda.is_available():
                    sentences = sentences.cuda(non_blocking=True)
                    attention_mask = attention_mask.cuda(non_blocking=True)
                    segments = segments.cuda(non_blocking=True)
                    video_frames = video_frames.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)

                test_outputs = model(sentences, attention_mask, segments, video_frames, choice='test',
                                  text_list=text_list, video_desc_list=video_desc_list)

                if len(test_outputs) == 8:
                    # 类别特定版本
                    txt_confidences = test_outputs[3].cpu().numpy()  # [batch_size, 1, n_classes]
                    video_confidences = test_outputs[4].cpu().numpy()  # [batch_size, 1, n_classes]
                    class_probs = test_outputs[7].cpu().numpy()  # [batch_size, n_classes]

                    all_txt_confidences.append(txt_confidences)
                    all_video_confidences.append(video_confidences)
                    all_class_probs.append(class_probs)

                clear_memory()

    # 返回详细信息
    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'predictions': all_preds,
        'labels': all_labels,
        'sample_ids': all_sample_ids,
        'original_ids': all_original_ids,
        'text_weights': all_text_weights,
        'video_weights': all_video_weights,
        'txt_confidences': all_txt_confidences,
        'video_confidences': all_video_confidences,
        'class_probs': all_class_probs
    }

    return results


def evaluate_memory_efficient(model, val_loader, criterion, args):
    """内存高效的评估函数"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_sample_ids = []
    all_original_ids = []
    all_text_weights = []
    all_video_weights = []
    all_txt_confidences = []
    all_video_confidences = []
    all_class_probs = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证", mininterval=2.0, maxinterval=10.0):
            sentences, attention_mask, segments, video_frames, labels, sample_ids, original_ids, text_list, video_desc_list = batch

            # 移动到GPU
            if torch.cuda.is_available():
                sentences = sentences.cuda(non_blocking=True)
                attention_mask = attention_mask.cuda(non_blocking=True)
                segments = segments.cuda(non_blocking=True)
                video_frames = video_frames.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            
            # 前向传播
            if args.use_dynamic_fusion:
                outputs = model(sentences, attention_mask, segments, video_frames, choice='test',
                             text_list=text_list, video_desc_list=video_desc_list)
                fused_logits = outputs[0]

                # 收集权重信息（处理不同版本的输出）
                if len(outputs) == 7:
                    # 原始版本：全局权重
                    text_weights = outputs[5].cpu().numpy().flatten()
                    video_weights = outputs[6].cpu().numpy().flatten()
                    all_text_weights.extend(text_weights)
                    all_video_weights.extend(video_weights)
                elif len(outputs) == 8:
                    # 类别特定版本
                    txt_confidences = outputs[3].cpu().numpy()  # [batch_size, 1, n_classes]
                    video_confidences = outputs[4].cpu().numpy()  # [batch_size, 1, n_classes]
                    text_weights = outputs[5].cpu().numpy().flatten()
                    video_weights = outputs[6].cpu().numpy().flatten()
                    class_probs = outputs[7].cpu().numpy()  # [batch_size, n_classes]

                    all_text_weights.extend(text_weights)
                    all_video_weights.extend(video_weights)
                    all_txt_confidences.append(txt_confidences)
                    all_video_confidences.append(video_confidences)
                    all_class_probs.append(class_probs)
            else:
                fused_logits, _, _ = model(sentences, attention_mask, segments, video_frames, choice='static')

            # 计算损失
            loss = criterion(fused_logits, labels)
            total_loss += loss.item()

            # 预测
            preds = torch.argmax(fused_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_sample_ids.extend(sample_ids.cpu().numpy().flatten())
            all_original_ids.extend(original_ids)

            # 清理内存
            clear_memory()
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    avg_loss = total_loss / len(val_loader)

    # 返回详细信息
    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'predictions': all_preds,
        'labels': all_labels,
        'sample_ids': all_sample_ids,
        'original_ids': all_original_ids,
        'text_weights': all_text_weights,
        'video_weights': all_video_weights,
        'txt_confidences': all_txt_confidences,
        'video_confidences': all_video_confidences,
        'class_probs': all_class_probs
    }

    return results


def main():
    """主训练函数"""
    args = get_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建带编号的实验目录
    base_checkpoint_dir = args.savedir
    experiment_dir, experiment_num = get_next_experiment_dir(base_checkpoint_dir, args.name)
    args.savedir = experiment_dir
    os.makedirs(args.savedir, exist_ok=True)

    print(f"🆔 实验编号: {experiment_num}")
    print(f"📁 实验目录: {args.savedir}")
    
    # 保存配置
    with open(os.path.join(args.savedir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    
    print("开始训练视频+文本多模态融合模型（增强版）")
    print("🚀 新增功能:")
    print("  ✓ 跨模态注意力机制")
    print("  ✓ 原型网络增强小样本学习")
    print("  ✓ 焦点损失处理类别不平衡")
    print("  ✓ 类别感知特征增强")
    print("  ✓ 动态融合权重优化")
    print(f"配置: {vars(args)}")
    
    # 清理初始内存
    clear_memory()
    
    # 获取数据加载器
    try:
        train_loader, val_loader, test_loader = get_video_data_loaders(args)
        print(f"数据加载成功: 训练集{len(train_loader.dataset)}, 测试集{len(test_loader.dataset)}")
        if val_loader:
            print(f"验证集: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 计算类别权重和样本分布
    print("\n📊 分析类别分布...")
    class_weights, samples_per_class = compute_class_weights_and_samples(train_loader)
    
    # 创建模型
    model = MultimodalLateFusionClf_video(args)
    if torch.cuda.is_available():
        model = model.cuda()
        class_weights = class_weights.cuda()
        print("模型已移动到GPU")

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=class_weights if args.use_class_weights else None)

    # 焦点损失
    focal_criterion = None
    if args.use_focal_loss:
        focal_alpha = class_weights if args.use_class_weights else None
        focal_criterion = FocalLoss(alpha=focal_alpha, gamma=args.focal_gamma)
        print(f"✓ 使用焦点损失 (gamma={args.focal_gamma})")

    # 类别平衡损失
    cb_criterion = None
    if len(samples_per_class) > 0:
        cb_criterion = ClassBalancedLoss(samples_per_class, beta=0.9999, gamma=args.focal_gamma)
        print(f"✓ 使用类别平衡损失")

    # 选择最终的损失函数
    if args.use_focal_loss and focal_criterion is not None:
        final_criterion = focal_criterion
        print("📊 使用焦点损失作为主要损失函数")
    elif cb_criterion is not None:
        final_criterion = cb_criterion
        print("📊 使用类别平衡损失作为主要损失函数")
    else:
        final_criterion = criterion
        print("📊 使用标准交叉熵损失")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and torch.cuda.is_available() else None
    
    # 训练循环
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    detailed_results = []

    for epoch in range(1, args.max_epochs + 1):
        print(f"\nEpoch {epoch}/{args.max_epochs}")
        print("=" * 60)

        # 训练
        train_results = train_epoch_memory_efficient(
            model, train_loader, optimizer, criterion, final_criterion, args, epoch, scaler
        )
        print(f"训练 - Loss: {train_results['loss']:.4f}, Acc: {train_results['accuracy']:.4f}, F1: {train_results['f1']:.4f}")

        # 训练阶段分析
        print("\n📊 训练阶段分析:")

        # 保存预测结果JSON
        save_predictions_json(
            train_results['predictions'], train_results['labels'],
            train_results['sample_ids'], train_results['original_ids'],
            args.labels, epoch, 'train', args.savedir
        )

        # 混淆矩阵
        plot_confusion_matrix(
            train_results['labels'], train_results['predictions'],
            args.labels, epoch, 'train', args.savedir
        )
        print(f"训练混淆矩阵已保存")

        # 错误分析
        analyze_errors(
            train_results['labels'], train_results['predictions'],
            args.labels, train_results['sample_ids'], train_results['original_ids'],
            epoch, 'train', args.savedir
        )

        # 权重分析
        if args.use_dynamic_fusion and train_results['text_weights']:
            # 检查是否有类别特定的数据
            if train_results['txt_confidences'] and train_results['class_probs']:
                try:
                    # 类别特定权重分析
                    import numpy as np
                    txt_confidences = np.concatenate(train_results['txt_confidences'], axis=0)
                    video_confidences = np.concatenate(train_results['video_confidences'], axis=0)
                    class_probs = np.concatenate(train_results['class_probs'], axis=0)

                    plot_class_specific_fusion_weights(
                        torch.from_numpy(txt_confidences), torch.from_numpy(video_confidences),
                        torch.from_numpy(class_probs), train_results['labels'], args.labels,
                        epoch, 'train', args.savedir
                    )
                except Exception as e:
                    print(f"⚠️ 类别特定权重分析失败: {e}")
                    print("📊 使用全局权重分析作为备选")
                    plot_fusion_weights(
                        train_results['text_weights'], train_results['video_weights'],
                        epoch, 'train', args.savedir
                    )
            else:
                # 全局权重分析
                plot_fusion_weights(
                    train_results['text_weights'], train_results['video_weights'],
                    epoch, 'train', args.savedir
                )

        # 打印文件结构总结
        print_epoch_summary(epoch, args.savedir)

        # 验证
        if val_loader:
            val_results = evaluate_memory_efficient(model, val_loader, criterion, args)
            print(f"\n验证 - Loss: {val_results['loss']:.4f}, Acc: {val_results['accuracy']:.4f}, F1: {val_results['f1']:.4f}")

            # 验证阶段分析
            print("\n📊 验证阶段分析:")

            # 保存验证预测结果JSON
            save_predictions_json(
                val_results['predictions'], val_results['labels'],
                val_results['sample_ids'], val_results['original_ids'],
                args.labels, epoch, 'val', args.savedir
            )

            # 混淆矩阵
            plot_confusion_matrix(
                val_results['labels'], val_results['predictions'],
                args.labels, epoch, 'val', args.savedir
            )
            print(f"验证混淆矩阵已保存")

            # 错误分析
            analyze_errors(
                val_results['labels'], val_results['predictions'],
                args.labels, val_results['sample_ids'], val_results['original_ids'],
                epoch, 'val', args.savedir
            )

            # 权重分析
            if args.use_dynamic_fusion and val_results['text_weights']:
                # 检查是否有类别特定的数据
                if val_results['txt_confidences'] and val_results['class_probs']:
                    try:
                        # 类别特定权重分析
                        import numpy as np
                        txt_confidences = np.concatenate(val_results['txt_confidences'], axis=0)
                        video_confidences = np.concatenate(val_results['video_confidences'], axis=0)
                        class_probs = np.concatenate(val_results['class_probs'], axis=0)

                        plot_class_specific_fusion_weights(
                            torch.from_numpy(txt_confidences), torch.from_numpy(video_confidences),
                            torch.from_numpy(class_probs), val_results['labels'], args.labels,
                            epoch, 'val', args.savedir
                        )
                    except Exception as e:
                        print(f"⚠️ 验证阶段类别特定权重分析失败: {e}")
                        print("📊 使用全局权重分析作为备选")
                        plot_fusion_weights(
                            val_results['text_weights'], val_results['video_weights'],
                            epoch, 'val', args.savedir
                        )
                else:
                    # 全局权重分析
                    plot_fusion_weights(
                        val_results['text_weights'], val_results['video_weights'],
                        epoch, 'val', args.savedir
                    )

            # 保存最佳模型
            if val_results['f1'] > best_val_f1:
                best_val_f1 = val_results['f1']
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(args.savedir, 'best_model.pth'))
                print(f"🎯🎯🎯🎯🎯 保存最佳模型 (F1: {best_val_f1:.4f})")

                # 创建best_epoch目录并复制最佳epoch的分析文件
                best_epoch_dir = os.path.join(args.savedir, 'best_epoch')
                if os.path.exists(best_epoch_dir):
                    import shutil
                    shutil.rmtree(best_epoch_dir)
                os.makedirs(best_epoch_dir, exist_ok=True)

                # 复制当前epoch的所有分析文件到best_epoch目录
                current_epoch_dir = os.path.join(args.savedir, f'epoch_{epoch}')
                if os.path.exists(current_epoch_dir):
                    import shutil
                    for file in os.listdir(current_epoch_dir):
                        src = os.path.join(current_epoch_dir, file)
                        dst = os.path.join(best_epoch_dir, file)
                        shutil.copy2(src, dst)
                    print(f"📁 已复制最佳epoch分析文件到 best_epoch/")
            else:
                patience_counter += 1

            # 早停
            if patience_counter >= args.patience:
                print(f"⏹️ 早停触发 (patience: {args.patience})")
                break

            # 保存详细结果
            epoch_results = {
                'epoch': epoch,
                'train': train_results,
                'val': val_results
            }
        else:
            # 没有验证集时直接保存
            torch.save(model.state_dict(), os.path.join(args.savedir, f'model_epoch_{epoch}.pth'))
            epoch_results = {
                'epoch': epoch,
                'train': train_results
            }

        detailed_results.append(epoch_results)

        # 清理内存
        clear_memory()
        print("=" * 60)
    
    # 最终测试
    print("\n🎯 最终测试阶段")
    print("=" * 60)

    if val_loader:
        model.load_state_dict(torch.load(os.path.join(args.savedir, 'best_model.pth')))
        print("已加载最佳模型进行测试")

    test_results = evaluate_memory_efficient(model, test_loader, criterion, args)
    print(f"最终测试结果 - Loss: {test_results['loss']:.4f}, Acc: {test_results['accuracy']:.4f}, F1: {test_results['f1']:.4f}")

    # 最终测试分析
    print("\n📊 最终测试分析:")

    # 保存最终测试预测结果JSON
    save_predictions_json(
        test_results['predictions'], test_results['labels'],
        test_results['sample_ids'], test_results['original_ids'],
        args.labels, 'final', 'test', args.savedir
    )

    # 混淆矩阵
    test_cm = plot_confusion_matrix(
        test_results['labels'], test_results['predictions'],
        args.labels, 'final', 'test', args.savedir
    )

    # 错误分析
    analyze_errors(
        test_results['labels'], test_results['predictions'],
        args.labels, test_results['sample_ids'], test_results['original_ids'],
        'final', 'test', args.savedir
    )

    # 权重分析
    if args.use_dynamic_fusion and test_results['text_weights']:
        plot_fusion_weights(
            test_results['text_weights'], test_results['video_weights'],
            'final', 'test', args.savedir
        )

    # 分类报告
    print("\n📋 详细分类报告:")
    report = classification_report(
        test_results['labels'], test_results['predictions'],
        target_names=args.labels, digits=4
    )
    print(report)

    # 保存分类报告
    with open(os.path.join(args.savedir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

    # 保存结果
    final_results = {
        'test_loss': test_results['loss'],
        'test_accuracy': test_results['accuracy'],
        'test_f1': test_results['f1'],
        'best_val_f1': best_val_f1 if val_loader else None,
        'confusion_matrix': test_cm.tolist(),
        'classification_report': report
    }

    with open(os.path.join(args.savedir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    # 保存详细的训练历史
    save_detailed_results(detailed_results, args.savedir)

    print(f"\n🎉 训练完成！")
    print("=" * 60)
    print(f"🆔 实验编号: {experiment_num}")
    print(f"📁 实验目录: {args.savedir}")
    print(f"📊 最终测试结果:")
    print(f"   - 准确率: {test_results['accuracy']:.4f}")
    print(f"   - F1分数: {test_results['f1']:.4f}")
    print(f"   - 损失: {test_results['loss']:.4f}")
    if val_loader and best_val_f1 > 0:
        print(f"   - 最佳验证F1: {best_val_f1:.4f} (Epoch {best_epoch})")
        print(f"   - 最佳epoch分析文件: best_epoch/")

    print(f"\n📂 生成的文件:")
    print(f"   - 模型权重: best_model.pth")
    print(f"   - 配置文件: config.json")
    print(f"   - 结果文件: results.json")
    print(f"   - 分类报告: classification_report.txt")
    print(f"   - 详细历史: detailed_results.json")

    print(f"\n📊 每个epoch的分析文件:")
    print(f"   - 混淆矩阵图表")
    print(f"   - 预测结果JSON（包含原始ID）")
    print(f"   - 错误分析JSON")
    print(f"   - 融合权重分布图")
    print(f"   - 标签映射文件")

    print(f"\n💡 查看结果:")
    print(f"   cd {args.savedir}")
    print(f"   ls -la  # 查看所有文件")
    print(f"   ls epoch_*/  # 查看各epoch的分析文件")
    print("=" * 60)


if __name__ == "__main__":
    main()
