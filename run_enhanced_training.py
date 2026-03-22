#!/usr/bin/env python3
#
# 增强版训练启动脚本
# 专门针对"技术操作"等小样本类别进行优化
#

import subprocess
import sys
import os

def run_enhanced_training():
    """运行增强版训练"""
    
    print("🚀 启动增强版多模态融合训练")
    print("=" * 60)
    print("主要优化:")
    print("  ✓ 跨模态注意力机制 - 增强特征表示")
    print("  ✓ 原型网络 - 专门处理小样本类别")
    print("  ✓ 焦点损失 - 解决类别不平衡问题")
    print("  ✓ 类别感知增强 - 为每个类别定制特征")
    print("  ✓ 动态权重调整 - 智能融合文本和视频")
    print("=" * 60)
    
    # 增强版训练参数
    cmd = [
        "python", "train_video_text_fusion_origin.py",
        
        # 基础参数
        "--batch_sz", "8",
        "--lr", "3e-5",  # 降低学习率以提高稳定性
        "--max_epochs", "200",  # 增加训练轮数
        "--patience", "15",  # 增加早停耐心值
        "--name", "enhanced_for_tech_ops",
        
        # 视频参数优化
        "--num_frames", "16",  # 增加帧数以捕获更多技术操作细节
        "--video_hidden_sz", "768",  # 增加视频特征维度
        "--video_lstm_layers", "3",  # 增加LSTM层数
        "--video_bidirectional", "True",  # 使用双向LSTM
        "--video_feature_dim", "1536",  # 增加视频特征维度
        "--freeze_video_cnn", "False",  # 解冻CNN以学习更好的特征
        "--video_augmentation", "True",  # 启用数据增强
        
        # 融合参数优化
        "--fusion_loss_weight", "1.0",
        "--text_loss_weight", "0.4",
        "--video_loss_weight", "0.6",  # 增加视频权重，技术操作更依赖视觉
        "--prototype_loss_weight", "0.5",  # 增加原型损失权重
        "--confidence_reg_weight", "0.01",
        
        # 类别不平衡处理
        "--use_focal_loss", "True",
        "--focal_alpha", "0.25",
        "--focal_gamma", "3.0",  # 增加gamma值，更关注困难样本
        "--use_class_weights", "True",
        
        # 内存优化
        "--gradient_accumulation_steps", "2",
        "--mixed_precision", "True",
        
        # 动态融合
        "--use_dynamic_fusion", "True",
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        # 运行训练
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n🎉 训练完成！")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败: {e}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")
        return 1


def run_analysis_training():
    """运行分析优化版训练"""
    
    print("🔍 启动分析优化版训练")
    print("=" * 60)
    print("专门针对技术操作类别的优化策略:")
    print("  ✓ 更高的视频权重 (0.7)")
    print("  ✓ 更强的原型网络 (0.8)")
    print("  ✓ 更激进的焦点损失 (gamma=4.0)")
    print("  ✓ 更多的视频帧 (20帧)")
    print("  ✓ 更深的网络结构")
    print("=" * 60)
    
    cmd = [
        "python", "train_video_text_fusion_origin.py",
        
        # 基础参数
        "--batch_sz", "2",  # 减小batch size以支持更大模型
        "--lr", "2e-5",  # 更低的学习率
        "--max_epochs", "300",  # 更多训练轮数
        "--patience", "20",
        "--name", "analysis_tech_ops_focus",
        
        # 视频参数 - 专门为技术操作优化
        "--num_frames", "20",  # 更多帧数捕获技术操作序列
        "--video_hidden_sz", "1024",  # 更大的隐藏层
        "--video_lstm_layers", "4",  # 更深的LSTM
        "--video_bidirectional", "True",
        "--video_feature_dim", "2048",  # 更大的特征维度
        "--freeze_video_cnn", "False",
        "--video_augmentation", "True",
        
        # 融合参数 - 偏向视频模态
        "--fusion_loss_weight", "1.0",
        "--text_loss_weight", "0.3",
        "--video_loss_weight", "0.7",  # 技术操作更依赖视觉信息
        "--prototype_loss_weight", "0.8",  # 强化原型网络
        "--confidence_reg_weight", "0.005",
        
        # 激进的类别不平衡处理
        "--use_focal_loss", "True",
        "--focal_alpha", "0.1",  # 更小的alpha，更关注少数类
        "--focal_gamma", "4.0",  # 更大的gamma，更关注困难样本
        "--use_class_weights", "True",
        
        # 内存优化
        "--gradient_accumulation_steps", "4",  # 增加梯度累积
        "--mixed_precision", "True",
        
        # 动态融合
        "--use_dynamic_fusion", "True",
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n🎉 分析优化训练完成！")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败: {e}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")
        return 1


def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == "analysis":
        return run_analysis_training()
    else:
        return run_enhanced_training()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
