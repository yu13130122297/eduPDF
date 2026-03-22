#!/bin/bash

# 设置遇到错误时终止脚本
set -e

echo "=========================================="
echo "          开始下载依赖模型和数据集"
echo "=========================================="

# 1. 下载 bert-base-chinese
echo ""
echo ">>> [1/2] 正在准备 bert-base-chinese 模型..."
if [ ! -d "bert-base-chinese" ]; then
    echo "正在从 HuggingFace 镜像源下载..."
    # 尝试安装 Git LFS
    git lfs install 2>/dev/null || true
    
    # 推荐使用国内镜像源以保证下载速度和稳定性
    GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/bert-base-chinese
    
    cd bert-base-chinese
    # 拉取大文件
    git lfs pull
    cd ..
    echo "bert-base-chinese 下载完成！"
else
    echo "目录 bert-base-chinese 已存在，跳过下载。"
fi

# 2. 下载 Glove 词向量
echo ""
echo ">>> [2/2] 正在准备 glove.840B.300d.txt 词向量..."
GLOVE_DIR="datasets/glove_embeds"
mkdir -p "$GLOVE_DIR"

if [ ! -f "$GLOVE_DIR/glove.840B.300d.txt" ]; then
    echo "正在从 Stanford NLP 下载 GloVe (文件较大，约 2GB，需要一些时间)..."
    wget https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip -O "$GLOVE_DIR/glove.840B.300d.zip"
    
    echo "下载完成，正在解压..."
    unzip "$GLOVE_DIR/glove.840B.300d.zip" -d "$GLOVE_DIR"
    
    echo "清理压缩包..."
    rm "$GLOVE_DIR/glove.840B.300d.zip"
    echo "GloVe 词向量准备完成！"
else
    echo "文件 $GLOVE_DIR/glove.840B.300d.txt 已存在，跳过下载。"
fi

echo ""
echo "=========================================="
echo "             所有资源准备完毕!"
echo "=========================================="
