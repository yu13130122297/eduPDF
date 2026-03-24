#!/usr/bin/env python3
"""
数据增强入口脚本（FR-2）

用法（Ollama）:
    OLLAMA_BASE_URL=http://localhost:11434/v1 python run_augmentation.py

用法（DeepSeek）:
    DEEPSEEK_API_KEY=sk-xxx python run_augmentation.py
"""

import json
import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── 配置区 ────────────────────────────────────────────────────────────────────

# Ollama 本地 / DeepSeek 远端，二选一
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "")          # e.g. http://localhost:11434/v1
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

if OLLAMA_BASE_URL:
    API_KEY  = "ollama"
    BASE_URL = OLLAMA_BASE_URL
    MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
elif DEEPSEEK_API_KEY:
    API_KEY  = DEEPSEEK_API_KEY
    BASE_URL = "https://api.deepseek.com"
    MODEL    = "deepseek-chat"
else:
    raise EnvironmentError(
        "请设置环境变量：\n"
        "  Ollama:   export OLLAMA_BASE_URL=http://localhost:11434/v1\n"
        "  DeepSeek: export DEEPSEEK_API_KEY=sk-xxx"
    )

TRAIN_PATH  = "datasets/education/train.jsonl"
OUTPUT_PATH = "datasets/education/train_augmented.jsonl"

# 每个类别的目标总数（不足则生成，已足则跳过）
TARGET_COUNTS = {
    "技术操作": 200,
    "教师反馈": 400,
    "教师指令": 300,
    "学生讨论": 300,
}

# ── 主逻辑 ────────────────────────────────────────────────────────────────────

def merge_and_save(original_path: str, augmented_path: str, final_path: str) -> None:
    """将原始数据和增强数据合并输出"""
    lines = []
    with open(original_path, encoding="utf-8") as f:
        lines.extend(f.readlines())
    if Path(augmented_path).exists():
        with open(augmented_path, encoding="utf-8") as f:
            lines.extend(f.readlines())
    with open(final_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    # 统计
    counts: dict = {}
    for line in lines:
        label = json.loads(line).get("label", "?")
        counts[label] = counts.get(label, 0) + 1
    logger.info("合并后各类别数量：")
    for label, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {label}: {cnt}")


def main():
    logger.info(f"使用模型: {MODEL}  base_url: {BASE_URL}")

    from src.data.augmentation import LLMAugmentor

    augmentor = LLMAugmentor(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
        temperature=0.8,
    )

    augmented_tmp = "datasets/education/augmented_new.jsonl"
    augmentor.augment_dataset(
        dataset_path=TRAIN_PATH,
        output_path=augmented_tmp,
        target_counts=TARGET_COUNTS,
    )

    merge_and_save(TRAIN_PATH, augmented_tmp, OUTPUT_PATH)
    logger.info(f"最终数据已保存至 {OUTPUT_PATH}")
    logger.info("后续训练将此文件路径替换 train.jsonl 即可，无需再次运行本脚本。")


if __name__ == "__main__":
    main()
