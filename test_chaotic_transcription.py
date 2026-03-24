#!/usr/bin/env python3
"""
测试混乱转录生成的多样性和真实性
"""

import json
import random

# 读取生成的文件
INPUT_FILE = "/Users/yuzhichuan/Desktop/eduPDF/new_single_with_transcription.jsonl"

def analyze_transcriptions():
    """分析转录的多样性和真实性"""

    print("正在读取文件...")
    transcriptions = []

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get('label') == '学生讨论':
                    text = entry.get('text', '')
                    if text and len(text) > 5:
                        transcriptions.append(text)
            except:
                continue

    print(f"共读取 {len(transcriptions)} 条学生讨论转录\n")

    if not transcriptions:
        print("没有找到学生讨论转录")
        return

    # 1. 长度分析
    print("=" * 60)
    print("1. 长度分析")
    print("=" * 60)
    lengths = [len(t) for t in transcriptions]
    print(f"平均长度: {sum(lengths)/len(lengths):.1f} 字")
    print(f"最短: {min(lengths)} 字")
    print(f"最长: {max(lengths)} 字")
    print(f"符合20-25字: {sum(1 for l in lengths if 20 <= l <= 25)} / {len(lengths)} ({100*sum(1 for l in lengths if 20 <= l <= 25)/len(lengths):.1f}%)")

    # 2. 重复率分析
    print("\n" + "=" * 60)
    print("2. 重复率分析")
    print("=" * 60)
    unique_transcriptions = set(transcriptions)
    print(f"唯一转录: {len(unique_transcriptions)} / {len(transcriptions)} ({100*len(unique_transcriptions)/len(transcriptions):.1f}%)")

    # 找出重复最多的
    from collections import Counter
    counts = Counter(transcriptions)
    duplicates = {k: v for k, v in counts.items() if v > 1}
    if duplicates:
        print(f"\n重复的转录（前10个）：")
        for text, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  [{count}次] {text}")
    else:
        print("\n✓ 没有重复的转录")

    # 3. 模式分析
    print("\n" + "=" * 60)
    print("3. 模式分析")
    print("=" * 60)

    # 统计常见模式
    patterns = {
        "开头有'不对'": sum(1 for t in transcriptions if t.startswith('不对')),
        "包含'不对'": sum(1 for t in transcriptions if '不对' in t),
        "开头有'那个'": sum(1 for t in transcriptions if t.startswith('那个')),
        "包含'等一下'": sum(1 for t in transcriptions if '等一下' in t),
        "包含'那个'": sum(1 for t in transcriptions if '那个' in t),
        "以句号结尾": sum(1 for t in transcriptions if t.endswith('。')),
    }

    for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
        percent = 100 * count / len(transcriptions)
        print(f"{pattern}: {count} / {len(transcriptions)} ({percent:.1f}%)")

    # 4. 随机抽样展示
    print("\n" + "=" * 60)
    print("4. 随机抽样展示（20条）")
    print("=" * 60)
    sample_size = min(20, len(transcriptions))
    samples = random.sample(transcriptions, sample_size)

    for i, text in enumerate(samples, 1):
        print(f"[{i}] {text}")

    # 5. 问题诊断
    print("\n" + "=" * 60)
    print("5. 问题诊断")
    print("=" * 60)

    issues = []

    # 重复率过高
    duplication_rate = 100 * (len(transcriptions) - len(unique_transcriptions)) / len(transcriptions)
    if duplication_rate > 50:
        issues.append(f"❌ 重复率过高: {duplication_rate:.1f}% (>50%)")
    elif duplication_rate > 20:
        issues.append(f"⚠️  重复率偏高: {duplication_rate:.1f}% (>20%)")
    else:
        issues.append(f"✓ 重复率正常: {duplication_rate:.1f}%")

    # "不对"开头过多
    wrong_start_rate = 100 * sum(1 for t in transcriptions if t.startswith('不对')) / len(transcriptions)
    if wrong_start_rate > 40:
        issues.append(f"❌ '不对'开头过多: {wrong_start_rate:.1f}% (>40%)")
    elif wrong_start_rate > 20:
        issues.append(f"⚠️  '不对'开头偏多: {wrong_start_rate:.1f}% (>20%)")
    else:
        issues.append(f"✓ '不对'开头正常: {wrong_start_rate:.1f}%")

    # "那个"过多
    that_rate = 100 * sum(1 for t in transcriptions if '那个' in t) / len(transcriptions)
    if that_rate > 60:
        issues.append(f"❌ '那个'出现过多: {that_rate:.1f}% (>60%)")
    elif that_rate > 40:
        issues.append(f"⚠️  '那个'出现偏多: {that_rate:.1f}% (>40%)")
    else:
        issues.append(f"✓ '那个'出现正常: {that_rate:.1f}%")

    for issue in issues:
        print(issue)

    print("\n" + "=" * 60)

if __name__ == "__main__":
    analyze_transcriptions()
