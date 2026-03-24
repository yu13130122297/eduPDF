#!/usr/bin/env python3
"""
将taolun.json合并到最终输出文件
"""

import json
import os

# 文件路径
INPUT_FILE = "/Users/yuzhichuan/Desktop/eduPDF/new_single.jsonl"
TEMP_FILE = "/Users/yuzhichuan/Desktop/eduPDF/taolun.json"
OUTPUT_FILE = "/Users/yuzhichuan/Desktop/eduPDF/new_single_with_transcription.jsonl"

def merge_to_final():
    """合并临时文件到最终文件"""

    # 检查文件是否存在
    if not os.path.exists(TEMP_FILE):
        print(f"错误：临时文件不存在: {TEMP_FILE}")
        print("请先运行 generate_chaotic_transcriptions.py 生成数据")
        return

    if not os.path.exists(INPUT_FILE):
        print(f"错误：输入文件不存在: {INPUT_FILE}")
        return

    print("=" * 70)
    print("合并工具：将taolun.json合并到最终输出文件")
    print("=" * 70)

    # 读取输入文件
    print(f"\n正在读取输入文件: {INPUT_FILE}")
    all_entries = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                all_entries.append(entry)
            except:
                continue

    print(f"共读取 {len(all_entries)} 条记录")

    # 读取临时文件
    print(f"正在读取临时文件: {TEMP_FILE}")
    with open(TEMP_FILE, 'r', encoding='utf-8') as f:
        taolun_data = json.load(f)

    print(f"共读取 {len(taolun_data)} 条学生讨论")

    # 创建ID到text的映射
    id_to_text = {item['id']: item['text'] for item in taolun_data}

    # 更新all_entries中的text
    updated_count = 0
    missing_count = 0
    for entry in all_entries:
        if entry.get('label') == '学生讨论':
            entry_id = entry.get('id', '')
            if entry_id in id_to_text:
                entry['text'] = id_to_text[entry_id]
                updated_count += 1
            else:
                missing_count += 1

    print(f"\n更新统计：")
    print(f"  成功更新: {updated_count} 条")
    print(f"  未找到: {missing_count} 条")

    # 写入最终文件
    print(f"\n正在写入最终文件: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\n合并完成！")
    print(f"  输出文件: {OUTPUT_FILE}")
    print(f"  总记录数: {len(all_entries)}")

    # 显示统计信息
    discussion_count = sum(1 for e in all_entries if e.get('label') == '学生讨论')
    print(f"\n文件统计：")
    print(f"  总记录数: {len(all_entries)}")
    print(f"  学生讨论: {discussion_count}")
    print(f"  其他类型: {len(all_entries) - discussion_count}")

if __name__ == "__main__":
    merge_to_final()
