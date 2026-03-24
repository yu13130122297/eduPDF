#!/usr/bin/env python3
"""
使用DeepSeek API + few-shot生成混乱的学生讨论转录
"""

import json
import os
import requests
import time
from collections import defaultdict

# DeepSeek API配置
API_KEY = "sk-11f388929ba64064a99422251adcb509"
API_URL = "https://api.deepseek.com/v1/chat/completions"

# 输入输出文件
INPUT_FILE = "/Users/yuzhichuan/Desktop/eduPDF/new_single.jsonl"
TEMP_FILE = "/Users/yuzhichuan/Desktop/eduPDF/taolun.json"  # 临时文件，可编辑
OUTPUT_FILE = "/Users/yuzhichuan/Desktop/eduPDF/new_single_with_transcription.jsonl"  # 最终文件

# 数学课few-shot示例（模拟真实多人混乱讨论转录）
MATH_EXAMPLES = """
示例1：斜率k等于 呃Δy除Δx 嗯y等于kx加b 截距是b 垂直k1k2负一
示例2：圆方程x平方加y平方 嗯r是半径 圆心坐标 切线距离等于r 呃相交d小于
示例3：公式 tanα 斜率 截距 点斜式 y等于kx加b 嗯两点式 两点式
示例4：Ax加By加C 呃移项 斜率负A除B 截距C除以B 要除A
示例5：k1等于k2平行 垂直 负一 垂足 向量法 投影 面积除底
示例6：距离根号A方加B方 分子绝对值 投影叉积 分母是模长
示例7：切线d等于r 呃相交d小于r 判别式 大于零 圆心距离 坐标代入
"""

# 语文课few-shot示例（模拟真实多人混乱讨论转录）
CHINESE_EXAMPLES = """
示例1：朱自清背影 父亲买橘子 嗯动作描写 语言描写 还有外貌
示例2：比喻拟人 修辞手法 动静结合 移步换景 散文线索 呃时间顺序
示例3：荷塘月色 作者情感 孤独 忧愁 嗯淡淡的喜悦 意境美
示例4：散文结构 总分总 开头点题 呃中间展开 结尾升华 议论抒情
示例5：鲁迅故乡 闰土 少年中年对比 嗯杨二嫂 反映社会现实
示例6：古诗词意象 借景抒情 直抒胸臆 炼字 推敲 托物言志
示例7：记叙文六要素 时间地点人物 呃起因经过结果 插叙倒叙
"""

def extract_lesson_id(video_path: str) -> str:
    """从视频路径提取课程ID"""
    try:
        parts = video_path.split('/')
        for part in parts:
            if part.startswith('T') and part[1:].isdigit():
                return part
    except:
        pass
    return None

def get_lesson_type(all_texts: str) -> str:
    """判断是数学课还是语文课"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    prompt = f"""根据以下课堂转录文本，判断是数学课还是语文课，只回答"数学"或"语文"。

转录文本:
{all_texts[:500]}

请直接回答："""

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 10
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return "数学"

def get_lesson_topic(all_texts: str) -> str:
    """获取课程主题摘要"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    prompt = f"""根据以下课堂的所有转录文本，简要总结这门课的主题内容和教学要点（20字以内）。

转录文本:
{all_texts}

请直接给出主题摘要。"""

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 50
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return "课堂讨论"

def generate_chaotic_transcription(lesson_type: str, lesson_topic: str) -> str:
    """使用few-shot生成混乱转录"""
    import random

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    if lesson_type == "语文":
        examples = CHINESE_EXAMPLES
        subject = "语文课"
    else:
        examples = MATH_EXAMPLES
        subject = "数学课"

    # 随机选择混乱风格提示
    style_prompts = [
        "模拟学生小声窃窃私语，打断别人的话",
        "模拟学生抢答、互相纠正、思维跳跃",
        "模拟学生自言自语、突然想起、自我否定",
        "模拟多个学生同时发言、声音交叠、观点冲突",
        "模拟学生边思考边说、思路不清、反复修改"
    ]
    style = random.choice(style_prompts)

    # 随机选择混乱模式
    confusion_patterns = [
        "说一半被打断，转而说另一个想法",
        "连续修正自己的说法，来回调整",
        "突然想起另一个相关知识点，插话说",
        "重复某个关键词或公式，强化记忆",
        "说到一半卡住，然后换种表达方式",
        "把两个相似概念混在一起说",
        "自我纠正，重新组织语言表达"
    ]
    pattern = random.choice(confusion_patterns)

    prompt = f"""{examples}

任务：生成1条混乱的学生课堂讨论转录（{subject}，主题：{lesson_topic}）

生成要求：
- 20-25个汉字
- 只有一句话，陈述句（以句号结尾）
- {style}
- {pattern}
- 语气：犹豫、疑惑、试探、自言自语
- 不要出现"老师"二字
- 不要出现"不对"二字
- 不要使用问号或疑问句
- 不要使用省略号（...、……）
- 内容要自然真实，避免机械重复

注意：参考上面示例的混乱风格，但要生成全新的、不同的内容。

直接输出20-25字转录："""

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.5,  # 提高随机性
        "top_p": 0.9,  # 增加多样性
        "max_tokens": 40
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        text = result["choices"][0]["message"]["content"].strip()

        # 移除可能的引号、破折号等符号
        text = text.strip('"\'-—～')

        # 移除省略号
        text = text.replace('...', '').replace('……', '').replace('…', '')

        # 禁止出现的词
        forbidden_words = ['不对']
        for word in forbidden_words:
            if word in text:
                # 替换为等价的混乱表达
                replacements = {
                    '不对': '那个',
                }
                for old, new in replacements.items():
                    text = text.replace(old, new)

        # 确保以句号结尾
        if text and not text.endswith(('。', '！', '？', '.', '!', '?')):
            text = text + '。'

        # 截取前25个字符（避免超长）
        if len(text) > 25:
            text = text[:25]
            # 确保最后是完整句子
            if not text.endswith(('。', '！', '？', '.', '!', '?')):
                text = text.rsplit('，', 1)[0] + '。'

        return text
    except Exception as e:
        print(f"API调用失败: {e}")
        return "嗯等一下这个是"

def update_taolun_file(all_entries, taolun_data):
    """更新taolun.json文件（每次处理完一节课后调用）"""
    taolun_entries = []
    for entry in all_entries:
        if entry.get('label') == '学生讨论' and entry.get('text', '').strip():
            taolun_entries.append({
                'id': entry.get('id', ''),
                'text': entry.get('text', ''),
                'video': entry.get('video', ''),
                'label': entry.get('label', '')
            })

    # 写入文件
    with open(TEMP_FILE, 'w', encoding='utf-8') as f:
        json.dump(taolun_entries, f, ensure_ascii=False, indent=2)

    return len(taolun_entries)


def merge_to_final(all_entries):
    """合并临时文件到最终文件"""
    print(f"\n正在合并到最终文件: {OUTPUT_FILE}")

    # 读取临时文件
    with open(TEMP_FILE, 'r', encoding='utf-8') as f:
        taolun_data = json.load(f)

    # 创建ID到text的映射
    id_to_text = {item['id']: item['text'] for item in taolun_data}

    # 更新all_entries中的text
    updated_count = 0
    for entry in all_entries:
        if entry.get('label') == '学生讨论':
            entry_id = entry.get('id', '')
            if entry_id in id_to_text:
                entry['text'] = id_to_text[entry_id]
                updated_count += 1

    # 写入最终文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"合并完成!")
    print(f"  更新了 {updated_count} 条学生讨论")
    print(f"  输出文件: {OUTPUT_FILE}")


def main():
    # 读取已生成的临时文件（如果存在）
    processed_ids = set()
    taolun_data = []

    if os.path.exists(TEMP_FILE):
        print(f"发现已存在的临时文件: {TEMP_FILE}")
        print(f"读取已处理的条目...")
        with open(TEMP_FILE, 'r', encoding='utf-8') as f:
            taolun_data = json.load(f)
            for item in taolun_data:
                processed_ids.add(item.get('id', ''))

        print(f"已处理 {len(processed_ids)} 个条目（将跳过）")

    # 读取所有条目并按课程分组
    all_entries = []
    lesson_texts = defaultdict(list)
    lesson_discussion_entries = defaultdict(list)

    print(f"正在读取文件: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                entry = json.loads(line.strip())
                all_entries.append(entry)

                video_path = entry.get('video', '')
                lesson_id = extract_lesson_id(video_path)

                if lesson_id:
                    text = entry.get('text', '')
                    if text and text.strip():
                        lesson_texts[lesson_id].append(text)

                    if entry.get('label') == '学生讨论':
                        lesson_discussion_entries[lesson_id].append({
                            'index': idx,
                            'entry': entry
                        })
            except json.JSONDecodeError:
                continue

    print(f"共读取 {len(all_entries)} 条记录")
    print(f"共 {len(lesson_texts)} 门课程: {list(lesson_texts.keys())}")

    total_discussions = sum(len(v) for v in lesson_discussion_entries.values())
    print(f"共 {total_discussions} 条学生讨论")

    # 处理每门课程
    for lesson_id in sorted(lesson_texts.keys()):
        discussion_count = len(lesson_discussion_entries[lesson_id])
        texts = lesson_texts[lesson_id]
        all_text = " ".join(texts)

        print(f"\n{'='*50}")
        print(f"课程 {lesson_id}: {discussion_count} 条学生讨论")

        if not all_text.strip():
            lesson_type = "数学"
            lesson_topic = "课堂讨论"
        else:
            print(f"[{lesson_id}] 正在判断课程类型...")
            lesson_type = get_lesson_type(all_text)
            print(f"[{lesson_id}] 课程类型: {lesson_type}")

            print(f"[{lesson_id}] 正在分析课程主题...")
            lesson_topic = get_lesson_topic(all_text)
            print(f"[{lesson_id}] 主题: {lesson_topic}")

        # 为每个学生讨论条目生成转录
        for i, disc_entry in enumerate(lesson_discussion_entries[lesson_id]):
            idx = disc_entry['index']
            entry = disc_entry['entry']
            entry_id = entry.get('id', '')

            # 检查是否已处理
            if entry_id in processed_ids:
                print(f"  [{i+1}/{discussion_count}] {entry_id} - 已跳过（已处理）")
                # 使用已生成的转录
                for taolun_item in taolun_data:
                    if taolun_item.get('id') == entry_id:
                        entry['text'] = taolun_item.get('text', '')
                        break
                continue

            print(f"  [{i+1}/{discussion_count}] {entry_id}")

            transcription = generate_chaotic_transcription(lesson_type, lesson_topic)
            entry['text'] = transcription
            print(f"    -> {transcription}")

            time.sleep(0.5)

        # 每处理完一节课，立即写入taolun.json
        print(f"\n[{lesson_id}] 正在保存到临时文件...")
        count = update_taolun_file(all_entries, taolun_data)
        print(f"[{lesson_id}] 保存完成！共 {count} 条学生讨论")

    # 全部完成后统计
    print(f"\n{'='*50}")
    print(f"所有课程处理完成！")

    # 统计总数
    total_taolun = sum(1 for e in all_entries if e.get('label') == '学生讨论')
    print(f"  共 {len(lesson_texts)} 门课程")
    print(f"  共 {total_taolun} 条学生讨论")

    # 询问是否合并到最终文件
    print(f"\n{'='*50}")
    choice = input("是否现在合并到最终文件？(y/n，默认n): ").strip().lower()

    if choice == 'y':
        merge_to_final(all_entries)
    else:
        print(f"\n稍后可以使用以下命令合并：")
        print(f"  python3 -c \"import json; exec(open('generate_chaotic_transcriptions.py').read().split('def ')[1])\"")

if __name__ == "__main__":
    main()
