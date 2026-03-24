#!/usr/bin/env python3
"""
快速测试混乱转录生成（只生成10条，用于快速验证效果）
"""

import sys
sys.path.append('.')

from generate_chaotic_transcriptions import generate_chaotic_transcription

def quick_test():
    """快速测试生成效果"""

    print("=" * 70)
    print("快速测试：混乱转录生成")
    print("=" * 70)

    test_cases = [
        ("数学", "直线与圆的方程"),
        ("数学", "直线斜率计算"),
        ("数学", "点到直线距离公式"),
        ("数学", "圆的相切相交相离"),
        ("数学", "直线平行与垂直"),
        ("语文", "朱自清背影"),
        ("语文", "散文结构分析"),
        ("语文", "修辞手法辨析"),
        ("语文", "古诗词鉴赏"),
        ("语文", "作者情感把握"),
    ]

    print("\n开始生成测试...\n")

    for i, (lesson_type, lesson_topic) in enumerate(test_cases, 1):
        print(f"[{i}/10] {lesson_type}课 - {lesson_topic}")

        try:
            transcription = generate_chaotic_transcription(lesson_type, lesson_topic)
            print(f"    -> {transcription}")
        except Exception as e:
            print(f"    ✗ 生成失败: {e}")

        print()

    print("=" * 70)
    print("测试完成！")
    print("\n检查要点：")
    print("  1. 每条转录是否独特（不重复）")
    print("  2. 长度是否在20-25字之间")
    print("  3. 是否有真实学生的混乱感")
    print("  4. 模式是否多样化（不只是'不对'开头）")
    print("  5. 内容是否与主题相关")
    print("=" * 70)

if __name__ == "__main__":
    quick_test()
