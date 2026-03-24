"""
LLM 数据增强模块（FR-2）
使用 DeepSeek / GPT-4o-mini 为小样本类别生成训练数据。

用法：
    augmentor = LLMAugmentor(api_key="...", model="deepseek-chat")
    samples = augmentor.generate_samples(
        class_name="技术操作",
        seed_samples=[{"text": "...", "video_description": "..."}, ...],
        target_count=162
    )
"""

import json
import logging
import time
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class LLMAugmentor:
    """使用 LLM 生成指定类别训练样本的数据增强器"""

    def __init__(self,
                 api_key: str,
                 model: str = "deepseek-chat",
                 base_url: str = "https://api.deepseek.com",
                 temperature: float = 0.8,
                 min_similarity: float = 0.7,
                 max_similarity: float = 0.95):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            raise ImportError("请安装 openai>=1.0.0: pip install openai")

        self.model = model
        self.temperature = temperature
        self.min_similarity = min_similarity
        self.max_similarity = max_similarity
        self._bert_model = None  # 懒加载

    # ------------------------------------------------------------------
    def generate_samples(self,
                         class_name: str,
                         seed_samples: List[Dict],
                         target_count: int,
                         max_attempts_ratio: int = 3) -> List[Dict]:
        """
        为指定类别生成训练样本。

        Returns:
            生成的样本列表，每项包含 text / video_description / label / source
        """
        generated: List[Dict] = []
        max_attempts = target_count * max_attempts_ratio
        attempts = 0
        prompt_template = self._build_prompt(class_name, seed_samples)

        while len(generated) < target_count and attempts < max_attempts:
            attempts += 1
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个教育场景数据生成专家。"},
                        {"role": "user", "content": prompt_template},
                    ],
                    temperature=self.temperature,
                    max_tokens=200,
                )
                text = response.choices[0].message.content.strip()

                if self._quality_check(text, seed_samples):
                    video_desc = self._generate_video_desc(text)
                    generated.append({
                        "text": text,
                        "video_description": video_desc,
                        "label": class_name,
                        "source": "llm_augmented",
                    })
                    if len(generated) % 10 == 0:
                        logger.info(f"[{class_name}] 已生成 {len(generated)}/{target_count}")

            except Exception as e:
                logger.warning(f"第 {attempts} 次生成失败: {e}")
                time.sleep(1)

        logger.info(
            f"[{class_name}] 完成：生成 {len(generated)} 条，尝试 {attempts} 次，"
            f"通过率 {len(generated)/max(attempts,1)*100:.1f}%"
        )
        return generated

    # ------------------------------------------------------------------
    def augment_dataset(self,
                        dataset_path: str,
                        output_path: str,
                        target_counts: Dict[str, int]) -> None:
        """
        批量增强整个数据集。

        Args:
            dataset_path: 原始 JSONL 路径
            output_path:  输出 JSONL 路径
            target_counts: {"技术操作": 200, "教师反馈": 400, ...}
        """
        # 加载原始数据并按类别分组
        class_samples: Dict[str, List[Dict]] = {}
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                label = item.get("label", "")
                class_samples.setdefault(label, []).append(item)

        all_generated: List[Dict] = []
        for class_name, target in target_counts.items():
            seeds = class_samples.get(class_name, [])
            current = len(seeds)
            if current >= target:
                logger.info(f"[{class_name}] 已有 {current} 条，跳过增强")
                continue
            need = target - current
            logger.info(f"[{class_name}] 当前 {current} 条，需生成 {need} 条")
            new_samples = self.generate_samples(class_name, seeds[:5], need)
            all_generated.extend(new_samples)

        with open(output_path, "w", encoding="utf-8") as f:
            for item in all_generated:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"共生成 {len(all_generated)} 条，已保存到 {output_path}")

    # ------------------------------------------------------------------
    def _build_prompt(self, class_name: str, seed_samples: List[Dict]) -> str:
        examples = "\n\n".join([
            f"示例{i+1}:\n文本: {s['text']}\n视频描述: {s.get('video_description', '')}"
            for i, s in enumerate(seed_samples[:5])
        ])
        return f"""请生成一个"{class_name}"类别的课堂行为描述。

已有示例：
{examples}

要求：
1. 文本长度在 20-100 字之间
2. 描述要具体、真实，符合实际课堂场景
3. 包含该类别的典型特征
4. 与示例相似但不完全相同

请生成：
文本:"""

    def _quality_check(self, text: str, seed_samples: List[Dict]) -> bool:
        """基于长度和可选 BERT 相似度的质量过滤"""
        if len(text) < 20 or len(text) > 200:
            return False
        # 如果 sentence-transformers 可用，则做相似度过滤
        try:
            sim = self._compute_avg_similarity(text, seed_samples)
            return self.min_similarity <= sim <= self.max_similarity
        except Exception:
            return True  # 无法计算时放行

    def _compute_avg_similarity(self, text: str, seed_samples: List[Dict]) -> float:
        if self._bert_model is None:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim
            self._bert_model = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2"
            )
            self._cos_sim = cos_sim

        seeds = [s["text"] for s in seed_samples if s.get("text")]
        if not seeds:
            return 0.8  # 无种子时默认通过

        gen_emb = self._bert_model.encode([text])
        seed_emb = self._bert_model.encode(seeds)
        sims = self._cos_sim(gen_emb, seed_emb)[0]
        return float(sims.mean())

    def _generate_video_desc(self, text: str) -> str:
        """根据文本生成对应的视频描述"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "根据课堂文本描述，生成对应的视频画面描述（50字以内）。"},
                    {"role": "user", "content": f"文本: {text}\n\n视频描述:"},
                ],
                temperature=0.7,
                max_tokens=100,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return ""
