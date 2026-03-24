"""
PRM 监督信号生成（FR-6）
使用 GPT-4o-mini 为样本生成五步监督信号，并缓存到本地。

用法：
    supervisor = PRMSupervisor(api_key="...")
    supervision = supervisor.generate_supervision(sample, model_outputs)
    # → {"text_quality": 0.88, "video_quality": 0.91, ...}

    supervisor.generate_dataset_supervision(
        dataset_path="data/train.jsonl",
        output_path="data/prm_supervision.json",
        n_samples=500
    )
"""

import json
import logging
import os
import re
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PRMSupervisor:
    """使用 LLM 生成 PRM 五步监督信号"""

    DEFAULT_SCORES = {
        "text_quality": 0.5,
        "video_quality": 0.5,
        "alignment": 0.5,
        "routing": 0.5,
        "fusion": 0.5,
    }

    def __init__(self,
                 api_key: str,
                 model: str = "gpt-4o-mini",
                 base_url: Optional[str] = None,
                 temperature: float = 0.3,
                 max_retries: int = 3):
        try:
            from openai import OpenAI
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self.client = OpenAI(**kwargs)
        except ImportError:
            raise ImportError("请安装 openai>=1.0.0: pip install openai")

        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    def generate_supervision(self,
                              sample: Dict,
                              model_outputs: Optional[Dict] = None) -> Dict:
        """
        为单个样本生成五步监督信号。

        Args:
            sample: {"text": str, "video_description": str, "label": str}
            model_outputs: {"prediction": str, "confidence": float}（可选）

        Returns:
            {"text_quality": float, "video_quality": float,
             "alignment": float, "routing": float, "fusion": float}
        """
        prompt = self._build_prompt(sample, model_outputs)
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system",
                         "content": "你是一个教育场景分析专家，负责评估多模态模型的预测质量。"},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=500,
                )
                return self._parse_response(resp.choices[0].message.content)
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"LLM 调用失败（第 {attempt+1} 次），{wait}s 后重试: {e}")
                time.sleep(wait)

        logger.error("LLM 监督信号生成失败，返回默认值")
        return dict(self.DEFAULT_SCORES)

    # ------------------------------------------------------------------
    def generate_dataset_supervision(self,
                                     dataset_path: str,
                                     output_path: str,
                                     n_samples: int = 500) -> None:
        """
        为数据集中前 n_samples 个样本批量生成监督信号并缓存。
        已存在的缓存文件将被跳过（断点续生）。
        """
        # 加载已有缓存
        cache: Dict[str, Dict] = {}
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            logger.info(f"加载缓存：{len(cache)} 条")

        # 读取数据集
        samples: List[Dict] = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                samples.append(item)
                if len(samples) >= n_samples:
                    break

        new_count = 0
        for i, sample in enumerate(samples):
            sid = sample.get("id", str(i))
            if sid in cache:
                continue

            supervision = self.generate_supervision(sample)
            cache[sid] = supervision
            new_count += 1

            if new_count % 20 == 0:
                self._save_cache(cache, output_path)
                logger.info(f"已处理 {i+1}/{len(samples)}，新增 {new_count} 条")

        self._save_cache(cache, output_path)
        logger.info(f"完成！共 {len(cache)} 条监督信号，保存至 {output_path}")

    # ------------------------------------------------------------------
    def _build_prompt(self, sample: Dict, model_outputs: Optional[Dict]) -> str:
        text = sample.get("text", "")
        video_desc = sample.get("video_description", "")
        label = sample.get("label", "未知")

        pred_info = ""
        if model_outputs:
            pred = model_outputs.get("prediction", "未知")
            conf = model_outputs.get("confidence", 0.0)
            pred_info = f"\n**模型输出**\n- 预测标签: {pred}\n- 预测置信度: {conf:.2f}"

        return f"""请评估以下课堂行为识别样本的质量：

**输入信息**
- 文本: {text}
- 视频描述: {video_desc}
- 真实标签: {label}
{pred_info}

请从以下五个维度评分（0-1分）：

1. **文本质量**: 文本描述是否清晰、完整、信息丰富？
   - 0分：空文本或无意义；0.5分：有信息但不完整；1分：清晰完整

2. **视频质量**: 视频描述是否准确、详细？
   - 0分：描述模糊；0.5分：基本准确；1分：准确详细

3. **对齐度**: 文本和视频描述是否语义一致？
   - 0分：完全不一致；0.5分：部分一致；1分：高度一致

4. **路由决策**: 模型应该更依赖文本(→1)还是视频(→0)，还是平衡(→0.5)？

5. **融合效果**: 模型预测是否合理？（无模型输出时评估数据质量）
   - 0分：预测错误；0.5分：部分正确；1分：完全正确

请以 JSON 格式输出：
{{
    "text_quality": 0.0-1.0,
    "video_quality": 0.0-1.0,
    "alignment": 0.0-1.0,
    "routing": 0.0-1.0,
    "fusion": 0.0-1.0,
    "reasoning": "简要说明"
}}"""

    def _parse_response(self, text: str) -> Dict:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return {
                    k: float(data.get(k, 0.5))
                    for k in self.DEFAULT_SCORES
                }
            except Exception:
                pass
        logger.warning("无法解析 LLM 响应，使用默认值")
        return dict(self.DEFAULT_SCORES)

    @staticmethod
    def _save_cache(cache: Dict, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
