"""
三模态文本编码器（FR-1）
将 text 和 video_description 智能融合后送入 BERT：
  - text 为空：仅使用 video_description
  - text 非空：双段拼接 [CLS] text [SEP] video_desc [SEP]，segment_ids 正确标记
"""

import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer


class TriModalTextEncoder(nn.Module):
    """三模态文本编码器，支持 text 和 video_description 的智能融合"""

    def __init__(self, args):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.bert_model)
        self.max_seq_length = getattr(args, "max_seq_length", 128)
        self.text_ratio = getattr(args, "text_ratio", 0.6)
        self.use_video_desc = getattr(args, "use_video_desc", True)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model,
                                                       do_lower_case=True)

    def forward(self, text_list, video_desc_list=None, return_segments=False):
        """
        Args:
            text_list: List[str]，原始文本（可能为空字符串）
            video_desc_list: List[str]，视频描述（99.9% 非空），当use_video_desc=False时可为None
            return_segments: 是否返回 segment_ids（可解释性用）

        Returns:
            pooled_output: [batch_size, 768]
            segment_ids: [batch_size, max_seq_length]（可选）
        """
        batch_size = len(text_list)
        input_ids_list, attention_mask_list, segment_ids_list = [], [], []

        for i in range(batch_size):
            txt = text_list[i].strip() if text_list[i] else ""

            # 根据 use_video_desc 决定是否使用 video_description
            if self.use_video_desc and video_desc_list is not None:
                desc = video_desc_list[i].strip() if video_desc_list[i] else ""
            else:
                desc = ""

            # 两者都为空时的兜底
            if not txt and not desc:
                desc = "[EMPTY]"

            if not txt:
                # 仅使用 video_description（segment 全为 0）
                tokens = self.tokenizer.tokenize(desc)
                tokens = ["[CLS]"] + tokens[: self.max_seq_length - 2] + ["[SEP]"]
                segment_ids = [0] * len(tokens)
            else:
                if self.use_video_desc and desc:
                    # 双段拼接
                    max_text_len = int((self.max_seq_length - 3) * self.text_ratio)
                    max_desc_len = self.max_seq_length - 3 - max_text_len

                    text_tokens = self.tokenizer.tokenize(txt)[:max_text_len]
                    desc_tokens = self.tokenizer.tokenize(desc)[:max_desc_len]

                    tokens = (
                        ["[CLS]"] + text_tokens + ["[SEP]"] + desc_tokens + ["[SEP]"]
                    )
                    segment_ids = (
                        [0] * (len(text_tokens) + 2) + [1] * (len(desc_tokens) + 1)
                    )
                else:
                    # 仅使用 text（不使用 video_description）
                    tokens = self.tokenizer.tokenize(txt)
                    tokens = ["[CLS]"] + tokens[: self.max_seq_length - 2] + ["[SEP]"]
                    segment_ids = [0] * len(tokens)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)

            # Padding
            pad_len = self.max_seq_length - len(input_ids)
            input_ids += [0] * pad_len
            attention_mask += [0] * pad_len
            segment_ids += [0] * pad_len

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            segment_ids_list.append(segment_ids)

        device = next(self.bert.parameters()).device
        input_ids_t = torch.tensor(input_ids_list, dtype=torch.long, device=device)
        attention_mask_t = torch.tensor(attention_mask_list, dtype=torch.long, device=device)
        segment_ids_t = torch.tensor(segment_ids_list, dtype=torch.long, device=device)

        _, pooled_output = self.bert(
            input_ids_t,
            token_type_ids=segment_ids_t,
            attention_mask=attention_mask_t,
            output_all_encoded_layers=False,
        )

        if return_segments:
            return pooled_output, segment_ids_t
        return pooled_output

    def build_modality_mask(self, text_list, video_desc_list=None):
        """返回 [batch_size, 2]，标记各样本是否有 text / video_description"""
        has_text = [1 if (t and t.strip()) else 0 for t in text_list]
        if self.use_video_desc and video_desc_list is not None:
            has_desc = [1 if (d and d.strip()) else 0 for d in video_desc_list]
        else:
            has_desc = [0] * len(text_list)
        device = next(self.bert.parameters()).device
        return torch.tensor(
            list(zip(has_text, has_desc)), dtype=torch.float, device=device
        )
