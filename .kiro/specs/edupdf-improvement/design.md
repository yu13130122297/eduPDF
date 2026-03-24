# eduPDF 改进方案设计文档

## 1. 概述

### 1.1 设计目标

本设计文档描述了eduPDF多模态课堂行为识别系统的六大核心改进方案。设计保留PDF（Probabilistic Distribution Fusion）框架的核心思想——动态权重调整，同时引入现代深度学习技术以解决空文本处理、小样本学习、长序列建模和可解释性等关键问题。

### 1.2 核心改进

1. **三模态融合（FR-1）**: 将video_description作为第三模态，解决28.5%空文本样本问题
2. **LLM数据增强（FR-2）**: 使用大语言模型生成小样本类别训练数据
3. **Mamba序列编码（FR-3）**: 用状态空间模型替换LSTM，支持100+帧长视频
4. **Sparse MoE ConfidNet（FR-4）**: 用8专家稀疏混合专家模型替换20个独立置信度网络
5. **层次化MoE融合层（FR-5）**: 将硬编码全息权重升级为可学习的两层MoE路由
6. **PRM过程监督（FR-6）**: 引入过程奖励模型提供五步监督信号

### 1.3 设计原则

- **保留核心**: 保持PDF框架的全息权重（holographic weights）计算思想
- **模块化**: 每个改进独立可测试，支持渐进式集成
- **可扩展**: 支持新增模态、类别和专家
- **可解释**: 输出融合权重、路由决策和过程评分
- **高效**: 推理速度≤20ms，内存占用≤4GB


## 2. 架构

### 2.1 整体架构

```mermaid
graph TB
    subgraph "输入层"
        A[text] --> B[BERT Encoder]
        C[video_description] --> B
        D[video frames] --> E[ResNet50 + Mamba]
    end
    
    subgraph "特征提取层"
        B --> F[text_features]
        E --> G[video_features]
    end
    
    subgraph "Sparse MoE ConfidNet"
        F --> H[8-Expert MoE]
        G --> H
        H --> I[text_confidence]
        H --> J[video_confidence]
    end
    
    subgraph "层次化MoE融合层"
        I --> K[Layer1: 模态路由MoE]
        J --> K
        K --> L[Layer2: 类别路由MoE]
        L --> M[融合权重 w_text, w_video]
    end
    
    subgraph "PRM过程监督"
        F --> N[Process Reward Model]
        G --> N
        M --> N
        N --> O[5步评分]
    end
    
    M --> P[最终预测]
    O --> P
```

### 2.2 数据流

```
输入样本 {text, video_description, video_frames, label}
    ↓
[文本处理] 
    if text == "":
        bert_input = video_description
    else:
        bert_input = text + [SEP] + video_description
    ↓
[特征提取]
    text_features = BERT(bert_input)           # [batch, 768]
    video_features = Mamba(ResNet50(frames))   # [batch, 512]
    ↓
[置信度估计 - Sparse MoE]
    expert_outputs = [Expert_i(features) for i in range(8)]
    routing_weights = Router(features)         # Top-2 sparse
    text_conf = Σ routing_weights[i] * expert_outputs[i]
    video_conf = Σ routing_weights[i] * expert_outputs[i]
    ↓
[全息权重计算 - 保留PDF核心]
    text_holo = log(video_conf) / log(text_conf * video_conf)
    video_holo = log(text_conf) / log(text_conf * video_conf)
    ↓
[层次化MoE融合]
    Layer1: 模态级路由 → 3个专家（text-dominant, video-dominant, balanced）
    Layer2: 类别级路由 → 10个专家（每类一个）
    最终权重: w_text, w_video = MoE_Fusion(text_holo, video_holo, features)
    ↓
[PRM过程监督]
    scores = PRM([text_quality, video_quality, alignment, routing, fusion])
    ↓
[最终预测]
    logits = w_text * text_logits + w_video * video_logits
    final_logits = logits + λ * PRM_adjustment(scores)
```


## 3. 组件与接口

### 3.1 三模态融合模块（FR-1）

#### 3.1.1 设计思路

当前系统仅使用text和video两个模态，但数据集中28.5%的样本text为空，导致BERT编码失效。同时，video_description字段覆盖率达99.9%，包含丰富的视频内容描述。设计将video_description作为第三模态，实现：

1. **空文本回退**: text=""时自动使用video_description
2. **双段拼接**: text非空时拼接text + [SEP] + video_description
3. **统一编码**: 通过BERT的segment_ids区分两段文本

#### 3.1.2 组件设计

**TriModalTextEncoder**
```python
class TriModalTextEncoder(nn.Module):
    """三模态文本编码器，支持text和video_description的智能融合"""
    
    def __init__(self, args):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.bert_model)
        self.max_seq_length = args.max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    
    def forward(self, text, video_description, return_segments=False):
        """
        Args:
            text: [batch_size] 原始文本
            video_description: [batch_size] 视频描述
            return_segments: 是否返回segment信息（用于可视化）
        
        Returns:
            features: [batch_size, 768] BERT [CLS] token特征
            segments: [batch_size, max_seq_length] segment_ids（可选）
        """
        batch_size = len(text)
        input_ids_list = []
        attention_mask_list = []
        segment_ids_list = []
        
        for i in range(batch_size):
            if text[i].strip() == "":
                # 空文本：仅使用video_description
                tokens = self.tokenizer.tokenize(video_description[i])
                tokens = ["[CLS]"] + tokens[:self.max_seq_length-2] + ["[SEP]"]
                segment_ids = [0] * len(tokens)
            else:
                # 非空文本：拼接text + [SEP] + video_description
                text_tokens = self.tokenizer.tokenize(text[i])
                desc_tokens = self.tokenizer.tokenize(video_description[i])
                
                # 动态分配长度：text占60%，video_description占40%
                max_text_len = int((self.max_seq_length - 3) * 0.6)
                max_desc_len = self.max_seq_length - 3 - max_text_len
                
                text_tokens = text_tokens[:max_text_len]
                desc_tokens = desc_tokens[:max_desc_len]
                
                tokens = ["[CLS]"] + text_tokens + ["[SEP]"] + desc_tokens + ["[SEP]"]
                segment_ids = [0] * (len(text_tokens) + 2) + [1] * (len(desc_tokens) + 1)
            
            # 转换为IDs
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)
            
            # Padding
            padding_length = self.max_seq_length - len(input_ids)
            input_ids += [0] * padding_length
            attention_mask += [0] * padding_length
            segment_ids += [0] * padding_length
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            segment_ids_list.append(segment_ids)
        
        # 转换为tensor
        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids_list, dtype=torch.long)
        
        # BERT编码
        _, pooled_output = self.bert(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            attention_mask=attention_mask,
            output_all_encoded_layers=False
        )
        
        if return_segments:
            return pooled_output, segment_ids
        return pooled_output
```


#### 3.1.3 接口规范

**输入接口**
```python
{
    "text": str,                    # 原始文本，可能为空字符串
    "video_description": str,       # 视频描述，99.9%非空
    "max_seq_length": int = 128     # BERT最大序列长度
}
```

**输出接口**
```python
{
    "features": Tensor,             # [batch_size, 768] BERT特征
    "segments": Tensor,             # [batch_size, max_seq_length] segment_ids（可选）
    "modality_mask": Tensor         # [batch_size, 2] [has_text, has_video_desc]
}
```

#### 3.1.4 实现细节

1. **空文本检测**: 使用`text.strip() == ""`判断，避免仅包含空格的情况
2. **长度分配**: text占60%，video_description占40%，基于统计分析得出
3. **Segment IDs**: text段为0，video_description段为1，符合BERT NSP任务设计
4. **回退机制**: video_description也为空时（0.1%样本），使用特殊token "[EMPTY]"


### 3.2 LLM数据增强模块（FR-2）

#### 3.2.1 设计思路

数据集存在严重类别不平衡：技术操作仅38条，教师反馈161条，而教师讲授1577条（41:1比例）。传统数据增强（如回译、同义词替换）难以生成高质量的教育场景文本。设计使用大语言模型（DeepSeek-V3或GPT-4o-mini）生成训练数据：

1. **Few-shot提示**: 从原始数据中选择3-5个高质量样本作为示例
2. **约束生成**: 指定生成文本的长度、风格和关键词
3. **质量过滤**: 使用BERT相似度和人工审核双重过滤
4. **多样性保证**: 通过temperature和top-p采样增加多样性

#### 3.2.2 组件设计

**LLMAugmentor**
```python
class LLMAugmentor:
    """使用LLM生成训练数据的增强器"""
    
    def __init__(self, api_key, model="deepseek-chat", temperature=0.8):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = model
        self.temperature = temperature
        self.bert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    def generate_samples(self, class_name, seed_samples, target_count, 
                        min_similarity=0.7, max_similarity=0.95):
        """
        为指定类别生成训练样本
        
        Args:
            class_name: 类别名称（如"技术操作"）
            seed_samples: 种子样本列表 [{"text": ..., "video_description": ...}, ...]
            target_count: 目标生成数量
            min_similarity: 与种子样本的最小相似度（避免过于离谱）
            max_similarity: 与种子样本的最大相似度（避免过于相似）
        
        Returns:
            generated_samples: 生成的样本列表
        """
        generated = []
        attempts = 0
        max_attempts = target_count * 3
        
        # 构建few-shot prompt
        prompt_template = self._build_prompt(class_name, seed_samples)
        
        while len(generated) < target_count and attempts < max_attempts:
            attempts += 1
            
            # 调用LLM生成
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个教育场景数据生成专家。"},
                    {"role": "user", "content": prompt_template}
                ],
                temperature=self.temperature,
                max_tokens=200
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            # 质量过滤
            if self._quality_check(generated_text, seed_samples, 
                                  min_similarity, max_similarity):
                generated.append({
                    "text": generated_text,
                    "video_description": self._generate_video_desc(generated_text),
                    "label": class_name,
                    "source": "llm_augmented"
                })
        
        return generated
    
    def _build_prompt(self, class_name, seed_samples):
        """构建few-shot prompt"""
        examples = "\n\n".join([
            f"示例{i+1}:\n文本: {s['text']}\n视频描述: {s['video_description']}"
            for i, s in enumerate(seed_samples[:5])
        ])
        
        prompt = f"""请生成一个"{class_name}"类别的课堂行为描述。

已有示例：
{examples}

要求：
1. 文本长度在20-100字之间
2. 描述要具体、真实，符合实际课堂场景
3. 包含该类别的典型特征
4. 与示例相似但不完全相同

请生成：
文本:"""
        return prompt
    
    def _quality_check(self, generated_text, seed_samples, min_sim, max_sim):
        """质量检查：相似度过滤"""
        if len(generated_text) < 20 or len(generated_text) > 200:
            return False
        
        # 计算与种子样本的相似度
        gen_embedding = self.bert_model.encode([generated_text])[0]
        seed_embeddings = self.bert_model.encode([s['text'] for s in seed_samples])
        
        similarities = cosine_similarity([gen_embedding], seed_embeddings)[0]
        avg_similarity = similarities.mean()
        
        return min_sim <= avg_similarity <= max_sim
    
    def _generate_video_desc(self, text):
        """根据文本生成对应的video_description"""
        # 简化版：使用LLM生成
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "根据课堂文本描述，生成对应的视频画面描述。"},
                {"role": "user", "content": f"文本: {text}\n\n视频描述:"}
            ],
            temperature=0.7,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
```


#### 3.2.3 接口规范

**输入接口**
```python
{
    "class_name": str,              # 目标类别名称
    "seed_samples": List[Dict],     # 种子样本
    "target_count": int,            # 目标生成数量
    "api_config": {
        "api_key": str,
        "model": str,               # "deepseek-chat" 或 "gpt-4o-mini"
        "temperature": float        # 0.7-0.9
    }
}
```

**输出接口**
```python
{
    "generated_samples": List[Dict],  # 生成的样本
    "statistics": {
        "total_generated": int,
        "total_attempts": int,
        "avg_similarity": float,
        "quality_pass_rate": float
    }
}
```

#### 3.2.4 实现细节

1. **API选择**: 优先使用DeepSeek-V3（免费），备选GPT-4o-mini（$0.15/1M tokens）
2. **生成策略**: 每次生成1个样本，循环直到达到目标数量
3. **相似度阈值**: min=0.7（避免离题），max=0.95（避免重复）
4. **人工审核**: 生成后随机抽取20%进行人工审核，通过率需>80%


### 3.3 Mamba序列编码模块（FR-3）

#### 3.3.1 设计思路

当前系统使用LSTM处理视频时序，存在三个问题：
1. **长度限制**: LSTM的O(n)复杂度导致只能处理16帧，无法捕捉长时依赖
2. **速度瓶颈**: 序列计算无法并行，推理速度慢
3. **内存占用**: 双向LSTM需要存储所有时间步的隐状态

Mamba（基于状态空间模型SSM）提供了更好的解决方案：
- **线性复杂度**: O(n)时间但常数更小，支持100+帧
- **并行计算**: 训练时可并行，推理时使用递归模式
- **选择性机制**: 根据输入动态调整状态转移，比LSTM更灵活

#### 3.3.2 组件设计

**MambaVideoEncoder**
```python
from mamba_ssm import Mamba

class MambaVideoEncoder(nn.Module):
    """使用Mamba替换LSTM的视频编码器"""
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # ResNet50帧特征提取器（保持不变）
        resnet = torchvision.models.resnet50(pretrained=True)
        self.frame_encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 特征投影：ResNet50输出2048维 → Mamba输入维度
        self.frame_projection = nn.Linear(2048, args.mamba_d_model)
        
        # Mamba层：替换LSTM
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=args.mamba_d_model,      # 模型维度，如512
                d_state=args.mamba_d_state,      # SSM状态维度，如16
                d_conv=args.mamba_d_conv,        # 卷积核大小，如4
                expand=args.mamba_expand         # 扩展因子，如2
            )
            for _ in range(args.mamba_n_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(args.mamba_d_model)
            for _ in range(args.mamba_n_layers)
        ])
        
        # 时序池化
        self.temporal_pooling = args.video_pooling_type  # "last", "mean", "max"
        
        # 输出投影
        self.output_projection = nn.Linear(args.mamba_d_model, args.video_feature_dim)
        
        # 是否冻结CNN
        if args.freeze_video_cnn:
            for param in self.frame_encoder.parameters():
                param.requires_grad = False
    
    def forward(self, video_frames):
        """
        Args:
            video_frames: [batch_size, num_frames, 3, H, W]
        
        Returns:
            video_features: [batch_size, video_feature_dim]
        """
        batch_size, num_frames, C, H, W = video_frames.shape
        
        # 1. 提取帧特征
        frames_flat = video_frames.view(-1, C, H, W)
        with torch.set_grad_enabled(not self.args.freeze_video_cnn):
            frame_features = self.frame_encoder(frames_flat)  # [B*T, 2048, 7, 7]
        
        frame_features = self.spatial_pool(frame_features)    # [B*T, 2048, 1, 1]
        frame_features = frame_features.view(batch_size, num_frames, -1)  # [B, T, 2048]
        
        # 2. 投影到Mamba维度
        frame_features = self.frame_projection(frame_features)  # [B, T, d_model]
        
        # 3. Mamba时序编码（多层）
        x = frame_features
        for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
            # 残差连接 + Mamba + LayerNorm
            residual = x
            x = mamba_layer(x)
            x = layer_norm(x + residual)
        
        # 4. 时序池化
        if self.temporal_pooling == "last":
            video_features = x[:, -1, :]
        elif self.temporal_pooling == "mean":
            video_features = x.mean(dim=1)
        elif self.temporal_pooling == "max":
            video_features = x.max(dim=1)[0]
        else:
            video_features = x[:, -1, :]
        
        # 5. 输出投影
        video_features = self.output_projection(video_features)  # [B, video_feature_dim]
        
        return video_features
```


#### 3.3.3 接口规范

**输入接口**
```python
{
    "video_frames": Tensor,         # [batch_size, num_frames, 3, H, W]
    "mamba_config": {
        "d_model": int,             # 512（模型维度）
        "d_state": int,             # 16（SSM状态维度）
        "d_conv": int,              # 4（卷积核大小）
        "expand": int,              # 2（扩展因子）
        "n_layers": int             # 4（Mamba层数）
    }
}
```

**输出接口**
```python
{
    "video_features": Tensor,       # [batch_size, video_feature_dim]
    "temporal_states": Tensor       # [batch_size, num_frames, d_model]（可选，用于可视化）
}
```

#### 3.3.4 实现细节

1. **Mamba配置**: d_model=512, d_state=16, d_conv=4, expand=2, n_layers=4
2. **帧数支持**: 16-100帧，动态适应不同视频长度
3. **内存优化**: 使用梯度检查点（gradient checkpointing）减少内存占用
4. **速度优化**: 推理时使用Mamba的递归模式，避免重复计算
5. **兼容性**: 保持与原LSTM相同的输出维度，便于集成


### 3.4 Sparse MoE ConfidNet模块（FR-4）

#### 3.4.1 设计思路

当前系统为每个类别（10类）和每个模态（2个）分别训练独立的置信度网络，共20个ConfidNet。这导致：
1. **参数冗余**: 20个独立网络，参数量大
2. **训练困难**: 小样本类别的ConfidNet难以训练
3. **缺乏协同**: 各网络独立，无法共享知识

Sparse MoE（稀疏混合专家）提供了更优方案：
- **参数共享**: 8个专家共享，参数量减少50%
- **稀疏激活**: Top-2路由，每次只激活2个专家，计算高效
- **专家分工**: 2个共享专家 + 2个模态专家 + 4个场景专家
- **模态感知**: 路由器根据输入特征自动选择合适的专家

#### 3.4.2 组件设计

**SparseMoEConfidNet**
```python
class SparseMoEConfidNet(nn.Module):
    """稀疏混合专家置信度网络"""
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_experts = 8
        self.top_k = 2  # Top-2稀疏激活
        
        # 8个专家网络
        self.experts = nn.ModuleList([
            self._build_expert(args) for _ in range(self.n_experts)
        ])
        
        # 专家类型标记（用于可视化和分析）
        self.expert_types = [
            "shared_1", "shared_2",           # 0-1: 共享专家
            "text_specialist", "video_specialist",  # 2-3: 模态专家
            "lecture", "discussion", "writing", "silence"  # 4-7: 场景专家
        ]
        
        # 路由器：根据特征选择专家
        self.text_router = nn.Sequential(
            nn.Linear(args.hidden_sz, 256),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(256, self.n_experts)
        )
        
        self.video_router = nn.Sequential(
            nn.Linear(args.video_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(256, self.n_experts)
        )
        
        # 负载均衡损失权重
        self.load_balance_weight = args.moe_load_balance_weight
    
    def _build_expert(self, args):
        """构建单个专家网络"""
        return nn.Sequential(
            nn.Linear(args.hidden_sz + args.video_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, text_features, video_features, return_routing=False):
        """
        Args:
            text_features: [batch_size, hidden_sz]
            video_features: [batch_size, video_feature_dim]
            return_routing: 是否返回路由信息
        
        Returns:
            text_confidence: [batch_size, 1]
            video_confidence: [batch_size, 1]
            routing_info: Dict（可选）
        """
        batch_size = text_features.size(0)
        
        # 1. 计算路由权重
        text_routing_logits = self.text_router(text_features)  # [B, 8]
        video_routing_logits = self.video_router(video_features)  # [B, 8]
        
        # 2. Top-2稀疏选择
        text_top_k_logits, text_top_k_indices = torch.topk(
            text_routing_logits, self.top_k, dim=-1
        )  # [B, 2], [B, 2]
        video_top_k_logits, video_top_k_indices = torch.topk(
            video_routing_logits, self.top_k, dim=-1
        )
        
        # 3. Softmax归一化（仅对Top-2）
        text_top_k_weights = F.softmax(text_top_k_logits, dim=-1)  # [B, 2]
        video_top_k_weights = F.softmax(video_top_k_logits, dim=-1)
        
        # 4. 拼接特征
        combined_features = torch.cat([text_features, video_features], dim=-1)
        
        # 5. 计算专家输出
        expert_outputs = torch.stack([
            expert(combined_features) for expert in self.experts
        ], dim=1)  # [B, 8, 1]
        
        # 6. 加权聚合（仅Top-2专家）
        text_confidence = torch.zeros(batch_size, 1, device=text_features.device)
        video_confidence = torch.zeros(batch_size, 1, device=video_features.device)
        
        for i in range(batch_size):
            for k in range(self.top_k):
                expert_idx = text_top_k_indices[i, k]
                weight = text_top_k_weights[i, k]
                text_confidence[i] += weight * expert_outputs[i, expert_idx]
                
                expert_idx = video_top_k_indices[i, k]
                weight = video_top_k_weights[i, k]
                video_confidence[i] += weight * expert_outputs[i, expert_idx]
        
        # 7. 负载均衡损失（训练时）
        if self.training:
            self.load_balance_loss = self._compute_load_balance_loss(
                text_routing_logits, video_routing_logits
            )
        
        if return_routing:
            routing_info = {
                "text_top_k_indices": text_top_k_indices,
                "text_top_k_weights": text_top_k_weights,
                "video_top_k_indices": video_top_k_indices,
                "video_top_k_weights": video_top_k_weights,
                "expert_types": [self.expert_types[i] for i in range(self.n_experts)]
            }
            return text_confidence, video_confidence, routing_info
        
        return text_confidence, video_confidence
    
    def _compute_load_balance_loss(self, text_logits, video_logits):
        """计算负载均衡损失，鼓励专家均匀使用"""
        # 计算每个专家被选中的概率
        text_probs = F.softmax(text_logits, dim=-1)  # [B, 8]
        video_probs = F.softmax(video_logits, dim=-1)
        
        # 平均概率
        text_avg_probs = text_probs.mean(dim=0)  # [8]
        video_avg_probs = video_probs.mean(dim=0)
        
        # 理想情况：每个专家被选中概率为1/8
        target = 1.0 / self.n_experts
        
        # L2损失
        text_loss = ((text_avg_probs - target) ** 2).sum()
        video_loss = ((video_avg_probs - target) ** 2).sum()
        
        return self.load_balance_weight * (text_loss + video_loss)
```


#### 3.4.3 接口规范

**输入接口**
```python
{
    "text_features": Tensor,        # [batch_size, hidden_sz]
    "video_features": Tensor,       # [batch_size, video_feature_dim]
    "moe_config": {
        "n_experts": int,           # 8
        "top_k": int,               # 2
        "load_balance_weight": float  # 0.01
    }
}
```

**输出接口**
```python
{
    "text_confidence": Tensor,      # [batch_size, 1]
    "video_confidence": Tensor,     # [batch_size, 1]
    "routing_info": {
        "text_top_k_indices": Tensor,     # [batch_size, 2]
        "text_top_k_weights": Tensor,     # [batch_size, 2]
        "video_top_k_indices": Tensor,
        "video_top_k_weights": Tensor,
        "expert_types": List[str]         # 专家类型标记
    },
    "load_balance_loss": float      # 负载均衡损失
}
```

#### 3.4.4 实现细节

1. **专家分工**:
   - Expert 0-1: 共享专家，处理通用特征
   - Expert 2-3: 模态专家，分别专注于文本和视频
   - Expert 4-7: 场景专家，对应高频场景（讲授、讨论、板书、沉寂）

2. **模态感知路由**: 
   - 空文本样本（text=""）自动路由到video_specialist（Expert 3）
   - 通过在路由器中添加模态掩码实现

3. **负载均衡**: 
   - 使用辅助损失鼓励专家均匀使用
   - 权重设为0.01，避免过度约束

4. **训练策略**:
   - 先训练共享专家（Epoch 1-5）
   - 再训练模态专家（Epoch 6-10）
   - 最后训练场景专家（Epoch 11-15）


### 3.5 层次化MoE融合层（FR-5）

#### 3.5.1 设计思路

当前PDF框架使用硬编码的全息权重公式：
```python
text_holo = log(video_conf) / log(text_conf * video_conf)
video_holo = log(text_conf) / log(text_conf * video_conf)
```

这个公式虽然数学上优雅，但存在问题：
1. **固定公式**: 无法根据数据学习最优权重
2. **缺乏上下文**: 不考虑样本的类别、场景等信息
3. **单层融合**: 直接计算最终权重，缺乏层次性

设计将全息权重计算升级为两层MoE路由：
- **Layer 1（模态路由）**: 3个专家，决定模态偏好（text-dominant, video-dominant, balanced）
- **Layer 2（类别路由）**: 10个专家，每个类别一个，细化权重
- **保留PDF核心**: 将全息权重作为Layer 1的输入特征，而非直接输出

#### 3.5.2 组件设计

**HierarchicalMoEFusion**
```python
class HierarchicalMoEFusion(nn.Module):
    """层次化MoE融合层，替代硬编码全息权重"""
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_classes = args.n_classes
        
        # Layer 1: 模态路由MoE（3个专家）
        self.modality_experts = nn.ModuleList([
            self._build_modality_expert(args, expert_type)
            for expert_type in ["text_dominant", "video_dominant", "balanced"]
        ])
        
        self.modality_router = nn.Sequential(
            nn.Linear(args.hidden_sz + args.video_feature_dim + 4, 128),  # +4: 置信度和全息权重
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(128, 3)  # 3个模态专家
        )
        
        # Layer 2: 类别路由MoE（10个专家）
        self.class_experts = nn.ModuleList([
            self._build_class_expert(args)
            for _ in range(self.n_classes)
        ])
        
        self.class_router = nn.Sequential(
            nn.Linear(args.hidden_sz + args.video_feature_dim + 2, 128),  # +2: Layer1输出
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(128, self.n_classes)
        )
    
    def _build_modality_expert(self, args, expert_type):
        """构建模态专家"""
        # 每个专家有不同的偏好
        if expert_type == "text_dominant":
            # 文本主导：输出偏向文本的权重
            bias_init = torch.tensor([0.7, 0.3])
        elif expert_type == "video_dominant":
            # 视频主导：输出偏向视频的权重
            bias_init = torch.tensor([0.3, 0.7])
        else:
            # 平衡：输出均衡的权重
            bias_init = torch.tensor([0.5, 0.5])
        
        expert = nn.Sequential(
            nn.Linear(args.hidden_sz + args.video_feature_dim + 4, 128),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(128, 2)  # 输出 [w_text, w_video]
        )
        
        # 初始化偏置
        with torch.no_grad():
            expert[-1].bias.copy_(bias_init)
        
        return expert
    
    def _build_class_expert(self, args):
        """构建类别专家"""
        return nn.Sequential(
            nn.Linear(args.hidden_sz + args.video_feature_dim + 2, 64),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(64, 2)  # 输出权重调整量 [Δw_text, Δw_video]
        )
    
    def forward(self, text_features, video_features, text_conf, video_conf, 
                class_logits, return_routing=False):
        """
        Args:
            text_features: [batch_size, hidden_sz]
            video_features: [batch_size, video_feature_dim]
            text_conf: [batch_size, 1] 文本置信度
            video_conf: [batch_size, 1] 视频置信度
            class_logits: [batch_size, n_classes] 类别预测logits
            return_routing: 是否返回路由信息
        
        Returns:
            w_text: [batch_size, 1] 文本权重
            w_video: [batch_size, 1] 视频权重
            routing_info: Dict（可选）
        """
        batch_size = text_features.size(0)
        
        # 1. 计算全息权重（保留PDF核心思想）
        eps = 1e-8
        text_holo = torch.log(video_conf + eps) / (torch.log(text_conf * video_conf + eps) + eps)
        video_holo = torch.log(text_conf + eps) / (torch.log(text_conf * video_conf + eps) + eps)
        
        # 2. Layer 1: 模态路由
        # 拼接特征：[text_features, video_features, text_conf, video_conf, text_holo, video_holo]
        modality_input = torch.cat([
            text_features, video_features, 
            text_conf, video_conf, text_holo, video_holo
        ], dim=-1)
        
        modality_routing_logits = self.modality_router(modality_input)  # [B, 3]
        modality_weights = F.softmax(modality_routing_logits, dim=-1)  # [B, 3]
        
        # 计算每个模态专家的输出
        modality_expert_outputs = torch.stack([
            expert(modality_input) for expert in self.modality_experts
        ], dim=1)  # [B, 3, 2]
        
        # 加权聚合
        layer1_weights = torch.einsum('bn,bnd->bd', modality_weights, modality_expert_outputs)  # [B, 2]
        
        # 3. Layer 2: 类别路由
        # 拼接特征：[text_features, video_features, layer1_weights]
        class_input = torch.cat([
            text_features, video_features, layer1_weights
        ], dim=-1)
        
        class_routing_logits = self.class_router(class_input)  # [B, n_classes]
        
        # 使用类别预测概率作为路由权重
        class_probs = F.softmax(class_logits, dim=-1)  # [B, n_classes]
        
        # 计算每个类别专家的输出
        class_expert_outputs = torch.stack([
            expert(class_input) for expert in self.class_experts
        ], dim=1)  # [B, n_classes, 2]
        
        # 加权聚合（使用类别概率）
        layer2_adjustments = torch.einsum('bn,bnd->bd', class_probs, class_expert_outputs)  # [B, 2]
        
        # 4. 最终权重：Layer1输出 + Layer2调整
        final_weights = layer1_weights + 0.3 * layer2_adjustments  # [B, 2]
        
        # 5. Softmax归一化
        final_weights = F.softmax(final_weights, dim=-1)
        w_text = final_weights[:, 0:1]  # [B, 1]
        w_video = final_weights[:, 1:2]  # [B, 1]
        
        if return_routing:
            routing_info = {
                "holographic_weights": torch.cat([text_holo, video_holo], dim=-1),
                "layer1_modality_weights": modality_weights,
                "layer1_output": layer1_weights,
                "layer2_class_probs": class_probs,
                "layer2_adjustments": layer2_adjustments,
                "final_weights": final_weights
            }
            return w_text, w_video, routing_info
        
        return w_text, w_video
```


#### 3.5.3 接口规范

**输入接口**
```python
{
    "text_features": Tensor,        # [batch_size, hidden_sz]
    "video_features": Tensor,       # [batch_size, video_feature_dim]
    "text_conf": Tensor,            # [batch_size, 1]
    "video_conf": Tensor,           # [batch_size, 1]
    "class_logits": Tensor,         # [batch_size, n_classes]
    "fusion_config": {
        "n_modality_experts": int,  # 3
        "n_class_experts": int,     # 10
        "adjustment_weight": float  # 0.3
    }
}
```

**输出接口**
```python
{
    "w_text": Tensor,               # [batch_size, 1]
    "w_video": Tensor,              # [batch_size, 1]
    "routing_info": {
        "holographic_weights": Tensor,      # [batch_size, 2] PDF全息权重
        "layer1_modality_weights": Tensor,  # [batch_size, 3] 模态路由权重
        "layer1_output": Tensor,            # [batch_size, 2] Layer1输出
        "layer2_class_probs": Tensor,       # [batch_size, n_classes] 类别概率
        "layer2_adjustments": Tensor,       # [batch_size, 2] Layer2调整量
        "final_weights": Tensor             # [batch_size, 2] 最终权重
    }
}
```

#### 3.5.4 实现细节

1. **保留PDF核心**: 全息权重仍然计算，但作为Layer 1的输入特征，而非最终输出
2. **层次化设计**:
   - Layer 1: 粗粒度模态选择（文本主导/视频主导/平衡）
   - Layer 2: 细粒度类别调整（每个类别有不同的模态偏好）
3. **专家初始化**: 
   - text_dominant专家偏置初始化为[0.7, 0.3]
   - video_dominant专家偏置初始化为[0.3, 0.7]
   - balanced专家偏置初始化为[0.5, 0.5]
4. **调整权重**: Layer2的调整量乘以0.3，避免过度修正
5. **可解释性**: 返回每层的路由决策，便于分析和可视化


### 3.6 PRM过程监督模块（FR-6）

#### 3.6.1 设计思路

当前系统缺乏对融合过程的监督，导致：
1. **黑盒融合**: 不知道融合过程哪一步出问题
2. **缺乏反馈**: 无法针对性优化
3. **可解释性差**: 无法向用户解释预测依据

PRM（Process Reward Model）提供过程级监督：
- **五步评分**: 文本质量、视频质量、对齐度、路由决策、融合效果
- **LLM监督**: 使用GPT-4o-mini生成监督信号
- **端到端训练**: PRM评分作为辅助损失，指导模型优化

#### 3.6.2 组件设计

**ProcessRewardModel**
```python
class ProcessRewardModel(nn.Module):
    """过程奖励模型，提供五步监督信号"""
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # 五个评分器
        self.text_quality_scorer = self._build_scorer(args.hidden_sz, "text_quality")
        self.video_quality_scorer = self._build_scorer(args.video_feature_dim, "video_quality")
        self.alignment_scorer = self._build_scorer(
            args.hidden_sz + args.video_feature_dim, "alignment"
        )
        self.routing_scorer = self._build_scorer(
            args.hidden_sz + args.video_feature_dim + 10, "routing"  # +10: 路由信息
        )
        self.fusion_scorer = self._build_scorer(
            args.hidden_sz + args.video_feature_dim + 4, "fusion"  # +4: 融合权重和置信度
        )
        
        # 聚合器：将五步评分聚合为最终调整
        self.score_aggregator = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(32, args.n_classes)  # 输出类别调整量
        )
    
    def _build_scorer(self, input_dim, scorer_name):
        """构建单个评分器"""
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出0-1评分
        )
    
    def forward(self, text_features, video_features, routing_info, fusion_weights,
                text_conf, video_conf, return_scores=True):
        """
        Args:
            text_features: [batch_size, hidden_sz]
            video_features: [batch_size, video_feature_dim]
            routing_info: Dict，包含路由决策信息
            fusion_weights: [batch_size, 2] [w_text, w_video]
            text_conf: [batch_size, 1]
            video_conf: [batch_size, 1]
            return_scores: 是否返回详细评分
        
        Returns:
            adjustment: [batch_size, n_classes] 类别调整量
            scores: Dict（可选）五步评分
        """
        batch_size = text_features.size(0)
        
        # 1. 文本质量评分
        text_quality = self.text_quality_scorer(text_features)  # [B, 1]
        
        # 2. 视频质量评分
        video_quality = self.video_quality_scorer(video_features)  # [B, 1]
        
        # 3. 对齐度评分（文本-视频语义一致性）
        alignment_input = torch.cat([text_features, video_features], dim=-1)
        alignment = self.alignment_scorer(alignment_input)  # [B, 1]
        
        # 4. 路由决策评分（路由是否合理）
        # 提取路由信息：Top-2专家索引和权重
        text_routing = routing_info.get("text_top_k_weights", torch.zeros(batch_size, 2))
        video_routing = routing_info.get("video_top_k_weights", torch.zeros(batch_size, 2))
        modality_routing = routing_info.get("layer1_modality_weights", torch.zeros(batch_size, 3))
        
        routing_input = torch.cat([
            text_features, video_features,
            text_routing, video_routing, modality_routing
        ], dim=-1)
        routing_score = self.routing_scorer(routing_input)  # [B, 1]
        
        # 5. 融合效果评分（融合权重是否合理）
        fusion_input = torch.cat([
            text_features, video_features,
            fusion_weights, text_conf, video_conf
        ], dim=-1)
        fusion_score = self.fusion_scorer(fusion_input)  # [B, 1]
        
        # 6. 聚合五步评分
        all_scores = torch.cat([
            text_quality, video_quality, alignment, routing_score, fusion_score
        ], dim=-1)  # [B, 5]
        
        adjustment = self.score_aggregator(all_scores)  # [B, n_classes]
        
        if return_scores:
            scores = {
                "text_quality": text_quality,
                "video_quality": video_quality,
                "alignment": alignment,
                "routing": routing_score,
                "fusion": fusion_score,
                "overall": all_scores.mean(dim=-1, keepdim=True)  # [B, 1] 总体评分
            }
            return adjustment, scores
        
        return adjustment
```


#### 3.6.3 LLM监督信号生成

**PRMSupervisor**
```python
class PRMSupervisor:
    """使用LLM生成PRM监督信号"""
    
    def __init__(self, api_key, model="gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate_supervision(self, sample, model_outputs):
        """
        为单个样本生成五步监督信号
        
        Args:
            sample: {
                "text": str,
                "video_description": str,
                "label": str
            }
            model_outputs: {
                "text_features": Tensor,
                "video_features": Tensor,
                "prediction": str,
                "confidence": float
            }
        
        Returns:
            supervision: {
                "text_quality": float,      # 0-1
                "video_quality": float,
                "alignment": float,
                "routing": float,
                "fusion": float
            }
        """
        prompt = self._build_supervision_prompt(sample, model_outputs)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个教育场景分析专家，负责评估多模态模型的预测质量。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # 低温度，保证评分稳定
            max_tokens=500
        )
        
        # 解析LLM输出
        supervision = self._parse_llm_response(response.choices[0].message.content)
        return supervision
    
    def _build_supervision_prompt(self, sample, model_outputs):
        """构建监督信号生成prompt"""
        prompt = f"""请评估以下课堂行为识别样本的质量：

**输入信息**
- 文本: {sample['text']}
- 视频描述: {sample['video_description']}
- 真实标签: {sample['label']}

**模型输出**
- 预测标签: {model_outputs['prediction']}
- 预测置信度: {model_outputs['confidence']:.2f}

请从以下五个维度评分（0-1分）：

1. **文本质量**: 文本描述是否清晰、完整、信息丰富？
   - 0分：空文本或无意义
   - 0.5分：有一定信息但不完整
   - 1分：清晰完整，信息丰富

2. **视频质量**: 视频描述是否准确、详细？
   - 0分：描述模糊或错误
   - 0.5分：描述基本准确但不详细
   - 1分：描述准确详细

3. **对齐度**: 文本和视频描述是否语义一致？
   - 0分：完全不一致或矛盾
   - 0.5分：部分一致
   - 1分：高度一致

4. **路由决策**: 模型是否应该更依赖文本还是视频？
   - 如果文本质量高，应该依赖文本（评分接近1）
   - 如果视频质量高，应该依赖视频（评分接近0）
   - 如果两者都好，应该平衡（评分接近0.5）

5. **融合效果**: 模型的预测是否合理？
   - 0分：预测完全错误
   - 0.5分：预测部分正确
   - 1分：预测完全正确

请以JSON格式输出：
{{
    "text_quality": 0.0-1.0,
    "video_quality": 0.0-1.0,
    "alignment": 0.0-1.0,
    "routing": 0.0-1.0,
    "fusion": 0.0-1.0,
    "reasoning": "简要说明评分理由"
}}"""
        return prompt
    
    def _parse_llm_response(self, response_text):
        """解析LLM返回的JSON"""
        import json
        import re
        
        # 提取JSON部分
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                supervision = json.loads(json_match.group())
                return {
                    "text_quality": float(supervision.get("text_quality", 0.5)),
                    "video_quality": float(supervision.get("video_quality", 0.5)),
                    "alignment": float(supervision.get("alignment", 0.5)),
                    "routing": float(supervision.get("routing", 0.5)),
                    "fusion": float(supervision.get("fusion", 0.5))
                }
            except:
                pass
        
        # 解析失败，返回默认值
        return {
            "text_quality": 0.5,
            "video_quality": 0.5,
            "alignment": 0.5,
            "routing": 0.5,
            "fusion": 0.5
        }
```


#### 3.6.4 接口规范

**输入接口**
```python
{
    "text_features": Tensor,        # [batch_size, hidden_sz]
    "video_features": Tensor,       # [batch_size, video_feature_dim]
    "routing_info": Dict,           # MoE路由信息
    "fusion_weights": Tensor,       # [batch_size, 2]
    "text_conf": Tensor,            # [batch_size, 1]
    "video_conf": Tensor,           # [batch_size, 1]
    "supervision_signals": Tensor   # [batch_size, 5] LLM生成的监督信号（训练时）
}
```

**输出接口**
```python
{
    "adjustment": Tensor,           # [batch_size, n_classes] 类别调整量
    "scores": {
        "text_quality": Tensor,     # [batch_size, 1]
        "video_quality": Tensor,    # [batch_size, 1]
        "alignment": Tensor,        # [batch_size, 1]
        "routing": Tensor,          # [batch_size, 1]
        "fusion": Tensor,           # [batch_size, 1]
        "overall": Tensor           # [batch_size, 1]
    },
    "prm_loss": float               # PRM监督损失（训练时）
}
```

#### 3.6.5 实现细节

1. **监督信号生成**:
   - 使用GPT-4o-mini生成500个样本的监督信号
   - 成本：500样本 × 500 tokens ≈ $0.04
   - 生成后缓存，避免重复调用

2. **训练策略**:
   - Phase 1: 仅训练PRM（冻结其他模块）
   - Phase 2: 联合训练PRM和融合层
   - Phase 3: 端到端微调

3. **损失函数**:
   ```python
   prm_loss = MSE(predicted_scores, llm_supervision_signals)
   total_loss = classification_loss + λ_prm * prm_loss
   ```
   其中λ_prm = 0.1

4. **推理时使用**:
   - 输出五步评分，提供可解释性
   - 调整量加到最终logits上：`final_logits = logits + 0.2 * adjustment`


## 4. 数据模型

### 4.1 输入数据格式

**训练样本**
```json
{
    "id": "sample_0001",
    "text": "教师正在讲解数学公式，学生们认真听讲",
    "video_description": "教师站在黑板前，手持粉笔，正在板书数学公式。学生们坐在座位上，目光集中在黑板上。",
    "video_path": "videos/classroom_001.mp4",
    "label": "教师讲授",
    "label_id": 0,
    "metadata": {
        "duration": 30.5,
        "num_frames": 915,
        "resolution": "1920x1080",
        "source": "original"
    }
}
```

**增强样本**
```json
{
    "id": "augmented_0001",
    "text": "教师使用投影仪展示PPT，讲解课程内容",
    "video_description": "教室内，教师站在讲台旁，使用遥控器切换PPT页面。投影屏幕上显示课程内容。",
    "video_path": null,
    "label": "技术操作",
    "label_id": 9,
    "metadata": {
        "source": "llm_augmented",
        "seed_samples": ["sample_0038", "sample_0102"],
        "generation_model": "deepseek-chat",
        "similarity_score": 0.82
    }
}
```

### 4.2 模型中间表示

**特征表示**
```python
{
    "text_features": Tensor,        # [batch_size, 768] BERT [CLS] token
    "video_features": Tensor,       # [batch_size, 512] Mamba编码后的视频特征
    "text_logits": Tensor,          # [batch_size, 10] 文本分类logits
    "video_logits": Tensor,         # [batch_size, 10] 视频分类logits
    "modality_mask": Tensor         # [batch_size, 2] [has_text, has_video_desc]
}
```

**置信度和权重**
```python
{
    "text_confidence": Tensor,      # [batch_size, 1] 文本置信度
    "video_confidence": Tensor,     # [batch_size, 1] 视频置信度
    "holographic_weights": Tensor,  # [batch_size, 2] PDF全息权重
    "fusion_weights": Tensor,       # [batch_size, 2] 最终融合权重 [w_text, w_video]
}
```

**路由信息**
```python
{
    "moe_routing": {
        "text_top_k_indices": Tensor,     # [batch_size, 2] 文本激活的专家索引
        "text_top_k_weights": Tensor,     # [batch_size, 2] 文本专家权重
        "video_top_k_indices": Tensor,    # [batch_size, 2] 视频激活的专家索引
        "video_top_k_weights": Tensor,    # [batch_size, 2] 视频专家权重
        "expert_types": List[str]         # 专家类型标记
    },
    "fusion_routing": {
        "layer1_modality_weights": Tensor,  # [batch_size, 3] 模态路由权重
        "layer2_class_probs": Tensor        # [batch_size, 10] 类别路由权重
    }
}
```

**PRM评分**
```python
{
    "text_quality": Tensor,         # [batch_size, 1] 0-1评分
    "video_quality": Tensor,        # [batch_size, 1]
    "alignment": Tensor,            # [batch_size, 1]
    "routing": Tensor,              # [batch_size, 1]
    "fusion": Tensor,               # [batch_size, 1]
    "overall": Tensor               # [batch_size, 1] 总体评分
}
```

### 4.3 输出数据格式

**预测结果**
```json
{
    "id": "sample_0001",
    "prediction": "教师讲授",
    "prediction_id": 0,
    "confidence": 0.92,
    "fusion_weights": {
        "text": 0.65,
        "video": 0.35
    },
    "prm_scores": {
        "text_quality": 0.88,
        "video_quality": 0.91,
        "alignment": 0.85,
        "routing": 0.79,
        "fusion": 0.92,
        "overall": 0.87
    },
    "routing_info": {
        "text_experts": ["shared_1", "text_specialist"],
        "video_experts": ["shared_2", "lecture"],
        "modality_preference": "text_dominant"
    }
}
```


## 5. 正确性属性

*属性（Property）是系统在所有有效执行中都应该保持为真的特征或行为——本质上是关于系统应该做什么的形式化陈述。属性是人类可读规范和机器可验证正确性保证之间的桥梁。*

### 属性 1: 三模态输入处理

*对于任意*输入样本，系统应该能够正确处理text、video_description和video三种模态的任意组合，包括空文本的情况。

**验证**: 需求 FR-1.1, FR-1.2

### 属性 2: 空文本自动回退

*对于任意*text为空字符串的样本，系统应该自动使用video_description作为文本输入，并且路由器应该将该样本路由到视频相关专家（video_specialist或场景专家）。

**验证**: 需求 FR-1.2, FR-4.3

### 属性 3: 双段文本拼接

*对于任意*text和video_description都非空的样本，BERT编码器应该将两者拼接为"[CLS] text_tokens [SEP] video_desc_tokens [SEP]"格式，且segment_ids正确标记两段（text段为0，video_description段为1）。

**验证**: 需求 FR-1.3

### 属性 4: 长视频序列支持

*对于任意*帧数在[16, 100]范围内的视频输入，Mamba编码器应该能够成功处理并输出固定维度的特征向量，不产生内存溢出或维度错误。

**验证**: 需求 FR-3.1, NFR-2.3

### 属性 5: Top-2稀疏激活

*对于任意*输入样本，Sparse MoE ConfidNet应该恰好激活2个专家（每个模态），且激活权重之和为1。

**验证**: 需求 FR-4.2

### 属性 6: 全息权重保留

*对于任意*输入样本，层次化MoE融合层应该计算PDF全息权重（text_holo和video_holo），并将其作为Layer 1的输入特征，保持PDF框架的核心思想。

**验证**: 需求 FR-5.3

### 属性 7: PRM五步评分输出

*对于任意*输入样本，Process Reward Model应该输出恰好5个评分（text_quality, video_quality, alignment, routing, fusion），每个评分的值域为[0, 1]。

**验证**: 需求 FR-6.1, FR-6.3, NFR-3.3

### 属性 8: 融合权重归一化

*对于任意*输入样本，最终的融合权重w_text和w_video应该满足：w_text + w_video = 1，且w_text, w_video ∈ [0, 1]。

**验证**: 需求 NFR-3.1

### 属性 9: 可解释性输出完整性

*对于任意*输入样本，系统应该输出完整的可解释性信息，包括：融合权重（w_text, w_video）、MoE路由决策（激活的专家类型和权重）、PRM五步评分。

**验证**: 需求 NFR-3.1, NFR-3.2, NFR-3.3

### 属性 10: 缺失模态鲁棒性

*对于任意*存在缺失模态的样本（text=""或video损坏），系统应该能够优雅处理，不抛出异常，并基于可用模态进行预测。

**验证**: 需求 NFR-4.1


## 6. 错误处理

### 6.1 输入验证错误

**错误类型**: 输入数据格式不正确

**处理策略**:
```python
class InputValidationError(Exception):
    pass

def validate_input(sample):
    """验证输入样本格式"""
    if "text" not in sample or "video_description" not in sample:
        raise InputValidationError("Missing required fields: text or video_description")
    
    if sample["text"] == "" and sample["video_description"] == "":
        # 两者都为空，使用特殊token
        sample["video_description"] = "[EMPTY]"
        logging.warning(f"Sample {sample.get('id', 'unknown')} has empty text and video_description")
    
    if "video_path" in sample and sample["video_path"] is not None:
        if not os.path.exists(sample["video_path"]):
            raise InputValidationError(f"Video file not found: {sample['video_path']}")
    
    return sample
```

### 6.2 模型推理错误

**错误类型**: 模型前向传播失败

**处理策略**:
```python
def safe_forward(model, inputs, fallback_strategy="zero"):
    """安全的前向传播，带回退策略"""
    try:
        outputs = model(**inputs)
        return outputs
    except RuntimeError as e:
        if "out of memory" in str(e):
            # GPU内存不足
            logging.error("GPU OOM, trying CPU fallback")
            torch.cuda.empty_cache()
            inputs = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            outputs = model(**inputs)
            return {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                   for k, v in outputs.items()}
        elif "dimension" in str(e):
            # 维度不匹配
            logging.error(f"Dimension mismatch: {e}")
            if fallback_strategy == "zero":
                # 返回零向量
                return {"logits": torch.zeros(inputs["text_features"].size(0), 10)}
            else:
                raise
        else:
            raise
```

### 6.3 LLM API错误

**错误类型**: LLM API调用失败

**处理策略**:
```python
def call_llm_with_retry(client, prompt, max_retries=3, backoff_factor=2):
    """带重试的LLM调用"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                timeout=30
            )
            return response
        except openai.APIError as e:
            if attempt < max_retries - 1:
                wait_time = backoff_factor ** attempt
                logging.warning(f"LLM API error, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                logging.error(f"LLM API failed after {max_retries} attempts")
                # 返回默认监督信号
                return {
                    "text_quality": 0.5,
                    "video_quality": 0.5,
                    "alignment": 0.5,
                    "routing": 0.5,
                    "fusion": 0.5
                }
```

### 6.4 数据增强错误

**错误类型**: LLM生成的样本质量不合格

**处理策略**:
```python
def filter_generated_samples(generated_samples, seed_samples, quality_threshold=0.7):
    """过滤低质量生成样本"""
    filtered = []
    rejected = []
    
    for sample in generated_samples:
        # 质量检查
        quality_score = compute_quality_score(sample, seed_samples)
        
        if quality_score >= quality_threshold:
            filtered.append(sample)
        else:
            rejected.append({
                "sample": sample,
                "quality_score": quality_score,
                "reason": "Below quality threshold"
            })
    
    # 记录拒绝的样本
    if rejected:
        logging.info(f"Rejected {len(rejected)}/{len(generated_samples)} generated samples")
        with open("rejected_samples.json", "w") as f:
            json.dump(rejected, f, indent=2, ensure_ascii=False)
    
    return filtered
```

### 6.5 训练不稳定错误

**错误类型**: MoE训练时负载不均衡

**处理策略**:
```python
def monitor_expert_usage(routing_info, step):
    """监控专家使用情况"""
    expert_counts = torch.zeros(8)
    
    for indices in routing_info["text_top_k_indices"]:
        for idx in indices:
            expert_counts[idx] += 1
    
    for indices in routing_info["video_top_k_indices"]:
        for idx in indices:
            expert_counts[idx] += 1
    
    # 检查是否有专家使用率过低
    min_usage = expert_counts.min().item()
    max_usage = expert_counts.max().item()
    
    if max_usage > 0 and min_usage / max_usage < 0.1:
        logging.warning(f"Step {step}: Expert usage imbalance detected. "
                       f"Min: {min_usage}, Max: {max_usage}")
        logging.warning(f"Expert usage: {expert_counts.tolist()}")
        
        # 增加负载均衡损失权重
        return True  # 需要调整
    
    return False  # 正常
```


## 7. 测试策略

### 7.1 双重测试方法

本项目采用单元测试和基于属性的测试（Property-Based Testing, PBT）相结合的策略：

- **单元测试**: 验证特定示例、边缘情况和错误条件
- **基于属性的测试**: 通过随机化验证所有输入的通用属性
- **互补性**: 单元测试捕获具体错误，属性测试验证通用正确性

### 7.2 单元测试

单元测试专注于：
- 特定示例：演示正确行为的具体案例
- 组件集成点：模块之间的接口
- 边缘情况和错误条件：空输入、极端值、异常情况

**示例单元测试**:
```python
def test_trimodal_encoder_empty_text():
    """测试空文本时的回退机制"""
    encoder = TriModalTextEncoder(args)
    
    # 空文本样本
    text = [""]
    video_desc = ["教师在黑板前讲课"]
    
    features, segments = encoder(text, video_desc, return_segments=True)
    
    # 验证：使用video_description
    assert features.shape == (1, 768)
    assert (segments[0] == 0).all()  # 所有segment_ids应该为0

def test_mamba_encoder_long_video():
    """测试100帧长视频处理"""
    encoder = MambaVideoEncoder(args)
    
    # 100帧视频
    video = torch.randn(1, 100, 3, 224, 224)
    
    features = encoder(video)
    
    # 验证：输出维度正确
    assert features.shape == (1, 512)

def test_sparse_moe_top2_activation():
    """测试Top-2稀疏激活"""
    moe = SparseMoEConfidNet(args)
    
    text_features = torch.randn(4, 768)
    video_features = torch.randn(4, 512)
    
    text_conf, video_conf, routing = moe(
        text_features, video_features, return_routing=True
    )
    
    # 验证：每个样本恰好激活2个专家
    assert routing["text_top_k_indices"].shape == (4, 2)
    assert routing["video_top_k_indices"].shape == (4, 2)
    
    # 验证：权重和为1
    assert torch.allclose(
        routing["text_top_k_weights"].sum(dim=-1),
        torch.ones(4),
        atol=1e-6
    )
```

### 7.3 基于属性的测试

基于属性的测试使用Hypothesis库（Python）生成随机输入，验证通用属性。

**配置要求**:
- 最小迭代次数：100次（由于随机化）
- 每个测试必须引用设计文档中的属性
- 标签格式：**Feature: edupdf-improvement, Property {number}: {property_text}**

**示例属性测试**:
```python
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as npst

@given(
    batch_size=st.integers(min_value=1, max_value=16),
    num_frames=st.integers(min_value=16, max_value=100),
    has_text=st.booleans()
)
@settings(max_examples=100)
def test_property_4_long_video_support(batch_size, num_frames, has_text):
    """
    Feature: edupdf-improvement, Property 4: 长视频序列支持
    
    对于任意帧数在[16, 100]范围内的视频输入，Mamba编码器应该能够
    成功处理并输出固定维度的特征向量。
    """
    encoder = MambaVideoEncoder(args)
    
    # 生成随机视频
    video = torch.randn(batch_size, num_frames, 3, 224, 224)
    
    # 前向传播
    features = encoder(video)
    
    # 验证：输出维度固定
    assert features.shape == (batch_size, 512)
    assert not torch.isnan(features).any()
    assert not torch.isinf(features).any()

@given(
    batch_size=st.integers(min_value=1, max_value=16),
    text_features=npst.arrays(
        dtype=np.float32,
        shape=npst.array_shapes(min_dims=2, max_dims=2),
        elements=st.floats(min_value=-10, max_value=10, allow_nan=False)
    ),
    video_features=npst.arrays(
        dtype=np.float32,
        shape=npst.array_shapes(min_dims=2, max_dims=2),
        elements=st.floats(min_value=-10, max_value=10, allow_nan=False)
    )
)
@settings(max_examples=100)
def test_property_5_top2_sparse_activation(batch_size, text_features, video_features):
    """
    Feature: edupdf-improvement, Property 5: Top-2稀疏激活
    
    对于任意输入样本，Sparse MoE ConfidNet应该恰好激活2个专家，
    且激活权重之和为1。
    """
    moe = SparseMoEConfidNet(args)
    
    text_features = torch.from_numpy(text_features[:batch_size, :768])
    video_features = torch.from_numpy(video_features[:batch_size, :512])
    
    _, _, routing = moe(text_features, video_features, return_routing=True)
    
    # 验证：恰好激活2个专家
    assert routing["text_top_k_indices"].shape[1] == 2
    assert routing["video_top_k_indices"].shape[1] == 2
    
    # 验证：权重和为1
    text_weight_sum = routing["text_top_k_weights"].sum(dim=-1)
    video_weight_sum = routing["video_top_k_weights"].sum(dim=-1)
    
    assert torch.allclose(text_weight_sum, torch.ones(batch_size), atol=1e-5)
    assert torch.allclose(video_weight_sum, torch.ones(batch_size), atol=1e-5)

@given(
    batch_size=st.integers(min_value=1, max_value=16),
    text_empty_mask=st.lists(st.booleans(), min_size=1, max_size=16)
)
@settings(max_examples=100)
def test_property_2_empty_text_fallback(batch_size, text_empty_mask):
    """
    Feature: edupdf-improvement, Property 2: 空文本自动回退
    
    对于任意text为空字符串的样本，系统应该自动使用video_description
    作为文本输入，并路由到视频相关专家。
    """
    encoder = TriModalTextEncoder(args)
    moe = SparseMoEConfidNet(args)
    
    # 生成样本
    text = ["" if mask else "教师讲课" for mask in text_empty_mask[:batch_size]]
    video_desc = ["教师在黑板前" for _ in range(batch_size)]
    
    # 编码
    text_features = encoder(text, video_desc)
    video_features = torch.randn(batch_size, 512)
    
    # 路由
    _, _, routing = moe(text_features, video_features, return_routing=True)
    
    # 验证：空文本样本路由到video_specialist（Expert 3）
    for i, is_empty in enumerate(text_empty_mask[:batch_size]):
        if is_empty:
            activated_experts = routing["text_top_k_indices"][i].tolist()
            # 应该激活video_specialist或场景专家（索引3-7）
            assert any(idx >= 3 for idx in activated_experts), \
                f"Empty text sample should route to video experts, got {activated_experts}"

@given(
    batch_size=st.integers(min_value=1, max_value=16)
)
@settings(max_examples=100)
def test_property_8_fusion_weight_normalization(batch_size):
    """
    Feature: edupdf-improvement, Property 8: 融合权重归一化
    
    对于任意输入样本，最终的融合权重w_text和w_video应该满足：
    w_text + w_video = 1，且w_text, w_video ∈ [0, 1]。
    """
    fusion = HierarchicalMoEFusion(args)
    
    # 生成随机输入
    text_features = torch.randn(batch_size, 768)
    video_features = torch.randn(batch_size, 512)
    text_conf = torch.rand(batch_size, 1)
    video_conf = torch.rand(batch_size, 1)
    class_logits = torch.randn(batch_size, 10)
    
    # 计算融合权重
    w_text, w_video = fusion(
        text_features, video_features, text_conf, video_conf, class_logits
    )
    
    # 验证：权重和为1
    weight_sum = w_text + w_video
    assert torch.allclose(weight_sum, torch.ones(batch_size, 1), atol=1e-5)
    
    # 验证：权重在[0, 1]范围内
    assert (w_text >= 0).all() and (w_text <= 1).all()
    assert (w_video >= 0).all() and (w_video <= 1).all()

@given(
    batch_size=st.integers(min_value=1, max_value=16)
)
@settings(max_examples=100)
def test_property_7_prm_five_step_scores(batch_size):
    """
    Feature: edupdf-improvement, Property 7: PRM五步评分输出
    
    对于任意输入样本，Process Reward Model应该输出恰好5个评分，
    每个评分的值域为[0, 1]。
    """
    prm = ProcessRewardModel(args)
    
    # 生成随机输入
    text_features = torch.randn(batch_size, 768)
    video_features = torch.randn(batch_size, 512)
    routing_info = {
        "text_top_k_weights": torch.rand(batch_size, 2),
        "video_top_k_weights": torch.rand(batch_size, 2),
        "layer1_modality_weights": torch.rand(batch_size, 3)
    }
    fusion_weights = torch.rand(batch_size, 2)
    text_conf = torch.rand(batch_size, 1)
    video_conf = torch.rand(batch_size, 1)
    
    # 计算评分
    _, scores = prm(
        text_features, video_features, routing_info, 
        fusion_weights, text_conf, video_conf, return_scores=True
    )
    
    # 验证：恰好5个评分
    assert len(scores) == 6  # 5个单项 + 1个overall
    
    # 验证：每个评分在[0, 1]范围内
    for score_name, score_value in scores.items():
        assert score_value.shape == (batch_size, 1)
        assert (score_value >= 0).all() and (score_value <= 1).all(), \
            f"{score_name} score out of range [0, 1]"
```

### 7.4 集成测试

**端到端测试**:
```python
def test_end_to_end_pipeline():
    """测试完整的推理流程"""
    # 加载模型
    model = load_full_model(checkpoint_path)
    
    # 准备测试样本
    sample = {
        "text": "教师正在讲解数学公式",
        "video_description": "教师站在黑板前，手持粉笔",
        "video_path": "test_videos/sample_001.mp4"
    }
    
    # 推理
    result = model.predict(sample)
    
    # 验证输出完整性
    assert "prediction" in result
    assert "confidence" in result
    assert "fusion_weights" in result
    assert "prm_scores" in result
    assert "routing_info" in result
    
    # 验证输出合理性
    assert result["confidence"] >= 0 and result["confidence"] <= 1
    assert result["fusion_weights"]["text"] + result["fusion_weights"]["video"] == 1.0
```

### 7.5 性能测试

**推理速度测试**:
```python
def test_inference_speed():
    """测试推理速度是否满足要求（≤20ms）"""
    model = load_full_model(checkpoint_path)
    model.eval()
    
    # 准备100个测试样本
    test_samples = load_test_samples(n=100)
    
    # 预热
    for _ in range(10):
        model.predict(test_samples[0])
    
    # 测速
    start_time = time.time()
    for sample in test_samples:
        model.predict(sample)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / len(test_samples) * 1000  # ms
    
    assert avg_time <= 20, f"Inference too slow: {avg_time:.2f}ms > 20ms"
```


## 8. 实现规范

### 8.1 目录结构

```
src/
├── models/
│   ├── bert.py                      # BERT编码器（已有）
│   ├── video.py                     # 视频编码器（已有，需修改）
│   ├── latefusion_video.py          # PDF融合（已有，需修改）
│   ├── trimodal_encoder.py          # 新增：三模态编码器
│   ├── mamba_encoder.py             # 新增：Mamba视频编码器
│   ├── sparse_moe.py                # 新增：Sparse MoE ConfidNet
│   ├── hierarchical_fusion.py       # 新增：层次化MoE融合层
│   └── prm.py                       # 新增：Process Reward Model
├── data/
│   ├── dataset.py                   # 数据集加载（已有）
│   ├── augmentation.py              # 新增：LLM数据增强
│   └── prm_supervision.py           # 新增：PRM监督信号生成
├── training/
│   ├── trainer.py                   # 训练器（已有，需修改）
│   ├── losses.py                    # 损失函数（已有，需修改）
│   └── callbacks.py                 # 新增：训练回调（监控专家使用率）
├── evaluation/
│   ├── metrics.py                   # 评估指标（已有）
│   └── visualization.py             # 新增：可视化工具
└── utils/
    ├── config.py                    # 配置管理（已有）
    └── logging.py                   # 日志工具（已有）

tests/
├── unit/
│   ├── test_trimodal_encoder.py
│   ├── test_mamba_encoder.py
│   ├── test_sparse_moe.py
│   ├── test_hierarchical_fusion.py
│   └── test_prm.py
├── property/
│   ├── test_properties_fr1.py       # FR-1相关属性测试
│   ├── test_properties_fr3.py       # FR-3相关属性测试
│   ├── test_properties_fr4.py       # FR-4相关属性测试
│   ├── test_properties_fr5.py       # FR-5相关属性测试
│   └── test_properties_fr6.py       # FR-6相关属性测试
└── integration/
    └── test_end_to_end.py

configs/
├── base.yaml                        # 基础配置
├── phase1.yaml                      # Phase 1配置
├── phase2.yaml                      # Phase 2配置
└── phase3.yaml                      # Phase 3配置
```

### 8.2 配置管理

**base.yaml**
```yaml
# 模型配置
model:
  bert_model: "bert-base-chinese"
  hidden_sz: 768
  video_feature_dim: 512
  n_classes: 10
  dropout: 0.3

# 三模态配置
trimodal:
  max_seq_length: 128
  text_ratio: 0.6
  video_desc_ratio: 0.4

# Mamba配置
mamba:
  d_model: 512
  d_state: 16
  d_conv: 4
  expand: 2
  n_layers: 4

# Sparse MoE配置
sparse_moe:
  n_experts: 8
  top_k: 2
  load_balance_weight: 0.01
  expert_types:
    - "shared_1"
    - "shared_2"
    - "text_specialist"
    - "video_specialist"
    - "lecture"
    - "discussion"
    - "writing"
    - "silence"

# 层次化融合配置
hierarchical_fusion:
  n_modality_experts: 3
  n_class_experts: 10
  adjustment_weight: 0.3

# PRM配置
prm:
  prm_loss_weight: 0.1
  adjustment_weight: 0.2

# 训练配置
training:
  batch_size: 32
  learning_rate: 2e-5
  num_epochs: 20
  warmup_steps: 500
  gradient_clip: 1.0
  
# 数据增强配置
augmentation:
  llm_model: "deepseek-chat"
  api_key: "${DEEPSEEK_API_KEY}"
  temperature: 0.8
  min_similarity: 0.7
  max_similarity: 0.95
  target_counts:
    技术操作: 200
    教师反馈: 400
```

### 8.3 API接口

**模型推理接口**
```python
class EduPDFModel:
    """eduPDF改进模型的统一接口"""
    
    def __init__(self, config_path, checkpoint_path=None):
        """
        初始化模型
        
        Args:
            config_path: 配置文件路径
            checkpoint_path: 检查点路径（可选）
        """
        self.config = load_config(config_path)
        self._build_model()
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
    
    def predict(self, sample, return_details=False):
        """
        预测单个样本
        
        Args:
            sample: {
                "text": str,
                "video_description": str,
                "video_path": str (可选)
            }
            return_details: 是否返回详细信息
        
        Returns:
            result: {
                "prediction": str,
                "prediction_id": int,
                "confidence": float,
                "fusion_weights": {"text": float, "video": float},
                "prm_scores": {...},  # 如果return_details=True
                "routing_info": {...}  # 如果return_details=True
            }
        """
        pass
    
    def predict_batch(self, samples, batch_size=32):
        """批量预测"""
        pass
    
    def explain(self, sample):
        """
        解释预测结果
        
        Returns:
            explanation: {
                "prediction": str,
                "confidence": float,
                "key_factors": [
                    {"factor": "text_quality", "score": 0.88, "impact": "high"},
                    {"factor": "video_quality", "score": 0.91, "impact": "high"},
                    ...
                ],
                "routing_decision": {
                    "modality_preference": "text_dominant",
                    "activated_experts": ["shared_1", "text_specialist"],
                    "reasoning": "文本质量高，内容完整"
                },
                "fusion_strategy": {
                    "text_weight": 0.65,
                    "video_weight": 0.35,
                    "reasoning": "文本信息更可靠"
                }
            }
        """
        pass
```

**数据增强接口**
```python
class LLMAugmentor:
    """LLM数据增强器"""
    
    def augment_class(self, class_name, seed_samples, target_count):
        """
        为指定类别生成增强数据
        
        Args:
            class_name: 类别名称
            seed_samples: 种子样本列表
            target_count: 目标生成数量
        
        Returns:
            generated_samples: 生成的样本列表
            statistics: 生成统计信息
        """
        pass
    
    def augment_dataset(self, dataset_path, output_path, target_counts):
        """
        批量增强整个数据集
        
        Args:
            dataset_path: 原始数据集路径
            output_path: 输出路径
            target_counts: 每个类别的目标数量 {"技术操作": 200, ...}
        """
        pass
```

### 8.4 依赖管理

**requirements.txt**
```
torch>=2.0.0
transformers>=4.30.0
mamba-ssm>=1.0.0
pytorch-pretrained-bert>=0.6.2
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
scikit-learn>=1.3.0
hypothesis>=6.90.0
openai>=1.0.0
sentence-transformers>=2.2.0
pyyaml>=6.0
tensorboard>=2.14.0
pytest>=7.4.0
```

**安装说明**
```bash
# 1. 创建虚拟环境
conda create -n edupdf python=3.10
conda activate edupdf

# 2. 安装PyTorch（根据CUDA版本）
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118

# 3. 安装Mamba
pip install mamba-ssm

# 4. 安装其他依赖
pip install -r requirements.txt

# 5. 验证安装
python -c "import torch; import mamba_ssm; print('Installation successful')"
```


## 9. 实施计划

### 9.1 Phase 1: 数据基建（Week 1-2）

**目标**: 完成三模态融合和LLM数据增强

**任务**:
1. 实现TriModalTextEncoder
   - 支持text和video_description拼接
   - 实现空文本回退机制
   - 单元测试和属性测试

2. 实现LLMAugmentor
   - 集成DeepSeek-V3 API
   - 实现质量过滤机制
   - 生成技术操作200条、教师反馈400条

3. 数据集更新
   - 合并原始数据和增强数据
   - 重新划分训练/验证/测试集
   - 数据统计分析

**验收标准**:
- 三模态编码器通过所有单元测试和属性测试
- 生成数据人工审核通过率>80%
- 整体准确率提升8-13%

### 9.2 Phase 2: 架构创新（Week 3-6）

**目标**: 实现Mamba编码器、Sparse MoE和层次化融合

**Week 3-4: Mamba编码器**
1. 实现MambaVideoEncoder
   - 替换LSTM为Mamba
   - 支持16-100帧输入
   - 内存优化和速度优化

2. 集成到现有系统
   - 修改latefusion_video.py
   - 保持接口兼容性

3. 测试和调优
   - 单元测试和属性测试
   - 推理速度测试（目标5倍提升）

**Week 5: Sparse MoE ConfidNet**
1. 实现SparseMoEConfidNet
   - 8个专家网络
   - Top-2稀疏路由
   - 负载均衡损失

2. 替换原有ConfidNet
   - 修改latefusion_video.py
   - 迁移训练权重

3. 测试和监控
   - 专家使用率监控
   - 参数量对比

**Week 6: 层次化MoE融合层**
1. 实现HierarchicalMoEFusion
   - Layer 1: 模态路由
   - Layer 2: 类别路由
   - 保留全息权重计算

2. 集成到融合流程
   - 替换硬编码权重
   - 保持PDF核心思想

3. 可解释性输出
   - 路由决策可视化
   - 权重分析工具

**验收标准**:
- Mamba编码器推理速度提升5倍
- Sparse MoE参数量减少50%
- 整体准确率提升19-29%（累计）

### 9.3 Phase 3: PRM过程监督（Week 7-10）

**目标**: 实现PRM并完成端到端优化

**Week 7-8: PRM实现**
1. 实现ProcessRewardModel
   - 五个评分器
   - 评分聚合器

2. LLM监督信号生成
   - 使用GPT-4o-mini生成500个样本的监督信号
   - 缓存监督信号

3. PRM训练
   - 冻结其他模块，仅训练PRM
   - 监督损失优化

**Week 9: 联合训练**
1. PRM与融合层联合训练
   - 端到端优化
   - 损失权重调优

2. 可解释性验证
   - 五步评分输出
   - 案例分析

**Week 10: 最终优化**
1. 超参数调优
   - 学习率、批大小、损失权重
   - 网格搜索或贝叶斯优化

2. 消融实验
   - 验证每个模块的贡献
   - 准备论文实验数据

3. 性能测试
   - 推理速度、内存占用
   - 准确率、F1等指标

**验收标准**:
- PRM输出五步评分
- 整体准确率目标：88-93%（较基线75%提升13-18个百分点）
- 推理速度≤20ms，内存≤4GB

### 9.4 里程碑和交付物

| 里程碑 | 时间 | 交付物 | 验收标准 |
|--------|------|--------|---------|
| M1: Phase 1完成 | Week 2 | 三模态编码器 + LLM增强数据 | 准确率+8-13% |
| M2: Mamba实现 | Week 4 | Mamba视频编码器 | 速度提升5倍 |
| M3: Sparse MoE实现 | Week 5 | Sparse MoE ConfidNet | 参数量-50% |
| M4: 层次化融合实现 | Week 6 | 层次化MoE融合层 | 准确率+19-29% |
| M5: PRM实现 | Week 8 | Process Reward Model | 输出五步评分 |
| M6: 最终优化 | Week 10 | 完整系统 + 实验报告 | 目标准确率 88-93% |

### 9.5 风险管理

**技术风险**:
1. **Mamba实现困难**
   - 风险：官方库API不稳定
   - 缓解：使用稳定版本（mamba-ssm==1.0.0），准备LSTM回退方案

2. **MoE训练不稳定**
   - 风险：专家使用不均衡
   - 缓解：添加负载均衡损失，监控专家使用率

3. **LLM生成质量差**
   - 风险：生成数据不符合要求
   - 缓解：人工审核，调整prompt，增加过滤阈值

**时间风险**:
1. **Phase 2超时**
   - 风险：Mamba和MoE实现复杂
   - 缓解：优先实现核心功能，可选功能后置

2. **PRM训练数据不足**
   - 风险：500个样本不够
   - 缓解：使用GPT-4生成伪标签，扩充到1000个


## 10. 性能优化

### 10.1 推理速度优化

**目标**: 推理速度≤20ms/样本

**优化策略**:

1. **模型量化**
   ```python
   # 使用PyTorch的动态量化
   model_int8 = torch.quantization.quantize_dynamic(
       model, {nn.Linear}, dtype=torch.qint8
   )
   # 预期速度提升：2-3倍
   ```

2. **批处理优化**
   ```python
   # 使用更大的批大小进行推理
   batch_size = 64  # 训练时32，推理时64
   # 预期速度提升：1.5-2倍
   ```

3. **Mamba递归模式**
   ```python
   # 推理时使用递归模式，避免重复计算
   mamba_layer.inference_mode = "recurrent"
   # 预期速度提升：3-5倍
   ```

4. **ONNX导出**
   ```python
   # 导出为ONNX格式，使用ONNX Runtime推理
   torch.onnx.export(model, dummy_input, "model.onnx")
   # 预期速度提升：1.5-2倍
   ```

**预期效果**:
- 基线：100ms/样本
- 优化后：15-20ms/样本（5-6倍提升）

### 10.2 内存优化

**目标**: 训练≤16GB，推理≤4GB

**优化策略**:

1. **梯度检查点**
   ```python
   # 在Mamba和BERT中使用梯度检查点
   self.mamba_layers = nn.ModuleList([
       torch.utils.checkpoint.checkpoint_wrapper(
           Mamba(...), preserve_rng_state=False
       )
       for _ in range(n_layers)
   ])
   # 内存节省：40-60%
   ```

2. **混合精度训练**
   ```python
   # 使用FP16训练
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   with autocast():
       outputs = model(inputs)
       loss = criterion(outputs, labels)
   scaler.scale(loss).backward()
   # 内存节省：30-50%
   ```

3. **参数共享**
   ```python
   # MoE专家之间共享部分参数
   shared_layers = nn.Sequential(...)
   for expert in self.experts:
       expert.shared = shared_layers
   # 内存节省：20-30%
   ```

4. **推理时冻结**
   ```python
   # 推理时冻结不需要的模块
   model.eval()
   with torch.no_grad():
       outputs = model(inputs)
   # 内存节省：50-70%
   ```

**预期效果**:
- 训练：24GB → 12-14GB
- 推理：8GB → 2-3GB

### 10.3 训练速度优化

**目标**: 训练速度≤10小时/epoch

**优化策略**:

1. **数据加载优化**
   ```python
   # 使用多进程数据加载
   dataloader = DataLoader(
       dataset, 
       batch_size=32,
       num_workers=8,  # 8个进程
       pin_memory=True,
       prefetch_factor=2
   )
   # 速度提升：2-3倍
   ```

2. **分布式训练**
   ```python
   # 使用DataParallel或DistributedDataParallel
   model = nn.DataParallel(model, device_ids=[0, 1])
   # 速度提升：1.8倍（2卡）
   ```

3. **梯度累积**
   ```python
   # 使用梯度累积模拟更大的批大小
   accumulation_steps = 4
   for i, batch in enumerate(dataloader):
       loss = model(batch) / accumulation_steps
       loss.backward()
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   # 速度提升：1.5-2倍
   ```

**预期效果**:
- 基线：20小时/epoch
- 优化后：8-10小时/epoch

### 10.4 模型压缩

**目标**: 模型大小≤500MB

**压缩策略**:

1. **知识蒸馏**
   ```python
   # 使用大模型（teacher）训练小模型（student）
   teacher_model = load_full_model()
   student_model = build_smaller_model()
   
   distillation_loss = KL_divergence(
       student_logits, teacher_logits
   )
   # 模型大小：1.2GB → 400MB
   ```

2. **剪枝**
   ```python
   # 剪枝不重要的连接
   import torch.nn.utils.prune as prune
   
   prune.l1_unstructured(model.layer, name='weight', amount=0.3)
   # 模型大小：减少30%
   ```

3. **低秩分解**
   ```python
   # 对大矩阵进行低秩分解
   W = nn.Linear(768, 768)
   # 分解为 W = U @ V，其中U: 768x128, V: 128x768
   # 参数量：768*768 → 768*128 + 128*768（减少83%）
   ```

**预期效果**:
- 完整模型：1.2GB
- 压缩后：400-500MB（保持95%+准确率）


## 11. 可扩展性设计

### 11.1 新增类别

**设计目标**: 支持新增类别，无需重新训练整个模型

**实现方案**:

1. **类别专家扩展**
   ```python
   class ExtensibleClassExperts(nn.Module):
       def __init__(self, initial_n_classes=10):
           super().__init__()
           self.experts = nn.ModuleDict()
           for i in range(initial_n_classes):
               self.experts[f"class_{i}"] = self._build_expert()
       
       def add_class(self, class_name):
           """添加新类别专家"""
           if class_name not in self.experts:
               self.experts[class_name] = self._build_expert()
               # 仅训练新专家，冻结其他模块
               for param in self.experts[class_name].parameters():
                   param.requires_grad = True
   ```

2. **增量学习**
   ```python
   def incremental_train(model, new_class_data, freeze_old=True):
       """增量训练新类别"""
       # 冻结旧模块
       if freeze_old:
           for name, param in model.named_parameters():
               if "new_class" not in name:
                   param.requires_grad = False
       
       # 仅训练新类别相关模块
       optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()))
       
       for epoch in range(num_epochs):
           for batch in new_class_data:
               loss = model(batch)
               loss.backward()
               optimizer.step()
   ```

**使用示例**:
```python
# 添加新类别"学生实验"
model.add_class("学生实验")

# 准备新类别数据
new_class_data = load_new_class_data("学生实验")

# 增量训练
incremental_train(model, new_class_data, freeze_old=True)
```

### 11.2 新增模态

**设计目标**: 支持新增模态（如音频），无需重构整个系统

**实现方案**:

1. **模态注册机制**
   ```python
   class ModalityRegistry:
       def __init__(self):
           self.encoders = {}
           self.fusion_weights = {}
       
       def register_modality(self, name, encoder, initial_weight=0.33):
           """注册新模态"""
           self.encoders[name] = encoder
           self.fusion_weights[name] = initial_weight
       
       def encode(self, inputs):
           """编码所有模态"""
           features = {}
           for name, encoder in self.encoders.items():
               if name in inputs:
                   features[name] = encoder(inputs[name])
           return features
   ```

2. **动态融合**
   ```python
   class DynamicFusion(nn.Module):
       def __init__(self, registry):
           super().__init__()
           self.registry = registry
           self.fusion_router = nn.ModuleDict()
       
       def forward(self, features):
           """动态融合所有可用模态"""
           available_modalities = list(features.keys())
           n_modalities = len(available_modalities)
           
           # 计算每个模态的权重
           weights = self._compute_weights(features, available_modalities)
           
           # 加权融合
           fused = sum(weights[m] * features[m] for m in available_modalities)
           return fused
   ```

**使用示例**:
```python
# 注册音频模态
audio_encoder = AudioEncoder(args)
registry.register_modality("audio", audio_encoder)

# 使用
inputs = {
    "text": text_data,
    "video": video_data,
    "audio": audio_data  # 新增
}
features = registry.encode(inputs)
output = fusion(features)
```

### 11.3 新增专家

**设计目标**: 支持动态添加MoE专家

**实现方案**:

1. **专家池管理**
   ```python
   class ExpertPool(nn.Module):
       def __init__(self, initial_experts=8):
           super().__init__()
           self.experts = nn.ModuleList([
               self._build_expert() for _ in range(initial_experts)
           ])
           self.expert_types = []
       
       def add_expert(self, expert_type):
           """添加新专家"""
           new_expert = self._build_expert()
           self.experts.append(new_expert)
           self.expert_types.append(expert_type)
           
           # 扩展路由器
           self._expand_router()
       
       def _expand_router(self):
           """扩展路由器以支持新专家"""
           old_router = self.router
           n_experts = len(self.experts)
           
           self.router = nn.Linear(
               old_router.in_features, n_experts
           )
           
           # 复制旧权重
           with torch.no_grad():
               self.router.weight[:old_router.out_features] = old_router.weight
               self.router.bias[:old_router.out_features] = old_router.bias
   ```

**使用示例**:
```python
# 添加新专家"group_work"
expert_pool.add_expert("group_work")

# 微调新专家
finetune_expert(expert_pool.experts[-1], group_work_data)
```

### 11.4 配置驱动

**设计目标**: 通过配置文件控制模型结构，无需修改代码

**实现方案**:

```yaml
# config.yaml
model:
  modalities:
    - name: "text"
      encoder: "BertEncoder"
      config:
        model_name: "bert-base-chinese"
        hidden_size: 768
    
    - name: "video"
      encoder: "MambaVideoEncoder"
      config:
        d_model: 512
        n_layers: 4
    
    - name: "audio"  # 新增模态
      encoder: "AudioEncoder"
      config:
        sample_rate: 16000
        n_mels: 128
  
  experts:
    - type: "shared"
      count: 2
    - type: "modality"
      count: 3  # text, video, audio
    - type: "scene"
      count: 4
  
  fusion:
    type: "hierarchical_moe"
    layers:
      - name: "modality_routing"
        n_experts: 3
      - name: "class_routing"
        n_experts: 10
```

```python
# 从配置构建模型
def build_model_from_config(config_path):
    config = load_yaml(config_path)
    
    # 构建编码器
    encoders = {}
    for modality in config["model"]["modalities"]:
        encoder_class = globals()[modality["encoder"]]
        encoders[modality["name"]] = encoder_class(**modality["config"])
    
    # 构建专家
    experts = build_experts(config["model"]["experts"])
    
    # 构建融合层
    fusion = build_fusion(config["model"]["fusion"])
    
    return Model(encoders, experts, fusion)
```


## 12. 监控与可视化

### 12.1 训练监控

**监控指标**:

1. **损失曲线**
   - 分类损失（classification_loss）
   - MoE负载均衡损失（load_balance_loss）
   - PRM监督损失（prm_loss）
   - 总损失（total_loss）

2. **准确率指标**
   - 整体准确率
   - 每个类别的准确率
   - 空文本样本准确率
   - 小样本类别准确率

3. **专家使用率**
   - 每个专家被激活的频率
   - 专家负载均衡度
   - 模态专家vs场景专家使用对比

4. **融合权重分布**
   - w_text和w_video的分布
   - 不同类别的权重偏好
   - 空文本样本的权重分布

**实现**:
```python
class TrainingMonitor:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.step = 0
    
    def log_losses(self, losses):
        """记录损失"""
        for name, value in losses.items():
            self.writer.add_scalar(f"Loss/{name}", value, self.step)
    
    def log_metrics(self, metrics):
        """记录指标"""
        for name, value in metrics.items():
            self.writer.add_scalar(f"Metrics/{name}", value, self.step)
    
    def log_expert_usage(self, routing_info):
        """记录专家使用情况"""
        expert_counts = torch.zeros(8)
        for indices in routing_info["text_top_k_indices"]:
            for idx in indices:
                expert_counts[idx] += 1
        
        # 绘制柱状图
        fig, ax = plt.subplots()
        ax.bar(range(8), expert_counts.cpu().numpy())
        ax.set_xlabel("Expert ID")
        ax.set_ylabel("Usage Count")
        ax.set_title("Expert Usage Distribution")
        self.writer.add_figure("Expert/Usage", fig, self.step)
    
    def log_fusion_weights(self, fusion_weights, labels):
        """记录融合权重分布"""
        w_text = fusion_weights[:, 0].cpu().numpy()
        w_video = fusion_weights[:, 1].cpu().numpy()
        
        # 按类别分组
        for class_id in range(10):
            mask = (labels == class_id)
            if mask.sum() > 0:
                class_w_text = w_text[mask]
                self.writer.add_histogram(
                    f"FusionWeights/class_{class_id}_text",
                    class_w_text, self.step
                )
```

### 12.2 可视化工具

**1. 注意力可视化**
```python
def visualize_attention(text, video_desc, attention_weights):
    """可视化BERT的注意力权重"""
    tokens = tokenizer.tokenize(text + " [SEP] " + video_desc)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        attention_weights.cpu().numpy(),
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="YlOrRd",
        ax=ax
    )
    ax.set_title("BERT Attention Weights")
    return fig
```

**2. 路由决策可视化**
```python
def visualize_routing(sample, routing_info):
    """可视化MoE路由决策"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 文本路由
    text_experts = routing_info["text_top_k_indices"][0].cpu().numpy()
    text_weights = routing_info["text_top_k_weights"][0].cpu().numpy()
    
    ax1.bar(text_experts, text_weights)
    ax1.set_xlabel("Expert ID")
    ax1.set_ylabel("Weight")
    ax1.set_title("Text Routing")
    
    # 视频路由
    video_experts = routing_info["video_top_k_indices"][0].cpu().numpy()
    video_weights = routing_info["video_top_k_weights"][0].cpu().numpy()
    
    ax2.bar(video_experts, video_weights)
    ax2.set_xlabel("Expert ID")
    ax2.set_ylabel("Weight")
    ax2.set_title("Video Routing")
    
    return fig
```

**3. PRM评分可视化**
```python
def visualize_prm_scores(sample, scores):
    """可视化PRM五步评分"""
    score_names = ["Text Quality", "Video Quality", "Alignment", "Routing", "Fusion"]
    score_values = [
        scores["text_quality"].item(),
        scores["video_quality"].item(),
        scores["alignment"].item(),
        scores["routing"].item(),
        scores["fusion"].item()
    ]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 雷达图
    angles = np.linspace(0, 2 * np.pi, len(score_names), endpoint=False).tolist()
    score_values += score_values[:1]
    angles += angles[:1]
    
    ax = plt.subplot(111, projection='polar')
    ax.plot(angles, score_values, 'o-', linewidth=2)
    ax.fill(angles, score_values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(score_names)
    ax.set_ylim(0, 1)
    ax.set_title("PRM Process Scores")
    
    return fig
```

**4. 融合过程可视化**
```python
def visualize_fusion_process(sample, model_outputs):
    """可视化完整的融合过程"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. 输入模态
    ax1 = fig.add_subplot(gs[0, :])
    ax1.text(0.1, 0.5, f"Text: {sample['text']}", fontsize=10)
    ax1.text(0.1, 0.3, f"Video Desc: {sample['video_description']}", fontsize=10)
    ax1.axis('off')
    ax1.set_title("Input Modalities")
    
    # 2. 置信度
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(["Text", "Video"], [
        model_outputs["text_confidence"].item(),
        model_outputs["video_confidence"].item()
    ])
    ax2.set_ylabel("Confidence")
    ax2.set_title("Modality Confidence")
    
    # 3. 全息权重
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(["Text Holo", "Video Holo"], [
        model_outputs["holographic_weights"][0].item(),
        model_outputs["holographic_weights"][1].item()
    ])
    ax3.set_ylabel("Weight")
    ax3.set_title("Holographic Weights")
    
    # 4. 最终融合权重
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.bar(["Text", "Video"], [
        model_outputs["fusion_weights"][0].item(),
        model_outputs["fusion_weights"][1].item()
    ])
    ax4.set_ylabel("Weight")
    ax4.set_title("Final Fusion Weights")
    
    # 5. PRM评分
    ax5 = fig.add_subplot(gs[2, :], projection='polar')
    # ... (使用上面的PRM可视化代码)
    
    return fig
```

### 12.3 实时监控仪表板

**使用Streamlit构建交互式仪表板**:

```python
import streamlit as st

def monitoring_dashboard():
    st.title("eduPDF Training Monitor")
    
    # 侧边栏：选择实验
    experiment = st.sidebar.selectbox(
        "Select Experiment",
        ["Phase1_Trimodal", "Phase2_Mamba", "Phase3_PRM"]
    )
    
    # 主面板：指标展示
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Accuracy", "87.5%", "+12.5%")
    
    with col2:
        st.metric("Inference Speed", "18ms", "-82ms")
    
    with col3:
        st.metric("Model Size", "450MB", "-750MB")
    
    # 损失曲线
    st.subheader("Training Loss")
    loss_data = load_loss_data(experiment)
    st.line_chart(loss_data)
    
    # 专家使用率
    st.subheader("Expert Usage")
    expert_usage = load_expert_usage(experiment)
    st.bar_chart(expert_usage)
    
    # 样本分析
    st.subheader("Sample Analysis")
    sample_id = st.selectbox("Select Sample", range(100))
    sample = load_sample(sample_id)
    
    # 显示输入
    st.text(f"Text: {sample['text']}")
    st.text(f"Video Desc: {sample['video_description']}")
    
    # 显示预测
    prediction = model.predict(sample)
    st.text(f"Prediction: {prediction['prediction']}")
    st.text(f"Confidence: {prediction['confidence']:.2f}")
    
    # 显示可视化
    fig = visualize_fusion_process(sample, prediction)
    st.pyplot(fig)

if __name__ == "__main__":
    monitoring_dashboard()
```

**启动仪表板**:
```bash
streamlit run monitoring_dashboard.py
```


## 13. 部署方案

### 13.1 模型导出

**导出为ONNX格式**:
```python
def export_to_onnx(model, output_path):
    """导出模型为ONNX格式"""
    model.eval()
    
    # 准备dummy输入
    dummy_text = torch.randint(0, 30522, (1, 128))
    dummy_mask = torch.ones(1, 128)
    dummy_segment = torch.zeros(1, 128)
    dummy_video = torch.randn(1, 16, 3, 224, 224)
    
    # 导出
    torch.onnx.export(
        model,
        (dummy_text, dummy_mask, dummy_segment, dummy_video),
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['text', 'mask', 'segment', 'video'],
        output_names=['logits', 'fusion_weights', 'prm_scores'],
        dynamic_axes={
            'text': {0: 'batch_size'},
            'mask': {0: 'batch_size'},
            'segment': {0: 'batch_size'},
            'video': {0: 'batch_size', 1: 'num_frames'}
        }
    )
```

**使用ONNX Runtime推理**:
```python
import onnxruntime as ort

class ONNXPredictor:
    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path)
    
    def predict(self, text, mask, segment, video):
        """使用ONNX模型推理"""
        inputs = {
            'text': text.numpy(),
            'mask': mask.numpy(),
            'segment': segment.numpy(),
            'video': video.numpy()
        }
        
        outputs = self.session.run(None, inputs)
        return {
            'logits': outputs[0],
            'fusion_weights': outputs[1],
            'prm_scores': outputs[2]
        }
```

### 13.2 服务化部署

**使用FastAPI构建REST API**:

```python
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="eduPDF API")

# 加载模型
model = load_model("checkpoints/best_model.pt")

class PredictionRequest(BaseModel):
    text: str
    video_description: str
    video_path: str = None

class PredictionResponse(BaseModel):
    prediction: str
    prediction_id: int
    confidence: float
    fusion_weights: dict
    prm_scores: dict
    routing_info: dict

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """预测接口"""
    sample = {
        "text": request.text,
        "video_description": request.video_description,
        "video_path": request.video_path
    }
    
    result = model.predict(sample, return_details=True)
    return result

@app.post("/predict_video")
async def predict_video(
    text: str,
    video_description: str,
    video: UploadFile = File(...)
):
    """上传视频预测接口"""
    # 保存视频
    video_path = f"temp/{video.filename}"
    with open(video_path, "wb") as f:
        f.write(await video.read())
    
    # 预测
    sample = {
        "text": text,
        "video_description": video_description,
        "video_path": video_path
    }
    
    result = model.predict(sample, return_details=True)
    
    # 清理临时文件
    os.remove(video_path)
    
    return result

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Docker部署**:

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码和模型
COPY src/ ./src/
COPY checkpoints/ ./checkpoints/
COPY configs/ ./configs/

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# 构建镜像
docker build -t edupdf-api:latest .

# 运行容器
docker run -d -p 8000:8000 --gpus all edupdf-api:latest
```

### 13.3 批量推理

**批量处理脚本**:
```python
def batch_inference(model, input_file, output_file, batch_size=32):
    """批量推理"""
    # 加载数据
    with open(input_file, 'r') as f:
        samples = [json.loads(line) for line in f]
    
    results = []
    
    # 分批处理
    for i in tqdm(range(0, len(samples), batch_size)):
        batch = samples[i:i+batch_size]
        
        # 批量预测
        batch_results = model.predict_batch(batch)
        results.extend(batch_results)
    
    # 保存结果
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    return results
```

**使用示例**:
```bash
python batch_inference.py \
    --model checkpoints/best_model.pt \
    --input data/test.jsonl \
    --output results/predictions.jsonl \
    --batch_size 64
```

### 13.4 边缘设备部署

**模型量化和优化**:
```python
def optimize_for_mobile(model, output_path):
    """优化模型用于移动端部署"""
    # 1. 量化
    model_int8 = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    
    # 2. 转换为TorchScript
    scripted_model = torch.jit.script(model_int8)
    
    # 3. 优化
    optimized_model = torch.jit.optimize_for_inference(scripted_model)
    
    # 4. 保存
    optimized_model.save(output_path)
```

**Android部署（使用PyTorch Mobile）**:
```java
// Android代码
import org.pytorch.Module;
import org.pytorch.Tensor;

public class EduPDFPredictor {
    private Module model;
    
    public EduPDFPredictor(String modelPath) {
        model = Module.load(modelPath);
    }
    
    public Prediction predict(String text, String videoDesc, float[] videoFrames) {
        // 准备输入
        Tensor textTensor = Tensor.fromBlob(text.getBytes(), new long[]{1, 128});
        Tensor videoTensor = Tensor.fromBlob(videoFrames, new long[]{1, 16, 3, 224, 224});
        
        // 推理
        IValue[] outputs = model.forward(IValue.from(textTensor), IValue.from(videoTensor)).toTuple();
        
        // 解析输出
        float[] logits = outputs[0].toTensor().getDataAsFloatArray();
        int prediction = argmax(logits);
        
        return new Prediction(prediction, logits[prediction]);
    }
}
```


## 14. 附录

### 14.1 术语表

| 术语 | 英文 | 解释 |
|------|------|------|
| PDF框架 | Probabilistic Distribution Fusion | 概率分布融合框架，通过动态权重融合多模态预测 |
| 全息权重 | Holographic Weights | 通过对数比值计算的互补权重，反映模态间的互补性 |
| MoE | Mixture-of-Experts | 混合专家模型，由多个专家网络和路由器组成 |
| 稀疏激活 | Sparse Activation | 每次只激活部分专家，降低计算成本 |
| Top-K路由 | Top-K Routing | 选择得分最高的K个专家进行激活 |
| PRM | Process Reward Model | 过程奖励模型，对中间步骤提供监督信号 |
| Mamba | Mamba | 基于状态空间模型的高效序列建模方法 |
| SSM | State Space Model | 状态空间模型，用于序列建模 |
| EARS | Easy Approach to Requirements Syntax | 需求规范语法，使用WHEN-THEN格式 |

### 14.2 参考文献

1. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**
   - Gu, A., & Dao, T. (2023)
   - arXiv:2312.00752
   - https://arxiv.org/abs/2312.00752

2. **GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding**
   - Lepikhin, D., et al. (2020)
   - ICLR 2021
   - https://arxiv.org/abs/2006.16668

3. **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity**
   - Fedus, W., et al. (2021)
   - JMLR 2022
   - https://arxiv.org/abs/2101.03961

4. **Let's Verify Step by Step (Process Reward Models)**
   - Lightman, H., et al. (2023)
   - OpenAI
   - https://arxiv.org/abs/2305.20050

5. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**
   - Devlin, J., et al. (2018)
   - NAACL 2019
   - https://arxiv.org/abs/1810.04805

6. **Deep Residual Learning for Image Recognition (ResNet)**
   - He, K., et al. (2015)
   - CVPR 2016
   - https://arxiv.org/abs/1512.03385

### 14.3 数据集统计

**原始数据集（new_single.jsonl）**:
```
总样本: 6318
类别分布:
  0. 教师讲授: 1577 (25.0%)
  1. 学生讨论: 807 (12.8%)
  2. 教师板书: 623 (9.9%)
  3. 教师提问: 584 (9.2%)
  4. 学生发言: 423 (6.7%)
  5. 课堂沉寂: 398 (6.3%)
  6. 教师反馈: 161 (2.5%)
  7. 教师巡视: 161 (2.5%)
  8. 教师指令: 146 (2.3%)
  9. 技术操作: 38 (0.6%)

模态完整性:
  text非空: 4520 (71.5%)
  text为空: 1798 (28.5%)
  video_description非空: 6313 (99.9%)
  video_description为空: 5 (0.1%)

数据划分:
  训练集: 5055 (80%)
  验证集: 632 (10%)
  测试集: 631 (10%)
```

**增强后数据集**:
```
总样本: 6918
新增样本: 600
  技术操作: +162 (38 → 200)
  教师反馈: +239 (161 → 400)
  其他类别: +199 (均衡增强)

类别分布（增强后）:
  0. 教师讲授: 1677 (24.2%)
  1. 学生讨论: 857 (12.4%)
  2. 教师板书: 673 (9.7%)
  3. 教师提问: 634 (9.2%)
  4. 学生发言: 473 (6.8%)
  5. 课堂沉寂: 448 (6.5%)
  6. 教师反馈: 400 (5.8%)
  7. 教师巡视: 211 (3.1%)
  8. 教师指令: 196 (2.8%)
  9. 技术操作: 200 (2.9%)

不平衡比例: 8.4:1 (原41:1)
```

### 14.4 超参数配置

**Phase 1: 三模态融合 + LLM增强**
```yaml
learning_rate: 2e-5
batch_size: 32
num_epochs: 10
warmup_steps: 500
weight_decay: 0.01
dropout: 0.3
max_seq_length: 128
text_ratio: 0.6
video_desc_ratio: 0.4
```

**Phase 2: Mamba + Sparse MoE + 层次化融合**
```yaml
learning_rate: 1e-5
batch_size: 24  # 降低以适应Mamba内存需求
num_epochs: 15
warmup_steps: 1000
gradient_clip: 1.0

mamba:
  d_model: 512
  d_state: 16
  d_conv: 4
  expand: 2
  n_layers: 4

sparse_moe:
  n_experts: 8
  top_k: 2
  load_balance_weight: 0.01

hierarchical_fusion:
  n_modality_experts: 3
  n_class_experts: 10
  adjustment_weight: 0.3
```

**Phase 3: PRM过程监督**
```yaml
learning_rate: 5e-6  # 更小的学习率
batch_size: 16
num_epochs: 10
warmup_steps: 200

prm:
  prm_loss_weight: 0.1
  adjustment_weight: 0.2
  supervision_samples: 500
  llm_model: "gpt-4o-mini"
  llm_temperature: 0.3
```

### 14.5 实验结果预期

**Phase 1预期结果**:
```
基线准确率: 75.0%
Phase 1准确率: 83.0-88.0% (+8-13%)

分类别准确率:
  教师讲授: 92% → 95%
  学生讨论: 85% → 88%
  技术操作: 30% → 50% (LLM增强效果)
  空文本样本: 40% → 65% (三模态融合效果)
```

**Phase 2预期结果**:
```
Phase 1准确率: 83.0-88.0%
Phase 2准确率: 88-93% (+13-18%累计)

性能指标:
  推理速度: 100ms → 20ms (5倍提升)
  内存占用: 8GB → 3GB (训练), 4GB → 2GB (推理)
  参数量: 350M → 175M (50%减少)
```

**Phase 3预期结果**:
```
Phase 2准确率: 88-93%
Phase 3准确率: 89-94% (+14-19%累计，较基线75%)

可解释性:
  PRM五步评分输出: ✓
  融合权重可视化: ✓
  路由决策分析: ✓
```

### 14.6 常见问题

**Q1: 为什么使用Mamba而不是Transformer？**
A: Mamba在长序列建模上比Transformer更高效（线性复杂度vs二次复杂度），且内存占用更小。对于100帧视频，Transformer需要16GB内存，而Mamba只需6GB。

**Q2: Sparse MoE的Top-2激活是否太少？**
A: Top-2是经过实验验证的最优配置。Top-1会导致专家过度专业化，Top-3会增加计算成本但收益有限。Top-2在效率和性能间取得最佳平衡。

**Q3: 为什么保留PDF的全息权重计算？**
A: 全息权重是PDF框架的核心创新，数学上优雅且效果好。我们将其作为特征输入到MoE中，而非直接输出，既保留了核心思想，又增加了可学习性。

**Q4: LLM生成的数据质量如何保证？**
A: 通过三重保障：1) 相似度过滤（0.7-0.95），2) 人工审核（20%抽样），3) 模型验证（生成数据在验证集上的表现）。

**Q5: PRM的监督信号从哪里来？**
A: 使用GPT-4o-mini生成500个样本的监督信号，成本约$0.04。这些信号用于训练PRM，之后PRM可以自主评分。

**Q6: 如何处理视频文件损坏的情况？**
A: 系统会捕获视频加载错误，自动回退到纯文本模式（仅使用text和video_description）。路由器会自动将这些样本路由到文本专家。

**Q7: 模型是否支持实时推理？**
A: 优化后的模型推理速度为15-20ms/样本，可以支持实时应用（30fps视频需要33ms/帧）。如需更快速度，可以使用ONNX Runtime或TensorRT加速。

**Q8: 如何添加新的课堂行为类别？**
A: 使用增量学习：1) 添加新的类别专家，2) 准备新类别数据，3) 冻结旧模块，仅训练新专家，4) 微调融合层。整个过程约需2-3小时。

