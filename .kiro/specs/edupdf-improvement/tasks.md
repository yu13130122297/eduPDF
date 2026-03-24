# 实施计划：eduPDF改进方案

## 概述

本实施计划将eduPDF多模态课堂行为识别系统的六大核心改进分解为可执行的编码任务。实施分为三个阶段：

- **Phase 1（Week 1-2）**: 数据基建 - 三模态融合 + LLM数据增强
- **Phase 2（Week 3-6）**: 架构创新 - Mamba编码器 + Sparse MoE + 层次化融合
- **Phase 3（Week 7-10）**: PRM过程监督 + 端到端优化

每个任务都引用了具体的需求条款，确保可追溯性。测试任务标记为可选（*），可根据时间安排跳过。

---

## Phase 1: 数据基建（Week 1-2）

### 1. 实现三模态文本编码器

- [ ] 1.1 创建TriModalTextEncoder类
  - 在`src/models/trimodal_encoder.py`中实现`TriModalTextEncoder(nn.Module)`
  - 支持text和video_description的智能拼接
  - 实现空文本自动回退机制（text=""时使用video_description）
  - 实现双段拼接（text非空时拼接text + [SEP] + video_description）
  - 动态分配长度：text占60%，video_description占40%
  - 正确设置segment_ids（text段为0，video_description段为1）
  - _需求: FR-1.1, FR-1.2, FR-1.3_

- [ ] 1.2 编写三模态编码器单元测试
  - 在`tests/unit/test_trimodal_encoder.py`中实现
  - 测试空文本回退机制
  - 测试双段拼接和segment_ids正确性
  - 测试长度分配逻辑
  - _需求: FR-1.1, FR-1.2, FR-1.3_

- [ ] 1.3 编写三模态编码器属性测试
  - 在`tests/property/test_properties_fr1.py`中实现
  - **属性1**: 三模态输入处理 - 验证任意模态组合都能正确处理
  - **属性2**: 空文本自动回退 - 验证空文本时自动使用video_description
  - **属性3**: 双段文本拼接 - 验证segment_ids正确标记
  - 使用Hypothesis生成随机输入，最小100次迭代
  - _需求: FR-1.1, FR-1.2, FR-1.3_

### 2. 实现LLM数据增强模块

- [ ] 2.1 创建LLMAugmentor类
  - 在`src/data/augmentation.py`中实现`LLMAugmentor`类
  - 集成DeepSeek-V3 API（使用OpenAI SDK）
  - 实现few-shot prompt构建（使用3-5个种子样本）
  - 实现质量过滤机制（BERT相似度：0.7-0.95）
  - 实现video_description生成功能
  - _需求: FR-2.1, FR-2.2, FR-2.3_

- [ ] 2.2 生成小样本类别训练数据
  - 为"技术操作"类别生成162条数据（38→200）
  - 为"教师反馈"类别生成239条数据（161→400）
  - 为其他类别生成199条均衡数据
  - 人工审核随机抽取20%样本，通过率需>80%
  - 保存生成数据到`data/augmented/`目录
  - _需求: FR-2.1, FR-2.2_

- [ ] 2.3 编写数据增强单元测试
  - 在`tests/unit/test_augmentation.py`中实现
  - 测试prompt构建逻辑
  - 测试质量过滤机制
  - 测试API调用和重试机制
  - _需求: FR-2.1, FR-2.2, FR-2.3_

### 3. 更新数据集加载器

- [ ] 3.1 修改Dataset类支持增强数据
  - 修改`src/data/dataset.py`中的数据加载逻辑
  - 合并原始数据（new_single.jsonl）和增强数据
  - 添加source字段标记数据来源（original/llm_augmented）
  - 重新划分训练/验证/测试集（80%/10%/10%）
  - 生成数据统计报告（类别分布、模态完整性）
  - _需求: FR-2.1_

- [ ] 3.2 验证数据集更新
  - 运行数据统计脚本，确认类别分布改善
  - 验证不平衡比例从41:1降至8.4:1
  - 确认总样本数从6318增至6918
  - _需求: FR-2.1_

### 4. Checkpoint - Phase 1验证

- [ ] 4.1 集成三模态编码器到现有系统
  - 修改`src/models/bert.py`或创建新的文本编码模块
  - 更新`src/models/latefusion_video.py`以使用TriModalTextEncoder
  - 确保接口兼容性，不破坏现有功能
  - _需求: FR-1.1_

- [ ] 4.2 训练基线模型（三模态+增强数据）
  - 使用Phase 1配置（configs/phase1.yaml）训练模型
  - 训练10个epoch，batch_size=32，learning_rate=2e-5
  - 记录训练损失和验证准确率
  - _需求: FR-1.1, FR-2.1_

- [ ] 4.3 评估Phase 1改进效果
  - 在测试集上评估准确率
  - 验证准确率提升8-13%（从75%到83-88%）
  - 特别关注空文本样本准确率（40%→65%）
  - 特别关注小样本类别准确率（技术操作：30%→50%）
  - 生成评估报告并保存到`results/phase1/`
  - _需求: FR-1.1, FR-2.1_

- [ ] 4.4 Phase 1里程碑检查
  - 确保所有测试通过，准确率达标
  - 询问用户是否有问题或需要调整
  - 如有问题，返回修改相关模块

---

## Phase 2: 架构创新（Week 3-6）

### 5. 实现Mamba视频编码器（Week 3-4）

- [ ] 5.1 创建MambaVideoEncoder类
  - 在`src/models/mamba_encoder.py`中实现`MambaVideoEncoder(nn.Module)`
  - 保留ResNet50帧特征提取器（可选冻结）
  - 实现特征投影层（2048→512维）
  - 实现多层Mamba编码（4层，d_model=512, d_state=16, d_conv=4, expand=2）
  - 实现残差连接和LayerNorm
  - 实现时序池化（支持last/mean/max三种模式）
  - 支持16-100帧动态输入
  - _需求: FR-3.1, FR-3.2_

- [ ] 5.2 编写Mamba编码器单元测试
  - 在`tests/unit/test_mamba_encoder.py`中实现
  - 测试16帧、50帧、100帧输入
  - 测试输出维度正确性
  - 测试不同时序池化模式
  - _需求: FR-3.1_

- [ ] 5.3 编写Mamba编码器属性测试
  - 在`tests/property/test_properties_fr3.py`中实现
  - **属性4**: 长视频序列支持 - 验证16-100帧输入都能成功处理
  - 验证输出维度固定，无NaN/Inf值
  - 使用Hypothesis生成随机帧数，最小100次迭代
  - _需求: FR-3.1, NFR-2.3_

- [ ] 5.4 集成Mamba编码器到现有系统
  - 修改`src/models/video.py`，添加Mamba作为可选编码器
  - 更新`src/models/latefusion_video.py`以支持Mamba编码器
  - 添加配置选项切换LSTM/Mamba（video_encoder_type: "lstm"/"mamba"）
  - 保持接口兼容性
  - _需求: FR-3.1_

- [ ] 5.5 优化Mamba编码器性能
  - 实现梯度检查点（gradient checkpointing）减少内存占用
  - 实现推理时的递归模式（inference_mode="recurrent"）
  - 测试推理速度，验证相比LSTM提升5倍
  - 测试内存占用，确保训练≤16GB，推理≤4GB
  - _需求: FR-3.2, NFR-2.1, NFR-2.3_

- [ ] 5.6 训练和评估Mamba模型
  - 使用Phase 2配置训练模型（batch_size=24，考虑内存）
  - 训练15个epoch，记录训练曲线
  - 在测试集上评估准确率和推理速度
  - 生成性能对比报告（LSTM vs Mamba）
  - _需求: FR-3.1, FR-3.2_

### 6. 实现Sparse MoE ConfidNet（Week 5）

- [ ] 6.1 创建SparseMoEConfidNet类
  - 在`src/models/sparse_moe.py`中实现`SparseMoEConfidNet(nn.Module)`
  - 实现8个专家网络（共享参数结构）
  - 实现文本和视频路由器（独立路由）
  - 实现Top-2稀疏激活机制
  - 实现负载均衡损失（load_balance_weight=0.01）
  - 标记专家类型：shared_1, shared_2, text_specialist, video_specialist, lecture, discussion, writing, silence
  - _需求: FR-4.1, FR-4.2, FR-4.3_

- [ ] 6.2 编写Sparse MoE单元测试
  - 在`tests/unit/test_sparse_moe.py`中实现
  - 测试Top-2激活机制
  - 测试权重归一化（和为1）
  - 测试负载均衡损失计算
  - _需求: FR-4.2_

- [ ] 6.3 编写Sparse MoE属性测试
  - 在`tests/property/test_properties_fr4.py`中实现
  - **属性5**: Top-2稀疏激活 - 验证恰好激活2个专家，权重和为1
  - **属性2（扩展）**: 空文本路由 - 验证空文本样本路由到视频相关专家
  - 使用Hypothesis生成随机输入，最小100次迭代
  - _需求: FR-4.2, FR-4.3_

- [ ] 6.4 替换原有ConfidNet
  - 修改`src/models/latefusion_video.py`，用SparseMoEConfidNet替换20个独立ConfidNet
  - 更新置信度计算逻辑
  - 添加专家使用率监控（在`src/training/callbacks.py`中）
  - 确保输出接口兼容（text_confidence, video_confidence）
  - _需求: FR-4.1_

- [ ] 6.5 训练和评估Sparse MoE模型
  - 分阶段训练：先共享专家（Epoch 1-5），再模态专家（Epoch 6-10），最后场景专家（Epoch 11-15）
  - 监控专家使用率，确保负载均衡（最小使用率/最大使用率 > 0.1）
  - 对比参数量：验证减少50%（从20个独立网络到8个共享专家）
  - 在测试集上评估准确率
  - _需求: FR-4.1, FR-4.2_

### 7. 实现层次化MoE融合层（Week 6）

- [ ] 7.1 创建HierarchicalMoEFusion类
  - 在`src/models/hierarchical_fusion.py`中实现`HierarchicalMoEFusion(nn.Module)`
  - 实现Layer 1模态路由MoE（3个专家：text_dominant, video_dominant, balanced）
  - 实现Layer 2类别路由MoE（10个专家，每类一个）
  - 保留PDF全息权重计算（作为Layer 1输入特征）
  - 实现两层输出聚合（Layer1输出 + 0.3 * Layer2调整）
  - 实现Softmax归一化确保权重和为1
  - _需求: FR-5.1, FR-5.2, FR-5.3_

- [ ] 7.2 编写层次化融合单元测试
  - 在`tests/unit/test_hierarchical_fusion.py`中实现
  - 测试全息权重计算
  - 测试两层路由逻辑
  - 测试权重归一化
  - _需求: FR-5.1, FR-5.3_

- [ ] 7.3 编写层次化融合属性测试
  - 在`tests/property/test_properties_fr5.py`中实现
  - **属性6**: 全息权重保留 - 验证全息权重被计算并作为Layer 1输入
  - **属性8**: 融合权重归一化 - 验证w_text + w_video = 1，且都在[0,1]范围
  - 使用Hypothesis生成随机输入，最小100次迭代
  - _需求: FR-5.3, NFR-3.1_

- [ ] 7.4 替换硬编码全息权重
  - 修改`src/models/latefusion_video.py`中的融合逻辑
  - 用HierarchicalMoEFusion替换硬编码的全息权重公式
  - 保留全息权重计算，但作为特征而非最终输出
  - 添加路由决策可视化功能
  - _需求: FR-5.1, FR-5.3_

- [ ] 7.5 训练和评估层次化融合模型
  - 联合训练Mamba + Sparse MoE + 层次化融合
  - 训练15个epoch，记录融合权重分布
  - 分析不同类别的模态偏好（通过Layer 2路由）
  - 在测试集上评估准确率
  - 验证准确率累计提升19-29%（从75%到94-104%）
  - _需求: FR-5.1, FR-5.2_

- [ ] 7.6 实现可解释性输出
  - 在`src/evaluation/visualization.py`中实现可视化工具
  - 实现融合权重分布可视化
  - 实现路由决策可视化（专家激活热力图）
  - 实现模态偏好分析（按类别统计）
  - 生成可解释性报告
  - _需求: NFR-3.1, NFR-3.2_

### 8. Checkpoint - Phase 2验证

- [ ] 8.1 端到端集成测试
  - 在`tests/integration/test_end_to_end.py`中实现完整流程测试
  - 测试从输入到输出的完整pipeline
  - 验证输出包含：prediction, confidence, fusion_weights, routing_info
  - 验证输出合理性（置信度在[0,1]，权重和为1）
  - _需求: NFR-3.1_

- [ ] 8.2 性能基准测试
  - 测试推理速度：验证≤20ms/样本
  - 测试内存占用：训练≤16GB，推理≤4GB
  - 测试模型大小：验证参数量减少50%
  - 生成性能基准报告
  - _需求: NFR-2.1, NFR-2.2, NFR-2.3_

- [ ] 8.3 Phase 2里程碑检查
  - 确保所有测试通过
  - 验证准确率累计提升19-29%
  - 验证推理速度提升5倍
  - 询问用户是否有问题或需要调整
  - 如有问题，返回修改相关模块

---

## Phase 3: PRM过程监督（Week 7-10）

### 9. 实现Process Reward Model（Week 7-8）

- [ ] 9.1 创建ProcessRewardModel类
  - 在`src/models/prm.py`中实现`ProcessRewardModel(nn.Module)`
  - 实现五个评分器：text_quality, video_quality, alignment, routing, fusion
  - 每个评分器输出0-1评分（使用Sigmoid激活）
  - 实现评分聚合器（5→n_classes调整量）
  - _需求: FR-6.1, FR-6.2, FR-6.3_

- [ ] 9.2 编写PRM单元测试
  - 在`tests/unit/test_prm.py`中实现
  - 测试五个评分器输出
  - 测试评分范围（0-1）
  - 测试聚合器输出维度
  - _需求: FR-6.1, FR-6.3_

- [ ] 9.3 编写PRM属性测试
  - 在`tests/property/test_properties_fr6.py`中实现
  - **属性7**: PRM五步评分输出 - 验证输出恰好5个评分，值域[0,1]
  - **属性9**: 可解释性输出完整性 - 验证输出包含融合权重、路由决策、PRM评分
  - 使用Hypothesis生成随机输入，最小100次迭代
  - _需求: FR-6.1, FR-6.3, NFR-3.3_

- [ ] 9.4 实现LLM监督信号生成
  - 在`src/data/prm_supervision.py`中实现`PRMSupervisor`类
  - 集成GPT-4o-mini API
  - 实现监督信号生成prompt（五步评分）
  - 实现JSON解析和错误处理
  - 为500个样本生成监督信号（成本约$0.04）
  - 缓存监督信号到`data/prm_supervision.json`
  - _需求: FR-6.2_

- [ ] 9.5 训练PRM模型
  - 冻结其他模块（Mamba, Sparse MoE, 层次化融合）
  - 仅训练PRM，使用LLM生成的监督信号
  - 训练10个epoch，batch_size=16
  - 损失函数：MSE(predicted_scores, llm_supervision_signals)
  - 记录训练曲线和评分分布
  - _需求: FR-6.2_

### 10. 联合训练和端到端优化（Week 9）

- [ ] 10.1 实现联合训练
  - 修改`src/training/trainer.py`，添加PRM损失
  - 总损失：classification_loss + λ_prm * prm_loss（λ_prm=0.1）
  - PRM与融合层联合训练，其他模块可选冻结
  - 训练10个epoch，learning_rate=5e-6
  - 监控各项损失和准确率
  - _需求: FR-6.2_

- [ ] 10.2 集成PRM调整到最终预测
  - 修改`src/models/latefusion_video.py`，添加PRM调整
  - 最终logits = 融合logits + 0.2 * PRM_adjustment
  - 确保PRM评分输出到结果中
  - _需求: FR-6.1, FR-6.3_

- [ ] 10.3 实现可解释性输出
  - 在预测结果中输出五步评分
  - 实现PRM评分可视化（雷达图）
  - 实现融合过程完整可视化（输入→置信度→全息权重→融合权重→PRM评分→预测）
  - 生成案例分析报告
  - _需求: NFR-3.3_

- [ ] 10.4 验证可解释性
  - 随机选择50个样本，生成可解释性报告
  - 人工审核可解释性质量
  - 验证PRM评分与预测结果的相关性
  - _需求: NFR-3.1, NFR-3.2, NFR-3.3_

### 11. 最终优化和实验（Week 10）

- [ ] 11.1 超参数调优
  - 调优学习率、批大小、损失权重
  - 使用网格搜索或贝叶斯优化
  - 记录最佳超参数配置
  - _需求: NFR-1.1_

- [ ] 11.2 消融实验
  - 实验1：仅三模态融合（Phase 1）
  - 实验2：三模态 + Mamba（Phase 1+2部分）
  - 实验3：三模态 + Mamba + Sparse MoE（Phase 1+2部分）
  - 实验4：三模态 + Mamba + Sparse MoE + 层次化融合（Phase 1+2完整）
  - 实验5：完整系统（Phase 1+2+3）
  - 验证每个模块的贡献
  - 生成消融实验报告
  - _需求: NFR-1.1_

- [ ] 11.3 最终性能评估
  - 在测试集上评估完整系统
  - 计算准确率、F1、精确率、召回率
  - 分类别评估（特别关注小样本类别）
  - 空文本样本单独评估
  - 验证准确率累计提升39-60%（从75%到114-135%）
  - _需求: NFR-1.1_

- [ ] 11.4 性能优化
  - 实现模型量化（INT8）
  - 导出ONNX模型
  - 测试优化后的推理速度和内存占用
  - 验证推理速度≤20ms，内存≤4GB
  - _需求: NFR-2.1, NFR-2.2, NFR-2.3_

- [ ] 11.5 鲁棒性测试
  - 测试缺失模态处理（text=""或video损坏）
  - 测试极端输入（超长文本、超长视频）
  - 测试错误处理（API失败、内存不足）
  - 验证系统不抛出异常，能优雅降级
  - _需求: NFR-4.1_

### 12. 文档和交付（Week 10）

- [ ] 12.1 生成实验报告
  - 汇总所有实验结果（Phase 1-3，消融实验）
  - 生成性能对比表格和图表
  - 分析每个改进的贡献
  - 准备论文实验数据
  - 保存到`results/final_report.md`

- [ ] 12.2 更新README和文档
  - 更新项目README，说明新功能
  - 更新安装说明（添加mamba-ssm依赖）
  - 更新使用示例（三模态输入、可解释性输出）
  - 更新配置说明（新增配置项）

- [ ] 12.3 代码清理和优化
  - 移除调试代码和注释
  - 统一代码风格（使用black格式化）
  - 添加必要的docstring
  - 运行linter检查（flake8, mypy）

- [ ] 12.4 最终检查
  - 运行所有单元测试和属性测试
  - 运行集成测试和性能测试
  - 验证所有功能正常工作
  - 确认所有需求都已实现

- [ ] 12.5 交付和演示
  - 准备演示脚本（展示三模态融合、可解释性、性能提升）
  - 生成演示视频或PPT
  - 准备答辩材料（针对硕士论文）
  - 询问用户是否满意，是否需要进一步调整

---

## 注意事项

1. **测试任务**: 标记为`*`的测试任务是可选的，可根据时间安排跳过以加快MVP开发
2. **需求追溯**: 每个任务都标注了对应的需求条款（FR-X.Y或NFR-X.Y），确保可追溯性
3. **Checkpoint任务**: 在关键阶段设置了checkpoint任务，确保增量验证和用户反馈
4. **渐进式实施**: 任务按依赖关系排序，每个任务都基于前面的任务构建
5. **错误处理**: 实施过程中如遇到问题，应返回修改相关模块，而非继续推进
6. **性能监控**: 在训练过程中持续监控性能指标，及时发现问题

## 实施建议

- **优先级**: 按Phase顺序实施，确保每个Phase完成后再进入下一个
- **测试驱动**: 建议先实现单元测试，再实现功能代码（TDD）
- **增量验证**: 每完成一个模块，立即集成测试，避免积累问题
- **文档同步**: 实施过程中同步更新文档，避免遗忘细节
- **版本控制**: 每个Phase完成后打tag，便于回滚和对比
