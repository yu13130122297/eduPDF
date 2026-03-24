# 混乱转录生成工作流程

## 📋 工作流程

### 步骤1：生成到临时文件
```bash
python3 generate_chaotic_transcriptions.py
```

**功能：**
- 生成所有学生讨论的混乱转录
- 保存到 `taolun.json`（临时文件）
- 自动跳过已处理的条目

**优点：**
- ✅ 支持断点续传（中断后继续）
- ✅ 方便手动编辑和审核
- ✅ 可删除不需要的条目重新生成

---

### 步骤2：编辑临时文件（可选）
打开 `taolun.json` 进行编辑：

```json
[
  {
    "id": "T12_0128_0134",
    "text": "不对那个距离是绝对值...",
    "video": "...",
    "label": "学生讨论"
  },
  {
    "id": "T12_0139_0144",
    "text": "你可以修改或删除这条",  ← 手动修改
    "video": "...",
    "label": "学生讨论"
  }
]
```

**可以进行的操作：**
1. **修改内容**：直接修改 `text` 字段
2. **删除条目**：删除不需要的条目，下次运行会重新生成
3. **添加条目**：手动添加特殊格式的转录

---

### 步骤3：再次生成（跳过已处理）
```bash
python3 generate_chaotic_transcriptions.py
```

**行为：**
- 跳过 `taolun.json` 中已存在的条目
- 只生成新的或被删除的条目
- 更新 `taolun.json`

---

### 步骤4：合并到最终文件

#### 方法1：在主脚本中选择合并
运行主脚本时，会提示：
```
是否现在合并到最终文件？(y/n，默认n):
```
输入 `y` 直接合并。

#### 方法2：使用独立合并脚本
```bash
python3 merge_to_final.py
```

**功能：**
- 读取原始数据文件
- 读取 `taolun.json`
- 将学生讨论的转录合并到原始数据
- 输出到 `new_single_with_transcription.jsonl`

---

## 🔄 常见使用场景

### 场景1：第一次运行
```bash
# 生成所有数据
python3 generate_chaotic_transcriptions.py

# 选择不合并（n）
# 现在 taolun.json 包含所有学生讨论

# 审核和编辑 taolun.json

# 合并到最终文件
python3 merge_to_final.py
```

### 场景2：中断后继续
```bash
# 假设生成到第100条时中断
python3 generate_chaotic_transcriptions.py

# 再次运行，会跳过前100条，继续生成
python3 generate_chaotic_transcriptions.py
```

### 场景3：部分不满意，重新生成
```bash
# 1. 打开 taolun.json
# 2. 删除不满意的条目（例如删除第50条）
# 3. 重新运行，会重新生成被删除的条目
python3 generate_chaotic_transcriptions.py
```

### 场景4：手动修改某些条目
```bash
# 1. 打开 taolun.json
# 2. 找到要修改的条目，修改 text 字段
# 3. 合并到最终文件
python3 merge_to_final.py
```

---

## 📁 文件说明

| 文件 | 说明 | 可编辑 |
|------|------|--------|
| `new_single.jsonl` | 原始输入数据 | ❌ |
| `taolun.json` | 临时文件（学生讨论转录） | ✅ |
| `new_single_with_transcription.jsonl` | 最终输出文件 | ❌ |

---

## 💡 最佳实践

### 1. 分批生成
不要一次性生成所有281条，可以：
```bash
# 生成50条，检查质量
# 编辑不满意的
# 继续生成
```

### 2. 质量控制
生成后使用测试脚本：
```bash
python3 test_chaotic_transcription.py
```

### 3. 版本管理
保存不同版本的 `taolun.json`：
```bash
cp taolun.json taolun_v1.json
cp taolun.json taolun_v2.json
```

### 4. 备份原始数据
始终保留原始文件：
```bash
cp new_single.jsonl new_single_backup.jsonl
```

---

## ⚠️ 注意事项

1. **不要直接编辑最终文件**
   - 编辑 `taolun.json` 而不是 `new_single_with_transcription.jsonl`
   - 最后通过合并脚本生成最终文件

2. **ID必须匹配**
   - `taolun.json` 中的 `id` 必须与原始数据匹配
   - 不要修改 `id` 字段

3. **JSON格式正确**
   - 编辑 `taolun.json` 后确保格式正确
   - 可以用JSON验证工具检查

4. **定期合并**
   - 不要只依赖 `taolun.json`
   - 定期合并到最终文件，避免数据丢失

---

## 🛠️ 故障排查

### 问题1：临时文件格式错误
**症状：** 读取 `taolun.json` 失败
**解决：** 删除 `taolun.json`，重新生成

### 问题2：合并后text为空
**症状：** 最终文件中学生讨论的text为空
**解决：**
- 检查 `taolun.json` 中是否有对应ID
- 检查ID是否匹配（注意大小写）

### 问题3：条目重复
**症状：** 最终文件中有重复条目
**解决：**
- 删除 `new_single_with_transcription.jsonl`
- 重新运行合并脚本

---

## 📊 数据统计

每次运行后，脚本会显示：
- 总记录数
- 学生讨论数
- 课程数量
- 已处理数量
- 跳过数量

示例输出：
```
共读取 6318 条记录
共 25 门课程
共 281 条学生讨论

发现已存在的临时文件: taolun.json
读取已处理的条目...
已处理 150 个条目（将跳过）
```
