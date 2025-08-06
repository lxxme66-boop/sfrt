# 🚀 RAFT Enhanced v4 - Quick Start Guide

**目标：生成高质量问答对用于半导体显示领域的RAG模型训练**

## 📋 前置要求

### 1. 环境准备
```bash
# 安装依赖
pip install -r requirements.txt
pip install matplotlib seaborn pandas

# 设置环境变量
export CHUNK_NUM=4
export CHUNK_NUM_MIN=2  
export NUM_distract=3
export PROMPT_KEY="deepseek-v2"
export COMPLETION_MODEL="deepseek-r1-250120"
export COMPLETION_OPENAI_BASE_URL="your_api_base_url"
export COMPLETION_OPENAI_API_KEY="your_api_key"
```

### 2. 数据准备
将你的文档放入 `data/` 目录：
```
data/
├── document1.md
├── document2.md
└── document3.md
```

## ⚡ 三种运行模式

### 🔥 模式1：完整增强管道（推荐）
**适用场景**：生产环境，追求最高质量

```bash
# 一键运行所有模式
bash utils/syndata_pipeline_v4.sh

# 或手动运行完整增强版
python -m utils.syndata_pipeline_v4 \
  --data_dir "data" \
  --syndatas_path "outputs_enhanced/syndatas/syndatas_enhanced.json" \
  --start_idx 0 \
  --end_idx 10
```

**特点**：
- ✅ 文档质量预筛选
- ✅ 增强版问题生成
- ✅ 问题质量验证
- ⏱️ 速度较慢，质量最高

### ⚡ 模式2：快速模式
**适用场景**：开发测试，平衡速度和质量

```bash
python -m utils.syndata_pipeline_v4 \
  --data_dir "data" \
  --skip_document_filter \
  --syndatas_path "outputs_enhanced/syndatas/syndatas_fast.json" \
  --start_idx 0 \
  --end_idx 10
```

**特点**：
- ❌ 跳过文档筛选
- ✅ 增强版问题生成  
- ✅ 问题质量验证
- ⚡ 速度中等，质量较高

### 🔄 模式3：兼容模式
**适用场景**：与v3版本对比，快速迭代

```bash
python -m utils.syndata_pipeline_v4 \
  --data_dir "data" \
  --skip_document_filter \
  --skip_question_validation \
  --syndatas_path "outputs_enhanced/syndatas/syndatas_compat.json" \
  --start_idx 0 \
  --end_idx 10
```

**特点**：
- ❌ 跳过文档筛选
- ❌ 跳过问题验证
- ⚡ 速度最快，标准质量

## 📊 质量评估

### 单个数据集评估
```bash
python enhanced_quality_evaluation.py \
  --input outputs_enhanced/syndatas/syndatas_enhanced.json
```

### 多数据集对比
```bash
python enhanced_quality_evaluation.py --compare \
  outputs_enhanced/syndatas/syndatas_enhanced.json \
  outputs_enhanced/syndatas/syndatas_fast.json \
  outputs_enhanced/syndatas/syndatas_compat.json
```

### 大数据集采样评估
```bash
python enhanced_quality_evaluation.py \
  --input your_large_dataset.json \
  --sample_size 100
```

## 🎯 关键参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--start_idx` | 文档起始索引 | 0 |
| `--end_idx` | 文档结束索引 | 根据文档数量设定 |
| `--max_workers` | 并发数量 | 4-8 |
| `--skip_document_filter` | 跳过文档筛选 | 开发时使用 |
| `--skip_question_validation` | 跳过问题验证 | 仅兼容模式使用 |

## 📈 质量指标解读

### 评估维度
1. **思维链逻辑质量** (35%权重)：推理步骤完整性和连贯性
2. **技术准确度** (30%权重)：材料特性和制程参数准确性  
3. **领域深度** (20%权重)：缺陷机理和技术趋势覆盖
4. **应用价值** (15%权重)：工程可行性和成本优化

### 质量等级
- **High**: 高质量问答对，可直接用于训练
- **Medium**: 中等质量，可能需要少量人工审核
- **Low**: 低质量，建议重新生成或丢弃

## 🔧 常见问题排查

### Q1: 输出文件为空
**解决方案**：检查 `start_idx` 和 `end_idx` 范围是否正确
```bash
# 查看文档数量
ls -la data/*.md | wc -l

# 相应调整索引范围
--start_idx 0 --end_idx 实际文档数量
```

### Q2: API调用失败
**解决方案**：确认环境变量设置正确
```bash
echo $COMPLETION_OPENAI_BASE_URL
echo $COMPLETION_OPENAI_API_KEY
```

### Q3: 质量评分偏低
**解决方案**：
1. 检查原始文档质量
2. 使用完整增强管道
3. 调整prompt模板设置

### Q4: 内存不足
**解决方案**：
1. 减少 `--max_workers` 数量
2. 分批处理文档（调整 `end_idx`）
3. 增加系统内存

## 📁 输出文件说明

```
outputs_enhanced/
├── chunks/                    # 文档分块结果
├── chunk4/                   # 组合块结果  
├── topics/                   # 主题提取结果
├── questions/                # 原始问题
├── validated_questions/      # 验证后问题
├── answers/                  # 答案生成结果
└── syndatas/                 # 最终训练数据
    ├── syndatas_enhanced.json    # 完整增强版
    ├── syndatas_fast.json        # 快速模式
    └── syndatas_compat.json      # 兼容模式
```

## 🎨 最佳实践

### 生产环境
1. 使用**完整增强管道**
2. 设置合适的文档范围
3. 定期监控质量指标
4. 保存评估报告备查

### 开发测试
1. 使用**快速模式**
2. 小范围测试（`end_idx=5`）
3. 对比不同模式效果
4. 逐步扩大处理范围

### 大规模处理
1. 分批处理文档
2. 增加并发数量
3. 使用采样评估
4. 监控系统资源使用

## 🔗 下一步

1. **模型训练**：使用生成的数据训练RAG模型
2. **效果评估**：在下游任务上测试模型性能
3. **迭代优化**：根据效果调整参数和prompt
4. **规模扩展**：处理更大规模的文档集合

---

💡 **提示**：建议先用少量文档测试整个流程，确认效果后再进行大规模处理。

📞 **支持**：遇到问题可查看详细日志或检查 `evaluation_results/` 目录下的质量报告。