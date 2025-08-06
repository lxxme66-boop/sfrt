# 本地大模型快速开始指南

## 概述

本指南将帮助您使用本地大模型运行 RAFT，无需任何 API 密钥。支持 Qwen、Llama 等开源模型。

## 环境准备

### 1. 安装依赖

```bash
# 安装 vLLM（用于高效推理）
pip install vllm

# 安装其他依赖
pip install -r requirements.txt
```

### 2. 验证安装

运行测试脚本检查环境：

```bash
python test_local_llm.py
```

## 快速开始

### 方式一：使用预设脚本

```bash
# Linux/Mac
bash raft_local.sh

# Windows
raft_local_run.bat
```

### 方式二：自定义运行

```bash
python raft_local_llm.py \
  --datapath data/你的文档.pdf \
  --output 输出目录 \
  --model-name qwq_32 \
  --questions 5
```

## 支持的模型

| 模型名称 | 描述 | 参数量 | 推荐用途 |
|---------|------|--------|---------|
| `qwq_32` | QwQ 32B（默认） | 32B | 通用推理，中文支持好 |
| `qw2_72` | Qwen2 72B | 72B | 高质量生成 |
| `qw2.5_32` | Qwen2.5 32B | 32B | 平衡性能和质量 |
| `qw2.5_72` | Qwen2.5 72B | 72B | 最高质量 |
| `llama3.1_70` | Llama 3.1 70B | 70B | 英文场景 |

## 常用参数说明

### 基础参数
- `--datapath`: 输入文档路径（支持 PDF、TXT、JSON）
- `--output`: 输出目录
- `--questions`: 每个文本块生成的问题数量（默认：5）
- `--chunk_size`: 文本块大小（默认：512 tokens）
- `--distractors`: 干扰文档数量（默认：3）

### 模型参数
- `--model-name`: 模型名称（见上表）
- `--model-path`: 自定义模型路径
- `--embedding-model`: 嵌入模型（默认：BAAI/bge-large-zh-v1.5）

### 推理参数
- `--temperature`: 生成温度（0-1，默认：0.6）
- `--top-p`: Top-p 采样（默认：0.95）
- `--max-tokens`: 最大生成长度（默认：4096）

### GPU 参数
- `--gpu-memory-utilization`: GPU 内存使用率（0-1，默认：0.95）
- `--tensor-parallel-size`: 张量并行数（多 GPU 时使用）

## 使用示例

### 1. 处理单个 PDF 文件

```bash
python raft_local_llm.py \
  --datapath data/论文.pdf \
  --output outputs/论文_qa \
  --model-name qw2.5_32 \
  --questions 3 \
  --chunk_size 1024
```

### 2. 批量处理文件夹

```bash
python raft_local_llm.py \
  --datapath data/ \
  --output outputs/批量处理 \
  --model-name qwq_32 \
  --qa-threshold 100  # 生成 100 个 QA 对后停止
```

### 3. 使用自定义模型

```bash
python raft_local_llm.py \
  --datapath data/文档.txt \
  --output outputs/自定义模型 \
  --model-path /path/to/your/model \
  --embedding-model BAAI/bge-base-zh-v1.5
```

### 4. 多 GPU 推理

```bash
python raft_local_llm.py \
  --datapath data/大文档.pdf \
  --output outputs/多gpu \
  --model-name qw2.5_72 \
  --tensor-parallel-size 4  # 使用 4 个 GPU
```

### 5. 半导体领域文档处理

```bash
python raft_local_llm.py \
  --datapath data/半导体论文.pdf \
  --output outputs/半导体_qa \
  --model-name qwq_32 \
  --system-prompt-key deepseek-v2 \
  --questions 5 \
  --temperature 0.3  # 降低温度以提高准确性
```

## 输出格式

生成的数据集包含以下字段：
- `id`: 唯一标识符
- `question`: 生成的问题
- `context`: 包含答案的上下文（含干扰文档）
- `cot_answer`: 思维链答案
- `instruction`: 格式化的指令

## 常见问题

### Q: 模型加载失败？
A: 检查模型路径是否正确，确保有足够的 GPU 内存。

### Q: 生成质量不高？
A: 尝试：
- 使用更大的模型（如 qw2.5_72）
- 降低 temperature（如 0.3）
- 调整 chunk_size

### Q: GPU 内存不足？
A: 尝试：
- 减小 `--gpu-memory-utilization`（如 0.8）
- 使用更小的模型
- 增加 `--tensor-parallel-size`（多 GPU）

### Q: 想使用其他开源模型？
A: 使用 `--model-path` 指定模型路径，确保模型兼容 vLLM。

## 性能优化建议

1. **批处理**：使用文件夹路径批量处理多个文档
2. **并行处理**：多 GPU 时增加 tensor-parallel-size
3. **合理分块**：根据文档类型调整 chunk_size
4. **缓存利用**：vLLM 会自动缓存 KV，处理相似文档时更快

## 与 API 版本对比

| 特性 | 本地模型版 | API 版 |
|-----|----------|--------|
| 成本 | 仅 GPU 成本 | 按 token 付费 |
| 隐私 | 完全本地 | 数据发送到云端 |
| 速度 | 取决于硬件 | 受网络影响 |
| 模型选择 | 开源模型 | 商业模型 |
| 配置复杂度 | 需要 GPU | 仅需 API Key |

## 下一步

1. 查看生成的数据集质量
2. 使用 `enhanced_quality_evaluation.py` 评估质量
3. 根据需要调整参数重新生成
4. 使用生成的数据集进行模型微调

祝您使用愉快！如有问题，请查看主 README 或提交 Issue。