#!/bin/bash

# RAFT with Local LLM - Run Script
# 使用本地大模型运行RAFT，无需API密钥

# 默认参数
MODEL_NAME="qwq_32"  # 可选: qwq_32, qw2_72, qw2.5_32, qw2.5_72, llama3.1_70
EMBEDDING_MODEL="BAAI/bge-large-zh-v1.5"  # HuggingFace上的中文嵌入模型
GPU_MEMORY=0.95
TENSOR_PARALLEL=1  # 如果使用多GPU，可以增加这个值

# 示例1: 处理单个PDF文件
echo "示例1: 处理单个PDF文件"
python3 raft_local_llm.py \
  --datapath data/RAFT.pdf \
  --output outputs_local \
  --output-format hf \
  --distractors 3 \
  --p 1.0 \
  --doctype pdf \
  --chunk_size 512 \
  --questions 2 \
  --model-name $MODEL_NAME \
  --embedding-model $EMBEDDING_MODEL \
  --gpu-memory-utilization $GPU_MEMORY \
  --tensor-parallel-size $TENSOR_PARALLEL \
  --system-prompt-key deepseek-v2 \
  --temperature 0.6 \
  --top-p 0.95 \
  --max-tokens 4096

# 示例2: 处理文本文件
echo "示例2: 处理文本文件"
python3 raft_local_llm.py \
  --datapath data/97平板显示综述_朱昌昌_llm_correct.md \
  --output outputs_local_txt \
  --output-format completion \
  --distractors 3 \
  --p 1.0 \
  --doctype txt \
  --chunk_size 512 \
  --questions 2 \
  --model-name $MODEL_NAME \
  --embedding-model $EMBEDDING_MODEL \
  --system-prompt-key deepseek-v2

# 示例3: 批量处理文件夹中的所有文档
echo "示例3: 批量处理文件夹"
python3 raft_local_llm.py \
  --datapath data/ \
  --output outputs_local_batch \
  --output-format hf \
  --distractors 3 \
  --p 1.0 \
  --doctype txt \
  --chunk_size 512 \
  --questions 2 \
  --model-name $MODEL_NAME \
  --embedding-model $EMBEDDING_MODEL \
  --system-prompt-key deepseek-v2 \
  --qa-threshold 100  # 生成100个QA对后停止

# 示例4: 使用自定义模型路径
echo "示例4: 使用自定义模型路径"
# python3 raft_local_llm.py \
#   --datapath data/RAFT.pdf \
#   --output outputs_custom_model \
#   --model-path /path/to/your/local/model \
#   --embedding-model $EMBEDDING_MODEL \
#   --doctype pdf \
#   --chunk_size 512 \
#   --questions 2

# 示例5: 使用不同的系统提示
echo "示例5: 使用llama系统提示"
# python3 raft_local_llm.py \
#   --datapath data/RAFT.pdf \
#   --output outputs_llama_prompt \
#   --model-name llama3.1_70 \
#   --tensor-parallel-size 4 \
#   --system-prompt-key llama \
#   --doctype pdf