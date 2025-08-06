#!/bin/bash

# 本地模型数据合成管道运行脚本
# 使用vLLM加载本地大模型

echo "🚀 启动本地模型数据合成管道"
echo "📅 $(date)"
echo "=================================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES='0,1,2,3'  # 根据你的GPU情况调整
export CHUNK_NUM=4
export CHUNK_NUM_MIN=2
export NUM_distract=3
export PROMPT_KEY="deepseek-v2"

# 基础配置
DATA_DIR="data"
FILTERED_DATA_DIR="data_filtered_local"
OUTPUT_BASE="outputs_local"
INTERMEDIATE_DIR="${OUTPUT_BASE}/intermediate"

# 创建输出目录
mkdir -p ${OUTPUT_BASE}/{chunks,chunk4,topics,questions,validated_questions,answers,syndatas,stats}
mkdir -p ${INTERMEDIATE_DIR}

# 模型配置 - 根据你的本地模型修改
MODEL_NAME="qwq_32"  # 可选: qw2_72, qw2.5_32, qw2.5_72, llama3.1_70, qwq_32
MODEL_PATH="/mnt/workspace/models/Qwen/QwQ-32B/"  # 修改为你的模型路径

# 性能配置
BATCH_SIZE=16  # 根据GPU内存调整
MAX_WORKERS=2
GPU_MEMORY_UTILIZATION=0.9
TENSOR_PARALLEL_SIZE=4  # 根据GPU数量调整

echo "📁 配置信息:"
echo "   数据目录: ${DATA_DIR}"
echo "   输出目录: ${OUTPUT_BASE}"
echo "   模型名称: ${MODEL_NAME}"
echo "   模型路径: ${MODEL_PATH}"
echo "   批量大小: ${BATCH_SIZE}"
echo "   GPU数量: ${TENSOR_PARALLEL_SIZE}"
echo ""

# 运行小规模测试
echo "🧪 运行小规模测试（10个文档）"
python -m utils.syndata_pipeline_v5 \
  --data_dir "${DATA_DIR}" \
  --filtered_data_dir "${FILTERED_DATA_DIR}" \
  --chunks_path "${OUTPUT_BASE}/chunks/article_chunks_local.json" \
  --chunk4_path "${OUTPUT_BASE}/chunk4/article_chunk4_local.json" \
  --topics_path "${OUTPUT_BASE}/topics/article_topics_local.json" \
  --questions_path "${OUTPUT_BASE}/questions/article_questions_local.json" \
  --validated_questions_path "${OUTPUT_BASE}/validated_questions/article_questions_local_validated.json" \
  --answers_path "${OUTPUT_BASE}/answers/article_answers_local.json" \
  --syndatas_path "${OUTPUT_BASE}/syndatas/syndatas_local.json" \
  --judge_output_path "${INTERMEDIATE_DIR}/judge_output_local.jsonl" \
  --question_output_path "${INTERMEDIATE_DIR}/question_output_local.jsonl" \
  --question_li_output_path "${INTERMEDIATE_DIR}/question_li_output_local.jsonl" \
  --start_idx 0 \
  --end_idx 10 \
  --max_workers ${MAX_WORKERS} \
  --batch_size ${BATCH_SIZE} \
  --use_vllm \
  --model_name ${MODEL_NAME} \
  --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
  --tensor_parallel_size ${TENSOR_PARALLEL_SIZE}

echo ""
echo "✅ 测试完成！"
echo "📊 输出文件: ${OUTPUT_BASE}/syndatas/syndatas_local.json"
echo ""

# 如果测试成功，可以运行更大规模的处理
echo "💡 提示: 如果测试成功，可以修改 --end_idx 参数来处理更多文档"
echo "   例如: --end_idx 1000 处理前1000个文档"
echo ""
echo "📅 完成时间: $(date)"
echo "=================================================="