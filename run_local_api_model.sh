#!/bin/bash

# 使用本地API服务的数据合成管道
# 适用于已经部署为API服务的本地模型

echo "🚀 使用本地API服务的数据合成管道"
echo "📅 $(date)"
echo "=================================================="

# 设置环境变量 - 指向你的本地模型API服务
export COMPLETION_OPENAI_API_KEY="your-api-key-or-dummy"  # 如果不需要可以设置为dummy
export COMPLETION_OPENAI_BASE_URL="http://localhost:8000/v1"  # 修改为你的API服务地址

# 基础配置
DATA_DIR="data"
FILTERED_DATA_DIR="data_filtered_api"
OUTPUT_BASE="outputs_api"
INTERMEDIATE_DIR="${OUTPUT_BASE}/intermediate"

# 创建输出目录
mkdir -p ${OUTPUT_BASE}/{chunks,chunk4,topics,questions,validated_questions,answers,syndatas,stats}
mkdir -p ${INTERMEDIATE_DIR}

echo "📁 配置信息:"
echo "   API地址: ${COMPLETION_OPENAI_BASE_URL}"
echo "   数据目录: ${DATA_DIR}"
echo "   输出目录: ${OUTPUT_BASE}"
echo ""

# 性能配置
BATCH_SIZE=8  # API模式下批量大小要适中
MAX_WORKERS=2

# 运行管道（不使用vLLM，使用API）
echo "🔥 运行数据合成管道（API模式）"
python -m utils.syndata_pipeline_v5 \
  --data_dir "${DATA_DIR}" \
  --filtered_data_dir "${FILTERED_DATA_DIR}" \
  --chunks_path "${OUTPUT_BASE}/chunks/article_chunks_api.json" \
  --chunk4_path "${OUTPUT_BASE}/chunk4/article_chunk4_api.json" \
  --topics_path "${OUTPUT_BASE}/topics/article_topics_api.json" \
  --questions_path "${OUTPUT_BASE}/questions/article_questions_api.json" \
  --validated_questions_path "${OUTPUT_BASE}/validated_questions/article_questions_api_validated.json" \
  --answers_path "${OUTPUT_BASE}/answers/article_answers_api.json" \
  --syndatas_path "${OUTPUT_BASE}/syndatas/syndatas_api.json" \
  --start_idx 0 \
  --end_idx 10 \
  --max_workers ${MAX_WORKERS} \
  --batch_size ${BATCH_SIZE}

echo ""
echo "✅ 执行完成！"
echo "📊 输出文件: ${OUTPUT_BASE}/syndatas/syndatas_api.json"
echo ""
echo "💡 提示："
echo "1. 确保你的本地API服务正在运行"
echo "2. 常见的本地API服务:"
echo "   - vLLM: python -m vllm.entrypoints.openai.api_server --model /path/to/model"
echo "   - FastChat: python -m fastchat.serve.openai_api_server --model-path /path/to/model"
echo "   - Text Generation Inference: docker run --gpus all -p 8080:80 ghcr.io/huggingface/text-generation-inference:latest --model-id /path/to/model"
echo "3. API服务通常在 http://localhost:8000 或 http://localhost:8080"
echo ""
echo "📅 完成时间: $(date)"
echo "=================================================="