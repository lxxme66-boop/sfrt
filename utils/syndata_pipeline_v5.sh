#!/bin/bash

# 增强版数据合成管道 v5 运行脚本
# 完整实现所有优化功能
# 目标：生成高质量问答对，支持批量推理和增量处理

echo "🚀 启动增强版数据合成管道 v5 - 完整优化版"
echo "🎯 目标：高效生成高质量问答对"
echo "📅 $(date)"
echo "=================================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES='4,5,6,7'  # 使用4张GPU
export CHUNK_NUM=4
export CHUNK_NUM_MIN=2
export NUM_distract=3
export PROMPT_KEY="deepseek-v2"

# 基础配置
DATA_DIR="data"
FILTERED_DATA_DIR="data_filtered_v5"
OUTPUT_BASE="outputs_v5"
INTERMEDIATE_DIR="${OUTPUT_BASE}/intermediate"

# 创建输出目录
mkdir -p ${OUTPUT_BASE}/{chunks,chunk4,topics,questions,validated_questions,answers,syndatas,stats}
mkdir -p ${INTERMEDIATE_DIR}

echo "📁 数据目录配置:"
echo "   原始数据: ${DATA_DIR}"
echo "   筛选数据: ${FILTERED_DATA_DIR}"
echo "   输出目录: ${OUTPUT_BASE}"
echo "   中间文件: ${INTERMEDIATE_DIR}"
echo ""

# 性能配置
BATCH_SIZE=32
MAX_WORKERS=4
MODEL_NAME="qwq_32"

echo "⚙️  性能配置:"
echo "   批量大小: ${BATCH_SIZE}"
echo "   并发数量: ${MAX_WORKERS}"
echo "   使用模型: ${MODEL_NAME}"
echo ""

# 模式1：完整增强管道（使用vLLM + 所有质量控制）
echo "🔥 模式1: 运行完整增强管道（vLLM + 批量推理 + 质量控制）"
python -m utils.syndata_pipeline_v5 \
  --data_dir "${DATA_DIR}" \
  --filtered_data_dir "${FILTERED_DATA_DIR}" \
  --chunks_path "${OUTPUT_BASE}/chunks/article_chunks_v5_full.json" \
  --chunk4_path "${OUTPUT_BASE}/chunk4/article_chunk4_v5_full.json" \
  --topics_path "${OUTPUT_BASE}/topics/article_topics_v5_full.json" \
  --questions_path "${OUTPUT_BASE}/questions/article_questions_v5_full.json" \
  --validated_questions_path "${OUTPUT_BASE}/validated_questions/article_questions_v5_validated.json" \
  --answers_path "${OUTPUT_BASE}/answers/article_answers_v5_full.json" \
  --syndatas_path "${OUTPUT_BASE}/syndatas/syndatas_v5_full.json" \
  --judge_output_path "${INTERMEDIATE_DIR}/judge_output_v5.jsonl" \
  --question_output_path "${INTERMEDIATE_DIR}/question_output_v5.jsonl" \
  --question_li_output_path "${INTERMEDIATE_DIR}/question_li_output_v5.jsonl" \
  --start_idx 0 \
  --end_idx 100 \
  --max_workers ${MAX_WORKERS} \
  --batch_size ${BATCH_SIZE} \
  --use_vllm \
  --model_name ${MODEL_NAME} \
  --gpu_memory_utilization 0.95 \
  --tensor_parallel_size 4

echo ""
echo "✅ 完整增强管道执行完成！"
echo ""

# 模式2：快速模式（跳过文档筛选，但保留问题验证）
echo "⚡ 模式2: 运行快速模式（跳过文档筛选）"
python -m utils.syndata_pipeline_v5 \
  --data_dir "${DATA_DIR}" \
  --filtered_data_dir "${FILTERED_DATA_DIR}_fast" \
  --chunks_path "${OUTPUT_BASE}/chunks/article_chunks_v5_fast.json" \
  --chunk4_path "${OUTPUT_BASE}/chunk4/article_chunk4_v5_fast.json" \
  --topics_path "${OUTPUT_BASE}/topics/article_topics_v5_fast.json" \
  --questions_path "${OUTPUT_BASE}/questions/article_questions_v5_fast.json" \
  --validated_questions_path "${OUTPUT_BASE}/validated_questions/article_questions_v5_fast_validated.json" \
  --answers_path "${OUTPUT_BASE}/answers/article_answers_v5_fast.json" \
  --syndatas_path "${OUTPUT_BASE}/syndatas/syndatas_v5_fast.json" \
  --question_output_path "${INTERMEDIATE_DIR}/question_output_v5_fast.jsonl" \
  --question_li_output_path "${INTERMEDIATE_DIR}/question_li_output_v5_fast.jsonl" \
  --start_idx 0 \
  --end_idx 100 \
  --skip_document_filter \
  --max_workers ${MAX_WORKERS} \
  --batch_size ${BATCH_SIZE} \
  --use_vllm \
  --model_name ${MODEL_NAME}

echo ""
echo "✅ 快速模式执行完成！"
echo ""

# 模式3：标准API模式（不使用vLLM，适合小规模测试）
echo "🔧 模式3: 运行标准API模式（不使用vLLM）"
python -m utils.syndata_pipeline_v5 \
  --data_dir "${DATA_DIR}" \
  --filtered_data_dir "${FILTERED_DATA_DIR}_api" \
  --chunks_path "${OUTPUT_BASE}/chunks/article_chunks_v5_api.json" \
  --chunk4_path "${OUTPUT_BASE}/chunk4/article_chunk4_v5_api.json" \
  --topics_path "${OUTPUT_BASE}/topics/article_topics_v5_api.json" \
  --questions_path "${OUTPUT_BASE}/questions/article_questions_v5_api.json" \
  --validated_questions_path "${OUTPUT_BASE}/validated_questions/article_questions_v5_api_validated.json" \
  --answers_path "${OUTPUT_BASE}/answers/article_answers_v5_api.json" \
  --syndatas_path "${OUTPUT_BASE}/syndatas/syndatas_v5_api.json" \
  --start_idx 0 \
  --end_idx 10 \
  --max_workers 2 \
  --batch_size 4

echo ""
echo "✅ 标准API模式执行完成！"
echo ""

# 模式4：兼容模式（与v3完全兼容）
echo "🔄 模式4: 运行兼容模式（跳过所有增强功能）"
python -m utils.syndata_pipeline_v5 \
  --data_dir "${DATA_DIR}" \
  --filtered_data_dir "${FILTERED_DATA_DIR}_compat" \
  --chunks_path "${OUTPUT_BASE}/chunks/article_chunks_v5_compat.json" \
  --chunk4_path "${OUTPUT_BASE}/chunk4/article_chunk4_v5_compat.json" \
  --topics_path "${OUTPUT_BASE}/topics/article_topics_v5_compat.json" \
  --questions_path "${OUTPUT_BASE}/questions/article_questions_v5_compat.json" \
  --validated_questions_path "${OUTPUT_BASE}/validated_questions/article_questions_v5_compat_validated.json" \
  --answers_path "${OUTPUT_BASE}/answers/article_answers_v5_compat.json" \
  --syndatas_path "${OUTPUT_BASE}/syndatas/syndatas_v5_compat.json" \
  --start_idx 0 \
  --end_idx 10 \
  --skip_document_filter \
  --skip_question_validation \
  --skip_text_preprocessing \
  --max_workers 1 \
  --batch_size 1

echo ""
echo "✅ 兼容模式执行完成！"
echo ""

# 统计和对比
echo "🎉 所有管道执行完成！"
echo ""
echo "📊 输出文件对比："
echo "   完整增强版: ${OUTPUT_BASE}/syndatas/syndatas_v5_full.json"
echo "   快速模式:   ${OUTPUT_BASE}/syndatas/syndatas_v5_fast.json"
echo "   标准API:    ${OUTPUT_BASE}/syndatas/syndatas_v5_api.json"
echo "   兼容模式:   ${OUTPUT_BASE}/syndatas/syndatas_v5_compat.json"
echo ""
echo "📈 性能统计文件:"
ls -la ${OUTPUT_BASE}/stats/
echo ""
echo "📝 中间结果文件:"
ls -la ${INTERMEDIATE_DIR}/
echo ""
echo "🔍 建议使用 cot数据质量评估.py 对生成的数据进行质量评估"
echo "📅 完成时间: $(date)"
echo "=================================================="

# 生成对比报告
echo ""
echo "📋 生成对比报告..."
python -c "
import json
import os

output_base = '${OUTPUT_BASE}'
stats_files = [
    f'{output_base}/syndatas/pipeline_v5_stats.json',
    f'{output_base}/quality_filter_stats.json',
    f'{output_base}/question_generation_stats.json'
]

print('\\n=== 管道执行统计 ===')
for stats_file in stats_files:
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        print(f'\\n{os.path.basename(stats_file)}:')
        for key, value in stats.items():
            print(f'  {key}: {value}')
"

echo ""
echo "✨ 全部完成！"