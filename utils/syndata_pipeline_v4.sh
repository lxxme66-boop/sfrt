#!/bin/bash

# 增强版数据合成管道 v4 运行脚本
# 目标：生成高质量问答对

echo "🚀 启动增强版数据合成管道 v4"
echo "🎯 目标：生成高质量问答对"
echo "📅 $(date)"
echo "=" * 50

# 设置环境变量
export CHUNK_NUM=4
export CHUNK_NUM_MIN=2
export NUM_distract=3
export PROMPT_KEY="deepseek-v2"

# 基础配置
DATA_DIR="data"
FILTERED_DATA_DIR="data_filtered"
OUTPUT_BASE="outputs_enhanced"

# 创建输出目录
mkdir -p ${OUTPUT_BASE}/{chunks,chunk4,topics,questions,validated_questions,answers,syndatas}

echo "📁 数据目录配置:"
echo "   原始数据: ${DATA_DIR}"
echo "   筛选数据: ${FILTERED_DATA_DIR}"
echo "   输出目录: ${OUTPUT_BASE}"
echo ""

# 完整管道运行（包含所有质量控制步骤）
echo "🔥 运行完整增强管道（包含文档筛选和问题验证）"
python -m utils.syndata_pipeline_v4 \
  --data_dir "${DATA_DIR}" \
  --filtered_data_dir "${FILTERED_DATA_DIR}" \
  --chunks_path "${OUTPUT_BASE}/chunks/article_chunks_enhanced.json" \
  --chunk4_path "${OUTPUT_BASE}/chunk4/article_chunk4_enhanced.json" \
  --topics_path "${OUTPUT_BASE}/topics/article_topics_enhanced.json" \
  --questions_path "${OUTPUT_BASE}/questions/article_questions_enhanced.json" \
  --validated_questions_path "${OUTPUT_BASE}/validated_questions/article_questions_validated.json" \
  --answers_path "${OUTPUT_BASE}/answers/article_answers_enhanced.json" \
  --syndatas_path "${OUTPUT_BASE}/syndatas/syndatas_enhanced.json" \
  --start_idx 0 \
  --end_idx 10 \
  --max_workers 4

echo ""
echo "✅ 完整增强管道执行完成！"
echo ""

# 快速模式运行（跳过文档筛选）
echo "⚡ 运行快速模式（跳过文档筛选）"
python -m utils.syndata_pipeline_v4 \
  --data_dir "${DATA_DIR}" \
  --filtered_data_dir "${FILTERED_DATA_DIR}" \
  --chunks_path "${OUTPUT_BASE}/chunks/article_chunks_fast.json" \
  --chunk4_path "${OUTPUT_BASE}/chunk4/article_chunk4_fast.json" \
  --topics_path "${OUTPUT_BASE}/topics/article_topics_fast.json" \
  --questions_path "${OUTPUT_BASE}/questions/article_questions_fast.json" \
  --validated_questions_path "${OUTPUT_BASE}/validated_questions/article_questions_fast_validated.json" \
  --answers_path "${OUTPUT_BASE}/answers/article_answers_fast.json" \
  --syndatas_path "${OUTPUT_BASE}/syndatas/syndatas_fast.json" \
  --start_idx 0 \
  --end_idx 10 \
  --skip_document_filter \
  --max_workers 4

echo ""
echo "✅ 快速模式执行完成！"
echo ""

# 兼容模式运行（跳过所有增强功能，与v3兼容）
echo "🔄 运行兼容模式（跳过所有增强功能）"
python -m utils.syndata_pipeline_v4 \
  --data_dir "${DATA_DIR}" \
  --filtered_data_dir "${FILTERED_DATA_DIR}" \
  --chunks_path "${OUTPUT_BASE}/chunks/article_chunks_compat.json" \
  --chunk4_path "${OUTPUT_BASE}/chunk4/article_chunk4_compat.json" \
  --topics_path "${OUTPUT_BASE}/topics/article_topics_compat.json" \
  --questions_path "${OUTPUT_BASE}/questions/article_questions_compat.json" \
  --validated_questions_path "${OUTPUT_BASE}/validated_questions/article_questions_compat_validated.json" \
  --answers_path "${OUTPUT_BASE}/answers/article_answers_compat.json" \
  --syndatas_path "${OUTPUT_BASE}/syndatas/syndatas_compat.json" \
  --start_idx 0 \
  --end_idx 10 \
  --skip_document_filter \
  --skip_question_validation \
  --max_workers 4

echo ""
echo "✅ 兼容模式执行完成！"
echo ""

echo "🎉 所有管道执行完成！"
echo "📊 输出文件对比："
echo "   完整增强版: ${OUTPUT_BASE}/syndatas/syndatas_enhanced.json"
echo "   快速模式:   ${OUTPUT_BASE}/syndatas/syndatas_fast.json" 
echo "   兼容模式:   ${OUTPUT_BASE}/syndatas/syndatas_compat.json"
echo ""
echo "🔍 建议使用 cot数据质量评估.py 对生成的数据进行质量评估"
echo "📅 完成时间: $(date)"