#!/bin/bash

# 自定义数据合成管道 v5 运行脚本
# 根据你的需求修改以下参数

echo "🚀 启动自定义数据合成管道 v5"
echo "📅 $(date)"
echo "=================================================="

# ========== 配置区域 ==========

# GPU配置（根据你的硬件调整）
export CUDA_VISIBLE_DEVICES='0'  # 使用哪些GPU，例如 '0,1,2,3' 或 '0'

# 基础配置
DATA_DIR="data"                    # 你的原始数据目录
OUTPUT_BASE="outputs_custom"       # 输出目录名称
START_IDX=0                       # 处理文档的起始索引
END_IDX=50                        # 处理文档的结束索引（None表示处理所有）

# 性能配置
BATCH_SIZE=16                     # 批量大小（根据GPU内存调整）
MAX_WORKERS=2                     # 并发数量
GPU_MEMORY=0.8                    # GPU内存利用率（0.8表示80%）
TENSOR_PARALLEL=1                 # 张量并行数（单GPU设为1）

# 模型配置
MODEL_NAME="qwq_32"               # 使用的模型名称
USE_VLLM="true"                   # 是否使用vLLM（true/false）

# 功能开关
SKIP_DOC_FILTER="false"           # 是否跳过文档质量筛选
SKIP_QUESTION_VAL="false"         # 是否跳过问题验证
SKIP_PREPROCESSING="false"        # 是否跳过文本预处理

# ========== 创建输出目录 ==========
mkdir -p ${OUTPUT_BASE}/{chunks,chunk4,topics,questions,validated_questions,answers,syndatas,stats,intermediate}

# ========== 构建命令 ==========
CMD="python -m utils.syndata_pipeline_v5"
CMD="$CMD --data_dir \"${DATA_DIR}\""
CMD="$CMD --filtered_data_dir \"${OUTPUT_BASE}/filtered_data\""
CMD="$CMD --chunks_path \"${OUTPUT_BASE}/chunks/article_chunks.json\""
CMD="$CMD --chunk4_path \"${OUTPUT_BASE}/chunk4/article_chunk4.json\""
CMD="$CMD --topics_path \"${OUTPUT_BASE}/topics/article_topics.json\""
CMD="$CMD --questions_path \"${OUTPUT_BASE}/questions/article_questions.json\""
CMD="$CMD --validated_questions_path \"${OUTPUT_BASE}/validated_questions/article_questions_validated.json\""
CMD="$CMD --answers_path \"${OUTPUT_BASE}/answers/article_answers.json\""
CMD="$CMD --syndatas_path \"${OUTPUT_BASE}/syndatas/syndatas.json\""
CMD="$CMD --judge_output_path \"${OUTPUT_BASE}/intermediate/judge_output.jsonl\""
CMD="$CMD --question_output_path \"${OUTPUT_BASE}/intermediate/question_output.jsonl\""
CMD="$CMD --question_li_output_path \"${OUTPUT_BASE}/intermediate/question_li_output.jsonl\""
CMD="$CMD --start_idx ${START_IDX}"
CMD="$CMD --end_idx ${END_IDX}"
CMD="$CMD --max_workers ${MAX_WORKERS}"
CMD="$CMD --batch_size ${BATCH_SIZE}"

# 添加vLLM相关参数
if [ "$USE_VLLM" = "true" ]; then
    CMD="$CMD --use_vllm"
    CMD="$CMD --model_name ${MODEL_NAME}"
    CMD="$CMD --gpu_memory_utilization ${GPU_MEMORY}"
    CMD="$CMD --tensor_parallel_size ${TENSOR_PARALLEL}"
fi

# 添加功能开关
if [ "$SKIP_DOC_FILTER" = "true" ]; then
    CMD="$CMD --skip_document_filter"
fi

if [ "$SKIP_QUESTION_VAL" = "true" ]; then
    CMD="$CMD --skip_question_validation"
fi

if [ "$SKIP_PREPROCESSING" = "true" ]; then
    CMD="$CMD --skip_text_preprocessing"
fi

# ========== 执行命令 ==========
echo "📋 执行命令："
echo "$CMD"
echo ""
echo "⏳ 开始处理..."

# 执行
eval $CMD

# ========== 完成后的操作 ==========
echo ""
echo "✅ 处理完成！"
echo ""
echo "📊 输出文件："
echo "   最终数据: ${OUTPUT_BASE}/syndatas/syndatas.json"
echo ""
echo "📈 查看统计信息："
if [ -f "${OUTPUT_BASE}/syndatas/pipeline_v5_stats.json" ]; then
    cat "${OUTPUT_BASE}/syndatas/pipeline_v5_stats.json"
fi
echo ""
echo "📅 完成时间: $(date)"