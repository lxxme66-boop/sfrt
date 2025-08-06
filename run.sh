#!/bin/bash

# RAFT运行脚本
# 用于处理文本文件并生成合成数据集

echo "==================================="
echo "RAFT 文本处理运行脚本"
echo "==================================="

# 设置默认参数
INPUT_DIR="data"
OUTPUT_BASE="outputs"
OUTPUT_FORMAT="completion"
DISTRACTORS=3
P=1.0
DOCTYPE="txt"
CHUNK_SIZE=512
QUESTIONS=2
COMPLETION_MODEL="deepseek-r1-250120"
SYSTEM_PROMPT="deepseek-v2"

# 显示使用说明
show_help() {
    echo "使用方法:"
    echo "  ./run.sh [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help              显示帮助信息"
    echo "  -i, --input DIR         输入目录或文件 (默认: data)"
    echo "  -o, --output DIR        输出基础目录 (默认: outputs)"
    echo "  -f, --file FILE         处理单个文件"
    echo "  -q, --questions NUM     每个块的问题数量 (默认: 2)"
    echo "  -d, --distractors NUM   干扰项数量 (默认: 3)"
    echo "  -c, --chunk-size SIZE   文本块大小 (默认: 512)"
    echo "  -m, --model MODEL       补全模型 (默认: deepseek-r1-250120)"
    echo ""
    echo "示例:"
    echo "  ./run.sh                           # 处理data目录下的所有文本文件"
    echo "  ./run.sh -f data/example.txt      # 处理单个文件"
    echo "  ./run.sh -i custom_data -q 5      # 处理custom_data目录，每块生成5个问题"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        -f|--file)
            SINGLE_FILE="$2"
            shift 2
            ;;
        -q|--questions)
            QUESTIONS="$2"
            shift 2
            ;;
        -d|--distractors)
            DISTRACTORS="$2"
            shift 2
            ;;
        -c|--chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        -m|--model)
            COMPLETION_MODEL="$2"
            shift 2
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3"
    exit 1
fi

# 检查必要的文件
if [ ! -f "raft.py" ]; then
    echo "错误: 未找到raft.py文件"
    exit 1
fi

if [ ! -f "run_raft.py" ]; then
    echo "错误: 未找到run_raft.py文件"
    exit 1
fi

# 构建命令
CMD="python3 run_raft.py"
CMD="$CMD --input \"$INPUT_DIR\""
CMD="$CMD --output-base \"$OUTPUT_BASE\""
CMD="$CMD --output-format \"$OUTPUT_FORMAT\""
CMD="$CMD --distractors $DISTRACTORS"
CMD="$CMD --p $P"
CMD="$CMD --doctype \"$DOCTYPE\""
CMD="$CMD --chunk-size $CHUNK_SIZE"
CMD="$CMD --questions $QUESTIONS"
CMD="$CMD --completion-model \"$COMPLETION_MODEL\""
CMD="$CMD --system-prompt-key \"$SYSTEM_PROMPT\""

# 如果指定了单个文件
if [ ! -z "$SINGLE_FILE" ]; then
    CMD="$CMD --single-file \"$SINGLE_FILE\""
fi

# 显示运行信息
echo ""
echo "运行参数:"
echo "  输入: ${SINGLE_FILE:-$INPUT_DIR}"
echo "  输出目录: $OUTPUT_BASE"
echo "  问题数量: $QUESTIONS"
echo "  干扰项: $DISTRACTORS"
echo "  块大小: $CHUNK_SIZE"
echo "  模型: $COMPLETION_MODEL"
echo ""

# 确认运行
read -p "是否开始处理? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 运行命令
echo "开始处理..."
eval $CMD

echo ""
echo "处理完成!"