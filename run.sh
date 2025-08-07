#!/bin/bash

# 通用文本文件处理运行脚本

echo "=========================================="
echo "文本文件处理脚本"
echo "=========================================="

# 显示帮助信息
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "用法:"
    echo "  $0                          # 处理默认文件 selected_papers.txt"
    echo "  $0 filename.txt             # 处理指定的文本文件"
    echo "  $0 filename.txt --count     # 处理文件并统计行数"
    echo ""
    echo "示例:"
    echo "  $0 selected_papers.txt"
    echo "  $0 selected_papers_2.txt --count"
    exit 0
fi

# 设置默认参数
INPUT_FILE="${1:-selected_papers.txt}"
OPERATION="list"

# 检查是否要统计行数
if [ "$2" = "--count" ]; then
    OPERATION="count_lines"
fi

# 检查 Python 脚本是否存在
if [ ! -f "process_txt_files.py" ]; then
    echo "错误: process_txt_files.py 不存在"
    exit 1
fi

# 运行 Python 脚本
echo "正在处理文件: $INPUT_FILE"
echo "操作模式: $OPERATION"
echo ""

python3 process_txt_files.py "$INPUT_FILE" --operation "$OPERATION"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "脚本执行完成！"
    echo "查看生成的文件:"
    echo "  - processing_summary.txt (处理摘要)"
    echo "  - txt_processing.log (详细日志)"
else
    echo "错误: 脚本执行失败"
    exit 1
fi