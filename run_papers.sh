#!/bin/bash

# 论文文件处理运行脚本
# 用于运行 run_papers_processor.py

set -e  # 遇到错误时退出

echo "=========================================="
echo "论文文件处理脚本启动"
echo "=========================================="

# 检查 Python 是否可用
if ! command -v python3 &> /dev/null; then
    echo "错误: python3 未找到，请确保已安装 Python 3"
    exit 1
fi

# 检查工作目录
WORKSPACE_DIR="/workspace"
if [ ! -d "$WORKSPACE_DIR" ]; then
    echo "错误: 工作目录 $WORKSPACE_DIR 不存在"
    exit 1
fi

# 切换到工作目录
cd "$WORKSPACE_DIR"

# 检查必要文件是否存在
if [ ! -f "selected_papers.txt" ] && [ ! -f "selected_papers_2.txt" ]; then
    echo "警告: 未找到 selected_papers.txt 或 selected_papers_2.txt 文件"
    echo "将继续执行，但可能没有数据处理"
fi

# 运行 Python 脚本
echo "正在运行论文处理器..."
python3 run_papers_processor.py

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "脚本执行完成！"
    echo "=========================================="
    
    # 显示生成的文件
    if [ -f "papers_processing_report.txt" ]; then
        echo "生成的报告文件: papers_processing_report.txt"
    fi
    
    if [ -f "papers_processing.log" ]; then
        echo "日志文件: papers_processing.log"
    fi
    
else
    echo "错误: 脚本执行失败"
    exit 1
fi