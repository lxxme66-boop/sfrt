#!/bin/bash

echo "RAFT运行脚本使用示例"
echo "===================="
echo ""

echo "1. 处理data目录下的所有文本文件（使用默认参数）:"
echo "   ./run.sh"
echo ""

echo "2. 处理单个文件:"
echo "   ./run.sh -f data/97平板显示综述_朱昌昌_llm_correct.md"
echo ""

echo "3. 处理特定目录，生成更多问题:"
echo "   ./run.sh -i data -q 5 -d 4"
echo ""

echo "4. 使用不同的模型:"
echo "   ./run.sh -m qwen-plus"
echo ""

echo "5. 直接使用Python脚本（更多控制选项）:"
echo "   python3 run_raft.py --input data --questions 3 --chunk-size 1024"
echo ""

echo "6. 处理单个文件的Python命令:"
echo "   python3 run_raft.py --single-file data/97平板显示综述_朱昌昌_llm_correct.md"
echo ""

echo "7. 查看帮助:"
echo "   ./run.sh -h"
echo "   python3 run_raft.py --help"
echo ""

echo "注意事项:"
echo "- 确保已安装所需的Python依赖（查看requirements.txt）"
echo "- 确保已配置API密钥（在.env文件中）"
echo "- 输出将保存在outputs目录下，按文件名和时间戳组织"