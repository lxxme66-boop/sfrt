#!/usr/bin/env python3
"""
RAFT运行脚本
用于处理文本文件并生成合成数据集
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import json
from datetime import datetime

def run_raft_command(datapath, output_dir, **kwargs):
    """运行单个RAFT命令"""
    cmd = [
        "python", "raft.py",
        "--datapath", str(datapath),
        "--output", str(output_dir),
    ]
    
    # 添加其他参数
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    print(f"\n运行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ 成功处理: {datapath}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 处理失败: {datapath}")
        print(f"错误信息: {e.stderr}")
        return False

def find_text_files(directory):
    """查找目录中的所有文本文件"""
    text_extensions = ['.txt', '.md']
    files = []
    
    if os.path.isfile(directory):
        return [Path(directory)]
    
    for ext in text_extensions:
        files.extend(Path(directory).glob(f"**/*{ext}"))
    
    return sorted(files)

def main():
    parser = argparse.ArgumentParser(description="RAFT批量处理脚本")
    parser.add_argument("--input", type=str, default="data", 
                       help="输入文件或目录路径")
    parser.add_argument("--output-base", type=str, default="outputs", 
                       help="输出基础目录")
    parser.add_argument("--output-format", type=str, default="completion",
                       choices=["hf", "completion", "chat"],
                       help="输出格式")
    parser.add_argument("--distractors", type=int, default=3,
                       help="干扰项数量")
    parser.add_argument("--p", type=float, default=1.0,
                       help="采样概率")
    parser.add_argument("--doctype", type=str, default="txt",
                       choices=["api", "pdf", "json", "txt"],
                       help="文档类型")
    parser.add_argument("--chunk-size", type=int, default=512,
                       help="文本块大小")
    parser.add_argument("--questions", type=int, default=2,
                       help="每个块生成的问题数量")
    parser.add_argument("--completion-model", type=str, default="deepseek-r1-250120",
                       help="补全模型")
    parser.add_argument("--system-prompt-key", type=str, default="deepseek-v2",
                       choices=["gpt", "llama", "deepseek", "deepseek-v2"],
                       help="系统提示词键")
    parser.add_argument("--embedding-model", type=str, default=None,
                       help="嵌入模型")
    parser.add_argument("--single-file", type=str, default=None,
                       help="处理单个文件")
    
    args = parser.parse_args()
    
    # 确定要处理的文件
    if args.single_file:
        files_to_process = [Path(args.single_file)]
    else:
        files_to_process = find_text_files(args.input)
    
    if not files_to_process:
        print(f"未找到可处理的文件在: {args.input}")
        return
    
    print(f"找到 {len(files_to_process)} 个文件待处理")
    
    # 创建输出基础目录
    os.makedirs(args.output_base, exist_ok=True)
    
    # 处理统计
    success_count = 0
    failed_files = []
    
    # 处理每个文件
    for idx, file_path in enumerate(files_to_process, 1):
        print(f"\n[{idx}/{len(files_to_process)}] 处理文件: {file_path}")
        
        # 生成输出目录名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_stem = file_path.stem.replace(" ", "_")[:50]  # 限制文件名长度
        output_dir = Path(args.output_base) / f"{file_stem}_{timestamp}"
        
        # 运行RAFT
        success = run_raft_command(
            datapath=file_path,
            output_dir=output_dir,
            output_format=args.output_format,
            distractors=args.distractors,
            p=args.p,
            doctype=args.doctype,
            chunk_size=args.chunk_size,
            questions=args.questions,
            completion_model=args.completion_model,
            system_prompt_key=args.system_prompt_key,
            embedding_model=args.embedding_model
        )
        
        if success:
            success_count += 1
        else:
            failed_files.append(str(file_path))
    
    # 打印总结
    print("\n" + "="*50)
    print("处理完成!")
    print(f"成功: {success_count}/{len(files_to_process)}")
    
    if failed_files:
        print(f"\n失败的文件:")
        for f in failed_files:
            print(f"  - {f}")
    
    # 保存处理日志
    log_file = Path(args.output_base) / f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "total_files": len(files_to_process),
        "success_count": success_count,
        "failed_files": failed_files,
        "parameters": vars(args)
    }
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n处理日志已保存到: {log_file}")

if __name__ == "__main__":
    main()