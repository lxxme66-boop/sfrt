#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用文本文件处理脚本
可以处理任何包含文件列表的文本文件
"""

import os
import sys
import argparse
from pathlib import Path
import logging
from typing import List, Dict, Tuple

def setup_logging(log_file: str = "txt_processing.log"):
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def read_txt_file(file_path: Path) -> List[str]:
    """读取文本文件，返回行列表"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return []

def find_files_in_directories(filenames: List[str], search_dirs: List[Path]) -> Tuple[List[str], List[str]]:
    """在指定目录中查找文件"""
    found_files = []
    missing_files = []
    
    for filename in filenames:
        found = False
        for search_dir in search_dirs:
            if search_dir.exists():
                file_path = search_dir / filename
                if file_path.exists():
                    found_files.append(str(file_path))
                    found = True
                    break
        
        if not found:
            missing_files.append(filename)
    
    return found_files, missing_files

def process_files(found_files: List[str], operation: str = "list") -> Dict:
    """处理找到的文件"""
    results = {
        'processed': 0,
        'errors': 0,
        'details': []
    }
    
    for file_path in found_files:
        try:
            if operation == "list":
                # 仅列出文件信息
                file_info = {
                    'path': file_path,
                    'size': os.path.getsize(file_path),
                    'exists': True
                }
                results['details'].append(file_info)
                results['processed'] += 1
                
            elif operation == "count_lines":
                # 统计行数
                with open(file_path, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
                
                file_info = {
                    'path': file_path,
                    'lines': line_count,
                    'size': os.path.getsize(file_path)
                }
                results['details'].append(file_info)
                results['processed'] += 1
                
        except Exception as e:
            results['errors'] += 1
            print(f"处理文件 {file_path} 时出错: {e}")
    
    return results

def generate_summary_report(input_file: str, total_files: int, found_files: List[str], 
                          missing_files: List[str], process_results: Dict, 
                          output_file: str = "processing_summary.txt"):
    """生成处理摘要报告"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("文本文件处理摘要报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"输入文件: {input_file}\n")
        f.write(f"处理时间: {os.popen('date').read().strip()}\n\n")
        
        f.write("统计信息:\n")
        f.write("-" * 30 + "\n")
        f.write(f"总文件数: {total_files}\n")
        f.write(f"找到文件数: {len(found_files)}\n")
        f.write(f"缺失文件数: {len(missing_files)}\n")
        f.write(f"成功处理: {process_results['processed']}\n")
        f.write(f"处理错误: {process_results['errors']}\n")
        f.write(f"完整性: {len(found_files)/total_files*100:.1f}%\n\n")
        
        if found_files:
            f.write("找到的文件:\n")
            f.write("-" * 30 + "\n")
            for file_path in found_files:
                f.write(f"{file_path}\n")
            f.write("\n")
        
        if missing_files:
            f.write("缺失的文件 (前50个):\n")
            f.write("-" * 30 + "\n")
            for file_path in missing_files[:50]:
                f.write(f"{file_path}\n")
            if len(missing_files) > 50:
                f.write(f"... 还有 {len(missing_files) - 50} 个文件\n")
    
    return output_file

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='处理包含文件列表的文本文件')
    parser.add_argument('input_file', nargs='?', default='selected_papers.txt',
                       help='输入的文本文件路径 (默认: selected_papers.txt)')
    parser.add_argument('--search-dirs', nargs='*', 
                       default=['/workspace/data', '/workspace/data_emb', '/workspace'],
                       help='搜索文件的目录列表')
    parser.add_argument('--operation', choices=['list', 'count_lines'], default='list',
                       help='对找到的文件执行的操作')
    parser.add_argument('--output', default='processing_summary.txt',
                       help='输出报告文件名')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging()
    
    # 检查输入文件
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        sys.exit(1)
    
    logger.info(f"开始处理文件: {input_path}")
    
    # 读取文件列表
    file_list = read_txt_file(input_path)
    if not file_list:
        logger.error("没有从输入文件中读取到任何内容")
        sys.exit(1)
    
    logger.info(f"从输入文件读取了 {len(file_list)} 行")
    
    # 搜索文件
    search_dirs = [Path(d) for d in args.search_dirs]
    found_files, missing_files = find_files_in_directories(file_list, search_dirs)
    
    logger.info(f"找到 {len(found_files)} 个文件，缺失 {len(missing_files)} 个文件")
    
    # 处理找到的文件
    process_results = process_files(found_files, args.operation)
    
    # 生成报告
    report_path = generate_summary_report(
        str(input_path), len(file_list), found_files, 
        missing_files, process_results, args.output
    )
    
    # 显示结果
    print("\n" + "=" * 60)
    print("处理完成！")
    print(f"输入文件: {input_path}")
    print(f"总文件数: {len(file_list)}")
    print(f"找到文件数: {len(found_files)}")
    print(f"缺失文件数: {len(missing_files)}")
    print(f"成功处理: {process_results['processed']}")
    print(f"处理错误: {process_results['errors']}")
    print(f"完整性: {len(found_files)/len(file_list)*100:.1f}%")
    print(f"详细报告: {report_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()