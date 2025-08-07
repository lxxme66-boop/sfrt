#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文文件处理脚本
用于处理 selected_papers.txt 和 selected_papers_2.txt 中的论文文件列表
"""

import os
import sys
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('papers_processing.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PapersProcessor:
    def __init__(self, workspace_path="/workspace"):
        self.workspace_path = Path(workspace_path)
        self.papers_files = [
            self.workspace_path / "selected_papers.txt",
            self.workspace_path / "selected_papers_2.txt"
        ]
        
    def read_paper_list(self, file_path):
        """读取论文文件列表"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                papers = [line.strip() for line in f if line.strip()]
            logger.info(f"从 {file_path} 读取了 {len(papers)} 篇论文")
            return papers
        except Exception as e:
            logger.error(f"读取文件 {file_path} 时出错: {e}")
            return []
    
    def process_papers(self):
        """处理所有论文文件"""
        all_papers = []
        
        for file_path in self.papers_files:
            if file_path.exists():
                papers = self.read_paper_list(file_path)
                all_papers.extend(papers)
            else:
                logger.warning(f"文件不存在: {file_path}")
        
        # 去重
        unique_papers = list(set(all_papers))
        logger.info(f"总共找到 {len(all_papers)} 篇论文，去重后 {len(unique_papers)} 篇")
        
        return unique_papers
    
    def check_paper_files_exist(self, papers):
        """检查论文文件是否存在"""
        existing_files = []
        missing_files = []
        
        # 可能的数据目录
        data_dirs = [
            self.workspace_path / "data",
            self.workspace_path / "data_emb",
            self.workspace_path
        ]
        
        for paper in papers:
            found = False
            for data_dir in data_dirs:
                if data_dir.exists():
                    paper_path = data_dir / paper
                    if paper_path.exists():
                        existing_files.append(str(paper_path))
                        found = True
                        break
            
            if not found:
                missing_files.append(paper)
        
        logger.info(f"找到 {len(existing_files)} 个存在的文件")
        logger.info(f"缺失 {len(missing_files)} 个文件")
        
        return existing_files, missing_files
    
    def generate_report(self, papers, existing_files, missing_files):
        """生成处理报告"""
        report_path = self.workspace_path / "papers_processing_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("论文文件处理报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"总论文数: {len(papers)}\n")
            f.write(f"存在文件数: {len(existing_files)}\n")
            f.write(f"缺失文件数: {len(missing_files)}\n\n")
            
            if existing_files:
                f.write("存在的文件:\n")
                f.write("-" * 30 + "\n")
                for file in existing_files:
                    f.write(f"{file}\n")
                f.write("\n")
            
            if missing_files:
                f.write("缺失的文件:\n")
                f.write("-" * 30 + "\n")
                for file in missing_files:
                    f.write(f"{file}\n")
        
        logger.info(f"报告已生成: {report_path}")
        return report_path
    
    def run(self):
        """运行主处理流程"""
        logger.info("开始处理论文文件...")
        
        # 1. 读取论文列表
        papers = self.process_papers()
        if not papers:
            logger.error("没有找到任何论文文件")
            return
        
        # 2. 检查文件是否存在
        existing_files, missing_files = self.check_paper_files_exist(papers)
        
        # 3. 生成报告
        report_path = self.generate_report(papers, existing_files, missing_files)
        
        # 4. 显示统计信息
        print("\n" + "=" * 60)
        print("处理完成！统计信息:")
        print(f"总论文数: {len(papers)}")
        print(f"存在文件数: {len(existing_files)}")
        print(f"缺失文件数: {len(missing_files)}")
        print(f"完整性: {len(existing_files)/len(papers)*100:.1f}%")
        print(f"详细报告: {report_path}")
        print("=" * 60)
        
        return {
            'total_papers': len(papers),
            'existing_files': existing_files,
            'missing_files': missing_files,
            'report_path': str(report_path)
        }

def main():
    """主函数"""
    processor = PapersProcessor()
    result = processor.run()
    
    if result:
        # 如果需要进一步处理，可以在这里添加代码
        logger.info("处理完成")
    else:
        logger.error("处理失败")
        sys.exit(1)

if __name__ == "__main__":
    main()