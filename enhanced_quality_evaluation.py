#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版数据质量评估工具
基于原有的cot数据质量评估，增加对比分析和统计功能
目标：评估生成的高质量问答对
"""

import os
import json
import pandas as pd
from openai import OpenAI
import datetime
import logging
from tqdm import tqdm
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 初始化客户端 - 使用环境变量配置
client = OpenAI(
    base_url=os.getenv("COMPLETION_OPENAI_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.getenv("COMPLETION_OPENAI_API_KEY", "EMPTY")
)

def get_quality_evaluation_prompt():
    """获取质量评估prompt"""
    return """
作为半导体显示领域的专业质量评估专家，请严格按以下标准评估问答对的质量。评估分为核心维度，每个维度包含具体评估点和示例参考。

### 评估维度
1. 思维链逻辑质量（权重35%）
   - 步骤完整性：推理步骤是否覆盖问题所有关键点？是否遗漏必要环节？
   - 因果连贯性：前后步骤是否存在清晰因果关系？有无逻辑断裂？
   - 技术参数合理性：工艺参数是否符合物理规律？（例：LTPS退火温度不能超过玻璃转化点）
   - 错误回溯机制：是否考虑可能故障点？（例：分析Mura缺陷应包含设备精度因素）

2. 技术准确度（权重30%）
   - 材料特性：材料描述是否符合物性？（例：IGZO迁移率范围是否正确）
   - 制程参数：工艺参数是否匹配行业标准？（例：光刻精度是否满足当前产线能力）
   - 标准引用：是否准确引用SEMI/SID等国际标准？
   - 专利技术：技术方案是否规避近期专利？

3. 领域深度（权重20%）
   - 缺陷机理：是否分析根本原因？（例：亮暗点应关联电致迁移机制）
   - 技术趋势：是否覆盖最新发展？（例：需提及Micro LED巨量转移技术）
   - 工艺瓶颈：是否识别关键限制？（例：指出QD-OLED的喷墨打印精度瓶颈）

4. 应用价值（权重15%）
   - 工程可行性：方案是否具备量产实施条件？
   - 成本优化：是否量化成本效益？
   - 良率提升路径：是否提供可验证的改善方案？

### 输出格式要求（JSON）
{{
    "quality_rating": {{
        "overall": "high/medium/low",
        "detailed_scores": {{
            "reasoning_chain": {{"score": "high/medium/low", "issues": ["具体问题1", "具体问题2"]}},
            "technical_accuracy": {{"score": "high/medium/low", "issues": ["具体问题1"]}},
            "domain_depth": {{"score": "high/medium/low", "issues": ["具体问题1"]}},
            "application_value": {{"score": "high/medium/low", "issues": ["具体问题1"]}}
        }}
    }},
    "improvement_suggestions": ["建议1", "建议2"],
    "confidence_level": "high/medium/low"
}}

问题：{question}
答案：{answer}
"""

def evaluate_qa_pair(question, answer, model_name="gpt-4"):
    """评估单个问答对的质量"""
    try:
        prompt = get_quality_evaluation_prompt().format(question=question, answer=answer)
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个专业的半导体显示技术领域质量评估专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2048
        )
        
        result = response.choices[0].message.content.strip()
        
        # 尝试解析JSON结果
        try:
            evaluation = json.loads(result)
            return evaluation
        except json.JSONDecodeError:
            logger.warning(f"无法解析评估结果JSON: {result[:100]}...")
            return None
            
    except Exception as e:
        logger.error(f"评估问答对时出错: {e}")
        return None

def load_syndatas(file_path):
    """加载合成数据文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"成功加载数据文件: {file_path}, 包含 {len(data)} 个问答对")
        return data
    except Exception as e:
        logger.error(f"加载数据文件失败 {file_path}: {e}")
        return []

def evaluate_dataset(data_file, output_file, sample_size=None, model_name="gpt-4"):
    """评估整个数据集的质量"""
    logger.info(f"开始评估数据集: {data_file}")
    
    # 加载数据
    data = load_syndatas(data_file)
    if not data:
        logger.error("数据加载失败，退出评估")
        return
    
    # 如果指定了样本大小，则随机采样
    if sample_size and sample_size < len(data):
        import random
        data = random.sample(data, sample_size)
        logger.info(f"随机采样 {sample_size} 个问答对进行评估")
    
    evaluations = []
    
    for i, item in enumerate(tqdm(data, desc="评估问答对")):
        try:
            question = item.get('question', '')
            answer = item.get('reasoning_answer', '') or item.get('answer', '')
            
            if not question or not answer:
                logger.warning(f"第 {i} 个项目缺少问题或答案，跳过")
                continue
            
            # 评估问答对
            evaluation = evaluate_qa_pair(question, answer, model_name)
            
            if evaluation:
                evaluation['item_index'] = i
                evaluation['question'] = question[:100] + "..." if len(question) > 100 else question
                evaluations.append(evaluation)
            else:
                logger.warning(f"第 {i} 个问答对评估失败")
                
        except Exception as e:
            logger.error(f"评估第 {i} 个问答对时出错: {e}")
    
    # 保存评估结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluations, f, ensure_ascii=False, indent=2)
    
    logger.info(f"评估完成，结果保存到: {output_file}")
    logger.info(f"成功评估 {len(evaluations)}/{len(data)} 个问答对")
    
    return evaluations

def analyze_evaluation_results(evaluation_file):
    """分析评估结果并生成统计报告"""
    try:
        with open(evaluation_file, 'r', encoding='utf-8') as f:
            evaluations = json.load(f)
    except Exception as e:
        logger.error(f"加载评估结果失败: {e}")
        return
    
    if not evaluations:
        logger.error("评估结果为空")
        return
    
    # 统计整体质量分布
    overall_scores = defaultdict(int)
    dimension_scores = defaultdict(lambda: defaultdict(int))
    confidence_levels = defaultdict(int)
    
    for eval_result in evaluations:
        # 整体评分统计
        overall = eval_result.get('quality_rating', {}).get('overall', 'unknown')
        overall_scores[overall] += 1
        
        # 各维度评分统计
        detailed_scores = eval_result.get('quality_rating', {}).get('detailed_scores', {})
        for dimension, score_info in detailed_scores.items():
            score = score_info.get('score', 'unknown')
            dimension_scores[dimension][score] += 1
        
        # 置信度统计
        confidence = eval_result.get('confidence_level', 'unknown')
        confidence_levels[confidence] += 1
    
    # 生成统计报告
    total_count = len(evaluations)
    
    print("\n" + "="*50)
    print("📊 数据质量评估统计报告")
    print("="*50)
    
    print(f"\n📈 整体质量分布 (总计: {total_count} 个问答对):")
    for score, count in sorted(overall_scores.items()):
        percentage = (count / total_count) * 100
        print(f"   {score:8}: {count:4} ({percentage:5.1f}%)")
    
    print(f"\n🔍 各维度质量分布:")
    for dimension, scores in dimension_scores.items():
        print(f"\n   {dimension}:")
        for score, count in sorted(scores.items()):
            percentage = (count / total_count) * 100
            print(f"     {score:8}: {count:4} ({percentage:5.1f}%)")
    
    print(f"\n🎯 评估置信度分布:")
    for confidence, count in sorted(confidence_levels.items()):
        percentage = (count / total_count) * 100
        print(f"   {confidence:8}: {count:4} ({percentage:5.1f}%)")
    
    # 计算高质量问答对比例
    high_quality_count = overall_scores.get('high', 0)
    high_quality_rate = (high_quality_count / total_count) * 100
    
    print(f"\n✨ 关键指标:")
    print(f"   高质量问答对数量: {high_quality_count}")
    print(f"   高质量问答对比例: {high_quality_rate:.1f}%")
    
    # 提取常见问题
    common_issues = defaultdict(int)
    for eval_result in evaluations:
        detailed_scores = eval_result.get('quality_rating', {}).get('detailed_scores', {})
        for dimension, score_info in detailed_scores.items():
            issues = score_info.get('issues', [])
            for issue in issues:
                common_issues[issue] += 1
    
    if common_issues:
        print(f"\n⚠️  常见问题 (TOP 10):")
        sorted_issues = sorted(common_issues.items(), key=lambda x: x[1], reverse=True)
        for i, (issue, count) in enumerate(sorted_issues[:10], 1):
            print(f"   {i:2}. {issue} ({count} 次)")
    
    print("\n" + "="*50)
    
    return {
        "total_count": total_count,
        "overall_scores": dict(overall_scores),
        "dimension_scores": dict(dimension_scores),
        "confidence_levels": dict(confidence_levels),
        "high_quality_rate": high_quality_rate,
        "common_issues": dict(sorted_issues[:10])
    }

def compare_datasets(datasets_info):
    """对比多个数据集的质量"""
    print("\n" + "="*60)
    print("🔄 数据集质量对比分析")
    print("="*60)
    
    comparison_results = {}
    
    for name, file_path in datasets_info.items():
        print(f"\n📊 分析数据集: {name}")
        eval_file = file_path.replace('.json', '_evaluation.json')
        
        if not os.path.exists(eval_file):
            print(f"   ⚠️  评估文件不存在: {eval_file}")
            continue
        
        stats = analyze_evaluation_results(eval_file)
        if stats:
            comparison_results[name] = stats
    
    # 生成对比表格
    if len(comparison_results) > 1:
        print(f"\n📈 数据集质量对比:")
        print(f"{'数据集':<20} {'总数':<8} {'高质量率':<10} {'置信度-高':<10}")
        print("-" * 60)
        
        for name, stats in comparison_results.items():
            high_quality_rate = stats['high_quality_rate']
            high_confidence_count = stats['confidence_levels'].get('high', 0)
            high_confidence_rate = (high_confidence_count / stats['total_count']) * 100
            
            print(f"{name:<20} {stats['total_count']:<8} {high_quality_rate:<9.1f}% {high_confidence_rate:<9.1f}%")
    
    return comparison_results

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="增强版数据质量评估工具")
    
    parser.add_argument("--input", type=str, help="单个数据文件路径")
    parser.add_argument("--compare", nargs='+', help="多个数据文件路径进行对比")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="评估结果保存目录")
    parser.add_argument("--sample_size", type=int, help="评估样本大小（用于大数据集采样）")
    parser.add_argument("--model", type=str, default="gpt-4", help="评估使用的模型")
    parser.add_argument("--skip_evaluation", action="store_true", help="跳过评估，仅分析已有结果")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = get_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.input:
        # 单文件评估
        input_path = Path(args.input)
        output_file = os.path.join(args.output_dir, f"{input_path.stem}_evaluation.json")
        
        if not args.skip_evaluation:
            evaluations = evaluate_dataset(args.input, output_file, args.sample_size, args.model)
        
        # 分析结果
        analyze_evaluation_results(output_file)
        
    elif args.compare:
        # 多文件对比评估
        datasets_info = {}
        
        for file_path in args.compare:
            file_path = Path(file_path)
            name = file_path.stem
            datasets_info[name] = str(file_path)
            
            if not args.skip_evaluation:
                output_file = os.path.join(args.output_dir, f"{name}_evaluation.json")
                evaluate_dataset(str(file_path), output_file, args.sample_size, args.model)
        
        # 对比分析
        compare_datasets(datasets_info)
        
    else:
        # 默认评估增强版管道的输出
        default_files = {
            "enhanced": "outputs_enhanced/syndatas/syndatas_enhanced.json",
            "fast": "outputs_enhanced/syndatas/syndatas_fast.json",
            "compat": "outputs_enhanced/syndatas/syndatas_compat.json"
        }
        
        existing_files = {name: path for name, path in default_files.items() if os.path.exists(path)}
        
        if not existing_files:
            logger.error("未找到默认的数据文件，请使用 --input 或 --compare 参数")
            return
        
        logger.info(f"找到 {len(existing_files)} 个默认数据文件进行评估")
        
        if not args.skip_evaluation:
            for name, file_path in existing_files.items():
                output_file = os.path.join(args.output_dir, f"{name}_evaluation.json")
                evaluate_dataset(file_path, output_file, args.sample_size, args.model)
        
        # 对比分析
        compare_datasets(existing_files)

if __name__ == "__main__":
    main()