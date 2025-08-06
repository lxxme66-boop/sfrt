import json
import re
from collections import defaultdict

def extract_quality_ratings(content_str):
    """
    使用正则表达式从content字符串中提取所有评分
    包括overall和detailed_scores中的各项评分
    返回字典包含overall和各个分项的评分
    """
    ratings = {
        'overall': None,
        'detailed_scores': {}
    }
    
    # 1. 提取overall评分
    overall_match = re.search(r'"overall"\s*:\s*"([^"]+)"', content_str, re.IGNORECASE)
    if overall_match:
        overall = overall_match.group(1).lower()
        if overall in ['high', 'medium', 'low']:
            ratings['overall'] = overall
    
    # 2. 提取detailed_scores中的各项评分
    # 匹配所有分项评分（Relevance, Logical Consistency, Terminology Usage, Factual Correctness）
    score_matches = re.finditer(
        r'"(Relevance|Logical Consistency|Terminology Usage|Factual Correctness)"\s*:\s*{[^}]*"score"\s*:\s*"([^"]+)"',
        content_str,
        re.IGNORECASE
    )
    
    for match in score_matches:
        category = match.group(1)
        score = match.group(2).lower()
        if score in ['high', 'medium', 'low']:
            ratings['detailed_scores'][category] = score
    
    return ratings

def calculate_medium_quality_score(detailed_scores):
    """
    计算中等质量数据的综合得分
    高分=3分，中分=2分，低分=1分
    """
    score_mapping = {'high': 3, 'medium': 2, 'low': 1}
    total_score = 0
    
    for category, score in detailed_scores.items():
        total_score += score_mapping.get(score, 0)
    
    return total_score

def filter_data(jsonl_file, output_file, target_count=2400):
    """
    筛选数据：
    1. 保留所有高质量数据
    2. 去除所有低质量数据
    3. 从中等质量数据中根据分项得分选取最好的，直到达到目标数量
    """
    high_quality = []
    medium_quality = []
    
    # 读取并分类数据
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                content_str = data['response']['body']['choices'][0]['message']['content']
                ratings = extract_quality_ratings(content_str)
                
                if ratings['overall'] == 'high':
                    high_quality.append((data, ratings))
                elif ratings['overall'] == 'medium':
                    medium_quality.append((data, ratings))
                # 忽略low质量数据
                
            except (json.JSONDecodeError, KeyError, AttributeError):
                continue
    
    print(f"Found {len(high_quality)} high quality entries")
    print(f"Found {len(medium_quality)} medium quality entries")
    
    # 检查高质量数据是否足够
    if len(high_quality) >= target_count:
        selected_data = [item[0] for item in high_quality[:target_count]]
        print(f"Using only high quality data (more than target count)")
    else:
        # 需要从中等质量数据中补充
        needed = target_count - len(high_quality)
        print(f"Need {needed} more entries from medium quality data")
        
        # 对中等质量数据按分项得分排序
        medium_quality_sorted = sorted(
            medium_quality,
            key=lambda x: calculate_medium_quality_score(x[1]['detailed_scores']),
            reverse=True
        )
        
        selected_medium = medium_quality_sorted[:needed]
        selected_data = [item[0] for item in high_quality] + [item[0] for item in selected_medium]
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for data in selected_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"Successfully saved {len(selected_data)} entries to {output_file}")
