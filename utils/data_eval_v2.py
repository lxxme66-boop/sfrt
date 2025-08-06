# import json
# import re
# import matplotlib.pyplot as plt
# from collections import defaultdict

# def extract_quality_ratings(content_str):
#     """
#     使用正则表达式从content字符串中提取所有评分
#     包括overall和detailed_scores中的各项评分
#     返回字典包含overall和各个分项的评分
#     """
#     ratings = {
#         'overall': None,
#         'detailed_scores': {}
#     }
    
#     # 1. 提取overall评分
#     overall_match = re.search(r'"overall"\s*:\s*"([^"]+)"', content_str, re.IGNORECASE)
#     if overall_match:
#         overall = overall_match.group(1).lower()
#         if overall in ['high', 'medium', 'low']:
#             ratings['overall'] = overall
    
#     # 2. 提取detailed_scores中的各项评分
#     # 首先匹配detailed_scores对象
#     detailed_scores_match = re.search(r'"detailed_scores"\s*:\s*({[^}]+})', content_str, re.IGNORECASE)
#     if detailed_scores_match:
#         detailed_scores_str = detailed_scores_match.group(1)
        
#         # 匹配每个category及其score
#         category_matches = re.finditer(
#             r'"([^"]+)"\s*:\s*{[^}]*"score"\s*:\s*"([^"]+)"[^}]*}',
#             detailed_scores_str,
#             re.IGNORECASE
#         )
        
#         for match in category_matches:
#             category = match.group(1)
#             score = match.group(2).lower()
#             if score in ['high', 'medium', 'low']:
#                 ratings['detailed_scores'][category] = score
    
#     return ratings

# def analyze_quality_ratings(jsonl_file):
#     # 初始化计数器
#     quality_counts = defaultdict(int)
#     detailed_counts = defaultdict(lambda: defaultdict(int))
#     total_entries = 0
#     parse_errors = 0
    
#     # 读取jsonl文件并统计质量评级
#     with open(jsonl_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             try:
#                 data = json.loads(line)
#                 # 获取content字符串
#                 content_str = data['response']['body']['choices'][0]['message']['content']
                
#                 # 提取评分信息
#                 ratings = extract_quality_ratings(content_str)
                
#                 # 统计overall评分
#                 if ratings['overall'] in ['high', 'medium', 'low']:
#                     quality_counts[ratings['overall']] += 1
#                     total_entries += 1
                    
#                     # 统计detailed_scores
#                     for category, score in ratings['detailed_scores'].items():
#                         detailed_counts[category][score] += 1
#                 else:
#                     parse_errors += 1
#             except (json.JSONDecodeError, KeyError, AttributeError) as e:
#                 parse_errors += 1
#                 continue
    
#     # 计算overall评分的百分比
#     quality_stats = {}
#     for quality, count in quality_counts.items():
#         percentage = (count / total_entries) * 100 if total_entries > 0 else 0
#         quality_stats[quality] = {'count': count, 'percentage': percentage}
    
#     # 计算detailed_scores的百分比
#     detailed_stats = {}
#     for category, scores in detailed_counts.items():
#         category_total = sum(scores.values())
#         category_stats = {}
#         for score, count in scores.items():
#             percentage = (count / category_total) * 100 if category_total > 0 else 0
#             category_stats[score] = {'count': count, 'percentage': percentage}
#         detailed_stats[category] = category_stats
    
#     return quality_stats, detailed_stats, total_entries, parse_errors

# def plot_quality_stats(quality_stats, detailed_stats, total_entries):
#     if total_entries == 0:
#         print("No valid data to plot.")
#         return
    
#     # 创建图表
#     fig = plt.figure(figsize=(18, 10))
    
#     # 整体评分饼图
#     ax1 = plt.subplot2grid((2, 3), (0, 0))
#     labels = ['High', 'Medium', 'Low']
#     counts = [quality_stats.get('high', {}).get('count', 0),
#               quality_stats.get('medium', {}).get('count', 0),
#               quality_stats.get('low', {}).get('count', 0)]
#     ax1.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90,
#             colors=['#4CAF50', '#FFC107', '#F44336'])
#     ax1.set_title('Overall Quality Rating Distribution')
    
#     # 整体评分柱状图
#     ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=2)
#     bars = ax2.bar(labels, counts, color=['#4CAF50', '#FFC107', '#F44336'])
#     ax2.set_title('Overall Quality Rating Counts')
#     ax2.set_ylabel('Count')
#     ax2.set_xlabel('Quality Rating')
    
#     # 在柱状图上添加数值标签
#     for bar in bars:
#         height = bar.get_height()
#         ax2.text(bar.get_x() + bar.get_width()/2., height,
#                 f'{height}\n({height/total_entries*100:.1f}%)',
#                 ha='center', va='bottom')
    
#     # 分项评分图表
#     if detailed_stats:
#         # 确定分项数量
#         categories = list(detailed_stats.keys())
#         num_categories = len(categories)
        
#         # 创建分项评分柱状图
#         ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=3)
        
#         # 准备数据
#         bar_width = 0.25
#         index = range(num_categories)
#         colors = {'high': '#4CAF50', 'medium': '#FFC107', 'low': '#F44336'}
        
#         # 绘制每个评分的柱状图
#         for i, score in enumerate(['high', 'medium', 'low']):
#             counts = [detailed_stats[cat].get(score, {}).get('count', 0) for cat in categories]
#             ax3.bar([x + i * bar_width for x in index], counts, bar_width,
#                     label=score.capitalize(), color=colors[score])
        
#         ax3.set_xlabel('Categories')
#         ax3.set_ylabel('Counts')
#         ax3.set_title('Detailed Quality Ratings by Category')
#         ax3.set_xticks([x + bar_width for x in index])
#         ax3.set_xticklabels(categories, rotation=45, ha='right')
#         ax3.legend()
    
#     plt.tight_layout()
#     plt.show()

# def show_quality(jsonl_file):
#     quality_stats, detailed_stats, total_entries, parse_errors = analyze_quality_ratings(jsonl_file)
    
#     # 打印统计结果
#     print(f"Total entries processed: {total_entries + parse_errors}")
#     print(f"Successfully parsed: {total_entries}")
#     print(f"Parse errors: {parse_errors}")
#     print("\nOverall Quality Statistics:")
#     for quality, stats in quality_stats.items():
#         print(f"{quality.capitalize()}: Count = {stats['count']}, Percentage = {stats['percentage']:.2f}%")
    
#     # 打印分项评分统计
#     if detailed_stats:
#         print("\nDetailed Quality Statistics:")
#         for category, scores in detailed_stats.items():
#             print(f"\nCategory: {category}")
#             for score, stats in scores.items():
#                 print(f"  {score.capitalize()}: Count = {stats['count']}, Percentage = {stats['percentage']:.2f}%")
    
#     # 绘制图表
#     plot_quality_stats(quality_stats, detailed_stats, total_entries)


import json
import re
import matplotlib.pyplot as plt
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

def analyze_quality_ratings(jsonl_file):
    # 初始化计数器
    quality_counts = defaultdict(int)
    detailed_counts = defaultdict(lambda: defaultdict(int))
    total_entries = 0
    parse_errors = 0
    
    # 读取jsonl文件并统计质量评级
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # 获取content字符串
                content_str = data['response']['body']['choices'][0]['message']['content']
                
                # 提取评分信息
                ratings = extract_quality_ratings(content_str)
                
                # 统计overall评分
                if ratings['overall'] in ['high', 'medium', 'low']:
                    quality_counts[ratings['overall']] += 1
                    total_entries += 1
                    
                    # 统计detailed_scores
                    for category, score in ratings['detailed_scores'].items():
                        detailed_counts[category][score] += 1
                else:
                    parse_errors += 1
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                parse_errors += 1
                continue
    
    # 计算overall评分的百分比
    quality_stats = {}
    for quality, count in quality_counts.items():
        percentage = (count / total_entries) * 100 if total_entries > 0 else 0
        quality_stats[quality] = {'count': count, 'percentage': percentage}
    
    # 计算detailed_scores的百分比
    detailed_stats = {}
    for category, scores in detailed_counts.items():
        category_total = sum(scores.values())
        category_stats = {}
        for score, count in scores.items():
            percentage = (count / category_total) * 100 if category_total > 0 else 0
            category_stats[score] = {'count': count, 'percentage': percentage}
        detailed_stats[category] = category_stats
    
    return quality_stats, detailed_stats, total_entries, parse_errors

def plot_quality_stats(quality_stats, detailed_stats, total_entries):
    if total_entries == 0:
        print("No valid data to plot.")
        return
    
    # 创建图表
    fig = plt.figure(figsize=(18, 12))
    
    # 整体评分饼图
    ax1 = plt.subplot2grid((3, 3), (0, 0))
    labels = ['High', 'Medium', 'Low']
    counts = [quality_stats.get('high', {}).get('count', 0),
              quality_stats.get('medium', {}).get('count', 0),
              quality_stats.get('low', {}).get('count', 0)]
    ax1.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90,
            colors=['#4CAF50', '#FFC107', '#F44336'])
    ax1.set_title('Overall Quality Rating Distribution')
    
    # 整体评分柱状图
    ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2)
    bars = ax2.bar(labels, counts, color=['#4CAF50', '#FFC107', '#F44336'])
    ax2.set_title('Overall Quality Rating Counts')
    ax2.set_ylabel('Count')
    ax2.set_xlabel('Quality Rating')
    
    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}\n({height/total_entries*100:.1f}%)',
                ha='center', va='bottom')
    
    # 分项评分图表
    if detailed_stats:
        # 确保所有分项都存在（即使计数为0）
        all_categories = ['Relevance', 'Logical Consistency', 'Terminology Usage', 'Factual Correctness']
        for category in all_categories:
            if category not in detailed_stats:
                detailed_stats[category] = {
                    'high': {'count': 0, 'percentage': 0},
                    'medium': {'count': 0, 'percentage': 0},
                    'low': {'count': 0, 'percentage': 0}
                }
        
        # 分项评分柱状图（按类别）
        ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=3)
        
        # 准备数据
        bar_width = 0.25
        index = range(len(all_categories))
        colors = {'high': '#4CAF50', 'medium': '#FFC107', 'low': '#F44336'}
        
        # 绘制每个评分的柱状图
        for i, score in enumerate(['high', 'medium', 'low']):
            counts = [detailed_stats[cat].get(score, {}).get('count', 0) for cat in all_categories]
            ax3.bar([x + i * bar_width for x in index], counts, bar_width,
                    label=score.capitalize(), color=colors[score])
        
        ax3.set_xlabel('Categories')
        ax3.set_ylabel('Counts')
        ax3.set_title('Detailed Quality Ratings by Category')
        ax3.set_xticks([x + bar_width for x in index])
        ax3.set_xticklabels(all_categories, rotation=45, ha='right')
        ax3.legend()
        
        # 分项评分饼图（按评分）
        ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        
        # 计算各评分在所有分项中的总次数
        score_totals = defaultdict(int)
        for category in all_categories:
            for score in ['high', 'medium', 'low']:
                score_totals[score] += detailed_stats[category][score]['count']
        
        # 绘制饼图
        labels = ['High', 'Medium', 'Low']
        sizes = [score_totals['high'], score_totals['medium'], score_totals['low']]
        ax4.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                colors=['#4CAF50', '#FFC107', '#F44336'])
        ax4.set_title('Aggregated Detailed Scores Distribution')
    
    plt.tight_layout()
    plt.show()

def show_quality(jsonl_file):
    quality_stats, detailed_stats, total_entries, parse_errors = analyze_quality_ratings(jsonl_file)
    
    # 打印统计结果
    print(f"Total entries processed: {total_entries + parse_errors}")
    print(f"Successfully parsed: {total_entries}")
    print(f"Parse errors: {parse_errors}")
    print("\nOverall Quality Statistics:")
    for quality, stats in quality_stats.items():
        print(f"{quality.capitalize()}: Count = {stats['count']}, Percentage = {stats['percentage']:.2f}%")
    
    # 打印分项评分统计
    if detailed_stats:
        print("\nDetailed Quality Statistics:")
        # 确保按固定顺序显示分项
        categories_order = ['Relevance', 'Logical Consistency', 'Terminology Usage', 'Factual Correctness']
        for category in categories_order:
            if category in detailed_stats:
                print(f"\nCategory: {category}")
                for score in ['high', 'medium', 'low']:
                    stats = detailed_stats[category].get(score, {'count': 0, 'percentage': 0})
                    print(f"  {score.capitalize()}: Count = {stats['count']}, Percentage = {stats['percentage']:.2f}%")
    
    # 绘制图表
    plot_quality_stats(quality_stats, detailed_stats, total_entries)
    
    