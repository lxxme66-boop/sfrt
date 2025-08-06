import pandas as pd
import json

quality_evaluation_prompt = {
    "v1": {
        "role": "system",
        "content": """
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
   - 专利技术：技术方案是否规避近期专利？（自动核对2020-2024专利数据库）

3. 领域深度（权重20%）
   - 缺陷机理：是否分析根本原因？（例：亮暗点应关联电致迁移机制）
   - 技术趋势：是否覆盖最新发展？（例：需提及Micro LED巨量转移技术）
   - 工艺瓶颈：是否识别关键限制？（例：指出QD-OLED的喷墨打印精度瓶颈）

4. 应用价值（权重15%）
   - 工程可行性：方案是否具备量产实施条件？
   - 成本优化：是否量化成本效益？（例：应计算采用MMG技术的材料节省）
   - 良率提升路径：是否提供可验证的改善方案？

### 领域关键点（自动核对）
| 要素类型       | 典型内容示例                  |
|----------------|------------------------------|
| 核心材料       | 氧化物TFT, QD材料, LTPO      |
| 工艺痛点       | 蒸镀均匀性, 水氧阻隔率       |
| 典型缺陷       | Mura, 亮点/暗点, 热应力翘曲   |

### 验证方法
1. 参数边界检查：对关键参数进行物理极限校验（例：若声称PPI>1500需验证光学混色距离）
2. 时效性验证：技术指标是否被近3年文献更新（自动索引IEEE期刊数据库）
3. 成本分解：对降本承诺进行材料/设备/良率因子分解

### 输出格式要求（JSON）
{
    "quality_rating": {
        "overall": "high/medium/low",
        "detailed_scores": {
            "reasoning_chain": {"score": int, "issues": [str]},
            "technical_accuracy": {"score": int, "validated": bool},
            "domain_depth": {"score": int, "benchmark": str}
        }
    },
    "improvement_suggestions": [str]
}

### 待评估样本
问题: {question_text}
文本块：{chunk_text}
思维链: {reasoning_chain}
答案: {answer_text}
"""
    },
    "v2": {
        "role": "system",
        "content": """你是一名资深显示技术领域专家。请严格评估以下显示技术相关的问答对是否适合用于监督微调（SFT）的数据集构建。评估需基于以下四个核心维度：

1.  **回答相关性 (Relevance)**：回答是否精准聚焦问题核心？是否存在答非所问、偏离主题或遗漏关键点？
2.  **逻辑一致性 (Logical Consistency)**：回答的推理过程是否清晰、连贯、无矛盾？是否存在逻辑跳跃、断裂或自相矛盾？
3.  **术语使用 (Terminology Usage)**：专业术语的使用是否准确、恰当、完整？是否存在术语误用、滥用、缺失或概念性错误？
4.  **事实正确性 (Factual Correctness)**：回答中的技术细节、参数、原理、行业现状等是否符合已知事实和行业共识？是否存在事实性错误或过时信息？

**总体质量评分标准：**
*   `low`：**存在严重缺陷**（如明显事实错误、完全偏离主题、逻辑混乱、关键术语错误），**不适合**用于SFT。
*   `medium`：**存在轻微问题或可优化项**（如部分表述不清、个别术语不严谨、次要逻辑不完美、相关性略有不足），需修改后方可考虑使用。
*   `high`：**无明显错误**，内容**准确、专业、逻辑清晰、紧扣主题**，**适合**直接用于SFT。

**你的任务：**
1.  对每个维度进行独立评分 (`high`/`medium`/`low`)。
2.  给出基于四个维度的**总体质量评分** (`high`/`medium`/`low`)。
3.  对于评分非`high`的维度，**必须具体指出**存在的问题及其**类型**（例如：“术语误用：将‘OLED’错误称为‘LED’”；“事实错误：声称当前主流Mini-LED背光分区数普遍超过5000区”）。
4.  基于你的专业知识和评估结果，**提供具体、可操作的改进建议**，以提升该问答对的质量。

**输出格式要求（严格遵循JSON）：**
{
    "quality_rating": {
        "overall": "high/medium/low", // 总体质量评分
        "detailed_scores": {
            "Relevance": {"score": "high/medium/low", "issues": ["具体问题描述1", "具体问题描述2", ...]}, // 如无问题，issues为空数组[]
            "Logical Consistency": {"score": "high/medium/low", "issues": [...]},
            "Terminology Usage": {"score": "high/medium/low", "issues": [...]},
            "Factual Correctness": {"score": "high/medium/low", "issues": [...]}
        }
    },
    "improvement_suggestions": ["具体建议1", "具体建议2", ...] // 即使总体是high，也可提供优化建议
}
### 待评估样本
问题: {question_text}
文本块：{chunk_text}
思维链: {reasoning_chain}
答案: {answer_text}
"""
    }
}

def excel_to_jsonl(input_excel_path, output_jsonl_path, prompt_version):
    # 读取Excel文件
    df = pd.read_excel(input_excel_path)
    
    # 确保所需的列存在
    required_columns = ['问题', '推理', '回答']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            question = row["问题"]
            chain = row["推理"]
            answer = row["回答"]
            quality_message = quality_evaluation_prompt[prompt_version].copy()
            quality_message["content"] = quality_message["content"].replace("{question_text}", question) \
                .replace("{reasoning_chain}", chain) \
                .replace("{answer_text}", answer)

            # 构建请求体
            request_body = {
                "custom_id": f"request-{idx + 1}",
                "body": {
                    "messages": [
                        quality_message
                    ]
                }
            }
            
            # 写入jsonl文件
            f.write(json.dumps(request_body, ensure_ascii=False) + '\n')
        print(f"转换完成，结果已保存到 {output_jsonl_path}")

import json
from typing import List, Dict, Any
def syndatas_to_jsonl(syndatas_path: str, output_jsonl_path: str, prompt_version: str) -> None:
    """
    将syndatas_path文件转换为符合要求的jsonl格式文件
    
    参数:
        syndatas_path: 输入syndatas文件路径
        output_jsonl_path: 输出jsonl文件路径
        prompt_version: 提示词版本标识
    """
    # 读取syndatas文件
    with open(syndatas_path, 'r', encoding='utf-8') as f:
        syndatas = json.load(f)
    
    # 确保prompt_version有效
    if prompt_version not in quality_evaluation_prompt:
        raise ValueError(f"无效的prompt_version: {prompt_version}")
    
    # 处理每条数据并写入jsonl文件
    with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
        for idx, item in enumerate(syndatas):
            # 构建请求体
            request_body = {
                "custom_id": f"request-{idx + 1}",
                "body": {
                    "messages": [
                        {
                            "role": quality_evaluation_prompt[prompt_version]["role"],
                            "content": quality_evaluation_prompt[prompt_version]["content"]
                                .replace("{question_text}", item["question"])
                                .replace("{reasoning_chain}", item["reasoning_content"])
                                .replace("{answer_text}", item["content"])
                        }
                    ]
                }
            }
            
            # 写入jsonl文件
            f_out.write(json.dumps(request_body, ensure_ascii=False) + '\n')
    
    print(f"转换完成，结果已保存到 {output_jsonl_path}")

import json
import re
import matplotlib.pyplot as plt
from collections import defaultdict

def extract_overall_quality(content_str):
    """
    使用正则表达式从content字符串中提取overall评分
    避免复杂的JSON解析，直接匹配"overall": "..."模式
    """
    # 匹配 "overall": "value" 的模式，不区分大小写
    match = re.search(r'"overall"\s*:\s*"([^"]+)"', content_str, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None

def analyze_quality_ratings(jsonl_file):
    # 初始化计数器
    quality_counts = defaultdict(int)
    total_entries = 0
    parse_errors = 0
    
    # 读取jsonl文件并统计质量评级
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # 获取content字符串
                content_str = data['response']['body']['choices'][0]['message']['content']
                
                # 使用正则表达式提取overall评分
                quality = extract_overall_quality(content_str)
                
                if quality in ['high', 'medium', 'low']:
                    quality_counts[quality] += 1
                    total_entries += 1
                else:
                    parse_errors += 1
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                parse_errors += 1
                continue
    
    # 计算百分比
    quality_stats = {}
    for quality, count in quality_counts.items():
        percentage = (count / total_entries) * 100 if total_entries > 0 else 0
        quality_stats[quality] = {'count': count, 'percentage': percentage}
    
    return quality_stats, total_entries, parse_errors

def plot_quality_stats(quality_stats, total_entries):
    if total_entries == 0:
        print("No valid data to plot.")
        return
    
    # 准备数据
    labels = ['High', 'Medium', 'Low']
    counts = [quality_stats.get('high', {}).get('count', 0),
              quality_stats.get('medium', {}).get('count', 0),
              quality_stats.get('low', {}).get('count', 0)]
    percentages = [quality_stats.get('high', {}).get('percentage', 0),
                   quality_stats.get('medium', {}).get('percentage', 0),
                   quality_stats.get('low', {}).get('percentage', 0)]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 饼图
    ax1.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90,
            colors=['#4CAF50', '#FFC107', '#F44336'])
    ax1.set_title('Quality Rating Distribution')
    
    # 柱状图
    bars = ax2.bar(labels, counts, color=['#4CAF50', '#FFC107', '#F44336'])
    ax2.set_title('Quality Rating Counts')
    ax2.set_ylabel('Count')
    ax2.set_xlabel('Quality Rating')
    
    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}\n({height/total_entries*100:.1f}%)',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def show_quality(jsonl_file):
    quality_stats, total_entries, parse_errors = analyze_quality_ratings(jsonl_file)
    
    # 打印统计结果
    print(f"Total entries processed: {total_entries + parse_errors}")
    print(f"Successfully parsed: {total_entries}")
    print(f"Parse errors: {parse_errors}")
    print("\nQuality Statistics:")
    for quality, stats in quality_stats.items():
        print(f"{quality.capitalize()}: Count = {stats['count']}, Percentage = {stats['percentage']:.2f}%")
    
    # 绘制图表
    plot_quality_stats(quality_stats, total_entries)






