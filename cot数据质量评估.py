# -*- coding: utf-8 -*-
# @Time : 2025/5/20 10:40 
# @Author : dumingyu
# @File : cot数据质量评估.py
# @Software: PyCharm


import os
import pandas as pd
from openai import OpenAI
import datetime
import json
import multiprocessing
import logging
from tqdm import tqdm

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

time_label = str(datetime.datetime.today())[:10].replace('-', '')


# 初始化客户端
client = OpenAI(
    base_url="http://8.130.143.102:81/v1",  # vLLM API 服务器的地址
    api_key="EMPTY"  # 如果没有设置 API 密钥，可以使用任意字符串
)




quality_evaluation_prompt_v1 = {
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
思维链: {reasoning_chain}
答案: {answer_text}
"""
}


quality_evaluation_prompt_v2 = {
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
思维链: {reasoning_chain}
答案: {answer_text}
"""
}


# 发送单个请求
def evaluate_qa_quality(question, chain, answer):
    try:
        prompt = quality_evaluation_prompt.copy()
        prompt["content"] = prompt["content"].replace("{question_text}", question) \
            .replace("{reasoning_chain}", chain) \
            .replace("{answer_text}", answer)

        response = client.chat.completions.create(
            model="qwen3-235b",  # 与 --served-model-name 参数指定的名称一致
            messages=[prompt],
            temperature=0.1
        )
        # 打印模型的响应
        return response.choices[0].message.content
    except Exception as e:
        return f"请求失败: {str(e)}"


# ================ 多进程工作函数 ================
def process_row(args):
    """
    处理单个数据行的质量评估任务
    """
    row_idx, row_data, server_config = args
    question, chain, answer = row_data['问题'], row_data['思维过程'], row_data['答案']

    # 1. 初始化客户端（每个进程单独创建）
    client = OpenAI(
        base_url=server_config["base_url"],
        api_key=server_config["api_key"]
    )

    try:
        # 2. 构建评估提示
        prompt = quality_evaluation_prompt.copy()
        prompt["content"] = prompt["content"].replace("{question_text}", question) \
            .replace("{reasoning_chain}", chain) \
            .replace("{answer_text}", answer)

        # 3. 发送请求
        response = client.chat.completions.create(
            model=server_config["model_name"],
            messages=[prompt],
            timeout=90  # 设置超时防止进程卡住
        )


        # 4. 解析响应
        result_text = response.choices[0].message.content
        if '```json' in result_text:
            result = json.loads(result_text.split("```json")[-1].strip().replace("```", "").strip())
        else:
            result = json.loads(result_text.split("</think>")[-1].strip().replace("```", "").strip())
        # 5. 更新行数据
        row_data['quality_rating'] = result['quality_rating']['overall']
        row_data['detailed_scores'] = str(result['quality_rating']['detailed_scores'])
        return (row_idx, row_data)

    except Exception as e:
        logger.error(f"处理行 {row_idx} 时出错: {str(e)}")
        # 保留原始行数据并标记错误
        row_data['quality_rating'] = "ERROR"
        row_data['detailed_scores'] = str(e)
        return (row_idx, row_data)





if __name__ == '__main__':

# 0. 服务器配置（避免重复传递）
server_config = {
    "base_url": "http://8.130.143.102:81/v1",
    "api_key": "EMPTY",
    "model_name": "qwen3-235b"
}

# 1. 加载源数据
file_name = "/mnt/data/LLM/dmy/华星数据&智慧研发数据&专家系统数据.xlsx"
data = pd.read_excel(file_name)
logger.info(f"加载数据完成，共 {data.shape[0]} 行")

# 2. 准备多进程处理
num_workers = 20
logger.info(f"启用 {num_workers} 个工作进程")


data = data.iloc[10848:,]


# 3. 创建进程池
with multiprocessing.Pool(processes=num_workers) as pool:
    # 准备任务参数 [(行索引, 行数据, 服务器配置), ...]
    task_args = [(idx, row.copy(), server_config) for idx, row in data.iterrows()]

    # 4. 并行处理并收集结果
    results = []
    for result in tqdm(pool.imap(process_row, task_args), total=len(task_args)):
        results.append(result)

# 5. 按原始顺序更新数据
logger.info("开始更新结果数据")
for row_idx, updated_row in results:
    data.loc[row_idx,'quality_rating'] = updated_row['quality_rating']
    data.loc[row_idx, 'detailed_scores'] = updated_row['detailed_scores']

# 6. 保存结果
save_dir = '/mnt/data/LLM/dmy/CoT数据生成/'
save_name = 'sft_data_华星数据&智慧研发数据&专家系统数据_judge_0618.xlsx'
save_path = os.path.join(save_dir, save_name)

with pd.ExcelWriter(save_path) as writer:
    data.to_excel(writer, index=False)
logger.info(f"结果已保存到: {save_path}")

