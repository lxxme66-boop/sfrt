import os
import json
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.common_utils import build_openai_client_chat

logger = logging.getLogger(__name__)

def get_document_quality_prompt():
    """文档质量评估prompt模板"""
    return """你的任务是依据以下评分规则对文本质量进行打分，并输出最终得分。评分流程如下：
1.依照每个标准依次评估文本。对每个子问题如实作答。若对某子问题答案为明确 "是"，则按标准相应加分或减分;
2.记录每个标准的累计分数，得出总分;
3.依据以下说明，将最终评估结果整理为有效的 JSON 对象。

## 评分标准：
1.标准 1：问题完整性
(1) 内容无清晰主要问题，或缺乏足够线索得出正确答案，得 0 分。
(2) 内容包含一个主要问题，且有足够线索得出正确答案，得 + 1 分。
(3) 文本体现多位作者间互动与讨论，如提出答案、评估反思答案、回应批评、修订编辑答案，得 + 1 分。

2.标准 2：问题复杂性和技术深度
(1) 内容难度为大学水平或以下，得 0 分。
(2) 内容难度为研究生水平或以上，仅领域专家能理解，得 + 1 分。
(3) 所讨论问题极具挑战性，高技能非专家花费 30 分钟上网搜索或阅读文献后，仍无法完全理解问题或给出正确答案，得 + 1 分。

3.标准 3：技术正确性和准确性
(1) 文本含显著技术错误或不准确，得 -1 分。
(2) 文本有一定技术正确性，但存在明显缺陷或遗漏（如单位错误、推导不完整），得 0 分。
(3) 文本技术正确，但有小缺陷或遗漏（如小代数错误、解释不完整），得 + 0.5 分。
(4) 文本技术高度正确，解释清晰准确（如精确定义、完整推导），得 + 0.5 分。
(5) 文本技术卓越正确，解释严格精确（如形式化证明、精确计算），得 + 1 分。

4.标准 4：思维和推理
(1) 文本无任何思维或推理迹象，得 -1 分。
(2) 文本展现一些基本思维和推理能力（如直接应用已知技术、简单分析问题），得 + 0.5 分。
(3) 文本展现一定思维和推理能力（如考虑多种解决方法、讨论不同方案权衡），得 + 0.5 分。
(4) 文本展现显著思维和推理能力（如通过多步推理链解决复杂问题、运用专业科学领域高级推理模式），得 + 1 分。
(5) 文本展现卓越思维和推理能力（如以高度创新方式解决专业领域复杂问题、结合多种推理技术对问题进行新抽象），得 + 1 分。

最终评判标准：若各项标准得分均大于零，且标准 4 得分大于等于 1 分，则该文本内容适合生成逻辑推理问题。

[文本内容的开始]
{document_content}
[文本内容的结束]

格式要求：只输出文本内容是否适合生成复杂推理问题，不输出任何别的内容。并且是否适合严格按照以下格式进行输出：
【是】或者【否】。不要输出为空，不要输出其他内容，输出是或否时，要带上【】符号进行输出。
"""

def evaluate_document_quality(document_content, chat_model):
    """评估单个文档的质量"""
    try:
        prompt = get_document_quality_prompt().format(document_content=document_content)
        messages = [
            {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
            {"role": "user", "content": prompt}
        ]
        
        response = chat_model(
            model=os.getenv("COMPLETION_MODEL", "deepseek-r1-250120"),
            messages=messages,
            temperature=0.1,
            max_tokens=100
        )
        
        result = response.choices[0].message.content.strip()
        return '【是】' in result
        
    except Exception as e:
        logger.error(f"评估文档质量时出错: {e}")
        return False

def filter_high_quality_documents(data_dir, filtered_data_dir, chat_model, max_workers=4):
    """
    筛选高质量文档
    
    Args:
        data_dir: 原始文档目录
        filtered_data_dir: 筛选后文档保存目录  
        chat_model: LLM客户端
        max_workers: 并发数量
    """
    if not os.path.exists(data_dir):
        logger.error(f"原始文档目录不存在: {data_dir}")
        return
    
    # 创建筛选后文档目录
    os.makedirs(filtered_data_dir, exist_ok=True)
    
    # 获取所有.md文件
    md_files = [f for f in os.listdir(data_dir) if f.endswith('.md')]
    logger.info(f"找到 {len(md_files)} 个文档待筛选")
    
    high_quality_count = 0
    
    def process_document(filename):
        nonlocal high_quality_count
        try:
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 评估文档质量
            is_high_quality = evaluate_document_quality(content, chat_model)
            
            if is_high_quality:
                # 复制高质量文档到筛选目录
                filtered_path = os.path.join(filtered_data_dir, filename)
                with open(filtered_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                high_quality_count += 1
                logger.info(f"✓ 高质量文档: {filename}")
                return True
            else:
                logger.info(f"✗ 低质量文档: {filename}")
                return False
                
        except Exception as e:
            logger.error(f"处理文档 {filename} 时出错: {e}")
            return False
    
    # 并发处理文档
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_document, filename) for filename in md_files]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="筛选文档"):
            future.result()
    
    logger.info(f"文档筛选完成: {high_quality_count}/{len(md_files)} 个高质量文档")
    
    # 保存筛选结果统计
    stats = {
        "total_documents": len(md_files),
        "high_quality_documents": high_quality_count,
        "quality_rate": high_quality_count / len(md_files) if md_files else 0,
        "filtered_files": os.listdir(filtered_data_dir)
    }
    
    stats_path = os.path.join(filtered_data_dir, "quality_filter_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    return filtered_data_dir