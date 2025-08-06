import os
import json
import logging
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from utils.common_utils import load_articles, get_chunkstr

logger = logging.getLogger(__name__)

def get_enhanced_question_prompt() -> str:
    """增强版问题生成prompt模板（包含思考过程）"""
    return """你是一位半导体显示技术领域的资深专家，擅长从技术文献中提炼核心知识点。你的职责是从论文中生成问题和相应的答案，问题和相应的答案对需要提供给资深的人员学习，问题和相应的答案的质量要高。请根据输入的学术论文内容，生成3个需要逻辑推理才能解答的高质量技术问题，请确保这些问题能够直接从论文中找到答案。这些问题将用于资深研究人员的专业能力评估，需满足以下要求：

## 【核心要求】
### 问题设计准则：
a) 仔细通读全文，找出涉及逻辑推理的文本部分，据此设计相关问题；
b) 问题必须基于论文中的技术原理进行设计，问题的描述必须明确清晰全面，问题中主语或名词的描述必须要精准、全面且具备通用性；
c) 问题中请不要引用文献或者文章定义的专有名词，请结合你自身半导体的显示领域的知识，将生成普适通用的问题，在不阅读论文的情况也能正常理解问题所表达的含义；
d) 问题中的名词描述不可以缩写，需要与论文中的描述一致。例如论文中提到的是"OLED材料"，问题中不能简化为"材料"。例如论文中提到的是"LTPS器件"，问题中不能简化为"器件"；
e) 不要针对于论文中的某个特定示例进行提问，问题尽量使顶尖科学家在不阅读论文的情况下也能理解和回答。且问题不能包含"书本"、"论文"、"本文"、"本实验"、"报道"、"xx等人的研究"等相关信息；
f) 保证问题的完整性，且完全不依赖论文内容，确保问题与论文完全解耦。若问题带有背景信息，一定要阐述清楚背景情况；
g) 问题要凝练简洁。

### 科学严谨性：
a) 因果链：问题需呈现完整技术逻辑链（如：机制A如何影响参数B，进而导致现象C）
b) 周密性：过程需要科学严谨，逐步思考，确保问题和对应的答案来源于论文的内容。且答案需要能在论文中完全找到详细的描述。

## 【禁止事项】
× 禁止使用"本文/本研究/本实验"等论文自指表述
× 禁止提问孤立概念（如：XX技术的定义是什么）
× 禁止超出论文技术范围的假设性问题

## 【输入】：
论文文本：  
{text}
概念列表：  
{concept}

## 【格式要求】：
用中文输出。当前阶段只设计问题，不输出答案。输出问题前必须用 </think> 结束思考后在输出问题。严格按照以下格式输出你设计的问题：
[[1]] 第1个问题
[[2]] 第2个问题
[[3]] 第3个问题"""

def get_question_validation_prompt() -> str:
    """问题质量验证prompt模板"""
    return """您是一位专家评估员，负责决定问题是否符合推理问题标准。您的评估必须结合给定文章内容和给定问题判断。

## 【评估标准】
### 因果性：
(1) 问题应展现出完整的技术逻辑链。比如，类似 "机制 A 怎样影响参数 B，最终致使现象 C 出现" 这种形式。

### 周密性：
(1) 思维过程要科学且严谨，需逐步思考。问题及对应的答案必须源于论文内容，且答案在论文中要有详细描述。

### 完整性：
(1) 问题是否全面涵盖文章相关内容的各个方面？
(2) 问题描述应简洁凝练，语义完整。
(3) 问题要与文章内容完全独立，不依赖文章也能被清晰理解，即问题需完整、自足。

[文章内容的开始]
{academic_paper}
[文章内容的结束]

[问题内容]
{academic_question}

格式要求：仅输出文本内容生成的问题是否符合标准，严格按以下格式，有且仅输出【是】或者【否】，不输出任何别的内容，不能输出为空，输出是或否时，要带上【】符号进行输出。用中文输出，严格按照以下格式进行输出：【是】或者【否】"""

def to_batch(lst: List, batch_size: int) -> List[List]:
    """将列表分批"""
    return [lst[i:i+batch_size] for i in range(0, len(lst), batch_size)]

def generate_questions_batch(
    papers_with_topics: List[Dict],
    llm_client,
    batch_size: int = 32
) -> List[Dict]:
    """
    批量生成问题
    
    使用vLLM或批量API调用来提高效率
    """
    results = []
    
    # 准备批量输入
    prompts = []
    metadata = []
    
    for item in papers_with_topics:
        paper_name = item['paper_name']
        paper_content = item['paper_content']
        concepts = item.get('concepts', [])
        
        prompt = get_enhanced_question_prompt().format(
            text=paper_content,
            concept=", ".join(concepts) if concepts else "无特定概念"
        )
        
        prompts.append(prompt)
        metadata.append({
            'paper_name': paper_name,
            'paper_content': paper_content,
            'concepts': concepts
        })
    
    # 批量推理
    if hasattr(llm_client, 'generate'):  # vLLM客户端
        from vllm import SamplingParams
        
        # 准备消息格式
        formatted_prompts = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
                {"role": "user", "content": prompt}
            ]
            # 需要tokenizer来格式化消息
            if hasattr(llm_client, 'tokenizer'):
                formatted_prompt = llm_client.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                formatted_prompts.append(formatted_prompt)
            else:
                formatted_prompts.append(prompt)
        
        sampling_params = SamplingParams(
            temperature=0.6,
            repetition_penalty=1.1,
            min_p=0,
            top_p=0.95,
            top_k=40,
            max_tokens=4096,
            stop_token_ids=llm_client.stop_token_ids if hasattr(llm_client, 'stop_token_ids') else None
        )
        
        # vLLM批量生成
        outputs = llm_client.generate(formatted_prompts, sampling_params, use_tqdm=True)
        
        for i, output in enumerate(outputs):
            question_text = output.outputs[0].text
            
            # 解析问题列表
            question_list = []
            if "</think>" in question_text:
                questions_part = question_text.split("</think>")[1].strip()
            else:
                questions_part = question_text
            
            # 提取问题
            lines = questions_part.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('[[') and ']]' in line:
                    question = line.split(']]', 1)[1].strip()
                    if question:
                        question_list.append(question)
            
            results.append({
                **metadata[i],
                'question_list': question_list[:3]  # 最多3个问题
            })
    
    else:  # 标准OpenAI客户端
        # 批量处理
        batches = to_batch(list(zip(prompts, metadata)), batch_size)
        
        for batch in tqdm(batches, desc="生成问题"):
            batch_results = []
            
            for prompt, meta in batch:
                try:
                    messages = [
                        {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
                        {"role": "user", "content": prompt}
                    ]
                    
                    response = llm_client(
                        model=os.getenv("COMPLETION_MODEL", "deepseek-r1-250120"),
                        messages=messages,
                        temperature=0.6,
                        max_tokens=4096
                    )
                    
                    question_text = response.choices[0].message.content
                    
                    # 解析问题列表
                    question_list = []
                    if "</think>" in question_text:
                        questions_part = question_text.split("</think>")[1].strip()
                    else:
                        questions_part = question_text
                    
                    # 提取问题
                    lines = questions_part.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith('[[') and ']]' in line:
                            question = line.split(']]', 1)[1].strip()
                            if question:
                                question_list.append(question)
                    
                    batch_results.append({
                        **meta,
                        'question_list': question_list[:3]
                    })
                    
                except Exception as e:
                    logger.error(f"生成问题失败: {e}")
                    batch_results.append({
                        **meta,
                        'question_list': []
                    })
            
            results.extend(batch_results)
    
    return results

def validate_questions_batch(
    questions_data: List[Dict],
    llm_client,
    batch_size: int = 32
) -> List[Dict]:
    """
    批量验证问题质量
    """
    results = []
    
    # 准备批量输入
    prompts = []
    metadata = []
    
    for item in questions_data:
        paper_content = item['paper_content']
        question = item['question']
        
        prompt = get_question_validation_prompt().format(
            academic_paper=paper_content,
            academic_question=question
        )
        
        prompts.append(prompt)
        metadata.append(item)
    
    # 批量推理
    if hasattr(llm_client, 'generate'):  # vLLM客户端
        from vllm import SamplingParams
        
        # 准备消息格式
        formatted_prompts = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
                {"role": "user", "content": prompt}
            ]
            if hasattr(llm_client, 'tokenizer'):
                formatted_prompt = llm_client.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                formatted_prompts.append(formatted_prompt)
            else:
                formatted_prompts.append(prompt)
        
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=100,
            stop_token_ids=llm_client.stop_token_ids if hasattr(llm_client, 'stop_token_ids') else None
        )
        
        # vLLM批量生成
        outputs = llm_client.generate(formatted_prompts, sampling_params, use_tqdm=True)
        
        for i, output in enumerate(outputs):
            evaluator_text = output.outputs[0].text.strip()
            is_valid = '【是】' in evaluator_text
            
            if is_valid:
                results.append(metadata[i])
    
    else:  # 标准OpenAI客户端
        # 批量处理
        batches = to_batch(list(zip(prompts, metadata)), batch_size)
        
        for batch in tqdm(batches, desc="验证问题"):
            for prompt, meta in batch:
                try:
                    messages = [
                        {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
                        {"role": "user", "content": prompt}
                    ]
                    
                    response = llm_client(
                        model=os.getenv("COMPLETION_MODEL", "deepseek-r1-250120"),
                        messages=messages,
                        temperature=0.1,
                        max_tokens=100
                    )
                    
                    evaluator_text = response.choices[0].message.content.strip()
                    is_valid = '【是】' in evaluator_text
                    
                    if is_valid:
                        results.append(meta)
                        
                except Exception as e:
                    logger.error(f"验证问题失败: {e}")
    
    return results

def convert_questionlist_to_individual(questions_data: List[Dict]) -> List[Dict]:
    """
    将问题列表展开为单个问题
    
    将包含question_list的数据展开为多个包含单个question的数据
    """
    individual_questions = []
    
    for item in questions_data:
        question_list = item.get('question_list', [])
        
        for question in question_list:
            individual_item = {
                'paper_name': item['paper_name'],
                'paper_content': item['paper_content'],
                'concepts': item.get('concepts', []),
                'question': question
            }
            individual_questions.append(individual_item)
    
    return individual_questions

def gen_enhanced_questions_with_validation_batch(
    topics_path: str,
    questions_path: str,
    validated_questions_path: str,
    llm_client,
    chunk4_path: str,
    batch_size: int = 32,
    max_workers: int = 4,
    question_output_path: Optional[str] = None,
    question_li_output_path: Optional[str] = None
):
    """
    批量生成增强版问题并进行质量验证
    
    Args:
        topics_path: 主题文件路径
        questions_path: 原始问题保存路径
        validated_questions_path: 验证后问题保存路径
        llm_client: LLM客户端
        chunk4_path: chunk4文件路径
        batch_size: 批量处理大小
        max_workers: 并发数量
        question_output_path: 问题生成中间结果路径
        question_li_output_path: 展开后问题中间结果路径
    """
    # 检查是否已存在结果
    if os.path.exists(validated_questions_path):
        logger.info(f"{validated_questions_path} 已存在，跳过...")
        return
    
    # 加载主题和chunk数据
    logger.info("加载主题和文档数据...")
    articles_topics = load_articles(topics_path)
    articles_chunks = load_articles(chunk4_path)
    
    # 准备生成问题的数据
    papers_with_topics = []
    for article_name, topics_list in articles_topics.items():
        if article_name not in articles_chunks:
            continue
        
        # 获取文档内容
        chunks = articles_chunks[article_name]
        if chunks:
            document_content = get_chunkstr(chunks[0].get("oracle_chunks", []))
        else:
            continue
        
        # 收集概念
        all_concepts = []
        for topic in topics_list:
            if 'concepts' in topic:
                all_concepts.extend(topic['concepts'])
        
        papers_with_topics.append({
            'paper_name': article_name,
            'paper_content': document_content,
            'concepts': list(set(all_concepts))  # 去重
        })
    
    logger.info(f"准备为 {len(papers_with_topics)} 个文档生成问题")
    
    # 第1步：批量生成问题
    logger.info("批量生成问题...")
    generated_questions = generate_questions_batch(
        papers_with_topics,
        llm_client,
        batch_size
    )
    
    # 保存生成的问题（中间结果）
    if question_output_path:
        os.makedirs(os.path.dirname(question_output_path), exist_ok=True)
        with open(question_output_path, 'w', encoding='utf-8') as f:
            for item in generated_questions:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"问题生成结果已保存到: {question_output_path}")
    
    # 第2步：展开问题列表
    logger.info("展开问题列表...")
    individual_questions = convert_questionlist_to_individual(generated_questions)
    
    # 保存展开后的问题（中间结果）
    if question_li_output_path:
        os.makedirs(os.path.dirname(question_li_output_path), exist_ok=True)
        with open(question_li_output_path, 'w', encoding='utf-8') as f:
            for item in individual_questions:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"展开后的问题已保存到: {question_li_output_path}")
    
    logger.info(f"生成了 {len(individual_questions)} 个问题")
    
    # 第3步：批量验证问题质量
    logger.info("批量验证问题质量...")
    validated_questions = validate_questions_batch(
        individual_questions,
        llm_client,
        batch_size
    )
    
    logger.info(f"验证通过 {len(validated_questions)}/{len(individual_questions)} 个问题")
    
    # 第4步：按文章组织验证后的问题
    validated_by_article = {}
    for item in validated_questions:
        article_name = item['paper_name']
        if article_name not in validated_by_article:
            validated_by_article[article_name] = []
        
        # 转换为原始格式
        question_item = {
            'concepts': item['concepts'],
            'question': item['question'],
            'oracle_chunks': articles_chunks[article_name][0].get("oracle_chunks", []) if article_name in articles_chunks else []
        }
        validated_by_article[article_name].append(question_item)
    
    # 保存最终结果
    os.makedirs(os.path.dirname(validated_questions_path), exist_ok=True)
    with open(validated_questions_path, 'w', encoding='utf-8') as f:
        json.dump(validated_by_article, f, ensure_ascii=False, indent=4)
    
    logger.info(f"验证后的问题已保存到: {validated_questions_path}")
    
    # 保存统计信息
    stats = {
        "total_documents": len(papers_with_topics),
        "total_generated_questions": len(individual_questions),
        "validated_questions": len(validated_questions),
        "validation_rate": len(validated_questions) / len(individual_questions) if individual_questions else 0,
        "batch_size": batch_size,
        "documents_with_questions": len(validated_by_article)
    }
    
    stats_path = os.path.join(os.path.dirname(validated_questions_path), "question_generation_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)