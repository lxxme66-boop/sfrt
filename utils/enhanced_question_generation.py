import os
import json
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.common_utils import load_articles, get_chunkstr, build_openai_client_chat

logger = logging.getLogger(__name__)

def get_enhanced_question_prompt():
    """增强版问题生成prompt模板"""
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
用中文输出。当前阶段只设计问题，不输出答案。严格按照以下格式输出你设计的问题：
[
    {{
        "concepts": ["概念1", "概念2", "仅填入2-3个概念"],  
        "question": "仅填入问题"  
    }},  
    {{
        "concepts": ["概念1", "概念2", "仅填入2-3个概念"],  
        "question": "仅填入问题"  
    }},
    {{
        "concepts": ["概念1", "概念2", "仅填入2-3个概念"],  
        "question": "仅填入问题"  
    }}
]
"""

def get_question_validation_prompt():
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
{document_content}
[文章内容的结束]

[问题内容]
{question_content}

格式要求：仅输出问题是否符合标准，严格按以下格式，有且仅输出【是】或者【否】，不输出任何别的内容，不能输出为空，输出是或否时，要带上【】符号进行输出。
"""

def generate_enhanced_questions(topics, chat_model):
    """使用增强版prompt生成问题"""
    try:
        chunkstr = get_chunkstr(topics["oracle_chunks"])
        prompt = get_enhanced_question_prompt().format(
            text=chunkstr, 
            concept=topics["topics"]
        )
        
        messages = [
            {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
            {"role": "user", "content": prompt}
        ]
        
        response = chat_model(
            model=os.getenv("COMPLETION_MODEL", "deepseek-r1-250120"),
            messages=messages,
            temperature=0.7,
            max_tokens=2048
        )
        
        result = response.choices[0].message.content.strip()
        
        # 尝试解析JSON格式的问题列表
        try:
            questions = json.loads(result)
            if isinstance(questions, list):
                return questions
        except json.JSONDecodeError:
            logger.warning("无法解析问题JSON格式，尝试文本解析")
        
        # 如果JSON解析失败，尝试文本解析
        questions = []
        lines = result.split('\n')
        for line in lines:
            if line.strip().startswith('[[') and ']]' in line:
                question_text = line.split(']]', 1)[1].strip()
                if question_text:
                    questions.append({
                        "concepts": [],
                        "question": question_text
                    })
        
        return questions[:3]  # 最多返回3个问题
        
    except Exception as e:
        logger.error(f"生成问题时出错: {e}")
        return []

def validate_question_quality(question, document_content, chat_model):
    """验证问题质量"""
    try:
        prompt = get_question_validation_prompt().format(
            document_content=document_content,
            question_content=question["question"]
        )
        
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
        logger.error(f"验证问题质量时出错: {e}")
        return False

def save_enhanced_questions(questions, article_name, filename):
    """保存增强版问题"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    if os.path.exists(filename):
        with open(filename, 'r', encoding="utf-8") as f:
            existing = json.load(f)
        if article_name in existing:
            existing[article_name].extend(questions)
        else:
            existing[article_name] = questions
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=4)
    else:
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump({article_name: questions}, f, ensure_ascii=False, indent=4)
    
    logger.info(f"增强版问题已保存到 {filename}")

def gen_enhanced_questions_with_validation(topics_path, questions_path, validated_questions_path, chat_model, chunk4_path, max_workers=4):
    """
    生成增强版问题并进行质量验证
    
    Args:
        topics_path: 主题文件路径
        questions_path: 原始问题保存路径
        validated_questions_path: 验证后问题保存路径
        chat_model: LLM客户端
        chunk4_path: chunk4文件路径
        max_workers: 并发数量
    """
    if os.path.exists(validated_questions_path):
        logger.info(f"{validated_questions_path} 已存在，跳过...")
        return
    
    # 加载主题和chunk数据
    articles_topics = load_articles(topics_path)
    articles_chunks = load_articles(chunk4_path)
    
    total_questions = 0
    validated_questions = 0
    
    for article_name, topics_list in articles_topics.items():
        logger.info(f"处理文章: {article_name}")
        
        article_validated_questions = []
        
        for topics in tqdm(topics_list, desc=f"生成问题-{article_name}"):
            # 生成增强版问题
            questions = generate_enhanced_questions(topics, chat_model)
            total_questions += len(questions)
            
            # 获取文档内容用于验证
            document_content = get_chunkstr(topics["oracle_chunks"])
            
            # 验证问题质量
            for question in questions:
                if validate_question_quality(question, document_content, chat_model):
                    # 添加必要的元数据
                    enhanced_question = {
                        **question,
                        **topics
                    }
                    article_validated_questions.append(enhanced_question)
                    validated_questions += 1
                    logger.debug(f"✓ 验证通过: {question['question'][:50]}...")
                else:
                    logger.debug(f"✗ 验证失败: {question['question'][:50]}...")
        
        # 保存验证后的问题
        if article_validated_questions:
            save_enhanced_questions(article_validated_questions, article_name, validated_questions_path)
    
    logger.info(f"问题生成和验证完成: {validated_questions}/{total_questions} 个问题通过验证")
    
    # 保存统计信息
    stats = {
        "total_generated_questions": total_questions,
        "validated_questions": validated_questions,
        "validation_rate": validated_questions / total_questions if total_questions > 0 else 0
    }
    
    stats_path = os.path.join(os.path.dirname(validated_questions_path), "question_validation_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)