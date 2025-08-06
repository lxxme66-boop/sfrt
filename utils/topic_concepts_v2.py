import os 
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils.common_utils import load_articles, get_chunkstr, get_chunk4
from utils.retrieve_nodes import rerank_chunks
import json
import json5
import re
import random
from typing import Any
from jsonschema import validate, ValidationError
import logging
logging.basicConfig(filename='failed_responses.log', level=logging.ERROR)

def save_chunk4(chunk4_list, article_name, filename):
    # 判断 filename 是否存在，如果存在则追加写入，否则创建新文件
    if os.path.exists(filename):
        with open(filename, 'r', encoding="utf-8") as f:
            existing = json.load(f)
        # 检查 article_name 是否已经存在于 questions 中
        if article_name in existing:
            existing[article_name].extend(chunk4_list)
        else:
            existing[article_name] = chunk4_list
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=4)
    else:
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump({article_name: chunk4_list}, f, ensure_ascii=False, indent=4)
    print(f"Chunk4 saved to {filename}")
# 1
def trans_chunk4(chunks_path, chunk4_path):
    if os.path.exists(chunk4_path):
        print(f"{chunk4_path} exists. Skipping...")
        return 
    articles_chunks = load_articles(chunks_path)
    for a_name, a_chunks in articles_chunks.items():
        chunk4_list = []
        for i in range(0, len(a_chunks), int(os.getenv("CHUNK_NUM"))):
            chunk4 = get_chunk4(i, a_chunks)
            if chunk4:
                chunk4_list.append(chunk4)
        save_chunk4(chunk4_list, a_name, chunk4_path)
        print(f"done {a_name} chunk4.")
        

prompt_topics = {
    "synthllm": """Here is an article crawl from the web, which our classifier has identified as having significant educational value for
        students learning math.
        Your task is to analyze this article and extract educational materials, specifically focusing on topics and key
        concepts that can enhance students’ understanding of mathematics and improve their problem-solving skills.
        Pay special attention to uncommon but important mathematical concepts that are crucial for a deeper understanding.
        ## Tasks
        1. **Determine Educational Level:**
        - Identify the appropriate educational level for the article based on its content. Choose from the
        following options:
        - Primary School
        - Middle School
        - High School
        - College
        - Graduate School
        - Competition
        - Other
        2. **Identify Subject Area:**
        - Specify the primary subject area of mathematics to which the article belongs (e.g., Calculus,
        Geometry, Algebra, etc.).
        3. **Extract Topics and Key Concepts:**
        - **Topics:**
        - List **1 to 5** main topics covered in the article.
        - Use terms commonly recognized in academia or industry.
        - **Key Concepts:**
        - For each identified topic, list **5 to 20** related key concepts.
        - Ensure these concepts are clearly articulated using standard academic or industry terms.
        ## Guidelines:
        - **Terminology:** Use precise and widely recognized academic or industry terminology for subjects, topics, and
        key concepts to maintain consistency and clarity.
        - **Educational Level Selection:** If appropriate, restrict the educational level to one of the following: "Primary
        School", "Middle School", "High School", "College", "Graduate School", or "Competition" to ensure accurate
        categorization.
        ## Text
        {{ text }}
        ## Output Format
        <level>Educational Level</level>
        <subject>Subject Area</subject>
        <topic> Topics:
        1. topic 1
        2. topic 2
        </topic>
        <key_concept>
        Key Concepts:
        1. topic 1:
        1.1. key concept
        1.2. key concept
        ...
        2. topic 2:
        2.1. key concept
        ... ...
        </key_concept>
        ## Output""",
    "deepseek": """以下是一篇研究论文。  
        您的任务是分析这篇文章并提取教学材料，特别关注能够增强对目标领域的理解并提高其问题解决能力的主题和关键概念。  
        请重点关注那些对深入理解目标领域至关重要但不常见的重要领域概念。  

        ## 任务：
        1. 确定学科领域：
        明确文章所属的主要学科领域。  
        2. 提取主题与关键概念：
        主题：  
            列出文章中涵盖的 1至3个 主要主题。  
            使用学术界或行业中公认的术语。  
        关键概念：  
            针对每个已确定的主题，列出 3至10个 相关关键概念。  
            确保这些概念使用标准的学术或行业术语清晰表述。  

        ## 指南：
        术语使用： 使用精确且广泛认可的学术或行业术语来描述学科、主题和关键概念，以确保一致性和清晰性。  

        ## 论文内容
        {{ text }}
        ## 输出格式
        <subject>学科领域</subject>
        <topic> 话题:
        1. 话题1
        2. 话题2
        </topic>
        <key_concept>
        关键概念:
        1. 话题1:
        1.1. 关键概念1
        1.2. 关键概念2
        ...
        2. 话题2:
        2.1. 关键概念1
        ... ...
        </key_concept>
        ## 输出""",
    "deepseek-v2": """以下是一篇研究论文。  
        您的任务是分析这篇文章并提取教学材料，特别关注能够增强对目标领域的理解并提高其问题解决能力的主题和关键概念。  
        请重点关注那些对深入理解目标领域至关重要但不常见的重要领域概念。  

        ## 任务：
        1. 确定学科领域：
        明确文章所属的主要学科领域。  
        2. 提取主题与关键概念：
        主题：  
            列出文章中涵盖的 1至3个 主要主题。  
            使用学术界或行业中公认的术语。  
        关键概念：  
            针对每个已确定的主题，列出 3至10个 相关关键概念。  
            确保这些概念使用标准的学术或行业术语清晰表述。  

        ## 指南：
        术语使用： 使用精确且广泛认可的学术或行业术语来描述学科、主题和关键概念，以确保一致性和清晰性。  

        ## 论文内容
        {{ text }}
        ## 输出格式
        <subject>学科领域</subject>
        <topic> 话题:
        1. 话题1
        2. 话题2
        </topic>
        <key_concept>
        关键概念:
        1. 话题1:
        1.1. 关键概念1
        1.2. 关键概念2
        ...
        2. 话题2:
        2.1. 关键概念1
        ... ...
        </key_concept>
        ## 输出"""
}
def save_topics(topics, article_name, topics_path):
    filename = topics_path
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # 自动创建目录
    # 判断 filename 是否存在，如果存在则追加写入，否则创建新文件
    if os.path.exists(filename):
        with open(filename, 'r', encoding="utf-8") as f:
            existing = json.load(f)
        # 检查 article_name 是否已经存在于 questions 中
        if article_name in existing:
            existing[article_name].append(topics)
        else:
            existing[article_name] = [topics]
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=4)
    else:
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump({article_name: [topics]}, f, ensure_ascii=False, indent=4)
    print(f"Topics saved to {filename}")

def gen_topic_prompt(chunk4: list[dict]) -> list[str]:
    """
    Encode multiple prompt instructions into a single string for the general case (`pdf`, `json`, or `txt`).
    """
    
    messages = []
    chunkstr = get_chunkstr(chunk4)
    prompt = prompt_topics[os.getenv("PROMPT_KEY")].replace("{{ text }}", chunkstr)
    messages.append({"role": "system", "content": "You are a helpful question answerer who can provide an answer given a question and relevant context."})
    messages.append({"role": "user", "content": prompt})
    return messages

def generate_topics(chat_completer: Any, chunk4: list[dict]) -> str | None:
    """
    Generates the label / answer to `question` using `context` and deepseek.
    """
    messages = gen_topic_prompt(chunk4)
    response = chat_completer(
        model=os.getenv("GENERATION_MODEL"),
        messages=messages,
        n=1,
        temperature=0,
        max_tokens=2048,
    )
    topics_concepts = response.choices[0].message.content
    topics = {
        "topics": topics_concepts,
        "oracle_chunks": chunk4,
    }
    return topics
# 2
def gen_topics(chunk4_path, topics_path, chat_model):
    if os.path.exists(topics_path):
        print(f"{topics_path} exists. Skipping...")
        return 
    articles_chunk4 = load_articles(chunk4_path)
    for a_name, a_chunk4 in articles_chunk4.items():
        futures = []
        num_chunks = len(a_chunk4)
        with tqdm(total=num_chunks, desc="Topicing", unit="file") as pbar:
            with ThreadPoolExecutor(max_workers=2) as executor:
                for chunk4 in a_chunk4:
                    futures.append(executor.submit(generate_topics, chat_model, chunk4))
                for future in as_completed(futures):
                    topics = future.result()
                    pbar.update(1)
                    save_topics(topics, a_name, topics_path)
                print(f"done {a_name} topics.")

# 2.2
from typing import List, Dict, Any
# def get_chunkstr(chunk4: List[Dict]) -> str:
#     """将chunk4列表中的文本内容拼接成一个字符串"""
#     return "\n".join([chunk["text"] for chunk in chunk4])

def gen_topic_prompt(chunk4: List[Dict]) -> List[Dict]:
    """
    生成主题提取的prompt消息
    """
    messages = []
    chunkstr = get_chunkstr(chunk4)
    prompt = prompt_topics[os.getenv("PROMPT_KEY")].replace("{{ text }}", chunkstr)
    
    messages.append({"role": "system", "content": "You are a helpful question answerer who can provide an answer given a question and relevant context."})
    messages.append({"role": "user", "content": prompt})
    return messages

def chunk4_to_jsonl(chunk4_path: str, output_jsonl_path: str):
    """
    将chunk4数据转换为jsonl格式的批量请求文件
    
    参数:
        chunk4_path: 包含chunk4数据的JSON文件路径
        output_jsonl_path: 输出的jsonl文件路径
        prompt_key: 使用的prompt模板键名 (默认为"deepseek")
    """
    # 加载chunk4数据
    with open(chunk4_path, 'r', encoding='utf-8') as f:
        articles_chunk4 = json.load(f)
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        request_id = 1
        
        # 遍历每篇文章及其chunk4数据
        for article_name, chunk4_list in articles_chunk4.items():
            for chunk4 in chunk4_list:
                # 生成prompt消息
                messages = gen_topic_prompt(chunk4)
                
                # 构建请求体
                request_body = {
                    "custom_id": f"request-{request_id}",
                    "body": {
                        "messages": messages,
                        # "max_tokens": 2048,
                        # "top_p": 1,
                        # "temperature": 0
                    }
                }
                
                # 写入jsonl文件
                f.write(json.dumps(request_body, ensure_ascii=False) + '\n')
                request_id += 1
    
    print(f"转换完成，结果已保存到 {output_jsonl_path}")

import os
import json
from typing import Dict, Any, List
from tqdm import tqdm

def process_response_file(response_path: str, chunk4_path: str, topics_path: str):
    """
    处理响应结果文件，并将结果保存到topics_path
    
    参数:
        response_path: 响应结果文件路径
        chunk4_path: 原始chunk4数据文件路径
        topics_path: 结果保存路径
    """
    # 加载响应结果数据
    with open(response_path, 'r', encoding='utf-8') as f:
        responses = [json.loads(line) for line in f]
    
    # 加载原始chunk4数据
    with open(chunk4_path, 'r', encoding='utf-8') as f:
        articles_chunk4 = json.load(f)
    
    # 创建结果目录（如果不存在）
    os.makedirs(os.path.dirname(topics_path), exist_ok=True)
    
    # 初始化结果字典
    results = {}
    if os.path.exists(topics_path):
        with open(topics_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    
    # 构建custom_id到article_name和chunk4的映射
    id_mapping = {}
    request_id = 1
    for article_name, chunk4_list in articles_chunk4.items():
        for chunk4 in chunk4_list:
            id_mapping[f"request-{request_id}"] = (article_name, chunk4)
            request_id += 1
    
    # 处理每个响应
    for response in tqdm(responses, desc="Processing responses"):
        custom_id = response.get("custom_id")
        if not custom_id or custom_id not in id_mapping:
            continue
        
        article_name, chunk4 = id_mapping[custom_id]
        
        # 提取响应内容
        if response.get("error") is not None:
            print(f"Error in response {custom_id}: {response['error']}")
            continue
        
        try:
            response_body = response.get("response", {}).get("body", {})
            choices = response_body.get("choices", [])
            if not choices:
                continue
            
            content = choices[0].get("message", {}).get("content", "")
            
            # 构建结果对象
            topics = {
                "topics": content,
                "oracle_chunks": chunk4
            }
            
            # 保存结果
            if article_name in results:
                results[article_name].append(topics)
            else:
                results[article_name] = [topics]
                
        except Exception as e:
            print(f"Error processing response {custom_id}: {str(e)}")
            continue
    
    # 保存最终结果
    with open(topics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"处理完成，结果已保存到 {topics_path}")


prompt_questions = {
    "synthllm": """As a senior **math** instructor, your task is to create **diverse and challenging computation-based math
        questions**. These questions should demonstrate the application of the provided topics and key concepts while
        enhancing students’ reasoning and critical-thinking skills. Ensure that questions are **non-redundant**, precise,
        and engaging.
        ### Guidelines for Creating Diverse and Challenging Computation-based Questions:
        1. **Concept Selection**:
        - Randomly select **up to 2-3 distinct key concepts** from the provided list for each question.
        - Ensure **broad coverage** of the provided concepts across the generated questions, avoiding over-reliance
        on a limited subset of concepts.
        - Avoid repeating the same **concept combinations** or **computational approach** across questions.
        2. **Diversity and Challenge**:
        - Ensure that each question explores **different combinations of key concepts** and is **sufficiently
        challenging** (e.g., requiring multi-step computations, integrating real-world scenarios, involving abstract or
        advanced reasoning.).
        3. **Clarity and Precision**:
        - Verify that the questions are **logically sound**.
        - Use precise and unambiguous language.
        - Write all mathematical expressions or formulas in LaTeX for clarity.
        - Clearly state all assumptions or conditions.
        4. **Reference Material**:
        - Use the provided **reference article** as a source of inspiration for generating **unique, diverse, and
        challenging questions**.
        - The reference material is intended to:
        - Supplement the concept list by introducing **novel perspectives**, **contexts**, or **applications**.
        - Help create questions that are **more complex, realistic, or uncommon** in traditional teaching scenarios.
        - Serve as a resource to craft **real-world scenarios** or **abstract extensions** beyond the given concepts.
        5. **Output Diversity**:
        - Create between **1 to 5 questions**.
        - Ensure each question is unique in **structure**, **approach**, and **concept usage**.
        - Minimize the use of **sub-questions**, unless they are essential to the problem’s complexity.
        - The answer should either be exact, or if not possible, then the question should clearly say the answer is only
        expected to be approximately correct.
        ### Inputs:
        - **Article**:
        {{ text }}
        - **Concept List**:
        {{ concept }}
        #### Output Format:
        [
            {
                "concepts": ["concept1", "Only insert 2-3 concepts here"],
                "question": "Only insert question here"
            },
            {
                "concepts": ["concept1", "concept2", "Only insert 2-3 concepts here"],
                "question": "Only insert question here"
            }
        ]""",
    "deepseek": """作为一名课程讲师，您的任务是设计多样化且具有挑战性的计算类问题。这些问题应能体现所提供主题和关键概念的应用，同时提升学生的推理和批判性思维能力。确保问题不重复、表述精确且具有吸引力。  
        设计多样化挑战性计算类问题的指南：
        1. 概念选择：  
        从提供的列表中随机选取2-3个不同的关键概念用于每个问题。  
        确保选取的关键概念属于同一个主题。
        确保生成的题目广泛覆盖所提供的概念，避免过度依赖少数概念。  
        避免在不同问题中重复相同的概念组合或计算方式。  
        
        2. 多样性与挑战性：  
        每个问题应探索不同的关键概念组合，并具备足够的挑战性（例如，需要多步计算、结合现实场景、涉及抽象或高阶推理）。  
        
        3. 清晰性与精确性：  
        确保问题逻辑严谨。  
        使用明确且无歧义的语言。  
        清晰说明所有假设或条件。  
        
        4. 参考资料使用：  
        利用提供的参考文章作为灵感来源，生成独特、多样且具有挑战性的问题。  
        参考资料的作用包括：  
            通过引入新颖视角、背景或应用场景补充概念列表。  
            帮助设计更复杂、贴近实际或非传统教学场景的问题。  
            作为资源，扩展出现实案例或抽象延伸的问题。  
            
        5. 输出多样性：  
        生成1至3个问题。  
        确保每个问题在结构、解题思路和概念运用上均独一无二。  
        除非必要，尽量减少子问题的使用。  
        答案应为精确解；若无法实现，需明确说明允许近似答案。  
        
        输入：
        论文文本：  
        {{ text }}  
        概念列表：  
        {{ concept }}  

        输出格式：
        [
            {

                "concepts": ["概念1", "仅填入2-3个概念"],  
                "question": "仅填入问题"  
            },  
            {
                "concepts": ["概念1", "概念2", "仅填入2-3个概念"],  
                "question": "仅填入问题"  
            }
        ]""",
    "deepseek-v1": """作为一名课程讲师，您的任务是设计多样化且具有挑战性的计算类问题。这些问题应能体现所提供主题和关键概念的应用，同时提升学生的推理和批判性思维能力。确保问题不重复、表述精确且具有吸引力。  
        设计多样化挑战性计算类问题的指南：
        1. 概念选择：  
        从提供的列表中随机选取2-3个不同的关键概念用于每个问题。  
        确保选取的关键概念属于同一个主题。
        确保生成的题目广泛覆盖所提供的概念，避免过度依赖少数概念。  
        避免在不同问题中重复相同的概念组合或计算方式。  
        
        2. 多样性与挑战性：  
        每个问题应探索不同的关键概念组合，并具备足够的挑战性（例如，需要多步计算、结合现实场景、涉及抽象或高阶推理）。  
        
        3. 清晰性与精确性：  
        确保问题逻辑严谨。  
        使用明确且无歧义的语言。  
        清晰说明所有假设或条件。  
        
        4. 参考资料使用：  
        利用提供的参考文章作为灵感来源，生成独特、多样且具有挑战性的问题。  
        参考资料的作用包括：  
            通过引入新颖视角、背景或应用场景补充概念列表。  
            帮助设计更复杂、贴近实际或非传统教学场景的问题。  
            作为资源，扩展出现实案例或抽象延伸的问题。  
            
        5. 输出多样性：  
        生成1至3个问题。  
        确保每个问题在结构、解题思路和概念运用上均独一无二。  
        除非必要，尽量减少子问题的使用。  
        答案应为精确解；若无法实现，需明确说明允许近似答案。  
        
        输入：
        论文文本：  
        {{ text }}  
        概念列表：  
        {{ concept }}  

        输出格式：
        [
            {

                "concepts": ["概念1", "仅填入2-3个概念"],  
                "question": "仅填入问题"  
            },  
            {
                "concepts": ["概念1", "概念2", "仅填入2-3个概念"],  
                "question": "仅填入问题"  
            }
        ]""",
    "deepseek-v2": """你是一位半导体显示技术领域的资深专家，擅长根据提供的主题和关键概念设计问题。你的职责是从论文中生成问题和相应的答案，问题和相应的答案对需要提供给资深的人员学习，问题和相应的答案的质量要高。请根据输入的学术论文内容以及主题和关键概念，生成{x}个需要逻辑推理才能解答的高质量技术问题，请确保这些问题能够直接从论文中找到答案。这些问题将用于资深研究人员的专业能力评估，需满足以下要求：
【核心要求】
概念选择：  
a) 从提供的列表中随机选取2-3个不同的关键概念用于每个问题。
b) 确保每个问题选取的关键概念属于同一个主题。
c) 确保生成的题目广泛覆盖所提供的概念，避免过度依赖少数概念。  
d) 避免在不同问题中重复相同的概念组合方式。

问题设计准则：
a) 首先你需要阅读全文，并判断哪些文本中涉及到逻辑推理的内容。然后你需要根据逻辑推理的内容设计相应的问题。
b) 问题必须基于论文中的技术原理进行设计，问题的描述必须明确清晰全面，问题中主语或名词的描述必须要精准、全面且具备通用性，专有名词应该让行业人员都能看懂。
c) 问题中请不要引用文献或者文章定义的专有名词，请结合你自身半导体的显示领域的知识和文章内容，生成普适通用的问题，在不阅读论文的情况也能正常理解问题所表达的含义。
d) 问题中的名词描述不可以缩写，需要与论文中的描述一致。例如论文中提到的是“OLED材料”，问题中不能简化为“材料”。例如论文中提到的是“LTPS器件”，问题中不能简化为“器件”。
e) 不要针对于论文中的某个特定示例进行提问，问题尽量使顶尖科学家在不阅读论文的情况下也能理解和回答。且问题不能包含“书本”、“论文”、“本文”、“本实验”、“报道”、“xx等人的研究”等相关信息； 

科学严谨性：
a) 因果链：问题需呈现完整技术逻辑链（如：机制A如何影响参数B，进而导致现象C）
b) 周密性：过程需要科学严谨，逐步思考，确保问题和对应的答案来源于论文的内容。且答案需要能在论文中完全找到详细的描述。
问题简洁：问题要凝练简洁。

【禁止事项】
× 禁止使用"本文/本研究/本实验"等论文自指表述
× 禁止提问孤立概念（如：XX技术的定义是什么）
× 禁止超出论文技术范围的假设性问题

【输入】：
论文文本：  
{{ text }}  
概念列表：  
{{ concept }}  

【格式要求】：用中文输出。当前阶段只设计问题，不输出答案。严格按照以下格式输出你设计的问题：
[
    {

        "concepts": ["概念1", "仅填入2-3个概念"],  
        "question": "仅填入问题"  
    },  
    {
        "concepts": ["概念1", "概念2", "仅填入2-3个概念"],  
        "question": "仅填入问题"  
    }
]"""
}

prompt_questions = {
    "deepseek-v2": """你是一位半导体显示技术领域的资深专家，擅长根据提供的主题和关键概念设计问题。你的职责是从论文中生成问题和相应的答案，问题和相应的答案对需要提供给资深的人员学习，问题和相应的答案的质量要高。请根据输入的学术论文内容以及主题和关键概念，生成{x}个需要逻辑推理才能解答的高质量技术问题，请确保这些问题能够直接从论文中找到答案。这些问题将用于资深研究人员的专业能力评估，需满足以下要求：
【核心要求】
概念选择：  
a) 从提供的列表中随机选取2-3个不同的关键概念用于每个问题。
b) 确保每个问题选取的关键概念属于同一个主题。
c) 确保生成的题目广泛覆盖所提供的概念，避免过度依赖少数概念。  
d) 避免在不同问题中重复相同的概念组合方式。

问题设计准则：
a) 首先你需要阅读全文，并判断哪些文本中涉及到逻辑推理的内容。然后你需要根据逻辑推理的内容设计相应的问题。
b) 问题必须基于论文中的技术原理进行设计，问题的描述必须明确清晰全面，问题中主语或名词的描述必须要精准、全面且具备通用性，专有名词应该让行业人员都能看懂。
c) 问题中请不要引用文献或者文章定义的专有名词，请结合你自身半导体的显示领域的知识和文章内容，生成普适通用的问题，在不阅读论文的情况也能正常理解问题所表达的含义。
d) 问题中的名词描述不可以缩写，需要与论文中的描述一致。例如论文中提到的是“OLED材料”，问题中不能简化为“材料”。例如论文中提到的是“LTPS器件”，问题中不能简化为“器件”。
e) 不要针对于论文中的某个特定示例进行提问，问题尽量使顶尖科学家在不阅读论文的情况下也能理解和回答。且问题不能包含“书本”、“论文”、“本文”、“本实验”、“报道”、“xx等人的研究”等相关信息； 

科学严谨性：
a) 因果链：问题需呈现完整技术逻辑链（如：机制A如何影响参数B，进而导致现象C）
b) 周密性：过程需要科学严谨，逐步思考，确保问题和对应的答案来源于论文的内容。且答案需要能在论文中完全找到详细的描述。
问题简洁：问题要凝练简洁。

【禁止事项】
× 禁止使用"本文/本研究/本实验"等论文自指表述
× 禁止提问孤立概念（如：XX技术的定义是什么）
× 禁止超出论文技术范围的假设性问题

【输入】：
论文文本：  
{{ text }}  
概念列表：  
{{ concept }}  

【格式要求】：用中文输出。当前阶段只设计问题，不输出答案。严格按照以下格式输出你设计的问题：
[
    {

        "concepts": ["概念1", "仅填入2-3个概念"],  
        "question": "仅填入问题"  
    },  
    {
        "concepts": ["概念1", "概念2", "仅填入2-3个概念"],  
        "question": "仅填入问题"  
    }
]"""
}
def gen_question_prompt(topics: dict) -> list[str]:
    """
    Encode multiple prompt instructions into a single string for the general case (`pdf`, `json`, or `txt`).
    """
    
    messages = []
    chunkstr = get_chunkstr(topics["oracle_chunks"])
    prompt = prompt_questions[os.getenv("PROMPT_KEY")].replace("{x}", "2").replace("{{ text }}", chunkstr).replace("{{ concept }}", topics["topics"])
    messages.append({"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"})
    messages.append({"role": "user", "content": prompt})
    return messages
QUESTION_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "concepts": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["question", "concepts"],
    }
}
def clean_and_parse(json_str):
    # 更健壮的清理逻辑，处理各种可能的Markdown格式
    cleaned = json_str.strip()
    # 处理可能的Markdown代码块
    if cleaned.startswith('```') and cleaned.endswith('```'):
        # 去除代码块标记
        cleaned = cleaned[3:-3].strip()
        # 如果还有json前缀（如```json）
        if cleaned.lower().startswith('json'):
            cleaned = cleaned[4:].strip()
    try:
        return json5.loads(cleaned)  # 主要修改点
    except json.JSONDecodeError as e:  # 修改异常类型
        print(f"JSON Decode Error: {e}")
        return None
def clean_and_parse_v2(json_str):
    """
    更健壮的 JSON 解析函数，处理以下情况：
    1. Markdown 代码块标记 (```json 或 ```)
    2. 字符串末尾的非 JSON 内容（如设计说明）
    3. 多种引号格式和注释（使用 json5）
    4. 前导/尾随空白字符
    """
    cleaned = json_str.strip()
    
    # 处理 Markdown 代码块
    if cleaned.startswith('```') and cleaned.endswith('```'):
        cleaned = cleaned[3:-3].strip()
        if cleaned.lower().startswith('json'):
            cleaned = cleaned[4:].strip()
    
    # 查找可能的 JSON 结束位置（处理末尾非 JSON 内容）
    json_end_chars = {'{': '}', '[': ']'}
    stack = []
    json_end_index = None
    
    for i, char in enumerate(cleaned):
        if char in json_end_chars:
            stack.append(json_end_chars[char])
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack:  # 当栈为空时，表示 JSON 结构完整
                json_end_index = i + 1
                break
    
    # 如果检测到完整的 JSON 结构，截取到结束位置
    if json_end_index is not None:
        cleaned = cleaned[:json_end_index]
    
    try:
        return json5.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"JSON 解析错误: {e}")
        print(f"正在尝试更宽松的解析...")
        
        # 尝试处理更多边界情况
        try:
            # 处理可能的多余逗号
            cleaned = cleaned.replace(',]', ']').replace(',}', '}')
            # 处理单引号（替换为双引号）
            cleaned = cleaned.replace("'", '"')
            return json5.loads(cleaned)
        except Exception as e:
            print(f"最终 JSON 解析失败: {e}")
            return None
def clean_and_parse_v3(json_str):
    """
    终极健壮版 JSON 解析函数，处理以下情况：
    1. 开头和结尾的非 JSON 文本
    2. Markdown 代码块标记 (```json 或 ```)
    3. 多种引号格式和注释（使用 json5）
    4. 前导/尾随空白字符
    5. 自动检测 JSON 部分的起始和结束位置
    """
    cleaned = json_str.strip()
    
    # 处理 Markdown 代码块
    if cleaned.startswith('```') and cleaned.endswith('```'):
        cleaned = cleaned[3:-3].strip()
        if cleaned.lower().startswith('json'):
            cleaned = cleaned[4:].strip()
    
    # 方法1：使用正则表达式直接提取 JSON 部分
    json_match = re.search(r'(\[.*\]|\{.*\})', cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group(1)
    else:
        # 方法2：手动查找 JSON 起始位置（如果正则失败）
        json_start_chars = {'[', '{'}
        json_end_chars = {'{': '}', '[': ']'}
        
        start_index = None
        stack = []
        
        # 查找第一个有效的 JSON 起始字符
        for i, char in enumerate(cleaned):
            if char in json_start_chars:
                start_index = i
                stack.append(json_end_chars[char])
                break
        
        if start_index is not None:
            # 继续查找匹配的结束字符
            for i in range(start_index + 1, len(cleaned)):
                char = cleaned[i]
                if char in json_end_chars:
                    stack.append(json_end_chars[char])
                elif stack and char == stack[-1]:
                    stack.pop()
                    if not stack:  # 栈为空表示 JSON 结构完整
                        cleaned = cleaned[start_index:i+1]
                        break
    
    # 最终清理和解析
    try:
        # 先尝试直接解析
        return json5.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"初始 JSON 解析错误: {e}")
        print("尝试修复常见问题...")
        
        # 尝试修复常见问题
        try:
            # 1. 处理单引号
            cleaned = cleaned.replace("'", '"')
            # 2. 处理多余逗号
            cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
            # 3. 处理无引号的键
            cleaned = re.sub(r'([\{\[,]\s*)([a-zA-Z_]\w*)(\s*:)', 
                           lambda m: f'{m.group(1)}"{m.group(2)}"{m.group(3)}', 
                           cleaned)
            
            return json5.loads(cleaned)
        except Exception as e:
            print(f"最终 JSON 解析失败: {e}")
            return None

def generate_questions(chat_completer: Any, topics: dict) -> str | None:
    """
    Generates the label / answer to `question` using `context` and deepseek.
    """
    messages = gen_question_prompt(topics)
    response = chat_completer(
        model=os.getenv("GENERATION_MODEL"),
        messages=messages,
        n=1,
        temperature=0,
        max_tokens=2048,
    )
    questions = response.choices[0].message.content
    try:
        output_questions = clean_and_parse(questions)
        validate(instance=output_questions, schema=QUESTION_SCHEMA)
    except (json.JSONDecodeError, ValidationError) as e:
        logging.error(f"Failed to parse response:\n{questions}\nError: {e}")
        output_questions = []
    return output_questions, topics

def load_other_chunks(article_name, chunk4_path):
    articles_chunks = load_articles(chunk4_path)
    chunk4_list = []
    for a_name, a_chunks in articles_chunks.items():
        if a_name != article_name:
            for chunk4 in a_chunks:
                chunk4_list.extend(chunk4)
    # print(f"all chunks - chunk4_list: {len(chunk4_list)}")
    return chunk4_list
def sort_noisy_chunks(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        existing_questions = json.load(f)
        for a_name, a_topics in tqdm(existing_questions.items(), desc="Sorted chunks."):
            for question_ele in tqdm(a_topics, desc="question_ele"):
                if "score" in question_ele["sorted_chunks"][0]:
                    print(f"score exists. Skipping...")
                    return 
                question = question_ele["question"]
                noisy_chunks = question_ele["sorted_chunks"]
                sorted_chunks = rerank_chunks(question, noisy_chunks)
                question_ele["sorted_chunks"] = sorted_chunks
    with open(filename, 'w', encoding="utf-8") as f:
        json.dump(existing_questions, f, ensure_ascii=False, indent=4)
        print(f"转换sorted_chunks成功。")
    
def sort_noisy_chunks_v2(filename: str) -> None:
    """
    对问题中的噪声chunks进行重新排序并保存
    参数:
        filename: 包含问题的JSON文件路径
    改动:
        1. 将全局跳过逻辑改为按文章跳过
        2. 添加文章级别的处理状态检测
    """
    try:
        # 1. 加载文件
        with open(filename, 'r', encoding="utf-8") as f:
            existing_questions = json.load(f)
            
            # 2. 处理每个问题
            total_articles = len(existing_questions)
            with tqdm(existing_questions.items(), 
                     desc="处理文章中", 
                     unit="article",
                     total=total_articles) as article_pbar:
                
                for a_name, a_topics in article_pbar:
                    article_pbar.set_postfix(article=a_name[:10])
                    
                    # 检查当前文章是否已处理过（任一问题有score则跳过整篇文章）
                    if any("score" in q.get("sorted_chunks", [{}])[0] for q in a_topics):
                        article_pbar.set_postfix(skip="已处理")
                        continue
                    
                    # 使用leave=False避免嵌套进度条混乱
                    with tqdm(a_topics, 
                             desc="处理问题", 
                             unit="question",
                             leave=False) as question_pbar:
                        
                        for question_ele in question_pbar:
                            try:
                                # 3. 重新排序chunks
                                question = question_ele["question"]
                                noisy_chunks = question_ele["sorted_chunks"]
                                sorted_chunks = rerank_chunks(question, noisy_chunks)
                                question_ele["sorted_chunks"] = sorted_chunks
                                
                                # 更新进度条状态
                                question_pbar.set_postfix(q_len=len(noisy_chunks))
                                
                            except KeyError as e:
                                print(f"问题格式错误，缺少必要字段: {e}")
                                continue
                            except Exception as e:
                                print(f"处理问题时出错: {e}")
                                continue
                    
                    # 每处理完一篇文章后保存一次（更安全）
                    with open(filename, 'w', encoding="utf-8") as f:
                        json.dump(existing_questions, f, 
                                 ensure_ascii=False, 
                                 indent=4)
    
    except json.JSONDecodeError:
        print(f"文件 {filename} 不是有效的JSON格式")
        return
    except FileNotFoundError:
        print(f"文件 {filename} 不存在")
        return
    except Exception as e:
        print(f"处理文件时发生未知错误: {e}")
        return
    
    print(f"成功处理并保存文件: {filename}")


def save_questions_v3(questions, topics, article_name, filename, chunk4_path):
    questions_list = []
    all_chunks = load_other_chunks(article_name, chunk4_path)
    for question in questions:
        question_ele = {
            **question,
            **topics
        }
        nums_distract = int(os.getenv("NUM_distract"))
        distract_chunks = random.sample(all_chunks, nums_distract)
        noisy_chunks = distract_chunks + question_ele["oracle_chunks"]
        question_ele["sorted_chunks"] = noisy_chunks
        questions_list.append(question_ele)
    # 判断 filename 是否存在，如果存在则追加写入，否则创建新文件
    if os.path.exists(filename):
        with open(filename, 'r', encoding="utf-8") as f:
            existing_questions = json.load(f)
        # 检查 article_name 是否已经存在于 questions 中
        if article_name in existing_questions:
            existing_questions[article_name].extend(questions_list)
        else:
            existing_questions[article_name] = questions_list
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump(existing_questions, f, ensure_ascii=False, indent=4)
    else:
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump({article_name: questions_list}, f, ensure_ascii=False, indent=4)
    # print(f"Questions saved to {filename}")

# 3
def gen_questions_with_topic_v3(topics_path, question_path, chat_model, chunk4_path) -> list[str]:
    if os.path.exists(question_path):
        print(f"{question_path} exists. Skipping...")
        return 
    articles_topics = load_articles(topics_path)
    for a_name, a_topics in articles_topics.items():
        futures = []
        with tqdm(total=len(a_topics), desc="Questioning", unit="file") as pbar:
            with ThreadPoolExecutor(max_workers=int(os.getenv("MAX_workers"))) as executor:
                for topics in a_topics:
                    futures.append(executor.submit(generate_questions, chat_model, topics))
                for future in as_completed(futures):
                    questions, topics = future.result()
                    pbar.update(1)
                    save_questions_v3(questions, topics, a_name, question_path, chunk4_path)
                print(f"done {a_name} questions.")

# 33
import json
import os
from tqdm import tqdm

def convert_topics_to_jsonl(topics_path, output_jsonl_path):
    """
    将topics_path文件转换为jsonl格式的批量请求文件
    
    参数:
        topics_path: 输入的topics文件路径
        output_jsonl_path: 输出的jsonl文件路径
    """
    # 加载topics文件
    articles_topics = load_articles(topics_path)
    
    # 创建输出文件
    with open(output_jsonl_path, 'w', encoding='utf-8') as out_file:
        request_id = 1  # 自定义ID计数器
        
        # 遍历所有文章和主题
        for a_name, a_topics in articles_topics.items():
            for topics in tqdm(a_topics, desc=f"Processing {a_name}"):
                # 生成问题提示
                messages = gen_question_prompt(topics)
                
                # 构建请求体
                request_body = {
                    "custom_id": f"request-{request_id}",
                    "body": {
                        "messages": messages,
                    }
                }
                
                # 写入jsonl文件
                out_file.write(json.dumps(request_body, ensure_ascii=False) + '\n')
                request_id += 1
    
    print(f"成功将 {topics_path} 转换为 {output_jsonl_path}")

import json
from tqdm import tqdm

def process_response_and_save(response_path, topics_path, question_path, chunk4_path):
    """
    处理响应结果文件并保存到question_path
    
    参数:
        response_path: 响应结果文件路径
        topics_path: 原始topics文件路径
        question_path: 输出question文件路径
        chunk4_path: chunk4文件路径
    """
    # 加载原始topics数据
    articles_topics = load_articles(topics_path)
    
    # 创建按custom_id索引的topics字典
    topics_dict = {}
    request_id = 1
    for a_name, a_topics in articles_topics.items():
        for topics in a_topics:
            custom_id = f"request-{request_id}"
            topics_dict[custom_id] = (a_name, topics)
            request_id += 1
    
    # 加载响应结果文件
    responses = []
    with open(response_path, 'r', encoding='utf-8') as f:
        for line in f:
            responses.append(json.loads(line.strip()))
    
    # 处理每个响应
    for response in tqdm(responses, desc="Processing responses"):
        custom_id = response['custom_id']
        
        if custom_id not in topics_dict:
            print(f"Warning: Custom ID {custom_id} not found in topics data")
            continue
        
        a_name, topics = topics_dict[custom_id]
        
        # 解析响应内容
        if response['error'] is not None:
            print(f"Error in response {custom_id}: {response['error']}")
            continue
        
        try:
            response_body = response['response']['body']
            content = response_body['choices'][0]['message']['content']
            
            # 清理和解析生成的questions
            questions = clean_and_parse_v3(content)
            if questions is None:
                print(f"Failed to parse questions for {custom_id}")
                continue
            
            # 保存到question文件
            save_questions_v3(questions, topics, a_name, question_path, chunk4_path)
            
        except (KeyError, IndexError, TypeError) as e:
            print(f"Error processing response {custom_id}: {str(e)}")
            continue




