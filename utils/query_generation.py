from typing import List, Any
from openai import BadRequestError, OpenAI
import os 
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from tqdm import tqdm
from utils.common_utils import *
from utils.retrieve_nodes import rerank_chunks


build_qa_messages = {
    "gpt": lambda chunk, x : [
            {"role": "system", "content": """You are a synthetic question-answer pair generator. Given a chunk of context about 
             some topic(s), generate %s example questions a user could ask and would be answered using information from the chunk. 
             For example, if the given context was a Wikipedia paragraph about the United States, an example question could be 
             'How many states are in the United States?'""" % (x)},
            {"role": "system", "content": "The questions should be able to be answered in a few words or less. Include only the questions in your response."},
            {"role": "user", "content": str(chunk)}
        ],
    "llama": lambda chunk, x : [
            {"role": "system", "content": 
                """You are a synthetic question generator.
                
                Instructions:
                - Given a chunk of context about some topic(s), generate %s example questions a user could ask
                - Questions should be answerable using only information from the chunk.
                - Generate one question per line
                - Generate only questions
                - Questions should be succinct

                Here are some samples:
                Context: A Wikipedia paragraph about the United States, 
                Question: How many states are in the United States?

                Context: A Wikipedia paragraph about vampire bats, 
                Question: What are the different species of vampire bats?
                """ % (x)},
            {"role": "system", "content": "The questions should be able to be answered in a few words or less. Include only the questions in your response."},
            {"role": "user", "content": str(chunk)}
        ],
    "deepseek": lambda chunk, x : [
            {"role": "system", "content": f"你是一个合成问答对的生成器。给定一个关于某些话题的上下文，生成{x}个用户可能会问到的示例问题，并且使用该上下文进行回答。例如，如果给定的上下文是维基百科中关于美国的段落，则示例问题可以是“美国的州有多少？”。"},
            {"role": "system", "content": "用中文进行提问，并且这些问题应该用简洁的语言回答。在回复中只包含问题。"},
            {"role": "user", "content": str(chunk)}
        ],
    "deepseek-v2": lambda academic_paper, x : [
            {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"}, 
            {"role": "user", "content": f"""你是一位半导体显示技术领域的资深专家，擅长从技术文献中提炼核心知识点。你的职责是从论文中生成问题和相应的答案，问题和相应的答案对需要提供给资深的人员学习，问题和相应的答案的质量要高。请根据输入的学术论文内容，生成{x}个需要逻辑推理才能解答的高质量技术问题，请确保这些问题能够直接从论文中找到答案。这些问题将用于资深研究人员的专业能力评估，需满足以下要求：
【核心要求】
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

【格式要求】：用中文输出。当前阶段只设计问题，不输出答案。严格按照以下格式输出你设计的问题：
[[1]] 第1个问题
[[2]] 第2个问题

[学术论文的开始]
{academic_paper}
[学术论文的结束]"""
            },
        ]
}

def generate_instructions_gen(chat_completer: Any, chunk: Any, x: int = 2) -> list[str]:
    """
    Generates `x` questions / use cases for `chunk`. Used when the input document is of general types 
    `pdf`, `json`, or `txt`.
    """
    try:
        # 判断 chunk 是否是 list
        if isinstance(chunk, list):
            chunk4 = chunk
            chunk2str = get_chunkstr(chunk)
            chunk = chunk2str
        response = chat_completer(
            model=os.getenv("GENERATION_MODEL"),
            messages=build_qa_messages[os.getenv("PROMPT_KEY")](chunk, x),
            max_tokens=min(100 * x, 512), # 25 tokens per question
        )
    except BadRequestError as e:
        if e.code == "content_filter":
            # logger.warning(f"Got content filter error, skipping chunk: {e.message}")
            return []
        raise e

    content = response.choices[0].message.content
    queries = content.split('\n') if content else []
    queries = [q for q in queries if any(c.isalpha() for c in q)]

    return queries, chunk4

def get_chunk4(i, chunks):
    # 取 i 和 i+4 个chunk
    if (i+4) >= len(chunks):
        chunk4 = chunks[i:]
    else:
        chunk4 = chunks[i: i+4] 
    return chunk4

def save_questions(questions, chunk4, article_name, filename):
    questions_list = []
    for question in questions:
        sorted_chunks = rerank_chunks(question, chunk4)
        question_dict = {
            "question": question,
            "oracle_chunks": chunk4,
            "sorted_chunks": sorted_chunks,
        }
        questions_list.append(question_dict)
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
    print(f"Questions saved to {filename}")

def gen_query(chunks_path, chat_model, questions_path):
    if os.path.exists(questions_path):
        # articles_questions = load_articles(questions_path)
        print(f"{questions_path} exists. Skipping...")
        return 
    articles_chunks = load_articles(chunks_path)
    print(f"articles: {len(articles_chunks)}")
    for a_name, chunks in articles_chunks.items():
        a_chunks = chunks
        print(f"processing {a_name}")
        futures = []
        # a_chunks 除 4 上取整数
        num_chunks = (len(a_chunks) // 4) + 1
        with tqdm(total=num_chunks, desc="Questioning", unit="file") as pbar:
            with ThreadPoolExecutor(max_workers=2) as executor:
                for i in range(0, len(a_chunks), 4):
                    chunk4 = get_chunk4(i, a_chunks)
                    # if len(chunk4) > 2:
                    if chunk4:
                        futures.append(executor.submit(generate_instructions_gen, chat_model, chunk4, x=2))
                for future in as_completed(futures):
                    questions, chunk4f = future.result()
                    # print(f"processing {a_name} questions {len(questions)}")
                    pbar.set_postfix({'chunks': i})
                    pbar.update(1)
                    save_questions(questions, chunk4f, a_name, questions_path)
                print(f"done {a_name} questions.")


