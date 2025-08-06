from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from mdc import MDC
from tqdm import tqdm
from logconf import log_setup
import logging
from typing import Literal, Any, get_args, List
import argparse
from openai import OpenAI, BadRequestError
import datasets
from datasets import Dataset, concatenate_datasets
import pyarrow as pa
from transformers import AutoTokenizer
import json
import PyPDF2
import random
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from client_utils import build_openai_client, build_langchain_embeddings, UsageStats, ChatCompleter
from math import ceil
from format import DatasetConverter, datasetFormats, outputDatasetTypes
from pathlib import Path
from dotenv import load_dotenv
from checkpointing import Checkpointing, checkpointed
import uuid
import shutil
from threading import Thread, Event
import os

log_setup()

load_dotenv()  # take environment variables from .env.

logger = logging.getLogger("raft")

DocType = Literal["api", "pdf", "json", "txt"]
docTypes = list(get_args(DocType))

SystemPromptKey = Literal["gpt", "llama", "deepseek", "deepseek-v2"]
systemPromptKeys = list(get_args(SystemPromptKey))

def get_args() -> argparse.Namespace:
    """
    Parses and returns the arguments specified by the user's command
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--datapath", type=Path, default="", help="If a file, the path at which the document is located. If a folder, the path at which to load all documents")
    parser.add_argument("--output", type=str, default="./", help="The path at which to save the dataset")
    parser.add_argument("--output-format", type=str, default="hf", help="The format of the output dataset.", choices=datasetFormats)
    parser.add_argument("--output-type", type=str, default="jsonl", help="Type to export the dataset to. Defaults to jsonl.", choices=outputDatasetTypes)
    parser.add_argument("--output-chat-system-prompt", type=str, help="The system prompt to use when the output format is chat")
    parser.add_argument("--output-completion-prompt-column", type=str, default="prompt", help="The prompt column name to use for the completion format")
    parser.add_argument("--output-completion-completion-column", type=str, default="completion", help="The completion column name to use for the completion format")
    parser.add_argument("--distractors", type=int, default=3, help="The number of distractor documents to include per data point / triplet")
    parser.add_argument("--p", type=float, default=1.0, help="The percentage that the oracle document is included in the context")
    parser.add_argument("--questions", type=int, default=5, help="The number of data points / triplets to generate per chunk")
    parser.add_argument("--chunk_size", type=int, default=512, help="The size of each chunk in number of tokens")
    parser.add_argument("--doctype", type=str, default="pdf", help="The type of the document, must be one of the accepted doctypes", choices=docTypes)
    parser.add_argument("--openai_key", type=str, default=None, help="Your OpenAI key used to make queries to GPT-3.5 or GPT-4")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-ada-002", help="The embedding model to use to encode documents chunks (text-embedding-ada-002, ...)")
    parser.add_argument("--completion_model", type=str, default="gpt-4", help="The model to use to generate questions and answers (gpt-3.5, gpt-4, ...)")
    parser.add_argument("--system-prompt-key", default="gpt", help="The system prompt to use to generate the dataset", choices=systemPromptKeys)
    parser.add_argument("--workers", type=int, default=2, help="The number of worker threads to use to generate the dataset")
    parser.add_argument("--auto-clean-checkpoints", type=bool, default=False, help="Whether to auto clean the checkpoints after the dataset is generated")
    parser.add_argument("--qa-threshold", type=int, default=None, help="The number of Q/A samples to generate after which to stop the generation process. Defaults to None, which means generating Q/A samples for all documents")

    args = parser.parse_args()
    return args


def get_chunks(
    data_path: Path, 
    doctype: DocType = "pdf", 
    chunk_size: int = 512, 
    openai_key: str | None = None,
    model: str = None
) -> list[str]:
    """
    Takes in a `data_path` and `doctype`, retrieves the document, breaks it down into chunks of size
    `chunk_size`, and returns the chunks.
    """
    chunks = []

    logger.info(f"Retrieving chunks from {data_path} of type {doctype} using the {model} model.")

    if doctype == "api":
        with open(data_path) as f:
            api_docs_json = json.load(f)
        chunks = list(api_docs_json)
        chunks = [str(api_doc_json) for api_doc_json in api_docs_json]

        for field in ["user_name", "api_name", "api_call", "api_version", "api_arguments", "functionality"]:
            if field not in chunks[0]:
                raise TypeError(f"API documentation is not in the format specified by the Gorilla API Store: Missing field `{field}`")

    else:
        embeddings = build_langchain_embeddings(openai_api_key=openai_key, model=model)
        chunks = []
        file_paths = [data_path]
        if data_path.is_dir():
            file_paths = list(data_path.rglob('**/*.' + doctype))

        futures = []
        with tqdm(total=len(file_paths), desc="Chunking", unit="file") as pbar:
            with ThreadPoolExecutor(max_workers=2) as executor:
                for file_path in file_paths:
                    futures.append(executor.submit(get_doc_chunks, embeddings, file_path, doctype, chunk_size))
                for future in as_completed(futures):
                    doc_chunks = future.result()
                    chunks.extend(doc_chunks)
                    pbar.set_postfix({'chunks': len(chunks)})
                    pbar.update(1)

    filename = os.path.basename(data_path)
    return chunks, filename

def get_doc_chunks(
    embeddings: OpenAIEmbeddings,
    file_path: Path, 
    doctype: DocType = "pdf", 
    chunk_size: int = 512,
 ) -> list[str]:
    if doctype == "json":
        with open(file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        text = data["text"]
    elif doctype == "pdf":
        text = ""
        with open(file_path, 'r', encoding="utf-8") as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text += page.extract_text()
    elif doctype == "txt":
        with open(file_path, 'r', encoding="utf-8") as file:
            data = file.read()
        text = str(data)
    else:
        raise TypeError("Document is not one of the accepted types: api, pdf, json, txt")
    
    num_chunks = ceil(len(text) / chunk_size)
    logger.debug(f"Splitting text into {num_chunks} chunks.")

    # text_splitter = SemanticChunker(embeddings, number_of_chunks=num_chunks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.create_documents([text])
    # chunks = text_splitter.create_documents([text])
    chunks = [chunk.page_content for chunk in chunks]
    return chunks

def generate_chunk_instructions(chat_completer: ChatCompleter, chunk: Any, x=5, model: str = None) -> list[str]:
    """
    Generates `x` questions / use cases for `api_call`. Used when the input document is of type `api`.
    """
    response = chat_completer(
        model=model,
        messages=[
            {"role": "system", "content": f"你是一个合成 指令-API对 的生成器。给定一个JSON对象形式的API endpoint，生成{x}个用户可能会问到的示例问题，并且示例问题应该能够被API调用所回答。例如，如果给定的API调用是Gmail API的`service.users().getProfile(userId='me').execute()`调用，一个示例问题可能为'如何获取我的Gmail账户的邮箱地址？'"},
            {"role": "system", "content": f"API endpoint是一个JSON对象，包含以下必选字段：user_name, api_name, api_call, api_version, api_arguments, functionality, 和可选字段 env_requirements, example_code, meta_data, Questions"},
            {"role": "system", "content": "例如，如果api调用包含：{'user_name': 'felixzhu555', 'api_name': 'Google Maps - Address Validation', 'api_call': 'Client.addressvalidation(addressLines, regionCode=region_code, locality=locality, enableUspsCass=boolean)', 'api_version': '4.10.0', 'api_arguments': {},'functionality': '验证一个地址和它的组件，将邮件地址标准化，并确定最佳的地理编码。', 'env_requirements': ['googlemaps'], 'example_code': 'client = googlemaps.Client(key='YOUR_API_KEY')\nresponse = client.addressvalidation('1600 Amphitheatre Pk', regionCode='US', locality='Mountain View', enableUspsCass=True)', 'meta_data': {'description': 'googlemaps python客户端是一个对Google Maps API的抽象，它要求Python 3.5+。每个Google Maps web服务请求都需要一个API密钥或客户端ID。API密钥应该在服务器上保密。', 'questions': []}，一个示例说明为：'验证以下地址： University Avenue and, Oxford St, Berkeley, CA 94720.' }"},
            {"role": "system", "content": "不要提及'API'，或者使用任何提示或API的名称。在三分之一的查询中，请确保包含一个具体的例子，例如'验证这个地址： 123 Harrison St, Oakland CA'。在你的回复中只包含查询。"},
            {"role": "user", "content": str(chunk)}
        ],
        thinking={
            "type": "enable"
        }
    )

    content = response.choices[0].message.content
    queries = content.split('\n')
    queries = [strip_str(q) for q in queries]
    queries = [q for q in queries if any(c.isalpha() for c in q)]

    return queries

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
            # {"role": "system", "content": "这些问题应该具备一定的难度，并且该上下文应该能够用来回答该问题。在回复中只包含问题。"},
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
[[3]] 第3个问题 

[学术论文的开始]
{academic_paper}
[学术论文的结束]"""
            },
        ]
}

def generate_instructions_gen(chat_completer: ChatCompleter, chunk: Any, x: int = 5, model: str = None, prompt_key : str = "gpt") -> list[str]:
    """
    Generates `x` questions / use cases for `chunk`. Used when the input document is of general types 
    `pdf`, `json`, or `txt`.
    """
    try:
        # 判断 chunk 是否是 list
        if isinstance(chunk, list):
            chunk2str = get_chunkstr(chunk)
            # chunk_content = ""
            # for idx,item in enumerate(chunk):
            #     chunk_content += f"chunk: {idx}, source: {item["source"]}\n{item["content"]}\n"
            chunk = chunk2str
        response = chat_completer(
            model=model,
            messages=build_qa_messages[prompt_key](chunk, x),
            max_tokens=min(100 * x, 512), # 25 tokens per question
        )
    except BadRequestError as e:
        if e.code == "content_filter":
            logger.warning(f"Got content filter error, skipping chunk: {e.message}")
            return []
        raise e

    content = response.choices[0].message.content
    queries = content.split('\n') if content else []
    #queries = [strip_str(q) for q in queries]
    queries = [q for q in queries if any(c.isalpha() for c in q)]

    return queries

def strip_str(s: str) -> str:
    """
    Helper function for helping format strings returned by GPT-4.
    """
    l, r = 0, len(s)-1
    beg_found = False
    for i in range(len(s)):
        if s[i].isalpha():
            if not beg_found:
                l = i
                beg_found = True
            else:
                r = i 
    r += 2
    return s[l:min(r, len(s))]

def encode_question(question: str, api: Any) -> list[str]:
    """
    Encode multiple prompt instructions into a single string for the `api` case.
    """
    prompts = []
        
    prompt = question + "\nWrite a python program to call API in " + str(api) + ".\n\nThe answer should follow the format: <<<domain>>> $DOMAIN \n, <<<api_call>>>: $API_CALL \n, <<<api_provider>>>: $API_PROVIDER \n, <<<explanation>>>: $EXPLANATION \n, <<<code>>>: $CODE}. Here are the requirements:\n \n2. The $DOMAIN should be the domain of the API ('N/A' if unknown). The $API_CALL should have only 1 line of code that calls api.\n3. The $API_PROVIDER should be the programming framework used.\n4. $EXPLANATION should be a numbered, step-by-step explanation.\n5. The $CODE is the python code.\n6. Do not repeat the format in your answer."
    prompts.append({"role": "system", "content": "You are a helpful API writer who can write APIs based on requirements."})
    prompts.append({"role": "user", "content": prompt})
    return prompts


prompt_templates = {
    "gpt": """
        Question: {question}\nContext: {context}\n
        Answer this question using the information given in the context above. Here is things to pay attention to: 
        - First provide step-by-step reasoning on how to answer the question. 
        - In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##. This would mean that things outside of ##begin_quote## and ##end_quote## are not directly copy paste from the context. 
        - End your response with final answer in the form <ANSWER>: $answer, the answer should be succinct.
        You MUST begin your final answer with the tag "<ANSWER>:".
    """,
    "llama": """
        Question: {question}
        Context: {context}

        Answer this question using the information given in the context above.
        
        Instructions:
        - Provide step-by-step reasoning on how to answer the question.
        - Explain which parts of the context are meaningful and why.
        - Copy paste the relevant sentences from the context in ##begin_quote## and ##end_quote##.
        - Provide a summary of how you reached your answer.
        - End your response with the final answer in the form <ANSWER>: $answer, the answer should be succinct.
        - You MUST begin your final answer with the tag "<ANSWER>:".

        Here are some samples:

        Example question: What movement did the arrest of Jack Weinberg in Sproul Plaza give rise to?
        Example answer: To answer the question, we need to identify the movement that was sparked by the arrest of Jack Weinberg in Sproul Plaza. 
        The context provided gives us the necessary information to determine this.
        First, we look for the part of the context that directly mentions Jack Weinberg's arrest. 
        We find it in the sentence: ##begin_quote##The arrest in Sproul Plaza of Jack Weinberg, a recent Berkeley alumnus and chair of Campus CORE, 
        prompted a series of student-led acts of formal remonstrance and civil disobedience that ultimately gave rise to the Free Speech Movement##end_quote##.
        From this sentence, we understand that the arrest of Jack Weinberg led to student-led acts which then gave rise to a specific movement. 
        The name of the movement is explicitly mentioned in the same sentence as the "Free Speech Movement."
        Therefore, based on the context provided, we can conclude that the arrest of Jack Weinberg in Sproul Plaza gave rise to the Free Speech Movement.
        <ANSWER>: Free Speech Movement
    """,
    "deepseek": """
        Question: {question}\nContext: {context}\n
        使用上述给定的上下文，回答问题。注意：
        - 首先，请提供有关如何回答问题的详细 reasoning。
        - 在 reasoning 中，如果需要复制上下文中的某些句子，请将其包含在 ##begin_quote## 和 ##end_quote## 中。 这意味着 ##begin_quote## 和 ##end_quote## 之外的内容不是直接从上下文中复制的。
        - 结束你的回答，以 final answer 的形式 <ANSWER>: $answer，答案应该简洁。
        你必须以<Reasoning>: 开头，包含 reasoning 相关的内容；以 <ANSWER>: 开头，包含答案。
    """,
    "deepseek-v2": """{
        "instruction":"你是一个半导体显示领域的资深专家，你掌握TFT、OLED、LCD、QLED、EE、Design等显示半导体显示领域内的相关知识。请根据输入中的切片信息和问题进行回答。切片信息是可能相关的资料，切片信息的内容庞杂，不一定会包含目标答案，请仔细阅读每个切片后再作答，不得出现错误。",
        "input": {
            "context": "{context}",
            "question": "{question}"
        },
        "output": {
            "answer": "根据切片中提供的有效信息对问题进行详尽的回答，推荐分点回答格式。"
        },
        "requirements": {
            "criteria": "根据提供的切片信息提取有效信息进行回答",
            "format": "输出内容必须用中文作答。"
        }
    }"""
}

def get_chunkstr(chunks: List[dict]):
    chunkstr = ""
    for idx,chunk in enumerate(chunks):
        chunkstr += f"chunk: {idx}, source: {chunk['source']}\n{chunk['chunk']}"
        chunkstr += "\n"
    return chunkstr
def encode_question_gen(question: str, chunk: Any, prompt_key : str = "gpt") -> list[str]:
    """
    Encode multiple prompt instructions into a single string for the general case (`pdf`, `json`, or `txt`).
    """
    
    prompts = []
    chunkstr = get_chunkstr(chunk)
    # print(f"Type of prompt template: {type(prompt_templates[prompt_key])}")
    # print(f"Content: {prompt_templates[prompt_key]}")
    # prompt = prompt_templates[prompt_key].format(question=question, context=chunkstr)
    prompt = prompt_templates[prompt_key].replace("{question}", question).replace("{context}", chunkstr)
    prompts.append({"role": "system", "content": "You are a helpful question answerer who can provide an answer given a question and relevant context."})
    prompts.append({"role": "user", "content": prompt})
    return prompts, prompt

def generate_label(chat_completer: ChatCompleter, question: str, context: Any, doctype: DocType = "pdf", model: str = None, prompt_key : str = "gpt") -> str | None:
    """
    Generates the label / answer to `question` using `context` and GPT-4.
    """
    question, prompt = encode_question(question, context) if doctype == "api" else encode_question_gen(question, context, prompt_key)
    response = chat_completer(
        model=model,
        messages=question,
        n=1,
        temperature=0,
        max_tokens=2048,
    )
    reasoning_content = response.choices[0].message.reasoning_content
    response = response.choices[0].message.content
    return response, reasoning_content, prompt

def generate_question_cot_answer(
        chat_completer: ChatCompleter,
        chunks: list[dict], 
        chunk2: list[dict], 
        question,
        doctype: DocType = "api", 
        num_distract: int = 3, 
        p: float = 0.8,
        model: str = None,
        prompt_key: str = "gpt",
        ):
    datapt = {
            "id": None,
            "type": None,
            "question": None,
            "context": None,
            "oracle_context": None,
            "cot_answer": None
        }

    datapt["id"] = str(uuid.uuid4())
    datapt["type"] = "api call" if doctype == "api" else "general"
    datapt["question"] = question

    # add num_distract distractor docs
    # docs = [chunk]
    # docs = chunk2.copy()
    # docs = [chunk["chunk"] for chunk in chunk2]
    docs = [chunk.copy() for chunk in chunk2]
    # indices = list(range(0, len(chunks)))
    # for chunk in chunk2:
    #     indices.remove(chunk["chunk_id"])
    # for j in random.sample(indices, num_distract):
    #     docs.append(chunks[j])
    indices = list(range(0, len(chunks)))
    for chunk in chunk2:
        if chunk["chunk_id"] in indices:
            indices.remove(chunk["chunk_id"])
    # 防止 num_distract 超出 indices 的长度
    actual_sample_size = min(num_distract, len(indices))
    for j in random.sample(indices, actual_sample_size):
        docs.append(chunks[j])
    # decides whether to add oracle document
    oracle = random.uniform(0, 1) < p
    if not oracle:
        # 采样4个文档，并替换docs[0], docs[1]
        nums_distract = len(chunk2)
        mydocs = chunks[random.sample(indices, nums_distract)]
        for idx, doc in enumerate(mydocs):
            docs[idx] = doc
        # docs[0], docs[1] = docs[0]["chunk"], docs[1]["chunk"]
        # docs[0] = chunks[random.sample(indices, 1)[0]]
    random.shuffle(docs)

    d = {
        "title": [],
        "sentences": []
    }

    d["title"].append(["placeholder_title"]*(num_distract+1))
    d["sentences"].append(docs)
    datapt["context"] = d
    # datapt["oracle_context"] = chunk
    datapt["oracle_context"] = chunk2
    # chunk2str = get_chunkstr(chunk2)
    # chunk2str = ""
    # for idx,chunk in enumerate(chunk2):
    #     chunk2str += f"chunk: {idx}, source: {chunk['source']}\n{chunk['chunk']}"
    # add answer to q
    answer, reasoning, instruc_prompt = generate_label(chat_completer, question, chunk2, doctype, model=model, prompt_key=prompt_key)
    datapt["cot_answer"] = f"<think>\n{reasoning}\n</think>\n{answer}"
    # construct model instruction 
    # context = ""
    # for doc in docs:
    #     context += "<DOCUMENT>" + str(doc) + "</DOCUMENT>\n"
    # context += question
    # instruction = """{
    #     "instruction":"你是一个半导体显示领域的资深专家，你掌握TFT、OLED、LCD、QLED、EE、Design等显示半导体显示领域内的相关知识。请根据输入中的切片信息和问题进行回答。切片信息是可能相关的资料，切片信息的内容庞杂，不一定会包含目标答案，请仔细阅读每个切片后再作答，不得出现错误。",
    #     "input": {
    #         "context": "{context}",
    #         "question": "{question}"
    #     },
    #     "output": {
    #         "answer": "根据切片中提供的有效信息对问题进行详尽的回答，推荐分点回答格式。"
    #     },
    #     "requirements": {
    #         "criteria": "根据提供的切片信息提取有效信息进行回答",
    #         "format": "输出内容必须用中文作答。"
    #     }
    # }"""
    instruction = prompt_templates["deepseek-v2"]
    ichunkstr = get_chunkstr(docs)
    instruction = instruction.replace("{context}", ichunkstr).replace("{question}", question)
    datapt["instruction"] = instruction
    return datapt

def build_or_load_chunks(
        datapath: Path, 
        doctype: str,
        CHUNK_SIZE: int, 
        OPENAPI_API_KEY: str,
        embedding_model: str,
        checkpoints_dir: Path, 
    ):
    """
    Builds chunks and checkpoints them if asked
    """
    chunks_ds: Dataset = None
    chunks = None
    checkpoints_chunks_path = checkpoints_dir / "chunks"
    logger.info(f"Using checkpoint chunks {checkpoints_chunks_path}")
    if checkpoints_chunks_path.exists():
        chunks_ds = Dataset.load_from_disk(checkpoints_chunks_path)
        chunks = chunks_ds
        chunks_list = chunks

    if not chunks:
        chunks, filename = get_chunks(datapath, doctype, CHUNK_SIZE, OPENAPI_API_KEY, model=embedding_model)
        chunks_list = [
            {"chunk": chunk, "source": filename, "chunk_id": idx}
            for idx, chunk in enumerate(chunks)
        ]
        
    if not chunks_ds:
        chunks_table = pa.table({ "chunk": chunks, "source": [filename] * len(chunks), "chunk_id": range(len(chunks)) })
        chunks_ds = Dataset(chunks_table)
        chunks_ds.save_to_disk(checkpoints_chunks_path)
    return chunks_list

def main():

    main_start = time.time()

    # run code
    args = get_args()

    # Validate arguments
    if args.output_chat_system_prompt and args.output_format != "chat":
        raise Exception("Parameter --output-chat-system-prompt can only be used with --output-format chat")

    # OPENAPI_API_KEY = args.openai_key
    OPENAPI_API_KEY = os.getenv("COMPLETION_OPENAI_API_KEY")
    print("Using OpenAI API Key")
    client = build_openai_client(
        api_key=OPENAPI_API_KEY,
        base_url=os.getenv("COMPLETION_OPENAI_BASE_URL")
        # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        # base_url="https://ark.cn-beijing.volces.com/api/v3"
    )
    chat_completer = ChatCompleter(client)

    CHUNK_SIZE = args.chunk_size
    NUM_DISTRACT_DOCS = args.distractors

    output_path = Path(args.output).absolute()

    checkpoints_dir = Path(str(output_path) + "-checkpoints").absolute()
    auto_clean_checkpoints = args.auto_clean_checkpoints
    if auto_clean_checkpoints:
        logger.info(f"Checkpoints will be automatically deleted after dataset generation. Remove --auto-clean-checkpoints to deactivate.")

    datapath: Path = args.datapath

    datasets.disable_progress_bars()

    # Chunks
    chunks_list = build_or_load_chunks(datapath, args.doctype, CHUNK_SIZE, OPENAPI_API_KEY, args.embedding_model, checkpoints_dir)
    chunks = chunks_list
    # logger.info(f"Loaded {len(chunks)} chunks")
    # logger.info(f"chunks[0]: {chunks[0]}")
    cot_answers_ds = None

    num_chunks = len(chunks)
    num_questions = args.questions
    max_workers = args.workers
    doctype = args.doctype
    completion_model = args.completion_model

    system_prompt_key = args.system_prompt_key

    logger.info(f"Using system prompt key {system_prompt_key}")

    logger.info(f"Using {max_workers} worker threads")

    cot_answers_ds = stage_generate(chat_completer, checkpoints_dir, chunks, num_questions, max_workers, doctype, completion_model, system_prompt_key, num_distract=NUM_DISTRACT_DOCS, p=args.p, qa_threshold=args.qa_threshold)

    # Save as .arrow format
    datasets.enable_progress_bars()
    cot_answers_ds.save_to_disk(str(output_path))

    # Save as .jsonl format
    formatter = DatasetConverter()

    # Extract format specific params
    format_params = {}
    if args.output_chat_system_prompt:
        format_params['system_prompt'] = args.output_chat_system_prompt

    if args.output_format == "completion":
        format_params['prompt_column'] = args.output_completion_prompt_column
        format_params['completion_column'] = args.output_completion_completion_column

    formatter.convert(ds=cot_answers_ds, format=args.output_format, output_path=str(output_path), output_type=args.output_type, params=format_params)
    # Save as .jsonl format - questions, answers
    formatter.convert_qa(ds=cot_answers_ds, format="qa", output_path=str(output_path), output_type=args.output_type)
    # Save as .jsonl format - questions, answers
    formatter.convert_sft(ds=cot_answers_ds, format="sft", output_path=str(output_path), output_type=args.output_type)

    # Warning, this deletes all intermediary checkpoint files
    if auto_clean_checkpoints:
        shutil.rmtree(checkpoints_dir)

    logger.info(f"Generated {len(cot_answers_ds)} question/answer/CoT/documents samples")
    logger.info(f"Dataset saved to {output_path}")
    logger.info(f"Done in {time.time() - main_start:.2f}s")

class StoppingException(Exception):
    """
    Raised by worker threads when the process is stopping early
    """
    pass

def stage_generate(chat_completer: ChatCompleter, checkpoints_dir, chunks, num_questions, max_workers, doctype, completion_model, system_prompt_key, num_distract, p, qa_threshold):
    """
    Given a chunk, create {Q, A, D} triplets and add them to the dataset.
    """

    questions_checkpointing = Checkpointing(checkpoints_dir / "questions")
    answers_checkpointing = Checkpointing(checkpoints_dir / "answers")
    num_chunks = len(chunks)

    # Tracking when the process is stopping, so we can stop the generation process early
    # Initial value is False
    is_stopping = Event()

    @checkpointed(questions_checkpointing)
    def generate_chunk_instructions_ds(chunk2: List[dict], chenkpoint_id: int, doctype: str, *args, **kwargs):
        """
        Generates a dataset of instructions for a given chunk.
        """
        questions = generate_chunk_instructions(chunk=chunk2, *args, **kwargs) if doctype == "api" else generate_instructions_gen(chunk=chunk2, *args, **kwargs)
        chunk2_question_pairs = [{"chunk": chunk2, "question": question} for question in questions]
        questions_ds = Dataset.from_list(chunk2_question_pairs)
        return questions_ds

    @checkpointed(answers_checkpointing)
    def generate_question_cot_answers(questions_ds, chenkpoint_id: int, *args, **kwargs):
        def process_example(chunk2, question):
            try:
                cot_answer = generate_question_cot_answer(chunk2=chunk2, chunks=chunks, question=question, *args, **kwargs)
            except BadRequestError as e:
                if e.code == "content_filter":
                    logger.warning(f"Got content filter error, skipping question '{question}': {e.message}")
                    return None
                raise e

            return cot_answer

        results = [process_example(chunk2, question) for chunk2, question in zip(questions_ds['chunk'], questions_ds['question']) if len(chunk2) > 2] if len(questions_ds) > 0 else []
        results = [r for r in results if r is not None]
        table = pa.Table.from_pylist(results)
        ds = Dataset(table)
        return ds

    def process_chunk(i):
        if is_stopping.is_set():
            raise StoppingException()
        # 取 i 和 i+4 个chunk
        if (i+4) >= len(chunks):
            chunk = chunks[i:]
        else:
            chunk = chunks[i: i+4]  
        questions_ds = generate_chunk_instructions_ds(chunk2=chunk, chenkpoint_id=i, chat_completer=chat_completer, x=num_questions, model=completion_model, doctype=doctype, prompt_key=system_prompt_key)
        answers_ds = generate_question_cot_answers(questions_ds=questions_ds, chenkpoint_id=i, chat_completer=chat_completer, model=completion_model, doctype=doctype, prompt_key=system_prompt_key, num_distract=num_distract, p=p)
        return answers_ds

    futures = []
    answers_ds_list = []
    usage_stats = UsageStats()

    # we use the checkpointing to keep track of the chunks that have already been processed
    # the answers are generated after the questions so the process might have been stopped in between a batch of answers and matching questions
    # so we need to use the answers checkpointing to keep track of which chunks we need to process
    # if the questions for a given chunk have already been checkpointed, they will just be loaded from the checkpoint
    # we set the tqdm's initial position to avoid having cached data skew the stats
    missing_chunks = answers_checkpointing.missing_checkpoints(num_chunks)

    gen_questions_count = 0
    if answers_checkpointing.has_checkpoints():
        ds = answers_checkpointing.collect_checkpoints()
        gen_questions_count = len(ds)

    done_chunks = num_chunks - len(missing_chunks)
    if done_chunks > 0 or gen_questions_count > 0:
        logger.info(f"Resuming generation from chunk {done_chunks}/{num_chunks} and {gen_questions_count} questions")

    # If we have a QA threshold, it makes more sense to keep track of the number of questions generated
    # Otherwise, track chunks
    track_questions = qa_threshold is not None

    if qa_threshold:
        logger.info(f"Will stop early as soon as the QA threshold is met: {qa_threshold}")

    if track_questions:
        tqdm_args = {"total": qa_threshold, "unit": "qa", "initial": gen_questions_count}
    else:
        tqdm_args = {"total": num_chunks, "unit": "chunk", "initial": done_chunks}

    tps = 0
    with tqdm(desc="Generating", **tqdm_args) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # for i in missing_chunks:
            #     futures.append(executor.submit(process_chunk, i))
            # # missing_chunks 每2个chunk取一个idx
            # def to_batch(lst, groupsize):
            #     for i in range(0, len(lst), groupsize):
            #         yield lst[i:i+groupsize]
            # for i in to_batch(missing_chunks, 2):
            #     futures.append(executor.submit(process_chunk, i))
            for i in range(0, len(missing_chunks), 4):
                futures.append(executor.submit(process_chunk, i))
                
            for future in as_completed(futures):
                if qa_threshold and gen_questions_count >= qa_threshold:
                    logger.info(f"Met threshold {gen_questions_count} >= {qa_threshold} questions, stopping generation")
                    is_stopping.set()
                    break
                answers_ds = future.result()
                answers_ds_list.append(answers_ds)
                increment = min(len(answers_ds), qa_threshold - gen_questions_count) if track_questions else 1
                gen_questions_count += len(answers_ds)
                done_chunks += 1
                stats = chat_completer.get_stats_and_reset()
                # if stats:
                #     tps = stats.total_tokens / stats.duration
                #     usage_stats += stats
                if stats.duration < 0.001:
                    tps = 0  # 避免除以接近零的小数
                else:
                    tps = stats.total_tokens / stats.duration
                    usage_stats += stats
                postfix = {'last tok/s': tps, 'avg tok/s': usage_stats.total_tokens / usage_stats.duration if usage_stats.duration > 0 else 0}
                if track_questions:
                    postfix['chunks'] = done_chunks
                else:
                    postfix['qa'] = gen_questions_count
                pbar.set_postfix(postfix)
                pbar.update(increment)

    ds = answers_checkpointing.collect_checkpoints()
    ds = ds.select(range(qa_threshold)) if qa_threshold else ds
    logger.info(f"Consumed {usage_stats.prompt_tokens} prompt tokens, {usage_stats.completion_tokens} completion tokens, {usage_stats.total_tokens} total tokens")

    return ds

if __name__ == "__main__":
    with MDC(progress="0%"):
        main()
