#!/usr/bin/env python3
"""
RAFT with Local LLM Support using vLLM

This version uses local large language models through vLLM instead of API keys.
Supports models like Qwen, Llama, and other open-source LLMs.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from mdc import MDC
from tqdm import tqdm
from logconf import log_setup
import logging
from typing import Literal, Any, get_args, List, Dict, Optional
import argparse
from datasets import Dataset, concatenate_datasets
import pyarrow as pa
from transformers import AutoTokenizer
import json
import PyPDF2
import random
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from math import ceil
from format import DatasetConverter, datasetFormats, outputDatasetTypes
from pathlib import Path
from checkpointing import Checkpointing, checkpointed
import uuid
import shutil
from threading import Thread, Event
import os
import sys

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

log_setup()
logger = logging.getLogger("raft_local")

DocType = Literal["api", "pdf", "json", "txt"]
docTypes = list(get_args(DocType))

SystemPromptKey = Literal["gpt", "llama", "deepseek", "deepseek-v2"]
systemPromptKeys = list(get_args(SystemPromptKey))

# Import vLLM client
try:
    from utils.vllm_client import build_vllm_client, get_default_sampling_params, VLLMClient
    from vllm import SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    logger.warning("vLLM not available. Please install with: pip install vllm")
    VLLM_AVAILABLE = False

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
    
    # Local model specific arguments
    parser.add_argument("--model-name", type=str, default="qwq_32", help="Local model name (qwq_32, qw2_72, qw2.5_32, qw2.5_72, llama3.1_70)")
    parser.add_argument("--model-path", type=str, help="Custom model path (overrides model-name)")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-large-zh-v1.5", help="HuggingFace embedding model")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95, help="GPU memory utilization for vLLM")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size for vLLM")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for sampling")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p for sampling")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens for generation")
    
    parser.add_argument("--system-prompt-key", default="deepseek-v2", help="The system prompt to use to generate the dataset", choices=systemPromptKeys)
    parser.add_argument("--workers", type=int, default=1, help="The number of worker threads to use to generate the dataset")
    parser.add_argument("--auto-clean-checkpoints", type=bool, default=False, help="Whether to auto clean the checkpoints after the dataset is generated")
    parser.add_argument("--qa-threshold", type=int, default=None, help="The number of Q/A samples to generate after which to stop the generation process")

    args = parser.parse_args()
    return args


class LocalLLMCompleter:
    """Wrapper for vLLM client to provide chat completion interface"""
    
    def __init__(self, vllm_client: VLLMClient, sampling_params: SamplingParams):
        self.client = vllm_client
        self.sampling_params = sampling_params
        self.tokenizer = vllm_client.tokenizer
        
    def __call__(self, messages: List[Dict[str, str]], **kwargs) -> Dict:
        """Convert messages to prompt and generate response"""
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Update sampling params if needed
        sampling_params = SamplingParams(
            temperature=kwargs.get('temperature', self.sampling_params.temperature),
            top_p=kwargs.get('top_p', self.sampling_params.top_p),
            max_tokens=kwargs.get('max_tokens', self.sampling_params.max_tokens),
            stop_token_ids=self.client.stop_token_ids
        )
        
        # Generate
        outputs = self.client.generate([prompt], sampling_params, use_tqdm=False)
        
        # Format response to match OpenAI structure
        response_text = outputs[0].outputs[0].text
        
        return {
            'choices': [{
                'message': {
                    'content': response_text,
                    'reasoning_content': None  # Local models don't have separate reasoning
                }
            }]
        }


def get_chunks(
    data_path: Path, 
    doctype: DocType = "pdf", 
    chunk_size: int = 512,
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
) -> list[str]:
    """
    Takes in a `data_path` and `doctype`, retrieves the document, breaks it down into chunks of size
    `chunk_size`, and returns the chunks using local embeddings.
    """
    chunks = []
    
    logger.info(f"Retrieving chunks from {data_path} of type {doctype} using {embedding_model}")
    
    if doctype == "api":
        with open(data_path) as f:
            api_docs_json = json.load(f)
        chunks = list(api_docs_json)
        chunks = [str(api_doc_json) for api_doc_json in api_docs_json]
        
        for field in ["user_name", "api_name", "api_call", "api_version", "api_arguments", "functionality"]:
            if field not in chunks[0]:
                raise TypeError(f"API documentation is not in the format specified by the Gorilla API Store: Missing field `{field}`")
    
    else:
        # Use HuggingFace embeddings instead of OpenAI
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        chunks = []
        if doctype == "json":
            with open(data_path, 'r') as f:
                data = json.load(f)
            text = data.get("text", json.dumps(data))
        elif doctype == "pdf":
            text = ""
            with open(data_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)
                for page_num in range(num_pages):
                    page = reader.pages[page_num]
                    text += page.extract_text()
        elif doctype == "txt":
            with open(data_path, 'r') as file:
                text = file.read()
        else:
            raise TypeError("Document is not one of the accepted types: api, pdf, json, txt")
        
        # Use semantic chunker with local embeddings
        text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
        texts = text_splitter.create_documents([text])
        
        for i, text in enumerate(texts):
            chunk_dict = {
                "chunk_id": i,
                "source": str(data_path),
                "content": text.page_content
            }
            chunks.append(chunk_dict)
    
    filename = data_path.stem
    return chunks, filename


# Message templates for different models
build_qa_messages = {
    "gpt": lambda chunk, x : [
            {"role": "system", "content": """You are a synthetic question-answer pair generator. Given a chunk of context about some topic(s), generate %s example questions a user could ask, and would be answered using information from the chunk. For example, if the given context was a Wikipedia paragraph about the United States, an example question could be "What is the capital of the United States?". The questions should be able to be answered in a few words or less. Include only the questions in your response.""" % (x)},
            {"role": "user", "content": str(chunk)}
        ],
    "llama": lambda chunk, x : [
            {"role": "system", "content": """You are a synthetic question-answer pair generator. Given a chunk of context about some topic(s), generate %s example questions a user could ask, and would be answered using information from the chunk. 
                
                For example:
                Context: A Wikipedia paragraph about the United States, 
                Question: What is the capital of the United States?

                Context: A Wikipedia paragraph about the United States, 
                Question: How many states are in the United States?

                Context: A Wikipedia paragraph about vampire bats, 
                Question: What are the different species of vampire bats?
                """ % (x)},
            {"role": "system", "content": "The questions should be able to be answered in a few words or less. Include only the questions in your response."},
            {"role": "user", "content": str(chunk)}
        ],
    "deepseek": lambda chunk, x : [
            {"role": "system", "content": f"你是一个合成问答对的生成器。给定一个关于某些话题的上下文，生成{x}个用户可能会问到的示例问题，并且使用该上下文进行回答。例如，如果给定的上下文是维基百科中关于美国的段落，则示例问题可以是"美国的州有多少？"。"},
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
d) 问题中的名词描述不可以缩写，需要与论文中的描述一致。例如论文中提到的是"OLED材料"，问题中不能简化为"材料"。例如论文中提到的是"LTPS器件"，问题中不能简化为"器件"。
e) 不要针对于论文中的某个特定示例进行提问，问题尽量使顶尖科学家在不阅读论文的情况下也能理解和回答。且问题不能包含"书本"、"论文"、"本文"、"本实验"、"报道"、"xx等人的研究"等相关信息； 

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
    "deepseek-v2": """
        Question: {question}
        Context: {context}

        作为半导体显示技术领域的专家，请基于给定的技术文献内容回答问题。

        【回答要求】
        1. 技术推理过程：
           - 明确指出回答所需的关键技术信息
           - 展示完整的逻辑推理链条
           - 使用 ##begin_quote## 和 ##end_quote## 标记引用的原文

        2. 答案规范：
           - 使用专业术语，保持技术准确性
           - 答案简洁明确，避免冗余
           - 必须以 <ANSWER>: 标记最终答案

        【示例格式】
        <Reasoning>: 
        根据文献内容，[具体技术分析过程]...
        关键信息：##begin_quote##[引用原文]##end_quote##
        因此，[推理结论]...

        <ANSWER>: [简洁的最终答案]
    """
}


def get_chunkstr(chunks: list[dict]) -> str:
    """Convert chunk list to string format"""
    if isinstance(chunks, list):
        chunk_content = ""
        for idx, item in enumerate(chunks):
            chunk_content += f"chunk: {idx}, source: {item.get('source', 'unknown')}\n{item.get('content', str(item))}\n"
        return chunk_content
    return str(chunks)


def generate_instructions_gen(llm_completer: LocalLLMCompleter, chunk: Any, x: int = 5, prompt_key: str = "deepseek-v2") -> list[str]:
    """
    Generates `x` questions / use cases for `chunk` using local LLM
    """
    try:
        if isinstance(chunk, list):
            chunk = get_chunkstr(chunk)
            
        response = llm_completer(
            messages=build_qa_messages[prompt_key](chunk, x),
            max_tokens=min(100 * x, 512),
        )
        
        content = response['choices'][0]['message']['content']
        queries = content.split('\n') if content else []
        queries = [q for q in queries if any(c.isalpha() for c in q)]
        
        return queries
        
    except Exception as e:
        logger.warning(f"Error generating instructions: {e}")
        return []


def encode_question_gen(question: str, context: Any, prompt_key: str = "deepseek-v2") -> tuple[list[dict], str]:
    """
    Encode question and context into prompt messages
    """
    prompts = []
    
    if isinstance(context, list):
        chunkstr = get_chunkstr(context)
    else:
        chunkstr = str(context)
    
    prompt = prompt_templates[prompt_key].replace("{question}", question).replace("{context}", chunkstr)
    prompts.append({"role": "system", "content": "You are a helpful question answerer who can provide an answer given a question and relevant context."})
    prompts.append({"role": "user", "content": prompt})
    return prompts, prompt


def generate_label(llm_completer: LocalLLMCompleter, question: str, context: Any, doctype: DocType = "pdf", prompt_key: str = "deepseek-v2") -> tuple[str, str, str]:
    """
    Generates the label / answer to `question` using `context` and local LLM
    """
    messages, prompt = encode_question_gen(question, context, prompt_key)
    
    response = llm_completer(
        messages=messages,
        temperature=0,
        max_tokens=2048,
    )
    
    response_content = response['choices'][0]['message']['content']
    reasoning_content = None  # Local models don't have separate reasoning
    
    return response_content, reasoning_content, prompt


def add_chunk_to_dataset(
    chunks: list[dict],
    chunk: list[dict],
    doctype: DocType,
    llm_completer: LocalLLMCompleter,
    x: int = 5,
    num_distract: int = 3,
    p: float = 1.0,
    prompt_key: str = "deepseek-v2"
) -> list[dict]:
    """
    Given chunks, create {Q, A, D} triplets and return as list of data points
    """
    data_points = []
    
    # Generate questions
    qs = generate_instructions_gen(llm_completer, chunk, x, prompt_key)
    
    for q in qs:
        datapt = {
            "id": str(uuid.uuid4()),
            "type": "api call" if doctype == "api" else "general",
            "question": q,
            "context": None,
            "oracle_context": None,
            "cot_answer": None
        }
        
        # Create context with distractors
        docs = [ch.copy() for ch in chunk]
        indices = list(range(len(chunks)))
        
        # Remove current chunks from indices
        for ch in chunk:
            if ch["chunk_id"] in indices:
                indices.remove(ch["chunk_id"])
        
        # Add distractor documents
        actual_sample_size = min(num_distract, len(indices))
        for j in random.sample(indices, actual_sample_size):
            docs.append(chunks[j])
        
        # Decide whether to include oracle document
        oracle = random.uniform(0, 1) < p
        if not oracle and len(indices) >= len(chunk):
            # Replace oracle chunks with distractors
            replace_indices = random.sample(indices, len(chunk))
            for i, idx in enumerate(replace_indices[:len(chunk)]):
                docs[i] = chunks[idx]
        
        random.shuffle(docs)
        
        # Format context
        datapt["context"] = {
            "title": ["placeholder_title"] * len(docs),
            "sentences": [doc["content"] for doc in docs]
        }
        datapt["oracle_context"] = get_chunkstr(chunk)
        
        # Generate answer
        cot_answer, reasoning_content, prompt = generate_label(
            llm_completer, q, chunk, doctype, prompt_key
        )
        datapt["cot_answer"] = cot_answer
        datapt["reasoning_content"] = reasoning_content
        
        # Construct instruction
        context_str = ""
        for doc in docs:
            context_str += f"<DOCUMENT>{doc['content']}</DOCUMENT>\n"
        context_str += q
        datapt["instruction"] = context_str
        
        data_points.append(datapt)
    
    return data_points


def main():
    args = get_args()
    
    if not VLLM_AVAILABLE:
        logger.error("vLLM is required but not installed. Please install with: pip install vllm")
        return
    
    # Initialize vLLM
    logger.info("Initializing vLLM client...")
    
    model_config = {
        "model_name": args.model_name,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size
    }
    
    if args.model_path:
        # Override with custom model path
        from utils.vllm_client import MODEL_CONFIGS
        MODEL_CONFIGS["custom"] = {
            "model_path": args.model_path,
            "stop_token_ids": []  # Will be auto-detected
        }
        model_config["model_name"] = "custom"
    
    vllm_client = build_vllm_client(model_config)
    
    # Create sampling params
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=vllm_client.stop_token_ids
    )
    
    # Create LLM completer
    llm_completer = LocalLLMCompleter(vllm_client, sampling_params)
    
    # Process documents
    if args.datapath.is_file():
        files = [args.datapath]
    else:
        files = list(args.datapath.glob("*"))
        files = [f for f in files if f.suffix in ['.pdf', '.txt', '.json']]
    
    logger.info(f"Processing {len(files)} files...")
    
    all_data_points = []
    
    for file_path in tqdm(files, desc="Processing files"):
        logger.info(f"Processing {file_path}")
        
        # Get chunks
        chunks, filename = get_chunks(file_path, args.doctype, args.chunk_size, args.embedding_model)
        logger.info(f"Generated {len(chunks)} chunks")
        
        # Process chunks
        chunk_size = 4  # Number of chunks to combine
        for i in range(0, len(chunks), 1):
            if i + chunk_size > len(chunks):
                break
                
            chunk_group = chunks[i:i+chunk_size]
            
            data_points = add_chunk_to_dataset(
                chunks=chunks,
                chunk=chunk_group,
                doctype=args.doctype,
                llm_completer=llm_completer,
                x=args.questions,
                num_distract=args.distractors,
                p=args.p,
                prompt_key=args.system_prompt_key
            )
            
            all_data_points.extend(data_points)
            
            if args.qa_threshold and len(all_data_points) >= args.qa_threshold:
                logger.info(f"Reached QA threshold of {args.qa_threshold}")
                break
        
        if args.qa_threshold and len(all_data_points) >= args.qa_threshold:
            break
    
    # Create dataset
    logger.info(f"Creating dataset with {len(all_data_points)} data points...")
    
    if all_data_points:
        # Convert to dataset format
        dataset_dict = {
            "id": [dp["id"] for dp in all_data_points],
            "type": [dp["type"] for dp in all_data_points],
            "question": [dp["question"] for dp in all_data_points],
            "context": [dp["context"] for dp in all_data_points],
            "oracle_context": [dp["oracle_context"] for dp in all_data_points],
            "cot_answer": [dp["cot_answer"] for dp in all_data_points],
            "instruction": [dp["instruction"] for dp in all_data_points]
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        # Save dataset
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        dataset.save_to_disk(str(output_path))
        logger.info(f"Dataset saved to {output_path}")
        
        # Convert to desired format if needed
        if args.output_format != "hf":
            converter = DatasetConverter(
                dataset,
                args.output_format,
                args.output_type,
                args.output_chat_system_prompt,
                args.output_completion_prompt_column,
                args.output_completion_completion_column
            )
            converter.convert()
            converter.save(output_path / f"dataset.{args.output_type}")
            logger.info(f"Converted dataset saved to {output_path}/dataset.{args.output_type}")
    else:
        logger.warning("No data points generated!")


if __name__ == "__main__":
    main()