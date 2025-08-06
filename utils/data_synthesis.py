import os
import uuid
import json
import random
from utils.common_utils import load_articles, get_chunkstr
from utils.retrieve_nodes import get_query_engine, get_reranked_nodes, rerank_chunks

def save_syndatas(datasyn_list, filename):
    # 判断 filename 是否存在，如果存在则追加写入，否则创建新文件
    if os.path.exists(filename):
        with open(filename, 'r', encoding="utf-8") as f:
            existing = json.load(f)
        existing.extend(datasyn_list)
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=4)
    else:
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump(datasyn_list, f, ensure_ascii=False, indent=4)
    print(f"Syndatas saved to {filename}")
    
def save_source_nodes(question, source_nodes, filename):
    # 判断 filename 是否存在，如果存在则追加写入，否则创建新文件
    if os.path.exists(filename):
        with open(filename, 'r', encoding="utf-8") as f:
            existing = json.load(f)
        # 检查 article_name 是否已经存在于 questions 中
        if question in existing:
            existing[question].extend(source_nodes)
        else:
            existing[question] = source_nodes
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=4)
    else:
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump({question: source_nodes}, f, ensure_ascii=False, indent=4)
    print(f"Source nodes saved to {filename}")
def get_distract_chunks(question, chunk4, num_distract, query_engine, p=1.0):
    # 添加干扰文档 检索召回
    source_nodes = get_reranked_nodes(question, query_engine)
    # save_source_nodes(question, source_nodes, "outputs_syndatas/source_nodes.json")
    chunk4_ids = [chunk["chunk_id"] for chunk in chunk4]
    source_nodes_without_chunk4 = [source_node.node for source_node in source_nodes if source_node.node.node_id not in chunk4_ids]
    distract_chunks = source_nodes_without_chunk4[:int(num_distract)]
    distract_chunks = [
        {
            "chunk_id": chunk.node_id,
            "chunk": chunk.text,
            "source": chunk.metadata["source"],
        } for chunk in distract_chunks
    ]
    return distract_chunks
def get_chunkdict(chunks_path):
    articles_chunks = load_articles(chunks_path)
    all_chunks = {}
    for a_name,chunks in articles_chunks.items():
        for chunk in chunks:
            all_chunks[chunk["chunk_id"]] = chunk
    print(f"syn_data all_chunks: {len(all_chunks.keys())}")
    return all_chunks
def syn_data(chunks_path, answers_path, syndatas_path, num_distract: int = 3):
    if os.path.exists(syndatas_path):
        print(f"{syndatas_path} exists. Skipping...")
        return
    # articles_chunks = load_articles(chunks_path)
    # all_chunks = {}
    # for a_name,chunks in articles_chunks.items():
    #     for chunk in chunks:
    #         all_chunks[chunk["chunk_id"]] = chunk
    # print(f"syn_data all_chunks: {len(all_chunks.keys())}")
    # all_chunks = get_chunkdict(chunks_path)
    articles_answers = load_articles(answers_path)
    query_engine = get_query_engine()
    for a_name,qa_dicts in articles_answers.items():
        datasyn_list = []
        datasyn_list1 = []
        for qa_dict in qa_dicts:
            chunk4 = qa_dict["oracle_chunks"]
            question = qa_dict["question"]
            reasoning_answer = qa_dict["reasoning_answer"]
            
            # 添加干扰文档
            docs = [chunk.copy() for chunk in qa_dict["oracle_chunks"]]
            # indices = list(all_chunks.keys())
            # for chunk in chunk4:
            #     if chunk["chunk_id"] in indices:
            #         indices.remove(chunk["chunk_id"])
            # for chunk_id in random.sample(indices, int(num_distract)):
            #     docs.append(all_chunks[chunk_id])
            # random.shuffle(docs)
            distract_chunks = get_distract_chunks(question, chunk4, num_distract, query_engine, p=1.0)
            docs.extend(distract_chunks)
            # chunks =[
            #     {"chunk": "placeholder_content", "chunk_id": "placeholder_id", "source": "placeholder_source"},
            #     {"chunk": "placeholder_content", "chunk_id": "placeholder_id", "source": "placeholder_source"}
            # ]
            sorted_chunks = rerank_chunks(question, docs)
            docs = sorted_chunks
            
            # 合成数据            
            datasyn = {
                "id": None,
                "question": None,
                "noisy_chunks": None,
                "oracle_chunks": None,
                "reasoning_answer": None
            }
            datasyn["id"] = str(uuid.uuid4())
            datasyn["question"] = question
            datasyn["oracle_chunks"] = chunk4
            datasyn["reasoning_answer"] = reasoning_answer
            datasyn["noisy_chunks"] = docs
            datasyn_list.append(datasyn)
            
            datasyn1 = {
                "instruction": "",
                "input": "",
                "output": ""
            }
            instruction_prompt = """{
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
            ichunkstr = get_chunkstr(docs)
            instruction = instruction_prompt.replace("{context}", ichunkstr).replace("{question}", question)
            datasyn1["instruction"] = instruction
            datasyn1["output"] = reasoning_answer
            datasyn_list1.append(datasyn1)
        
        # 保存合成数据
        save_syndatas(datasyn_list, syndatas_path)
        syndatas_path1 = syndatas_path.replace(".json", "_instruction.json")
        save_syndatas(datasyn_list1, syndatas_path1)
        print(f"done {a_name} syndata.")

def syn_data_v2(answers_path, syndatas_path):
    if os.path.exists(syndatas_path):
        print(f"{syndatas_path} exists. Skipping...")
        return
    articles_answers = load_articles(answers_path)
    for a_name,qa_dicts in articles_answers.items():
        datasyn_list = []
        datasyn_list1 = []
        for qa_dict in qa_dicts:
            chunk4 = qa_dict["oracle_chunks"]
            sorted_chunks = qa_dict["sorted_chunks"]
            question = qa_dict["question"]
            reasoning_answer = qa_dict["reasoning_answer"]
            
            # 合成数据            
            datasyn = {
                "id": None,
                "question": None,
                "noisy_chunks": None,
                "oracle_chunks": None,
                "reasoning_answer": None
            }
            datasyn["id"] = str(uuid.uuid4())
            datasyn["question"] = question
            datasyn["oracle_chunks"] = chunk4
            datasyn["reasoning_answer"] = reasoning_answer
            datasyn["noisy_chunks"] = sorted_chunks
            datasyn_list.append(datasyn)
            
            datasyn1 = {
                "instruction": "",
                "input": "",
                "output": ""
            }
            instruction_prompt = """{
                "instruction":"你是一个半导体显示领域的资深专家，你掌握TFT、OLED、LCD、QLED、EE、Design等显示半导体显示领域内的相关知识。请根据输入中的切片信息和问题进行回答。切片信息是可能相关的资料，切片信息的内容庞杂，不一定会包含目标答案，可能含有与问题相近的干扰信息，请仔细阅读每个切片后再作答，不得出现错误。"
                "input": {
                    "context": "{context}",
                    "question": "{question}"
                },
                "output": {
                    "answer": "根据切片中提供的有效信息和自身知识对问题进行详尽的回答，推荐分点回答格式。"
                },
                "requirements": {
                    "criteria": "根据提供的切片信息提取有效信息，同时结合自身已有的半导体显示知识进行完整、准确的回答",
                    "format": "输出内容必须用中文作答且有逻辑条理性。"
                }
            }"""
            ichunkstr = get_chunkstr(sorted_chunks)
            instruction = instruction_prompt.replace("{context}", ichunkstr).replace("{question}", question)
            datasyn1["instruction"] = instruction
            datasyn1["output"] = reasoning_answer
            datasyn_list1.append(datasyn1)
        
        # 保存合成数据
        save_syndatas(datasyn_list, syndatas_path)
        syndatas_path1 = syndatas_path.replace(".json", "_instruction.json")
        save_syndatas(datasyn_list1, syndatas_path1)
        print(f"done {a_name} syndata.")


