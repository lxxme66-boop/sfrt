import os
import uuid
import json
from utils.common_utils import load_articles, get_chunkstr
from utils.answer_generation_v2 import prompt_templates

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
    # print(f"Syndatas saved to {filename}")
    
def syn_data(answers_path, syndatas_path):
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
                "content": None,
                "reasoning_content": None
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
            # instruction_prompt = """{
            #     "instruction":"你是一个半导体显示领域的资深专家，你掌握TFT、OLED、LCD、QLED、EE、Design等显示半导体显示领域内的相关知识。请根据输入中的切片信息和问题进行回答。切片信息是可能相关的资料，切片信息的内容庞杂，不一定会包含目标答案，可能含有与问题相近的干扰信息，请仔细阅读每个切片后再作答，不得出现错误。"
            #     "input": {
            #         "context": "{context}",
            #         "question": "{question}"
            #     },
            #     "output": {
            #         "answer": "根据切片中提供的有效信息和自身知识对问题进行详尽的回答，推荐分点回答格式。"
            #     },
            #     "requirements": {
            #         "criteria": "根据提供的切片信息提取有效信息，同时结合自身已有的半导体显示知识进行完整、准确的回答",
            #         "format": "输出内容必须用中文作答且有逻辑条理性。"
            #     }
            # }"""
            instruction_prompt = prompt_templates[os.getenv("PROMPT_KEY")]
            ichunkstr = get_chunkstr(sorted_chunks)
            instruction = instruction_prompt.replace("{context}", ichunkstr).replace("{question}", question)
            datasyn1["instruction"] = instruction
            datasyn1["output"] = reasoning_answer
            datasyn_list1.append(datasyn1)
        
        # 保存合成数据
        save_syndatas(datasyn_list, syndatas_path)
        syndatas_path1 = syndatas_path.replace(".json", "_instruction.json")
        save_syndatas(datasyn_list1, syndatas_path1)
        # print(f"done {a_name} syndata.")

def syn_data_v2(answers_path, syndatas_path):
    if os.path.exists(syndatas_path):
        print(f"{syndatas_path} exists. Skipping...")
        return
    articles_answers = load_articles(answers_path)
    for a_name,qa_dicts in articles_answers.items():
        datasyn_list = []
        datasyn_list1 = []
        for qa_dict in qa_dicts:
            sorted_chunks = qa_dict["sorted_chunks"]
            question = qa_dict["question"]
            reasoning_answer = qa_dict["reasoning_answer"]
            
            # 合成数据            
            datasyn = {
                "id": None,
                "question": None,
                "content": None,
                "reasoning_content": None
            }
            datasyn["id"] = str(uuid.uuid4())
            datasyn["question"] = question
            datasyn["reasoning_answer"] = reasoning_answer
            datasyn_list.append(datasyn)
            
            datasyn1 = {
                "instruction": "",
                "input": "",
                "output": ""
            }
            instruction_prompt = prompt_templates[os.getenv("PROMPT_KEY")]
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

def syn_data_v3(answers_path, syndatas_path):
    if os.path.exists(syndatas_path):
        print(f"{syndatas_path} exists. Skipping...")
        return
    articles_answers = load_articles(answers_path)
    for a_name,qa_dicts in articles_answers.items():
        datasyn_list = []
        datasyn_list1 = []
        for qa_dict in qa_dicts:
            sorted_chunks = qa_dict["sorted_chunks"]
            question = qa_dict["question"]
            reasoning_answer = f"<think>\n{qa_dict['reasoning_content']}</think>{qa_dict['content']}"
            
            # 合成数据            
            datasyn = {
                "id": None,
                "question": None,
                "noisy_chunks": None,
                "content": None,
                "reasoning_content": None
            }
            datasyn["id"] = str(uuid.uuid4())
            datasyn["question"] = question
            datasyn["content"] = qa_dict["content"]
            datasyn["reasoning_content"] = qa_dict["reasoning_content"]
            datasyn["noisy_chunks"] = sorted_chunks
            datasyn_list.append(datasyn)
            
            datasyn1 = {
                "instruction": "",
                "input": "",
                "output": ""
            }
            instruction_prompt = prompt_templates[os.getenv("PROMPT_KEY")]
            ichunkstr = get_chunkstr(sorted_chunks)
            instruction = instruction_prompt.replace("{context}", ichunkstr).replace("{question}", question)
            datasyn1["instruction"] = "You will be given a problem. Please reason step by step, and put your final answer within \\boxed{{}}"
            datasyn1["input"] = instruction
            datasyn1["output"] = reasoning_answer
            datasyn_list1.append(datasyn1)
        
        # 保存合成数据
        save_syndatas(datasyn_list, syndatas_path)
        syndatas_path1 = syndatas_path.replace(".json", "_instruction.json")
        save_syndatas(datasyn_list1, syndatas_path1)
        print(f"done {a_name} syndata.")

