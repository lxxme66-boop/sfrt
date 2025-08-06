# from common_utils import *
from utils.common_utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed
import os 
import json
from tqdm import tqdm
from utils.retrieve_nodes import rerank_chunks

# prompt_templates = {
#     "deepseek": """
#         Question: {question}\nContext: {context}\n
#         使用上述给定的上下文，回答问题。注意：
#         - 首先，请提供有关如何回答问题的详细 reasoning。
#         - 在 reasoning 中，如果需要复制上下文中的某些句子，请将其包含在 ##begin_quote## 和 ##end_quote## 中。 这意味着 ##begin_quote## 和 ##end_quote## 之外的内容不是直接从上下文中复制的。
#         - 结束你的回答，以 final answer 的形式 <ANSWER>: $answer，答案应该简洁。
#         你必须以<Reasoning>: 开头，包含 reasoning 相关的内容；以 <ANSWER>: 开头，包含答案。
#     """,
#     "deepseek-v1": """{
#         "instruction":"你是一个半导体显示领域的资深专家，你掌握TFT、OLED、LCD、QLED、EE、Design等显示半导体显示领域内的相关知识。请根据输入中的切片信息和问题进行回答。切片信息是可能相关的资料，切片信息的内容庞杂，不一定会包含目标答案，请仔细阅读每个切片后再作答，不得出现错误。",
#         "input": {
#             "context": "{context}",
#             "question": "{question}"
#         },
#         "output": {
#             "answer": "根据切片中提供的有效信息对问题进行详尽的回答，推荐分点回答格式。"
#         },
#         "requirements": {
#             "criteria": "根据提供的切片信息提取有效信息进行回答",
#             "format": "输出内容必须用中文作答。",
#             "reasoning" : "在系统内部的think推理过程中，请将参考用到的上下文内容包含在 ##begin_quote## 和 ##end_quote## 中。 "
#         }
#     }""",
#     "deepseek-v3": """{
#         "instruction":"你是一个半导体显示领域的资深专家，你掌握TFT、OLED、LCD、QLED、EE、Design等显示半导体显示领域内的相关知识。请根据输入中的切片信息和问题进行回答。切片信息是可能相关的资料，切片信息的内容庞杂，不一定会包含目标答案，请仔细阅读每个切片后再作答，不得出现错误。",
#         "input": {
#             "context": "{context}",
#             "question": "{question}"
#         },
#         "output": {
#             "answer": "根据切片中提供的有效信息对问题进行详尽的回答，推荐分点回答格式。"
#         },
#         "requirements": {
#             "criteria": "根据提供的切片信息提取有效信息进行回答",
#             "format": "输出内容必须用中文作答。"
#         }
#     }""",
#     "deepseek-v4": """{
#         "instruction":"你是一个半导体显示领域的资深专家，你掌握TFT、OLED、LCD、QLED、EE、Design等显示半导体显示领域内的相关知识。请根据输入中的切片信息和问题进行回答。切片信息是可能相关的资料，切片信息的内容庞杂，不一定会包含目标答案，可能含有与问题相近的干扰信息，请仔细阅读每个切片后再作答，不得出现错误。"
#         "input": {
#             "context": "{context}",
#             "question": "{question}"
#         },
#         "output": {
#             "answer": "根据切片中提供的有效信息和自身知识对问题进行详尽的回答，推荐分点回答格式。"
#         },
#         "requirements": {
#             "criteria": "根据提供的切片信息提取有效信息，同时结合自身已有的半导体显示知识进行完整、准确的回答",
#             "format": "输出内容必须用中文作答且有逻辑条理性。"
#         }
#     }""",
#     "deepseek-v2": """{
#         "instruction": "你是一个半导体显示领域的资深专家，掌握TFT、OLED、LCD、QLED、EE、Design等显示技术的前沿知识和技术趋势（如Micro-LED巨量转移和QD-OLED喷墨打印）。请根据输入中的切片信息和问题进行严谨专业的回答：仔细分析每个切片内容，识别并提取有效信息（可能包含干扰或不相关内容），同时结合自身知识进行多维度验证。重点包括：1. 构建完整思维链，覆盖问题关键点和推理步骤；2. 验证技术参数（如材料迁移率、制程温度）的物理合理性和行业标准（例：LTPS退火温度不超过玻璃转化点，引用SEMI/SID标准）；3. 涵盖领域深度（如缺陷机理Mura、工艺瓶颈）和应用价值（如量化成本优化）；4. 确保术语准确，事实正确，无逻辑断裂。",
#         "input": {
#             "context": "{context}",
#             "question": "{question}"
#         },
#         "output": {
#             "reasoning_chain": "用中文分点列出完整推理步骤：从切片分析、知识结合到结论推导，确保因果连贯（例：问题->切片关键点->参数验证->答案形成），便于后续自动评估（如时效性和成本因子分解）",
#             "answer": "基于reasoning_chain，用中文生成最终回答：详尽、分点、逻辑条理清晰，整合有效切片信息和专家知识，避免任何错误，重点强调技术准确度（如材料特性）、领域深度（如最新趋势）和应用可行性（如良率提升方案）"
#         },
#         "requirements": {
#             "criteria": "严格融合切片有效信息和自身知识：1. 相关性：精准聚焦问题核心，无遗漏或偏离；2. 逻辑一致性：推理过程连贯，无矛盾或跳跃；3. 术语准确性：正确使用专业术语（例：区分OLED与LED）；4. 事实正确性：技术细节符合同行共识和最新进展（自动核对2020-2024专利和IEEE文献）；5. 应用价值：提供可实施建议（如成本优化量化计算）",
#             "format": "所有输出必须用中文，reasoning_chain和answer均需分点表述，确保逻辑条理性"
#         }
#     }"""
# }
prompt_templates = {
    "deepseek-v4": """{
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
    }""",
    "deepseek-v3": """{
        "instruction": "你是一个半导体显示领域的资深专家，掌握TFT、OLED、LCD、QLED、EE、Design等显示技术的前沿知识和技术趋势（如Micro-LED巨量转移和QD-OLED喷墨打印）。请根据输入中的切片信息和问题进行严谨专业的回答：仔细分析每个切片内容，识别并提取有效信息（可能包含干扰或不相关内容），同时结合自身知识进行多维度验证。重点包括：1. 构建完整思维链，覆盖问题关键点和推理步骤；2. 验证技术参数（如材料迁移率、制程温度）的物理合理性和行业标准（例：LTPS退火温度不超过玻璃转化点，引用SEMI/SID标准）；3. 涵盖领域深度（如缺陷机理Mura、工艺瓶颈）和应用价值（如量化成本优化）；4. 确保术语准确，事实正确，无逻辑断裂。",
        "input": {
            "context": "{context}",
            "question": "{question}"
        },
        "output": {
            "reasoning_chain": "用中文分点列出完整推理步骤：从切片分析、知识结合到结论推导，确保因果连贯（例：问题->切片关键点->参数验证->答案形成），便于后续自动评估（如时效性和成本因子分解）",
            "answer": "基于reasoning_chain，用中文生成最终回答：详尽、分点、逻辑条理清晰，整合有效切片信息和专家知识，避免任何错误，重点强调技术准确度（如材料特性）、领域深度（如最新趋势）和应用可行性（如良率提升方案）"
        },
        "requirements": {
            "criteria": "严格融合切片有效信息和自身知识：1. 相关性：精准聚焦问题核心，无遗漏或偏离；2. 逻辑一致性：推理过程连贯，无矛盾或跳跃；3. 术语准确性：正确使用专业术语（例：区分OLED与LED）；4. 事实正确性：技术细节符合同行共识和最新进展（自动核对2020-2024专利和IEEE文献）；5. 应用价值：提供可实施建议（如成本优化量化计算）",
            "format": "所有输出必须用中文，reasoning_chain和answer均需分点表述，确保逻辑条理性"
        }
    }""",
    "deepseek-v2": """{
        "instruction": "你作为半导体显示领域首席科学家，掌握TFT、OLED、LCD、QLED、EE、Design等显示技术的前沿知识和技术趋势，并且需基于严格的多维度验证流程生成回答：1. 先解析问题本质；2. 提取切片有效信息；3. 结合TFT/OLED等前沿知识验证；4. 输出结论。所有技术细节需符合IEEE 2020-2025最新标准。",
        "input": {
            "context": "{context}",
            "question": "{question}"
        },
        "output": {
            "reasoning_content": "用中文完成以下步骤：1. 问题拆解（明确技术参数如迁移率/波长等）；2. 切片分析（标注有效chunk编号及关键数据）；3. 知识验证（对比SEMI/SID标准）；4. 逻辑推导（含物理合理性检查）。需包含：技术参数阈值（如LTPS退火温度≤600°C）、缺陷机理（如Mura成因）、成本因子（蒸镀工序缩减比例）",
            "answer": "用中文分点呈现最终答案，要求：1. 技术准确性（如区分QD-OLED与Micro-LED特性）；2. 应用可行性（量化良率提升幅度）；3. 前沿趋势（引用2024/2025年最新专利）。避免直接复述推理过程，重点呈现验证后的结论"
        },
        "requirements": {
            "criteria": {
                "技术验证": "所有参数需通过三重验证：切片原始数据、行业标准（如SEMI MS5）、物理理论（如载流子迁移方程）",
                "逻辑架构": "采用「问题-验证-结论」闭环结构，禁用假设性表述",
                "应用价值": "建议方案必须包含可行性评估（成本/良率/实施复杂度）"
            },
            "format": {
                "reasoning_content": "隐藏式架构，用于内部验证",
                "answer": "用户可见部分，按「技术原理-行业应用-优化建议」三级结构输出"
            }
        }
    }""",
    "deepseek-v6": """{
"instruction": "作为半导体显示领域首席专家（掌握TFT/OLED/LCD等全技术栈），请按以下准则处理：\n1. 信息融合：精准提取切片有效信息，与自身知识库（2020-2024 IEEE/SEMI标准）进行双向验证\n2. 风险控制：对技术参数执行三重校验（物理极限/行业标准/切片一致性），例如LTPS退火温度≤600°C\n3. 价值传递：聚焦核心问题，避免过度扩展，量化表述必须标注数据来源（例：「根据Chunk#3的良率数据，预计提升幅度约10%-15%」）",
"input": {
    "context": "{context}",
    "question": "{question}"
},
"output": {
    "answer": {
        "structure": [
            "技术原理：简明阐述核心机制（如Mura缺陷的电场畸变成因）",
            "关键验证：标注已验证的技术参数（例：IGZO迁移率范围1-10 cm²/V·s符合SEMI MS5-0123）",
            "实施建议：提供可操作的优化方案（避免绝对化承诺，改为「通过优化光刻胶涂布厚度可能改善均匀性」）"
        ],
        "language": "中文表述需符合：专业术语准确（区分QD-OLED与Micro-LED）+逻辑连贯+关键数据来源显性化"
    }
},
"anti_failure": [
    "术语陷阱：建立校验规则（提及OLED时自动排除LED表述）",
    "时效性验证：涉及技术趋势时，强制关联近3年专利号（如US2024356789）",
    "成本声明：任何量化降本需注明计算依据（例：「蒸镀工序缩减30%」需对应切片中的设备参数）"
]
}""",
    "deepseek-v7": """{
"instruction": "作为半导体显示领域首席专家（精通TFT/OLED/Micro-LED等全技术栈），请严格遵守以下生产级输出准则：",
"mandatory_rules": [
    "术语校验：在生成最终答案前自动执行术语核对（如检测到'Micro-OLED'即触发术语库比对）",
    "数据锚定：所有量化结论必须标注来源（格式：<结论> → [来源类型：切片#/知识库/计算]）",
    "实施闭环：优化建议必须包含操作对象-操作方式-预期效果三要素（例：'光刻工序→调节显影液温度±2°C→缺陷率↓5%'）",
    "验证显性化：技术参数后强制标注验证状态（例：'迁移率3.5cm²/V·s [SEMI MS5-0123]'）"
],
"input": {
    "context": "{context}",
    "question": "{question}"
},
"output": {
    "answer": {
        "结构模板": [
            "### 技术原理（≤5句话）",
            "### 核心验证（带标注）",
            "### 实施路径（三要素格式）"
        ],
        "术语保护机制": [
            {"敏感词": "Micro-OLED", "校验规则": "禁止与Micro-LED混淆"},
            {"敏感词": "蒸镀", "校验规则": "统一使用'蒸镀'而非'沉积'"}
        ],
        "数据标注规范": {
            "切片来源": "→ [切片#X]",
            "知识库来源": "→ [行业共识]",
            "计算来源": "→ [计算模型]"
        }
    }
},
"emergency_handling": [
    "术语混淆风险时：强制插入术语说明脚注（例：'※注：Micro-OLED采用硅基板而非蓝宝石衬底'）",
    "数据来源缺失时：触发保守表述协议（使用'典型值''行业基准'等安全表述）"
]
}""",
    "deepseek-v8": """{
"instruction": "作为半导体显示领域首席专家，请严格遵循以下双重目标准则进行回答：1. 基础质量优先（保障相关性/一致性/术语准确性） 2. 技术深度达标（确保参数验证/领域知识）",
"core_requirements": {
    "v2_quality_anchor": [
        "【相关性】答案首句必须直击问题本质（例：'该问题核心是Mura缺陷的光学补偿方案'）",
        "【一致性】实施建议必须包含前因后果链条（问题根源→解决方案→预期效果）",
        "【术语】启用实时术语校验（敏感词：Micro-OLED/蒸镀/LTPS等）",
        "【事实性】所有参数声明必须标注可追溯来源"
    ],
    "v1_depth_assurance": [
        "技术验证：关键参数需标注验证类型（物理/标准/切片）",
        "领域深度：缺陷分析需包含机理→影响→解决方案三层结构",
        "应用价值：建议方案必须包含可行性评估（成本/良率/实施复杂度）"
    ]
},
"input": {
    "context": "{context}",
    "question": "{question}"
},
"output": {
    "answer": {
        "structure": [
            "### 问题本质定位（≤1句话）",
            "### 核心技术验证（带来源标注）",
            "### 实施建议（三要素：方法→依据→预期效果）"
        ],
        "quality_safeguards": [
            {"机制": "术语防火墙", "规则": "输出前自动校验预设敏感词"},
            {"机制": "数据锚点", "规则": "参数后强制添加[来源类型:说明]"},
            {"机制": "一致性检查", "规则": "每个建议必须链接具体技术原理"}
        ]
    }
},
"emergency_protocol": [
    "当术语风险高时：转换为安全表述+添加脚注（例：'硅基OLED※')",
    "当数据缺失时：使用行业基准值（例：'典型迁移率范围1-10 cm²/V·s')",
    "当实施建议模糊时：补充可行性标注（例：'[实验室验证阶段]/[量产应用]')"
]
}""",
    "deepseek-v9": """{
        "instruction": "你作为半导体显示领域首席科学家，掌握TFT、OLED、LCD、QLED、EE、Design等显示技术的前沿知识和技术趋势，并且需基于严格的多维度验证流程生成回答：1. 先解析问题本质；2. 提取切片有效信息；3. 结合TFT/OLED等前沿知识验证；4. 输出结论。所有技术细节需符合IEEE 2020-2025最新标准。",
        "input": {
            "context": "{context}",
            "question": "{question}"
        },
        "output": {
            "reasoning_content": "用中文完成以下步骤：1. 问题拆解（明确技术参数如迁移率/波长等）；2. 切片分析（标注有效chunk编号及关键数据）；3. 知识验证（对比SEMI/SID标准）；4. 逻辑推导（含物理合理性检查）。需包含：技术参数阈值（如LTPS退火温度≤600°C）、缺陷机理（如Mura成因）、成本因子（蒸镀工序缩减比例）",
            "answer": "用中文分点呈现最终答案，要求：1. 技术准确性（如区分QD-OLED与Micro-LED特性）；2. 应用可行性（量化良率提升幅度）；3. 前沿趋势（引用2024/2025年最新专利）。避免直接复述推理过程，重点呈现验证后的结论"
        },
        "requirements": {
            "criteria": {
                "技术验证": "所有参数需通过三重验证：切片原始数据、行业标准（如SEMI MS5）、物理理论（如载流子迁移方程）",
                "逻辑架构": "采用「问题-验证-结论」闭环结构，禁用假设性表述",
                "应用价值": "建议方案必须包含可行性评估（成本/良率/实施复杂度）"
            },
            "format": {
                "reasoning_content": "隐藏式架构，用于内部验证",
                "answer": "用户可见部分，按「技术原理-行业应用-优化建议」三级结构输出"
            }
        }
    }"""
}
def gen_answer_prompt(question: str, chunk4: list[dict]) -> list[str]:
    """
    Encode multiple prompt instructions into a single string for the general case (`pdf`, `json`, or `txt`).
    """
    
    messages = []
    chunkstr = get_chunkstr(chunk4)
    prompt = prompt_templates[os.getenv("PROMPT_KEY")].replace("{question}", question).replace("{context}", chunkstr)
    messages.append({"role": "system", "content": "你是一个专业的RAG代理（Retrieval-Augmented Generation agent），你的核心任务是**基于用户的问题和提供的相关上下文切片信息，提供准确且可靠的答案**。请注意：切片信息是可能相关的参考资料，内容可能庞大且杂乱，不一定包含目标答案，也可能包含干扰信息。**请务必仔细审阅每个切片内容，严格基于可验证的信息进行作答，避免引入错误信息。**"})
    messages.append({"role": "user", "content": prompt})
    return messages

def generate_label(chat_completer: Any, question_dict: dict) -> str | None:
    """
    Generates the label / answer to `question` using `context` and deepseek.
    """
    chunk4 = question_dict["oracle_chunks"]
    question = question_dict["question"]
    messages = gen_answer_prompt(question, chunk4)
    response = chat_completer(
        model=os.getenv("GENERATION_MODEL"),
        messages=messages,
        n=1,
        temperature=0,
        max_tokens=2048,
    )
    reasoning_content = response.choices[0].message.reasoning_content
    response = response.choices[0].message.content
    return response, reasoning_content, question_dict

def generate_label_with_sorted_chunk(chat_completer: Any, question_dict: dict) -> str | None:
    """
    Generates the label / answer to `question` using `context` and deepseek.
    """
    sorted_chunks = question_dict["sorted_chunks"]
    question = question_dict["question"]
    messages = gen_answer_prompt(question, sorted_chunks)
    response = chat_completer(
        model=os.getenv("GENERATION_MODEL"),
        messages=messages,
        n=1,
        temperature=0,
        max_tokens=2048,
    )
    reasoning_content = response.choices[0].message.reasoning_content
    response = response.choices[0].message.content
    return response, reasoning_content, question_dict
               
def save_answers(response, reasoning_content, question_dict, article_name, answers_path):
    question_dict["content"] = response
    question_dict["reasoning_content"] = reasoning_content
    # 删除 question_dict 的 oracle_chunk 属性
    del question_dict["oracle_chunks"]
    # 判断 filename 是否存在，如果存在则追加写入，否则创建新文件
    if os.path.exists(answers_path):
        with open(answers_path, 'r', encoding="utf-8") as f:
            existing_questions = json.load(f)
        # 检查 article_name 是否已经存在于 questions 中
        if article_name in existing_questions:
            existing_questions[article_name].append(question_dict)
        else:
            existing_questions[article_name] = [question_dict]
        with open(answers_path, 'w', encoding="utf-8") as f:
            json.dump(existing_questions, f, ensure_ascii=False, indent=4)
    else:
        with open(answers_path, 'w', encoding="utf-8") as f:
            json.dump({article_name: [question_dict]}, f, ensure_ascii=False, indent=4)
    # print(f"Answers saved to {answers_path}")

def gen_answer_v3(questions_path, chat_model, answers_path):
    if os.path.exists(answers_path):
        print(f"{answers_path} exists. Skipping...")
        return 
    articles_questions = load_articles(questions_path)
    for a_name,question_dicts in articles_questions.items():
        futures = []
        num_questions = len(question_dicts)
        with tqdm(total=num_questions, desc="Answering", unit="ans") as pbar:
            with ThreadPoolExecutor(max_workers=8) as executor:
                for question_dict in question_dicts:
                    futures.append(executor.submit(generate_label_with_sorted_chunk, chat_model, question_dict))
                for future in as_completed(futures):
                    response, reasoning_content, question_dict = future.result()
                    pbar.update(1)
                    save_answers(response, reasoning_content, question_dict, a_name, answers_path)
                print(f"done {a_name} answers.")

def convert_to_jsonl(questions_path, output_jsonl_path):
    """
    将questions_path文件转换为jsonl格式的批量请求文件
    :param questions_path: 原始问题文件路径
    :param output_jsonl_path: 输出的jsonl文件路径
    """
    # 加载原始问题数据
    articles_questions = load_articles(questions_path)
    
    with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
        request_id = 1
        
        # 遍历所有文章和问题
        for article_name, question_dicts in articles_questions.items():
            for question_dict in question_dicts:
                # 获取问题和上下文
                question = question_dict["question"]
                chunks = question_dict.get("sorted_chunks", question_dict.get("oracle_chunks", []))
                chunk_str = get_chunkstr(chunks)
                
                # 构建prompt
                prompt = prompt_templates["deepseek-v2"].replace("{question}", question).replace("{context}", chunk_str)
                
                # 构建请求体
                request_body = {
                    "custom_id": f"request-{request_id}",
                    "body": {
                        "messages": [
                            {"role": "system", "content": "你是一个专业的RAG代理（Retrieval-Augmented Generation agent），你的核心任务是**基于用户的问题和提供的相关上下文切片信息，提供准确且可靠的答案**。请注意：切片信息是可能相关的参考资料，内容可能庞大且杂乱，不一定包含目标答案，也可能包含干扰信息。**请务必仔细审阅每个切片内容，严格基于可验证的信息进行作答，避免引入错误信息。**"},
                            {"role": "user", "content": prompt}
                        ]
                    }
                }
                
                # 写入jsonl文件
                f_out.write(json.dumps(request_body, ensure_ascii=False) + '\n')
                request_id += 1
        print(f"转换完成，结果已保存到 {output_jsonl_path}")


def process_response_file(response_file_path, questions_path, answers_path):
    """
    处理响应结果文件并保存到answer_path
    
    :param response_file_path: 响应结果文件路径(jsonl格式)
    :param questions_path: 原始问题文件路径(用于获取原始问题信息)
    :param answers_path: 结果保存路径
    """
    # 加载原始问题数据
    articles_questions = load_articles(questions_path)
    
    # 创建问题ID到问题详情的映射
    question_id_map = {}
    request_id = 1
    for article_name, question_dicts in articles_questions.items():
        for question_dict in question_dicts:
            custom_id = f"request-{request_id}"
            question_id_map[custom_id] = (article_name, question_dict)
            request_id += 1
    
    # 读取响应结果文件
    responses = []
    with open(response_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            responses.append(json.loads(line.strip()))
    
    # 处理每个响应
    for response_data in tqdm(responses, desc="Processing responses"):
        custom_id = response_data['custom_id']
        
        if custom_id not in question_id_map:
            print(f"Warning: Custom ID {custom_id} not found in original questions")
            continue
        
        article_name, question_dict = question_id_map[custom_id]
        
        if response_data['error'] is not None:
            print(f"Error in response {custom_id}: {response_data['error']}")
            continue
        
        try:
            # 提取响应内容
            response_body = response_data['response']['body']
            choices = response_body['choices']
            
            if not choices:
                print(f"No choices in response {custom_id}")
                continue
                
            # 获取主要内容和推理内容
            content = choices[0]['message']['content']
            reasoning_content = choices[0]['message'].get('reasoning_content', '')
            
            # 保存结果
            save_answers(content, reasoning_content, question_dict, article_name, answers_path)
            
        except Exception as e:
            print(f"Error processing response {custom_id}: {str(e)}")
            continue
    print(f"处理完成，结果已保存到 {answers_path}")
    

