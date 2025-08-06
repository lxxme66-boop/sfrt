# -*- coding: utf-8 -*-
import os, re
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
import json
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams
import numpy as np

def is_to_drop(text):
    
    text = text.strip()[:10]    
    patterns = ["", "#"]
    for pattern in patterns:
        if text == pattern:
            return True 
    patterns = ['http://www.cnki.net', 'https://www.cnki.net','^\[\d{1,4}\]', '^\*\s+\[\d{1,4}\]', '^\*\s+\(\d{1,4}\)', 
                '^致谢.*[0-9]$', '.*致\s*谢.*','.*目\s*录.*','\.\.\.\.\.\.\.\.', '\…\…\…',r"(http://www|doi:|DOI:|please contact)",
                r"(work was supported by|study was supported by|China|Republic of Korea|Authorized licensed use limited to)",
                r"\s[1-9]\d{5}(?!\d)",  # 邮编
                r"\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*", r"(received in revised form|All rights reserved|©)", r"[a-zA-z]+://[^\s]*",
                r"(13[0-9]|14[5|7]|15[0|1|2|3|5|6|7|8|9]|18[0|1|2|3|5|6|7|8|9])\d{8}", r"\d{3}-\d{8}|\d{4}-\d{7}",
                '^分\s*类\s*号', '^学\s*科\s*专\s*业', '^签\s*字\s*日\s*期', '^申\s*请\s*人\s*员\s*姓\s*名',
                '^日\s*期', '^指\s*定\s*教\s*师', '学\s*位\s*论\s*文', '^工\s*作\s*单\s*位', '^电\s*话', '^通讯地址', '^邮\s*编', 
                '^中\s*图\s*分\s*类\s*号', '^评\s*阅\s*人', '^签\s*名', '^分\s*类\s*号', '^密\s*级', '^学\s*号', '^院\s*系', 
                '^委\s*员', '^国内图书分类号', '^国际图书分类号', '^导\s*师', '^申\s*请\s*学\s*位', '^工\s*程\s*领\s*域', '^所\s*在\s*单\s*位', 
                '^答\s*辩',  '^作\s*者', '^专\s*业', '^保\s*密', '^不\s*保\s*密', '^硕\s*土\s*姓\s*名', '^导\s*师', '^职\s*称', '^声\s*明', 
                '^申请学位', '^学科、专业', '^学\s*校\s*代\s*码', '^邢\s*坤\s*太\s*学', '^学\s*科\s*门\s*类', '^培\s*养\s*院\s*系',
                '^研\s*究\s*生', '^专\s*业', '^完\s*成\s*日\s*期', '^年\s*月\s*日', '^审\s*级', '^单\s*位\s*代\s*码', '^密\s*码', 
                '^学\s*位\s*授\s*予', '^校\s*址', '^授\s*予', '^论\s*文\s*分\s*类\s*号', '^研\s*突\s*生', '^研\s*究\s*方\s*向:', 
                '^研\s*究\s*生', '^学\s*校\s*代\s*号', '^主\s*席', '^U\s*D\s*C', '^U.D.C','^论\s*文\s*起\s*止', '^论\s*文\s*样\s*纸', 
                '^完\s*成\s*时\s*间', '^学\s*校\s*编\s*码', '^声\s*明\s*人', '^分\s*类\s*号', '^培\s*养\s*单\s*位', '^提\s*交\s*论\s*文', 
                '^资\s*助', '^学科(专业)', '^提\s*交\s*日\s*期', '^学\s*科\s*名\s*称', '^课\s*题\s*人', '^学\s*科\s*门\s*类', 
                '^一\s*级\s*学\s*科', '^学\s*位\s*申\s*请', '^学\s*院\s*名\s*称', '^主\s*任', '^院\s*系', '^专\s*业', '^姓\s*名', 
                '^完\s*成\s*日\s*期', '^作\s*者', '^申\s*请\s*学\s*位', '^工\s*程\s*领\s*域', '^学\s*科\s*名\s*称', '^领\s*域', '^学\s*院', 
                '^提\s*交\s*日\s*期', '^授\s*予\s*学\s*位', '^学\s*科', '^所\s*在\s*单\s*位',  '^电\s*子\s*邮\s*箱', '^联\s*系\s*地\s*址',
#                r'^!\[\](images/.*',  # 多余（可在检查有无中文字符时去掉）且导致报错
                r'^\[?\d+\]?',  r'^\s*\[?\d+\]?', r'^\［?\d+\］?', r'^\s*\［?\d+\］?' # mineru解析的参考文献格式
                ]
    for pattern in patterns:
        if re.search(pattern, text):
            return True
        
    patterns = ['申请号|专利号|已录用|学报|研究生|已收录|攻读|第一作者|第二作者|参考文献|专业名称|863项目|导师',
                '教授|感谢|致谢|谢谢|指导|朋友|家人|亲友|师弟|师妹|老师|同学|父母|充实|答辩|祝愿|独创性声明|作者签名',
                '发表文章|论文使用授权声明|本人|知网|论文使用权|发表的论文|申请的专利|申请专利|发表的文章|发表学术论文|发表论文',
                '参与科研项目|作者简介|三年的学习|大学硕士学位论文|大学博士学位论文|涉密论文|学校代码|论文提交日期|委员：|中图分类号',
                '原创性声明|顺利完成学业|All rights reserved|参 考 文 献|参考文献|所在学院|国家自然科学基金|教育部重点学科建设',
                '时间飞梭|时光飞梭|光阴似箭|白驹过隙|论文版权|本学位论文|使用授权书|References|Acknowledgements',
                '论文著作权|保密的学位论文|中国第一所现代大学|参加科研情况|独 创 性 声 明|论文使用授权|获得的专利|家庭的爱|文献标识码|文章编号'
                ]
    for pattern in patterns:
        if re.findall(pattern, text):
            return True   
        
    """
    判断是否不包含中文字符（暂时把公式也去掉）
    """
    num = 0
    for t in text:
        if  '\u4e00' <= t <= '\u9fa5':
            num += 1    
    if num / len(text) < 0.01:
        return True
                
    return False

def drop(texts, concatenation= "\n"):
    new_texts = []
    texts = texts.split("\n")
    for i, text in enumerate(texts):
        if not is_to_drop(text):
            new_texts.append(text)
    return concatenation.join(new_texts)

def extract(folder):
    files = os.listdir(folder)
    files.sort()
    return files

def load_json(file_path):
    with open(file_path, "r+", encoding="utf8") as load_f:
        dicts = json.load(load_f)
    dicts.sort(key=lambda s: s["id"])
    return dicts

def load_paper(file_path):
    with open(file_path, "r", encoding="utf8") as f:
        content = f.read()
        deal_content = drop(content) 
        
    return deal_content

def dcts2json(dcts, save_path):
    with open(save_path, 'w', encoding='utf8') as f:
        json.dump(dcts , f, indent=4, ensure_ascii=False) 

def to_batch(lst, groupsize):  # [a,b,c,d,e] -> [[a,b], [c,d], [e]], for batch inference
    return [lst[i:i+groupsize] for i in range(0,len(lst),groupsize)]

def ask(raw_folders, save_paths, model_len = 14 * 1024 , questioner = "qwq_32"):
    
    assert questioner in ["qw2_72", "qw2_72_awq","llama3.1_70", "qw2.5_32", "qw2.5_72", "qwq_32"]
    save_steps = 2
    batchsize = 32
    
    if questioner == "qw2_72":
        model_path = "/data/lc/openmodels/qw2_72b_instruct"
        model_stopid = "<|im_end|>"
    elif questioner == "qw2_72_awq":
        model_path = "/data/lc/openmodels/qw2_72b_instruct_awq"
        model_stopid = "<|im_end|>"
    elif questioner == "qw2.5_32":
        model_path = "/data/lc/openmodels/qw2.5_32b_instruct"
        model_stopid = "<|im_end|>"
    elif questioner == "qw2.5_72":
        model_path = "/data/lc/openmodels/qw2.5_72b_instruct"
        model_stopid = "<|im_end|>"
    elif questioner == "llama3.1_70":
        model_path = "/data/lc/openmodels/llama3.1_70b_instruct"
        model_stopid = "<|eot_id|>"
    elif questioner == "qwq_32":
        model_path = '/mnt/workspace/models/Qwen/QwQ-32B/'
        model_stopid = "<|im_end|>"

    tokenizer = AutoTokenizer.from_pretrained(model_path, rust_remote_code=True)
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(model_stopid)]
    llm = LLM(model= model_path, trust_remote_code=True, gpu_memory_utilization=0.95, tensor_parallel_size=2, max_model_len = model_len) # decrease max_model_len for oom
    sampling_params = SamplingParams(temperature = 0.3, repetition_penalty=1.1, max_tokens=2048, top_p= 0.9, stop_token_ids = terminators) 

    score_template = """你的职责是按照以下评分要求，对文本质量内容进行打分，并输出最终的得分分数，评分说明
按照每个标准逐步评估文本。对每个子问题提供诚实的回答。如果对某个子问题的回答是明确的“是”，则根据标准加分或减分。
记录每个标准的累计分数以获得总分。
根据以下说明，将最终评估结果总结为一个有效的 JSON 对象。
评分标准
标准 1：问题完整性

内容没有清晰的主要问题，或没有足够的线索来得出正确答案。（0 分）
内容包含一个主要问题，并有足够的线索来得出正确答案。（+1 分）
文本显示出多位作者之间的互动和讨论的证据，包括提出答案、评估和反思答案、回应批评、修订和编辑答案。（+1 分）
标准 2：问题复杂性和技术深度

内容的难度为大学水平或以下。（0 分）
内容的难度为研究生水平或以上，且只有领域专家才能理解。（+1 分）
所讨论的问题非常具有挑战性，即使是高技能的非专家在花费 30 分钟搜索互联网或阅读文献后，也无法完全理解问题或提供正确答案。（+1 分）
标准 3：技术正确性和准确性

文本包含显著的技术错误或不准确之处。（-1 分）
文本表现出一定的技术正确性，但存在明显的缺陷或遗漏（例如，单位错误、不完整的推导）。（0 分）
文本表现出技术正确性，但有一些小的缺陷或遗漏（例如，小的代数错误、不完整的解释）。（+0.5 分）
文本表现出高度的技术正确性，具有清晰且准确的解释（例如，精确的定义、完整的推导）。（+0.5 分）
文本表现出卓越的技术正确性，具有严格且精确的解释（例如，形式化证明、精确计算）。（+1 分）
标准 4：思维和推理

文本缺乏任何思维或推理的证据。（-1 分）
文本表现出一些基本的思维和推理能力（+0.5 分），例如：
对已知技术的直接应用。
对问题的简单分析。
文本表现出一些思维和推理能力（+0.5 分），例如：
考虑问题的多种解决方法。
讨论不同解决方案之间的权衡。
文本表现出显著的思维和推理能力（+1 分），例如：
通过多步推理链解决复杂问题。
在专业科学领域中常用的高级推理模式。
文本表现出卓越的思维和推理能力（+1 分），例如：
以高度创新和创造性的方式解决专业领域中的复杂问题。
结合多种推理和思维技术，对问题进行新的抽象。

最终评判：
如果各项标准都大于零，且标准4得分大于等于1分，则该文本内容适合生成逻辑推理问题

[文本内容的开始]
{academic_paper}
[文本内容的结束]

格式要求：只输出文本内容是否适合生成复杂推理问题，不输出任何别的内容。用中文输出，严格按照以下格式进行输出：
【是】或者【否】

""" 
   
    prompt_template = """你是一位半导体显示技术领域的资深专家，擅长从技术文献中提炼核心知识点。你的职责是从论文中生成问题和相应的答案，问题和相应的答案对需要提供给资深的人员学习，问题和相应的答案的质量要高。请根据输入的学术论文内容，生成3个需要逻辑推理才能解答的高质量技术问题，请确保这些问题能够直接从论文中找到答案。这些问题将用于资深研究人员的专业能力评估，需满足以下要求：
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
[学术论文的结束]
"""

    evaluator_template = """您是一位专家评估员，负责决定问题是否符合推理问题标准。您的评估必须结合给定文章内容和给定问题判断。
### 评估标准：
1. **因果性**：
- 问题需呈现完整技术逻辑链（如：机制A如何影响参数B，进而导致现象C）
2. **周密性**：
- 思维过程需要科学严谨，逐步思考，确保问题和对应的答案来源于论文的内容。
- 答案需要能在论文中完全找到详细的描述
3. **完整性**：
- 问题是否充分包含文章相关内容的所有方面？
- 问题描述要凝练简洁，且语义完整

[文章内容的开始]
{academic_paper}
[文章内容的结束]

[问题内容]
{academic_question}

格式要求：只输出问题是否符合标准，不输出任何别的内容。用中文输出，严格按照以下格式进行输出：
【是】或者【否】
"""

    for raw_folder, save_path in zip(raw_folders, save_paths):
        files = extract(raw_folder) # txt _files
        results = []
        already_ids = []
        to_do = []
        if os.path.exists(save_path):
            results = load_json(save_path)
            for already_sample in results:
                already_ids.append(already_sample["id"])
        print(len(already_ids),  len(results))
        
        for file in tqdm(files, desc = "check:" + raw_folder):
            if file.endswith("md"):
                if not (file in already_ids):
                    to_do.append(file)             
        batches = to_batch(to_do, batchsize)
        current_step = 0
        for batch in tqdm(batches, desc = "judge:" + raw_folder):
            paper_contents = []
            short_papers = []
            for paper in batch:
                try:
                    score_inputs = []
                    inputs = []
                    evaluator_inputs = []
                    paper_name = paper.split("_part")[0]
                    paper_content = load_paper(raw_folder + paper)
                    print(paper_name)
                    print(paper_content[-20:])
                
                    score_prompt = score_template.replace("{academic_paper}", paper_content)
                    score_messages = [{"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"}, {"role": "user", "content": score_prompt}]
                
                    if len(tokenizer.encode(score_prompt)) < model_len - 1024:
                        score_inputs.append(tokenizer.apply_chat_template(score_messages,tokenize=False,add_generation_prompt=True))

                    score_outputs = llm.generate(score_inputs, sampling_params, use_tqdm = False)
                    assert len(score_outputs) == len(score_inputs)
                    score_text = score_outputs[0].outputs[0].text.split('\n')[-1]
                    print('score_text', score_text)

                    if '【是】' in score_text:       
                        prompt = prompt_template.replace("{academic_papername}", paper_name).replace("{academic_paper}", paper_content)
                        messages = [{"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"}, {"role": "user", "content": prompt}]

                        if len(tokenizer.encode(prompt)) < model_len - 1024:
                            inputs.append(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True))
                            paper_contents.append(paper_content)
                            short_papers.append(paper)
                    
                        outputs = llm.generate(inputs, sampling_params, use_tqdm = False)
                        assert len(inputs) == len(outputs)
   
                        question_list = outputs[0].outputs[0].text.split("</think>")[1].strip().split('\n')
                        for question in question_list:
                            if len(question.strip()) < 5 : continue
                            evaluator_prompt = evaluator_template.replace("{academic_question}", question).replace("{academic_paper}", paper_content)
                            evaluator_messages = [{"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"}, {"role": "user", "content": evaluator_prompt}]
                            if len(tokenizer.encode(evaluator_prompt)) < model_len - 1024:
                                evaluator_inputs.append(tokenizer.apply_chat_template(evaluator_messages,tokenize=False,add_generation_prompt=True))
                            evaluator_outputs = llm.generate(evaluator_inputs, sampling_params, use_tqdm = False)
                            assert len(evaluator_outputs) == len(evaluator_inputs)
                            evaluator_text = evaluator_outputs[0].outputs[0].text.split('\n')[-1]
                            print('evaluator_text', question, evaluator_text) 

                            if '【是】' in evaluator_text:
                                sample = dict()
                                sample["id"] = paper
                                sample["paper_content"]= paper_content
                                sample["question_output"] = question
                                results.append(sample)
                    """
                    for paper, output, paper_content in zip(short_papers, outputs, paper_contents):
                        sample = dict()
                        sample["id"] = paper
                        sample["paper_content"]= paper_content
                        sample["question_output"] = output.outputs[0].text
                        results.append(sample)
                    """
               
                except Exception as e:
                    print("The error is: ",e)
                    pass
            current_step = current_step + 1
            if current_step == save_steps:
                dcts2json(results, save_path)
                current_step = 0
            dcts2json(results, save_path)
                
ask(["/mnt/workspace/LLM/xuleliu/field_reinforcement/clean_data/"],
    ["/mnt/workspace/LLM/xuleliu/field_reinforcement/generate_question/0324.json"])
