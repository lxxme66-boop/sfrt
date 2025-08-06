import os
import re
import json
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

def is_to_drop(text: str) -> bool:
    """
    判断文本是否应该被过滤掉
    
    过滤规则：
    1. 空文本或过短文本
    2. 参考文献、致谢等无关内容
    3. 非中文内容（中文字符比例过低）
    """
    text = text.strip()[:10]
    
    # 空文本或特定模式
    patterns = ["", "#"]
    for pattern in patterns:
        if text == pattern:
            return True
    
    # 正则表达式模式匹配
    regex_patterns = [
        'http://www.cnki.net', 'https://www.cnki.net',
        r'^\[\d{1,4}\]', r'^\*\s+\[\d{1,4}\]', r'^\*\s+\(\d{1,4}\)',
        r'^致谢.*[0-9]$', r'.*致\s*谢.*', r'.*目\s*录.*',
        r'\.\.\.\.\.\.\.\.', r'\…\…\…',
        r"(http://www|doi:|DOI:|please contact)",
        r"(work was supported by|study was supported by|China|Republic of Korea|Authorized licensed use limited to)",
        r"\s[1-9]\d{5}(?!\d)",  # 邮编
        r"\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*",  # 邮箱
        r"(received in revised form|All rights reserved|©)",
        r"[a-zA-z]+://[^\s]*",  # URL
        r"(13[0-9]|14[5|7]|15[0|1|2|3|5|6|7|8|9]|18[0|1|2|3|5|6|7|8|9])\d{8}",  # 手机号
        r"\d{3}-\d{8}|\d{4}-\d{7}",  # 电话号码
        # 学位论文相关
        r'^分\s*类\s*号', r'^学\s*科\s*专\s*业', r'^签\s*字\s*日\s*期',
        r'^申\s*请\s*人\s*员\s*姓\s*名', r'^日\s*期', r'^指\s*定\s*教\s*师',
        r'学\s*位\s*论\s*文', r'^工\s*作\s*单\s*位', r'^电\s*话',
        r'^通讯地址', r'^邮\s*编', r'^中\s*图\s*分\s*类\s*号',
        r'^评\s*阅\s*人', r'^签\s*名', r'^分\s*类\s*号', r'^密\s*级',
        r'^学\s*号', r'^院\s*系', r'^委\s*员', r'^国内图书分类号',
        r'^国际图书分类号', r'^导\s*师', r'^申\s*请\s*学\s*位',
        r'^工\s*程\s*领\s*域', r'^所\s*在\s*单\s*位', r'^答\s*辩',
        r'^作\s*者', r'^专\s*业', r'^保\s*密', r'^不\s*保\s*密',
        # 参考文献格式
        r'^\[?\d+\]?', r'^\s*\[?\d+\]?', r'^\［?\d+\］?', r'^\s*\［?\d+\］?'
    ]
    
    for pattern in regex_patterns:
        if re.search(pattern, text):
            return True
    
    # 关键词匹配
    keywords = [
        '申请号', '专利号', '已录用', '学报', '研究生', '已收录', '攻读',
        '第一作者', '第二作者', '参考文献', '专业名称', '863项目', '导师',
        '教授', '感谢', '致谢', '谢谢', '指导', '朋友', '家人', '亲友',
        '师弟', '师妹', '老师', '同学', '父母', '充实', '答辩', '祝愿',
        '独创性声明', '作者签名', '发表文章', '论文使用授权声明', '本人',
        '知网', '论文使用权', '发表的论文', '申请的专利', '申请专利',
        '发表的文章', '发表学术论文', '发表论文', '参与科研项目', '作者简介',
        '三年的学习', '大学硕士学位论文', '大学博士学位论文', '涉密论文',
        '学校代码', '论文提交日期', '委员：', '中图分类号', '原创性声明',
        '顺利完成学业', 'All rights reserved', '参 考 文 献', '参考文献',
        '所在学院', '国家自然科学基金', '教育部重点学科建设', '时间飞梭',
        '时光飞梭', '光阴似箭', '白驹过隙', '论文版权', '本学位论文',
        '使用授权书', 'References', 'Acknowledgements', '论文著作权',
        '保密的学位论文', '中国第一所现代大学', '参加科研情况', '独 创 性 声 明',
        '论文使用授权', '获得的专利', '家庭的爱', '文献标识码', '文章编号'
    ]
    
    for keyword in keywords:
        if re.findall(keyword, text):
            return True
    
    # 判断中文字符比例
    chinese_count = 0
    for char in text:
        if '\u4e00' <= char <= '\u9fa5':
            chinese_count += 1
    
    if len(text) > 0 and chinese_count / len(text) < 0.01:
        return True
    
    return False

def preprocess_document(content: str, concatenation: str = "\n") -> str:
    """
    预处理文档内容，过滤无关内容
    """
    new_lines = []
    lines = content.split("\n")
    
    for line in lines:
        if not is_to_drop(line):
            new_lines.append(line)
    
    return concatenation.join(new_lines)

def get_document_quality_prompt() -> str:
    """文档质量评估prompt模板"""
    return """你的任务是依据以下评分规则对文本质量进行打分，并输出最终得分。评分流程如下：
1.依照每个标准依次评估文本。对每个子问题如实作答。若对某子问题答案为明确 "是"，则按标准相应加分或减分;
2.记录每个标准的累计分数，得出总分;
3.依据以下说明，将最终评估结果整理为有效的 JSON 对象。

## 评分标准：
1.标准 1：问题完整性
(1) 内容无清晰主要问题，或缺乏足够线索得出正确答案，得 0 分。
(2) 内容包含一个主要问题，且有足够线索得出正确答案，得 + 1 分。
(3) 文本体现多位作者间互动与讨论，如提出答案、评估反思答案、回应批评、修订编辑答案，得 + 1 分。

2.标准 2：问题复杂性和技术深度
(1) 内容难度为大学水平或以下，得 0 分。
(2) 内容难度为研究生水平或以上，仅领域专家能理解，得 + 1 分。
(3) 所讨论问题极具挑战性，高技能非专家花费 30 分钟上网搜索或阅读文献后，仍无法完全理解问题或给出正确答案，得 + 1 分。

3.标准 3：技术正确性和准确性
(1) 文本含显著技术错误或不准确，得 -1 分。
(2) 文本有一定技术正确性，但存在明显缺陷或遗漏（如单位错误、推导不完整），得 0 分。
(3) 文本技术正确，但有小缺陷或遗漏（如小代数错误、解释不完整），得 + 0.5 分。
(4) 文本技术高度正确，解释清晰准确（如精确定义、完整推导），得 + 0.5 分。
(5) 文本技术卓越正确，解释严格精确（如形式化证明、精确计算），得 + 1 分。

4.标准 4：思维和推理
(1) 文本无任何思维或推理迹象，得 -1 分。
(2) 文本展现一些基本思维和推理能力（如直接应用已知技术、简单分析问题），得 + 0.5 分。
(3) 文本展现一定思维和推理能力（如考虑多种解决方法、讨论不同方案权衡），得 + 0.5 分。
(4) 文本展现显著思维和推理能力（如通过多步推理链解决复杂问题、运用专业科学领域高级推理模式），得 + 1 分。
(5) 文本展现卓越思维和推理能力（如以高度创新方式解决专业领域复杂问题、结合多种推理技术对问题进行新抽象），得 + 1 分。

最终评判标准：若各项标准得分均大于零，且标准 4 得分大于等于 1 分，则该文本内容适合生成逻辑推理问题。

[文本内容的开始]
{document_content}
[文本内容的结束]

格式要求：只输出文本内容是否适合生成复杂推理问题，不输出任何别的内容。并且是否适合严格按照以下格式进行输出：
【是】或者【否】。不要输出为空，不要输出其他内容，输出是或否时，要带上【】符号进行输出。
"""

def to_batch(lst: List, batch_size: int) -> List[List]:
    """将列表分批"""
    return [lst[i:i+batch_size] for i in range(0, len(lst), batch_size)]

def evaluate_documents_batch(
    documents: List[Dict[str, str]], 
    llm_client, 
    batch_size: int = 32
) -> List[Dict[str, any]]:
    """
    批量评估文档质量
    
    使用vLLM或批量API调用来提高效率
    """
    results = []
    
    # 准备批量输入
    prompts = []
    for doc in documents:
        prompt = get_document_quality_prompt().format(
            document_content=doc['content']
        )
        prompts.append(prompt)
    
    # 批量推理
    if hasattr(llm_client, 'generate'):  # vLLM客户端
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=100,
            stop_token_ids=llm_client.stop_token_ids if hasattr(llm_client, 'stop_token_ids') else None
        )
        
        # vLLM批量生成
        outputs = llm_client.generate(prompts, sampling_params, use_tqdm=True)
        
        for i, output in enumerate(outputs):
            score_text = output.outputs[0].text.strip()
            is_high_quality = '【是】' in score_text
            
            results.append({
                'filename': documents[i]['filename'],
                'content': documents[i]['content'],
                'is_high_quality': is_high_quality,
                'score_text': score_text
            })
    
    else:  # 标准OpenAI客户端
        # 使用线程池并发调用
        def evaluate_single(doc, prompt):
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
                
                score_text = response.choices[0].message.content.strip()
                is_high_quality = '【是】' in score_text
                
                return {
                    'filename': doc['filename'],
                    'content': doc['content'],
                    'is_high_quality': is_high_quality,
                    'score_text': score_text
                }
            except Exception as e:
                logger.error(f"评估文档 {doc['filename']} 时出错: {e}")
                return {
                    'filename': doc['filename'],
                    'content': doc['content'],
                    'is_high_quality': False,
                    'score_text': f"评估失败: {str(e)}"
                }
        
        # 批量处理
        batches = to_batch(list(zip(documents, prompts)), batch_size)
        for batch in tqdm(batches, desc="评估文档质量"):
            batch_results = []
            for doc, prompt in batch:
                result = evaluate_single(doc, prompt)
                batch_results.append(result)
            results.extend(batch_results)
    
    return results

def filter_high_quality_documents_batch(
    data_dir: str,
    filtered_data_dir: str,
    llm_client,
    batch_size: int = 32,
    max_workers: int = 4,
    skip_preprocessing: bool = False,
    judge_output_path: Optional[str] = None
) -> str:
    """
    批量筛选高质量文档
    
    Args:
        data_dir: 原始文档目录
        filtered_data_dir: 筛选后文档保存目录
        llm_client: LLM客户端（支持vLLM或OpenAI）
        batch_size: 批量处理大小
        max_workers: 并发数量
        skip_preprocessing: 是否跳过文本预处理
        judge_output_path: 评估结果保存路径（用于增量处理）
    
    Returns:
        筛选后的文档目录路径
    """
    if not os.path.exists(data_dir):
        logger.error(f"原始文档目录不存在: {data_dir}")
        return data_dir
    
    # 创建筛选后文档目录
    os.makedirs(filtered_data_dir, exist_ok=True)
    
    # 获取所有文档文件
    doc_files = []
    for ext in ['.md', '.txt']:
        doc_files.extend([f for f in os.listdir(data_dir) if f.endswith(ext)])
    
    logger.info(f"找到 {len(doc_files)} 个文档待筛选")
    
    # 检查已处理的文档（增量处理）
    already_processed = set()
    if judge_output_path and os.path.exists(judge_output_path):
        with open(judge_output_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                already_processed.add(data['filename'])
        logger.info(f"已处理 {len(already_processed)} 个文档，跳过")
    
    # 准备待处理文档
    documents_to_process = []
    for filename in doc_files:
        if filename in already_processed:
            continue
        
        try:
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 文本预处理
            if not skip_preprocessing:
                content = preprocess_document(content)
            
            # 检查处理后的内容是否过短
            if len(content.strip()) < 100:
                logger.info(f"✗ 文档内容过短，跳过: {filename}")
                continue
            
            documents_to_process.append({
                'filename': filename,
                'content': content
            })
            
        except Exception as e:
            logger.error(f"读取文档 {filename} 时出错: {e}")
    
    logger.info(f"准备评估 {len(documents_to_process)} 个文档")
    
    # 批量评估文档质量
    evaluation_results = evaluate_documents_batch(
        documents_to_process,
        llm_client,
        batch_size
    )
    
    # 保存评估结果（用于增量处理）
    if judge_output_path:
        with open(judge_output_path, 'a', encoding='utf-8') as f:
            for result in evaluation_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # 复制高质量文档到筛选目录
    high_quality_count = 0
    for result in evaluation_results:
        if result['is_high_quality']:
            src_path = os.path.join(data_dir, result['filename'])
            dst_path = os.path.join(filtered_data_dir, result['filename'])
            
            # 保存筛选后的文档
            with open(dst_path, 'w', encoding='utf-8') as f:
                f.write(result['content'])
            
            high_quality_count += 1
            logger.info(f"✓ 高质量文档: {result['filename']}")
        else:
            logger.info(f"✗ 低质量文档: {result['filename']}")
    
    # 处理已经评估过的文档（从judge_output_path加载）
    if judge_output_path and os.path.exists(judge_output_path):
        with open(judge_output_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data['filename'] not in [d['filename'] for d in documents_to_process]:
                    if data.get('is_high_quality', False):
                        # 如果之前评估为高质量，也复制到筛选目录
                        src_path = os.path.join(data_dir, data['filename'])
                        dst_path = os.path.join(filtered_data_dir, data['filename'])
                        if os.path.exists(src_path) and not os.path.exists(dst_path):
                            with open(src_path, 'r', encoding='utf-8') as src_f:
                                content = src_f.read()
                            with open(dst_path, 'w', encoding='utf-8') as dst_f:
                                dst_f.write(content)
                            high_quality_count += 1
    
    total_processed = len(evaluation_results) + len(already_processed)
    logger.info(f"文档筛选完成: {high_quality_count}/{total_processed} 个高质量文档")
    
    # 保存筛选结果统计
    stats = {
        "total_documents": len(doc_files),
        "processed_documents": total_processed,
        "high_quality_documents": high_quality_count,
        "quality_rate": high_quality_count / total_processed if total_processed > 0 else 0,
        "filtered_files": os.listdir(filtered_data_dir),
        "batch_size": batch_size,
        "preprocessing_enabled": not skip_preprocessing
    }
    
    stats_path = os.path.join(filtered_data_dir, "quality_filter_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    return filtered_data_dir