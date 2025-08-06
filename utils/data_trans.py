import json
import pandas as pd
import re



def trans_syn2excel(syndata_path, output_path):
    # 1. 解析 JSON 数据
    with open(syndata_path, "r", encoding="utf-8") as file:
        json_data = json.load(file)
    
    # 2. 处理数据提取
    processed_data = []
    for entry in json_data:
        # 提取问题
        question = entry["question"]
        # 提取完整响应内容
        full_content = entry["reasoning_answer"]
        # 使用正则表达式分离推理和回答部分
        reasoning_match = re.search(r'<think>(.*?)</think>', full_content, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        # 提取答案部分：</think>之后的所有内容
        answer_start = full_content.find('</think>') + len('</think>')
        answer = full_content[answer_start:].strip()
        # 添加到处理后的数据集
        processed_data.append({
            "问题": question,
            "推理": reasoning,
            "回答": answer
        })
    
    # 3. 转换为 DataFrame
    df = pd.DataFrame(processed_data)
    # 4. 保存为 Excel 文件
    
    df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"Excel文件已生成: {output_path}")
    print(f"总处理条目数: {len(processed_data)}")
    
