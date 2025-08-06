import json

def merge_jsonl_files(input_files, output_file):
    """
    合并多个jsonl文件为一个新的jsonl文件
    
    参数:
        input_files: 要合并的jsonl文件路径列表
        output_file: 合并后的输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in input_files:
            with open(file_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    # 验证是否为有效的JSON行
                    try:
                        json.loads(line)
                        outfile.write(line)
                    except json.JSONDecodeError:
                        print(f"警告: 文件 {file_path} 中跳过无效的JSON行: {line.strip()}")
