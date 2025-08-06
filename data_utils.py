import json
import logging

def get_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

import os
def get_filenames(dir):
    filenames = [filename for filename in os.listdir(dir) if "_sft" in filename]
    print(f"filenames: {filenames}")
    return filenames

from collections import OrderedDict
def ordered_json_data(json_data):
    ordered_data = []
    for item in json_data:
        ordered_item = OrderedDict([
            ("instruction", item["instruction"]),
            ("input", item["input"]),
            ("output", item["output"])
        ])
        ordered_data.append(ordered_item)
    return ordered_data
    
def jsonl2json(mydir, json_path):
    filenames = get_filenames(mydir)
    json_data = []
    for filename in filenames:
        filepath = os.path.join(mydir, filename)
        json_data.extend(get_jsonl(filepath))

    print(f"total json data: {len(json_data)}")
    # 在保存前调用这个函数
    json_data = ordered_json_data(json_data)
    with open(json_path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(json_data, ensure_ascii=False, indent=4))
        print("JSONL file converted to JSON")

if __name__ == "__main__":
    jsonl_path = "outputs_jsonl/outputs_jsonl/"
    json_path = "outputs_jsonl/outputs_jsonl/outputs_05.json"
    # 把所有的jsonl文件合并到一个json文件中
    jsonl2json(jsonl_path, json_path)