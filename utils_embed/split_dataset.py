import json
import random
import os

# 设置路径
input_file = 'data_emb/output62hn.jsonl'
output_dir = 'data_emb'

os.makedirs(output_dir, exist_ok=True)

train_file = os.path.join(output_dir, 'train.jsonl')
dev_file = os.path.join(output_dir, 'dev.jsonl')
test_file = os.path.join(output_dir, 'test.jsonl')

# 拆分比例：80% 训练，10% 验证，10% 测试
train_ratio = 0.8
dev_ratio = 0.1
test_ratio = 0.1

# 读取所有数据
with open(input_file, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

# 打乱顺序确保分布均匀
random.shuffle(lines)

# 计算分割点
total = len(lines)
train_end = int(total * train_ratio)
dev_end = train_end + int(total * dev_ratio)

train_data = lines[:train_end]
dev_data = lines[train_end:dev_end]
test_data = lines[dev_end:]

# 写入文件
def write_lines(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line + '\n')

write_lines(train_file, train_data)
write_lines(dev_file, dev_data)
write_lines(test_file, test_data)

print(f"Total samples: {total}")
print(f"Train: {len(train_data)}")
print(f"Dev:   {len(dev_data)}")
print(f"Test:  {len(test_data)}")
print("Split complete.")