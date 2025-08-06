# RAFT运行脚本说明

本项目提供了多个运行脚本来处理文本文件并生成合成数据集。

## 脚本文件说明

1. **run_raft.py** - Python主运行脚本，提供批量处理功能
2. **run.sh** - Shell包装脚本，提供简化的命令行接口
3. **example_usage.sh** - 使用示例脚本

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境

创建 `.env` 文件并配置API密钥：

```bash
# OpenAI API配置
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=your_api_base_url  # 可选

# 其他模型配置
DEEPSEEK_API_KEY=your_deepseek_key
```

### 3. 运行脚本

#### 处理所有文本文件
```bash
./run.sh
```

#### 处理单个文件
```bash
./run.sh -f data/example.txt
```

#### 使用Python脚本（更多选项）
```bash
python3 run_raft.py --input data --questions 5 --chunk-size 1024
```

## 参数说明

### run.sh 参数

- `-h, --help` - 显示帮助信息
- `-i, --input DIR` - 输入目录或文件（默认: data）
- `-o, --output DIR` - 输出基础目录（默认: outputs）
- `-f, --file FILE` - 处理单个文件
- `-q, --questions NUM` - 每个块的问题数量（默认: 2）
- `-d, --distractors NUM` - 干扰项数量（默认: 3）
- `-c, --chunk-size SIZE` - 文本块大小（默认: 512）
- `-m, --model MODEL` - 补全模型（默认: deepseek-r1-250120）

### run_raft.py 参数

```
--input              输入文件或目录路径
--output-base        输出基础目录
--output-format      输出格式 (hf/completion/chat)
--distractors        干扰项数量
--p                  采样概率
--doctype            文档类型 (txt/pdf/json/api)
--chunk-size         文本块大小
--questions          每个块生成的问题数量
--completion-model   补全模型
--system-prompt-key  系统提示词键 (gpt/llama/deepseek/deepseek-v2)
--embedding-model    嵌入模型
--single-file        处理单个文件
```

## 输出格式

处理结果将保存在输出目录中，结构如下：

```
outputs/
├── 文件名_20240125_143022/
│   ├── train_dataset.jsonl
│   └── metadata.json
├── processing_log_20240125_143022.json
└── ...
```

## 支持的文件类型

- `.txt` - 纯文本文件
- `.md` - Markdown文件
- `.pdf` - PDF文档（需要指定 --doctype pdf）
- `.json` - JSON文件（需要指定 --doctype json）

## 模型支持

### 补全模型
- deepseek-r1-250120（默认）
- qwen-plus
- gpt-3.5-turbo
- gpt-4

### 嵌入模型
- text-embedding-v2
- doubao-embedding-large-text-250515

## 使用示例

查看更多使用示例：
```bash
./example_usage.sh
```

## 注意事项

1. 确保API密钥已正确配置
2. 大文件处理可能需要较长时间
3. 输出文件会自动按时间戳命名，避免覆盖
4. 处理日志会保存在输出目录中

## 故障排除

1. **找不到Python3**: 确保已安装Python 3.7+
2. **缺少依赖**: 运行 `pip install -r requirements.txt`
3. **API错误**: 检查.env文件中的API密钥配置
4. **内存不足**: 减小chunk-size参数值