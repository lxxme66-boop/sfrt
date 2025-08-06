# 本地大模型使用指南

本指南说明如何在数据合成管道中使用本地大模型。

## 目录
1. [环境准备](#环境准备)
2. [方式一：使用vLLM（推荐）](#方式一使用vllm推荐)
3. [方式二：使用API服务](#方式二使用api服务)
4. [常见问题](#常见问题)

## 环境准备

### 安装依赖
```bash
# 安装vLLM（用于高性能推理）
pip install vllm

# 或者使用特定CUDA版本
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118
```

### 检查GPU
```bash
nvidia-smi
```

## 方式一：使用vLLM（推荐）

vLLM提供了最高的推理性能，特别适合批量处理。

### 1. 配置模型路径

编辑 `utils/vllm_client.py`，在 `MODEL_CONFIGS` 中添加你的模型：

```python
MODEL_CONFIGS = {
    # ... 现有配置 ...
    "my_model": {
        "model_path": "/path/to/your/model",
        "stop_token_ids": ["<|im_end|>"]  # 根据模型调整
    }
}
```

### 2. 运行脚本

使用提供的脚本运行：

```bash
# 给脚本执行权限
chmod +x run_syndata_v5_local.sh
chmod +x run_local_model_example.sh

# 运行本地模型
./run_syndata_v5_local.sh
```

### 3. 自定义命令

直接使用命令行参数：

```bash
python -m utils.syndata_pipeline_v5 \
  --data_dir "data" \
  --filtered_data_dir "data_filtered" \
  --chunks_path "outputs/chunks.json" \
  --chunk4_path "outputs/chunk4.json" \
  --topics_path "outputs/topics.json" \
  --questions_path "outputs/questions.json" \
  --validated_questions_path "outputs/validated_questions.json" \
  --answers_path "outputs/answers.json" \
  --syndatas_path "outputs/syndatas.json" \
  --start_idx 0 \
  --end_idx 100 \
  --use_vllm \
  --model_name "custom" \
  --model_path "/path/to/your/model" \
  --gpu_memory_utilization 0.9 \
  --tensor_parallel_size 4 \
  --max_model_len 32768 \
  --batch_size 16
```

### 参数说明

- `--use_vllm`: 启用vLLM推理引擎
- `--model_name`: 模型名称（预定义的或"custom"）
- `--model_path`: 模型文件夹路径（覆盖默认路径）
- `--gpu_memory_utilization`: GPU内存使用率（0-1）
- `--tensor_parallel_size`: 使用的GPU数量
- `--max_model_len`: 最大上下文长度
- `--batch_size`: 批处理大小

## 方式二：使用API服务

如果你已经有运行的模型API服务，可以直接使用。

### 1. 启动本地API服务

#### 使用vLLM API服务器
```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/your/model \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 4
```

#### 使用FastChat
```bash
python -m fastchat.serve.openai_api_server \
  --model-path /path/to/your/model \
  --port 8000
```

### 2. 配置环境变量

```bash
export COMPLETION_OPENAI_API_KEY="dummy"
export COMPLETION_OPENAI_BASE_URL="http://localhost:8000/v1"
```

### 3. 运行管道

```bash
# 使用API模式脚本
chmod +x run_local_api_model.sh
./run_local_api_model.sh

# 或直接运行（不带--use_vllm参数）
python -m utils.syndata_pipeline_v5 \
  --data_dir "data" \
  --filtered_data_dir "data_filtered" \
  --chunks_path "outputs/chunks.json" \
  --chunk4_path "outputs/chunk4.json" \
  --topics_path "outputs/topics.json" \
  --questions_path "outputs/questions.json" \
  --validated_questions_path "outputs/validated_questions.json" \
  --answers_path "outputs/answers.json" \
  --syndatas_path "outputs/syndatas.json" \
  --start_idx 0 \
  --end_idx 100 \
  --batch_size 8
```

## 支持的模型

### 预配置模型
- QwQ-32B
- Qwen2系列（7B, 14B, 72B）
- LLaMA3系列（8B, 70B）
- DeepSeek-v2

### 自定义模型
任何兼容Transformers的模型都可以使用，只需指定正确的路径。

## 性能优化建议

### GPU内存优化
```bash
# 单GPU，大模型
--gpu_memory_utilization 0.95 \
--tensor_parallel_size 1 \
--batch_size 4

# 多GPU，更大批量
--gpu_memory_utilization 0.9 \
--tensor_parallel_size 4 \
--batch_size 32
```

### 上下文长度
```bash
# 短上下文，更快速度
--max_model_len 8192

# 长上下文，更好质量
--max_model_len 65536
```

## 常见问题

### 1. CUDA内存不足
- 减小 `batch_size`
- 减小 `gpu_memory_utilization`
- 增加 `tensor_parallel_size`（如果有多GPU）

### 2. 模型加载失败
- 检查模型路径是否正确
- 确保模型文件完整（config.json, tokenizer文件等）
- 检查模型格式是否兼容

### 3. 推理速度慢
- 使用vLLM而不是API模式
- 增加 `batch_size`
- 使用更多GPU（增加 `tensor_parallel_size`）

### 4. 生成质量问题
- 调整模型的prompt模板
- 确保使用正确的stop tokens
- 考虑使用更大的模型

## 示例：使用不同模型

### Qwen2-7B
```bash
python -m utils.syndata_pipeline_v5 \
  --use_vllm \
  --model_name "qwen2-7b" \
  --model_path "/models/Qwen2-7B-Instruct" \
  --tensor_parallel_size 1 \
  --max_model_len 32768 \
  # ... 其他参数
```

### LLaMA-3-70B
```bash
python -m utils.syndata_pipeline_v5 \
  --use_vllm \
  --model_name "llama3-70b" \
  --model_path "/models/Meta-Llama-3-70B-Instruct" \
  --tensor_parallel_size 4 \
  --max_model_len 8192 \
  # ... 其他参数
```

### 自定义模型
```bash
python -m utils.syndata_pipeline_v5 \
  --use_vllm \
  --model_name "custom" \
  --model_path "/models/my-custom-model" \
  --tensor_parallel_size 2 \
  --max_model_len 16384 \
  # ... 其他参数
```

## 监控和调试

### 查看GPU使用情况
```bash
watch -n 1 nvidia-smi
```

### 查看日志
管道会输出详细的日志信息，包括：
- 模型加载进度
- 批处理进度
- 性能统计
- 错误信息

### 测试小批量
建议先用小批量测试：
```bash
--start_idx 0 --end_idx 10
```

成功后再处理更多数据。