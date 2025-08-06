@echo off
REM RAFT with Local LLM - Windows Run Script
REM 使用本地大模型运行RAFT，无需API密钥

REM 默认参数
SET MODEL_NAME=qwq_32
SET EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
SET GPU_MEMORY=0.95
SET TENSOR_PARALLEL=1

echo 示例1: 处理单个PDF文件
python raft_local_llm.py ^
  --datapath data/RAFT.pdf ^
  --output outputs_local ^
  --output-format hf ^
  --distractors 3 ^
  --p 1.0 ^
  --doctype pdf ^
  --chunk_size 512 ^
  --questions 2 ^
  --model-name %MODEL_NAME% ^
  --embedding-model %EMBEDDING_MODEL% ^
  --gpu-memory-utilization %GPU_MEMORY% ^
  --tensor-parallel-size %TENSOR_PARALLEL% ^
  --system-prompt-key deepseek-v2 ^
  --temperature 0.6 ^
  --top-p 0.95 ^
  --max-tokens 4096

pause