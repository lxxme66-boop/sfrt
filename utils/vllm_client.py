import os
import logging
from typing import Dict, List, Optional
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# 模型配置映射
MODEL_CONFIGS = {
    "qw2_72": {
        "model_path": "/data/lc/openmodels/qw2_72b_instruct",
        "stop_token_ids": ["<|im_end|>"]
    },
    "qw2_72_awq": {
        "model_path": "/data/lc/openmodels/qw2_72b_instruct_awq",
        "stop_token_ids": ["<|im_end|>"]
    },
    "qw2.5_32": {
        "model_path": "/data/lc/openmodels/qw2.5_32b_instruct",
        "stop_token_ids": ["<|im_end|>"]
    },
    "qw2.5_72": {
        "model_path": "/data/lc/openmodels/qw2.5_72b_instruct",
        "stop_token_ids": ["<|im_end|>"]
    },
    "llama3.1_70": {
        "model_path": "/data/lc/openmodels/llama3.1_70b_instruct",
        "stop_token_ids": ["<|eot_id|>"]
    },
    "qwq_32": {
        "model_path": "/mnt/workspace/models/Qwen/QwQ-32B/",
        "stop_token_ids": [151329, 151336, 151338]
    }
}

class VLLMClient:
    """vLLM客户端封装，提供统一的接口"""
    
    def __init__(self, llm, tokenizer, stop_token_ids=None):
        self.llm = llm
        self.tokenizer = tokenizer
        self.stop_token_ids = stop_token_ids
    
    def generate(self, prompts, sampling_params, use_tqdm=True):
        """生成文本，与vLLM的generate方法兼容"""
        return self.llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)
    
    def __call__(self, *args, **kwargs):
        """兼容OpenAI风格的调用"""
        # 这个方法主要用于兼容性，实际使用时应该用generate方法
        raise NotImplementedError("请使用generate方法进行批量推理")

def build_vllm_client(model_config: Dict) -> VLLMClient:
    """
    构建vLLM客户端
    
    Args:
        model_config: 模型配置字典，包含：
            - model_name: 模型名称
            - gpu_memory_utilization: GPU内存利用率
            - tensor_parallel_size: 张量并行大小
    
    Returns:
        VLLMClient实例
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        logger.error("vLLM未安装，请运行: pip install vllm")
        raise
    
    model_name = model_config.get("model_name", "qwq_32")
    gpu_memory_utilization = model_config.get("gpu_memory_utilization", 0.95)
    tensor_parallel_size = model_config.get("tensor_parallel_size", 4)
    
    # 获取模型配置
    if model_name not in MODEL_CONFIGS:
        logger.warning(f"未知的模型名称: {model_name}，使用默认配置")
        model_info = MODEL_CONFIGS["qwq_32"]
    else:
        model_info = MODEL_CONFIGS[model_name]
    
    model_path = model_info["model_path"]
    stop_token_ids = model_info["stop_token_ids"]
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        logger.error(f"模型路径不存在: {model_path}")
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    
    logger.info(f"加载vLLM模型: {model_name}")
    logger.info(f"模型路径: {model_path}")
    logger.info(f"GPU内存利用率: {gpu_memory_utilization}")
    logger.info(f"张量并行大小: {tensor_parallel_size}")
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    
    # 初始化vLLM
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=96 * 1024  # 96K上下文长度
    )
    
    # 创建客户端
    client = VLLMClient(llm, tokenizer, stop_token_ids)
    
    logger.info("vLLM客户端初始化完成")
    
    return client

def get_default_sampling_params():
    """获取默认的采样参数"""
    from vllm import SamplingParams
    
    return SamplingParams(
        temperature=0.6,
        repetition_penalty=1.1,
        min_p=0,
        top_p=0.95,
        top_k=40,
        max_tokens=4096
    )