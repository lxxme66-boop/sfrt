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
    },
    # 添加更多常见模型的默认配置
    "qwen2-7b": {
        "model_path": None,  # 需要用户指定
        "stop_token_ids": ["<|im_end|>"]
    },
    "qwen2-14b": {
        "model_path": None,
        "stop_token_ids": ["<|im_end|>"]
    },
    "qwen2-72b": {
        "model_path": None,
        "stop_token_ids": ["<|im_end|>"]
    },
    "llama3-8b": {
        "model_path": None,
        "stop_token_ids": ["<|eot_id|>"]
    },
    "llama3-70b": {
        "model_path": None,
        "stop_token_ids": ["<|eot_id|>"]
    },
    "deepseek-v2": {
        "model_path": None,
        "stop_token_ids": ["<|end_of_sentence|>"]
    },
    "custom": {
        "model_path": None,
        "stop_token_ids": None  # 需要用户指定
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
            - model_path: 模型路径（可选，如果提供则覆盖默认路径）
            - gpu_memory_utilization: GPU内存利用率
            - tensor_parallel_size: 张量并行大小
            - stop_token_ids: 停止词ID（可选）
            - max_model_len: 最大模型长度（可选）
    
    Returns:
        VLLMClient实例
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        logger.error("vLLM未安装，请运行: pip install vllm")
        raise
    
    model_name = model_config.get("model_name", "qwq_32")
    custom_model_path = model_config.get("model_path")
    gpu_memory_utilization = model_config.get("gpu_memory_utilization", 0.95)
    tensor_parallel_size = model_config.get("tensor_parallel_size", 4)
    max_model_len = model_config.get("max_model_len", 96 * 1024)  # 默认96K
    custom_stop_token_ids = model_config.get("stop_token_ids")
    
    # 获取模型配置
    if model_name not in MODEL_CONFIGS:
        logger.warning(f"未知的模型名称: {model_name}，使用custom配置")
        model_info = MODEL_CONFIGS["custom"]
    else:
        model_info = MODEL_CONFIGS[model_name].copy()
    
    # 如果提供了自定义路径，使用自定义路径
    if custom_model_path:
        model_path = custom_model_path
        logger.info(f"使用自定义模型路径: {model_path}")
    else:
        model_path = model_info.get("model_path")
        if not model_path:
            raise ValueError(f"模型 {model_name} 需要指定model_path参数")
    
    # 处理停止词
    stop_token_ids = custom_stop_token_ids or model_info.get("stop_token_ids")
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        logger.error(f"模型路径不存在: {model_path}")
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    
    logger.info(f"加载vLLM模型: {model_name}")
    logger.info(f"模型路径: {model_path}")
    logger.info(f"GPU内存利用率: {gpu_memory_utilization}")
    logger.info(f"张量并行大小: {tensor_parallel_size}")
    logger.info(f"最大模型长度: {max_model_len}")
    
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
        max_model_len=max_model_len
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