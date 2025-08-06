#!/usr/bin/env python3
"""
Test script for local LLM setup
测试本地大模型配置
"""

import os
import sys

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

def test_vllm_import():
    """Test if vLLM can be imported"""
    try:
        import vllm
        print("✓ vLLM 已成功导入")
        print(f"  版本: {vllm.__version__}")
        return True
    except ImportError:
        print("✗ vLLM 未安装")
        print("  请运行: pip install vllm")
        return False

def test_embeddings():
    """Test if HuggingFace embeddings work"""
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("\n✓ HuggingFace embeddings 可用")
        
        # Try to initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",  # Use smaller model for testing
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Test embedding
        test_text = "这是一个测试文本"
        embedding = embeddings.embed_query(test_text)
        print(f"  嵌入维度: {len(embedding)}")
        return True
        
    except Exception as e:
        print(f"✗ HuggingFace embeddings 错误: {e}")
        return False

def test_vllm_client():
    """Test vLLM client"""
    try:
        from utils.vllm_client import MODEL_CONFIGS
        print("\n✓ vLLM 客户端配置可用")
        print("  支持的模型:")
        for model_name, config in MODEL_CONFIGS.items():
            print(f"    - {model_name}: {config['model_path']}")
        return True
    except Exception as e:
        print(f"✗ vLLM 客户端错误: {e}")
        return False

def main():
    print("=== 本地大模型环境测试 ===\n")
    
    # Run tests
    tests = [
        test_vllm_import(),
        test_embeddings(),
        test_vllm_client()
    ]
    
    # Summary
    print("\n=== 测试结果 ===")
    if all(tests):
        print("✓ 所有测试通过！可以使用本地大模型运行RAFT。")
        print("\n运行示例:")
        print("  bash raft_local.sh")
        print("  或")
        print("  python raft_local_llm.py --datapath data/your_file.pdf --output outputs_local")
    else:
        print("✗ 部分测试失败，请检查上述错误信息。")
        print("\n建议:")
        print("1. 确保已安装 vLLM: pip install vllm")
        print("2. 确保已安装其他依赖: pip install -r requirements.txt")
        print("3. 检查模型路径是否正确")

if __name__ == "__main__":
    main()