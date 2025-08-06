import os
import sys
import json
import time
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 导入基础模块
from utils.common_utils import build_openai_client_chat, load_articles, get_chunkstr
from utils.article_chunks import gen_chunks
from utils.topic_concepts import trans_chunk4, gen_topics, gen_questions_with_topic_v3
from utils.answer_generation import gen_answer_v3
from utils.data_synthesis import syn_data_v2

# 导入增强模块
from utils.document_quality_filter_v5 import filter_high_quality_documents_batch
from utils.enhanced_question_generation_v5 import gen_enhanced_questions_with_validation_batch

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print(f"Python executable: {sys.executable}")

def get_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="增强版数据合成管道 v5 - 完整优化实现",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基础路径参数
    parser.add_argument("--data_dir", type=str, required=True, help="原始文档目录路径")
    parser.add_argument("--filtered_data_dir", type=str, default="data_filtered", help="筛选后文档保存目录")
    parser.add_argument("--chunks_path", type=str, required=True, help="文档分块保存路径")
    parser.add_argument("--chunk4_path", type=str, required=True, help="Chunk4数据保存路径")
    parser.add_argument("--topics_path", type=str, required=True, help="主题数据保存路径")
    parser.add_argument("--questions_path", type=str, required=True, help="原始问题保存路径")
    parser.add_argument("--validated_questions_path", type=str, required=True, help="验证后问题保存路径")
    parser.add_argument("--answers_path", type=str, required=True, help="答案数据保存路径")
    parser.add_argument("--syndatas_path", type=str, required=True, help="最终合成数据保存路径")
    
    # 处理范围参数
    parser.add_argument("--start_idx", type=int, default=0, help="处理文档的起始索引")
    parser.add_argument("--end_idx", type=int, default=None, help="处理文档的结束索引")
    
    # 功能开关参数
    parser.add_argument("--skip_document_filter", action="store_true", help="跳过文档质量筛选步骤")
    parser.add_argument("--skip_question_validation", action="store_true", help="跳过问题质量验证步骤")
    parser.add_argument("--skip_text_preprocessing", action="store_true", help="跳过文本预处理步骤")
    
    # 性能优化参数
    parser.add_argument("--max_workers", type=int, default=4, help="并发处理数量")
    parser.add_argument("--batch_size", type=int, default=32, help="批量推理大小")
    parser.add_argument("--use_vllm", action="store_true", help="使用vLLM进行高效推理")
    parser.add_argument("--model_name", type=str, default="qwq_32", help="使用的模型名称")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95, help="GPU内存利用率")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="张量并行大小")
    
    # 中间文件路径（用于增量处理）
    parser.add_argument("--judge_output_path", type=str, help="文档评估结果保存路径")
    parser.add_argument("--question_output_path", type=str, help="问题生成结果保存路径")
    parser.add_argument("--question_li_output_path", type=str, help="展开后问题保存路径")
    
    args = parser.parse_args()
    return args

def enhanced_data_synthesis_pipeline_v5(
    data_dir,
    filtered_data_dir,
    chunks_path,
    chunk4_path,
    topics_path,
    questions_path,
    validated_questions_path,
    answers_path,
    syndatas_path,
    start_idx=0,
    end_idx=None,
    skip_document_filter=False,
    skip_question_validation=False,
    skip_text_preprocessing=False,
    max_workers=4,
    batch_size=32,
    use_vllm=False,
    model_config=None,
    intermediate_paths=None
):
    """
    增强版数据合成管道 v5 - 完整优化实现
    
    主要改进：
    1. 批量推理优化（vLLM支持）
    2. 文本预处理
    3. 增量处理
    4. 更好的错误处理
    5. 详细的进度跟踪
    """
    pipeline_start = time.time()
    
    # 初始化模型客户端
    if use_vllm:
        logger.info("🚀 初始化vLLM推理引擎...")
        from utils.vllm_client import build_vllm_client
        llm_client = build_vllm_client(model_config)
    else:
        logger.info("🔧 使用标准OpenAI客户端...")
        llm_client = build_openai_client_chat()
    
    # 第0步：文档质量筛选（批量处理）
    if not skip_document_filter:
        logger.info("🔍 步骤1: 开始文档质量筛选（批量处理）...")
        filter_start = time.time()
        
        actual_data_dir = filter_high_quality_documents_batch(
            data_dir=data_dir,
            filtered_data_dir=filtered_data_dir,
            llm_client=llm_client,
            batch_size=batch_size,
            max_workers=max_workers,
            skip_preprocessing=skip_text_preprocessing,
            judge_output_path=intermediate_paths.get('judge_output_path') if intermediate_paths else None
        )
        
        filter_time = time.time() - filter_start
        logger.info(f"✅ 文档质量筛选完成，耗时: {filter_time/60:.2f}分钟")
    else:
        logger.info("⏭️ 跳过文档质量筛选步骤")
        actual_data_dir = data_dir
    
    # 第1步：文档分块
    logger.info("📄 步骤2: 开始文档分块...")
    chunk_start = time.time()
    
    gen_chunks(actual_data_dir, chunks_path, start_idx, end_idx)
    
    chunk_time = time.time() - chunk_start
    logger.info(f"✅ 文档分块完成，耗时: {chunk_time/60:.2f}分钟")
    
    # 第2步：Chunk4转换
    logger.info("🔄 步骤3: 开始Chunk4转换...")
    chunk4_start = time.time()
    
    trans_chunk4(chunks_path, chunk4_path)
    
    chunk4_time = time.time() - chunk4_start
    logger.info(f"✅ Chunk4转换完成，耗时: {chunk4_time/60:.2f}分钟")
    
    # 第3步：主题生成
    logger.info("🎯 步骤4: 开始主题生成...")
    topic_start = time.time()
    
    gen_topics(chunk4_path, topics_path, llm_client)
    
    topic_time = time.time() - topic_start
    logger.info(f"✅ 主题生成完成，耗时: {topic_time/60:.2f}分钟")
    
    # 第4步：增强版问题生成和验证（批量处理）
    logger.info("❓ 步骤5: 开始增强版问题生成和验证（批量处理）...")
    question_start = time.time()
    
    if not skip_question_validation:
        gen_enhanced_questions_with_validation_batch(
            topics_path=topics_path,
            questions_path=questions_path,
            validated_questions_path=validated_questions_path,
            llm_client=llm_client,
            chunk4_path=chunk4_path,
            batch_size=batch_size,
            max_workers=max_workers,
            question_output_path=intermediate_paths.get('question_output_path') if intermediate_paths else None,
            question_li_output_path=intermediate_paths.get('question_li_output_path') if intermediate_paths else None
        )
        actual_questions_path = validated_questions_path
    else:
        logger.info("⏭️ 跳过问题质量验证，使用原始问题生成")
        gen_questions_with_topic_v3(topics_path, questions_path, llm_client, chunk4_path)
        actual_questions_path = questions_path
    
    question_time = time.time() - question_start
    logger.info(f"✅ 问题生成和验证完成，耗时: {question_time/60:.2f}分钟")
    
    # 第5步：答案生成
    logger.info("💬 步骤6: 开始答案生成...")
    answer_start = time.time()
    
    gen_answer_v3(actual_questions_path, llm_client, answers_path)
    
    answer_time = time.time() - answer_start
    logger.info(f"✅ 答案生成完成，耗时: {answer_time/60:.2f}分钟")
    
    # 第6步：数据合成
    logger.info("🔗 步骤7: 开始数据合成...")
    syndata_start = time.time()
    
    syn_data_v2(answers_path, syndatas_path)
    
    syndata_time = time.time() - syndata_start
    logger.info(f"✅ 数据合成完成，耗时: {syndata_time/60:.2f}分钟")
    
    # 总结统计
    total_time = time.time() - pipeline_start
    logger.info("🎉 增强版数据合成管道v5执行完成！")
    logger.info("=" * 50)
    logger.info("⏱️  各步骤耗时统计:")
    if not skip_document_filter:
        logger.info(f"   文档筛选: {filter_time/60:.2f}分钟")
    logger.info(f"   文档分块: {chunk_time/60:.2f}分钟")
    logger.info(f"   Chunk4转换: {chunk4_time/60:.2f}分钟")
    logger.info(f"   主题生成: {topic_time/60:.2f}分钟")
    logger.info(f"   问题生成: {question_time/60:.2f}分钟")
    logger.info(f"   答案生成: {answer_time/60:.2f}分钟")
    logger.info(f"   数据合成: {syndata_time/60:.2f}分钟")
    logger.info(f"   总耗时: {total_time/60:.2f}分钟")
    logger.info("=" * 50)
    
    # 返回详细统计信息
    return {
        "filter_time": filter_time if not skip_document_filter else 0,
        "chunk_time": chunk_time,
        "chunk4_time": chunk4_time,
        "topic_time": topic_time,
        "question_time": question_time,
        "answer_time": answer_time,
        "syndata_time": syndata_time,
        "total_time": total_time,
        "use_vllm": use_vllm,
        "batch_size": batch_size
    }

if __name__ == "__main__":
    args = get_args()
    
    logger.info("🚀 启动增强版数据合成管道 v5")
    logger.info(f"📁 原始数据目录: {args.data_dir}")
    logger.info(f"📁 筛选数据目录: {args.filtered_data_dir}")
    logger.info(f"📄 最终输出: {args.syndatas_path}")
    logger.info(f"🔧 使用vLLM: {args.use_vllm}")
    logger.info(f"📦 批量大小: {args.batch_size}")
    
    # 创建必要的输出目录
    for path in [args.chunks_path, args.chunk4_path, args.topics_path,
                 args.questions_path, args.validated_questions_path,
                 args.answers_path, args.syndatas_path]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 准备模型配置
    model_config = {
        "model_name": args.model_name,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size
    }
    
    # 准备中间文件路径
    intermediate_paths = {
        "judge_output_path": args.judge_output_path,
        "question_output_path": args.question_output_path,
        "question_li_output_path": args.question_li_output_path
    }
    
    try:
        timing_stats = enhanced_data_synthesis_pipeline_v5(
            data_dir=args.data_dir,
            filtered_data_dir=args.filtered_data_dir,
            chunks_path=args.chunks_path,
            chunk4_path=args.chunk4_path,
            topics_path=args.topics_path,
            questions_path=args.questions_path,
            validated_questions_path=args.validated_questions_path,
            answers_path=args.answers_path,
            syndatas_path=args.syndatas_path,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            skip_document_filter=args.skip_document_filter,
            skip_question_validation=args.skip_question_validation,
            skip_text_preprocessing=args.skip_text_preprocessing,
            max_workers=args.max_workers,
            batch_size=args.batch_size,
            use_vllm=args.use_vllm,
            model_config=model_config,
            intermediate_paths=intermediate_paths
        )
        
        logger.info("✨ 管道执行成功完成！")
        
        # 保存执行统计
        stats_path = os.path.join(os.path.dirname(args.syndatas_path), "pipeline_v5_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(timing_stats, f, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"❌ 管道执行失败: {str(e)}")
        raise