from utils.common_utils import build_openai_client_chat
from utils.article_chunks import gen_chunks
from utils.topic_concepts import trans_chunk4, gen_topics
from utils.document_quality_filter import filter_high_quality_documents
from utils.enhanced_question_generation import gen_enhanced_questions_with_validation
from utils.answer_generation import gen_answer_v3
from utils.data_synthesis import syn_data_v2
import time
import argparse
import sys
import logging
import os

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print(sys.executable)

def get_args() -> argparse.Namespace:
    """
    解析和返回用户命令行参数
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data_dir", type=str, required=True, help="原始文档目录路径")
    parser.add_argument("--filtered_data_dir", type=str, default="data_filtered", help="筛选后文档保存目录")
    parser.add_argument("--chunks_path", type=str, required=True, help="文档分块保存路径")
    parser.add_argument("--chunk4_path", type=str, required=True, help="Chunk4数据保存路径")
    parser.add_argument("--topics_path", type=str, required=True, help="主题数据保存路径")
    parser.add_argument("--questions_path", type=str, required=True, help="原始问题保存路径")
    parser.add_argument("--validated_questions_path", type=str, required=True, help="验证后问题保存路径")
    parser.add_argument("--answers_path", type=str, required=True, help="答案数据保存路径")
    parser.add_argument("--syndatas_path", type=str, required=True, help="最终合成数据保存路径")
    parser.add_argument("--start_idx", type=int, default=0, help="处理文档的起始索引")
    parser.add_argument("--end_idx", type=int, default=None, help="处理文档的结束索引")
    parser.add_argument("--skip_document_filter", action="store_true", help="跳过文档质量筛选步骤")
    parser.add_argument("--skip_question_validation", action="store_true", help="跳过问题质量验证步骤")
    parser.add_argument("--max_workers", type=int, default=4, help="并发处理数量")

    args = parser.parse_args()
    return args

def enhanced_data_synthesis_pipeline(
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
    max_workers=4
):
    """
    增强版数据合成管道
    
    流程：
    1. 文档质量筛选（可选）
    2. 文档分块
    3. Chunk4转换
    4. 主题生成
    5. 增强版问题生成
    6. 问题质量验证（可选）
    7. 答案生成
    8. 数据合成
    """
    pipeline_start = time.time()
    
    # 构建LLM客户端
    chat_model = build_openai_client_chat()
    logger.info("LLM客户端初始化完成")
    
    # 第0步：文档质量筛选（可选）
    if not skip_document_filter:
        logger.info("🔍 步骤1: 开始文档质量筛选...")
        filter_start = time.time()
        
        actual_data_dir = filter_high_quality_documents(
            data_dir, 
            filtered_data_dir, 
            chat_model, 
            max_workers=max_workers
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
    
    gen_topics(chunk4_path, topics_path, chat_model)
    
    topic_time = time.time() - topic_start
    logger.info(f"✅ 主题生成完成，耗时: {topic_time/60:.2f}分钟")
    
    # 第4步：增强版问题生成和验证
    logger.info("❓ 步骤5: 开始增强版问题生成和验证...")
    question_start = time.time()
    
    if not skip_question_validation:
        gen_enhanced_questions_with_validation(
            topics_path, 
            questions_path,
            validated_questions_path,
            chat_model, 
            chunk4_path,
            max_workers=max_workers
        )
        actual_questions_path = validated_questions_path
    else:
        logger.info("⏭️ 跳过问题质量验证，使用原始问题生成")
        # 这里可以调用原始的问题生成函数
        from utils.topic_concepts import gen_questions_with_topic_v3
        gen_questions_with_topic_v3(topics_path, questions_path, chat_model, chunk4_path)
        actual_questions_path = questions_path
    
    question_time = time.time() - question_start
    logger.info(f"✅ 问题生成和验证完成，耗时: {question_time/60:.2f}分钟")
    
    # 第5步：答案生成
    logger.info("💬 步骤6: 开始答案生成...")
    answer_start = time.time()
    
    gen_answer_v3(actual_questions_path, chat_model, answers_path)
    
    answer_time = time.time() - answer_start
    logger.info(f"✅ 答案生成完成，耗时: {answer_time/60:.2f}分钟")
    
    # 第6步：数据合成
    logger.info("🔗 步骤7: 开始数据合成...")
    syndata_start = time.time()
    
    syn_data_v2(answers_path, syndatas_path)
    
    syndata_time = time.time() - syndata_start
    logger.info(f"✅ 数据合成完成，耗时: {syndata_time/60:.2f}分钟")
    
    # 总结
    total_time = time.time() - pipeline_start
    logger.info("🎉 增强版数据合成管道执行完成！")
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
    
    return {
        "filter_time": filter_time if not skip_document_filter else 0,
        "chunk_time": chunk_time,
        "chunk4_time": chunk4_time,
        "topic_time": topic_time,
        "question_time": question_time,
        "answer_time": answer_time,
        "syndata_time": syndata_time,
        "total_time": total_time
    }

if __name__ == "__main__":
    args = get_args()
    
    logger.info("🚀 启动增强版数据合成管道 v4")
    logger.info(f"📁 原始数据目录: {args.data_dir}")
    logger.info(f"📁 筛选数据目录: {args.filtered_data_dir}")
    logger.info(f"📄 最终输出: {args.syndatas_path}")
    
    # 创建必要的输出目录
    for path in [args.chunks_path, args.chunk4_path, args.topics_path, 
                 args.questions_path, args.validated_questions_path, 
                 args.answers_path, args.syndatas_path]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    try:
        timing_stats = enhanced_data_synthesis_pipeline(
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
            max_workers=args.max_workers
        )
        
        logger.info("✨ 管道执行成功完成！")
        
    except Exception as e:
        logger.error(f"❌ 管道执行失败: {str(e)}")
        raise