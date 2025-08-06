from utils.article_chunks import gen_chunks
from utils.common_utils import build_openai_client_chat
from utils.query_generation import gen_query
from utils.answer_generation import gen_answer
from utils.data_synthesis import syn_data, syn_data_v2
from utils.vector_store import get_embedding_then_storage
import time
import argparse
import sys
print(sys.executable)
def get_args() -> argparse.Namespace:
    """
    Parses and returns the arguments specified by the user's command
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data_dir", type=str, default="", help="If a file, the path at which the document is located. If a folder, the path at which to load all documents")
    parser.add_argument("--chunks_path", type=str, default="./", help="The path at which to save the dataset")
    parser.add_argument("--embeddings_path", type=str, default="./", help="The path at which to embedding the dataset")
    parser.add_argument("--questions_path", type=str, default="hf", help="The format of the output dataset.")
    parser.add_argument("--answers_path", type=str, default="jsonl", help="Type to export the dataset to. Defaults to jsonl.")
    parser.add_argument("--syndatas_path", type=str, help="The system prompt to use when the output format is chat")
    parser.add_argument("--num_distract", type=str, default="prompt", help="The prompt column name to use for the completion format")
    parser.add_argument("--start_idx", type=str, default="completion", help="The completion column name to use for the completion format")
    parser.add_argument("--end_idx", type=int, default=3, help="The number of distractor documents to include per data point / triplet")

    args = parser.parse_args()
    return args



def data_synthesis_pipeline(data_dir, chunks_path, embeddings_path, questions_path, answers_path, syndatas_path, num_distract=4, start_idx=7, end_idx=9):
    gen_chunks(data_dir, chunks_path, start_idx, end_idx)
    get_embedding_then_storage(chunks_path, embeddings_path)
    embedding_time = time.time() - main_start
    chat_model = build_openai_client_chat()
    gen_query(chunks_path, chat_model, questions_path)
    question_time = time.time() - embedding_time
    gen_answer(questions_path, chat_model, answers_path)
    answer_time = time.time() - question_time
    # syn_data(chunks_path, answers_path, syndatas_path, num_distract)
    syn_data_v2(answers_path, syndatas_path)
    syndata_time = time.time() - answer_time
    return embedding_time, question_time, answer_time, syndata_time
    
    
if __name__ == "__main__":
    
    main_start = time.time()
    
    data_dir = "data"
    chunks_path = "outputs_chunks/article_chunks04.json"
    embeddings_path = "outputs_chunks/chunk_embedding04.json"
    questions_path = "outputs_questions/article_questions04.json"
    answers_path = "outputs_answers/article_answers04.json"
    syndatas_path = "outputs_syndatas/syndatas05.json"
    
    args = get_args()
    time1, time2, time3, time4 = data_synthesis_pipeline(
        args.data_dir, 
        args.chunks_path, 
        args.embeddings_path, 
        args.questions_path, 
        args.answers_path, 
        args.syndatas_path, 
        num_distract=args.num_distract, 
        start_idx=args.start_idx, 
        end_idx=args.end_idx
    )
    print(f"embedding time: {time1 / 60}min")
    print(f"question time: {time2 / 60}min")
    print(f"answer time: {time3 / 60}min")
    print(f"syndata time: {time4 / 60}min")
    print(f"total time: {(time.time() - main_start) / 60}min")
    
    
    