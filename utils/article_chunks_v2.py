import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
from typing import Literal
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import uuid
from utils.common_utils import load_articles

DocType = Literal["txt", "md"]
# 2.1 要读取文件的文件名
def read_file_names(data_dir, ext=".md"):
    """\n"""
    file_names = []
    for filename in os.listdir(data_dir):
        file_names.append(filename)
    # 过滤掉 ~$ 开头的文件
    file_names = [file_name for file_name in file_names if not file_name.startswith("~$")]
    file_names = [file_name for file_name in file_names if file_name.endswith(ext)]
    print(f"len(file_names): {len(file_names)}")
    return file_names

def get_chunks(
    data_path: Path, 
    doctype: DocType = "txt", 
    chunk_size: int = 512, 
) -> list[str]:
    """
    Takes in a `data_path` and `doctype`, retrieves the document, breaks it down into chunks of size
    `chunk_size`, and returns the chunks.
    """

    chunks = []
    file_paths = [data_path]
    if data_path.is_dir():
        file_paths = list(data_path.rglob('**/*.' + doctype))

    futures = []
    with tqdm(total=len(file_paths), desc="Chunking", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=2) as executor:
            for file_path in file_paths:
                futures.append(executor.submit(get_doc_chunks, file_path, doctype, chunk_size))
            for future in as_completed(futures):
                doc_chunks = future.result()
                chunks.extend(doc_chunks)
                pbar.set_postfix({'chunks': len(chunks)})
                pbar.update(1)

    filename = os.path.basename(data_path)
    return chunks, filename

def get_doc_chunks(
    file_path: Path, 
    doctype: DocType = "txt", 
    chunk_size: int = 512,
 ) -> list[str]:
    if doctype == "json":
        with open(file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        text = data["text"]
    elif doctype == "txt":
        with open(file_path, 'r', encoding="utf-8") as file:
            data = file.read()
        text = str(data)
    elif doctype == "md":
        with open(file_path, 'r', encoding="utf-8") as file:
            data = file.read()
        text = str(data)
    else:
        raise TypeError("Document is not one of the accepted types: api, pdf, json, txt")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.create_documents([text])
    chunks = [chunk.page_content for chunk in chunks]
    return chunks

def build_or_load_chunks(
        datapath: Path, 
        doctype: str,
        CHUNK_SIZE: int
    ):
    """
    Builds chunks and checkpoints them if asked
    """
    chunks = None

    if not chunks:
        chunks, filename = get_chunks(datapath, doctype, CHUNK_SIZE)
        chunks_dict = [
            {"chunk": chunk, "source": filename, "chunk_id": str(uuid.uuid4())}
            for chunk in chunks
        ]

    return chunks_dict, filename

def save_chunks(chunks, article_name, filename):
    # 判断 filename 是否存在，如果存在则追加写入，否则创建新文件
    if os.path.exists(filename):
        with open(filename, 'r', encoding="utf-8") as f:
            existing = json.load(f)
        # 检查 article_name 是否已经存在于 questions 中
        if article_name in existing:
            existing[article_name].extend(chunks)
        else:
            existing[article_name] = chunks
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=4)
    else:
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump({article_name: chunks}, f, ensure_ascii=False, indent=4)
    print(f"Chunks saved to {filename}")
    
def gen_chunks(data_dir, chunks_path, start_idx=None, end_idx=None):
    if os.path.exists(chunks_path):
        print(f"{chunks_path} exists. Skipping...")
        return 
    filenames = read_file_names(data_dir, ext=".md")
    filenames = filenames[int(start_idx):int(end_idx)]
    articles = {}
    count = 0
    for filename in filenames:
        data_path = os.path.join(data_dir, filename)
        data_path = Path(data_path)
        chunks_dict, a_name = build_or_load_chunks(data_path, "md", 512)
        save_chunks(chunks_dict, a_name, chunks_path)
        count += len(chunks_dict)
    print(f"all chunks: {count}")
    return articles

def read_titles_from_txt(input_file) -> list[str]:
    """
    从TXT文件中读取论文标题，每行一个标题，返回标题列表
    
    参数:
        input_file: 输入的TXT文件名(默认为selected_papers.txt)
    
    返回:
        list[str]: 论文标题列表
    """
    paper_titles = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                # 去除行末的换行符和空白字符
                title = line.strip()
                if title:  # 忽略空行
                    paper_titles.append(title)
                    
        print(f"成功从 {input_file} 读取 {len(paper_titles)} 篇论文标题")
        return paper_titles
    
    except FileNotFoundError:
        print(f"错误：文件 {input_file} 不存在")
        return []
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return []


def gen_chunks_v2(data_dir, chunks_path, input_file, start_idx=None, end_idx=None):
    if os.path.exists(chunks_path):
        print(f"{chunks_path} exists. Skipping...")
        return 
    filenames = read_titles_from_txt(input_file)
    filenames = filenames[start_idx:end_idx]
    print(f"filenames: {len(filenames)}")
    articles = {}
    count = 0
    for filename in filenames:
        data_path = os.path.join(data_dir, filename)
        data_path = Path(data_path)
        chunks_dict, a_name = build_or_load_chunks(data_path, "md", 512)
        save_chunks(chunks_dict, a_name, chunks_path)
        count += len(chunks_dict)
    print(f"all chunks: {count}")
    return articles


