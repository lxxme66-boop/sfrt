from typing import List, Any
import json
from openai import BadRequestError, OpenAI
import os
from volcenginesdkarkruntime import Ark

def get_chunk4(i, chunks):
    chunknum = int(os.getenv("CHUNK_NUM"))
    # 取 i 和 i+4 个chunk
    if (i+chunknum) >= len(chunks):
        chunk4 = chunks[i:]
        if len(chunk4) < int(os.getenv("CHUNK_NUM_MIN")):
            return None
    else:
        chunk4 = chunks[i: i + chunknum] 
    return chunk4

def get_chunkstr(chunks: List[dict]):
    chunkstr = ""
    for idx,chunk in enumerate(chunks):
        chunkstr += f"chunk: {idx}, source: {chunk['source']}\n{chunk['chunk']}"
        chunkstr += "\n"
    return chunkstr

def load_articles(chunks_path):
    with open(chunks_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
        return articles

def build_openai_client_chat(env_prefix : str = "COMPLETION", **kwargs: Any) -> OpenAI:
    """
    Build OpenAI client based on the environment variables.
    """

    client = OpenAI(
        api_key=os.getenv("COMPLETION_OPENAI_API_KEY"),
        base_url=os.getenv("COMPLETION_OPENAI_BASE_URL")
    )

    return client.chat.completions.create

def build_doubao_embedding(env_prefix : str = "COMPLETION", **kwargs: Any) -> Ark:
    """
    Build OpenAI client based on the environment variables.
    """

    emb_client = Ark(
        api_key=os.getenv("COMPLETION_OPENAI_API_KEY"),
    )

    return emb_client.embeddings.create