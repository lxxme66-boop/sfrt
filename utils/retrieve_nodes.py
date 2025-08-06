# 在线流程 
# 混合检索

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
)
from llama_index.core.schema import NodeWithScore
from llama_index.core import QueryBundle
from typing import List, Any
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from utils.common_utils import load_articles, build_doubao_embedding
from llama_index.core.base.embeddings.base import BaseEmbedding
import os
from llama_index.core.query_engine import RetrieverQueryEngine
from pydantic import SkipValidation

# 4. 创建自定义的检索器
class CustomRetriever(BaseRetriever):
    """custom retriever that performs both vector and keyword table retrieval"""
    def __init__(self,
                 vector_retriever: VectorIndexRetriever,
                 bm25_retriever: BM25Retriever,
                 mode: str = "OR",
    ) -> None:
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        if mode not in ["AND", "OR"]:
            raise ValueError("mode must be either AND or OR")
        self._mode = mode
        super().__init__()
    def _retrieve(self, query_bundle: QueryBundle|str) -> List[NodeWithScore]:
        """retrieve nodes given query"""
        print(f"Retrieving nodes for query: {query_bundle.query_str}")
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        bm25_nodes = self._bm25_retriever.retrieve(query_bundle)
        
        vector_ids = {node.node.node_id for node in vector_nodes}
        bm25_ids = {node.node.node_id for node in bm25_nodes}
        
        combined_dict = {node.node.node_id: node for node in vector_nodes}
        combined_dict.update({node.node.node_id: node for node in bm25_nodes})
        
        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(bm25_ids)
        if self._mode == "OR":
            retrieve_ids = vector_ids.union(bm25_ids)
        
        retrieve_nodes = [combined_dict[node_id] for node_id in retrieve_ids]
        print(f"{len(retrieve_nodes)} nodes retrieved")
        return retrieve_nodes

class DouBaoEmbedding(BaseEmbedding):
    emb_model: SkipValidation[Any] = None  # 声明字段，避免 pydantic 等框架报错
    def __init__(self, model_name: str = "doubao-embedding-text-240715", emb_model: Any = None, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.emb_model = emb_model
    def _get_embedding(self, texts: list[str] | str) -> List[float] | List[List[float]]:
        # 这里替换为实际调用豆包平台的 API 获取 embedding 的逻辑
        # 例如通过 requests 请求、认证等
        single_text = isinstance(texts, str)
        if single_text:
            texts = [texts]
        response = self.emb_model(
            model=self.model_name,
            input=texts
        )
        embeddings = [
            embedding_data.embedding for embedding_data in response.data
        ]
        if single_text:
            return embeddings[0]
        return embeddings  # 返回浮点数列表

    async def _aget_embedding(self, text: str) -> List[float]:
        return self._get_embedding(text)

    def _get_text_embedding(self, text: list[str]) -> List[List[float]]:
        return self._get_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_embedding(query)
    async def _aget_text_embedding(self, text: list[str]) -> List[List[float]]:
        return self._get_text_embedding(text)
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

def get_doubao_embedding(model="doubao-embedding-text-240715"):
    emb_model = build_doubao_embedding()
    
    embedding_model = DouBaoEmbedding(
        model=model,
        emb_model=emb_model,
        api_key=os.environ.get("COMPLETION_OPENAI_API_KEY"),
        api_base=os.environ.get("COMPLETION_OPENAI_BASE_URL"),
    )
    return embedding_model

def get_retriever(
    docstore_path,
    chroma_db,
    storage_dir,
    similarity_top_k=10
):
    docstore = SimpleDocumentStore.from_persist_path(docstore_path)
    db = chromadb.PersistentClient(path=chroma_db)
    chroma_collection = db.get_or_create_collection(name="sc_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(
        persist_dir=storage_dir,
        vector_store=vector_store, 
        docstore=docstore
    )

    doubao_embedding = get_doubao_embedding()
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=doubao_embedding,
        show_progress=True,
    )
    vector_retriever = vector_index.as_retriever(
        similarity_top_k=similarity_top_k, 
        verbose=True
    )
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=docstore,
        similarity_top_k=similarity_top_k,
    )
    
    custom_retriever = CustomRetriever(
        vector_retriever, 
        bm25_retriever, 
    )
    return custom_retriever

# 重排
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
model_name = r"C:\Users\Administrator\.cache\modelscope\hub\models\BAAI\bge-reranker-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
# 检查是否有可用的GPU，如果有则将模型移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
model = model.to(device)
def rerank_chunks(query, chunks):
    """
    对 chunks 进行重新排序
    
    参数:
        query: 查询文本
        chunks: 待排序的 chunks 列表
        model: reranker 模型
        tokenizer: 对应的 tokenizer
        top_k: 返回前 k 个结果，None 表示返回全部
        
    返回:
        排序后的 chunks 列表
    """
    # 准备模型输入
    features = tokenizer(
        [query]*len(chunks),
        [chunk["chunk"] for chunk in chunks],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    # 将特征张量移动到与模型相同的设备
    features = {k: v.to(device) for k, v in features.items()}
    # 计算分数
    with torch.no_grad():
        scores = model(**features).logits.squeeze()
    # 将分数添加到每个 chunk 中
    for i, chunk in enumerate(chunks):
        chunk["score"] = float(scores[i].cpu())  # 确保将分数移回CPU
    # 降序排序
    sorted_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)
    return sorted_chunks

# 检索文档
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
reranker_model = r"C:\Users\Administrator\.cache\modelscope\hub\models\BAAI\bge-reranker-large"
reranker = FlagEmbeddingReranker(
    model=reranker_model, top_n=int(os.getenv("top_n"))
)

def get_query_engine():
    retriever = get_retriever(
        docstore_path=os.getenv("docstore_path"), 
        chroma_db=os.getenv("chroma_db"), 
        storage_dir=os.getenv("storage_dir"), 
        similarity_top_k=int(os.getenv("similarity_top_k"))
    )
    Settings.llm = None
    query_engine = RetrieverQueryEngine.from_args(
        llm=None,
        response_mode="no_text",
        retriever=retriever, 
        node_postprocessors=[reranker]
    )
    return query_engine
def get_reranked_nodes(query, query_engine):
    response = query_engine.query(query)
    return response.source_nodes




















