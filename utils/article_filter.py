from utils.article_chunks import read_file_names
from utils.vector_store import build_doubao_embedding

def get_embedding(texts: list[str], model) -> list[list[float]]:
    response = model(
        model="doubao-embedding-text-240715",
        input=texts,
        encoding_format="float"
    ) 
    embeddings = [item.embedding for item in response.data]
    return embeddings
def get_embedding_batch(filenames_dir):
    embedding_model = build_doubao_embedding()
    filenames_all = read_file_names(filenames_dir)
    filenames_batch = [filenames_all[i:i+20] for i in range(0, len(filenames_all), 20)]
    embeddings_all = []
    for i, filenames in enumerate(filenames_batch):
        print(f"batch {i}, total {len(filenames_batch)}, now processing: {len(filenames)}, {filenames}")
        embeddings = get_embedding(filenames, embedding_model)
        embeddings_all.extend(embeddings)
    return embeddings_all, filenames_all



import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def save_titles_to_txt(selected_papers, output_file="selected_papers.txt"):
    """
    将选中的论文标题写入TXT文件，每行一个标题
    
    参数:
        selected_papers: 选中的论文列表(包含文件路径)
        output_file: 输出的TXT文件名(默认为selected_papers.txt)
    """
    # 从文件路径中提取纯标题(去掉路径和扩展名)
    paper_titles = []
    for paper_path in selected_papers:
        # 获取文件名(去掉路径)
        filename = paper_path.split('/')[-1]
        paper_titles.append(filename)
        # 去掉.md扩展名
        # title = filename[:-3] if filename.endswith('.md') else filename
        # paper_titles.append(title)
    
    # 写入TXT文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for title in paper_titles:
            f.write(title + '\n')
    
    print(f"成功将 {len(paper_titles)} 篇论文标题写入 {output_file}")


def balanced_sample_papers(filenames, embeddings, n_clusters=6, total_samples=400, min_samples=20):
    """
    改进的平衡抽样方法，确保每个聚类至少有min_samples篇论文
    
    参数:
        filenames: 论文文件名列表
        embeddings: 对应的嵌入向量
        n_clusters: 聚类数量(默认为6)
        total_samples: 需要抽取的总论文数(默认为400)
        min_samples: 每个聚类至少抽取的论文数(默认为20)
        
    返回:
        选中的论文文件名列表
        聚类标签数组
        抽样分布信息
    """
    # 执行K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # 统计每个簇的论文数量
    cluster_counts = np.bincount(clusters)
    total_papers = len(filenames)
    
    # 计算初始按比例分配的样本数
    samples_per_cluster = (cluster_counts / total_papers * total_samples).astype(int)
    
    # 应用最小样本数限制
    samples_per_cluster = np.maximum(samples_per_cluster, min_samples)
    
    # 调整分配，确保总数不超过400
    while samples_per_cluster.sum() > total_samples:
        # 找到最大的簇(不是按原始大小，而是按当前分配)
        largest_cluster = np.argmax(samples_per_cluster)
        samples_per_cluster[largest_cluster] -= 1
    
    # 从每个簇中随机抽取指定数量的论文
    selected_papers = []
    selected_clusters = []
    cluster_info = defaultdict(dict)
    
    for cluster_id in range(n_clusters):
        cluster_mask = (clusters == cluster_id)
        cluster_filenames = np.array(filenames)[cluster_mask]
        
        # 记录聚类信息
        cluster_info[cluster_id]['total_papers'] = cluster_counts[cluster_id]
        cluster_info[cluster_id]['sampled_papers'] = samples_per_cluster[cluster_id]
        
        # 随机抽样
        n_samples = samples_per_cluster[cluster_id]
        if n_samples >= len(cluster_filenames):
            selected = cluster_filenames  # 如果样本数超过簇大小，取全部
        else:
            selected_indices = np.random.choice(
                len(cluster_filenames), 
                size=n_samples, 
                replace=False
            )
            selected = cluster_filenames[selected_indices]
        
        selected_papers.extend(selected.tolist())
        selected_clusters.extend([cluster_id] * len(selected))
    save_titles_to_txt(selected_papers, output_file="selected_papers.txt")
    return selected_papers, np.array(selected_clusters), cluster_info

import pickle

# 保存 embeddings 到文件
def save_embeddings_pickle(embeddings, file_path):
    """使用 pickle 保存 embeddings"""
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings 已保存到 {file_path}")

# 从文件加载 embeddings
def load_embeddings_pickle(file_path):
    """使用 pickle 加载 embeddings"""
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)
    print(f"从 {file_path} 加载的 embeddings 形状: len={len(embeddings)}, len[0]={len(embeddings[0])}")
    return embeddings

import os

def balanced_sample_papers_v2(filenames, embeddings, n_clusters=6, total_samples=400, min_samples=20, output_file=None):
    """
    改进的平衡抽样方法，确保每个聚类至少有min_samples篇论文，并排除已选论文
    
    参数:
        filenames: 论文文件名列表
        embeddings: 对应的嵌入向量
        n_clusters: 聚类数量(默认为6)
        total_samples: 需要抽取的总论文数(默认为400)
        min_samples: 每个聚类至少抽取的论文数(默认为20)
        output_file: 输出文件名，用于读取已选论文
        
    返回:
        选中的论文文件名列表
        聚类标签数组
        抽样分布信息
    """
    # 确保embeddings是NumPy数组
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)
    # 读取已选论文（如果文件存在）
    existing_papers = set()
    if output_file and os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_papers = set(line.strip() for line in f.readlines())
    
    # 过滤掉已选论文
    available_indices = [i for i, filename in enumerate(filenames) if filename not in existing_papers]
    available_filenames = [filenames[i] for i in available_indices]
    available_embeddings = embeddings[available_indices]
    
    # 如果没有足够的可用论文，则返回空
    if len(available_filenames) == 0:
        print("警告：没有可用的新论文可选择")
        return [], np.array([]), {}
    
    # 执行K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(available_embeddings)
    
    # 统计每个簇的论文数量
    cluster_counts = np.bincount(clusters)
    total_available_papers = len(available_filenames)
    
    # 计算初始按比例分配的样本数
    samples_per_cluster = (cluster_counts / total_available_papers * total_samples).astype(int)
    
    # 应用最小样本数限制
    samples_per_cluster = np.maximum(samples_per_cluster, min_samples)
    
    # 调整分配，确保总数不超过total_samples
    while samples_per_cluster.sum() > total_samples:
        # 找到最大的簇(不是按原始大小，而是按当前分配)
        largest_cluster = np.argmax(samples_per_cluster)
        samples_per_cluster[largest_cluster] -= 1
    
    # 从每个簇中随机抽取指定数量的论文
    selected_papers = []
    selected_clusters = []
    cluster_info = defaultdict(dict)
    
    for cluster_id in range(n_clusters):
        cluster_mask = (clusters == cluster_id)
        cluster_filenames = np.array(available_filenames)[cluster_mask]
        
        # 记录聚类信息
        cluster_info[cluster_id]['total_papers'] = cluster_counts[cluster_id]
        cluster_info[cluster_id]['sampled_papers'] = samples_per_cluster[cluster_id]
        
        # 随机抽样
        n_samples = samples_per_cluster[cluster_id]
        if n_samples >= len(cluster_filenames):
            selected = cluster_filenames  # 如果样本数超过簇大小，取全部
        else:
            selected_indices = np.random.choice(
                len(cluster_filenames), 
                size=n_samples, 
                replace=False
            )
            selected = cluster_filenames[selected_indices]
        
        selected_papers.extend(selected.tolist())
        selected_clusters.extend([cluster_id] * len(selected))
    
    # 保存新选中的论文（追加到现有文件或创建新文件）
    if output_file:
        with open(output_file, 'a', encoding='utf-8') as f:  # 使用追加模式
            for title in selected_papers:
                f.write(title + '\n')
    
    print(f"成功从 {total_available_papers} 篇可用论文中选取 {len(selected_papers)} 篇新论文")
    return selected_papers, np.array(selected_clusters), cluster_info


