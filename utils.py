import fitz  # PyMuPDF，用于处理 PDF 文件
import os  # 操作系统相关功能
import numpy as np  # NumPy，用于数值计算
import json  # JSON 数据处理
from openai import OpenAI  # OpenAI API 客户端
import utils
from dotenv import load_dotenv

# =========== 从 PDF 文件中提取文本 ===========
def extract_text_from_pdf(pdf_path):

    mypdf = fitz.open(pdf_path)  # 打开 PDF 文件
    all_text = ""  # 初始化文本内容

    for page_num in range(mypdf.page_count):  # 遍历每一页
        page = mypdf[page_num]  # 获取当前页
        text = page.get_text("text")  # 提取当前页的文本内容
        all_text += text  # 将当前页的文本内容添加到总文本内容中

    return all_text  # 返回提取的文本内容


# =========== 将得到的文本分块 ===========
def split_text_into_chunks(text, n, overlap):

    chunks = []

    # 遍历文本，每次取长度为 n 的块，重叠部分为 overlap
    for i in range(0, len(text), n - overlap):
        chunks.append(text[i:i+n])

    return chunks   


# =========== 文本向量化 ===========
def get_embedding(texts, client):
    completion = client.embeddings.create(
    model="text-embedding-v2",
    input=texts,
    dimensions=1024,  
    encoding_format="float"
    )

    return np.array(completion.data[0].embedding)


# =========== 为每个 chunk 创建 embedding ===========
def get_chunk_embedding(text_chunks, client):
    return [get_embedding(chunk, client) for chunk in text_chunks]


# =========== 通过 余弦相似度 计算两个 embedding 之间的相似度 ===========
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# =========== 找到与 用户查询 相似度最高的 top_k 个 chunks ===========
def get_top_k_chunks(query, chunks, embeddings, top_k, client):

    query_embedding = get_embedding(query, client) # 用户查询的向量表示

    similarity_scores = []

    # 遍历计算：用户查询 与 所有的 chunks 之间的相似度
    for idx, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(np.array(query_embedding), np.array(chunk_embedding))
        similarity_scores.append((idx, similarity_score))

    similarity_scores.sort(key=lambda x: x[1], reverse=True) # 按分数从高到低排序

    top_idx = [idx for idx, _ in similarity_scores[:top_k]]

    return [chunks[idx] for idx in top_idx]


# =========== 定义 AI 助手的系统提示 ===========
def generate_response(client, system_prompt, user_message, model="qwen-plus"):
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return response


