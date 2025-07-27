# 通过在添加每个 chunk 之前，为其添加一个标题，来增强检索效果

import os  # 操作系统相关功能
import numpy as np  # NumPy，用于数值计算
import json  # JSON 数据处理
from openai import OpenAI  # OpenAI API 客户端
import utils

from dotenv import load_dotenv

# =========== 初始化 OpenAI API 客户端 ===========
load_dotenv() # # 默认加载当前目录下的 .env 文件

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 替换成自己的 API
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的 base_url
)

# =========== 使用 LLM 为每个分块生成描述性标题 ===========
def generate_chunk_header(chunk, model="qwen-plus"):
    system_prompt = "Generate a concise and informative title for the given text."
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk}
        ]
    )
    return response.choices[0].message.content.strip()

# =========== 生成包含 标题 + 片段 的列表 ===========
def chunk_text_with_header(text, n, overlap):
    chunks = []
    for i in range(0, len(text), n - overlap):
        chunk = text[i:i+n]
        header = generate_chunk_header(chunk)
        chunks.append({"header": header, "content": chunk})
    return chunks


# =========== 生成包含 标题 + 片段 的 Embedding 列表 ===========
def generate_embedding_with_header(text_chunks):
    embeddings = []
    for chunk in text_chunks:
        header_embedding = utils.get_embedding(chunk["header"], client=client)
        content_embedding = utils.get_embedding(chunk["content"], client=client)
        embeddings.append({"header": chunk["header"], "content": chunk["content"], "header_embedding": header_embedding, "content_embedding": content_embedding})

    return embeddings


# =========== 找到与 用户查询 相似度最高的 top_k 个 chunks ===========
def get_top_k_chunks(query, text_chunks, chunks_embeddings, top_k, client):

    query_embedding = utils.get_embedding(query, client) # 用户查询的向量表示

    similarity_scores = []

    for idx, embedding in enumerate(chunks_embeddings):
        sim_header = utils.cosine_similarity(query_embedding, embedding["header_embedding"])
        sim_content = utils.cosine_similarity(query_embedding, embedding["content_embedding"])
        similarity_scores.append((idx, (sim_header + sim_content) / 2) ) 

    similarity_scores.sort(key=lambda x: x[1], reverse=True) # 按分数从高到低排序

    top_idx = [idx for idx, _ in similarity_scores[:top_k]]

    return [text_chunks[idx] for idx in top_idx]

# ====================== 主程序调用 ======================
pdf_path = "data/AI_Information.pdf"

extracted_text = utils.extract_text_from_pdf(pdf_path)  # 提取 PDF 内容

text_chunks = chunk_text_with_header(extracted_text, 2048, 200) # 文本分块

chunks_embeddings = generate_embedding_with_header(text_chunks)


# ====================== 测试 ======================
# 从JSON文件中加载验证数据集  
with open('data/val.json') as f:  
    data = json.load(f)  

# 从数据集中提取第一个问题作为我们的查询  
query = data[0]['question']  

top_k_chunks = get_top_k_chunks(query, text_chunks, chunks_embeddings, top_k=5,client=client) # 获取最相关的 5 个块和标题


# ====================== 打印测试结果 ======================
print("Query:", query)
for idx, chunk in enumerate(top_k_chunks):
    print(f"Top {idx+1} Chunk:")
    print(f"Header: {chunk['header']}")
    print(f"Content: {chunk['content']}")
    print("----------")


# 根据上下文扩展的相似文本片段创建用户提示
user_message = "\n".join([f"Header: {chunk['header']}\nContent:\n{chunk['content']}" for chunk in top_k_chunks])
user_message = f"{user_message}\nQuestion: {query}"

# 生成AI回复
system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"

ai_response = utils.generate_response(client, system_prompt, user_message)

print(f"==================== AI's Answer ==================== \n {ai_response.choices[0].message.content}")

# ====================== 评估 ======================
# 定义评估系统提示
evaluate_system_prompt = """You are an intelligent evaluation system. 
Assess the AI assistant's response based on the provided context. 
- Assign a score of 1 if the response is very close to the true answer. 
- Assign a score of 0.5 if the response is partially correct. 
- Assign a score of 0 if the response is incorrect.
Return only the score (0, 0.5, or 1)."""

# 从验证数据中提取真实答案
true_answer = data[0]['ideal_answer']

# 构造评估提示
evaluation_prompt = f"""
User Query: {query}
AI Response: {ai_response}
True Answer: {true_answer}
{evaluate_system_prompt}
"""

# 生成评估分数
evaluation_response = utils.generate_response(client, evaluate_system_prompt, evaluation_prompt)

# 打印评估分数
print("Evaluation Score:", evaluation_response.choices[0].message.content)