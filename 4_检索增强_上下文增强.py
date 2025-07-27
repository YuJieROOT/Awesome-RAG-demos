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


# =========== 上下文扩展的相似文本查询 ===========
def context_enriched_search(query, text_chunks, chunks_embeeddings, top_k=3, context_size = 1):
    query_embedding = utils.get_embedding(query, client)

    similarity_scores = []

    for idx, chunk_embedding in enumerate(chunks_embeeddings):
        similarity_score = utils.cosine_similarity(query_embedding, chunk_embedding)
        similarity_scores.append((idx, similarity_score))

    similarity_scores.sort(key=lambda x: x[1], reverse=True) # 按分数从高到低排序

    top_idx = [idx for idx, _ in similarity_scores[:top_k]] # 获得相似度最高的 top_k 个块的索引

    # 定义上下文范围，即从当前块的前后各取多少块
    # 确保不超出文本块的范围
    start_idx = [max(0, idx - context_size) for idx in top_idx]
    end_idx = [min(len(text_chunks), idx + context_size + 1) for idx in top_idx]

    context_texts = []
    for start, end in zip(start_idx, end_idx):
        context_texts.append(text_chunks[start:end])

    return context_texts


# ====================== 主程序调用 ======================
pdf_path = "data/AI_Information.pdf"

extracted_text = utils.extract_text_from_pdf(pdf_path)  # 提取 PDF 内容

text_chunks = utils.split_text_into_chunks(extracted_text, 2048, 200) # 文本分块

print("Number of chunks:", len(text_chunks))

chunks_embeeddings = utils.get_chunk_embedding(text_chunks, client)  # 获取每个块的向量表示

# ====================== 测试 ======================
# 从JSON文件中加载验证数据集  
with open('data/val.json') as f:  
    data = json.load(f)  

# 从数据集中提取第一个问题作为我们的查询  
query = data[0]['question']  

# 检索：最相关的片段 + 其上下文片段 以提供背景信息  
context_texts = context_enriched_search(query, text_chunks, chunks_embeeddings, top_k=3, context_size = 1) 

# 根据上下文扩展的相似文本片段创建用户提示
user_message = "\n".join([f"Context {idx + 1}:\n{chunk}\n=====================================\n" for idx, chunk in enumerate(context_texts)])
user_message = f"{user_message}\nQuestion: {query}"

# 生成AI回复
system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"

ai_response = utils.generate_response(client, system_prompt, user_message)

print(f"==================== AI's Answer ==================== \n {ai_response.choices[0].message.content}")

# ====================== 评估 ======================

# 定义评估系统的系统提示
evaluate_system_prompt = "You are an intelligent evaluation system tasked with assessing the AI assistant's responses. If the AI assistant's response is very close to the true response, assign a score of 1. If the response is incorrect or unsatisfactory in relation to the true response, assign a score of 0. If the response is partially aligned with the true response, assign a score of 0.5."

# 通过组合用户查询、AI回复、真实回复和评估系统提示来创建评估提示
evaluation_prompt = f"User Query: {query}\nAI Response:\n{ai_response.choices[0].message.content}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"

# 使用评估系统提示和评估提示生成评估回复
evaluation_response = utils.generate_response(client, evaluate_system_prompt, evaluation_prompt)

# 打印评估回复
print(evaluation_response.choices[0].message.content)