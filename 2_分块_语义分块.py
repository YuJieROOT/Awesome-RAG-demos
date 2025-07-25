# 语义分块：文本基于句子之间的内容相似性，被划分为有意义的片段，以提高检索的准确度。
# 固定分块：文本被划分为固定大小的片段，以提高检索的效率。

# 语义分块的常见方法包括：
# 1. 百分位法 (percentile)：计算所有句子之间的差异，然后任何大于 X 百分位数的差异都会被分割。
# 2. 标准差法 (Standard Deviation)：任何超过 X 个标准差的差异都会被分割。
# 3. 四分位距法 (Interquartile Range, IQR)：使用四分位距来分割块。


import fitz  # PyMuPDF 库，用于处理 PDF 文件
import os  # 操作系统相关功能
import numpy as np  # NumPy 库，用于数值计算
import json  # JSON 数据处理 
from openai import OpenAI  # OpenAI API客户端
from dotenv import load_dotenv

# =========== 初始化 OpenAI API 客户端 ===========
load_dotenv() # # 默认加载当前目录下的 .env 文件

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 替换成自己的 API
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的 base_url
)


# =========== 从 PDF 文件中提取文本 ===========
def extract_text_from_pdf(pdf_path):

    mypdf = fitz.open(pdf_path)  # 打开 PDF 文件
    all_text = ""  # 初始化文本内容

    for page_num in range(mypdf.page_count):  # 遍历每一页
        page = mypdf[page_num]  # 获取当前页
        text = page.get_text("text")  # 提取当前页的文本内容
        all_text += text  # 将当前页的文本内容添加到总文本内容中

    return all_text  # 返回提取的文本内容

# =========== 文本向量化 ===========
def get_embedding(texts):
    completion = client.embeddings.create(
    model="text-embedding-v2",
    input=texts,
    dimensions=1024,  
    encoding_format="float"
    )

    return np.array(completion.data[0].embedding)


# =========== 通过 余弦相似度 计算 连续句子之间的相似度 ===========
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# =========== 确定语义分块的断点位置 ===========
def find_breakpoints(similarities, method="percetile", threshold=90):
    if method == "percetile":
        threshold_val = np.percentile(similarities, threshold)
    elif method == "standard_deviation":
        mean = np.mean(similarities)
        std_dev = np.std(similarities)
        threshold_val = mean - (threshold * std_dev)
    elif method == "interquartile":
        q1 = np.percentile(similarities, 25)
        q3 = np.percentile(similarities, 75)
        iqr = q3 - q1
        threshold_val = q1 - (1.5 * iqr)
    else:
        raise ValueError("Invalid method. Choose 'percetile', 'standard_deviation', or 'interquartile'.")

    # 找到相似度低于阈值的索引
    return [idx for idx, sim in enumerate(similarities) if sim < threshold_val]

# =========== 根据断点的位置分割文本为 chunks ===========
def split_text_into_chunks(sentences, breakpoints):
    chunks = []
    start = 0

    for breakpoint in breakpoints:
        chunks.append(sentences[start:breakpoint+1])
        start = breakpoint + 1
    chunks.append(sentences[start:])  # 添加最后一个 chunk
    
    return chunks

# =========== 为每个 chunk 创建 embedding ===========
def get_chunk_embedding(text_chunks):
    return [get_embedding(chunk) for chunk in text_chunks]


# =========== 找到与 用户查询 相似度最高的 k 个 chunks ===========
def get_top_k_chunks(query, text_chunks, chunk_embeddings, top_k):

    query_embedding = get_embedding(query) # 用户查询的向量表示

    similarity_scores = []

    # 遍历计算：用户查询 与 所有的 chunks 之间的相似度
    for idx, chunk_embedding in enumerate(chunk_embeddings):
        similarity_score = cosine_similarity(np.array(query_embedding), np.array(chunk_embedding))
        similarity_scores.append((idx, similarity_score))

    similarity_scores.sort(key=lambda x: x[1], reverse=True) # 按分数从高到低排序

    top_idx = [idx for idx, _ in similarity_scores[:top_k]]

    return [text_chunks[idx] for idx in top_idx]

# ====================== 主程序调用 ======================
pdf_path = "data/AI_Information.pdf"

extracted_text = extract_text_from_pdf(pdf_path)  # 提取 PDF 内容

sentences = extracted_text.split(".") # 将文本按句子分割

embeddings = [get_embedding(sentence) for sentence in sentences] # 为每个句子创建嵌入向量

print(f"\n====================== Generated {len(embeddings)} sentence embeddings. ======================\n")

similarities = [cosine_similarity(embeddings[i], embeddings[i+1]) for i in range(len(embeddings)-1)] # 计算连续句子之间的相似度

breakpoints = find_breakpoints(similarities, method="percetile", threshold=90) # 找到需要分块的断点

text_chunks = split_text_into_chunks(sentences, breakpoints) # 根据断点位置将文本分割为 chunks

print(f"\n====================== Number of text chunks: {len(text_chunks)} ======================\n")

print("\nFirst chunk:\n")
print(text_chunks[0])

chunk_embeddings = get_chunk_embedding(text_chunks) # 为每个 chunk 创建 embedding

print(f"\n====================== Begin Data Verify ======================\n")

# 加载验证数据集
with open('data/val.json') as f:
    data = json.load(f)

query = data[0]['question']

top_k_chunks = get_top_k_chunks(query, text_chunks, chunk_embeddings, top_k=2)

# 定义AI助手的系统提示
system_prompt = "You are an AI assistant that strictly answers based on the given context. \
                If the answer cannot be derived directly from the provided context, \
                respond with: 'I do not have enough information to answer that.'"

def generate_answer(system_prompt, user_message, model="qwen-plus"):
    completion = client.chat.completions.create(
    model=model,
    messages=[{'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message}],
    )

    return completion

# 根据顶级片段创建用户提示
user_prompt = "\n".join([f"Context {idx + 1}:\n{chunk}\n=====================================\n" for idx, chunk in enumerate(top_k_chunks)])
user_prompt = f"{user_prompt}\nQuestion: {query}"

# 生成AI回复
ai_response = generate_answer(system_prompt, user_prompt)
print(f"==================== AI's Answer ==================== \n {ai_response.choices[0].message.content}")

# 对 AI 回答的评估
# 定义评估系统的系统提示
evaluate_system_prompt = '''
You are an intelligent evaluation system tasked with assessing the AI assistant's responses.\
    If the AI assistant's response is very close to the true response, assign a score of 1. \
    If the response is incorrect or unsatisfactory in relation to the true response, assigna score of 0. \
    If the response is partially aligned with the true response, assign a score of 0.5.
'''
# 通过组合用户查询、AI回复、真实回复和评估系统提示创建评估提示
evaluation_prompt = f"User Query: {query}\nAI Response:\n{ai_response.choices[0].message.content}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"

# 使用评估系统提示和评估提示生成评估回复
evaluation_response = generate_answer(evaluate_system_prompt, evaluation_prompt)

# 打印评估回复
print(f"==================== Evaluation ==================== \n {evaluation_response.choices[0].message.content}")