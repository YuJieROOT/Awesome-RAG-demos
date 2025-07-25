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


# =========== 将得到的文本分块 ===========
def split_text_into_chunks(text, n, overlap):

    chunks = []

    # 遍历文本，每次取长度为 n 的块，重叠部分为 overlap
    for i in range(0, len(text), n - overlap):
        chunks.append(text[i:i+n])

    return chunks   

# =========== 文本向量化 ===========
def create_embedding(chunks):

    completion = client.embeddings.create(
    model="text-embedding-v2",
    input=chunks,
    dimensions=1024,  
    encoding_format="float"
    )

    return completion

# =========== 通过 余弦相似度 计算 用户查询 与 chunks 的相似度 ===========
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# =========== 找到与 用户查询 相似度最高的 k 个 chunks ===========
def get_top_k_chunks(query, chunks, embeddings, top_k):

    query_embedding = create_embedding(query).data[0].embedding # 用户查询的向量表示

    similarity_scores = []

    # 遍历计算：用户查询 与 所有的 chunks 之间的相似度
    for idx, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(np.array(query_embedding), np.array(chunk_embedding.embedding))
        similarity_scores.append((idx, similarity_score))

    similarity_scores.sort(key=lambda x: x[1], reverse=True) # 按分数从高到低排序

    top_idx = [idx for idx, _ in similarity_scores[:top_k]]

    return [chunks[idx] for idx in top_idx]


# ====================== 主程序调用 ======================
pdf_path = "data/AI_Information.pdf"

extracted_text = extract_text_from_pdf(pdf_path)  # 提取 PDF 内容

chunks = split_text_into_chunks(extracted_text, 2048, 200) # 文本分块

print("Number of chunks:", len(chunks))

embeddings = create_embedding(chunks) # 为文本块创建 Embedding

# 加载验证数据集
with open('data/val.json') as f:
    data = json.load(f)

query = data[0]['question']

print("Query:", query)

# 查询与验证数据集中的问题最相关的块
top_chunks = get_top_k_chunks(query, chunks, embeddings.data, top_k=2)

for idx, chunk in enumerate(top_chunks):
    print(f"====================== Top {idx+1} Chunk ======================")
    print(chunk)
    print()

# 基于检索到的块，生成回答
system_prompt = "You are an AI assistant that strictly answers based on the given context.If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"

def generate_answer(system_prompt, user_message, model="qwen-plus"):
    completion = client.chat.completions.create(
    model=model,
    messages=[{'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message}],
    )

    return completion
    

# 上下文组装：合并 用户query + 相关文档
user_prompt = "\n".join([f"Context {idx + 1}:\n{chunk}\n ====================\n" for idx, chunk in enumerate(top_chunks)])
user_prompt = f"{user_prompt}\nQuestion: {query}"

# 生成回答
answer = generate_answer(system_prompt, user_prompt)
print(f"==================== AI's Answer ==================== \n {answer.choices[0].message.content}")

# 对 AI 回答的评估
# 定义评估系统的系统提示
evaluate_system_prompt = '''
You are an intelligent evaluation system tasked with assessing the AI assistant's responses.\
    If the AI assistant's response is very close to the true response, assign a score of 1. \
    If the response is incorrect or unsatisfactory in relation to the true response, assigna score of 0. \
    If the response is partially aligned with the true response, assign a score of 0.5.
'''
# 通过组合用户查询、AI回复、真实回复和评估系统提示创建评估提示
evaluation_prompt = f"User Query: {query}\nAI Response:\n{answer.choices[0].message.content}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"

# 使用评估系统提示和评估提示生成评估回复
evaluation_response = generate_answer(evaluate_system_prompt, evaluation_prompt)

# 打印评估回复
print(f"==================== Evaluation ==================== \n {evaluation_response.choices[0].message.content}")