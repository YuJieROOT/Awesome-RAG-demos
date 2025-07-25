# 如何选择合适的块大小
# 评估不同的块大小对结果的影响：
#   1. 从PDF中提取文本。
#   2. 将文本拆分为不同大小的块。
#   3. 为每个块创建嵌入向量。
#   4. 根据用户查询检索相关的块。
#   5. 使用检索到的块生成回复。
#   6. 评估 忠实性 + 相关性。
#   7. 比较不同块大小的结果。

import fitz  # PyMuPDF 库，用于处理 PDF 文件
import os  # 操作系统相关功能
import numpy as np  # NumPy 库，用于数值计算
import json  # JSON 数据处理 
from openai import OpenAI  # OpenAI API客户端
from dotenv import load_dotenv
from tqdm import tqdm  # 进度条库，用于显示处理进度

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
def split_text_into_chunks(text, chunk_size, overlap):

    chunks = []

    # 遍历文本，每次取长度为 chunk_size 的块，重叠部分为 overlap
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])

    return chunks   

# =========== 文本向量化 ===========
def create_embedding(chunks):

    completion = client.embeddings.create(
    model="text-embedding-v2",
    input=chunks,
    dimensions=1024,  
    encoding_format="float"
    )

    # 将结果转换为 numpy 数组的列表并返回
    return np.array(completion.data[0].embedding)

# =========== 为每个 chunk 创建 embedding ===========
def get_chunk_embedding(text_chunks):
    return [create_embedding(chunk) for chunk in text_chunks]


# =========== 通过 余弦相似度 计算 用户查询 与 chunks 的相似度 ===========
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# =========== 找到与 用户查询 相似度最高的 k 个 chunks ===========
def get_top_k_chunks(query, chunks, chunk_embeddings, top_k):

    query_embedding = create_embedding(query) # 用户查询的向量表示

    similarities = [cosine_similarity(query_embedding, chunk_embedding) for chunk_embedding in chunk_embeddings] # 计算查询嵌入与每个文本块嵌入之间的余弦相似度

    # 从数组 similarities 中找出最大的 k 个元素的索引，并按从大到小的顺序返回这些索引。
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [chunks[idx] for idx in top_indices] # 返回与查询最相似的 k 个文本块

# =========== 定制 AI 助手 ===========
system_prompt = """
You are an AI assistant that strictly answers based on the given context. 
If the answer cannot be derived directly from the provided context, 
respond with: 'I do not have enough information to answer that.'
"""

def generate_answer(query, system_prompt, top_k_chunks, model="qwen-plus"):
    # 将检索到的片段组合成单一上下文字符串
    context = "\n".join([f"Context {idx+1}:\n{chunk}" for idx, chunk in enumerate(top_k_chunks)])    
    # 通过结合上下文和查询创建用户提示
    user_prompt = f"{context}\n\nQuestion: {query}"

    completion = client.chat.completions.create(
    model=model,
    messages=[{'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}],
    )

    return completion.choices[0].message.content

# =========== 定制 评估系统 ===========
# 定义 评估系统常量
SCORE_FULL = 1.0     # 完全匹配或完全令人满意
SCORE_PARTIAL = 0.5  # 部分匹配或部分令人满意
SCORE_NONE = 0.0     # 无匹配或不令人满意

# 定义严格的 忠实性评估提示模板
FAITHFULNESS_PROMPT_TEMPLATE = """
Evaluate the faithfulness of the AI response compared to the true answer.
User Query: {question}
AI Response: {response}
True Answer: {true_answer}

Faithfulness measures how well the AI response aligns with facts in the true answer, without hallucinations.

INSTRUCTIONS:
- Score STRICTLY using only these values:
    * {full} = Completely faithful, no contradictions with true answer
    * {partial} = Partially faithful, minor contradictions
    * {none} = Not faithful, major contradictions or hallucinations
- Return ONLY the numerical score ({full}, {partial}, or {none}) with no explanation or additional text.
"""

# 定义严格的 相关性评估提示模板
RELEVANCY_PROMPT_TEMPLATE = """
Evaluate the relevancy of the AI response to the user query.
User Query: {question}
AI Response: {response}

Relevancy measures how well the response addresses the user's question.

INSTRUCTIONS:
- Score STRICTLY using only these values:
    * {full} = Completely relevant, directly addresses the query
    * {partial} = Partially relevant, addresses some aspects
    * {none} = Not relevant, fails to address the query
- Return ONLY the numerical score ({full}, {partial}, or {none}) with no explanation or additional text.
"""

def evaluate_response(question, response, true_answer):
    # 格式化评估提示
    faithfulness_prompt = FAITHFULNESS_PROMPT_TEMPLATE.format(
            question=question, 
            response=response, 
            true_answer=true_answer,
            full=SCORE_FULL,
            partial=SCORE_PARTIAL,
            none=SCORE_NONE
    )
    relevancy_prompt = RELEVANCY_PROMPT_TEMPLATE.format(
            question=question, 
            response=response,
            full=SCORE_FULL,
            partial=SCORE_PARTIAL,
            none=SCORE_NONE
    )
    # 请求模型进行忠实性评估
    faithfulness_response = client.chat.completions.create(
           model="qwen-plus",
            temperature=0,
            messages=[
                    {"role": "system", "content": "You are an objective evaluator. Return ONLY the numerical score."},
                    {"role": "user", "content": faithfulness_prompt}
            ]
    )
    # 请求模型进行相关性评估
    relevancy_response = client.chat.completions.create(
            model="qwen-plus",
            temperature=0,
            messages=[
                    {"role": "system", "content": "You are an objective evaluator. Return ONLY the numerical score."},
                    {"role": "user", "content": relevancy_prompt}
            ]
    )

    # 提取分数并处理潜在的解析错误
    try:
            faithfulness_score = float(faithfulness_response.choices[0].message.content.strip())
    except ValueError:
            print("Warning: Could not parse faithfulness score, defaulting to 0")
            faithfulness_score = 0.0
            
    try:
            relevancy_score = float(relevancy_response.choices[0].message.content.strip())
    except ValueError:
            print("Warning: Could not parse relevancy score, defaulting to 0")
            relevancy_score = 0.0

    return faithfulness_score, relevancy_score


# ====================== 主程序调用 ======================
pdf_path = "data/AI_Information.pdf"

extracted_text = extract_text_from_pdf(pdf_path)  # 提取 PDF 内容

chunk_sizes = [128, 256, 512] # 定义不同的块大小进行评估

text_chunks_dict = {size:split_text_into_chunks(extracted_text, chunk_size=size, overlap=size // 5) for size in chunk_sizes} # 创建一个字典，其中键是块大小，值是分块后的文本列表

for chunk_size, chunks in text_chunks_dict.items():
    print(f"块大小为 {chunk_size} 的分块数量为 {len(chunks)}")

chunk_embeddings_dict = {size:get_chunk_embedding(chunks) for size, chunks in tqdm(text_chunks_dict.items(), desc="Creating embeddings")} # 创建一个字典，其中键是块大小，值是分块后的文本嵌入

print(f"\n====================== Begin Data Verify ======================\n")

# 加载验证数据集
with open('data/val.json') as f:
    data = json.load(f)

query = data[3]['question']

top_k_chunks_dict = {size:get_top_k_chunks(query, text_chunks_dict[size],chunk_embeddings_dict[size], top_k=5) for size in chunk_sizes} # 获取与查询最相似的 top_k 个文本块

ai_answers_dict = {size:generate_answer(query, system_prompt, top_k_chunks_dict[size]) for size in chunk_sizes} # 使用 top_k 个文本块生成答案

print(f"==================== AI's Answer ==================== \n {ai_answers_dict[128]}")

print(f"==================== Evaluation ==================== \n")

# 第一个验证数据的真实答案
true_answer = data[3]['ideal_answer']

# 对于块大小 256 和 128 评估回复
faithfulness, relevancy = evaluate_response(query, ai_answers_dict[256], true_answer)
faithfulness2, relevancy2 = evaluate_response(query, ai_answers_dict[128], true_answer)

# 打印评估分数
print(f"Faithfulness Score (Chunk Size 256): {faithfulness}")
print(f"Relevancy Score (Chunk Size 256): {relevancy}")

print(f"\n")

print(f"Faithfulness Score (Chunk Size 128): {faithfulness2}")
print(f"Relevancy Score (Chunk Size 128): {relevancy2}")