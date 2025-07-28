# 对每个文本块生成相关的问题，并使用这些问题来增强文本块

import fitz  # PyMuPDF，用于处理 PDF 文件
import os  # 操作系统相关功能，如文件路径管理
import numpy as np  # 科学计算库 NumPy
import json  # JSON 数据处理
from openai import OpenAI  # OpenAI API 客户端
import re  # 正则表达式模块
from tqdm import tqdm  # 进度条显示工具
from dotenv import load_dotenv  # 加载环境变量
import utils  # 自定义工具函数

# =========== 初始化 OpenAI API 客户端 ===========
load_dotenv() # # 默认加载当前目录下的 .env 文件

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 替换成自己的 API
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的 base_url
)


# =========== 为文本片段生成问题 ===========
def generate_questions(text_chunk, num_questions, model="qwen-plus"):
    questions = []
    
    system_prompt = "你是一名擅长从文本中提炼关键信息并生成相关问题的专家。请只用中文，根据给定文本生成可以用该文本直接回答的简明问题，聚焦于核心信息和重要概念。"
    
    user_prompt = f"""
                   请根据以下文本内容，生成 {num_questions} 个不同的问题，这些问题都可以仅通过该文本内容来回答：{text_chunk}
                   请只用中文作答，输出格式为编号的问题列表，不要有其他多余内容。
                   """
    response = client.chat.completions.create(
    model=model,
    temperature=0,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    )
    
    questions_text = response.choices[0].message.content.strip()

    for line in questions_text.split('\n'):
        cleaned_line = re.sub(r'^\d+\.\s*', '', line.strip()) # 去掉行首的编号和点  
        if cleaned_line and cleaned_line.endswith('?'):
            questions.append(cleaned_line)

    return questions

# =========== 创建一个简单的向量存储系统 ===========
class SimpleVectorStore:
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add_item(self, text, embedding, metadata=None):
        self.vectors.append(embedding)
        self.texts.append(text)
        self.metadata.append(metadata or {})

    def get_top_k_similar(self, query_embedding, top_k=5):

        similarites = []

        for idx, vector in enumerate(self.vectors):
            similarity = utils.cosine_similarity(query_embedding, vector)
            similarites.append((idx, similarity))

        similarites.sort(key=lambda x: x[1], reverse=True)

        results = []

        for i in range(min(top_k, len(similarites))):
            idx, similarity = similarites[i]
            results.append({
                "text": self.texts[idx],
                "similarity": similarity,
                "metadata": self.metadata[idx]
            })

        return results



# =========== 语义搜索 ===========
def semantic_search(query, vector_store, top_k):
    query_embedding = utils.get_embedding(query, client)

    results = vector_store.get_top_k_similar(query_embedding, top_k)

    return results


# ====================== 主程序调用 ======================
pdf_path = "data/2024年AI行业报告.pdf"

extracted_text = utils.extract_text_from_pdf(pdf_path)  # 提取 PDF 内容

text_chunks = utils.split_text_into_chunks(extracted_text, 2048, 200)  # 将提取的内容切分成多个文本块

vector_store = SimpleVectorStore()

# ===== 提取和处理文档 =====
print("Processing chunks and generating questions...\n")

for idx, chunk in enumerate(tqdm(text_chunks, desc="Processing chunks")):
    chunk_embedding = utils.get_embedding(chunk, client)

    # 将这个块存储到向量存储中
    vector_store.add_item(
        text = chunk,
        embedding = chunk_embedding,
        metadata = {
            "type":"chunk", # 标记为文本块
            "chunk_index": idx
        }
    )

    # 为该块生成问题
    questions = generate_questions(chunk, num_questions=3)

    # 将问题存储到向量存储中
    for idx, question in enumerate(questions):
        vector_store.add_item(
            text = question,
            embedding = utils.get_embedding(question, client),
            metadata = {
                "type": "question",  # 标记为问题
                "chunk_index": idx,
                "original_chunk": chunk
            }
        )
    

# ===== 结合 相关片段 + 问题 中的信息来准备上下文 =====
def prepare_context(rearch_results):
    chunk_idxs = set()
    context_chunk = []

    for result in search_results:
        if result['metadata']['type'] == 'chunk':
            chunk_idxs.add(result['metadata']['chunk_index'])
            context_chunk.append(f"Chunk {result['metadata']['chunk_index']}: \n {result['text']}")
        elif result['metadata']['type'] == 'question':
            chunk_idx = result['metadata']['chunk_index']
            if chunk_idx not in chunk_idxs:
                chunk_idxs.add(chunk_idx)
                context_chunk.append(f"Chunk {chunk_idx} (referenced by question '{result['text']}'):\n{result['metadata']['original_chunk']}")

    # 合并所有上下文片段
    full_context = "\n\n".join(context_chunk)
    return full_context



# ===== 测试 =====
# 从JSON文件中加载验证数据集  
with open('data/val.json') as f:  
    data = json.load(f)  

# 从数据集中提取第一个问题作为我们的查询  
query = data[4]['question']  

# 执行语义搜索以查找相关的内容
search_results = semantic_search(query, vector_store, top_k=5)

# 按类型组织结果
chunk_results = []
question_results = []

for result in search_results:
    if result['metadata']['type'] == 'chunk':
        chunk_results.append(result)
    elif result['metadata']['type'] == 'question':
        question_results.append(result)

# ===== 打印 块结果 ===== 
print("\n======================= Chunk Results =======================\n")
for idx, result in enumerate(chunk_results):
    print(f"Chunk {idx + 1}:")
    print(result['text'][:300] + "...")
    print("============================================================")

# ===== 打印 问题匹配结果 =====
print("\n======================= Question Results =======================\n")
for idx, result in enumerate(question_results):
    print(f"Question {idx + 1}: Similarity: {result['similarity']:.4f}")
    print(result['text'][:300] + "...")
    chunk_idx = result['metadata']['chunk_index']
    print(f"Related chunk: {chunk_idx + 1}")
    print("============================================================")


# ===== 准备搜索结果的上下文 =====
context = prepare_context(search_results)


# ===== 生成 AI 回复 =====
system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"
    
user_prompt = f"""
                Context:{context}

                Question: {query}

                Please answer the question based only on the context provided above. Be concise and accurate.
                """

ai_response = utils.generate_response(client, system_prompt, user_prompt, model="qwen-plus")

print(f"==================== AI's Answer ==================== \n {ai_response}")

# ===== 评估 =====
reference_answer = data[4]['ideal_answer']  

evaluate_system_prompt = """你是一个智能评测系统，负责评估AI助手的回复质量。
                            请将AI助手的回复与真实/参考答案进行对比，主要从以下三个方面进行评价：
                            1. 事实准确性：回复内容是否准确、无错误。
                            2. 完整性：是否涵盖了参考答案中的所有重要信息点。
                            3. 相关性：回复是否直接回答了用户的问题。

                            请根据以下评分标准打分（0~1分）：
                            - 1.0：内容和含义完全一致
                            - 0.8：非常好，仅有轻微遗漏或差异
                            - 0.6：较好，覆盖主要要点但遗漏部分细节
                            - 0.4：部分回答，存在明显遗漏
                            - 0.2：仅有极少相关信息
                            - 0.0：内容错误或完全无关

                            请给出你的评分，并用中文简要说明理由。
                            """

evaluation_prompt = f"""
                        用户问题: {query}
                        AI助手回复:{ai_response}
                        参考答案:{reference_answer}
                        请根据上述标准，对AI助手的回复进行评价和打分。
                        """


# 评估回复  
evaluation = utils.generate_response(client, evaluate_system_prompt, evaluation_prompt)

# 打印评估分数
print("Evaluation Score:", evaluation)