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



# ====================== 主程序调用 ======================
pdf_path = "data/AI_Information.pdf"

extracted_text = utils.extract_text_from_pdf(pdf_path)  # 提取 PDF 内容

text_chunks = utils.split_text_into_chunks(extracted_text, 2048, 200) # 文本分块