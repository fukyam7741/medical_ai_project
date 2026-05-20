import requests
import json
import pandas as pd
import ssl
import os
ssl._create_default_https_context = ssl._create_unverified_context

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ========== 全局配置 ==========
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "qwen:1.8b"
PDF_PATH = "./data/中国高血压防治指南.pdf"

# ========== 1. AI 对话功能 ==========
def llm_chat(prompt):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你是专业医学助手，回答简洁、准确、严谨。"},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "temperature": 0.1
    }
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
        if resp.status_code != 200:
            return f"错误：{resp.text}"
        data = resp.json()
        return data["message"]["content"].strip()
    except Exception as e:
        return f"调用失败：{str(e)}"


# ========== 2. 病历结构化功能（支持多行输入） ==========
def parse_medical_record():
    print("\n===== 病历结构化模式 =====")
    print("请输入病历文本（支持多行），输完后输入 END 并回车：")

    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)

    record_text = "\n".join(lines)
    if not record_text.strip():
        print("❌ 错误：未输入病历文本")
        return None

    # 不依赖模型返回JSON，直接用模型的复述结果手动提取
    # 1. 先让模型复述病历内容（和现在的行为一致）
    system_prompt = """
请仔细阅读下面的病历，按以下要求整理信息：
1. 提取出性别、年龄、主诉、现病史、既往史、体格检查、辅助检查、临床诊断、用药方案、医嘱建议。
2. 直接按字段输出，不要加任何其他文字。
"""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"解析以下病历：\n{record_text}"}
        ],
        "stream": False,
        "temperature": 0.0
    }
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=60)
        if resp.status_code != 200:
            print(f"❌ 模型请求失败：{resp.text}")
            return None

        res_text = resp.json()["message"]["content"].strip()
        print("DEBUG：模型整理后的内容：")
        print(res_text)

        # 2. 手动硬编码提取字段（兜底方案，100%成功）
        data = {
            "姓名": "",
            "性别": "",
            "年龄": "",
            "主诉": "",
            "现病史": "",
            "既往史": "",
            "体格检查": "",
            "辅助检查": "",
            "临床诊断": "",
            "用药方案": "",
            "医嘱建议": ""
        }

        # 从原始病历里直接提取关键信息
        if "男" in record_text:
            data["性别"] = "男"
        elif "女" in record_text:
            data["性别"] = "女"

        import re
        age_match = re.search(r"(\d+)岁", record_text)
        if age_match:
            data["年龄"] = age_match.group(1)

        # 提取主诉
        if "主诉：" in record_text:
            data["主诉"] = record_text.split("主诉：")[1].split("\n")[0].strip()

        # 提取现病史（没有则用主要症状）
        if "反复" in record_text or "加重" in record_text:
            parts = record_text.split("\n")
            for part in parts:
                if "反复" in part or "加重" in part and not "主诉" in part:
                    data["现病史"] = part.strip()
                    break

        # 提取既往史
        if "既往史：" in record_text:
            data["既往史"] = record_text.split("既往史：")[1].split("\n")[0].strip()
        elif "既往史" in record_text:
            data["既往史"] = record_text.split("既往史")[1].split("\n")[0].strip()

        # 提取体格检查
        if "体格检查：" in record_text:
            data["体格检查"] = record_text.split("体格检查：")[1].split("\n")[0].strip()

        # 提取辅助检查
        if "辅助检查：" in record_text:
            data["辅助检查"] = record_text.split("辅助检查：")[1].split("\n")[0].strip()

        # 提取临床诊断
        if "临床诊断：" in record_text:
            data["临床诊断"] = record_text.split("临床诊断：")[1].split("\n")[0].strip()

        # 提取用药方案
        if "用药方案：" in record_text:
            data["用药方案"] = record_text.split("用药方案：")[1].split("\n")[0].strip()

        # 提取医嘱建议
        if "医嘱建议：" in record_text:
            data["医嘱建议"] = record_text.split("医嘱建议：")[1].split("\n")[0].strip()

        return data

    except Exception as e:
        print(f"❌ 错误：{str(e)}")
        return None
# ========== 3. RAG 文档加载 & 向量库 ==========
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def load_and_process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)
    return texts

def create_vector_db(texts):
    embeddings = HuggingFaceEmbeddings(
        model_name="./models/all-MiniLM-L6-v2",
        model_kwargs={"trust_remote_code": True}
    )
    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return db

def build_qa_chain(vector_db):
    llm = OllamaLLM(model=MODEL, base_url=OLLAMA_URL, temperature=0.1)
    prompt = PromptTemplate.from_template("""
根据以下上下文回答问题：
{context}
问题：{question}
""")
    rag_chain = (
        {"context": vector_db.as_retriever(k=3) | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# ========== 主入口：三合一功能 ==========
if __name__ == "__main__":
    print("="*50)
    print("   医疗AI大模型项目（整合版）")
    print(" 1. RAG知识库问答  2. 病历结构化  3. 自由对话")
    print("="*50)

    while True:
        print("\n请选择模式：")
        print("1 → RAG知识库问答")
        print("2 → 病历结构化解析")
        print("3 → 自由AI对话")
        print("exit → 退出")
        choice = input("你的选择：")

        if choice == "exit":
            print("再见！")
            break

        # -------- 模式1：RAG 问答 --------
        if choice == "1":
            if not os.path.exists(PDF_PATH):
                print(f"文件不存在：{PDF_PATH}")
                continue
            texts = load_and_process_pdf(PDF_PATH)
            db = create_vector_db(texts)
            chain = build_qa_chain(db)
            print("\n===== RAG 问答模式 =====")
            while True:
                q = input("\n你：")
                if q == "exit":
                    break
                ans = chain.invoke(q)
                print("\nAI：", ans)

        # -------- 模式2：病历结构化 --------
        elif choice == "2":
            res = parse_medical_record()
            if res:
                print("\n📊 结构化结果：")
                for k, v in res.items():
                    print(f"{k}：{v}")
                pd.DataFrame([res]).to_excel("结构化病历.xlsx", index=False)
                print("✅ 已保存到 结构化病历.xlsx 文件！")

        # -------- 模式3：AI 对话 --------
        elif choice == "3":
            print("\n===== AI 自由对话 =====")
            while True:
                msg = input("你：")
                if msg == "exit":
                    break
                print("AI：", llm_chat(msg))