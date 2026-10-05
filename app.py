import os
import glob
import pandas as pd
import numpy as np
import streamlit as st
import faiss

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="แอ่วเชียงใหม่ - ระบบแนะนำการท่องเที่ยวด้วย RAG",
    page_icon="🏔️",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #F3F4F6;
        border-left: 4px solid #3B82F6;
        padding: 10px;
        margin-top: 10px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏔️ แอ่วเชียงใหม่ RAG Tourist Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">ระบบตอบคำถามการท่องเที่ยวจังหวัดเชียงใหม่จากคลังเอกสารความรู้แบบแม่นยำ อ้างอิงแหล่งที่มาได้</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Secrets & API Key Setup
# ---------------------------------------------------------
# 🔑 ใส่ Groq API Key ของคุณตรงนี้ได้เลยครับ
groq_api_key = "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"

# Priority 1: Check Streamlit Secrets (ถ้ามีตั้งไว้ใน secrets จะใช้ในนี้แทน)
if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]

# Priority 2: Sidebar Input for Local Testing
with st.sidebar:
    st.header("⚙️ การตั้งค่าระบบ")
    st.markdown("---")
    if not groq_api_key or groq_api_key == "ใส่_GROQ_API_KEY_ของคุณที่นี่":
        groq_api_key = st.text_input("กรอก Groq API Key:", type="password")
        st.caption("🔒 *ระบุ API Key เพื่อเริ่มใช้งาน*")
    else:
        st.success("✅ เชื่อมต่อ Groq API Key เรียบร้อยแล้ว")
    
    st.markdown("---")
    st.subheader("📚 เกี่ยวกับคลังข้อมูล")
    st.info("ระบบดึงข้อมูลจากเอกสารท่องเที่ยวเชียงใหม่ 10 หมวดหมู่ ได้แก่ ดอยอินทนนท์, วัดโบราณ, ดอยสุเทพ, ย่านนิมมาน, ตลาดนัด, อาหารเหนือ, ม่อนแจ่ม, เชียงดาว, เทศกาล และการเดินทาง")

# ---------------------------------------------------------
# Load Embedding Model & RAG Indexing
# ---------------------------------------------------------
@st.cache_resource(show_spinner="🔄 กำลังโหลดโมเดล Sentence Embedding และสร้าง Vector Database (FAISS)...")
def initialize_rag_system(data_folder="data"):
    # 1. Load Text Embedder (Multilingual model for Thai & English)
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # 2. Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=80,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    chunk_metadata = []
    
    # Read files from data directory
    file_paths = glob.glob(os.path.join(data_folder, "*.txt")) + glob.glob(os.path.join(data_folder, "*.md"))
    
    if not file_paths:
        st.error(f"❌ ไม่พบไฟล์เอกสารในโฟลเดอร์ {data_folder}/ กรุณาตรวจสอบโครงสร้างไฟล์")
        return None, None, None, None
        
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        file_chunks = text_splitter.split_text(content)
        for idx, chunk in enumerate(file_chunks):
            chunks.append(chunk)
            chunk_metadata.append({
                "source_file": file_name,
                "chunk_id": idx,
                "text": chunk
            })
            
    # 3. Compute Embeddings
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    embeddings = np.array(embeddings).astype('float32')
    
    # Normalize for cosine similarity via IndexFlatIP
    faiss.normalize_L2(embeddings)
    
    # 4. Build FAISS Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    return embedding_model, index, chunks, chunk_metadata

# Initialize RAG Pipeline
embed_model, faiss_index, all_chunks, metadata_list = initialize_rag_system()

# ---------------------------------------------------------
# Vector Search Function
# ---------------------------------------------------------
def search_context(query, top_k=5, score_threshold=0.15):
    if faiss_index is None or embed_model is None:
        return []
        
    query_vector = embed_model.encode([query], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(query_vector)
    
    scores, indices = faiss_index.search(query_vector, top_k)
    
    retrieved_docs = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(metadata_list) and score >= score_threshold:
            retrieved_docs.append({
                "score": float(score),
                "source_file": metadata_list[idx]["source_file"],
                "text": metadata_list[idx]["text"]
            })
            
    return retrieved_docs

# ---------------------------------------------------------
# Chat Session State
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "สวัสดีครับ! ผมคือไกด์ท่องเที่ยวเชียงใหม่ AI ยินดีให้ข้อมูลสถานที่เที่ยว เวลาเปิด-ปิด ร้านอาหาร และการเดินทางครับ มีเรื่องไหนให้ผมช่วยแนะนำไหมครับ?"}
    ]

# Display Previous Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📌 เอกสารอ้างอิงที่ใช้ในการตอบ (Retrieved Context)"):
                for s in msg["sources"]:
                    st.markdown(f"📄 **ไฟล์:** `{s['source_file']}` (ค่าความเกี่ยวข้อง: {s['score']:.2f})")
                    st.caption(s['text'])

# ---------------------------------------------------------
# Handle User Input
# ---------------------------------------------------------
user_query = st.chat_input("พิมพ์คำถามเกี่ยวกับการท่องเที่ยวเชียงใหม่ที่นี่...")

if user_query:
    # 1. Display User Question
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Check API Key
    if not groq_api_key or groq_api_key == "ใส่_GROQ_API_KEY_ของคุณที่นี่":
        st.warning("⚠️ กรุณากรอก Groq API Key ที่แถบด้านข้าง (Sidebar) หรือระบุในโค้ดก่อนใช้งาน")
        st.stop()

    client = Groq(api_key=groq_api_key)

    # 3. Retrieve Context via FAISS
    retrieved_results = search_context(user_query, top_k=3, score_threshold=0.30)
    
    # Prepare Context String for Prompt
    if retrieved_results:
        context_str = "\n\n---\n\n".join([
            f"[อ้างอิงไฟล์: {item['source_file']}]\n{item['text']}" 
            for item in retrieved_results
        ])
    else:
        context_str = "ไม่พบข้อมูลที่เกี่ยวข้องในคลังเอกสารความรู้"

    # 4. Strict RAG Prompt Engineering
    system_prompt = f"""คุณคือไกด์ท่องเที่ยวเชียงใหม่ผู้สุภาพ มีหน้าที่ตอบคำถามและแนะนำการท่องเที่ยวโดยอิงตาม "บริบทคลังข้อมูล (Context)" ที่กำหนดให้เท่านั้น

กฎที่คุณต้องปฏิบัติตามอย่างเคร่งครัด:
1. ตอบคำถามหรือสรุปข้อมูลโดยใช้ข้อมูลจาก [บริบทคลังข้อมูล (Context)] ด้านล่างนี้เท่านั้น
2. หากผู้ใช้พิมพ์คำถามสั้นๆ หรือคีย์เวิร์ด (เช่น "ตลาดนัด", "ดอยอินทนนท์", "อาหารเหนือ") ให้สรุปรายละเอียดของสถานที่/หมวดหมู่นั้นๆ ที่พบในบริบทมาแนะนำผู้ใช้ทันที
3. หากใน [บริบทคลังข้อมูล (Context)] ไม่พบข้อมูลที่เกี่ยวข้องกับเรื่องที่ผู้ใช้ถามเลยแม้แต่น้อย ให้ตอบปฏิเสธอย่างสุภาพว่า: "ขออภัยครับ/ค่ะ ไม่พบข้อมูลเกี่ยวกับเรื่องนี้ในคลังเอกสารความรู้การท่องเที่ยวเชียงใหม่ที่มีอยู่"
4. ห้ามคิดหรือคาดเดาข้อมูลเองนอกเหนือจากคลังข้อมูลที่มีอยู่โดยเด็ดขาด
5. คำตอบต้องมีความกระชับ อ่านง่าย สุภาพ และจัดหมวดหมู่ให้อ่านสบายตา
6. ระบุชื่อไฟล์อ้างอิงที่ใช้ในการตอบสั้นๆ ไว้ท้ายคำตอบ

[บริบทคลังข้อมูล (Context)]:
{context_str}
"""

    # 5. Call LLM (Qwen 3.8 27B)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            response = client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.1,
                max_tokens=800
            )
            bot_reply = response.choices[0].message.content
            message_placeholder.markdown(bot_reply)

            # Display Source Accordion
            if retrieved_results:
                with st.expander("📌 เอกสารอ้างอิงที่ใช้ในการตอบ (Retrieved Context)"):
                    for s in retrieved_results:
                        st.markdown(f"📄 **ไฟล์:** `{s['source_file']}` (ค่าความเกี่ยวข้อง: {s['score']:.2f})")
                        st.caption(s['text'])

            # Save Message History
            st.session_state.messages.append({
                "role": "assistant",
                "content": bot_reply,
                "sources": retrieved_results
            })

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ Groq API: {str(e)}")