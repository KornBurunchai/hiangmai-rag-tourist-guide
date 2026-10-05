# 🏔️ แอ่วเชียงใหม่ RAG Tourism Assistant Web Application

เว็บแอปพลิเคชันแชตบอตแนะนำข้อมูลการท่องเที่ยวจังหวัดเชียงใหม่ ด้วยเทคนิค **Retrieval-Augmented Generation (RAG)** อิงข้อมูลจากคลังเอกสารจริง สามารถระบุเวลาเปิด-ปิด ค่าธรรมเนียม แหล่งท่องเที่ยว และอ้างอิงที่มาได้อย่างแม่นยำ

- **URL หน้าเว็บ Streamlit:** `[ใส่ URL ที่ Deploy บน Streamlit Cloud ของนักศึกษา]`
- **GitHub Repository:** `[ใส่ URL GitHub Repository ของนักศึกษา]`

---

## 🌟 คุณสมบัติทางเทคนิค (Technical Features)

1. **Document Loading & Chunking:** โหลดเอกสารความรู้การท่องเที่ยวเชียงใหม่ 10 ไฟล์ (>18,000 ตัวอักษร) ตัดแบ่งข้อความด้วย `RecursiveCharacterTextSplitter` (Chunk Size: 450, Overlap: 80)
2. **Embedding & Vector Search:** แปลงข้อความด้วยโมเดล `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` ค้นหาข้อมูลที่ตรงกับคำถามด้วย **FAISS Index**
3. **Prompt Engineering:** กำหนด System Prompt บังคับให้ LLM ตอบจาก Context ที่ค้นเจอเท่านั้น หากไม่พบข้อมูลจะตอบว่า *"ไม่พบข้อมูลเกี่ยวกับเรื่องนี้ในคลังเอกสารความรู้..."*
4. **Large Language Model:** ประมวลผลสร้างคำตอบด้วย **Groq API** (โมเดล `llama-3.3-70b-versatile`)
5. **Chatbot Interface & Sources:** แสดงผลด้วย Streamlit Chat UI แสดงไฟล์อ้างอิงพร้อมค่า Similarity Score ทุกครั้งที่ตอบ

---

## 📁 โครงสร้างโปรเจกต์

- `app.py`: ไฟล์หลักของแอปพลิเคชัน Streamlit
- `requirements.txt`: รายชื่อไลบรารีสำหรับการ Deploy
- `test_questions.csv`: คำถามทดสอบ 10 ข้อ (พร้อมคำถามนอกบริบท 2 ข้อ)
- `data/`: โฟลเดอร์เก็บเอกสารความรู้ 10 ไฟล์

---

## 🚀 วิธีการติดตั้งและรันในเครื่อง Local

1. **Clone Repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/chiangmai-rag-tourist-guide.git](https://github.com/YOUR_USERNAME/chiangmai-rag-tourist-guide.git)
   cd chiangmai-rag-tourist-guide