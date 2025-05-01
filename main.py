import os
import re
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
import PyPDF2
import docx
import tempfile
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant
from datetime import datetime
from typing import List, Dict
from pydantic import BaseModel
import sqlite3
from uuid import uuid4
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# Khởi tạo FastAPI app
app = FastAPI(title="Character AI Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo API Key của OpenRouter
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
# Khởi tạo API Key của HuggingFace
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("HUGGINGFACE_API_KEY", "")

# Thay OpenAIEmbeddings bằng HuggingFaceEmbeddings (miễn phí, chạy cục bộ)
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = HuggingFaceHubEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

# Kết nối đến Qdrant
# qdrant_client = QdrantClient(
#     "localhost", 
#     port=6333
# )
qdrant_client = QdrantClient(
    url="https://46cfbe6c-39e3-4932-9c35-f785698fbdd3.us-west-2-0.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0._vpwH9FFClLO1JH-Cp4RwD7anj769IL4rM58VmPXVj0",
)

collection_name = "chatbot_documents"

# Định nghĩa kích thước batch
BATCH_SIZE = 100

# Tạo cơ sở dữ liệu cho history
def init_db():
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Gọi hàm khởi tạo
init_db()

# Thêm các hàm để quản lý history
def get_history(session_id):
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    cursor.execute('SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY id', (session_id,))
    history = [{"role": role, "content": content} for role, content in cursor.fetchall()]
    conn.close()
    return history

def save_message(session_id, role, content):
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)',
                  (session_id, role, content))
    conn.commit()
    conn.close()

# Định nghĩa mô hình dữ liệu cho request/response
class ChatRequest(BaseModel):
    message: str
    session_id: str = None

class ChatResponse(BaseModel):
    history: List[Dict[str, str]]
    message: str

# Hàm đọc nội dung từ file (giữ nguyên)
def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
    elif ext == ".docx":
        doc = docx.Document(file_path)
        text = " ".join([para.text for para in doc.paragraphs])
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
    else:
        return "Định dạng file không hỗ trợ!"
    return text

@app.get("/test")
async def test():
    return {"message": "API v2 is running!"}

@app.get("/health")
async def health_check():
    """Health check endpoint for Koyeb"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# Hàm xử lý file upload (chuyển sang FastAPI)
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp file để upload!")
    
    # Kiểm tra kích thước file (giới hạn 20MB)
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File quá lớn! Giới hạn là 20MB.")
    
    # Tạo thư mục tạm
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Ghi file vào thư mục tạm
        with open(temp_file_path, "wb") as f:
            f.write(content)
        
        # Trích xuất văn bản
        text = extract_text_from_file(temp_file_path)
        if "Định dạng file không hỗ trợ" in text:
            raise HTTPException(status_code=400, detail=text)
        
        # Chia nhỏ văn bản
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        documents = text_splitter.split_text(text)
        
        # Tạo collection name an toàn
        file_base_name = file.filename.split('.')[0]
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', file_base_name)
        current_collection = f"doc_{safe_name}"
        
        total_chunks = len(documents)
        
        try:
            # Kiểm tra nếu collection đã tồn tại
            try:
                if qdrant_client.collection_exists(current_collection):
                    # Xóa collection cũ
                    qdrant_client.delete_collection(current_collection)
            except Exception as e:
                print(f"Lỗi khi kiểm tra collection: {e}")
                
            # Tạo collection mới
            vector_size = len(embeddings.embed_query("test"))
            qdrant_client.create_collection(
                collection_name=current_collection,
                vectors_config={"size": vector_size, "distance": "Cosine"}
            )
            
            # Tạo vector store sử dụng qdrant_client đã cấu hình
            vector_store = Qdrant(
                client=qdrant_client,
                collection_name=current_collection,
                embeddings=embeddings
            )
            
            # Thêm documents theo batch
            for i in range(0, total_chunks, BATCH_SIZE):
                batch = documents[i:min(i+BATCH_SIZE, total_chunks)]
                vector_store.add_texts(batch)
            
            # Lưu collection_name mới vào file
            with open("current_collection.txt", "w", encoding="utf-8") as f:
                f.write(current_collection)
                
            return {
                "message": f"File đã được xử lý thành công! Collection: {current_collection}",
                "total_chunks": total_chunks
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý file: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# Hàm trò chuyện với chatbot (chuyển sang FastAPI)
@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    # Nếu không có session_id, tạo mới
    session_id = request.session_id or str(uuid4())
    
    try:
        # Đọc collection_name
        with open("current_collection.txt", "r", encoding="utf-8") as f:
            current_collection = f.read().strip()
    except:
        raise HTTPException(status_code=400, detail="Vui lòng tải file lên trước!")
    
    vector_store = Qdrant(
        client=qdrant_client,
        collection_name=current_collection,
        embeddings=embeddings
    )
    
    # Cấu hình ChatOpenAI để dùng OpenRouter
    # llm = ChatOpenAI(
    #     openai_api_key=openrouter_api_key,
    #     openai_api_base="https://openrouter.ai/api/v1",
    #     model_name="nvidia/llama-3.1-nemotron-70b-instruct:free",
    #     temperature=0
    # )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
        temperature=0
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3})
    )
    
    # Lưu câu hỏi vào history
    save_message(session_id, "user", request.message)
    
    # Truy vấn và lưu câu trả lời
    response_text = qa_chain.invoke({"query": request.message})["result"]
    save_message(session_id, "assistant", response_text)
    
    # Lấy toàn bộ history
    history = get_history(session_id)
    
    return ChatResponse(history=history, message=session_id)

# Chạy ứng dụng
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)