from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModel
from peft import PeftModel, PeftConfig
from pyngrok import ngrok, conf
from pymongo import MongoClient
import torch
import faiss
import json
import os
import uuid
import shutil
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import docx
from tika import parser
import tika

# 初始化 Tika
tika.initVM()

# === Ngrok 設定 ===
conf.get_default().auth_token = "2zdWETPLyoPte8AydDb2uyo2aHi_3FZCs9z9EPoaB8W1v1nGo"
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# === 載入 RAG 模型 ===
print("\U0001F680 載入 LLM 模型中...")
lora_model_path = "./output/result_mistral7b-lora"
base_model_path = PeftConfig.from_pretrained(lora_model_path).base_model_name_or_path

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    offload_folder="./offload",
    torch_dtype="auto"
)
model = PeftModel.from_pretrained(base_model, lora_model_path)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print("✅ LLM 載入完成")

# === Embedding 查詢設定 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = "./sustainability_embedding_model"
faiss_dir = "./faiss_data"
os.makedirs(faiss_dir, exist_ok=True)

embedding_tokenizer = AutoTokenizer.from_pretrained(model_dir)
embedding_base_model = AutoModel.from_pretrained(model_dir)

class EmbeddingModel(torch.nn.Module):
    def __init__(self, base_model, tokenizer, num_labels=3):
        super().__init__()
        self.model = base_model
        self.tokenizer = tokenizer
        self.classifier = torch.nn.Linear(base_model.config.hidden_size, num_labels)

    def get_embeddings(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        extended_attention_mask = attention_mask.unsqueeze(-1)
        sum_embeddings = torch.sum(token_embeddings * extended_attention_mask, 1)
        sum_mask = torch.sum(extended_attention_mask, 1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask

    def get_embedding(self, text, device):
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512).to(device)
        with torch.no_grad():
            return self.get_embeddings(inputs['input_ids'], inputs['attention_mask'])

embedding_model = EmbeddingModel(embedding_base_model, embedding_tokenizer)
embedding_model.load_state_dict(torch.load(os.path.join(model_dir, "embedding_head.pt"), map_location=device), strict=False)
embedding_model = embedding_model.to(device).eval()

# === MongoDB 與 FAISS ===
mongo_client = MongoClient("mongodb+srv://0506ppm:tt920506@cluster0.ozc6lzs.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
collection = mongo_client["vector_db"]["paragraphs"]

faiss_index_path = os.path.join(faiss_dir, "faiss_index.index")
faiss_ids_path = os.path.join(faiss_dir, "faiss_ids.json")

if os.path.exists(faiss_index_path):
    try:
        index = faiss.read_index(faiss_index_path)
    except Exception as e:
        print("❌ 讀取 FAISS 索引失敗：", e)
        index = None
else:
    index = None


if os.path.exists(faiss_ids_path):
    with open(faiss_ids_path, "r") as f:
        paragraph_ids = json.load(f)
else:
    paragraph_ids = []


label_map = {0: "初級做法", 1: "進階做法", 2: "領先做法"}

class QueryRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: QueryRequest):
    prompt = f"""你是一位 ESG 永續報告分析師，請直接針對以下問題清楚簡短地回答，不要產生額外問題：
問題：{req.message}
回答："""
    
    result = pipe(prompt, max_new_tokens=200)[0]['generated_text']

    # 只取第一個「回答」段落
    if "回答：" in result:
        answer = result.split("回答：")[-1].split("問題：")[0].strip()
    else:
        answer = result.strip()

    return {"reply": answer}

@app.post("/rag_chat")
def rag_chat(req: QueryRequest):
    # Step 1：將 query 轉為向量
    query_vec = embedding_model.get_embedding(req.message, device).cpu().numpy()

    # Step 2：使用 FAISS 搜尋前 3 筆段落
    D, I = index.search(query_vec, 3)
    retrieved_texts = []
    context_snippets = []

    for idx in I[0]:
        para_id = paragraph_ids[idx]
        doc = collection.find_one({"_id": para_id})
        if doc:
            label = label_map.get(doc.get("label", -1), "未標記")
            para_text = doc['text']
            context_snippets.append(f"[{label}] {para_text}")
            retrieved_texts.append({
                "id": para_id,
                "label": label,
                "text": para_text
            })

    # Step 3：構建 prompt 並產生回覆
    context = "\n\n".join(context_snippets)
    prompt = f"""你是一位 ESG 永續報告分析師，請根據以下參考資料回答問題，若找不到答案請誠實說明。\n\n參考資料：\n{context}\n\n問題：{req.message}\n回答："""

    result = pipe(prompt, max_new_tokens=200)[0]['generated_text']

    # Step 4：清洗回答
    if "回答：" in result:
        answer = result.split("回答：")[-1].split("問題：")[0].strip()
    else:
        answer = result.strip()

    return {
        "reply": answer,
        "references": retrieved_texts
    }


class DocumentProcessor:
    def __init__(self):
        self.extractors = {
            '.pdf': self._extract_pdf,
            '.docx': self._extract_docx,
            '.doc': self._extract_doc,
            '.txt': self._extract_txt
        }

    def _extract_pdf(self, file_path):
        try:
            doc = fitz.open(file_path)
            text = "\n".join([page.get_text("text") for page in doc])
            doc.close()
            return text
        except Exception as e:
            print(f"PDF處理錯誤 {file_path}: {e}")
            return ""

    def _extract_docx(self, file_path):
        try:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            print(f"DOCX處理錯誤 {file_path}: {e}")
            return ""

    def _extract_doc(self, file_path):
        try:
            parsed = parser.from_file(file_path)
            return parsed.get("content", "") or ""
        except Exception as e:
            print(f"DOC處理錯誤 {file_path}: {e}")
            return ""

    def _extract_txt(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"TXT處理錯誤 {file_path}: {e}")
            return ""

    def extract_text(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        extractor = self.extractors.get(ext)
        if extractor:
            return extractor(file_path)
        else:
            print(f"不支援的檔案類型: {ext}")
            return ""

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    upload_dir = "./upload"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    processor = DocumentProcessor()
    text = processor.extract_text(file_path)
    os.remove(file_path)

    if not text.strip():
        return {"error": "❌ 檔案無法擷取文字內容"}

    def chunk_text(text, max_length=500, overlap=50):
        if len(text) <= max_length:
            return [text]
        chunks, start = [], 0
        while start < len(text):
            end = start + max_length
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    paragraphs = chunk_text(text)
    paragraph_labels = [{"source": file.filename, "category": "uploaded_document", "chunk_id": i} for i in range(len(paragraphs))]
    embeddings, new_ids = [], []

    for para, label in zip(paragraphs, paragraph_labels):
        emb = embedding_model.get_embedding(para, device).cpu().numpy()
        embeddings.append(emb)
        para_id = str(uuid.uuid4())
        new_ids.append(para_id)

        collection.insert_one({
            "_id": para_id,
            "text": para,
            "category": label["category"],
            "source": label["source"],
            "chunk_id": label["chunk_id"],
            "meta": {
                "source_file": label["source"],
                "category": label["category"],
                "length": len(para),
                "chunk_index": label["chunk_id"]
            }
        })

    new_embeddings = np.vstack(embeddings)
    if index is None:
        index = faiss.IndexFlatL2(new_embeddings.shape[1])
    index.add(new_embeddings)
    paragraph_ids.extend(new_ids)

    faiss.write_index(index, faiss_index_path)
    with open(faiss_ids_path, "w") as f:
        json.dump(paragraph_ids, f)
    
        # ✅ 取前 3 段當作 preview
    preview_texts = paragraphs[:3]

    return {
        "message": "✅ 文件已向量化",
        "paragraphs": len(paragraphs),
        "dimension": new_embeddings.shape[1],
        "preview": preview_texts,
        "new_faiss_added": len(new_ids)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
