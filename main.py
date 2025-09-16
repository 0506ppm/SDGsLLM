# ✅ main.py（TWCC）- 支援 URL 下載 + 優化文字解析
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModel
from peft import PeftModel, PeftConfig
from pyngrok import ngrok, conf
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
import requests
from dotenv import load_dotenv
import re  # 新增：用於文字清理
from groq import Groq  # 新增：Groq API

load_dotenv()

# 初始化 Tika
tika.initVM()

local_api = os.getenv("LOCAL_API")
groq_api_key = os.getenv("GROQ_API_KEY")

# 初始化 Groq 客戶端
groq_client = Groq(api_key=groq_api_key)
# === Ngrok 設定 ===
conf.get_default().auth_token = os.getenv("NGROK_TOKEN")
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
embedding_model.load_state_dict(torch.load(os.path.join(model_dir, "model_weights.pt"), map_location=device), strict=False)
embedding_model = embedding_model.to(device).eval()

# === FAISS ===
faiss_index_path = os.path.join(faiss_dir, "faiss_index.index")
faiss_ids_path = os.path.join(faiss_dir, "faiss_ids.json")

if os.path.exists(faiss_index_path):
    try:
        index = faiss.read_index(faiss_index_path)
    except Exception as e:
        print("⚠ 讀取 FAISS 索引失敗：", e)
        index = None
else:
    index = None

if os.path.exists(faiss_ids_path):
    with open(faiss_ids_path, "r") as f:
        paragraph_ids = json.load(f)
else:
    paragraph_ids = []

# === Pydantic 模型定義 ===
class QueryRequest(BaseModel):
    message: str

class UploadDocumentRequest(BaseModel):
    file_url: str
    file_name: str

class BatchUploadRequest(BaseModel):
    files: list[dict]  # [{"file_url": "...", "file_name": "..."}]

# === 文字清理和驗證函數（參考你提供的程式碼）===
def clean_text(text):
    """清理文字：去除多餘空白和無效字符"""
    if not text:
        return ""
    
    # 合併多個空白字符為單一空格
    text = re.sub(r'\s+', ' ', text)
    
    # 只保留：中文字、英文字母、數字、常見標點符號
    text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？：；（）、""''「」『』《》〈〉.,:;!?"\'()[\]{}\-—%$€£¥₩₹]', '', text)
    
    return text.strip()

def is_valid_sentence(text):
    """檢查句子是否為有效句子（排除奇怪符號和無意義內容）"""
    if not text or len(text.strip()) < 10:  # 最少10個字符
        return False
    
    # 允許：中文字、中文標點、英文字母、阿拉伯數字、常見符號
    valid_pattern = re.compile(r'^[\u4e00-\u9fffA-Za-z0-9，。！？：；（）、「」『』《》〈〉.,:;!?"\'()\[\]{}\-—\s%$€£¥₩₹]{10,}$')
    
    if not valid_pattern.match(text):
        return False
    
    # 排除只有數字和符號的內容
    chinese_char_count = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_char_count = len(re.findall(r'[A-Za-z]', text))
    
    # 確保至少有一定比例的中文或英文字符
    if (chinese_char_count + english_char_count) / len(text) < 0.3:
        return False
    
    # 排除頁碼、目錄等無意義內容
    meaningless_patterns = [
        r'^\d+\s*$',  # 純數字
        r'^第\s*\d+\s*頁\s*$',  # 頁碼
        r'^頁\s*\d+\s*$',  # 頁碼
        r'^目\s*錄\s*$',  # 目錄
        r'^\.\.\.\s*$',  # 省略號
        r'^[-—]+\s*$',  # 分隔線
        r'^\s*[IVX]+\s*\.\s*$',  # 羅馬數字章節
    ]
    
    for pattern in meaningless_patterns:
        if re.match(pattern, text.strip()):
            return False
    
    return True

def extract_meaningful_sentences(text):
    """從文字中提取有意義的句子"""
    if not text:
        return []
    
    meaningful_sentences = []
    
    # 先按段落分割
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
        
        # 按句號分割句子
        sentences = re.split(r'(?<=[。！？])\s*', paragraph)
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # 清理文字
            clean_sentence = clean_text(sentence)
            
            # 驗證是否為有效句子
            if is_valid_sentence(clean_sentence):
                meaningful_sentences.append(clean_sentence)
    
    return meaningful_sentences

# === 文件處理類別（優化版）===
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
            text = ""
            for page_num, page in enumerate(doc):
                page_text = page.get_text("text")
                if page_text.strip():
                    text += f"\n{page_text}\n"
            doc.close()
            
            # 如果 PyMuPDF 沒有提取到內容，嘗試使用 Tika
            if not text.strip():
                print("📄 PyMuPDF 未提取到內容，嘗試 Tika...")
                parsed = parser.from_file(file_path)
                text = parsed.get("content", "") or ""
            
            return text.strip()
        except Exception as e:
            print(f"PDF處理錯誤 {file_path}: {e}")
            # 備用方案：使用 Tika
            try:
                parsed = parser.from_file(file_path)
                return (parsed.get("content", "") or "").strip()
            except Exception as e2:
                print(f"Tika 備用方案也失敗: {e2}")
                return ""
    
    def _extract_docx(self, file_path):
        try:
            doc = docx.Document(file_path)
            text = []
            
            # 提取段落
            for para in doc.paragraphs:
                if para.text.strip():
                    text.append(para.text)
            
            # 提取表格內容
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        text.append(" | ".join(row_text))
            
            result = "\n".join(text)
            
            # 如果 python-docx 沒有提取到內容，嘗試使用 Tika
            if not result.strip():
                print("📄 python-docx 未提取到內容，嘗試 Tika...")
                parsed = parser.from_file(file_path)
                result = parsed.get("content", "") or ""
            
            return result.strip()
        except Exception as e:
            print(f"DOCX處理錯誤 {file_path}: {e}")
            # 備用方案：使用 Tika
            try:
                parsed = parser.from_file(file_path)
                return (parsed.get("content", "") or "").strip()
            except Exception as e2:
                print(f"Tika 備用方案也失敗: {e2}")
                return ""
    
    def _extract_doc(self, file_path):
        try:
            parsed = parser.from_file(file_path)
            content = parsed.get("content", "") or ""
            return content.strip()
        except Exception as e:
            print(f"DOC處理錯誤 {file_path}: {e}")
            return ""
    
    def _extract_txt(self, file_path):
        # 嘗試多種編碼
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'big5', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read().strip()
                    if content:
                        print(f"✅ 成功使用 {encoding} 編碼讀取 TXT 檔案")
                        return content
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"TXT處理錯誤 (encoding: {encoding}): {e}")
                continue
        
        print(f"⚠ 所有編碼都無法讀取 TXT 檔案: {file_path}")
        return ""
    
    def extract_text(self, file_path):
        if not os.path.exists(file_path):
            print(f"⚠ 檔案不存在: {file_path}")
            return ""
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            print(f"⚠ 檔案為空: {file_path}")
            return ""
        
        print(f"🔍 開始處理檔案: {file_path} (大小: {file_size} bytes)")
        
        ext = os.path.splitext(file_path)[1].lower()
        print(f"🔍 檔案副檔名: {ext}")
        
        if ext not in self.extractors:
            print(f"⚠ 不支援的檔案格式: {ext}")
            print(f"✅ 支援的格式: {list(self.extractors.keys())}")
            return ""
        
        extractor = self.extractors[ext]
        raw_text = extractor(file_path)
        
        if not raw_text or not raw_text.strip():
            print(f"⚠ 未提取到任何文字內容")
            return ""
        
        print(f"🔄 開始清理和驗證文字內容...")
        
        # 提取有意義的句子
        meaningful_sentences = extract_meaningful_sentences(raw_text)
        
        if not meaningful_sentences:
            print(f"⚠ 清理後沒有有效的句子")
            return ""
        
        # 重新組合有效句子
        clean_text_result = '\n'.join(meaningful_sentences)
        
        print(f"✅ 成功提取文字，原始長度: {len(raw_text)} 字元")
        print(f"✅ 清理後長度: {len(clean_text_result)} 字元")
        print(f"✅ 有效句子數量: {len(meaningful_sentences)}")
        print(f"🔍 預覽前3句: {meaningful_sentences[:3]}")
        
        return clean_text_result

# === 下載函數 ===
def download_file(url: str, local_path: str, timeout: int = 30) -> bool:
    """
    從 URL 下載檔案到本機路徑
    """
    try:
        print(f"🌐 開始下載檔案: {url}")
        
        # 設定請求標頭
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # 檢查內容長度
        content_length = response.headers.get('Content-Length')
        if content_length:
            file_size = int(content_length)
            print(f"📊 檔案大小: {file_size} bytes")
            
            # 檢查檔案大小限制（例如 100MB）
            max_size = 100 * 1024 * 1024  # 100MB
            if file_size > max_size:
                print(f"⚠ 檔案過大: {file_size} bytes > {max_size} bytes")
                return False
        
        # 下載檔案
        with open(local_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        print(f"✅ 檔案下載完成: {local_path} ({downloaded} bytes)")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"⚠ 網路請求錯誤: {e}")
        return False
    except Exception as e:
        print(f"⚠ 下載檔案時發生錯誤: {e}")
        return False

# === 文字分塊函數（優化版）===
def chunk_text(text, max_length=400, overlap=50):
    """
    智能文字分塊，優先按句號分割，避免句子被截斷
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    sentences = re.split(r'(?<=[。！？])\s*', text)
    
    current_chunk = ""
    
    for sentence in sentences:
        # 如果當前句子本身就超過最大長度，強制分割
        if len(sentence) > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # 對長句子進行強制分割
            start = 0
            while start < len(sentence):
                end = start + max_length
                chunks.append(sentence[start:end])
                start = end - overlap
        else:
            # 檢查加入這個句子是否會超過長度限制
            if len(current_chunk) + len(sentence) > max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += sentence
    
    # 添加最後一個塊
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # 過濾掉太短的塊
    chunks = [chunk for chunk in chunks if len(chunk.strip()) >= 20]
    
    return chunks

# === 向量化處理函數 ===
def process_document_to_vectors(text, source_name, source_url=None):
    """
    將文字轉換為向量並存入 FAISS 和 MongoDB
    """
    global index, paragraph_ids
    
    # 使用優化的分塊函數
    paragraphs = chunk_text(text)
    print(f"📝 文字分塊完成，共 {len(paragraphs)} 個段落")
    
    paragraph_labels = [
        {
            "source": source_name,
            "category": "remote_document" if source_url else "uploaded_document",
            "chunk_id": i,
            "source_url": source_url
        } 
        for i in range(len(paragraphs))
    ]
    
    embeddings, new_ids = [], []

    # 生成向量並準備文檔
    for para, label in zip(paragraphs, paragraph_labels):
        if len(para.strip()) < 10:  # 跳過太短的段落
            continue
            
        emb = embedding_model.get_embedding(para, device).cpu().numpy()
        embeddings.append(emb)
        para_id = str(uuid.uuid4())
        new_ids.append(para_id)
        
        doc = {
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
        }
        
        # 如果有 URL，添加到 meta 中
        if source_url:
            doc["meta"]["source_url"] = source_url
        
        # 插入 MongoDB
        try:
            # 修正：移除多餘的 https://
            api_url = local_api if local_api.startswith('http') else f"https://{local_api}"
            res = requests.post(f"{api_url}/insert_doc", json=doc, timeout=10)
            res.raise_for_status()
            print(f"✅ 成功插入文檔片段到 MongoDB: {para_id}")
        except Exception as e:
            print(f"⚠ 插入 MongoDB 失敗: {e}")

    # 更新 FAISS 索引
    if embeddings:
        new_embeddings = np.vstack(embeddings)
        
        if index is None:
            index = faiss.IndexFlatL2(new_embeddings.shape[1])
            print("🔍 創建新的 FAISS 索引")
        
        index.add(new_embeddings)
        paragraph_ids.extend(new_ids)

        # 保存 FAISS 索引
        faiss.write_index(index, faiss_index_path)
        with open(faiss_ids_path, "w") as f:
            json.dump(paragraph_ids, f)
        
        print(f"✅ FAISS 索引已更新，新增 {len(new_ids)} 個向量")
    
    return len(paragraphs), new_embeddings.shape[1] if embeddings else 0, paragraphs[:3], len(new_ids)

# === 聊天端點 ===
@app.post("/chat")
def chat(req: QueryRequest):
    prompt = f"""你是一位 ESG 永續報告分析師，請直接針對以下問題清楚簡短地回答，不要產生額外問題：\n問題：{req.message}\n回答："""
    result = pipe(prompt, max_new_tokens=200)[0]['generated_text']
    answer = result.split("回答：")[-1].split("問題：")[0].strip() if "回答：" in result else result.strip()
    return {"reply": answer}

@app.post("/rag_chat")
def rag_chat(req: QueryRequest):
    if index is None:
        return {"reply": "⚠ FAISS 索引尚未初始化，請先上傳文件。", "references": []}
    
    query_vec = embedding_model.get_embedding(req.message, device).cpu().numpy()
    D, I = index.search(query_vec, 3)

    # 列出取得的索引位置與對應的 Mongo _id
    print("🔍 FAISS 找到的向量 ID 索引：", I)
    top_ids = [paragraph_ids[idx] for idx in I[0] if idx < len(paragraph_ids)]
    print("🧾 FAISS 找到的 MongoDB _id：", top_ids)

    if not top_ids:
        return {"reply": "⚠ 沒有找到相關的文件段落。", "references": []}

    # 向本機 API 請求原文
    try:
        response = requests.post(f"{local_api}/get_docs", json={"ids": top_ids}, timeout=10)
        documents = response.json()
        if isinstance(documents, dict) and "error" in documents:
            return {"reply": f"⚠ 本機 API 回傳錯誤：{documents['error']}", "references": []}
        print("📚 取得 documents：", documents)
    except Exception as e:
        return {"reply": f"⚠ 無法從本機 API 獲取段落：{str(e)}", "references": []}

    if not documents:
        return {"reply": "⚠ 沒有找到任何相關段落。", "references": []}

    # 建立 context
    context_snippets = [doc['text'] for doc in documents]
    prompt = f"你是一位 ESG 永續報告分析師，請根據以下參考資料回答問題，若找不到答案請誠實說明。\n\n參考資料：\n{chr(10).join(context_snippets)}\n\n問題：{req.message}\n回答："
    result = pipe(prompt, max_new_tokens=3000)[0]['generated_text']
    answer = result.split("回答：")[-1].split("問題：")[0].strip() if "回答：" in result else result.strip()

    return {"reply": answer, "references": documents}

@app.post("/gpt_chat")
def gpt_chat(req: QueryRequest):
    """
    使用 RAG 檢索相關文件，然後透過 Groq API 回答問題
    """
    # 檢查 Groq API Key
    if not groq_api_key:
        return {"error": "⚠ GROQ_API_KEY 未設定", "references": []}
    
    # 檢查 FAISS 索引是否存在
    if index is None:
        return {"reply": "⚠ FAISS 索引尚未初始化，請先上傳文件。", "references": []}
    
    try:
        # === RAG 檢索階段 ===
        query_vec = embedding_model.get_embedding(req.message, device).cpu().numpy()
        D, I = index.search(query_vec, 5)  # 獲取更多相關文檔

        print("🔍 FAISS 找到的向量 ID 索引：", I)
        top_ids = [paragraph_ids[idx] for idx in I[0] if idx < len(paragraph_ids)]
        print("🧾 FAISS 找到的 MongoDB _id：", top_ids)

        if not top_ids:
            # 如果沒有找到相關文檔，直接使用 Groq 回答
            try:
                completion = groq_client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一位專業的 ESG 永續報告分析師。請根據你的專業知識回答用戶的問題。如果問題超出你的知識範圍，請誠實說明。"
                        },
                        {
                            "role": "user",
                            "content": req.message
                        }
                    ],
                    temperature=0.7,
                    max_completion_tokens=1000,
                    top_p=1,
                    reasoning_effort="medium",
                    stream=False,
                    stop=None
                )
                
                answer = completion.choices[0].message.content
                return {"reply": answer, "references": [], "note": "基於一般知識回答（未找到相關文檔）"}
            
            except Exception as e:
                return {"error": f"⚠ Groq API 調用失敗：{str(e)}", "references": []}

        # === 獲取相關文檔 ===
        try:
            api_url = local_api if local_api.startswith('http') else f"https://{local_api}"
            response = requests.post(f"{api_url}/get_docs", json={"ids": top_ids}, timeout=10)
            documents = response.json()
            
            if isinstance(documents, dict) and "error" in documents:
                return {"reply": f"⚠ 本機 API 回傳錯誤：{documents['error']}", "references": []}
            
            print("📚 取得 documents：", len(documents))
            
        except Exception as e:
            return {"error": f"⚠ 無法從本機 API 獲取段落：{str(e)}", "references": []}

        if not documents:
            return {"error": "⚠ 沒有找到任何相關段落。", "references": []}

        # === 準備上下文並呼叫 Groq API ===
        context_snippets = [doc['text'] for doc in documents]
        context_text = "\n\n".join(context_snippets)
        
        # 限制上下文長度避免超過 token 限制
        if len(context_text) > 4000:  # 預留給問題和系統提示的空間
            context_text = context_text[:4000] + "..."
        
        try:
            completion = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "system",
                        "content": """你是一位專業的 ESG 永續報告分析師。請根據提供的參考資料回答用戶的問題。

回答要求：
1. 主要依據提供的參考資料回答
2. 如果參考資料中沒有相關資訊，請明確說明
3. 回答要準確、專業且易懂
4. 如果可能，請引用具體的數據或事實
5. 回答中不得包含任何特殊換行符號（如換行）、HTML 標籤（如 <br>），而應直接輸出正常文字段落。"""
                    },
                    {
                        "role": "user", 
                        "content": f"""參考資料：
{context_text}

問題：{req.message}

請根據以上參考資料回答問題。"""
                    }
                ],
                temperature=0.7,
                max_completion_tokens=1000,
                top_p=1,
                reasoning_effort="medium",
                stream=False,
                stop=None
            )
            
            answer = completion.choices[0].message.content
            
            return {
                "reply": answer,
                "references": documents,
                "note": "基於文檔內容 + Groq GPT 回答"
            }
            
        except Exception as e:
            return {"error": f"⚠ Groq API 調用失敗：{str(e)}", "references": documents}
    
    except Exception as e:
        print(f"gpt_chat 發生錯誤: {e}")
        return {"error": f"⚠ 處理請求時發生錯誤：{str(e)}", "references": []}

# === 上傳端點（原有的檔案上傳功能）===
@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    print(f"📤 收到上傳檔案: {file.filename}")
    
    upload_dir = "./upload"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        processor = DocumentProcessor()
        text = processor.extract_text(file_path)
        
        # 清理臨時檔案
        os.remove(file_path)
        
        if not text.strip():
            return {"error": "⚠ 檔案無法擷取文字內容"}

        # 處理向量化
        paragraphs_count, dimension, preview, new_faiss_added = process_document_to_vectors(
            text, file.filename
        )

        return {
            "message": "✅ 文件已向量化",
            "paragraphs": paragraphs_count,
            "dimension": dimension,
            "preview": preview,
            "new_faiss_added": new_faiss_added,
            "text_length": len(text)
        }
    
    except Exception as e:
        # 確保清理臨時檔案
        if os.path.exists(file_path):
            os.remove(file_path)
        return {"error": f"⚠ 檔案處理失敗: {str(e)}"}

# === 新的 URL 下載端點 ===
@app.post("/upload_documents")
async def upload_documents(req: UploadDocumentRequest):
    """
    從 URL 下載文件並進行向量化處理
    """
    print(f"📤 收到文件處理請求: {req.file_name} from {req.file_url}")

    # 驗證檔案名稱
    if not req.file_name:
        return {"error": "⚠ 檔案名稱為空"}

    # 驗證 URL
    if not req.file_url or not req.file_url.startswith(('http://', 'https://')):
        return {"error": "⚠ 無效的檔案 URL"}

    # 檢查檔案副檔名
    ext = os.path.splitext(req.file_name)[1].lower()
    supported_formats = ['.pdf', '.docx', '.doc', '.txt']
    if ext not in supported_formats:
        return {
            "error": f"⚠ 不支援的檔案格式 '{ext}'",
            "supported_formats": supported_formats
        }

    # 建立臨時目錄
    temp_dir = "./temp_downloads"
    os.makedirs(temp_dir, exist_ok=True)

    # 生成唯一的本機檔案路徑
    unique_id = str(uuid.uuid4())[:8]
    safe_filename = f"{unique_id}_{req.file_name}"
    local_file_path = os.path.join(temp_dir, safe_filename)

    try:
        # 下載檔案
        if not download_file(req.file_url, local_file_path):
            return {"error": "⚠ 檔案下載失敗"}

        # 處理文件
        processor = DocumentProcessor()
        text = processor.extract_text(local_file_path)

        if not text or not text.strip():
            return {
                "error": "⚠ 檔案無法擷取文字內容",
                "file_name": req.file_name,
                "possible_reasons": [
                    "檔案可能是掃描的圖片 PDF（需要 OCR）",
                    "檔案內容為空或損壞",
                    "檔案格式不正確",
                    "檔案使用不支援的編碼",
                    "檔案包含太多無意義的內容（已被過濾）"
                ]
            }

        # 處理向量化
        paragraphs_count, dimension, preview, new_faiss_added = process_document_to_vectors(
            text, req.file_name, req.file_url
        )

        return {
            "message": "✅ 文件已成功處理並向量化",
            "file_name": req.file_name,
            "source_url": req.file_url,
            "paragraphs": paragraphs_count,
            "dimension": dimension,
            "preview": preview,
            "new_faiss_added": new_faiss_added,
            "text_length": len(text),
            "chunks_processed": paragraphs_count
        }

    except Exception as e:
        print(f"⚠ 處理檔案時發生錯誤: {e}")
        return {
            "error": f"⚠ 檔案處理失敗: {str(e)}",
            "file_name": req.file_name
        }

    finally:
        # 清理臨時檔案
        try:
            if os.path.exists(local_file_path):
                os.remove(local_file_path)
                print(f"🗑️ 已清理臨時檔案: {local_file_path}")
        except Exception as e:
            print(f"⚠️ 清理臨時檔案失敗: {e}")

# === 批次處理端點 ===
@app.post("/upload_documents_batch")
async def upload_documents_batch(req: BatchUploadRequest):
    """
    批次處理多個文件
    """
    results = []

    for file_info in req.files:
        try:
            file_req = UploadDocumentRequest(**file_info)
            result = await upload_documents(file_req)
            results.append({
                "file_name": file_info.get("file_name"),
                "status": "success" if "error" not in result else "error",
                "result": result
            })
        except Exception as e:
            results.append({
                "file_name": file_info.get("file_name", "unknown"),
                "status": "error",
                "result": {"error": f"⚠ 處理失敗: {str(e)}"}
            })

    return {
        "message": f"批次處理完成",
        "total_files": len(req.files),
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
