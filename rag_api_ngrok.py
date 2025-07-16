from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModel
from peft import PeftModel, PeftConfig
from pyngrok import ngrok, conf
from pymongo import MongoClient
import torch
import faiss
import json
import os

# === Ngrok 設定 ===
conf.get_default().auth_token = "2zdWETPLyoPte8AydDb2uyo2aHi_3FZCs9z9EPoaB8W1v1nGo"
app = FastAPI()

# === 載入 RAG 模型 ===
print("🚀 載入 LLM 模型中...")
lora_model_path = "./output/not_finetuned_mistral7b-lora"
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

embedding_tokenizer = AutoTokenizer.from_pretrained(model_dir)
embedding_base_model = AutoModel.from_pretrained(model_dir)

# 你的 EmbeddingModel class（請放在前面或單獨檔案引入）
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
embedding_model = embedding_model.to(device)

# === MongoDB 與 FAISS ===
mongo_client = MongoClient("mongodb+srv://0506ppm:tt920506@cluster0.ozc6lzs.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
collection = mongo_client["vector_db"]["paragraphs"]

index = faiss.read_index("faiss_index.index")
with open("faiss_ids.json", "r") as f:
    paragraph_ids = json.load(f)

label_map = {0: "初級做法", 1: "進階做法", 2: "領先做法"}

# === API Schema ===
class QueryRequest(BaseModel):
    message: str

# === 第一個 API：單純 LLM ===
@app.post("/chat")
def chat(req: QueryRequest):
    prompt = f"請根據以下問題給出正確答案：\n問題：{req.message}"
    result = pipe(prompt, max_new_tokens=200)[0]['generated_text']
    return {"reply": result}

# === 第二個 API：向量檢索 + LLM ===
@app.post("/rag_chat")
def rag_chat(req: QueryRequest):
    # 1. 查 FAISS
    query_vec = embedding_model.get_embedding(req.message, device).cpu().numpy()
    D, I = index.search(query_vec, 3)

    # 2. 查 MongoDB
    retrieved_texts = []
    for idx in I[0]:
        para_id = paragraph_ids[idx]
        doc = collection.find_one({"_id": para_id})
        if doc:
            label = label_map.get(doc.get("label", -1), "未標記")
            retrieved_texts.append(f"[{label}] {doc['text']}")

    context = "\n".join(retrieved_texts)

    # 3. 組 prompt 丟給 LLM
    prompt = f"""請根據以下參考資料回答問題：

參考資料：
{context}

問題：{req.message}
請根據資料清楚回答問題，如果找不到答案請說明。"""

    result = pipe(prompt, max_new_tokens=200)[0]['generated_text']
    return {
        "reply": result,
        "context": context
    }

# === 啟動服務 ===
if __name__ == "__main__":
    import uvicorn
    port = 8000
    public_url = ngrok.connect(port)
    print(f"🌐 公開網址：{public_url}/chat 或 /rag_chat")
    uvicorn.run(app, host="0.0.0.0", port=port)
