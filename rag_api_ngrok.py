from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from pyngrok import ngrok, conf
import torch
import os

conf.get_default().auth_token = "2zdWETPLyoPte8AydDb2uyo2aHi_3FZCs9z9EPoaB8W1v1nGo"

app = FastAPI()

# === Step 1: 載入模型 ===
print("🚀 載入模型中...")
lora_model_path = "./output/result_mistral7b-lora"
base_model_path = PeftConfig.from_pretrained(lora_model_path).base_model_name_or_path

tokenizer = AutoTokenizer.from_pretrained(base_model_path)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    offload_folder="./offload",  # ✅ 確保這個資料夾存在
    torch_dtype="auto"
)

model = PeftModel.from_pretrained(base_model, lora_model_path)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

print("✅ 模型載入完成")

# === Step 2: 定義 API Schema ===
class QueryRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(request: QueryRequest):
    prompt = f"請根據以下問題給出正確答案：\n問題：{request.message}"
    output = pipe(prompt, max_new_tokens=100)[0]['generated_text']
    return {"reply": output}

# === Step 3: 開啟 ngrok tunnel 並啟動 uvicorn ===
if __name__ == "__main__":
    import uvicorn

    port = 8000
    public_url = ngrok.connect(port)
    print(f"🌐 公開網址：{public_url}/chat")
    print("📡 FastAPI 伺服器啟動中...")

    uvicorn.run(app, host="0.0.0.0", port=port)

