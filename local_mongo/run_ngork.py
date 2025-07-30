from pyngrok import ngrok, conf
from dotenv import load_dotenv
from pathlib import Path
import time
import os

load_dotenv(dotenv_path=Path("/Users/chenjiaxiang/SDGsLLM/.env"))

# ✅ 輸入你自己的 ngrok token
conf.get_default().auth_token = os.getenv("NGROK_TOKEN")

# ✅ 開啟 8000 port
public_url = ngrok.connect(8000)
print("🚀 你的 API 可從這裡存取：", public_url)

# 🛑 程式不可以馬上結束！用 while True 撐住
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("🔌 中斷 ngrok")
    ngrok.disconnect(public_url)

