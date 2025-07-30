from pyngrok import ngrok, conf
import os

load_dotenv()  # 預設會找 .env 檔案

# ✅ 輸入你自己的 ngrok token
conf.get_default().auth_token = os.getenv("NGROK_TOKEN")

# ✅ 開啟 8000 port
public_url = ngrok.connect(8000)
print("🚀 你的 API 可從這裡存取：", public_url)

