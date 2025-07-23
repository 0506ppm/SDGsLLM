from pyngrok import ngrok, conf

# ✅ 輸入你自己的 ngrok token
conf.get_default().auth_token = "30GlQy4n4ri3sbqPuLQMjbFELlr_7aWLtdRFWrevtcawL29XX"

# ✅ 開啟 8000 port
public_url = ngrok.connect(8000)
print("🚀 你的 API 可從這裡存取：", public_url)

