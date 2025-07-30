from flask import Flask, request, jsonify
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path
import os

app = Flask(__name__)
load_dotenv(dotenv_path=Path("/Users/chenjiaxiang/SDGsLLM/.env"))  # 預設會找 .env 檔案

# ✅ 使用 Atlas MongoDB URI
client = MongoClient(os.getenv("MONGO_URI"))
# ✅ 指定資料庫與集合
db = client["vector_db"]
collection = db["paragraphs"]

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "pong"})

@app.route("/insert_doc", methods=["POST"])
def insert_doc():
    doc = request.json
    collection.insert_one(doc)
    return jsonify({"message": "✅ 文件已寫入"})

@app.route("/get_docs", methods=["POST"])
def get_docs():
    ids = request.json.get("ids", [])
    try:
        results = collection.find({"_id": {"$in": ids}})  # 不轉 ObjectId，直接比對字串
        docs = [{"_id": str(doc["_id"]), "text": doc["text"], "label": doc.get("label", -1)} for doc in results]
        return jsonify(docs)
    except Exception as e:
        return jsonify({"error": f"❌ 查詢時發生錯誤：{str(e)}"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

