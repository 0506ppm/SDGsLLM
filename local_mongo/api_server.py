from flask import Flask, request, jsonify
from pymongo import MongoClient
import os

app = Flask(__name__)

# ✅ 使用 Atlas MongoDB URI
client = MongoClient("mongodb+srv://0506ppm:tt920506@cluster0.ozc6lzs.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

# ✅ 指定資料庫與集合
db = client["SDGs"]
collection = db["documents"]

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

