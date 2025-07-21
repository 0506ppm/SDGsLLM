from pymongo import MongoClient

try:
    uri = "mongodb+srv://0506ppm:<db_password>@cluster0.ozc6lzs.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(uri)
    dbs = client.list_database_names()
    print("✅ 成功連線，資料庫列表：", dbs)
except Exception as e:
    print("❌ 無法連線：", e)

