# SDGs ESG 永續報告智能問答系統

這是一個基於 RAG (Retrieval-Augmented Generation) 架構的 ESG 永續報告智能問答系統，專注於台灣企業的永續發展報告分析。系統整合了向量檢索、語言模型和 Rerank 技術，提供精確的 ESG 相關問題解答。

## 🌟 專案特色

- **進階 RAG 架構**：結合 FAISS 向量檢索 + Cohere Rerank + LLM 生成
- **多模型支援**：
  - 本地微調模型（Mistral-7B + LoRA）
  - Groq API（GPT-OSS-120B）
- **智慧文件處理**：支援 PDF、DOCX、DOC、TXT 等多種格式
- **角色權限管理**：內部人士可查看精確數字，外部人士僅能獲取模糊資訊
- **信心門檻過濾**：只回答信心度 ≥ 0.8 的問題，避免錯誤資訊

## 📋 系統架構
```
使用者提問
    ↓
FAISS 向量檢索 (Top 25)
    ↓
Cohere Rerank 精選 (Top 5)
    ↓
信心門檻過濾 (≥ 0.8)
    ↓
LLM 生成回答
    ↓
角色權限處理
    ↓
返回結果
```

## 🛠️ 技術棧

### 後端框架
- **FastAPI**：RESTful API 服務
- **PyTorch**：深度學習框架
- **Transformers**：Hugging Face 模型庫

### 核心技術
- **向量檢索**：FAISS (Facebook AI Similarity Search)
- **向量化模型**：自訓練 Sustainability Embedding Model
- **語言模型**：
  - Mistral-7B-Instruct-v0.2 (微調版)
  - Groq API (GPT-OSS-120B)
- **Rerank 模型**：Cohere Rerank Multilingual v3.0

### 資料儲存
- **MongoDB Atlas**：文件段落儲存
- **FAISS**：向量索引

### 文件處理
- **PyMuPDF (fitz)**：PDF 解析
- **python-docx**：DOCX 處理
- **Apache Tika**：通用文件解析

## 📦 安裝指南

### 環境需求（跑在國網）

- Python 3.10+
- CUDA 11.8+ (使用 GPU 時)
- 64GB+ RAM 建議

### 安裝步驟

1. **Clone專案**
```bash
git clone <repository-url>
cd SDGsLLM
```

2. **建立虛擬環境**
```bash
conda create -n llm python=3.10
conda activate llm
```

3. **安裝依賴套件**
```bash
pip install -r requirements.txt
```

4. **設定環境變數**

建立 `.env` 檔案：
```env
# MongoDB 連線
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/

# API 服務
LOCAL_API=https://your-ngrok-url.ngrok.io

# Groq API
GROQ_API_KEY=your_groq_api_key

# Cohere API (Rerank)
COHERE_API_KEY=your_cohere_api_key

# Ngrok (可選)
NGROK_TOKEN=your_ngrok_token
```

5. **準備模型檔案**

確保以下路徑存在：
- `./output/result_mistral7b-lora/` - 微調後的 LoRA 模型
- `./sustainability_embedding_model/` - Embedding 模型

## 🚀 使用方式

### 啟動本地 API 伺服器
```bash
# 方法 1: 直接啟動 (本地測試)
python main.py

# 方法 2: 使用 Ngrok 公開服務
python local_mongo/run_ngork.py  # Terminal 1
python local_mongo/api_server.py  # Terminal 2
```

### API 端點說明

#### 1. 一般聊天（無檢索）
```bash
POST /chat
Content-Type: application/json

{
  "message": "什麼是 ESG？"
}
```

#### 2. RAG 問答（本地模型）⭐ 使用這個
```bash
POST /rag_chat
Content-Type: application/json

{
  "message": "台積電的減碳策略是什麼？",
  "role": "內部人士"  # 或 "外部人士"
}
```

#### 3. GPT 問答（Groq API，無檢索）
```bash
POST /gpt_chat
Content-Type: application/json

{
  "message": "解釋 ESG 的重要性"
}
```

#### 4. GPT RAG 問答（Groq + 檢索 + Rerank）
```bash
POST /gpt_rag_chat
Content-Type: application/json

{
  "message": "台積電在水資源管理方面的策略？"
}
```

#### 5. 上傳文件（本地檔案）
```bash
POST /upload-document
Content-Type: multipart/form-data

file: <your-file.pdf>
```

#### 6. 上傳文件（URL）
```bash
POST /upload_documents
Content-Type: application/json

{
  "file_url": "https://example.com/report.pdf",
  "file_name": "sustainability_report_2023.pdf"
}
```

#### 7. 批次上傳
```bash
POST /upload_documents_batch
Content-Type: application/json

{
  "files": [
    {
      "file_url": "https://example.com/report1.pdf",
      "file_name": "report1.pdf"
    },
    {
      "file_url": "https://example.com/report2.pdf",
      "file_name": "report2.pdf"
    }
  ]
}
```

### 回應格式範例
```json
{
  "reply": "台積電的水資源管理策略包括...",
  "references": [
    {
      "_id": "uuid-xxxx",
      "text": "台積電透過水資源回收再利用...",
      "source": "TSMC_2023_Report.pdf",
      "rerank_score": 0.95,
      "rerank_position": 1
    }
  ],
  "rerank_enabled": true,
  "faiss_candidates": 25,
  "documents_used": 3,
  "role": "外部人士",
  "confidence_threshold": 0.8
}
```

## 📊 訓練自己的模型

### LLM 微調

1. **準備訓練資料**
   - 格式：`llama_instruction_qa_dataset.json`
   - 結構：instruction-input-output

2. **執行訓練**
```bash
# 本地訓練
python LLM_Fine_Tune.py

# SLURM 叢集訓練
sbatch train_llm.sh
```

### Embedding 模型訓練

參考 `sustainability_embedding_model/` 內的訓練腳本

## 🔧 核心功能說明

### 文件處理流程

1. **文字提取**：支援多種格式，智慧降級處理
2. **文字清理**：移除無效字符、過濾無意義內容
3. **智能分塊**：按句號分割，避免句子截斷（400 字/塊）
4. **向量化**：使用自訓練 Embedding 模型
5. **儲存**：MongoDB（原文） + FAISS（向量）

### RAG 檢索增強

1. **第一階段**：FAISS 快速檢索 25 個候選文檔
2. **第二階段**：Cohere Rerank 精選前 5 個最相關文檔
3. **第三階段**：信心門檻過濾（≥ 0.8）
4. **第四階段**：LLM 生成回答

### 角色權限控制

- **內部人士**：可查看精確數字與細節
- **外部人士**：數字以「約」「大約」等模糊表達

## 📁 專案結構
```
SDGsLLM/
├── main.py                          # 主要 API 服務
├── LLM_Fine_Tune.py                 # LLM 微調腳本
├── LLM_inference.py                 # 推論測試
├── requirements.txt                 # 依賴套件
├── .env                             # 環境變數（需自行建立）
├── .gitignore                       # Git 忽略檔案
│
├── local_mongo/                     # MongoDB API 服務
│   ├── api_server.py                # 本地 API 伺服器
│   └── run_ngork.py                 # Ngrok 啟動腳本
│
├── output/                          # 模型輸出（不納入版控）
│   └── result_mistral7b-lora/       # 微調後的模型
│
├── sustainability_embedding_model/  # Embedding 模型（不納入版控）
│
├── faiss_data/                      # FAISS 索引（不納入版控）
│   ├── faiss_index.index
│   └── faiss_ids.json
│
├── upload/                          # 暫存上傳檔案
├── temp_downloads/                  # 暫存下載檔案
│
└── llama_instruction_qa_dataset.json # 訓練資料集
```

## ⚠️ 注意事項

### 資安提醒
- ✅ `.env` 檔案已加入 `.gitignore`，請勿提交 API Key
- ✅ MongoDB URI 包含密碼，請妥善保管
- ✅ Ngrok Token 為個人專用，切勿公開

### 效能優化
- 使用 GPU 可大幅提升推論速度
- FAISS 索引建議定期重建優化
- 批次上傳時建議每批不超過 50 個檔案

### 常見問題

1. **記憶體不足**
   - 調整 `max_length` 參數
   - 使用 `offload_folder` 分流記憶體

2. **文件無法解析**
   - 確認 Tika 已正確安裝
   - 檢查文件是否為掃描 PDF（需 OCR）

3. **Rerank 失敗**
   - 檢查 `COHERE_API_KEY` 是否設定
   - 系統會自動降級為 FAISS 檢索

## 🤝 貢獻指南

歡迎提交 Issue 和 Pull Request！

## 📄 授權

本專案僅供學術研究使用。

## 👥 聯絡方式

如有任何問題，請聯繫 0506ppm@gmail.com

---

**最後更新**：2025年1月
