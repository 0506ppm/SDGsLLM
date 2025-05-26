#!/bin/bash
#SBATCH --job-name=llm-finetune
#SBATCH --account=mst113116
#SBATCH --partition=ct56
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=llm.out
#SBATCH --error=llm.err

# === 載入模組 ===
module purge
module load compiler/gcc/11.2.0 cuda/11.8 python/3.10

# === 啟用 Conda 環境 ===
source ~/miniconda3/bin/activate
conda activate llm

# === 設定 Hugging Face 快取資料夾（避免寫滿 home）===
export HF_HOME=/work/ppmm0506/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

# === 切換目錄到你的專案資料夾 ===
cd ~/SDGsLLM

# === 執行訓練 ===
python LLM_Fine_Tune.py

