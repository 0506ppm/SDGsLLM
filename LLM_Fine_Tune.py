import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# === 設定快取與路徑 ===
os.environ["HF_HOME"] = "/work/ppmm0506/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
os.environ["HF_DATASETS_CACHE"] = os.environ["HF_HOME"]

output_dir = "/work/ppmm0506/fine-tuned-models/mistral7b-lora"
data_file = "/home/ppmm0506/SDGsLLM/llama_instruction_qa_dataset.json"

# === 載入資料集 ===
dataset = load_dataset("json", data_files=data_file)["train"]
dataset = dataset.train_test_split(test_size=0.1)

# === 載入 tokenizer 與模型（不使用 bitsandbytes） ===
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16  # 或換成 torch.float16 若遇 bfloat16 不支援
)
model = prepare_model_for_kbit_training(model)

# === LoRA 設定 ===
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# === 格式化資料集 ===
def format_example(example):
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')

    prompt = f"{instruction}\n{input_text}" if input_text else instruction
    return prompt + "\n" + output

train_texts = [format_example(e) for e in dataset['train']]
test_texts = [format_example(e) for e in dataset['test']]

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings["input_ids"])
    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}

train_dataset = TextDataset(train_encodings)
test_dataset = TextDataset(test_encodings)

# === 訓練參數 ===
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=True,   # 或改成 fp16=True
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator
)

trainer.train()

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

