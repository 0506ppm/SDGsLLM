# ✅ 推論測試
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model = AutoModelForCausalLM.from_pretrained(
    output_dir,
    device_map="auto",
    offload_folder="/content/offload"
)
tokenizer = AutoTokenizer.from_pretrained(output_dir)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

prompt = "請根據以下問題給出正確答案：\n問題：台泥的減碳策略包含哪些關鍵措施？"
print(pipe(prompt, max_new_tokens=100)[0]['generated_text'])
