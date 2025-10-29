#!/usr/bin/env python3
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

model_name = "Qwen/Qwen2-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)

# 加载数据集（假设 CSV: input, output）
dataset = load_dataset("csv", data_files="dataset.csv", split="train")

def format_example(example):
    messages = [{"role": "system", "content": "Parse to JSON."}, {"role": "user", "content": example["input"]}, {"role": "assistant", "content": example["output"]}]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

dataset = dataset.map(format_example)

training_args = TrainingArguments(output_dir="./qwen_finetuned", num_train_epochs=3, per_device_train_batch_size=4, fp16=True)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer)
trainer.train()
model.save_pretrained("./qwen_finetuned")
