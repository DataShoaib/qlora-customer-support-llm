from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

BASE_DIR = Path(__file__).resolve().parents[2]
TRAIN_FILE = BASE_DIR / "data/processed/train.jsonl"
EVAL_FILE = BASE_DIR / "data/processed/eval.jsonl"
OUTPUT_DIR = BASE_DIR / "outputs/adapter"
MODEL_ID = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
dataset = load_dataset("json", data_files={"train":str(TRAIN_FILE),"eval":str(EVAL_FILE)})

def format_example(example):
    return {"text": tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False,
        enable_thinking=False
    )}

dataset = dataset.map(format_example)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto"
)
model.config.use_cache = False

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj"]
)

args = SFTConfig(
    output_dir=str(OUTPUT_DIR), num_train_epochs=2,
    per_device_train_batch_size=2, per_device_eval_batch_size=2,
    gradient_accumulation_steps=4, learning_rate=2e-4,
    logging_steps=5, eval_strategy="steps", eval_steps=20,
    save_steps=20, save_total_limit=2, max_length=512,
    packing=True, fp16=True, report_to="none", seed=42
)

trainer = SFTTrainer(
    model=model, args=args, train_dataset=dataset["train"],
    eval_dataset=dataset["eval"], processing_class=tokenizer,
    peft_config=peft_config
)

trainer.train()
print(trainer.evaluate())
trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
print(f"Adapter saved to {OUTPUT_DIR}")
