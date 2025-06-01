#!/usr/bin/env python3
"""
Plain LoRA-SFT with Unsloth; 4-bit QLoRA; single or multi-GPU.
Outputs ./checkpoints/latest
"""
from pathlib import Path
import json, torch
from datasets import load_dataset
from unsloth import PatchLoraModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, TaskType
from trl import SFTTrainer

# ----- paths -----------------------------------------------------------
MODEL_NAME   = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DATA_JSONL   = "data/balatro_sft.jsonl"     # or balatro_mix.jsonl
OUTPUT_DIR   = "checkpoints/latest"

# ----- tokenizer -------------------------------------------------------
tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
tok.pad_token = tok.eos_token

# ----- dataset ---------------------------------------------------------
ds = load_dataset("json", data_files=DATA_JSONL, split="train")
def tok_fn(b): return tok(b["prompt"] + b["completion"])
ds = ds.map(tok_fn, batched=False, remove_columns=["prompt","completion"])

# ----- base model ------------------------------------------------------
base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, load_in_4bit=True, device_map="auto", trust_remote_code=True
)

# ----- patch LoRA ------------------------------------------------------
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = PatchLoraModel(base, lora_cfg)

# ----- training args ---------------------------------------------------
args = TrainingArguments(
    per_device_train_batch_size = 128,
    gradient_accumulation_steps = 16,
    num_train_epochs = 2,
    learning_rate = 2e-4,
    fp16 = True,
    logging_steps = 50,
    output_dir = OUTPUT_DIR,
    save_strategy = "epoch",
)

trainer = SFTTrainer(
    model           = model,
    tokenizer       = tok,
    train_dataset   = ds,
    args            = args,
    dataset_text_field = None,
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
print("LoRA-SFT done →", OUTPUT_DIR)

