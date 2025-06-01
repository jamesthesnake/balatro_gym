#!/usr/bin/env python3
"""
02_peft.py  –  LoRA supervised fine-tune (SFT) for DeepSeek-8B

* Uses bitsandbytes 4-bit QLoRA + PEFT.
* Expects a jsonl dataset where each line has {"prompt": ..., "completion": ...}
* Writes merged checkpoint (base + LoRA) to ./checkpoints/latest
"""

from pathlib import Path
import json, torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer
from accelerate import Accelerator # IMPORT ACCELERATOR

# ──────────────────────────────────────────────────────────────
# CONFIG — tweak as needed
# ──────────────────────────────────────────────────────────────
MODEL_NAME   = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DATA_JSONL   = "data/balatro_sft.jsonl"       # or data/balatro_mix.jsonl
OUTPUT_DIR   = "checkpoints/latest"
EPOCHS       = 2
BATCH_SIZE   = 32       # per-GPU micro-batch
GRAD_ACCUM   = 32       # Gradient accumulation
SEQ_LEN      = 256
LR           = 2e-4
SEED         = 42
LORA_RANK    = 64
LORA_ALPHA   = 16
# ──────────────────────────────────────────────────────────────


def main():
    # Initialize Accelerator early to get device information for each process
    accelerator = Accelerator() # INITIALIZE ACCELERATOR

    # --- Tokenizer ────────────────────────────────────────────
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    tok.pad_token = tok.eos_token

    # --- Dataset ──────────────────────────────────────────────
    # Load the full dataset first
    ds = load_dataset("json", data_files=DATA_JSONL, split="train")
    
    # MODIFIED: Select a 5% random subset of the dataset
    percentage_to_use = 0.01  # 5%
    num_examples_to_use = int(len(ds) * percentage_to_use)
    
    # Ensure there's at least 1 example if 5% is too small (e.g., for very small datasets)
    if num_examples_to_use == 0 and len(ds) > 0:
        print(f"Warning: 5% of dataset size {len(ds)} is 0. Using 1 example instead for subset.")
        num_examples_to_use = 1 
    elif num_examples_to_use == 0 and len(ds) == 0:
        print(f"Error: Dataset {DATA_JSONL} is empty.")
        return # Exit if dataset is empty

    if accelerator.is_main_process: # Print dataset info only on main process
        print(f"Original dataset size: {len(ds)}")
        print(f"Selecting {percentage_to_use*100}% of the dataset: {num_examples_to_use} examples.")

    ds = ds.shuffle(seed=SEED) # Shuffle before selecting for a random subset
    ds = ds.select(range(num_examples_to_use))
    
    if accelerator.is_main_process:
        print(f"Using a random subset of {len(ds)} examples for training.")


    def concat_fn(example):
        return {"text": example["prompt"] + example["completion"]}

    ds = ds.map(concat_fn, remove_columns=list(ds.column_names))

    # --- Base model w/ 4-bit quant ────────────────────────────
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    current_process_device = accelerator.device
    if accelerator.is_main_process:
        print(f"Process {accelerator.process_index} (main) attempting to load model to device: {current_process_device}")
    else:
        print(f"Process {accelerator.process_index} attempting to load model to device: {current_process_device}")

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_cfg,
        device_map={'': current_process_device}, # USE ACCELERATOR.DEVICE
        trust_remote_code=True,
    )

    # --- Add LoRA adapters with PEFT ──────────────────────────
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)
    if accelerator.is_main_process:
        model.print_trainable_parameters()

    # --- Training args (using SFTConfig) ─────────────────────
    args = SFTConfig(
        output_dir                = OUTPUT_DIR,
        num_train_epochs          = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate             = LR,
        fp16                      = True,
        logging_steps             = 10, # Log more frequently with smaller dataset
        seed                      = SEED,
        save_strategy             = "epoch",
        report_to                 = "none", 
        max_seq_length            = SEQ_LEN,
        packing                   = False,  # Explicitly disable packing
        # dataset_text_field        = "text", # Omitted
    )

    # --- TRL SFTTrainer handles packing, padding, etc. ────────
    trainer = SFTTrainer(
        model               = model,
        train_dataset       = ds,
        args                = args,
    )
    
    print(f"Process {accelerator.process_index} starting training...")
    trainer.train()
    
    accelerator.wait_for_everyone() 
    if accelerator.is_main_process:
        print(f"\n[SFT Process {accelerator.process_index}] Training complete. Saving final model/adapter to {OUTPUT_DIR}")
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(OUTPUT_DIR)
        tok.save_pretrained(OUTPUT_DIR)
        print(f"[SFT Process {accelerator.process_index}] Model/adapter and tokenizer saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
