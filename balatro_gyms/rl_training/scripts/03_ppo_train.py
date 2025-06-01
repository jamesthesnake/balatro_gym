#!/usr/bin/env python3
"""
TRL PPO fine-tune for Balatro-Bot using trl.PPOTrainer.
(Revised for robust multi-GPU device handling with accelerate)
"""
from __future__ import annotations
import math
import re
import json
from pathlib import Path
import torch
import os # For LOCAL_RANK
from tqdm import tqdm

from datasets import Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import PeftModel
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
# from trl.core import LengthSampler # Only if you use it explicitly

# ---- Constants and Your Existing Functions ----
THOUGHT_RE = re.compile(r"^Thought:.*$", re.MULTILINE)
SFT_MODEL_ADAPTER_PATH = "checkpoints/latest"
BASE_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
PPO_OUTPUT_DIR = "checkpoints/ppo_balatro_bot_v2" # Use a distinct output dir

# Your reward function (ensure it returns Python floats or CPU tensors, PPOTrainer will handle device for rewards)
def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    rewards_py = []
    for comp in completions:
        try:
            js = json.loads(THOUGHT_RE.sub("", comp).strip())
            raw = js.get("raw_score", 0)
        except Exception:
            raw = 0
        rewards_py.append(math.sqrt(raw) / 100.0 - 0.01)
    return rewards_py


def main():
    # --- Accelerator and Device Setup (Crucial for Multi-GPU) ---
    # accelerate launch sets LOCAL_RANK. Each process uses this to set its PyTorch device.
    # This must happen BEFORE any CUDA operations or model loading that relies on current_device.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    current_device = torch.cuda.current_device() # This will now be correctly cuda:0 for rank0, cuda:1 for rank1
    
    # Initialize Accelerator AFTER setting the device for the process.
    # PPOTrainer will internally create an Accelerator or use one if configured globally.
    # For simplicity, we'll let PPOTrainer manage its internal accelerator.
    # We primarily needed the correct `current_device` for `device_map`.

    print(f"Process LOCAL_RANK {local_rank} set to use CUDA device: {current_device}")

    # --- PPO Configuration ---
    # (Using values from your trlx_ppo.yml and PPOConfig signature)
    ppo_config = PPOConfig(
        seed=1234,
        num_ppo_epochs=4,
        kl_coef=0.05,
        batch_size=128, # Rollout buffer size (experiences collected before PPO update)
        # mini_batch_size will be per_device for the PPO updates.
        # If None, PPOTrainer calculates: batch_size / (num_gpus * gradient_accumulation_steps)
        # For 2 GPUs, 128 / (2 * 8) = 8. This will be the per-device mini_batch_size.
        mini_batch_size=None, # Let TRL calculate, or set to 8
        gradient_accumulation_steps=8, # For PPO optimization loop
        learning_rate=5e-05, # From your PPOConfig signature default
        output_dir=PPO_OUTPUT_DIR, # Use the new constant
        report_to=None,
        fp16=True, # Assuming you want mixed precision for PPO updates
        gamma=1.0,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.1,
        remove_unused_columns=False, # Often good to set False with PPOTrainer and custom data
    )

    # --- Load Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    special_tokens_to_add = ["<SEP>", "<EOS>"] # From your YML
    num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token # Also set the token string

    # --- Load SFT Model as Actor and create Reference Model ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print(f"Process {local_rank} loading Actor model onto device: {current_device}")
    # Load base for actor, apply PEFT, then wrap with ValueHead
    # The key is that `device_map` uses the `current_device` correctly set for this process.
    actor_model_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map={'': current_device}, # Model loaded onto this process's assigned GPU
        trust_remote_code=True,
    )
    actor_peft_model = PeftModel.from_pretrained(actor_model_base, SFT_MODEL_ADAPTER_PATH, is_trainable=True)
    actor = AutoModelForCausalLMWithValueHead.from_pretrained(actor_peft_model)

    print(f"Process {local_rank} loading Reference model onto device: {current_device}")
    ref_model_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map={'': current_device}, # Model loaded onto this process's assigned GPU
        trust_remote_code=True,
    )
    ref_model = PeftModel.from_pretrained(ref_model_base, SFT_MODEL_ADAPTER_PATH)

    if num_added_toks > 0:
        print(f"Process {local_rank} resizing token embeddings for actor and ref models.")
        actor.resize_token_embeddings(len(tokenizer))
        ref_model.resize_token_embeddings(len(tokenizer))
        # Also resize the value head if it's tied to embeddings.
        # AutoModelForCausalLMWithValueHead often handles this if a new model is passed.
        # If issues arise, explicitly re-initialize or resize actor.v_head.
        if hasattr(actor, 'v_head'): # Check if v_head exists
             actor.v_head = torch.nn.Linear(actor.config.hidden_size, 1, bias=False).to(current_device)


    # --- Prepare Prompts (Dataset/Generator) ---
    # This needs to be your actual prompt source.
    # Your trlx script used: dummy_prompts = ["<START>"] * cfg.train.total_steps
    total_ppo_steps = 8000 # From your trlx_ppo.yml
    # Example placeholder for prompt generation:
    example_prompts_text = ["<START_BALATRO_PROMPT_EXAMPLE>"] * ppo_config.batch_size


    # --- Initialize PPOTrainer ---
    # PPOTrainer will initialize its own Accelerator internally using the environment
    # variables set by `accelerate launch`.
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=actor,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    # Generation kwargs
    # SEQ_LEN from your old SFT config was 256.
    # max_new_tokens should be SEQ_LEN - typical_prompt_length
    SEQ_LEN = 256 
    generation_kwargs = {
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id, # Ensure EOS is correctly handled
        "max_new_tokens": 150, # Adjust: e.g., SEQ_LEN - (avg prompt token length)
    }

    # --- PPO Training Loop ---
    for ppo_step in tqdm(range(total_ppo_steps), desc=f"PPO Steps (Rank {local_rank})", disable=not ppo_trainer.accelerator.is_local_main_process):
        # Replace this with your actual prompt fetching logic for each PPO batch
        current_prompts_text = example_prompts_text 

        # Tokenize prompts. PPOTrainer expects a list of 1D tensors (input_ids).
        # Max length for prompt tokenization should allow space for generated response.
        prompt_max_len = SEQ_LEN - generation_kwargs["max_new_tokens"]
        query_tokens_dict = tokenizer(
            current_prompts_text,
            padding=True, # Pad to longest in batch
            truncation=True,
            max_length=prompt_max_len, 
            return_tensors="pt" 
        )
        # PPOTrainer needs a list of tensors for queries
        query_tensors_list = [tensor.to(ppo_trainer.accelerator.device) for tensor in query_tokens_dict["input_ids"]]

        # Get model responses (completions only, as PPOTrainer.generate with return_prompt=False)
        response_tensors_list = ppo_trainer.generate(
            query_tensors_list, # Must be list of 1D tensors
            return_prompt=False, # Generate only the response part
            **generation_kwargs
        )
        
        # For reward function, create the "completion text" (generated part only)
        completions_text = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors_list]

        # Calculate reward using your reward function
        rewards_py = reward_fn(prompts=current_prompts_text, completions=completions_text)
        reward_tensors = [torch.tensor(r, dtype=torch.float32, device=ppo_trainer.accelerator.device) for r in rewards_py]
        
        # PPO step
        stats = ppo_trainer.step(query_tensors_list, response_tensors_list, reward_tensors)
        
        # Log statistics (only on main process)
        if ppo_trainer.accelerator.is_main_process:
            log_stats = {f"ppo/{k}": v for k, v in stats.items()}
            log_stats["ppo/mean_reward"] = torch.mean(torch.stack(reward_tensors)).item()
            # ppo_trainer.accelerator.log(log_stats, step=ppo_step) # If using accelerator's tracker
            tqdm.write(f"Step {ppo_step}: Mean Reward: {log_stats['ppo/mean_reward']:.4f}, Policy Loss: {stats.get('ppo/loss/policy'):.4f}, Value Loss: {stats.get('ppo/loss/value'):.4f}")

        # Save model periodically
        checkpoint_interval = 1000 # From your YML
        if (ppo_step + 1) % checkpoint_interval == 0:
            ppo_trainer.accelerator.wait_for_everyone() # Ensure all processes are ready before saving
            if ppo_trainer.accelerator.is_main_process:
                save_dir = Path(ppo_config.output_dir) / f"step_{ppo_step+1}"
                save_dir.mkdir(parents=True, exist_ok=True)
                ppo_trainer.save_pretrained(str(save_dir)) # Saves actor's adapter
                tokenizer.save_pretrained(str(save_dir)) # Save tokenizer too
                print(f"Saved PPO model adapter and tokenizer to {save_dir}")

    # --- Save Final Model ---
    ppo_trainer.accelerator.wait_for_everyone()
    if ppo_trainer.accelerator.is_main_process:
        final_save_dir = Path(ppo_config.output_dir) / "final"
        final_save_dir.mkdir(parents=True, exist_ok=True)
        ppo_trainer.save_pretrained(str(final_save_dir))
        tokenizer.save_pretrained(str(final_save_dir))
        print(f"Saved final PPO model adapter and tokenizer to {final_save_dir}")

if __name__ == "__main__":
    # Constants used in main that might have been in global scope or CONFIG before
    # BASE_MODEL_NAME and SFT_MODEL_ADAPTER_PATH are already global constants here.
    # SEQ_LEN is also global.
    main()
