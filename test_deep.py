#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import model_info
import time, os

# Verify critical dependencies are installed
try:
    import accelerate
except ImportError:
    raise ImportError("Please install accelerate: pip install accelerate")

MODEL_ID = "deepseek-ai/deepseek-moe-16b-base"

def verify_model_access():
    print("\n🔍 Testing DeepSeek-MoE-16B-Base access...")
    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ Set HF_TOKEN first: export HF_TOKEN='your_token_here'")
        return False
    
    try:
        info = model_info(MODEL_ID, token=token)
        print("✅ Repository access confirmed")
        return True
    except Exception as e:
        print(f"❌ Access check failed: {e}")
        print(f"\nVisit and accept the license: https://huggingface.co/{MODEL_ID}")
        return False

def load_model():
    print("\n⬇️ Loading DeepSeek-MoE-16B-Base (Optimized for 2x A100)...")
    
    # 4-bit config for A100 efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    try:
        # First load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True
        )
        
        # Then load model with device mapping
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        print(f"✅ Model loaded across devices: {model.hf_device_map}")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Load failed: {str(e)[:500]}")  # Truncate long errors
        return None, None

if __name__ == "__main__":
    # Verify and install missing packages
    try:
        import flash_attn
    except ImportError:
        print("⚠️ Installing flash-attn for better performance...")
        os.system("pip install flash-attn --no-build-isolation")
    
    if verify_model_access():
        model, tokenizer = load_model()
        if model:
            print("\n🎉 Success! Model is ready for inference.")
            print(f"Device map: {model.hf_device_map}")
