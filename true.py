from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-MoE-16B-Base")
model.save_pretrained("/workspace/deepseek-model")
