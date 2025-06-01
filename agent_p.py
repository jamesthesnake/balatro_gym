# agent_balatro.py
# -----------------------------------------------------------
# 1. pip install transformers accelerate bitsandbytes pydantic
# 2. make sure balatro_gym or your own wrappers expose:
#       check_cards(hand)          -> str (analysis of current 8-card hand)
#       get_optimal_pairs(hand)    -> list[list[int]]  (candidate 5-card keeps)
#       discard_cards(indices)     -> str (env.step discarding those slots)
# 3. run:  python agent_balatro.py
# -----------------------------------------------------------

import os, json, torch
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# ─────────── Balatro tool stubs (replace with real calls) ──────────────
def check_cards(hand: List[str]) -> str:
    return f"You are holding: {' '.join(hand)}"

def get_optimal_pairs(hand: List[str]) -> List[List[int]]:
    # dummy: keep high ranks
    return [sorted(range(8))[-5:]]

def discard_cards(indices: List[int]) -> str:
    return f"Discarded slots {indices}"

# ─────────── Function-call schema for the LLM tools ────────────────────
class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

TOOLS = [
    ToolSchema(
        name="check_cards",
        description="Summarise the current hand in natural language.",
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    ToolSchema(
        name="get_optimal_pairs",
        description="Return best 5-card subsets (indices 0-7) ranked by raw chip value.",
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    ToolSchema(
        name="discard_cards",
        description="Discard the given list of card positions (0-based).",
        parameters={
            "type": "object",
            "properties": {
                "indices": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0, "maximum": 7},
                }
            },
            "required": ["indices"],
        },
    ),
]

TOOLS_JSON = json.dumps([t.dict() for t in TOOLS], ensure_ascii=False)

# ─────────── Load DeepSeek 8-B (4-bit QLoRA for a 24 GB GPU) ────────────
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="left")
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)
streamer = TextStreamer(tok)

# ─────────── Helper to wrap prompt+chat history ────────────────────────
SYSTEM_MSG = (
    "You are Balatro-Bot. You may call these tools via JSON:\n"
    f"{TOOLS_JSON}\n"
    "When you decide on a tool call, output ONLY valid JSON: "
    '{"name": "...", "arguments": {...}}. Otherwise, reply normally.\n'
)

def build_prompt(history: List[Dict[str, str]]) -> str:
    msgs = [("system", SYSTEM_MSG)] + history
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

# ─────────── Main REPL loop ───────────────────────────────────────────
def run_agent(hand: List[str]):
    history = [
        {"role": "user", "content": f"Here is the hand: {' '.join(hand)}. What should I do?"}
    ]
    
    for _ in range(4):  # max depth
        prompt = build_prompt(history)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            out = model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=256,
                temperature=0.2,
            )
            
        new_text = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        
        # tool call or plain answer?
        try:
            tool_call = json.loads(new_text)
            name = tool_call["name"]
            args = tool_call.get("arguments", {})
            
            if name == "check_cards":
                result = check_cards(hand)
            elif name == "get_optimal_pairs":
                result = get_optimal_pairs(hand)
            elif name == "discard_cards":
                result = discard_cards(args["indices"])
            else:
                result = f"Unknown tool {name}"
                
            history.append({"role": "assistant", "content": new_text})
            history.append({"role": "tool", "content": str(result)})
            
        except json.JSONDecodeError:
            print("LLM final answer:", new_text)
            break
        except KeyError:
            print("Invalid tool format:", new_text)
            break

# ─────────── demo run ──────────────────────────────────────────────────
if __name__ == "__main__":
    demo_hand = ["8♠", "3♣", "9♥", "J♥", "K♠", "7♣", "5♦", "4♣"]
    run_agent(demo_hand)
