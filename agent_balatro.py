#!/usr/bin/env python3
"""
LLM ReAct agent for Balatro using DeepSeek-R1-Distill-Llama-8B.
* Supports private Chain-of-Thought (“Thought:” lines).
* Emits JSON tool calls that are routed to balatro_tools.py functions.
"""
from __future__ import annotations
import json, torch, re
from typing import List, Dict, Any
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# ── Balatro tools -------------------------------------------------------
from balatro_tools import (
    check_cards,
    get_optimal_pairs,
    discard_cards,
    keep_cards,          # new – finalise and score a hand
)

# ── Tool schema for the system prompt ----------------------------------
class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

TOOLS: List[ToolSchema] = [
    ToolSchema(
        name="check_cards",
        description="Return the current 8-card hand as codes like '8S'.",
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    ToolSchema(
        name="get_optimal_pairs",
        description="Return best 5-card subsets and their raw-chip scores.",
        parameters={
            "type": "object",
            "properties": {
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10}
            },
            "required": [],
        },
    ),
    ToolSchema(
        name="discard_cards",
        description="Discard the specified card positions (0-7) and redraw.",
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
    ToolSchema(
        name="keep_cards",
        description="Lock exactly five cards (positions 0-7) and finish hand.",
        parameters={
            "type": "object",
            "properties": {
                "indices": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0, "maximum": 7},
                    "minItems": 5,
                    "maxItems": 5,
                }
            },
            "required": ["indices"],
        },
    ),
]

TOOLS_JSON = json.dumps([t.dict() for t in TOOLS], ensure_ascii=False)

# ── System prompt with CoT instructions --------------------------------
SYSTEM_MSG = f"""
You are Balatro-Bot.

**Formatting rules**

1. (Optional) Start with a single line beginning with `Thought:` to reason.
2. Follow immediately with ONE pure-JSON tool call, e.g.:

   Thought: I need one more heart for a flush.
   {{"name":"discard_cards","arguments":{{"indices":[1,3]}}}}

3. When you are done playing, reply normally with no JSON.
Available tools:
{TOOLS_JSON}
"""

# ── Load model & tokenizer ---------------------------------------------
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True,
)
streamer = TextStreamer(tok)

# ── Utility -------------------------------------------------------------
THOUGHT_RE = re.compile(r"^Thought:.*$", re.MULTILINE)

def build_prompt(history: List[Dict[str, str]]) -> str:
    msgs = [("system", SYSTEM_MSG)] + history
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def extract_json_block(text: str) -> str | None:
    cleaned = THOUGHT_RE.sub("", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        return None
    return cleaned[start : end + 1]

# ── Main interaction loop ----------------------------------------------
def run_agent(initial_hand: str):
    history: List[tuple[str, str]] = [
        ("user", f"Here is the hand: {initial_hand}. What should I do?")
    ]
    for _ in range(6):                               # max steps per dialog
        prompt = build_prompt(history)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=256,
                temperature=0.3,
            )
        new_text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        tool_json = extract_json_block(new_text)
        if tool_json:
            call = json.loads(tool_json)
            name, args = call["name"], call.get("arguments", {})
            # route ------------------------------------------------------
            if name == "check_cards":
                result = check_cards()
            elif name == "get_optimal_pairs":
                result = get_optimal_pairs(**args)
            elif name == "discard_cards":
                result = discard_cards(**args)
            elif name == "keep_cards":
                result = keep_cards(**args)
            else:
                result = f"ERROR: unknown tool {name}"
            # update chat -----------------------------------------------
            history.append(("assistant", new_text))
            history.append(("tool", json.dumps(result, ensure_ascii=False)))
        else:
            print("\nAssistant:", THOUGHT_RE.sub("", new_text).strip())
            break

# ── Demo ---------------------------------------------------------------
if __name__ == "__main__":
    print("=== Demo run ===")
    run_agent(check_cards())

