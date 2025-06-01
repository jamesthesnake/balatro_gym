#!/usr/bin/env python3
"""
agent_balatro_hier.py
────────────────────────────────────────────────────────────
A hierarchical DeepSeek-8B agent for Balatro.

* High-level “Head A” chooses a round strategy via the tool
  `set_strategy(target, budget)`.
* Low-level “Head B” makes per-hand discard / keep decisions,
  conditioned on a context token
      <STRATEGY={FLUSH|STRAIGHT|HIGHPAIR|ANY}|BUDGET={n}>
  that is appended to every prompt.

The file is drop-in compatible with the rest of the rl_training/ stack.
"""

from __future__ import annotations
import json, re, torch
from typing import List, Dict, Any
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
)

# ── Balatro tool imports ──────────────────────────────────
from balatro_tools import (
    check_cards,
    get_optimal_pairs,
    discard_cards,
    keep_cards,
)
from balatro_strategy import (
    set_strategy,
    get_context,
    consume_discards,
)

# ── Helper schema for JSON tool calls ─────────────────────
class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

TOOLS: List[ToolSchema] = [
    ToolSchema(
        name="check_cards",
        description="Return the current 8-card hand.",
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    ToolSchema(
        name="get_optimal_pairs",
        description="Return best 5-card subsets and raw-chip scores.",
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
        description="Discard specified card positions (0-7) and redraw.",
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
        description="Keep exactly five card positions (0-7) and finish hand.",
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
    ToolSchema(
        name="set_strategy",
        description="Choose round-level target and discard budget.",
        parameters={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": ["ANY", "FLUSH", "STRAIGHT", "HIGHPAIR"],
                },
                "budget": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 4,
                    "default": 4,
                },
            },
            "required": ["target"],
        },
    ),
]

TOOLS_JSON = json.dumps([t.dict() for t in TOOLS], ensure_ascii=False)

# ── System prompt ─────────────────────────────────────────
SYSTEM_MSG = f"""
You are Balatro-Bot, a hierarchical agent.

Formatting rules
────────────────
1. (Optional) Begin with a line:  Thought: ...
2. Follow with ONE pure-JSON tool call.
3. When the round starts, you may first call set_strategy(target,budget).
4. When the game is over, reply normally with no JSON.

Available tools:
{TOOLS_JSON}
"""

# ── Load DeepSeek model & tokenizer ───────────────────────
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

# ── Regex helpers ─────────────────────────────────────────
THOUGHT_RE = re.compile(r"^Thought:.*$", re.MULTILINE)

def build_prompt(history: List[Dict[str, str]]) -> str:
    msgs = [("system", SYSTEM_MSG)] + history
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def extract_json(text: str) -> str | None:
    cleaned = THOUGHT_RE.sub("", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        return None
    return cleaned[start : end + 1]

# ── Main interaction loop ─────────────────────────────────
def run_agent():
    first_hand = check_cards()
    context = get_context()
    history: List[tuple[str, str]] = [
        ("user", f"{context}\nHere is the hand: {first_hand}\nWhat should I do?")
    ]

    while True:
        prompt = build_prompt(history)
        inputs = tok(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=256,
                temperature=0.3,
            )
        new_text = tok.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        tool_json = extract_json(new_text)

        if tool_json:
            call = json.loads(tool_json)
            name, args = call["name"], call.get("arguments", {})
            # ── route calls ───────────────────────────────
            if name == "check_cards":
                result = check_cards()
            elif name == "get_optimal_pairs":
                result = get_optimal_pairs(**args)
            elif name == "discard_cards":
                consume_discards(len(args["indices"]))
                result = discard_cards(**args)
            elif name == "keep_cards":
                result = keep_cards(**args)
            elif name == "set_strategy":
                result = set_strategy(**args)
            else:
                result = f"ERROR: unknown tool {name}"

            # append tool result
            history.append(("assistant", new_text))
            history.append(("tool", json.dumps(result, ensure_ascii=False)))
            # prepend updated context on next user turn
            context = get_context()
            history.append(("user", f"{context}\n(Next hand state?)"))
        else:
            print("\nAssistant:", THOUGHT_RE.sub("", new_text).strip())
            break

# ── Demo entry point ─────────────────────────────────────
if __name__ == "__main__":
    run_agent()

