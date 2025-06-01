#!/usr/bin/env python3
"""
TRLX PPO fine-tune for Balatro-Bot.
Assumes:
* Chat history includes Thought lines + JSON keep_cards result.
* Reward = sqrt(raw_score)/100 − 0.01.
"""
from __future__ import annotations
import math, re, json, trlx
from trlx.data.configs import TRLConfig

THOUGHT_RE = re.compile(r"^Thought:.*$", re.MULTILINE)

# ---- TRLX configuration (edit as needed) ------------------------------
cfg = TRLConfig.load_yaml("cfg/trlx_ppo.yml")

# ---- reward function ---------------------------------------------------
def reward_fn(prompts, completions):
    rewards = []
    for comp in completions:
        try:
            js = json.loads(THOUGHT_RE.sub("", comp).strip())
            raw = js.get("raw_score", 0)
        except Exception:
            raw = 0
        rewards.append(math.sqrt(raw) / 100.0 - 0.01)
    return rewards

# ---- dummy prompts (env will feed real data via tool calls) -----------
dummy_prompts = ["<START>"] * cfg.train.total_steps

if __name__ == "__main__":
    trlx.train(
        reward_fn=reward_fn,
        prompts=dummy_prompts,
        config=cfg,
    )

