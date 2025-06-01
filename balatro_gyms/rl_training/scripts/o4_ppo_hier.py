#!/usr/bin/env python3
"""
Head-A (round): action = JSON set_strategy → reward = pass/fail
Head-B (hand) : action = discard/keep      → reward = sqrt(raw_chips)
"""
from trlx import train
from trlx.data.configs import TRLConfig
from balatro_tools import discard_cards, keep_cards, check_cards
from balatro_strategy import set_strategy

cfg = TRLConfig.load_yaml("rl_training/cfg/trlx_ppo.yml")
cfg.train.total_steps = 8000

def reward_fn(prompts, completions):
    r = []
    for txt in completions:
        # simple heuristic: +1 if round_pass, else hand-level score
        if '"round_pass": true' in txt.lower():
            r.append(1.0)
        else:
            # extract raw_score
            import json, re
            try:
                js = json.loads(re.search(r"{.*}", txt).group(0))
                r.append(js.get("raw_score", 0)**0.5 / 100)
            except Exception:
                r.append(0.0)
    return r

train(
    reward_fn = reward_fn,
    prompts   = ["<NEW ROUND>"] * cfg.train.total_steps,
    config    = cfg,
)

