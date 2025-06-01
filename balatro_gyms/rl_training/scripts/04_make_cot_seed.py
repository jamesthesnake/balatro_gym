#!/usr/bin/env python3
"""
04_make_cot_seed.py
Generate a few thousand Chain-of-Thought (CoT) demo lines so the
LLM sees the “Thought: … <JSON>” pattern before PPO.

It reads a pickle produced by save_traj.py, throws a trivial heuristic
explanation in front of each Phase-1 KEEP action, and writes JSONL lines
compatible with 01_extract_jsonl / TRLX SFT.
"""

import pickle, json, glob, random, re, argparse
from pathlib import Path
from datetime import datetime
from balatro_game import BalatroGame, Card

# ─── card helpers (same mapping as balatro_tools) ───────────────────────
RANK = {
    Card.Ranks.TWO: "2", Card.Ranks.THREE: "3", Card.Ranks.FOUR: "4",
    Card.Ranks.FIVE: "5", Card.Ranks.SIX: "6",  Card.Ranks.SEVEN: "7",
    Card.Ranks.EIGHT: "8", Card.Ranks.NINE: "9", Card.Ranks.TEN: "T",
    Card.Ranks.JACK: "J", Card.Ranks.QUEEN: "Q", Card.Ranks.KING: "K",
    Card.Ranks.ACE: "A",
}
SUIT = {
    Card.Suits.SPADES: "S",
    Card.Suits.CLUBS: "C",
    Card.Suits.HEARTS: "H",
    Card.Suits.DIAMONDS: "D",
}

def code(card):
    return f"{RANK[card.rank]}{SUIT[card.suit]}"

# ─── quick hand evaluator for a naive explanation -----------------------
def explain_keep(cards):
    suits = [c.suit for c in cards]
    ranks = sorted([c.rank.value for c in cards])
    if len(set(suits)) == 1:
        return "I have a flush already."
    if ranks == [0,1,2,3,12] or all(ranks[i]+1==ranks[i+1] for i in range(4)):
        return "I already have a straight."
    if len(set(ranks)) == 1:
        return "Four of a kind hit."
    if len(set(ranks)) == 2:
        return "Full house / trips."
    return "Highest raw-chip subset."

# ─── main ---------------------------------------------------------------
def make_cot_seed(pkl_glob: str, out_path: str, limit: int = 2000):
    lines = 0
    with open(out_path, "w") as w:
        for pkl in glob.glob(pkl_glob):
            rounds = pickle.load(open(pkl, "rb"))
            for rnd_i, rnd in enumerate(rounds):
                for hand_i, hand in enumerate(rnd):
                    phase1 = hand[1]                       # second step = KEEP
                    keep_idx = phase1["keep_indices"]
                    if keep_idx is None:                   # skip malformed
                        continue
                    kept = [Card(*divmod(int(idx), 13)) for idx in phase1["hand_before"][list(keep_idx)]]
                    thought = explain_keep(kept)
                    obs_cards = " ".join(code(Card(*divmod(int(c),13))) for c in phase1["hand_before"])
                    keep_json = {
                        "name": "keep_cards",
                        "arguments": {"indices": list(keep_idx)}
                    }
                    prompt = f"<ROUND={rnd_i}><HAND={hand_i}><PHASE=1>\n{obs_cards}\n<SEP>"
                    completion = f"Thought: {thought}\n" + json.dumps(keep_json) + " <EOS>"
                    w.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")
                    lines += 1
                    if lines >= limit:
                        print(f"Wrote {lines} CoT lines → {out_path}")
                        return
    print(f"Wrote {lines} CoT lines → {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl_glob", default="pickles/rounds_mix_*.pkl")
    ap.add_argument("--out", default="data/cot_seed.jsonl")
    ap.add_argument("--limit", type=int, default=2000)
    make_cot_seed(**vars(ap.parse_args()))

