#!/usr/bin/env python3
"""
01_extract_jsonl.py  –  robust pickle ➜ JSONL converter for Balatro.

• Handles both trajectory schemas:
    phase-0 : cards_discarded | num_discards
    phase-1 : cards_kept      | keep_indices
• Accepts card values as int, np.int*, str, bytes.
• Writes one JSON line / step: {"prompt": ..., "completion": ...}
"""

import argparse, glob, json, os, pickle, re, numpy as np

# ── helpers ─────────────────────────────────────────────────────────────
CARD_RE   = re.compile(r"(\d|T|J|Q|K|A)([♠♣♥♦])")
SUIT_TXT  = {"♠": "S", "♣": "C", "♥": "H", "♦": "D"}
RANKS     = "A23456789TJQK"

def canon_int(cid: int) -> str:
    return RANKS[cid % 13] + "SCDH"[cid // 13]

def canon_utf(card_str: str) -> str:
    m = CARD_RE.match(card_str)
    return m.group(1) + SUIT_TXT[m.group(2)]

def to_code(x) -> str:
    """int | np.integer | str | bytes  -> '7H'"""
    if isinstance(x, (np.integer, int)):
        return canon_int(int(x))
    if isinstance(x, (bytes, bytearray, np.bytes_)):
        try:
            x = x.decode("utf-8")
        except Exception:
            return "UNK"
    if isinstance(x, str):
        try:
            return canon_utf(x)
        except Exception:
            return "UNK"
    return "UNK"

# ── main conversion routine ─────────────────────────────────────────────
pairs = 0
def dump_pickle(path, fh_out):
    global pairs
    rounds = pickle.load(open(path, "rb"))
    for r_i, rnd in enumerate(rounds):
        for h_i, hand in enumerate(rnd):
            for step in hand:
                ph    = step["phase"]
                obs   = " ".join(to_code(c) for c in step["hand_before"])

                if ph == 0:                                   # discard
                    disc = step.get("cards_discarded")
                    if disc is None:
                        disc_cnt = step.get("num_discards", 0)
                        disc     = [] if disc_cnt == 0 else ["UNK"]
                    if disc == ['None'] or disc == []:
                        action = "DISCARD_NONE"
                    else:
                        action = "DISCARD " + " ".join(to_code(c) for c in disc)
                else:                                         # keep
                    kept = step.get("cards_kept")
                    if kept is None:
                        idx  = step["keep_indices"]
                        kept = [to_code(step["hand_before"][i]) for i in idx]
                    action = "KEEP " + " ".join(kept)

                prompt = f"<ROUND={r_i}><HAND={h_i}><PHASE={ph}>\n{obs}\n<SEP>"
                fh_out.write(json.dumps({
                    "prompt": prompt,
                    "completion": action + " <EOS>"
                }) + "\n")
                pairs += 1

# ── CLI ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl_glob", default="pickles/*.pkl")
    ap.add_argument("--out",      default="data/balatro_sft.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fout:
        for pkl in sorted(glob.glob(args.pkl_glob)):
            print("Converting", pkl)
            dump_pickle(pkl, fout)

    print(f"Wrote {pairs:,} pairs → {args.out}")

if __name__ == "__main__":
    main()

