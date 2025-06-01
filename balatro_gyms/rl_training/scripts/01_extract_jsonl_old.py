import pickle, json, re, glob, os, argparse

CARD_RE = re.compile(r"([23456789TJQKA])([♠♣♥♦])")
SUIT_MAP = {"♠": "S", "♣": "C", "♥": "H", "♦": "D"}

def canon(card: str) -> str:
    rank, suit = CARD_RE.match(card).groups()
    return f"{rank}{SUIT_MAP[suit]}"

def main(pkl_glob: str, out: str):
    pairs = 0
    with open(out, "w") as w:
        for pkl in glob.glob(pkl_glob):
            rounds = pickle.load(open(pkl, "rb"))
            for r_i, rnd in enumerate(rounds):
                for h_i, hand in enumerate(rnd["hands"]):
                    for step in hand["steps"]:
                        ph = step["phase"]
                        obs = " ".join(map(canon, step["hand_before"]))
                        if ph == 0:                   # discard / no-op
                            disc = step["cards_discarded"]
                            action = ("DISCARD_NONE" if disc == ['None']
                                      else "DISCARD " +
                                      " ".join(map(canon, disc)))
                        else:                         # keep phase
                            action = "KEEP " + " ".join(
                                map(canon, step["cards_kept"]))
                        prompt = (f"<ROUND={r_i}><HAND={h_i}><PHASE={ph}>\n"
                                  f"{obs}\n<SEP>")
                        completion = action + " <EOS>"
                        w.write(json.dumps({"prompt": prompt,
                                            "completion": completion}) + "\n")
                        pairs += 1
    print(f"Wrote {pairs:,} pairs → {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl_glob", default="data/*.pkl")
    ap.add_argument("--out", default="data/balatro_sft.jsonl")
    main(**vars(ap.parse_args()))
