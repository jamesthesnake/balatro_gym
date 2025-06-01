#!/usr/bin/env python3
"""
eval_confusion.py — baseline-vs-PPO “keep” decision confusion matrix.

Hand categories follow BalatroGame._evaluate_hand() heuristic:
  HIGH, PAIR, 2PAIR, TRIPS, STRAIGHT, FLUSH, FULL-HOUSE,
  QUADS, STRAIGHT-FLUSH, FIVE-KIND / FLUSH-FIVE (rare).

The script requires that both pickles were generated with identical seeds
so each Round/Hand index refers to the same original 8-card deal.
"""

import argparse, os, pickle, itertools, numpy as np
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from balatro_game import BalatroGame, Card   # uses your local implementation

CAT = ["HIGH","PAIR","2PAIR","TRIPS","STRAIGHT","FLUSH",
       "FULL","QUADS","ST_FL","FIVE"]
def classify(cards):
    """Return category string for a 5-card list of Card objects."""
    chips = BalatroGame._evaluate_hand(cards)
    # thresholds copied from your scoring table
    if chips >= 160:  return "FIVE"          # flush 5 or royal
    if chips >= 120:  return "ST_FL"
    if chips >= 100:  return "QUADS"
    if chips >=  60:  return "FULL"
    if chips >=  40:  return "FLUSH"
    if chips >=  35:  return "STRAIGHT"
    if chips >=  30:  return "TRIPS"
    if chips >=  20:  return "2PAIR"
    if chips >=  10:  return "PAIR"
    return "HIGH"

def recode_hand(step):
    """Return category string for one Phase-1 transition dict."""
    kept = step.get("cards_kept")
    if kept is None:                              # new schema
        idx = step["keep_indices"]
        kept = [ Card(*divmod(int(step["hand_before"][i]), 13))
                 for i in idx ]
    else:                                         # older str list
        # convert '7♥' → Card obj
        cvt = lambda s: Card(
            BalatroGame.Card.Ranks("A23456789TJQK".index(s[0])),
            BalatroGame.Card.Suits("SCDH".index({"♠":"S","♣":"C","♥":"H","♦":"D"}[s[1]]))
        )
        kept = [cvt(c) for c in kept]
    return classify(kept)

def table_from_pickle(path):
    tbl = {}
    rounds = pickle.load(open(path,"rb"))
    for r_idx, rnd in enumerate(rounds):
        for h_idx, hand in enumerate(rnd):
            cat = recode_hand(hand[1])
            tbl[(r_idx,h_idx)] = cat
    return tbl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--ppo", required=True)
    ap.add_argument("--outdir", default="eval_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    base = table_from_pickle(args.baseline)
    ppo  = table_from_pickle(args.ppo)

    # Align by (round,hand) key
    common_keys = set(base) & set(ppo)
    data = [(base[k], ppo[k]) for k in common_keys]
    df = pd.DataFrame(data, columns=["baseline","ppo"])

    # Confusion counts
    conf = (df.groupby(["ppo","baseline"])
              .size().unstack(fill_value=0).reindex(index=CAT, columns=CAT))
    conf.to_csv(os.path.join(args.outdir,"confusion.csv"))

    # Heat-map plot
    plt.figure(figsize=(8,7))
    sns.heatmap(conf, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Baseline kept category")
    plt.ylabel("PPO kept category")
    plt.title("Confusion matrix of kept-hand categories")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir,"confusion_heatmap.png"))
    print("Saved confusion.csv and confusion_heatmap.png →", args.outdir)

if __name__ == "__main__":
    main()

