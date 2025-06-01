#!/usr/bin/env python3
"""
Compare two trajectory pickles and write:
  • eval_report.csv  (summary table)
  • score_hist_*.png (histograms)
"""

import pickle, pandas as pd, matplotlib.pyplot as plt
import argparse, os, numpy as np

def flatten(pkl):
    rows = []
    rounds = pickle.load(open(pkl,"rb"))
    for r in rounds:
        pass_flag = r[-1][1].get("round_pass", False)
        round_score = r[-1][1]["round_score_so_far"]
        for hand in r:
            ph1 = hand[1]                       # phase-1 step
            rows.append({
                "hand_score" : ph1["balatro_raw_score"],
                "round_score": round_score,
                "round_pass" : pass_flag,
            })
    return pd.DataFrame(rows)

def summarize(df):
    return pd.Series({
        "blind_pass_rate"     : df["round_pass"].mean(),
        "avg_round_score"     : df.groupby(level=0)["round_score"].max().mean(),
        "avg_hand_score"      : df["hand_score"].mean(),
    })

def main(baseline_pkl, ppo_pkl, outdir="eval_out"):
    os.makedirs(outdir, exist_ok=True)
    base_df = flatten(baseline_pkl)
    ppo_df  = flatten(ppo_pkl)

    report = pd.concat(
        {"baseline": summarize(base_df), "ppo": summarize(ppo_df)},
        axis=1
    )
    report.to_csv(os.path.join(outdir, "eval_report.csv"))
    print(report)

    # histogram per-hand raw chips
    for name, df in [("baseline", base_df), ("ppo", ppo_df)]:
        plt.figure()
        plt.hist(df["hand_score"], bins=50, density=True, alpha=0.7)
        plt.title(f"Hand score distribution – {name}")
        plt.xlabel("raw chip value")
        plt.ylabel("density")
        plt.savefig(os.path.join(outdir, f"score_hist_{name}.png"))
        plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="baseline pickle path")
    ap.add_argument("--ppo",      required=True, help="PPO pickle path")
    main(**vars(ap.parse_args()))

