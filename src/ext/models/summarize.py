"""Summaries for the extra models (read-only use of ~/rs3d/src).

main:    reads per_window_newmodels.csv written by `evaluate2.py score` -> median ATE per model / cond / bin with n,
         plus the paired RS/GS ratio of medians (windows with both GS and RS).
control: GS-vs-GS control (calibration sequence) scored exactly like EXP0's evaluate.control_summary
         (evaluate.compute_per_window("control"): tercile bins when a fixed bin has < 15 windows), but for our
         models and written to our own files (EXP0's control_summary.csv is not touched).

usage: RS3D_RES=~/rs3d/results/exp2/gap/tum python summarize.py main --csv <per_window_newmodels.csv>
       RS3D_RES=~/rs3d/results/exp2/gap/tum python summarize.py control --models da3,streamvggt --out <dir>
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser("~/rs3d/src"))


def main_summary(csv):
    """median ATE per model / cond / bin with n; RS/GS ratio of medians over windows that have both conds."""
    df = pd.read_csv(csv)
    rows = []
    for m in df.model.unique():
        d = df[df.model == m]
        P = d.pivot_table(index="key", columns="cond", values="ATE")
        P = P.join(d.drop_duplicates("key").set_index("key").bin)
        conds = [c for c in ("GS", "RS") if c in P]
        for bn in ["slow", "medium", "fast", "all"]:
            Q = P if bn == "all" else P[P.bin == bn]
            row = dict(model=m, bin=bn)
            for c in conds:
                row[f"n_{c}"] = int(Q[c].notna().sum())
                row[f"{c}_median_ATE_m"] = Q[c].median()
            if len(conds) == 2:
                R = Q.dropna(subset=conds)
                row.update(n_paired=len(R), RS_over_GS=R.RS.median() / R.GS.median())
            rows.append(row)
    out = pd.DataFrame(rows)
    print(out.round(5).to_string(index=False))
    out.to_csv(csv.replace(".csv", "_summary.csv"), index=False)


def control(models, out):
    import evaluate  # EXP0 analysis module; only its per-window function and ratio CI are used
    evaluate.MODELS = models
    dfc, rule, cuts = evaluate.compute_per_window("control")
    dfc.to_csv(f"{out}/control_per_window_newmodels.csv", index=False)
    rng = np.random.default_rng(2)
    rows = []
    for m in dfc.model.unique():
        P = evaluate.paired(dfc, m, "ATE")
        for b in ["slow", "medium", "fast", "all"]:
            Q = P if b == "all" else P[P.bin == b]
            lo, hi = evaluate.ratio_ci(Q.GS.values, Q.RS.values, rng)
            rows.append(dict(model=m, metric="ATE", bin=b, n=len(Q), median_cam0=Q.GS.median(),
                             median_cam1=Q.RS.median(), ratio=Q.RS.median() / Q.GS.median(), ratio_ci_lo=lo,
                             ratio_ci_hi=hi))
    cs = pd.DataFrame(rows)
    cs.to_csv(f"{out}/control_summary_newmodels.csv", index=False)
    print("control bin rule:", rule)
    print(cs.round(5).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["main", "control"])
    ap.add_argument("--csv", default="")
    ap.add_argument("--models", default="da3,streamvggt")
    ap.add_argument("--out", default=os.environ.get("RS3D_RES", ""))
    a = ap.parse_args()
    if a.what == "main":
        main_summary(a.csv)
    else:
        control(a.models.split(","), a.out)
