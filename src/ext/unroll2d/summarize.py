"""Score pi3 / vggt on every U2D condition and summarise vs plain RS and GS.

Uses src/evaluate2.score (non-GS conditions vs the centre-row RS GT `{key}_RS`) and additionally scores every
non-GS condition against the first-row GT `{key}_RSfirst` with common.pose_metrics (needed for first-row methods;
reported for all as a check of the time convention). Only windows present in every compared condition are used
for the summary (paired). Run with the main env and RS3D_RES=~/rs3d/results/exp2/gap/tum.

Writes ~/rs3d/results/exp2/baselines/unroll2d/
  per_window_u2d.csv      one row per (model, cond, window): evaluate2 metrics + ATE_RSfirst etc.
  summary_u2d.csv         median ATE (+ bootstrap 95% CI) per model x cond x bin with n, ratio to RS and GS medians,
                          fraction of windows where the condition beats plain RS (paired)
  summary_u2d.md          the same as a readable table
usage: python summarize.py [--conds U2D-a,U2D-b]   (default: every preds/pi3_U2D-*.npz)
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser("~/rs3d/src"))
from common import RES, pose_metrics  # noqa: E402
from evaluate2 import score  # noqa: E402
from stats2 import boot_median_ci  # noqa: E402

OUT = os.path.expanduser("~/rs3d/results/exp2/baselines/unroll2d")
BINS = ["slow", "medium", "fast", "all"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="")
    ap.add_argument("--models", default="pi3,vggt")
    ap.add_argument("--tag", default="u2d")
    a = ap.parse_args()
    models = a.models.split(",")
    if a.conds:
        conds = a.conds.split(",")
    else:
        conds = sorted({os.path.basename(p)[len("pi3_"):-4] for p in glob.glob(f"{RES}/preds/pi3_U2D-*.npz")} |
                       {os.path.basename(p)[len("vggt_"):-4] for p in glob.glob(f"{RES}/preds/vggt_U2D-*.npz")})
    df = score(models, ["GS", "RS"] + conds)
    gt = np.load(f"{RES}/gt_poses.npz")
    first = []
    for m in models:
        for c in ["RS"] + conds:
            p = f"{RES}/preds/{m}{'' if c == 'RS' else '_' + c}.npz"
            if not os.path.exists(p):
                continue
            pr = np.load(p)
            for k in df[(df.model == m) & (df.cond == c)].key:
                met = pose_metrics(pr[f"{k}_{c}"], gt[f"{k}_RSfirst"])
                first.append(dict(model=m, cond=c, key=k, **{f"{n}_RSfirst": v for n, v in met.items()}))
    df = df.merge(pd.DataFrame(first), on=["model", "cond", "key"], how="left")
    df.to_csv(f"{OUT}/per_window_{a.tag}.csv", index=False)

    rows = []
    for m in models:
        d = df[df.model == m]
        base = d[d.cond == "RS"].set_index("key")
        gs = d[d.cond == "GS"].set_index("key")
        for c in ["GS", "RS"] + conds:
            dc = d[d.cond == c].set_index("key")
            if len(dc) == 0:
                continue
            keys = dc.index.intersection(base.index).intersection(gs.index)
            for b in BINS:
                kb = keys if b == "all" else keys[dc.loc[keys, "bin"] == b]
                if len(kb) == 0:
                    continue
                x = dc.loc[kb, "ATE"].values
                med, lo, hi = boot_median_ci(x)
                rs_med = float(np.median(base.loc[kb, "ATE"]))
                gs_med = float(np.median(gs.loc[kb, "ATE"]))
                rows.append(dict(model=m, cond=c, bin=b, n=len(kb), ATE_median=med, ci_lo=lo, ci_hi=hi,
                                 RS_median=rs_med, GS_median=gs_med, ratio_vs_RS=med / rs_med, ratio_vs_GS=med / gs_med,
                                 gap_closed=(rs_med - med) / (rs_med - gs_med) if rs_med != gs_med else np.nan,
                                 frac_better_than_RS=float((x < base.loc[kb, "ATE"].values).mean()),
                                 ATE_RSfirst_median=float(np.median(dc.loc[kb, "ATE_RSfirst"]))
                                 if c != "GS" else np.nan))
    s = pd.DataFrame(rows)
    s.to_csv(f"{OUT}/summary_{a.tag}.csv", index=False)
    with open(f"{OUT}/summary_{a.tag}.md", "w") as f:
        f.write("Median ATE (Sim3-aligned, m) per speed bin; n windows; paired over windows present in GS, RS and "
                "the condition. ATE vs centre-row RS GT (GS vs GS GT); last column: vs first-row RS GT.\n\n")
        for m in models:
            f.write(f"### {m}\n\n| cond | " + " | ".join(f"{b} (n)" for b in BINS) +
                    " | fast ratio vs RS | fast ratio vs GS | fast gap closed | all frac better than RS | "
                    "all ATE vs RSfirst |\n|---|" + "---|" * (len(BINS) + 5) + "\n")
            for c in ["GS", "RS"] + conds:
                sc = s[(s.model == m) & (s.cond == c)].set_index("bin")
                if len(sc) == 0:
                    continue
                cells = [f"{sc.loc[b, 'ATE_median']:.4f} ({int(sc.loc[b, 'n'])})" if b in sc.index else "-"
                         for b in BINS]
                fr = sc.loc["fast"] if "fast" in sc.index else None
                al = sc.loc["all"]
                f.write(f"| {c} | " + " | ".join(cells) +
                        (f" | {fr.ratio_vs_RS:.2f} | {fr.ratio_vs_GS:.2f} | {fr.gap_closed:+.2f}" if fr is not None
                         else " | - | - | -") +
                        f" | {al.frac_better_than_RS:.2f} | {al.ATE_RSfirst_median:.4f} |\n")
            f.write("\n")
    print(open(f"{OUT}/summary_{a.tag}.md").read(), flush=True)


if __name__ == "__main__":
    main()
