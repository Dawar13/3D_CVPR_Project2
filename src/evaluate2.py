"""EXP2 per-window pose evaluation of any TUM condition, plus the reproduction check against EXP0.

Conditions are window folders `{wid}_{cond}`. GS is scored against GS GT; everything else (RS, V3, V4, SE2P_*, ...)
is a version of the RS frames and is scored against the RS centre-row GT, as in EXP0/EXP1.

usage: RS3D_RES=~/rs3d/results/exp2/gap/tum python evaluate2.py score --models pi3,vggt --conds GS,RS,V3
       python evaluate2.py repro --new ~/rs3d/results/exp2/gap/tum --old ~/rs3d/results/exp0
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import N_FRAMES, RES, SPEED_BINS, STRIDE, pose_metrics
from metrics2 import pose_auc
from stats2 import boot_median_ci

FPS = 20.0
BINS = ["slow", "medium", "fast"]


def gt_key(key, cond):
    return f"{key}_GS" if cond == "GS" else f"{key}_RS"


def score(models, conds, res=RES):
    win = pd.read_csv(f"{res}/windows.csv")
    win = win[win.keep].copy()
    win["v_mean"] = win.gt_path_m / ((N_FRAMES - 1) * STRIDE / FPS)
    win["bin"] = pd.cut(win.omega_mean, [0, *SPEED_BINS, np.inf], labels=BINS, right=False).astype(str)
    gt = np.load(f"{res}/gt_poses.npz")
    rows = []
    for m in models:
        for c in conds:
            tag = "" if c in ("GS", "RS") else f"_{c}"
            p = f"{res}/preds/{m}{tag}.npz"
            if not os.path.exists(p):
                print("missing", p, flush=True)
                continue
            pr = np.load(p)
            for r in win.itertuples():
                k = f"{r.key}_{c}"
                if k not in pr:
                    continue
                g = gt[gt_key(r.key, c)]
                met = pose_metrics(pr[k], g)
                rows.append(dict(seq=r.seq, window_id=r.window_id, key=r.key, model=m, cond=c, omega_mean=r.omega_mean,
                                 v_mean=r.v_mean, rs_shift_px=r.rs_shift_px, bin=r.bin, **met,
                                 AUC30=pose_auc(pr[k], g)))
    return pd.DataFrame(rows)


def repro(new, old):
    """Reproduction check (EXP2 §2.5): new medians must sit inside EXP0's bootstrap CI, per model/camera/bin."""
    a = pd.read_csv(f"{old}/per_window.csv")
    b = pd.read_csv(f"{new}/per_window_repro.csv").rename(columns={"cond": "camera"})
    out = []
    for m in sorted(set(b.model)):
        for cam in ("GS", "RS"):
            for bn in BINS + ["all"]:
                A = a[(a.model == m) & (a.camera == cam) & ((a.bin == bn) | (bn == "all"))]
                B = b[(b.model == m) & (b.camera == cam) & ((b.bin == bn) | (bn == "all"))]
                if len(A) == 0 or len(B) == 0:
                    continue
                ma, la, ha = boot_median_ci(A.ATE)
                mb, lb, hb = boot_median_ci(B.ATE)
                j = A.merge(B, on="key", suffixes=("_old", "_new"))
                out.append(dict(model=m, camera=cam, bin=bn, n_old=len(A), n_new=len(B), median_old=ma, ci_old_lo=la,
                                ci_old_hi=ha, median_new=mb, ci_new_lo=lb, ci_new_hi=hb,
                                new_in_old_ci=bool(la <= mb <= ha), old_in_new_ci=bool(lb <= ma <= hb),
                                per_window_spearman=float(j.ATE_old.corr(j.ATE_new, method="spearman")),
                                per_window_median_absdiff=float((j.ATE_old - j.ATE_new).abs().median())))
    df = pd.DataFrame(out)
    df.to_csv(f"{new}/repro_check.csv", index=False)
    ok = bool(df.new_in_old_ci.all() and df.old_in_new_ci.all())
    print(df.round(4).to_string(), flush=True)
    print("REPRO_PASS" if ok else "REPRO_FAIL", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["score", "repro"])
    ap.add_argument("--models", default="pi3,vggt")
    ap.add_argument("--conds", default="GS,RS")
    ap.add_argument("--out", default="")
    ap.add_argument("--new", default=RES)
    ap.add_argument("--old", default=os.path.expanduser("~/rs3d/results/exp0"))
    a = ap.parse_args()
    if a.what == "score":
        df = score(a.models.split(","), a.conds.split(","))
        out = a.out or f"{RES}/per_window_{'_'.join(a.conds.split(','))}.csv"
        df.to_csv(out, index=False)
        print(df.groupby(["model", "cond", "bin"]).ATE.agg(["count", "median"]).round(4).to_string(), flush=True)
    else:
        repro(a.new, a.old)


if __name__ == "__main__":
    main()
