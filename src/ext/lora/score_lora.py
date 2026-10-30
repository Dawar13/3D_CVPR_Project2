"""Scoring and tables for the EXP2 §8.6 LoRA controls.

  tum_clips  --out <csv>                 infer2 clips CSV (key, dir) for the kept TUM RS-VIO windows, GS and RS
  tum_score  --preds <npz> --out <csv>   per-window pose metrics (common.pose_metrics vs gt_poses.npz, keys
                                         {key}_GS / {key}_RS), speed bin from omega_mean (slow < 0.5 <= medium < 1.5
                                         <= fast rad/s)
  summary    --out <dir>                 per-run and mean +- std (over seeds) tables from every finished eval
  curves     --runs a,b,... --out <png>  validation / training curves

Per-run statistics: synthetic test -- median ATE (moving clips, EXP0's 0.20 m path filter as in synth_eval), mean
AUC30, median chamfer, median plan_rms, per input condition (rs, gs); TUM -- median ATE per speed bin per camera.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser("~/rs3d/src"))
TUM = os.path.expanduser("~/rs3d/results/exp2/gap/tum")
TR = os.path.expanduser("~/rs3d/results/exp2/training")
BINS = ["slow", "medium", "fast"]


def tum_clips(out, every=4):
    """Every `every`-th kept window (in windows.csv order: sequence, window id), both cameras. Windows start every 8
    frames and span 46, so neighbours overlap by ~5/6; every 4th keeps ~30 % overlap and cuts GPU time 4x."""
    w = pd.read_csv(f"{TUM}/windows.csv")
    w = w[w.keep].iloc[::every]
    rows = []
    for r in w.itertuples():
        for cam in ("GS", "RS"):
            rows.append(dict(key=f"{r.key}_{cam}",
                             dir=os.path.expanduser(f"~/rs3d/data/windows/seq{r.seq}/{r.window_id:04d}_{cam}")))
    missing = [r["dir"] for r in rows if not os.path.isdir(r["dir"])]
    assert not missing, missing[:3]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(rows)} rows -> {out}")


def tum_score(preds, out):
    from common import SPEED_BINS, pose_metrics
    from metrics2 import pose_auc
    w = pd.read_csv(f"{TUM}/windows.csv")
    w = w[w.keep].copy()
    w["bin"] = pd.cut(w.omega_mean, [0, *SPEED_BINS, np.inf], labels=BINS, right=False).astype(str)
    gt = np.load(f"{TUM}/gt_poses.npz")
    pr = np.load(preds)
    rows = []
    for r in w.itertuples():
        for cam in ("GS", "RS"):
            k = f"{r.key}_{cam}"
            if k not in pr:
                continue
            P, G = pr[k], gt[k]
            rows.append(dict(key=r.key, seq=r.seq, cam=cam, omega_mean=r.omega_mean, bin=r.bin,
                             **pose_metrics(P, G), AUC30=pose_auc(P, G)))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.groupby(["cam", "bin"]).ATE.agg(["median", "count"]).round(4).to_string())
    print("TUM_SCORE_DONE", len(df))


def run_stats(tag):
    """One evaluated run -> dict of per-run statistics (None when not finished)."""
    d = f"{TR}/eval/{tag}"
    s, t = f"{d}/synth_eval.csv", f"{d}/tum_windows.csv"
    if not (os.path.exists(s) and os.path.exists(t)):
        return None
    out = {}
    se = pd.read_csv(s)
    for c in ("rs", "gs"):
        x = se[se.cond == c]
        out[f"syn_{c}_ATE"] = x.ATE.median()
        out[f"syn_{c}_AUC30"] = x.AUC30.mean()
        out[f"syn_{c}_chamfer"] = x.chamfer.median()
        out[f"syn_{c}_plan_rms"] = x.plan_rms.median()
        out[f"syn_{c}_n"] = int(len(x))
        out[f"syn_{c}_ATE_fast"] = x[x.bin == "fast"].ATE.median()
    tw = pd.read_csv(t)
    for cam in ("RS", "GS"):
        for b in BINS + ["all"]:
            x = tw[(tw.cam == cam) & ((tw.bin == b) | (b == "all"))]
            out[f"tum_{cam}_{b}_ATE"] = x.ATE.median()
            out[f"tum_{cam}_{b}_n"] = int(len(x))
    return out


def summary(outdir):
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for d in sorted(glob.glob(f"{TR}/eval/*/eval_config.json")):
        cfg = json.load(open(d))
        st = run_stats(cfg["tag"])
        if st is None:
            continue
        tag = cfg["tag"]
        arm = "base" if cfg["ckpt"] == "base" else tag.split("_")[1]
        seed = np.nan if arm == "base" else int(tag.split("_s")[-1])
        info = {}
        rj = f"{TR}/{tag}/run.json"
        if os.path.exists(rj):
            j = json.load(open(rj))
            info = dict(best_step=j["best_step"], last_step=j["last_step"], stop_reason=j["stop_reason"])
        rows.append(dict(tag=tag, model=cfg["model"], arm=arm, seed=seed, **info, **st))
    df = pd.DataFrame(rows)
    df.to_csv(f"{outdir}/per_run.csv", index=False)
    num = [c for c in df.columns if c.startswith(("syn_", "tum_"))]
    g = df.groupby(["model", "arm"])[num]
    agg = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std")).join(
        df.groupby(["model", "arm"]).size().rename("n_seeds"))
    agg.to_csv(f"{outdir}/mean_std.csv")

    def fmt(m, s, nd=4):
        return f"{m:.{nd}f}" if not np.isfinite(s) else f"{m:.{nd}f} ± {s:.{nd}f}"
    lines = ["| model | arm | seeds | " + " | ".join(f"syn {c} {k}" for c in ("rs", "gs")
                                                    for k in ("ATE", "AUC30", "chamfer", "plan_rms")) + " |",
             "|---" * 11 + "|"]
    for (m, a), r in agg.iterrows():
        cells = [fmt(r[f"syn_{c}_{k}_mean"], r[f"syn_{c}_{k}_std"]) for c in ("rs", "gs")
                 for k in ("ATE", "AUC30", "chamfer", "plan_rms")]
        lines.append(f"| {m} | {a} | {int(r.n_seeds)} | " + " | ".join(cells) + " |")
    lines += ["", "| model | arm | seeds | " + " | ".join(f"TUM {cam} {b} ATE (n)" for cam in ("RS", "GS")
                                                         for b in BINS) + " |", "|---" * 9 + "|"]
    for (m, a), r in agg.iterrows():
        cells = [f"{fmt(r[f'tum_{cam}_{b}_ATE_mean'], r[f'tum_{cam}_{b}_ATE_std'])} ({int(r[f'tum_{cam}_{b}_n_mean'])})"
                 for cam in ("RS", "GS") for b in BINS]
        lines.append(f"| {m} | {a} | {int(r.n_seeds)} | " + " | ".join(cells) + " |")
    open(f"{outdir}/tables.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def curves(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    for run in runs:
        v = pd.read_csv(f"{TR}/{run}/val_curve.csv")
        t = pd.read_csv(f"{TR}/{run}/train_log.csv")
        t = t[t.skipped.isna()]
        sm = t.loss.rolling(25, min_periods=1).median()
        ax[0, 0].plot(t.step, sm, label=run)
        ax[0, 1].plot(t.step, t.grad_norm.rolling(25, min_periods=1).median(), label=run)
        for c, a in (("rs", ax[1, 0]), ("gs", ax[1, 1])):
            a.plot(v.step, v[f"{c}_ATE_mean"], "o-", label=run)
        ax[1, 2].plot(v.step, v["rs_absrel"], "o-", label=f"{run} rs")
        ax[0, 2].plot(v.step, v["rs_AUC30"], "o-", label=f"{run} rs")
    for a, ttl in zip(ax.flat, ["train loss (rolling median 25)", "grad norm (pre-clip, rolling median)",
                                "val AUC30 (rs)", "val mean ATE, rs input (m)", "val mean ATE, gs input (m)",
                                "val depth AbsRel (rs)"]):
        a.set_title(ttl)
        a.set_xlabel("step")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tum_clips", "tum_score", "summary", "curves"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--preds")
    ap.add_argument("--runs")
    ap.add_argument("--every", type=int, default=4)
    a = ap.parse_args()
    if a.mode == "tum_clips":
        tum_clips(a.out, a.every)
    elif a.mode == "tum_score":
        tum_score(a.preds, a.out)
    elif a.mode == "summary":
        summary(a.out)
    else:
        curves(a.runs.split(","), a.out)


if __name__ == "__main__":
    main()
