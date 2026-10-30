"""Assemble COLMAP / GLOMAP window results into prediction npz files, per-window metric CSVs and a summary against
pi3 / VGGT on the same windows.

Registration policy (applied identically to both mappers and all windows):
  * the largest model (most registered images) is the window's reconstruction;
  * status 'full'    = all 16 frames registered;
    status 'partial' = 3..15 frames registered -> metrics on the registered frames only, window flagged;
    status 'fail'    = < 3 frames registered (no Sim(3) possible), no model, or a crash -> no metrics.
  * summaries report (A) all non-failed windows, n given, and (B) the 'common-full' subset: windows where BOTH
    COLMAP and GLOMAP registered all 16 frames in that condition and pi3 / VGGT predictions exist (they always
    predict 16).

Outputs (in --out):
  colmap.npz, glomap.npz          {key}_{GS|RS} -> (16,4,4) cam-to-world (OpenCV); NaN rows = unregistered frame
  colmap_reg.npz, glomap_reg.npz  {key}_{GS|RS} -> (16,) bool registered mask
  per_window_sfm.csv              one row per (method, cond, window) incl. pi3 / vggt
  summary_sfm.csv / summary_sfm.md
usage: python collect_sfm.py [--jobs <jobs.jsonl>] [--out <dir>]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser("~/rs3d/src"))
from common import SPEED_BINS, pose_metrics  # noqa: E402
from metrics2 import pose_auc  # noqa: E402

RES_TUM = os.path.expanduser("~/rs3d/results/exp2/gap/tum")
OUT = os.path.expanduser("~/rs3d/results/exp2/baselines/classical")
BINS = ["slow", "medium", "fast"]
MAPPERS = ["colmap", "glomap"]
FF = ["pi3", "vggt"]


def metrics(P, G):
    m = np.isfinite(P[:, 0, 0])
    if m.sum() < 3:
        return {}
    d = pose_metrics(P[m], G[m])
    d["AUC30"] = pose_auc(P[m], G[m])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=f"{OUT}/jobs.jsonl")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    win = pd.read_csv(f"{RES_TUM}/windows.csv")
    win = win[win.keep].copy()
    win["bin"] = pd.cut(win.omega_mean, [0, *SPEED_BINS, np.inf], labels=BINS, right=False).astype(str)
    winfo = win.set_index("key")
    gt = np.load(f"{RES_TUM}/gt_poses.npz")

    jobs = {}
    with open(a.jobs) as f:
        for line in f:
            j = json.loads(line)
            jobs[(j["key"], j["cond"])] = j  # last record wins
    rows, npz, reg = [], {m: {} for m in MAPPERS}, {m: {} for m in MAPPERS}
    for (key, cond), j in sorted(jobs.items()):
        G = gt[f"{key}_{cond}"]
        w = winfo.loc[key]
        base = dict(seq=int(w.seq), window_id=int(w.window_id), key=key, cond=cond, bin=w.bin,
                    omega_mean=float(w.omega_mean), rs_shift_px=float(w.rs_shift_px))
        for m in MAPPERS:
            r = dict(base, method=m)
            if j["status"] != "ok":
                r.update(status="fail", n_reg=0, err="job error: " + j.get("trace", "").strip().splitlines()[-1])
                P = np.full((16, 4, 4), np.nan)
            else:
                mm = j["mappers"][m]
                P = np.array(mm["poses"], dtype=np.float64)
                n = int(np.isfinite(P[:, 0, 0]).sum())
                r.update(n_reg=n, status="full" if n == 16 else ("partial" if n >= 3 else "fail"),
                         n_models=mm["n_models"], n_points3D=mm["n_points3D"], mean_reproj=mm["mean_reproj"],
                         t_feat=j["t_feat"], t_map=mm["t_map"], n_kp_median=j["n_kp_median"],
                         n_pairs_verified=j["n_pairs_verified"], err=mm["err"])
                r.update(metrics(P, G))
            npz[m][f"{key}_{cond}"] = P
            reg[m][f"{key}_{cond}"] = np.isfinite(P[:, 0, 0])
            rows.append(r)
    for m in FF:
        p = f"{RES_TUM}/preds/{m}.npz"
        if not os.path.exists(p):
            continue
        pr = np.load(p)
        for (key, cond) in sorted(jobs):
            k = f"{key}_{cond}"
            if k not in pr:
                continue
            w = winfo.loc[key]
            r = dict(seq=int(w.seq), window_id=int(w.window_id), key=key, cond=cond, bin=w.bin,
                     omega_mean=float(w.omega_mean), rs_shift_px=float(w.rs_shift_px), method=m, n_reg=16,
                     status="full")
            r.update(metrics(pr[k], gt[k]))
            rows.append(r)
    df = pd.DataFrame(rows)
    for m in MAPPERS:
        np.savez(f"{a.out}/{m}.npz", **npz[m])
        np.savez(f"{a.out}/{m}_reg.npz", **reg[m])
    # common-full subset per condition: both mappers registered 16/16
    full = df[df.method.isin(MAPPERS)].pivot_table(index=["key", "cond"], columns="method", values="n_reg")
    cf = set(full[(full == 16).all(axis=1)].index)
    for m in FF:  # ... and every feed-forward model has a prediction for it
        if m in set(df.method):
            cf &= set(zip(df[df.method == m].key, df[df.method == m].cond))
    df["common_full"] = [(k, c) in cf for k, c in zip(df.key, df.cond)]
    cfb = {k for k in df.key.unique() if (k, "GS") in cf and (k, "RS") in cf}
    df["common_full_both"] = df.key.isin(cfb)
    # restrict pi3/vggt to windows the SfM job covered
    df.to_csv(f"{a.out}/per_window_sfm.csv", index=False)

    out = []
    methods = [m for m in MAPPERS + FF if m in set(df.method)]
    for cond in ["GS", "RS"]:
        for b in BINS + ["all"]:
            for m in methods:
                d = df[(df.method == m) & (df.cond == cond) & ((df.bin == b) if b != "all" else True)]
                if not len(d):
                    continue
                ok = d[d.status != "fail"]
                c = d[d.common_full]
                cb = d[d.common_full_both]
                out.append(dict(
                    cond=cond, bin=b, method=m, n_windows=len(d),
                    reg_rate=float(d.n_reg.sum() / (16 * len(d))),
                    full_rate=float((d.status == "full").mean()), partial_n=int((d.status == "partial").sum()),
                    fail_n=int((d.status == "fail").sum()), fail_rate=float((d.status == "fail").mean()),
                    n_ok=len(ok), ATE_med_ok=ok.ATE.median(), RPErot_med_ok=ok.RPE_rot.median(),
                    AUC30_med_ok=ok.AUC30.median(), AUC30_mean_ok=ok.AUC30.mean(),
                    n_cf=len(c), ATE_med_cf=c.ATE.median(), RPErot_med_cf=c.RPE_rot.median(),
                    AUC30_med_cf=c.AUC30.median(), AUC30_mean_cf=c.AUC30.mean(),
                    n_cfb=len(cb), ATE_med_cfb=cb.ATE.median()))
    s = pd.DataFrame(out)
    s.to_csv(f"{a.out}/summary_sfm.csv", index=False)
    # RS / GS ratio of median ATE on windows that are common-full in both conditions
    rat = []
    for b in BINS + ["all"]:
        for m in methods:
            d = df[(df.method == m) & df.common_full_both & ((df.bin == b) if b != "all" else True)]
            g, r_ = d[d.cond == "GS"].ATE.median(), d[d.cond == "RS"].ATE.median()
            rat.append(dict(bin=b, method=m, n=int((d.cond == "RS").sum()), ATE_GS=g, ATE_RS=r_, ratio=r_ / g))
    rat = pd.DataFrame(rat)
    rat.to_csv(f"{a.out}/summary_sfm_rs_gs_ratio.csv", index=False)
    with open(f"{a.out}/summary_sfm.md", "w") as f:
        f.write(f"jobs: {len(jobs)} (window, cond) of {2 * len(win)}\n\n")
        cols1 = ["cond", "bin", "method", "n_windows", "reg_rate", "full_rate", "partial_n", "fail_n", "n_ok",
                 "ATE_med_ok", "RPErot_med_ok", "AUC30_med_ok"]
        cols2 = ["cond", "bin", "method", "n_cf", "ATE_med_cf", "RPErot_med_cf", "AUC30_med_cf", "AUC30_mean_cf"]
        f.write("## Registration and all-window metrics (failures excluded)\n\n")
        f.write(s[cols1].round(4).to_markdown(index=False) + "\n\n")
        f.write("## Common-full subset (COLMAP and GLOMAP both 16/16)\n\n")
        f.write(s[cols2].round(4).to_markdown(index=False) + "\n\n")
        f.write("## RS/GS median ATE ratio, windows common-full in both conditions\n\n")
        f.write(rat.round(4).to_markdown(index=False) + "\n")
    print(open(f"{a.out}/summary_sfm.md").read())


if __name__ == "__main__":
    main()
