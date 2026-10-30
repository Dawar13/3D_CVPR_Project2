"""Evaluate Barath RS-7AC / GS-5PC pairwise poses on EXP2 windows against the centre-row GT, next to pi3 / VGGT /
COLMAP / GLOMAP on the same pairs (relative poses taken from their window predictions), plus per-frame angular
velocity of the 7AC model against the gyro.

Relative pose convention (Barath's tester): X_c2 = R X_c1 + t, t unit. From cam-to-world poses P1, P2 (centres c):
  R = R2^T R1,  t = R2^T (c1 - c2) / |.|
Errors: rotation = angle(R_gt^T R_est); translation = angle(t_est, t_gt) (signed, primary) and the sign-free
version min(e, 180-e) used by the repo's evaluate_R_t; pose error = max(rot, trans). A pair a method could not solve
counts as 180 deg (repo convention) and is counted in n_fail.

Angular velocity: the RS model is R_i(tau) = (I + tau [w_i]x) R_i(0), tau = y/h - 0.5 (world->camera), so a camera
with camera-frame angular velocity omega_c gives w = -omega_c * readout. We convert omega_est = -w / readout
(readout = 1024 rows x 29.4737 us) and compare with the gyro (src/unroll.py frame_motion: mean gyro over the
readout, rotated into cam1). The sign is also checked empirically (median cosine to the gyro with either sign).

usage: python eval_barath.py [--cond RS]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser("~/rs3d/src"))
from common import READOUT, SPEED_BINS  # noqa: E402
from unroll import frame_motion  # noqa: E402

RES_TUM = os.path.expanduser("~/rs3d/results/exp2/gap/tum")
CLS = os.path.expanduser("~/rs3d/results/exp2/baselines/classical")
OUT = f"{CLS}/barath"
BINS = ["slow", "medium", "fast"]
N = 16


def rel_from_c2w(P1, P2):
    R1, R2, c1, c2 = P1[:3, :3], P2[:3, :3], P1[:3, 3], P2[:3, 3]
    t = R2.T @ (c1 - c2)
    return R2.T @ R1, t / (np.linalg.norm(t) + 1e-12)


def rot_err(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def t_err(ta, tb):
    ta, tb = ta / (np.linalg.norm(ta) + 1e-12), tb / (np.linalg.norm(tb) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(ta @ tb, -1, 1))))


def auc(errs, thr):
    """Repo's continuous pose AUC (area under the recall curve up to thr, normalised)."""
    e = np.sort(np.asarray(errs, dtype=float))
    rec = (np.arange(len(e)) + 1) / len(e)
    e, rec = np.r_[0.0, e], np.r_[0.0, rec]
    i = np.searchsorted(e, thr)
    r, x = np.r_[rec[:i], rec[i - 1]], np.r_[e[:i], thr]
    return float(np.trapezoid(r, x=x) / thr) if hasattr(np, "trapezoid") else float(np.trapz(r, x=x) / thr)


def parse_vec(s):
    return np.array([float(x) for x in str(s).split()]) if isinstance(s, str) and s.strip() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="RS")
    ap.add_argument("--sfm", default=f"{CLS}/sfm_cpu", help="dir with colmap.npz / glomap.npz")
    a = ap.parse_args()
    cond = a.cond
    win = pd.read_csv(f"{RES_TUM}/windows.csv")
    win = win[win.keep].copy()
    win["bin"] = pd.cut(win.omega_mean, [0, *SPEED_BINS, np.inf], labels=BINS, right=False).astype(str)
    gt = np.load(f"{RES_TUM}/gt_poses.npz")
    gkey = "GS" if cond == "GS" else "RS"
    pr = pd.read_csv(f"{OUT}/pairs_{cond}.csv", dtype={"ta": str, "tb": str})
    pr = pr.drop_duplicates(["seq", "ta", "tb", "method"], keep="last")
    sol = {(int(r.seq), r.ta, r.tb, r.method): r for r in pr.itertuples()}
    others = {}
    for m in ["pi3", "vggt"]:
        p = f"{RES_TUM}/preds/{m}.npz"
        if os.path.exists(p):
            others[m] = np.load(p)
    for m in ["colmap", "glomap"]:
        p = f"{a.sfm}/{m}.npz"
        if os.path.exists(p):
            others[m] = np.load(p)

    rows, omg = [], []
    for r in win.itertuples():
        d = f"{os.path.expanduser('~/rs3d/data/windows')}/seq{r.seq}/{r.window_id:04d}_{cond}"
        names = sorted(n for n in os.listdir(d) if n.endswith(".png"))
        ts = [n.split("_")[1][:-4] for n in names]
        G = gt[f"{r.key}_{gkey}"]
        w_est = [[] for _ in range(N)]
        for g in (1, 2):
            for k in range(N - g):
                Rg, tg = rel_from_c2w(G[k], G[k + g])
                base = dict(seq=r.seq, key=r.key, bin=r.bin, omega_mean=r.omega_mean, gap=g, k=k, ta=ts[k],
                            tb=ts[k + g], gt_rot_deg=rot_err(np.eye(3), Rg),
                            gt_baseline_m=float(np.linalg.norm(G[k][:3, 3] - G[k + g][:3, 3])))
                for m in ("ac_rs_7direct", "gs_5pt"):
                    s = sol.get((int(r.seq), ts[k], ts[k + g], m))
                    if s is None:
                        continue  # pair not solved (yet)
                    R, t = parse_vec(s.R), parse_vec(s.t)
                    ok = bool(s.ok) and R is not None
                    e_r = rot_err(Rg, R.reshape(3, 3)) if ok else 180.0
                    e_t = t_err(t, tg) if ok else 180.0
                    rows.append(dict(base, method="barath_7ac" if m == "ac_rs_7direct" else "barath_gs5pc",
                                     ok=ok, rot_err=e_r, t_err=e_t, t_err_unsigned=min(e_t, 180 - e_t),
                                     n_acs=s.n_acs, n_inliers=s.n_inliers))
                    if ok and m == "ac_rs_7direct" and g == 1:
                        w1, w2 = parse_vec(s.w1), parse_vec(s.w2)
                        w_est[k].append(("w1", w1))
                        w_est[k + 1].append(("w2", w2))
                for m, z in others.items():
                    kk = f"{r.key}_{cond}"
                    if kk not in z:
                        continue
                    P = z[kk]
                    ok = bool(np.isfinite(P[k, 0, 0]) and np.isfinite(P[k + g, 0, 0]))
                    if ok:
                        R, t = rel_from_c2w(P[k], P[k + g])
                        e_r, e_t = rot_err(Rg, R), t_err(t, tg)
                    else:
                        e_r = e_t = 180.0
                    rows.append(dict(base, method=m, ok=ok, rot_err=e_r, t_err=e_t,
                                     t_err_unsigned=min(e_t, 180 - e_t)))
        # per-frame omega vs gyro
        if cond == "RS" or cond == "GS":
            mot = frame_motion(int(r.seq)) if r.seq not in _MOT else _MOT[r.seq]
            _MOT[r.seq] = mot
            for k in range(N):
                wg = mot[int(ts[k])][0]
                for src, w in w_est[k]:
                    omg.append(dict(seq=r.seq, key=r.key, bin=r.bin, k=k, t=ts[k], src=src,
                                    est_x=-w[0] / READOUT, est_y=-w[1] / READOUT, est_z=-w[2] / READOUT,
                                    gyro_x=wg[0], gyro_y=wg[1], gyro_z=wg[2]))
    df = pd.DataFrame(rows)
    df["pose_err"] = np.maximum(df.rot_err, df.t_err)
    df["pose_err_unsigned"] = np.maximum(df.rot_err, df.t_err_unsigned)
    df.to_csv(f"{OUT}/per_pair_{cond}.csv", index=False)

    # Barath windows are evaluated only when all their consecutive pairs are solved (complete windows)
    b7 = df[(df.method == "barath_7ac") & (df.gap == 1)].groupby("key").size()
    complete = set(b7[b7 == N - 1].index)
    df = df[df.key.isin(complete)]
    have = df.pivot_table(index=["key", "gap", "k"], columns="method", values="rot_err", aggfunc="first")
    sets = {"A": ["barath_7ac", "barath_gs5pc", "pi3", "vggt"],
            "B": ["barath_7ac", "barath_gs5pc", "pi3", "vggt", "colmap", "glomap"]}
    out = []
    dfi = df.set_index(["key", "gap", "k"])
    for sname, methods in sets.items():
        methods = [m for m in methods if m in have.columns]
        common = have[methods].dropna().index
        dfc = dfi.loc[dfi.index.isin(common)].reset_index()
        for g in sorted(dfc.gap.unique()):
            for b in BINS + ["all"]:
                for m in methods:
                    d = dfc[(dfc.gap == g) & (dfc.method == m) & ((dfc.bin == b) if b != "all" else True)]
                    if not len(d):
                        continue
                    out.append(dict(set=sname, cond=cond, gap=g, bin=b, method=m, n_windows=d.key.nunique(),
                                    n_pairs=len(d), n_fail=int((~d.ok).sum()), rot_med=d.rot_err.median(),
                                    t_med=d.t_err.median(), t_unsigned_med=d.t_err_unsigned.median(),
                                    AUC5=auc(d.pose_err, 5), AUC10=auc(d.pose_err, 10), AUC20=auc(d.pose_err, 20),
                                    AUC10_unsigned=auc(d.pose_err_unsigned, 10)))
    s = pd.DataFrame(out)
    s.to_csv(f"{OUT}/summary_pairs_{cond}.csv", index=False)
    with open(f"{OUT}/summary_pairs_{cond}.md", "w") as f:
        f.write(f"cond={cond}; complete Barath windows: {len(complete)}. Set A = Barath + pi3/VGGT on those windows; "
                f"set B = additionally COLMAP/GLOMAP (windows also done by the SfM run; a pair with an unregistered "
                f"frame counts as 180 deg)\n\n")
        f.write(s.round(4).to_markdown(index=False) + "\n")
    print(open(f"{OUT}/summary_pairs_{cond}.md").read())

    omg = [r for r in omg if r["key"] in complete]
    if omg:
        o = pd.DataFrame(omg)
        E = o[["est_x", "est_y", "est_z"]].to_numpy()
        Gy = o[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()
        ng = np.linalg.norm(Gy, axis=1)
        o["err"] = np.linalg.norm(E - Gy, axis=1)
        o["err_flipped"] = np.linalg.norm(-E - Gy, axis=1)
        o["gyro_norm"] = ng
        o["est_norm"] = np.linalg.norm(E, axis=1)
        o["cos"] = (E * Gy).sum(1) / (np.linalg.norm(E, axis=1) * ng + 1e-12)
        o.to_csv(f"{OUT}/omega_vs_gyro_{cond}.csv", index=False)
        # per-frame (16,3) average of the available consecutive-pair estimates (rad/s, camera frame)
        if cond == "RS":
            om = {}
            for key, d in o.groupby("key"):
                A = np.full((N, 3), np.nan)
                for k, dk in d.groupby("k"):
                    A[k] = dk[["est_x", "est_y", "est_z"]].mean().to_numpy()
                om[key] = A
            np.savez(f"{OUT}/barath_omega.npz", **om)
        fr = o.groupby(["key", "k"]).agg(bin=("bin", "first"), gyro_norm=("gyro_norm", "first"),
                                          est_x=("est_x", "mean"), est_y=("est_y", "mean"),
                                          est_z=("est_z", "mean"), gx=("gyro_x", "first"),
                                          gy=("gyro_y", "first"), gz=("gyro_z", "first")).reset_index()
        Ef, Gf = fr[["est_x", "est_y", "est_z"]].to_numpy(), fr[["gx", "gy", "gz"]].to_numpy()
        fr["err"] = np.linalg.norm(Ef - Gf, axis=1)
        fr["rel_err"] = fr.err / (fr.gyro_norm + 1e-12)
        fr["cos"] = (Ef * Gf).sum(1) / (np.linalg.norm(Ef, axis=1) * fr.gyro_norm + 1e-12)
        fr["ratio"] = np.linalg.norm(Ef, axis=1) / (fr.gyro_norm + 1e-12)
        res = []
        for b in BINS + ["all"]:
            d = fr[(fr.bin == b) if b != "all" else slice(None)]
            dd = d[d.gyro_norm > 0.5]
            res.append(dict(cond=cond, bin=b, n_frames=len(d), gyro_norm_med=d.gyro_norm.median(),
                            err_med=d.err.median(), zero_pred_err_med=d.gyro_norm.median(),
                            rel_err_med=d.rel_err.median(), n_fast_frames=len(dd),
                            cos_med_fast=dd.cos.median(), angle_med_fast_deg=float(np.degrees(np.arccos(
                                np.clip(dd.cos.median(), -1, 1)))) if len(dd) else np.nan,
                            mag_ratio_med_fast=dd.ratio.median(), frac_frames_est_zero=float((d.ratio == 0).mean())))
        res = pd.DataFrame(res)
        for ax, i in zip("xyz", range(3)):
            res.loc[res.bin == "all", f"pearson_{ax}"] = np.corrcoef(Ef[:, i], Gf[:, i])[0, 1]
        sign = dict(median_cos_as_is=float(o.cos[o.gyro_norm > 0.5].median()),
                    median_err_as_is=float(o.err.median()), median_err_flipped=float(o.err_flipped.median()),
                    frac_est_exactly_zero=float((o.est_norm == 0).mean()), n_estimates=len(o))
        # 3x3 Pearson correlation est axis i vs gyro axis j (a frame/sign convention error would show off-diagonal)
        Eo, Go = o[["est_x", "est_y", "est_z"]].to_numpy(), o[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()
        C = np.array([[np.corrcoef(Eo[:, i], Go[:, j])[0, 1] for j in range(3)] for i in range(3)])
        sign["corr_est_i_vs_gyro_j"] = np.round(C, 3).tolist()
        # least-squares linear map gyro ~ M est (diagnostic only)
        M, *_ = np.linalg.lstsq(Eo, Go, rcond=None)
        sign["lstsq_R2_gyro_from_est"] = float(1 - ((Go - Eo @ M) ** 2).sum() / ((Go - Go.mean(0)) ** 2).sum())
        res.to_csv(f"{OUT}/summary_omega_{cond}.csv", index=False)
        with open(f"{OUT}/summary_omega_{cond}.md", "w") as f:
            f.write(f"omega_est = -w/readout (rad/s, camera frame); sign check: {sign}\n\n")
            f.write(res.round(4).to_markdown(index=False) + "\n")
        print(open(f"{OUT}/summary_omega_{cond}.md").read())


_MOT = {}

if __name__ == "__main__":
    main()
