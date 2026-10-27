"""EXP0 analysis: per-window metrics, speed bins, bootstrap CIs, Wilcoxon, trend test, plots, verdict.

writes results/exp0/{per_window.csv, bins_summary.csv, trend.csv, verdict.json, key_table.md,
                     err_vs_speed_<model>.png, ratio_vs_speed.png, qual_traj_<key>.png}
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(__file__))
from common import N_FRAMES, RES, SPEED_BINS, STRIDE, pose_metrics, set_suffix

FPS = 20.0

MODELS = ["mapanything", "vggt", "pi3"]
LABEL = dict(mapanything="MapAnything", vggt="VGGT", pi3="π³")
METRICS = ["ATE", "RPE_rot", "RPE_trans", "RRA5", "RTA5"]
ERR_METRICS = ["ATE", "RPE_rot", "RPE_trans"]  # lower is better; ratio = RS / GS
UNIT = dict(ATE="m", RPE_rot="deg", RPE_trans="m")
BINS = ["slow", "medium", "fast"]
N_BOOT = 1000

# reference palette (dataviz skill): categorical slots 1-3, ink and chrome tokens, light surface
C_GS, C_RS = "#2a78d6", "#eb6834"
C_MODEL = dict(mapanything="#2a78d6", vggt="#eb6834", pi3="#1baf7a")
MARK_MODEL = dict(mapanything="o", vggt="s", pi3="^")
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "legend.frameon": False,
})


def compute_per_window(which="main"):
    sfx = set_suffix(which)
    win = pd.read_csv(f"{RES}/windows{sfx}.csv")
    win = win[win.keep].copy()
    win["v_mean"] = win.gt_path_m / ((N_FRAMES - 1) * STRIDE / FPS)  # GT camera speed over the window (m/s)
    if (pd.cut(win.omega_mean, [0, *SPEED_BINS, np.inf]).value_counts() < 15).any():
        cuts = list(win.omega_mean.quantile([1 / 3, 2 / 3]))
        bin_rule = f"terciles of omega_mean, cut points {cuts[0]:.3f}, {cuts[1]:.3f} rad/s"
    else:
        cuts = SPEED_BINS
        bin_rule = f"fixed: slow < {cuts[0]} <= medium < {cuts[1]} <= fast (rad/s)"
    win["bin"] = pd.cut(win.omega_mean, [0, *cuts, np.inf], labels=BINS, right=False).astype(str)
    gt = np.load(f"{RES}/gt_poses{sfx}.npz")
    rows = []
    for m in MODELS:
        p = f"{RES}/preds/{m}{sfx}.npz"
        if not os.path.exists(p):
            print("missing", p)
            continue
        pr = np.load(p)
        for r in win.itertuples():
            for cam in ["GS", "RS"]:
                k = f"{r.key}_{cam}"
                if k not in pr:
                    continue
                met = pose_metrics(pr[k], gt[k])
                extra = {}
                if cam == "RS" and which == "main":  # sensitivity: RS GT at the first-row timestamp
                    extra["ATE_rs_firstrow_gt"] = pose_metrics(pr[k], gt[f"{r.key}_RSfirst"])["ATE"]
                rows.append(dict(seq=r.seq, window_id=r.window_id, key=r.key, model=m, camera=cam,
                                 omega_mean=r.omega_mean, v_mean=r.v_mean, rs_shift_px=r.rs_shift_px,
                                 gt_path_m=r.gt_path_m, bin=r.bin, **met, **extra))
    return pd.DataFrame(rows), bin_rule, cuts


def paired(df, m, metric):
    d = df[df.model == m].pivot_table(index="key", columns="camera", values=metric)
    meta = df[(df.model == m) & (df.camera == "GS")].set_index("key")[["omega_mean", "v_mean", "bin", "seq"]]
    return d.dropna().join(meta)


def regress_logratio(df):
    """OLS per model/metric: log(RS/GS) = b0 + b_w * omega + b_v * v (rotation- vs translation-driven RS effect)."""
    out = []
    for m in df.model.unique():
        for metric in ["ATE", "RPE_rot"]:
            P = paired(df, m, metric)
            y = np.log(P.RS / P.GS).values
            X = np.column_stack([np.ones(len(P)), P.omega_mean, P.v_mean])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            res = y - X @ beta
            se = np.sqrt(np.diag(res @ res / (len(y) - 3) * np.linalg.inv(X.T @ X)))
            out.append(dict(model=m, metric=metric, n=len(P), b0=beta[0], b_omega=beta[1], se_omega=se[1],
                            b_v=beta[2], se_v=se[2], corr_omega_v=float(np.corrcoef(P.omega_mean, P.v_mean)[0, 1]),
                            ratio_at_zero_motion=float(np.exp(beta[0]))))
    return pd.DataFrame(out)


def control_summary():
    """GS-vs-GS control (calibration sequence, both cameras global shutter): cam1/cam0 error ratio."""
    if not any(os.path.exists(f"{RES}/preds/{m}_control.npz") for m in MODELS):
        return None, None
    dfc, rule, _ = compute_per_window("control")
    dfc.to_csv(f"{RES}/control_per_window.csv", index=False)
    rng = np.random.default_rng(2)
    rows = []
    for m in dfc.model.unique():
        for metric in ERR_METRICS:
            P = paired(dfc, m, metric)
            for b in sorted(P.bin.unique()) + ["all"]:
                Q = P if b == "all" else P[P.bin == b]
                if len(Q) < 5:
                    continue
                lo, hi = ratio_ci(Q.GS.values, Q.RS.values, rng)
                rows.append(dict(model=m, metric=metric, bin=b, n=len(Q), median_cam0=Q.GS.median(),
                                 median_cam1=Q.RS.median(), ratio=Q.RS.median() / Q.GS.median(),
                                 ratio_ci_lo=lo, ratio_ci_hi=hi, wilcoxon_p=wilcoxon(Q.RS, Q.GS).pvalue))
    cs = pd.DataFrame(rows)
    cs.to_csv(f"{RES}/control_summary.csv", index=False)
    return cs, rule


def ratio_ci(gs, rs, rng):
    n = len(gs)
    idx = rng.integers(0, n, (N_BOOT, n))
    r = np.median(rs[idx], 1) / np.median(gs[idx], 1)
    return np.percentile(r, [2.5, 97.5])


def summarize(df):
    rng = np.random.default_rng(0)
    rows, trend = [], []
    for m in df.model.unique():
        for metric in METRICS:
            P = paired(df, m, metric)
            for b in BINS + ["all"]:
                Q = P if b == "all" else P[P.bin == b]
                gs, rs = Q.GS.values, Q.RS.values
                if len(Q) < 2:
                    continue
                mg, mr = np.median(gs), np.median(rs)
                lo, hi = ratio_ci(gs, rs, rng) if mg > 0 else (np.nan, np.nan)
                diff = rs - gs
                p = wilcoxon(rs, gs).pvalue if np.any(diff != 0) else 1.0
                rows.append(dict(model=m, bin=b, metric=metric, n=len(Q), median_GS=mg, median_RS=mr,
                                 ratio=mr / mg if mg > 0 else np.nan, ratio_ci_lo=lo, ratio_ci_hi=hi,
                                 wilcoxon_p=p, median_diff_RS_minus_GS=np.median(diff),
                                 frac_RS_worse=float(np.mean(diff > 0)) if metric in ERR_METRICS
                                 else float(np.mean(diff < 0))))
            if metric in ERR_METRICS:
                lr = np.log(P.RS / P.GS)
                rho, pv = spearmanr(P.omega_mean, lr)
                trend.append(dict(model=m, metric=metric, n=len(P), spearman_rho_omega_vs_logratio=rho, p=pv))
    return pd.DataFrame(rows), pd.DataFrame(trend)


def verdict(bs):
    out = {}
    for m in bs.model.unique():
        a = bs[(bs.model == m) & (bs.metric == "ATE")].set_index("bin")
        rs, rm, rf = a.loc["slow", "ratio"], a.loc["medium", "ratio"], a.loc["fast", "ratio"]
        increases = bool(rf > rs)
        monotone = bool(rs < rm < rf)
        confound = bool(rs > 1.3)
        corrected = rf / rs
        if rf >= 1.5 and increases and rs <= 1.2:
            v = "GO"
        elif rf >= 1.2:
            v = "GRAY"
        else:
            v = "STOP"
        rot = bs[(bs.model == m) & (bs.metric == "RPE_rot")].set_index("bin")["ratio"].to_dict()
        out[m] = dict(verdict=v, ATE_ratio_slow=rs, ATE_ratio_medium=rm, ATE_ratio_fast=rf,
                      increases_slow_to_fast=increases, monotone=monotone, slow_ratio_gt_1p3_confound=confound,
                      corrected_gap_fast_over_slow=corrected, RPE_rot_ratios=rot)
    vs = {m: out[m]["verdict"] for m in out}
    if vs.get("mapanything") == "GO" and sum(v == "GO" for k, v in vs.items() if k != "mapanything") >= 1:
        overall = "GO"
    elif all(out[m]["ATE_ratio_fast"] < 1.2 for m in out):
        overall = "STOP"
    else:
        overall = "GRAY"
    return dict(per_model=out, overall=overall)


def binned_median(x, y, edges):
    c, med = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (x >= a) & (x < b)
        if s.sum() >= 5:
            c.append(np.median(x[s]))
            med.append(np.median(y[s]))
    return np.array(c), np.array(med)


def plot_err_vs_speed(df, m, cuts):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    sub = df[df.model == m]
    edges = np.quantile(sub.omega_mean, np.linspace(0, 1, 9))
    edges[-1] += 1e-6
    for ax, metric in zip(axs, ERR_METRICS):
        for cam, col in [("GS", C_GS), ("RS", C_RS)]:
            s = sub[sub.camera == cam]
            ax.scatter(s.omega_mean, s[metric], s=6, color=col, alpha=0.25, linewidths=0)
            cx, my = binned_median(s.omega_mean.values, s[metric].values, edges)
            ax.plot(cx, my, "-o", color=col, lw=2, ms=6, mec=SURF, mew=1.5,
                    label=f"{'global shutter (cam0)' if cam == 'GS' else 'rolling shutter (cam1)'}: binned median")
        for c in cuts:
            ax.axvline(c, color=MUTED, lw=1, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel("mean gyro speed over window, ω (rad/s)")
        ax.set_ylabel(f"{metric.replace('_', '-')} ({UNIT[metric]})")
        ax.set_title(metric.replace("_", "-"), color=INK, loc="left")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, fontsize=9)
    fig.suptitle(f"{LABEL[m]} — pose error vs camera rotation speed, TUM RS-VIO (measured; dashed = bin edges)",
                 color=INK, x=0.01, ha="left")
    fig.savefig(f"{RES}/err_vs_speed_{m}.png", dpi=140)
    plt.close(fig)


def plot_ratio_vs_speed(df):
    rng = np.random.default_rng(1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for ax, metric in zip(axs, ["ATE", "RPE_rot"]):
        for m in df.model.unique():
            P = paired(df, m, metric)
            edges = np.quantile(P.omega_mean, np.linspace(0, 1, 7))
            edges[-1] += 1e-6
            xs, rs, lo, hi = [], [], [], []
            for a, b in zip(edges[:-1], edges[1:]):
                Q = P[(P.omega_mean >= a) & (P.omega_mean < b)]
                xs.append(Q.omega_mean.median())
                rs.append(Q.RS.median() / Q.GS.median())
                l_, h_ = ratio_ci(Q.GS.values, Q.RS.values, rng)
                lo.append(l_)
                hi.append(h_)
            xs, rs = np.array(xs), np.array(rs)
            ax.errorbar(xs, rs, yerr=[rs - np.array(lo), np.array(hi) - rs], fmt="-" + MARK_MODEL[m],
                        color=C_MODEL[m], lw=2, ms=7, mec=SURF, mew=1.5, capsize=3, elinewidth=1, label=LABEL[m])
            ax.annotate(LABEL[m], (xs[-1], rs[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                        color=INK2, fontsize=9)
        for yv in [1.0, 1.2, 1.5]:
            ax.axhline(yv, color=MUTED, lw=1, ls=":" if yv != 1 else "-", zorder=0)
        ax.set_xlabel("mean gyro speed over window, ω (rad/s) — sextile bins")
        ax.set_ylabel(f"median RS / median GS ({metric.replace('_', '-')})")
        ax.set_title(f"{metric.replace('_', '-')} ratio, 95% bootstrap CI", color=INK, loc="left")
    axs[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Rolling-shutter / global-shutter error ratio vs rotation speed (measured). "
                 "Solid line = no gap (1.0×); dotted = 1.2× and 1.5× (GO threshold)", color=INK, x=0.01, ha="left")
    fig.savefig(f"{RES}/ratio_vs_speed.png", dpi=140)
    plt.close(fig)


def plot_qualitative(df, n=3):
    gt = np.load(f"{RES}/gt_poses.npz")
    preds = {m: np.load(f"{RES}/preds/{m}.npz") for m in df.model.unique()}
    from common import umeyama_sim3
    complete = df.groupby("key").size() == 2 * len(preds)
    win = df[(df.camera == "GS") & df.key.isin(complete[complete].index)].drop_duplicates("key")
    win = win.sort_values("omega_mean", ascending=False).head(n)
    keys = list(win.key)
    for key, om in zip(keys, win.omega_mean):
        fig, axs = plt.subplots(1, len(preds), figsize=(4.4 * len(preds), 4.4), constrained_layout=True)
        Gg = gt[f"{key}_GS"][:, :3, 3]
        c0 = Gg.mean(0)
        _, _, Vt = np.linalg.svd(Gg - c0)
        B = Vt[:2]  # plane of largest GT motion ("top-down" view of the window)
        for ax, (m, pr) in zip(np.atleast_1d(axs), preds.items()):
            for cam, col in [("GS", C_GS), ("RS", C_RS)]:
                G = gt[f"{key}_{cam}"][:, :3, 3]
                P = pr[f"{key}_{cam}"][:, :3, 3]
                s, R, t = umeyama_sim3(P, G)
                Pa = (s * (R @ P.T)).T + t
                g2, p2 = (G - c0) @ B.T, (Pa - c0) @ B.T
                ate = df[(df.key == key) & (df.model == m) & (df.camera == cam)].ATE.iloc[0]
                ax.plot(g2[:, 0], g2[:, 1], "-", color=col, lw=1, alpha=0.45)
                ax.plot(p2[:, 0], p2[:, 1], "-o", color=col, lw=2, ms=4,
                        label=f"{cam} pred (ATE {ate * 100:.1f} cm)")
            ax.plot([], [], "-", color=MUTED, lw=1, label="thin line = GT for that camera")
            ax.set_aspect("equal")
            ax.set_title(LABEL[m], color=INK, loc="left")
            ax.set_xlabel("m")
            ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
        fig.suptitle(f"{key}: ω = {om:.2f} rad/s — Sim(3)-aligned camera centres, projected on the main motion plane "
                     "(measured)", color=INK, x=0.01, ha="left", fontsize=10)
        fig.savefig(f"{RES}/qual_traj_{key}.png", dpi=130)
        plt.close(fig)
    return keys


def key_table(bs, bin_rule):
    lines = [f"Bins: {bin_rule}", "",
             "| model | metric | bin | n | median GS | median RS | ratio RS/GS [95% CI] | Wilcoxon p |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in bs[bs.metric.isin(["ATE", "RPE_rot", "RPE_trans"])].iterrows():
        f = (lambda v: f"{v * 100:.2f} cm") if UNIT[r.metric] == "m" else (lambda v: f"{v:.3f}°")
        lines.append(f"| {LABEL[r.model]} | {r.metric} | {r.bin} | {r.n} | {f(r.median_GS)} | {f(r.median_RS)} | "
                     f"{r.ratio:.2f} [{r.ratio_ci_lo:.2f}, {r.ratio_ci_hi:.2f}] | {r.wilcoxon_p:.1e} |")
    return "\n".join(lines)


def main():
    df, bin_rule, cuts = compute_per_window()
    df.to_csv(f"{RES}/per_window.csv", index=False)
    bs, tr = summarize(df)
    bs.to_csv(f"{RES}/bins_summary.csv", index=False)
    tr.to_csv(f"{RES}/trend.csv", index=False)
    v = verdict(bs)
    v["bin_rule"] = bin_rule
    v["n_windows_per_model"] = df[df.camera == "GS"].groupby("model").size().to_dict()
    reg = regress_logratio(df)
    reg.to_csv(f"{RES}/regression_logratio.csv", index=False)
    cs, crule = control_summary()
    if cs is not None:
        v["control_bin_rule"] = crule
        for m in cs.model.unique():
            c = cs[(cs.model == m) & (cs.metric == "ATE") & (cs.bin == "all")].iloc[0]
            v["per_model"][m]["control_ATE_ratio_cam1_over_cam0"] = c.ratio
            v["per_model"][m]["fast_ratio_over_control_ratio"] = v["per_model"][m]["ATE_ratio_fast"] / c.ratio
        print(cs.round(4).to_string())
    print(reg.round(4).to_string())
    with open(f"{RES}/verdict.json", "w") as f:
        json.dump(v, f, indent=1, default=float)
    with open(f"{RES}/key_table.md", "w") as f:
        f.write(key_table(bs, bin_rule) + "\n")
    for m in df.model.unique():
        plot_err_vs_speed(df, m, cuts)
    plot_ratio_vs_speed(df)
    keys = plot_qualitative(df)
    print(key_table(bs, bin_rule))
    print(tr.round(4).to_string())
    print(json.dumps(v, indent=1, default=float))
    print("qualitative:", keys)
    rs = df[df.camera == "RS"]
    print("RS ATE sensitivity (first-row GT / centre-row GT), median per model:",
          (rs.groupby("model").ATE_rs_firstrow_gt.median() / rs.groupby("model").ATE.median()).round(3).to_dict())
    print("EVAL_DONE")


if __name__ == "__main__":
    main()
