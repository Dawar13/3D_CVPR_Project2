"""EXP1 analysis: how much of the GS-RS gap closes when RS frames are unrolled with GT motion.

recovered = (median RS - median RSU) / (median RS - median GS), per model x speed bin x metric x variant.
GO >= 50 %, STOP < 25 % (PROJECT_CONTEXT, Phase 1). Primary: ATE, fast bin, rotation-only (E1a).
writes results/exp1/{per_window_e1.csv, e1_summary.csv, e1_verdict.json, e1_key_table.md, *.png}
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(__file__))
from common import RES, pose_metrics
from evaluate import BINS, INK, INK2, LABEL, MODELS, MUTED, SURF, C_GS, C_RS, C_MODEL, MARK_MODEL  # noqa: F401

OUT = os.path.join(os.path.dirname(RES), "exp1")
VARIANTS = {"RSU": ("_e1rot", "E1a: rotation-only unrolling"), "RSUT": ("_e1rotrans", "E1b: rotation + translation")}
C_VAR = {"GS": C_GS, "RS": C_RS, "RSU": "#1baf7a", "RSUT": "#eda100"}
NAME_CAM = {"GS": "global shutter", "RS": "rolling shutter (raw)", "RSU": "RS unrolled, rotation",
            "RSUT": "RS unrolled, rotation + translation"}
METRICS = ["ATE", "RPE_rot", "RPE_trans"]
N_BOOT = 1000


def per_window():
    base = pd.read_csv(f"{RES}/per_window.csv")
    meta = base[base.camera == "GS"][["key", "model", "seq", "bin", "omega_mean", "v_mean"]]
    rows = [base[["key", "model", "camera"] + METRICS]]
    gt = np.load(f"{RES}/gt_poses.npz")
    for cam, (tag, _) in VARIANTS.items():
        for m in MODELS:
            p = f"{RES}/preds/{m}{tag}.npz"
            if not os.path.exists(p):
                continue
            pr = np.load(p)
            recs = []
            for key in meta[meta.model == m].key:
                k = f"{key}_{cam}"
                if k in pr:
                    met = pose_metrics(pr[k], gt[f"{key}_RS"])  # unrolled to the centre-row time = RS GT
                    recs.append(dict(key=key, model=m, camera=cam, **{x: met[x] for x in METRICS}))
            rows.append(pd.DataFrame(recs))
    df = pd.concat(rows, ignore_index=True).merge(meta, on=["key", "model"])
    return df


def recovered(gs, rs, ru):
    g, r, u = np.median(gs), np.median(rs), np.median(ru)
    return (r - u) / (r - g) if r - g > 1e-12 else np.nan


def summarize(df, ctrl):
    rng = np.random.default_rng(0)
    out = []
    for m in df.model.unique():
        for metric in METRICS:
            P = df[df.model == m].pivot_table(index="key", columns="camera", values=metric)
            P = P.join(df[(df.model == m) & (df.camera == "GS")].set_index("key")[["bin"]])
            for cam in [c for c in VARIANTS if c in P.columns]:
                Q0 = P.dropna(subset=["GS", "RS", cam])
                for b in BINS + ["all"]:
                    Q = Q0 if b == "all" else Q0[Q0.bin == b]
                    if len(Q) < 5:
                        continue
                    gs, rs, ru = Q.GS.values, Q.RS.values, Q[cam].values
                    rec = recovered(gs, rs, ru)
                    idx = rng.integers(0, len(Q), (N_BOOT, len(Q)))
                    boots = np.array([recovered(gs[i], rs[i], ru[i]) for i in idx])
                    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
                    c = ctrl.get(m, 1.0) if metric == "ATE" else 1.0
                    floor = np.median(gs) * c
                    rec_floor = ((np.median(rs) - np.median(ru)) / (np.median(rs) - floor)
                                 if np.median(rs) - floor > 1e-12 else np.nan)
                    out.append(dict(model=m, metric=metric, variant=cam, bin=b, n=len(Q), median_GS=np.median(gs),
                                    median_RS=np.median(rs), median_unrolled=np.median(ru),
                                    ratio_RS_over_GS=np.median(rs) / np.median(gs),
                                    ratio_unrolled_over_GS=np.median(ru) / np.median(gs),
                                    recovered=rec, recovered_ci_lo=lo, recovered_ci_hi=hi,
                                    recovered_vs_control_floor=rec_floor,
                                    wilcoxon_p_unrolled_vs_RS=wilcoxon(ru, rs).pvalue,
                                    frac_windows_improved=float(np.mean(ru < rs))))
    return pd.DataFrame(out)


def verdict(sm):
    v = {}
    for m in sm.model.unique():
        v[m] = {}
        for cam in sm.variant.unique():
            a = sm[(sm.model == m) & (sm.variant == cam) & (sm.metric == "ATE")].set_index("bin")
            if "fast" not in a.index:
                continue
            r = a.loc["fast", "recovered"]
            v[m][cam] = dict(recovered_fast_ATE=r, ci=[a.loc["fast", "recovered_ci_lo"], a.loc["fast", "recovered_ci_hi"]],
                             recovered_by_bin={b: a.loc[b, "recovered"] for b in a.index},
                             result="GO" if r >= 0.5 else ("STOP" if r < 0.25 else "MIDDLE (25-50%)"))
    return v


def plot_bars(sm):
    cams = ["GS", "RS"] + [c for c in VARIANTS if c in sm.variant.unique()]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for ax, m in zip(axs, MODELS):
        a = sm[(sm.model == m) & (sm.metric == "ATE")]
        w = 0.8 / len(cams)
        for j, cam in enumerate(cams):
            ys = []
            for b in BINS:
                q = a[(a.bin == b)]
                if cam in ("GS", "RS"):
                    ys.append(q[f"median_{cam}"].iloc[0] * 100)
                else:
                    ys.append(q[q.variant == cam].median_unrolled.iloc[0] * 100)
            xs = np.arange(3) + (j - (len(cams) - 1) / 2) * w
            ax.bar(xs, ys, width=w * 0.92, color=C_VAR[cam], label=NAME_CAM[cam], zorder=2)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["slow", "medium", "fast"])
        ax.set_title(LABEL[m], loc="left", color=INK)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("median path error, ATE (cm)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=len(cams), fontsize=9, frameon=False)
    fig.suptitle("EXP1: does unrolling the RS frames with GT motion bring the error back to global-shutter level? "
                 "(measured)", color=INK, x=0.01, ha="left")
    fig.savefig(f"{OUT}/e1_ate_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_recovered(df):
    rng = np.random.default_rng(1)
    cams = [c for c in VARIANTS if c in df.camera.unique()]
    fig, axs = plt.subplots(1, len(cams), figsize=(6.2 * len(cams), 4.3), constrained_layout=True, squeeze=False)
    for ax, cam in zip(axs[0], cams):
        for m in MODELS:
            P = df[df.model == m].pivot_table(index="key", columns="camera", values="ATE").dropna(
                subset=["GS", "RS", cam])
            om = df[(df.model == m) & (df.camera == "GS")].set_index("key").omega_mean.reindex(P.index)
            edges = np.quantile(om, np.linspace(0, 1, 7))
            edges[-1] += 1e-6
            xs, ys, lo, hi = [], [], [], []
            for a_, b_ in zip(edges[:-1], edges[1:]):
                Q = P[(om >= a_) & (om < b_)]
                xs.append(om[(om >= a_) & (om < b_)].median())
                r = recovered(Q.GS.values, Q.RS.values, Q[cam].values)
                idx = rng.integers(0, len(Q), (300, len(Q)))
                bt = [recovered(Q.GS.values[i], Q.RS.values[i], Q[cam].values[i]) for i in idx]
                l_, h_ = np.nanpercentile(bt, [2.5, 97.5])
                ys.append(r * 100)
                lo.append(r * 100 - l_ * 100)
                hi.append(h_ * 100 - r * 100)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="-" + MARK_MODEL[m], color=C_MODEL[m], lw=2, ms=7, mec=SURF,
                        mew=1.5, capsize=3, elinewidth=1, label=LABEL[m])
        for yv in [0, 25, 50, 100]:
            ax.axhline(yv, color=MUTED, lw=1, ls="-" if yv in (0, 100) else ":", zorder=0)
        ax.set_ylim(-60, 160)
        ax.set_xlabel("mean gyro speed over window, ω (rad/s), sextile bins")
        ax.set_ylabel("share of the GS–RS ATE gap closed (%)")
        ax.set_title(VARIANTS[cam][1], loc="left", color=INK)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Gap closed by oracle unrolling vs speed (measured)\n"
                 "dotted: 25 % (STOP below) and 50 % (GO); 100 % = back to GS level", color=INK, x=0.01, ha="left")
    fig.savefig(f"{OUT}/e1_recovered_vs_speed.png", dpi=150)
    plt.close(fig)


def key_table(sm):
    lines = ["| model | variant | bin | n | GS | RS raw | RS unrolled | gap closed [95% CI] | vs control floor | "
             "p (unrolled vs raw) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in sm[sm.metric == "ATE"].iterrows():
        lines.append(f"| {LABEL[r.model]} | {r.variant} | {r.bin} | {r.n} | {r.median_GS * 100:.2f} cm | "
                     f"{r.median_RS * 100:.2f} cm | {r.median_unrolled * 100:.2f} cm | {r.recovered * 100:.0f}% "
                     f"[{r.recovered_ci_lo * 100:.0f}, {r.recovered_ci_hi * 100:.0f}] | "
                     f"{r.recovered_vs_control_floor * 100:.0f}% | {r.wilcoxon_p_unrolled_vs_RS:.1e} |")
    lines += ["", "RPE-rot (deg):", "", "| model | variant | bin | GS | RS raw | RS unrolled | gap closed |",
              "|---|---|---|---|---|---|---|"]
    for _, r in sm[sm.metric == "RPE_rot"].iterrows():
        lines.append(f"| {LABEL[r.model]} | {r.variant} | {r.bin} | {r.median_GS:.3f} | {r.median_RS:.3f} | "
                     f"{r.median_unrolled:.3f} | {r.recovered * 100:.0f}% |")
    return "\n".join(lines)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = per_window()
    df.to_csv(f"{OUT}/per_window_e1.csv", index=False)
    cs = pd.read_csv(f"{RES}/control_summary.csv")
    ctrl = cs[(cs.metric == "ATE") & (cs.bin == "all")].set_index("model").ratio.to_dict()
    sm = summarize(df, ctrl)
    sm.to_csv(f"{OUT}/e1_summary.csv", index=False)
    v = verdict(sm)
    with open(f"{OUT}/e1_verdict.json", "w") as f:
        json.dump(v, f, indent=1, default=float)
    kt = key_table(sm)
    with open(f"{OUT}/e1_key_table.md", "w") as f:
        f.write(kt + "\n")
    plot_bars(sm)
    plot_recovered(df)
    print(kt)
    print(json.dumps(v, indent=1, default=float))
    print("E1_EVAL_DONE")


if __name__ == "__main__":
    main()
