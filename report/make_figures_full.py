"""Plain-language figures for the combined EXP0 + EXP1 team report (data from results/, measured)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from make_figures import C_GS, C_RS, GRID, INK, INK2, MODELS, MUTED, NAME, OUT

HERE = os.path.dirname(os.path.abspath(__file__))
RES1 = os.path.join(HERE, "..", "results", "exp1")

# same series colours as the technical EXP1 plots: aqua = straightened for turning, yellow = turning + walking
C_RSU, C_RSUT = "#1baf7a", "#eda100"
CAM = ["GS", "RS", "RSU", "RSUT"]
CAM_COL = dict(GS=C_GS, RS=C_RS, RSU=C_RSU, RSUT=C_RSUT)
CAM_NAME = dict(GS="normal camera", RS="rolling shutter, bent", RSU="straightened: turning",
                RSUT="straightened: turning + walking")
BIN_NAME = dict(slow="slow (under 29°/s)", medium="medium (29–86°/s)", fast="fast (86°/s or more)")


def recovered(gs, rs, ru):
    g, r, u = np.median(gs), np.median(rs), np.median(ru)
    return (r - u) / (r - g)


def fig_unroll_concept():
    """Three steps: bent rows -> known motion says how far each row moved -> straight again."""
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.5))
    rows = 12
    steps = [("1. Bent photo: each row\ntaken a little later", "bent"),
             ("2. The known motion tells us\nhow far each row moved", "arrows"),
             ("3. Slide every row back:\nstraight again", "straight")]
    for ax, (title, kind) in zip(axs, steps):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, rows)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(MUTED)
        for r in range(rows):
            ax.axhline(r, color=GRID, lw=0.6)
            x_bent, x_ok = 3.0 + 0.2 * (r - rows / 2) + 1.2, 4.2
            if kind == "bent":
                ax.add_patch(mpatches.Rectangle((x_bent, r + 0.06), 1.0, 0.88, color=C_RS))
            elif kind == "arrows":
                ax.add_patch(mpatches.Rectangle((x_bent, r + 0.06), 1.0, 0.88, color=C_RS, alpha=0.3))
                if abs(x_bent - x_ok) > 0.15:
                    ax.annotate("", xy=(x_ok + 0.5, r + 0.5), xytext=(x_bent + 0.5, r + 0.5),
                                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.1))
            else:
                ax.add_patch(mpatches.Rectangle((x_ok, r + 0.06), 1.0, 0.88, color=C_RSU))
        ax.set_title(title, color=INK, fontsize=10.5, loc="left")
    # rows are lined up to the middle row's moment (the same time the true camera position is taken at)
    axs[1].text(6.3, 0.4, "middle row:\nstays put\n\ntop and bottom\nrows: biggest\nshift (up to\n35 pixels here)",
                fontsize=8, color=INK2, va="top")
    fig.tight_layout()
    fig.savefig(f"{OUT}/g1_unroll_concept.png", dpi=200)
    plt.close(fig)


def fig_unroll_photos():
    im = np.asarray(Image.open(f"{RES1}/check_unroll_s9_w0077.jpg").convert("L"))
    h, w = im.shape
    pw = (w - 2 * 5) // 3  # three panels separated by ~5 px white bars (12 px at 0.4 scale)
    starts = [0, (w - pw) // 2, w - pw]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
    titles = [("Normal camera", C_GS), ("Rolling shutter: bent", C_RS), ("Straightened (turning only)", C_RSU)]
    for ax, x0, (title, col) in zip(axs, starts, titles):
        ax.imshow(im[:, x0:x0 + pw], cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(col)
            s.set_linewidth(3)
        ax.set_title(title, color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/g2_unroll_photos.png", dpi=170)
    plt.close(fig)


def fig_gap_closed():
    sm = pd.read_csv(f"{RES1}/e1_summary.csv")
    a = sm[sm.metric == "ATE"]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 4.2), sharey=True)
    w = 0.36
    for ax, b in zip(axs, ["slow", "medium", "fast"]):
        for j, (var, col) in enumerate([("RSU", C_RSU), ("RSUT", C_RSUT)]):
            for i, m in enumerate(MODELS):
                r = a[(a.model == m) & (a.bin == b) & (a.variant == var)].iloc[0]
                x = i + (j - 0.5) * (w + 0.03)
                y, lo, hi = r.recovered * 100, r.recovered_ci_lo * 100, r.recovered_ci_hi * 100
                ax.bar(x, y, width=w, color=col, zorder=2)
                ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5,
                            zorder=3)
                ax.text(x, max(hi, y) + 4 if y >= 0 else min(lo, y) - 4, f"{y:.0f}%", ha="center",
                        va="bottom" if y >= 0 else "top", fontsize=7.5, color=INK)
        ax.axhline(0, color=INK2, lw=1)
        for yv, lab, ls in [(25, "STOP below 25%", ":"), (50, "GO from 50%", "--"), (100, "normal-camera level", "-")]:
            ax.axhline(yv, color=MUTED, lw=1, ls=ls, zorder=1)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["MapAnything", "VGGT", "π³"], fontsize=8.5)
        ax.set_title(BIN_NAME[b], loc="left", fontsize=11)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(-115, 180)
    for yv, lab in [(25, "STOP line 25%"), (50, "GO line 50%"), (100, "back to normal (100%)")]:
        axs[2].text(2.62, yv, " " + lab, fontsize=7.5, color=INK2, va="center", clip_on=False)
    axs[0].set_ylabel("share of the extra error removed (%)")
    fig.legend(handles=[mpatches.Patch(color=C_RSU, label="straightened for turning (E1a, the planned test)"),
                        mpatches.Patch(color=C_RSUT, label="straightened for turning + walking (E1b)")],
               loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 0.93, 1))
    fig.savefig(f"{OUT}/g3_gap_closed.png", dpi=200)
    plt.close(fig)


def fig_error_cm():
    sm = pd.read_csv(f"{RES1}/e1_summary.csv")
    a = sm[sm.metric == "ATE"]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.9))
    w = 0.2
    for ax, m in zip(axs, MODELS):
        for j, cam in enumerate(CAM):
            ys = []
            for b in ["slow", "medium", "fast"]:
                q = a[(a.model == m) & (a.bin == b)]
                if cam in ("GS", "RS"):
                    ys.append(q[f"median_{cam}"].iloc[0] * 100)
                else:
                    ys.append(q[q.variant == cam].median_unrolled.iloc[0] * 100)
            ax.bar(np.arange(3) + (j - 1.5) * (w + 0.015), ys, width=w, color=CAM_COL[cam], label=CAM_NAME[cam],
                   zorder=2)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["slow", "medium", "fast"])
        ax.set_title(NAME[m], loc="left")
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("typical path error (cm)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Orange (bent) is always highest; once straightened (aqua, yellow) the error drops back to blue "
                 "(normal). Each panel has its own scale.", fontsize=9.5, color=INK2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(f"{OUT}/g4_error_cm.png", dpi=200)
    plt.close(fig)


def fig_gap_vs_speed():
    df = pd.read_csv(f"{RES1}/per_window_e1.csv")
    rng = np.random.default_rng(1)
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.9), sharey=True)
    for ax, m in zip(axs, MODELS):
        P = df[df.model == m].pivot_table(index="key", columns="camera", values="ATE")
        om = df[(df.model == m) & (df.camera == "GS")].set_index("key").omega_mean.reindex(P.index) * 180 / np.pi
        edges = np.quantile(om, np.linspace(0, 1, 7))
        edges[-1] += 1e-6
        for var, mk in [("RSU", "o"), ("RSUT", "s")]:
            xs, ys, lo, hi = [], [], [], []
            for a_, b_ in zip(edges[:-1], edges[1:]):
                sel = (om >= a_) & (om < b_)
                Q = P[sel]
                xs.append(om[sel].median())
                r = recovered(Q.GS.values, Q.RS.values, Q[var].values) * 100
                idx = rng.integers(0, len(Q), (300, len(Q)))
                bt = [recovered(Q.GS.values[i], Q.RS.values[i], Q[var].values[i]) * 100 for i in idx]
                l_, h_ = np.nanpercentile(bt, [2.5, 97.5])
                ys.append(r)
                lo.append(r - l_)
                hi.append(h_ - r)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="-" + mk, color=CAM_COL[var], lw=2, ms=6, mec="white", mew=1.2,
                        capsize=2.5, elinewidth=1, label=CAM_NAME[var])
        for yv, ls in [(0, "-"), (25, ":"), (50, "--"), (100, "-")]:
            ax.axhline(yv, color=MUTED, lw=1, ls=ls, zorder=0)
        ax.set_ylim(-60, 160)
        ax.set_title(NAME[m] + (" (small gap, so noisy)" if m == "mapanything" else ""), loc="left", fontsize=11)
        ax.set_xlabel("how fast the camera turns (°/s)")
    axs[0].set_ylabel("share of extra error removed (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("The fix works at every speed, not only for fast motion. Dotted = 25% (STOP), dashed = 50% (GO), "
                 "top line = 100% (back to normal).", fontsize=9.5, color=INK2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    fig.savefig(f"{OUT}/g5_gap_vs_speed.png", dpi=200)
    plt.close(fig)


def fig_fastest_clip_e1(key="s9_w0077"):
    df = pd.read_csv(f"{RES1}/per_window_e1.csv")
    d = df[df.key == key].pivot_table(index="model", columns="camera", values="ATE") * 100
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 2.9))
    for ax, m in zip(axs, MODELS):
        vals = [d.loc[m, c] for c in CAM]
        ys = np.arange(4)[::-1]
        ax.barh(ys, vals, color=[CAM_COL[c] for c in CAM], height=0.68, zorder=2)
        for y, v in zip(ys, vals):
            ax.text(v + max(vals) * 0.02, y, f"{v:.1f} cm", va="center", fontsize=9, color=INK)
        ax.set_yticks(ys)
        ax.set_yticklabels([CAM_NAME[c] for c in CAM] if m == MODELS[0] else [""] * 4, fontsize=9)
        ax.set_xlim(0, max(vals) * 1.3)
        ax.set_title(NAME[m], loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("path error (cm)")
    fig.suptitle("The fastest 2-second clip (about 200°/s): path error on each version of the same 16 photos",
                 fontsize=9.5, color=INK2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(f"{OUT}/g6_fastest_clip_e1.png", dpi=200)
    plt.close(fig)


def fig_roadmap():
    fig, ax = plt.subplots(figsize=(9, 8.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    DONE, NEXT, LATER = ("#e8f4ee", "#1a7f37"), ("#fdf3e0", "#c27c00"), ("#f4f4f1", MUTED)

    def node(x, y, w, h, title, text, style, dashed=False):
        fc, ec = style
        ax.add_patch(mpatches.FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                             fc=fc, ec=ec, lw=1.6, ls="--" if dashed else "-"))
        ax.text(x, y + h / 2 - 2.2, title, ha="center", va="top", fontsize=10, fontweight="bold", color=INK)
        ax.text(x, y + h / 2 - 6.4, text, ha="center", va="top", fontsize=8.6, color=INK2, linespacing=1.35)

    def arrow(x1, y1, x2, y2, label=None, lx=0, ly=0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.3))
        if label:
            ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=8.3, color=INK, ha="center",
                    va="center", bbox=dict(fc="white", ec="none", pad=1.5))

    node(40, 93, 46, 10, "EXP0 · gap test · DONE: GRAY", "The problem is real: π³ up to 2.6× worse", DONE)
    node(40, 77, 46, 10, "EXP1 · fix with true motion · DONE: GO", "Straightening removes 82–100% of the extra error",
         DONE)
    node(84, 77, 26, 14, "Prior-work check", "now, in parallel:\nhas anyone done this?\ngyroscope fixes, etc.", LATER,
         dashed=True)
    node(40, 59, 52, 13, "EXP2 · NEXT", "Can the model fix itself from its own output?\nNo sensors, no training. "
                                        "~3–5 h (projected)", NEXT)
    node(18, 37, 32, 13, "Paper A", "training-free\n“two passes fix rolling shutter”", LATER)
    node(66, 37, 36, 13, "EXP3", "train a small motion head + adapter\non synthetic bent videos", LATER)
    node(54, 20, 26, 11, "Paper B", "rolling-shutter-aware\n3D model", LATER)
    node(86, 20, 22, 11, "Fallback", "workshop paper", LATER)
    node(30, 5, 44, 9, "EXP4 + writing", "full benchmark, comparisons, teaser figure", LATER)

    arrow(40, 88, 40, 82.5)
    arrow(63.5, 77, 70.5, 77)
    arrow(40, 72, 40, 65.8)
    arrow(30, 52.3, 20, 43.8, "close to EXP1\n(≥ 70% of its gain)", lx=-6, ly=1)
    arrow(52, 52.3, 62, 43.8, "far below\n(< 40% of its gain)", lx=8, ly=1)
    arrow(62, 30.3, 56, 25.8, "beats adapter-only", lx=-10)
    arrow(74, 30.3, 84, 25.8, "no better", lx=6)
    arrow(18, 30.3, 22, 10.3)
    arrow(47, 14.3, 41, 10.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/g7_roadmap.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_unroll_concept()
    fig_unroll_photos()
    fig_gap_closed()
    fig_error_cm()
    fig_gap_vs_speed()
    fig_fastest_clip_e1()
    fig_roadmap()
    print(sorted(os.listdir(OUT)))
