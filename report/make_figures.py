"""Plain-language figures for the EXP0 team report (all data from results/exp0, measured)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "exp0")
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)

# dataviz reference palette: slot 1 blue = normal camera, slot 2 orange = rolling-shutter camera
C_GS, C_RS = "#2a78d6", "#eb6834"
SPEED_RAMP = ["#9ec5f4", "#5598e7", "#1c5cab"]  # sequential blue for slow / medium / fast
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"
MODELS = ["mapanything", "vggt", "pi3"]
NAME = dict(mapanything="MapAnything", vggt="VGGT", pi3="π³ (Pi3)")
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11, "legend.frameon": False,
    "axes.titlesize": 12, "axes.titleweight": "bold",
})


def fig_rolling_shutter_sketch():
    """A vertical pole photographed while the camera turns: all-at-once vs row-by-row."""
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    rows = 12
    for ax, rs, title, col in [(axs[0], False, "Normal camera: whole photo at once", C_GS),
                               (axs[1], True, "Rolling-shutter camera: one row at a time", C_RS)]:
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
            x = (3.4 + 0.17 * r) if rs else 4.2
            ax.add_patch(mpatches.Rectangle((x, r + 0.04), 1.0, 0.92, color=col))
            ax.axhline(r, color=GRID, lw=0.6)
        ax.set_title(title, color=INK, fontsize=11)
        if rs:
            ax.annotate("", xy=(9.3, rows - 0.5), xytext=(9.3, 0.5),
                        arrowprops=dict(arrowstyle="->", color=INK2, lw=1.5))
            ax.text(9.0, 0.6, "row 1\n0 ms", ha="right", va="top", fontsize=8, color=INK2)
            ax.text(9.0, rows - 0.6, "last row\n30 ms later", ha="right", va="bottom", fontsize=8, color=INK2)
            ax.text(0.3, rows - 0.6, "the camera kept turning,\nso a straight pole\nlooks slanted", fontsize=9,
                    color=INK2, va="bottom")
        else:
            ax.text(0.3, rows - 0.6, "every row taken\nat the same instant:\nthe pole stays straight", fontsize=9,
                    color=INK2, va="bottom")
    fig.tight_layout()
    fig.savefig(f"{OUT}/f1_sketch.png", dpi=200)
    plt.close(fig)


def fig_photo_pair():
    im = np.asarray(Image.open(f"{RES}/check_fastest_pair.jpg").convert("L"))
    h, w = im.shape
    half = (w - 8) // 2
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.9))
    for ax, sl, title, col in [(axs[0], im[:, :half], "Normal camera (left)", C_GS),
                               (axs[1], im[:, -half:], "Rolling-shutter camera (right)", C_RS)]:
        ax.imshow(sl, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(col)
            s.set_linewidth(3)
        ax.set_title(title, color=INK)
    fig.suptitle("Same instant, camera turning fast (about 200° per second). The two cameras sit 11 cm apart.",
                 fontsize=10, color=INK2, y=0.03, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(f"{OUT}/f2_photos.png", dpi=170)
    plt.close(fig)


def fig_how_much_worse():
    bs = pd.read_csv(f"{RES}/bins_summary.csv")
    a = bs[(bs.metric == "ATE") & bs.bin.isin(["slow", "medium", "fast"])]
    fig, ax = plt.subplots(figsize=(9, 4.3))
    w = 0.26
    labels = ["slow turning (under 30°/s)", "medium (30–85°/s)", "fast (over 85°/s)"]
    for k, (b, lab) in enumerate(zip(["slow", "medium", "fast"], labels)):
        xs, ys, lo, hi = [], [], [], []
        for i, m in enumerate(MODELS):
            r = a[(a.model == m) & (a.bin == b)].iloc[0]
            xs.append(i + (k - 1) * (w + 0.02))
            ys.append(r.ratio)
            lo.append(r.ratio - r.ratio_ci_lo)
            hi.append(r.ratio_ci_hi - r.ratio)
        ax.bar(xs, ys, width=w, color=SPEED_RAMP[k], label=lab, zorder=2)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
        for x, y, h_ in zip(xs, ys, hi):
            ax.text(x, y + h_ + 0.04, f"{y:.2f}×", ha="center", va="bottom", fontsize=9, color=INK)
    ax.axhline(1.0, color=INK2, lw=1.2, zorder=1)
    ax.axhline(1.5, color=MUTED, lw=1.2, ls="--", zorder=1)
    ax.text(2.55, 1.03, " no difference", va="bottom", fontsize=9, color=INK2)
    ax.text(2.55, 1.53, " GO line (1.5×)", va="bottom", fontsize=9, color=MUTED)
    ax.set_xticks(range(3))
    ax.set_xticklabels([NAME[m] for m in MODELS], fontsize=11)
    ax.set_xlim(-0.5, 3.15)
    ax.set_ylim(0, 3.1)
    ax.set_ylabel("rolling-shutter error ÷ normal error")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("How many times worse each model is on the rolling-shutter camera", loc="left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/f3_how_much_worse.png", dpi=200)
    plt.close(fig)


def fig_error_vs_speed():
    t = pd.read_csv(f"{RES}/ate_fine_bins.csv")
    t["deg"] = t.omega.str.extract(r"\(([\d.]+), ([\d.]+)\]").astype(float).mean(1) * 180 / np.pi
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, m in zip(axs, MODELS):
        s = t[t.model == m]
        ax.plot(s.deg, s.GS_cm, "-o", color=C_GS, lw=2, ms=6, label="normal camera")
        ax.plot(s.deg, s.RS_cm, "-o", color=C_RS, lw=2, ms=6, label="rolling-shutter camera")
        ax.fill_between(s.deg, s.GS_cm, s.RS_cm, color=C_RS, alpha=0.12, lw=0)
        ax.set_title(NAME[m], loc="left")
        ax.set_xlabel("how fast the camera turns (°/s)")
        ax.set_ylim(bottom=0)
    axs[0].set_ylabel("path error (cm)")
    axs[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Error grows with speed, and the rolling-shutter camera (orange) is always worse. "
                 "Each panel has its own scale.", fontsize=10, color=INK2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{OUT}/f4_error_vs_speed.png", dpi=200)
    plt.close(fig)


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / ((xs ** 2).sum() / len(src))
    return s, R, mu_d - s * R @ mu_s


def fig_fastest_clip(key="s9_w0077", model="pi3"):
    gt = np.load(f"{RES}/gt_poses.npz")
    pr = np.load(f"{RES}/preds/{model}.npz")
    G0 = gt[f"{key}_GS"][:, :3, 3]
    c0 = G0.mean(0)
    _, _, Vt = np.linalg.svd(G0 - c0)
    B = Vt[:2]
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.1), sharex=True, sharey=True)
    for ax, cam, col, title in [(axs[0], "GS", C_GS, "Normal camera"), (axs[1], "RS", C_RS, "Rolling-shutter camera")]:
        G = gt[f"{key}_{cam}"][:, :3, 3]
        P = pr[f"{key}_{cam}"][:, :3, 3]
        s, R, t_ = umeyama(P, G)
        Pa = (s * (R @ P.T)).T + t_
        err = np.sqrt(((Pa - G) ** 2).sum(1).mean()) * 100
        g2, p2 = (G - c0) @ B.T * 100, (Pa - c0) @ B.T * 100
        ax.plot(g2[:, 0], g2[:, 1], "-", color=INK2, lw=2.5, alpha=0.35, label="true path (motion capture)")
        ax.plot(p2[:, 0], p2[:, 1], "-o", color=col, lw=2, ms=5, label="π³'s guess")
        ax.set_aspect("equal")
        ax.set_title(f"{title}\noff by {err:.1f} cm on average", loc="left")
        ax.set_xlabel("cm")
        ax.legend(loc="lower left", fontsize=9)
    axs[0].set_ylabel("cm")
    fig.suptitle("The fastest 2-second clip (about 200°/s), flattened to a 2D view. Same moment, same model.",
                 fontsize=10, color=INK2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f"{OUT}/f5_fastest_clip.png", dpi=200)
    plt.close(fig)


def fig_fairness():
    cs = pd.read_csv(f"{RES}/control_summary.csv")
    bs = pd.read_csv(f"{RES}/bins_summary.csv")
    fig, ax = plt.subplots(figsize=(9, 3.9))
    w = 0.36
    for i, m in enumerate(MODELS):
        c = cs[(cs.model == m) & (cs.metric == "ATE") & (cs.bin == "all")].iloc[0]
        f = bs[(bs.model == m) & (bs.metric == "ATE") & (bs.bin == "fast")].iloc[0]
        for x, r, col in [(i - w / 2 - 0.01, c, MUTED), (i + w / 2 + 0.01, f, C_RS)]:
            ax.bar(x, r.ratio, width=w, color=col, zorder=2)
            ax.errorbar(x, r.ratio, yerr=[[r.ratio - r.ratio_ci_lo], [r.ratio_ci_hi - r.ratio]], fmt="none",
                        ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
            ax.text(x, r.ratio_ci_hi + 0.04, f"{r.ratio:.2f}×", ha="center", va="bottom", fontsize=9)
    ax.axhline(1.0, color=INK2, lw=1.2, zorder=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels([NAME[m] for m in MODELS])
    ax.set_ylim(0, 3.1)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("camera 2 error ÷ camera 1 error")
    ax.legend(handles=[mpatches.Patch(color=MUTED, label="fairness test: BOTH cameras take normal photos"),
                       mpatches.Patch(color=C_RS, label="real test: camera 2 in rolling-shutter mode, fast turning")],
              loc="upper left", fontsize=9)
    ax.set_title("Is camera 2 just a worse camera? Mostly no.", loc="left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/f6_fairness.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_rolling_shutter_sketch()
    fig_photo_pair()
    fig_how_much_worse()
    fig_error_vs_speed()
    fig_fastest_clip()
    fig_fairness()
    print(sorted(os.listdir(OUT)))
