"""Side-by-side check image RS | straightened | GS (+ 4x |straightened - RS|) for one frame of one window.

usage: python check_image.py --conds U2D-jamnet-fastec[,U2D-...] [--key s5_w0123] [--k 8] [--out <png>]
Without --key, the fastest kept window (omega_mean) that has all requested conditions is used.
GS is cam0 (a different camera, ~stereo baseline away), so it is only a visual reference for straight lines.
"""
import argparse
import glob
import os

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/rs3d")
WIN = f"{ROOT}/data/windows"
OUT = f"{ROOT}/results/exp2/baselines/unroll2d/check"


def label(im, txt):
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if im.ndim == 2 else im
    cv2.putText(im, txt, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
    return im


def grid_lines(im, step=128):
    im = im.copy()
    for y in range(step, im.shape[0], step):
        cv2.line(im, (0, y), (im.shape[1] - 1, y), (0, 255, 0), 1)
    for x in range(step, im.shape[1], step):
        cv2.line(im, (x, 0), (x, im.shape[0] - 1), (0, 255, 0), 1)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", required=True)
    ap.add_argument("--key", default="")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default="")
    ap.add_argument("--half", action="store_true", help="downscale panels by 2")
    a = ap.parse_args()
    conds = a.conds.split(",")
    win = pd.read_csv(f"{ROOT}/results/exp2/gap/tum/windows.csv")
    win = win[win.keep].sort_values("omega_mean", ascending=False)
    if a.key:
        r = win[win.key == a.key].iloc[0]
    else:
        r = next(r for r in win.itertuples()
                 if all(os.path.isdir(f"{WIN}/seq{r.seq}/{r.window_id:04d}_{c}") for c in conds))
    d = f"{WIN}/seq{r.seq}/{r.window_id:04d}"
    rs_p = sorted(glob.glob(f"{d}_RS/*.png"))[a.k]
    gs_p = sorted(glob.glob(f"{d}_GS/*.png"))[a.k]
    rs = cv2.imread(rs_p, cv2.IMREAD_GRAYSCALE)
    gs = cv2.imread(gs_p, cv2.IMREAD_GRAYSCALE)
    rows = []
    for c in conds:
        u = cv2.imread(f"{d}_{c}/{os.path.basename(rs_p)}", cv2.IMREAD_GRAYSCALE)
        diff = np.clip(4 * np.abs(u.astype(np.int16) - rs.astype(np.int16)), 0, 255).astype(np.uint8)
        panels = [label(grid_lines(cv2.cvtColor(rs, cv2.COLOR_GRAY2BGR)), "RS (cam1)"),
                  label(grid_lines(cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)), c),
                  label(grid_lines(cv2.cvtColor(gs, cv2.COLOR_GRAY2BGR)), "GS (cam0, other camera)"),
                  label(diff, "4x |U2D - RS|")]
        row = np.concatenate(panels, 1)
        rows.append(row)
    img = np.concatenate(rows, 0)
    if a.half:
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    os.makedirs(OUT, exist_ok=True)
    out = a.out or f"{OUT}/{r.key}_k{a.k:02d}_{'_'.join(conds)}.png"
    cv2.imwrite(out, img)
    print(out, f"omega_mean={r.omega_mean:.2f} rad/s rs_shift_px={r.rs_shift_px:.1f}", flush=True)


if __name__ == "__main__":
    main()
