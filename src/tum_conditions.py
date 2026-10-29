"""EXP2 TUM RS-VIO conditions, built on the GPU (gpu_unroll.py) from the raw 16-bit cam1 frames.

  V3               rotation only, gyro w (EXP1-a recipe)
  V4               rotation + translation: gyro w, GT v, pi3 depth scaled to metres by the GT Sim(3) (EXP1-b; P7)
  V3_r<f>, V4_r<f> the unroller assumes readout x f (f = 0.5, 0.75, 1.25, 1.5): oracle sensitivity (EXP2 §7)
  V3_bt, V4_bt     the unroller assumes a bottom->top scan (wrong direction)
  SE2P_<model>     training-free two-pass, rotation only: w from <model>'s own pass-1 poses on the RS window
  SE2PT_<model>    ... plus v from pass-1 camera centres and the model's own pass-1 depth (scale cancels in v/Z)
  SE2P_<model>_shuf  rotation only, from a pass 1 that saw the frames shuffled (unordered photo set)
  SE2P_<model>_onGS  the same method on the GS camera (cam0) frames, assuming the RS readout: no-harm check

Frame-level conditions (V*) write data/prep_<cond>/seq{s}/cam1/{t}.png plus window symlinks, as unroll.py did.
Window-level conditions (SE2P*) write each window's frames into windows/seq{s}/{wid:04d}_<cond>/, because the
estimate differs per window. The output then goes through the same tone curve -> 8-bit PNG as prep.py.

pass-1 inputs: preds/<model>.npz (RS poses), preds/<model>_RSshuf.npz (shuffled pass 1, keys <key>_RSshuf),
               preds/<model>_RSdense_dense/<key>_RS.npz (infer2 --dense output on the RS windows)
usage: python tum_conditions.py <cond> [--batch 8] [--pass1 <preds npz>] [--dense <dir>]
"""
import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(__file__))
import gpu_unroll as G
from common import DATA, H, N_FRAMES, RES, ROW_TIME, STRIDE, W, WIN_DIR, load_calib, load_frames, seq_dir
from dense_utils import pmap_of, to_orig_grid
from infer2 import frame_order
from unroll import DEPTH_DIR, frame_motion

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FPS = 20.0


def parse(cond):
    m = re.fullmatch(r"(V3|V4)(?:_r([0-9.]+))?(?:_(bt))?", cond)
    if m:
        return dict(kind=m.group(1), factor=float(m.group(2) or 1.0), direction=m.group(3) or "tb")
    m = re.fullmatch(r"(SE2PT?)_(\w+?)(_shuf)?(_onGS)?", cond)
    if m:
        return dict(kind=m.group(1), model=m.group(2), shuf=bool(m.group(3)), ongs=bool(m.group(4)), factor=1.0,
                    direction="tb")
    raise ValueError(cond)


def make_unroller(K_new, factor, direction, cam="cam1"):
    cal = load_calib()[cam]
    K, D = cal["K"].tolist(), cal["D"].tolist()
    scan = G.ScanModel(ROW_TIME * factor, H, W, direction)
    return G.Unroller(K_new, (H, W), lambda X: G.fisheye_project(X, K, D), scan, device=DEV)


def read_raw(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0


def write_u8(path, im8):
    cv2.imwrite(path, im8, [cv2.IMWRITE_PNG_COMPRESSION, 1])


def unroll_batch(un, raws, w, v=None, depth=None):
    """raws: list of (H, W) float32 raw frames -> list of uint8 straightened frames."""
    mu, mv = un.maps(np.asarray(w), None if v is None else np.asarray(v), depth)
    x = torch.from_numpy(np.stack(raws))[:, None].to(DEV)
    out = G.tone_map_u8(G.Unroller.remap(x, mu, mv)[:, 0])
    return list(out.cpu().numpy())


def run_frames(cond, p, K_new, win, batch, io):
    un = make_unroller(K_new, p["factor"], p["direction"])
    root = f"{os.path.dirname(DATA)}/prep_{cond}"
    for s in sorted(win.seq.unique()):
        ts = load_frames(s, "cam1")
        w_s = win[win.seq == s]
        used = sorted({int(ts[st + STRIDE * k]) for st in w_s.start_frame for k in range(N_FRAMES)})
        os.makedirs(f"{root}/seq{s}/cam1", exist_ok=True)
        todo = [t for t in used if not os.path.exists(f"{root}/seq{s}/cam1/{t}.png")]
        mot = frame_motion(s)
        dep = None
        if p["kind"] == "V4":
            z = np.load(f"{DEPTH_DIR}/seq{s}.npz")
            dep = {int(k): z[k].astype(np.float32) for k in z.files}
        for i in range(0, len(todo), batch):
            tb = todo[i:i + batch]
            raws = list(io.map(read_raw, [f"{seq_dir(s)}/cam1/data/{t}.png" for t in tb]))
            w = np.stack([mot[t][0] for t in tb])
            if dep is None:
                ims = unroll_batch(un, raws, w)
            else:
                ims = unroll_batch(un, raws, w, np.stack([mot[t][1] for t in tb]), np.stack([dep[t] for t in tb]))
            list(io.map(write_u8, [f"{root}/seq{s}/cam1/{t}.png" for t in tb], ims))
        for r in w_s.itertuples():
            d = f"{WIN_DIR}/seq{s}/{r.window_id:04d}_{cond}"
            os.makedirs(d, exist_ok=True)
            for k in range(N_FRAMES):
                t = int(ts[r.start_frame + STRIDE * k])
                lnk = f"{d}/{k:02d}_{t}.png"
                if not os.path.lexists(lnk):
                    os.symlink(f"{root}/seq{s}/cam1/{t}.png", lnk)
        print(f"seq{s}: {len(todo)} frames ({cond})", flush=True)


def twist_from_poses(P, t_s):
    """Camera-frame (w, v) per frame from cam-to-world poses P (N, 4, 4) at times t_s (s): central differences,
    one-sided at the ends. v is in the model's own (arbitrary) scale."""
    n = len(P)
    w, v = np.zeros((n, 3)), np.zeros((n, 3))
    for i in range(n):
        a, b = max(i - 1, 0), min(i + 1, n - 1)
        dt = t_s[b] - t_s[a]
        Om = Rotation.from_matrix(P[b, :3, :3] @ P[a, :3, :3].T).as_rotvec() / dt  # world frame
        w[i] = P[i, :3, :3].T @ Om
        v[i] = P[i, :3, :3].T @ ((P[b, :3, 3] - P[a, :3, 3]) / dt)
    return w, v


def run_windows(cond, p, K_new, win, batch, io, pass1, dense_dir):
    """Self-estimate two-pass: per window, w (and v, depth) from the model's own pass-1 output on the RS window.
    Shuffled: pass 1 saw the frames in infer2's fixed random order, and the finite differences are taken in that
    order with a uniform frame gap, since an unordered photo set has no timestamps."""
    cam = "cam0" if p.get("ongs") else "cam1"  # _onGS: the same method given the GS camera's frames (no-harm)
    un = make_unroller(K_new, 1.0, "tb", cam)
    preds = np.load(pass1)
    for s in sorted(win.seq.unique()):
        ts = load_frames(s, cam)
        for r in win[win.seq == s].itertuples():
            d = f"{WIN_DIR}/seq{s}/{r.window_id:04d}_{cond}"
            if p.get("ongs"):  # GS frames: take the stamps from the GS window itself
                tk = [int(os.path.basename(f)[3:-4]) for f in sorted(glob.glob(f"{WIN_DIR}/seq{s}/{r.window_id:04d}_GS/*.png"))]
            else:
                tk = [int(ts[r.start_frame + STRIDE * k]) for k in range(N_FRAMES)]
            if len(glob.glob(f"{d}/*.png")) == N_FRAMES:
                continue
            os.makedirs(d, exist_ok=True)
            P = preds[f"{r.key}_GS" if p.get("ongs") else (f"{r.key}_RSshuf" if p["shuf"] else f"{r.key}_RS")]
            if p["shuf"]:
                order = frame_order(f"{r.key}_RSshuf", N_FRAMES, "shuffled")  # the order pass 1 saw
                ws, vs = twist_from_poses(P[order], np.arange(N_FRAMES) * STRIDE / FPS)
                w, v = np.empty_like(ws), np.empty_like(vs)
                w[order], v[order] = ws, vs
            else:
                w, v = twist_from_poses(P, (np.array(tk) - tk[0]) * 1e-9)
            z = None
            if p["kind"] == "SE2PT":  # the model's own depth (model units, like v), on the image grid
                dz = np.load(f"{dense_dir}/{r.key}_RS.npz")
                z = to_orig_grid(dz["local_pts"][..., 2].astype(np.float32), pmap_of(dz))
                assert not p.get("ongs"), "SE2PT_onGS needs GS dense pass-1 output (not produced)"
            raws = list(io.map(read_raw, [f"{seq_dir(s)}/{cam}/data/{t}.png" for t in tk]))
            ims = []
            for i in range(0, N_FRAMES, batch):
                sl = slice(i, i + batch)
                if z is None:
                    ims += unroll_batch(un, raws[sl], w[sl])
                else:
                    ims += unroll_batch(un, raws[sl], w[sl], v[sl], z[sl])
            list(io.map(write_u8, [f"{d}/{k:02d}_{t}.png" for k, t in enumerate(tk)], ims))
        print(f"seq{s}: windows done ({cond})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cond")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--pass1", default="", help="pass-1 preds npz for SE2P* (default: RES/preds/<model>.npz)")
    ap.add_argument("--dense", default="", help="pass-1 dense dir for SE2PT* (default: RES/preds/<model>_dense)")
    a = ap.parse_args()
    p = parse(a.cond)
    with open(f"{RES}/prep_info.json") as f:
        K_new = np.array(json.load(f)["K_new"])
    win = pd.read_csv(f"{RES}/windows.csv")
    win = win[win.keep]
    with ThreadPoolExecutor(12) as io:
        if p["kind"] in ("V3", "V4"):
            run_frames(a.cond, p, K_new, win, a.batch, io)
        else:
            pass1 = f"{RES}/preds/{p['model']}_RSshuf.npz" if p["shuf"] else f"{RES}/preds/{p['model']}.npz"
            run_windows(a.cond, p, K_new, win, a.batch, io, a.pass1 or pass1,
                        a.dense or f"{RES}/preds/{p['model']}_RSdense_dense")
    print("COND_DONE", a.cond, flush=True)


if __name__ == "__main__":
    main()
