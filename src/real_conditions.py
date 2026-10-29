"""EXP2 straightened conditions on ADVIO and ETH3D windows (built on the GPU with gpu_unroll.py; pinhole frames).

  ADVIO (real phone RS; readout and the sideways scan direction are both unknown, DEVIATIONS D9):
    GYRO_r<ms>_<dir>      gyroscope straightening (brief §8 item 5): w = R_cam_imu (gyro - bias) at each frame time,
                          assuming readout <ms> and direction lr or rl. Swept over a grid because neither is known:
                          this is a sensitivity analysis, not a tuned method.
    SE2P_<m>_r30_<dir>    training-free self-estimate with an a-priori 30 ms readout (both directions reported)
  ETH3D (global shutter; no-harm check, brief §9 item 11):
    SE2P_<m>_r30          the same self-estimate applied to GS frames, assuming a 30 ms top->bottom readout as a
                          fixed-readout method would
Writes data/windows_<ds>/<seq>/<wid:04d>_<cond>/<same names as the input window>, and clips_<cond>.csv.

usage: python real_conditions.py advio GYRO_r20_lr   |   python real_conditions.py eth3d SE2P_pi3_r30
"""
import argparse
import glob
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(__file__))
import gpu_unroll as G
from common import N_FRAMES, ROOT
from real_windows import CFG
from tum_conditions import twist_from_poses

DEV = "cuda" if torch.cuda.is_available() else "cpu"
# ADVIO calibration batches (github.com/AaltoVision/ADVIO/tree/master/calibration): sequences -> iphone-0x.yaml
ADVIO_CALIB = [(range(1, 13), "iphone-02"), (range(13, 18), "iphone-03"), (range(18, 20), "iphone-01"),
               (range(20, 24), "iphone-04")]
ADVIO_GYRO_BIAS = [(range(1, 13), (-0.0067, 0.0070, -0.0065)), (range(13, 18), (-0.0065, 0.0055, -0.0064)),
                   (range(18, 20), (-0.0069, 0.0062, -0.0067)), (range(20, 24), (-0.0066, 0.0065, -0.0063))]
CALIB_URL = "https://raw.githubusercontent.com/AaltoVision/ADVIO/master/calibration/{}.yaml"


def advio_calib(seq):
    n = int(seq.split("-")[1])
    name = next(f for r, f in ADVIO_CALIB if n in r)
    bias = np.array(next(b for r, b in ADVIO_GYRO_BIAS if n in r))
    path = f"{ROOT}/data/advio/calibration/{name}.yaml"
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(CALIB_URL.format(name), path)
    with open(path) as f:
        c = yaml.safe_load(f)["cameras"][0]["camera"]
    fx, fy, cx, cy = c["intrinsics"]["data"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    T = np.array(c["T_cam_imu"]["data"], dtype=float)
    return K, T[:3, :3], bias


def eth3d_K(d):
    fx, fy, cx, cy = np.loadtxt(f"{d}/calibration.txt")[:4]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def frame_times(ds, d):
    if ds == "advio":
        return np.loadtxt(f"{d}/iphone/frames.csv", delimiter=",")[:, 0]
    return pd.read_csv(f"{d}/rgb.txt", sep=" ", header=None, comment="#", names=["t", "p"]).t.values


def unroll(imgs, K, W, H, readout, direction, w, v=None, depth=None):
    L = W if direction in ("lr", "rl") else H
    scan = G.ScanModel(readout / L, H, W, direction)
    un = G.Unroller(K, (H, W), lambda X: G.pinhole_project(X, K), scan, device=DEV)
    mu, mv = un.maps(np.asarray(w), None if v is None else np.asarray(v), depth)
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).to(DEV).float()
    out = G.Unroller.remap(x, mu.float(), mv.float())
    return (out + 0.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds", choices=["advio", "eth3d"])
    ap.add_argument("cond")
    ap.add_argument("--thin", type=int, default=1, help="every k-th kept window per sequence (DEVIATIONS D14)")
    a = ap.parse_args()
    c = CFG[a.ds]
    raw = f"{ROOT}/data/advio" if a.ds == "advio" else f"{ROOT}/data/eth3d_slam/training"
    res = f"{ROOT}/results/exp2/gap/{a.ds}"
    win_root = f"{ROOT}/data/windows_{a.ds}"
    cam = c["cam"]
    m_g = re.fullmatch(r"GYRO_r(\d+)_(lr|rl)", a.cond)
    m_s = re.fullmatch(r"SE2P_(\w+?)_r(\d+)(?:_(lr|rl|tb))?", a.cond)
    assert m_g or m_s, a.cond
    readout = int((m_g or m_s).group(2 if m_s else 1)) / 1e3
    direction = (m_g.group(2) if m_g else (m_s.group(3) or "tb"))
    P1 = np.load(f"{res}/preds/{m_s.group(1)}.npz") if m_s else None
    win = pd.read_csv(f"{res}/windows.csv")
    win = win[win.keep].sort_values(["seq", "window_id"])
    win = win[win.groupby("seq").cumcount() % a.thin == 0]
    io = ThreadPoolExecutor(8)
    rows = []
    for seq, ws in win.groupby("seq"):
        d = f"{raw}/{seq}"
        ft = frame_times(a.ds, d)
        if a.ds == "advio":
            K, R_ci, bias = advio_calib(seq)
            gy = np.loadtxt(f"{d}/iphone/gyro.csv", delimiter=",")
        else:
            K = eth3d_K(d)
        for r in ws.itertuples():
            src = f"{win_root}/{seq}/{r.window_id:04d}_{cam}"
            out = f"{win_root}/{seq}/{r.window_id:04d}_{a.cond}"
            names = sorted(os.listdir(src))
            rows.append(dict(key=f"{r.key}_{a.cond}", dir=out))
            if os.path.isdir(out) and len(os.listdir(out)) == N_FRAMES:
                continue
            idx = r.start_frame + c["stride"] * np.arange(N_FRAMES)
            t = ft[idx]
            if m_g:
                g = np.stack([np.interp(t, gy[:, 0], gy[:, k]) for k in (1, 2, 3)], -1) - bias
                w = g @ R_ci.T  # camera frame
            else:
                w, _ = twist_from_poses(P1[f"{r.key}_{cam}"], t - t[0])
            imgs = list(io.map(lambda n: cv2.imread(f"{src}/{n}", cv2.IMREAD_COLOR)[..., ::-1], names))
            H, W = imgs[0].shape[:2]
            ims = unroll(imgs, K, W, H, readout, direction, w)
            os.makedirs(out, exist_ok=True)
            list(io.map(lambda kv: cv2.imwrite(f"{out}/{kv[0]}", kv[1][..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1]),
                        zip(names, ims)))
        print(seq, "done", a.cond, flush=True)
    pd.DataFrame(rows).to_csv(f"{res}/clips_{a.cond}.csv", index=False)
    print("REAL_COND_DONE", a.cond, len(rows), flush=True)


if __name__ == "__main__":
    main()
