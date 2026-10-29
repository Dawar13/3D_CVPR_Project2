"""EXP2 real-data windows beyond TUM, in EXP0's format (16 frames ~150 ms apart, static filter < 0.20 m of travel).

  advio  ADVIO iPhone 6s video (rolling shutter, readout unknown: the calibration's line delay is 0), 60 fps,
         stride 9 (150 ms). GT: ground-truth/pose.csv, documented as the camera pose at 100 Hz (INS + manual fixes).
         Angular speed for the speed axis from the iPhone gyroscope (|w| does not depend on the device frame).
         Camera key "RS".
  eth3d  ETH3D SLAM training sequences, RGB camera (global shutter), ~27 Hz, stride 4 (148 ms). GT: groundtruth.txt
         (TUM format). Angular speed from the GT orientation. Camera key "GS" (the no-harm check).
Frame-to-GT timing: GT is interpolated (linear position, slerp rotation) at each frame's own timestamp. For ADVIO the
timestamp's position within the RS readout is not documented (UNVERIFIED); for ETH3D (global shutter) it is the
frame time. Window starts every `step` frames; a window is kept if every frame has GT within 20 ms on both sides
and the GT camera path is >= 0.20 m.

Writes  results/exp2/gap/<ds>/windows.csv, gt_poses.npz (keys <key>_<cam>), clips_<cam>.csv, prep_info.json
        data/windows_<ds>/<seq>/<wid:04d>_<cam>/<k:02d>_<frame>.png  (symlinks; ADVIO frames are decoded once to
        data/frames_advio/<seq>/<idx:05d>.png)
usage: python real_windows.py advio|eth3d [--check_quat]
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, os.path.dirname(__file__))
from common import MIN_PATH_M, N_FRAMES, ROOT

CFG = dict(
    advio=dict(cam="RS", stride=9, step=72, fps=60.0),  # windows every 1.2 s
    eth3d=dict(cam="GS", stride=4, step=11, fps=27.1),  # windows every ~0.4 s, as EXP0
)
ADVIO_QUAT = "wxyz"  # pose.csv quaternion order; decided by --check_quat (GT angular speed vs gyro)


class PoseTrack:
    def __init__(self, t, p, R, max_gap=0.02):
        o = np.argsort(t)
        self.t, self.p = t[o], p[o]
        self.slerp = Slerp(self.t, Rotation.from_matrix(R[o]))
        self.max_gap = max_gap

    def valid(self, tq):
        i = np.searchsorted(self.t, tq)
        ok = (i > 0) & (i < len(self.t))
        i = np.clip(i, 1, len(self.t) - 1)
        return ok & (tq - self.t[i - 1] <= self.max_gap) & (self.t[i] - tq <= self.max_gap)

    def __call__(self, tq):
        tq = np.clip(tq, self.t[0], self.t[-1])
        T = np.tile(np.eye(4), (len(tq), 1, 1))
        T[:, :3, :3] = self.slerp(tq).as_matrix()
        T[:, :3, 3] = np.stack([np.interp(tq, self.t, self.p[:, k]) for k in range(3)], -1)
        return T

    def omega(self, t0, t1, n=200):
        """Mean |angular velocity| (rad/s) over [t0, t1] from the orientation track."""
        ts = np.linspace(t0, t1, n)
        R = self.slerp(ts)
        return float(np.mean((R[1:] * R[:-1].inv()).magnitude() / np.diff(ts)))


def advio_track(d, quat=ADVIO_QUAT):
    a = np.loadtxt(f"{d}/ground-truth/pose.csv", delimiter=",")
    q = a[:, 4:8] if quat == "xyzw" else a[:, [5, 6, 7, 4]]
    return PoseTrack(a[:, 0], a[:, 1:4], Rotation.from_quat(q).as_matrix())


def eth3d_track(d):
    a = np.loadtxt(f"{d}/groundtruth.txt", comments="#")
    return PoseTrack(a[:, 0], a[:, 1:4], Rotation.from_quat(a[:, 4:8]).as_matrix())


def check_quat(root):
    """Which quaternion order makes the GT angular speed agree with the gyroscope? Prints both correlations."""
    for d in sorted(glob.glob(f"{root}/advio-*"))[:4]:
        g = np.loadtxt(f"{d}/iphone/gyro.csv", delimiter=",")
        for quat in ("wxyz", "xyzw"):
            tr = advio_track(d, quat)
            ts = np.arange(tr.t[0] + 1, tr.t[-1] - 1, 0.5)
            w_gt = np.array([tr.omega(t - 0.25, t + 0.25, 50) for t in ts])
            w_gy = np.array([np.linalg.norm(g[(g[:, 0] > t - 0.25) & (g[:, 0] < t + 0.25), 1:4], axis=1).mean()
                             for t in ts])
            print(os.path.basename(d), quat, "corr(|w| GT, |w| gyro) = %.3f" % np.corrcoef(w_gt, w_gy)[0, 1],
                  "median ratio %.2f" % np.median(w_gt / np.maximum(w_gy, 1e-3)))


def extract_advio(d, idx_needed, out):
    """Decode the needed frames of iphone/frames.mov (k-th decoded frame = k-th row of frames.csv)."""
    os.makedirs(out, exist_ok=True)
    need = set(int(i) for i in idx_needed if not os.path.exists(f"{out}/{int(i):05d}.png"))
    if not need:
        return None
    cap = cv2.VideoCapture(f"{d}/iphone/frames.mov")
    k, shape = 0, None
    while need:
        ok, im = cap.read()
        if not ok:
            break
        if k in need:  # write then rename, so an interrupted run never leaves a truncated frame behind
            cv2.imwrite(f"{out}/{k:05d}.tmp.png", im, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            os.replace(f"{out}/{k:05d}.tmp.png", f"{out}/{k:05d}.png")
            need.discard(k)
            shape = im.shape
        k += 1
    cap.release()
    assert not need, f"{d}: {len(need)} frames not decoded"
    return shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds", choices=list(CFG))
    ap.add_argument("--check_quat", action="store_true")
    a = ap.parse_args()
    c = CFG[a.ds]
    raw = f"{ROOT}/data/advio" if a.ds == "advio" else f"{ROOT}/data/eth3d_slam/training"
    if a.check_quat:
        check_quat(raw)
        return
    res = f"{ROOT}/results/exp2/gap/{a.ds}"
    win_root = f"{ROOT}/data/windows_{a.ds}"
    os.makedirs(res, exist_ok=True)
    rows, gt, shapes = [], {}, {}
    span = (N_FRAMES - 1) * c["stride"]
    for d in sorted(glob.glob(f"{raw}/*")):
        if not os.path.isdir(d):
            continue
        seq = os.path.basename(d)
        if a.ds == "advio":
            fr = np.loadtxt(f"{d}/iphone/frames.csv", delimiter=",")
            ft = fr[:, 0]
            names = [f"{k:05d}" for k in range(len(ft))]
            tr = advio_track(d)
            gy = np.loadtxt(f"{d}/iphone/gyro.csv", delimiter=",")
        else:
            if not os.path.exists(f"{d}/groundtruth.txt"):
                continue
            fr = pd.read_csv(f"{d}/rgb.txt", sep=" ", header=None, comment="#", names=["t", "path"])
            ft = fr.t.values
            names = [os.path.splitext(os.path.basename(p))[0] for p in fr.path]
            tr = eth3d_track(d)
        sel = []
        for wid, st in enumerate(range(0, len(ft) - span, c["step"])):
            idx = st + c["stride"] * np.arange(N_FRAMES)
            t = ft[idx]
            T = tr(t)
            path = float(np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1).sum())
            ok_gt = bool(tr.valid(t).all())
            if a.ds == "advio":
                m = (gy[:, 0] >= t[0]) & (gy[:, 0] <= t[-1])
                om = float(np.linalg.norm(gy[m, 1:4], axis=1).mean()) if m.any() else np.nan
            else:
                om = tr.omega(t[0], t[-1]) if ok_gt else np.nan
            key = f"{seq}_w{wid:04d}"
            keep = ok_gt and path >= MIN_PATH_M and np.isfinite(om)
            rows.append(dict(seq=seq, window_id=wid, key=key, start_frame=st, t0=t[0], keep=keep, gt_valid=ok_gt,
                             gt_path_m=path, omega_mean=om, v_mean=path / (t[-1] - t[0]), rs_shift_px=np.nan))
            if keep:
                gt[f"{key}_{c['cam']}"] = T
                sel.append((wid, idx))
        if a.ds == "advio" and sel:
            shapes[seq] = extract_advio(d, np.concatenate([i for _, i in sel]), f"{ROOT}/data/frames_advio/{seq}")
        for wid, idx in sel:
            wd = f"{win_root}/{seq}/{wid:04d}_{c['cam']}"
            os.makedirs(wd, exist_ok=True)
            for k, i in enumerate(idx):
                src = (f"{ROOT}/data/frames_advio/{seq}/{int(i):05d}.png" if a.ds == "advio"
                       else f"{d}/rgb/{names[i]}.png")
                lnk = f"{wd}/{k:02d}_{names[i]}.png"
                if not os.path.lexists(lnk):
                    os.symlink(src, lnk)
        print(seq, len(sel), "kept windows", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{res}/windows.csv", index=False)
    np.savez(f"{res}/gt_poses.npz", **gt)
    k = df[df.keep]
    pd.DataFrame(dict(key=[f"{x}_{c['cam']}" for x in k.key],
                      dir=[f"{win_root}/{s}/{w:04d}_{c['cam']}" for s, w in zip(k.seq, k.window_id)])
                 ).to_csv(f"{res}/clips_{c['cam']}.csv", index=False)
    info = dict(c, n_frames=N_FRAMES, min_path_m=MIN_PATH_M, quat=ADVIO_QUAT if a.ds == "advio" else "xyzw",
                n_windows=len(df), n_kept=int(df.keep.sum()), n_seqs=int(df.seq.nunique()),
                frame_shapes={s: list(v) for s, v in shapes.items() if v is not None})
    with open(f"{res}/prep_info.json", "w") as f:
        json.dump(info, f, indent=1)
    print("TOTAL windows", len(df), "kept", int(df.keep.sum()), flush=True)


if __name__ == "__main__":
    main()
