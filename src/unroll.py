"""EXP1 oracle unrolling: re-render every RS frame (cam1) as if all rows were captured at the centre-row time.

Modes
  rot      rotation only, from the gyro (depth-free; the plan's E1)          -> windows/<seq>/<wid>_RSU
  rotrans  rotation + GT linear velocity, with pi3 metric depth (E1b)      -> windows/<seq>/<wid>_RSUT

Camera model: R_wc(t) = R_wc(t_mid) exp([w_c]x dt), c(t) = c_mid + v_w dt, dt = t_row - t_mid.
A ray d_mid of the centre-row camera was seen in row-camera r as  X_r = exp(-[w_c]x dt) (Z d_mid - v_c dt).
The source row depends on the answer, so the raw row is found by fixed-point iteration.
Output goes through the same remap -> tone curve -> 8-bit PNG as prep.py, in a single interpolation.

usage: python unroll.py {rot|rotrans|check}
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import (DATA, H, READOUT, RES, ROW_TIME, RS_MID_OFFSET, STRIDE, N_FRAMES, W, WIN_DIR, GTInterp,
                    load_calib, load_frames, load_imu, seq_dir)
from prep import GAMMA

PREP_E1 = {"rot": f"{os.path.dirname(DATA)}/prep_e1rot", "rotrans": f"{os.path.dirname(DATA)}/prep_e1rotrans"}
CAM_LABEL = {"rot": "RSU", "rotrans": "RSUT"}
DEPTH_DIR = f"{os.path.dirname(DATA)}/pi3_depth"
N_ITER = 4


def fisheye_project(X, K, D):
    """Kalibr pinhole-equidistant (= OpenCV fisheye) projection of camera-frame points (..., 3) to raw pixels."""
    z = np.maximum(X[..., 2], 1e-6)
    a, b = X[..., 0] / z, X[..., 1] / z
    r = np.sqrt(a * a + b * b)
    th = np.arctan(r)
    t2 = th * th
    thd = th * (1 + D[0] * t2 + D[1] * t2 ** 2 + D[2] * t2 ** 3 + D[3] * t2 ** 4)
    sc = np.where(r > 1e-9, thd / np.maximum(r, 1e-9), 1.0)
    return (K[0, 0] * a * sc + K[0, 2]).astype(np.float32), (K[1, 1] * b * sc + K[1, 2]).astype(np.float32)


def rotate(X, axis, ang):
    """Rodrigues rotation of vectors X (H, W, 3) about a unit axis (3,) by a per-pixel angle (H, W)."""
    c, s = np.cos(ang)[..., None], np.sin(ang)[..., None]
    kxX = np.cross(np.broadcast_to(axis, X.shape), X)
    kdX = (X @ axis)[..., None]
    return X * c + kxX * s + axis * kdX * (1 - c)


class Unroller:
    def __init__(self, K_new):
        cal = load_calib()["cam1"]
        self.K, self.D = cal["K"], cal["D"]
        self.R_ci = cal["T_cam_imu"][:3, :3]
        u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
        self.K_new = np.asarray(K_new)
        self.rays = np.stack([(u - self.K_new[0, 2]) / self.K_new[0, 0], (v - self.K_new[1, 2]) / self.K_new[1, 1],
                              np.ones_like(u)], -1)  # z = 1, so Z * ray is the 3D point at z-depth Z
        self.u0, self.v0 = fisheye_project(self.rays, self.K, self.D)  # identity map (no motion)

    def maps(self, w_c, v_c=None, depth=None):
        th = np.linalg.norm(w_c)
        axis = w_c / th if th > 1e-12 else np.array([0.0, 0.0, 1.0])
        v_raw = self.v0
        X_mid = self.rays if depth is None else None
        for _ in range(N_ITER):
            dt = v_raw.astype(np.float64) * ROW_TIME - RS_MID_OFFSET
            if depth is None:
                X_r = rotate(X_mid, axis, -th * dt)
            else:
                # depth (metric z) is sampled where the source pixel currently lands in the RS pinhole image
                if X_mid is None:
                    xs, ys = self.rays[..., 0], self.rays[..., 1]
                else:
                    xs, ys = X_r[..., 0] / X_r[..., 2], X_r[..., 1] / X_r[..., 2]
                pu = (xs * self.K_new[0, 0] + self.K_new[0, 2]).astype(np.float32)
                pv = (ys * self.K_new[1, 1] + self.K_new[1, 2]).astype(np.float32)
                Z = cv2.remap(depth, pu, pv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float64)
                X_mid = self.rays * Z[..., None]
                X_r = rotate(X_mid - v_c * dt[..., None], axis, -th * dt)
            u_raw, v_raw = fisheye_project(X_r, self.K, self.D)
        return u_raw, v_raw


def frame_motion(s):
    """Per cam1 frame: camera-frame angular velocity (gyro mean over the readout) and linear velocity (GT)."""
    calib = load_calib()
    R_ci = calib["cam1"]["T_cam_imu"][:3, :3]
    T_imu_cam = np.linalg.inv(calib["cam1"]["T_cam_imu"])
    ts = load_frames(s, "cam1")
    imu_t, gyro = load_imu(s)
    gt = GTInterp(s)
    out = {}
    d = int(0.025e9)
    for t in ts:
        m = (imu_t >= t) & (imu_t <= t + int(READOUT * 1e9))
        w_c = R_ci @ gyro[m].mean(0) if m.any() else np.zeros(3)
        t_mid = t + int(round(RS_MID_OFFSET * 1e9))
        if gt.valid(np.array([t_mid - d, t_mid + d])).all():
            Tm, Tp, T0 = gt(np.array([t_mid - d, t_mid + d, t_mid])) @ T_imu_cam
            v_w = (Tp[:3, 3] - Tm[:3, 3]) / 0.05
            v_c = T0[:3, :3].T @ v_w
        else:
            v_c = np.zeros(3)
        out[int(t)] = (w_c, v_c)
    return out


_U = {}


def _init(K_new, mode):
    _U["u"] = Unroller(K_new)
    _U["mode"] = mode


def _do(job):
    s, t, w_c, v_c, depth = job
    dst = f"{PREP_E1[_U['mode']]}/seq{s}/cam1/{t}.png"
    if os.path.exists(dst):
        return
    if depth is not None:  # small pi3 depth -> full resolution here, in the worker
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
    mu, mv = _U["u"].maps(w_c, v_c, depth)
    im = cv2.imread(f"{seq_dir(s)}/cam1/data/{t}.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
    # rows shifted past the sensor edge repeat the edge pixel, so no black border becomes a cue for the models
    im = cv2.remap(im, mu, mv, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    im8 = np.clip(255.0 * np.power(np.clip(im, 0, 1), GAMMA) + 0.5, 0, 255).astype(np.uint8)
    cv2.imwrite(dst, im8, [cv2.IMWRITE_PNG_COMPRESSION, 1])


def load_K_new():
    import json
    with open(f"{RES}/prep_info.json") as f:
        return np.array(json.load(f)["K_new"])


def run(mode):
    K_new = load_K_new()
    win = pd.read_csv(f"{RES}/windows.csv")
    win = win[win.keep]
    lab = CAM_LABEL[mode]
    with ProcessPoolExecutor(12, initializer=_init, initargs=(K_new, mode)) as ex:
        for s in sorted(win.seq.unique()):
            ts = load_frames(s, "cam1")
            w = win[win.seq == s]
            used = sorted({int(ts[st + STRIDE * k]) for st in w.start_frame for k in range(N_FRAMES)})
            mot = frame_motion(s)
            dep = None
            if mode == "rotrans":
                z = np.load(f"{DEPTH_DIR}/seq{s}.npz")
                dep = {int(k): z[k].astype(np.float32) for k in z.files}
                missing = [t for t in used if t not in dep]
                assert not missing, f"seq{s}: {len(missing)} frames lack pi3 depth"
            os.makedirs(f"{PREP_E1[mode]}/seq{s}/cam1", exist_ok=True)
            jobs = [(s, t, *mot[t], dep[t] if dep else None) for t in used]
            list(ex.map(_do, jobs, chunksize=4))
            for r in w.itertuples():
                d = f"{WIN_DIR}/seq{s}/{r.window_id:04d}_{lab}"
                os.makedirs(d, exist_ok=True)
                for k in range(N_FRAMES):
                    t = int(ts[r.start_frame + STRIDE * k])
                    lnk = f"{d}/{k:02d}_{t}.png"
                    if not os.path.lexists(lnk):
                        os.symlink(f"{PREP_E1[mode]}/seq{s}/cam1/{t}.png", lnk)
            print(f"seq{s}: {len(used)} frames unrolled ({mode})", flush=True)
    print("UNROLL_DONE", flush=True)


def check():
    """(1) zero motion reproduces the EXP0 undistortion map; (2) gyro-predicted rotation over half a readout
    matches the mocap GT rotation (sign and frame check); (3) side-by-side image of the fastest frame."""
    K_new = load_K_new()
    u = Unroller(K_new)
    cal = load_calib()["cam1"]
    mx, my = cv2.fisheye.initUndistortRectifyMap(cal["K"], cal["D"], np.eye(3), K_new, (W, H), cv2.CV_32FC1)
    a, b = u.maps(np.zeros(3))
    print("identity map max |diff| px:", float(np.abs(a - mx).max()), float(np.abs(b - my).max()))

    from scipy.spatial.transform import Rotation
    calib = load_calib()
    T_imu_cam = np.linalg.inv(calib["cam1"]["T_cam_imu"])
    errs, mags = [], []
    for s in [9, 10]:
        mot = frame_motion(s)
        gt = GTInterp(s)
        for t, (w_c, _) in list(mot.items())[100:700:15]:
            t0 = t
            t1 = t + int(READOUT * 1e9)
            if not gt.valid(np.array([t0, t1])).all():
                continue
            A, B = gt(np.array([t0, t1])) @ T_imu_cam
            R_gt = A[:3, :3].T @ B[:3, :3]
            R_gyro = Rotation.from_rotvec(w_c * READOUT).as_matrix()
            R_wrong = Rotation.from_rotvec(-w_c * READOUT).as_matrix()
            ang = lambda R: np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
            errs.append((ang(R_gt.T @ R_gyro), ang(R_gt.T @ R_wrong)))
            mags.append(ang(R_gt))
    e = np.array(errs)
    print(f"rotation over one readout: GT median {np.median(mags):.3f} deg; gyro-vs-GT median error "
          f"{np.median(e[:, 0]):.3f} deg (opposite sign would give {np.median(e[:, 1]):.3f} deg)")

    win = pd.read_csv(f"{RES}/windows.csv")
    r = win[win.keep].sort_values("omega_mean").iloc[-1]
    ts = load_frames(int(r.seq), "cam1")
    t = int(ts[r.start_frame + STRIDE * 7])
    w_c, _ = frame_motion(int(r.seq))[t]
    mu, mv = u.maps(w_c)
    raw = cv2.imread(f"{seq_dir(int(r.seq))}/cam1/data/{t}.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
    tm = lambda im: np.clip(255.0 * np.power(np.clip(im, 0, 1), GAMMA) + 0.5, 0, 255).astype(np.uint8)
    rs = tm(cv2.remap(raw, mx, my, cv2.INTER_LINEAR))
    rsu = tm(cv2.remap(raw, mu, mv, cv2.INTER_LINEAR))
    gs = cv2.imread(f"{os.path.dirname(DATA)}/prep/seq{int(r.seq)}/cam0/{t}.png", 0)
    print("max per-pixel shift applied (px):", float(np.hypot(mu - mx, mv - my).max()), "omega",
          float(np.linalg.norm(w_c)))
    sep = np.full((H, 12), 255, np.uint8)
    im = cv2.resize(np.concatenate([gs, sep, rs, sep, rsu], 1), None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"{RES}/../exp1/check_unroll_{r.key}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("CHECK_DONE")


if __name__ == "__main__":
    os.makedirs(f"{RES}/../exp1", exist_ok=True)
    {"check": check, "rot": lambda: run("rot"), "rotrans": lambda: run("rotrans")}[sys.argv[1]]()
