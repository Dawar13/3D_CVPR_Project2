"""EXP0 preprocessing: undistort + tone-map TUM RS-VIO frames identically for GS and RS, build matched windows and GT.

Outputs
  data/prep/seq{s}/{cam}/{ts}.png     8-bit grayscale, undistorted to a shared pinhole K_new (1280x1024)
  data/windows/seq{s}/{wid:04d}_{GS|RS}/  16 symlinks per window (input for the models)
  results/exp0/windows.csv            per-window speed, parallax and filter info
  results/exp0/gt_poses.npz           GT cam-to-world poses per window/camera
  results/exp0/prep_info.json         K_new, calibration and timing constants actually used
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import (CAMS, H, MIN_PATH_M, N_FRAMES, PREP, READOUT, RES, ROW_TIME, RS_MID_OFFSET, SEQS, STRIDE,
                    W, WIN_DIR, WIN_STEP, GTInterp, load_calib, load_frames, load_imu, seq_dir, set_suffix)

GAMMA = 1 / 2.2  # fixed tone curve for the linear 16-bit images, identical for both cameras


def shared_new_K(calib):
    """One pinhole K for both cameras, zoomed until every output pixel is valid for both."""
    fs = []
    for cam in ["cam0", "cam1"]:
        Kn = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            calib[cam]["K"], calib[cam]["D"], (W, H), np.eye(3), balance=0.0)
        fs.append(Kn[0, 0])
    f = max(fs)
    while True:
        K_new = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]])
        bad = 0.0
        for cam in ["cam0", "cam1"]:
            mx, my = cv2.fisheye.initUndistortRectifyMap(
                calib[cam]["K"], calib[cam]["D"], np.eye(3), K_new, (W, H), cv2.CV_32FC1)
            bad = max(bad, ((mx < 0) | (mx > W - 1) | (my < 0) | (my > H - 1)).mean())
        if bad == 0:
            return K_new, fs
        f *= 1.01


_MAPS = {}


def _init_worker(K_new):
    calib = load_calib()
    for cam in ["cam0", "cam1"]:
        _MAPS[cam] = cv2.fisheye.initUndistortRectifyMap(calib[cam]["K"], calib[cam]["D"], np.eye(3), K_new, (W, H),
                                                         cv2.CV_32FC1)


def _prep_one(args):
    src, dst, cam = args
    if os.path.exists(dst):
        return
    mx, my = _MAPS[cam]
    im = cv2.imread(src, cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
    im = cv2.remap(im, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    im8 = np.clip(255.0 * np.power(np.clip(im, 0, 1), GAMMA) + 0.5, 0, 255).astype(np.uint8)
    cv2.imwrite(dst, im8, [cv2.IMWRITE_PNG_COMPRESSION, 1])


def main(which="main"):
    sfx = set_suffix(which)
    seqs = ["cal"] if which == "control" else SEQS
    rs_offset = 0.0 if which == "control" else RS_MID_OFFSET  # cam1 is global shutter in the calibration sequence
    os.makedirs(RES, exist_ok=True)
    calib = load_calib()
    K_new, fs_cv = shared_new_K(calib)
    f_px = float(K_new[0, 0])
    print("K_new f =", f_px, "(per-camera balance=0 f:", fs_cv, ")", flush=True)

    rows, gt_store, info_seq = [], {}, {}
    for s in [s for s in seqs if os.path.isdir(seq_dir(s))]:
        t0 = load_frames(s, "cam0")
        t1 = load_frames(s, "cam1")
        # match RS frames to GS frames by nearest timestamp (tolerance 5 ms)
        j = np.clip(np.searchsorted(t1, t0), 1, len(t1) - 1)
        j = np.where(np.abs(t1[j - 1] - t0) < np.abs(t1[j] - t0), j - 1, j)
        dt = np.abs(t1[j] - t0) * 1e-9
        matched = dt < 0.005
        gt = GTInterp(s)
        imu_t, gyro = load_imu(s)
        gyro_norm = np.linalg.norm(gyro, axis=1)

        T_imu_cam = {cam: np.linalg.inv(calib[cam]["T_cam_imu"]) for cam in ["cam0", "cam1"]}
        t_rs_mid = t1[j] + int(round(rs_offset * 1e9))
        valid = matched & gt.valid(t0) & gt.valid(t_rs_mid) & gt.valid(t1[j])
        T_gs = gt(t0) @ T_imu_cam["cam0"]
        T_rs_mid = gt(t_rs_mid) @ T_imu_cam["cam1"]
        T_rs_first = gt(t1[j]) @ T_imu_cam["cam1"]
        info_seq[s] = dict(n_frames=int(len(t0)), n_matched=int(matched.sum()), n_valid_gt=int(valid.sum()),
                           max_match_dt_ms=float(dt[matched].max() * 1e3), duration_s=float((t0[-1] - t0[0]) * 1e-9))

        # undistort + tone-map every frame that any window uses
        jobs = []
        for cam, ts in [("cam0", t0), ("cam1", t1[j])]:
            os.makedirs(f"{PREP}/seq{s}/{cam}", exist_ok=True)
            for t in ts:
                jobs.append((f"{seq_dir(s)}/{cam}/data/{t}.png", f"{PREP}/seq{s}/{cam}/{t}.png", cam))
        with ProcessPoolExecutor(12, initializer=_init_worker, initargs=(K_new,)) as ex:
            list(ex.map(_prep_one, jobs, chunksize=16))

        span = STRIDE * (N_FRAMES - 1)
        for wid, st in enumerate(range(0, len(t0) - span, WIN_STEP)):
            idx = st + STRIDE * np.arange(N_FRAMES)
            ok = bool(valid[idx].all())
            c = T_gs[idx, :3, 3]
            path_len = float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())
            extent = float(np.linalg.norm(c - c.mean(0), axis=1).max())
            ta, tb = t0[idx[0]], t0[idx[-1]] + int(READOUT * 1e9)
            m = (imu_t >= ta) & (imu_t <= tb)
            om = float(gyro_norm[m].mean()) if m.any() else np.nan
            om_max = float(gyro_norm[m].max()) if m.any() else np.nan
            # GT rotation speed of the camera from mocap (cross-check of the gyro axis)
            Rg = T_gs[idx, :3, :3]
            ang = np.radians([np.degrees(np.arccos(np.clip((np.trace(Rg[k].T @ Rg[k + 1]) - 1) / 2, -1, 1)))
                              for k in range(N_FRAMES - 1)])
            om_mocap = float(ang.sum() / ((t0[idx[-1]] - t0[idx[0]]) * 1e-9))
            keep = ok and path_len >= MIN_PATH_M
            key = f"s{s}_w{wid:04d}"
            rows.append(dict(seq=s, window_id=wid, key=key, start_frame=int(st), t_start_ns=int(t0[idx[0]]),
                             gt_valid=ok, gt_path_m=path_len, gt_extent_m=extent, omega_mean=om, omega_max=om_max,
                             omega_mocap=om_mocap, rs_shift_px=om * READOUT * f_px, keep=keep))
            if not ok:
                continue
            gt_store[f"{key}_GS"] = T_gs[idx]
            gt_store[f"{key}_RS"] = T_rs_mid[idx]
            gt_store[f"{key}_RSfirst"] = T_rs_first[idx]
            for camlab, cam, ts in [("GS", "cam0", t0), ("RS", "cam1", t1[j])]:
                d = f"{WIN_DIR}/seq{s}/{wid:04d}_{camlab}"
                os.makedirs(d, exist_ok=True)
                for k, fi in enumerate(idx):
                    lnk = f"{d}/{k:02d}_{ts[fi]}.png"
                    if not os.path.lexists(lnk):
                        os.symlink(f"{PREP}/seq{s}/{cam}/{ts[fi]}.png", lnk)
        print(f"seq{s}: {info_seq[s]}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/windows{sfx}.csv", index=False)
    np.savez_compressed(f"{RES}/gt_poses{sfx}.npz", **gt_store)
    info = dict(K_new=K_new.tolist(), f_px=f_px, gamma=GAMMA, row_time_s=ROW_TIME, readout_s=READOUT,
                rs_mid_offset_s=rs_offset, n_frames=N_FRAMES, stride=STRIDE, win_step=WIN_STEP,
                min_path_m=MIN_PATH_M, cams=CAMS,
                calib={c: {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in calib[c].items()} for c in calib},
                seqs=info_seq)
    with open(f"{RES}/prep_info{sfx}.json", "w") as f:
        json.dump(info, f, indent=1)
    print(df.groupby("seq")[["keep", "gt_path_m", "omega_mean"]].agg(["count", "mean"]).round(2), flush=True)
    print("TOTAL windows", len(df), "kept", int(df.keep.sum()), flush=True)
    print("PREP_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
