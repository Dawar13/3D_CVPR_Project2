"""EXP2 Waymo Open Dataset (v1.4.3, validation): FRONT camera frames, per-image RS timing and GT, EXP0-style windows.

Reads segment TFRecords with Waymo's own protos (compiled to ~/rs3d/ext/waymo_pb, no TensorFlow) and writes
  data/waymo/front/<segment>/<i:04d>.jpg            the FRONT camera JPEG as stored
  results/exp2/gap/waymo/frames_<segment>.csv       per image: times, readout, shutter, velocity
  results/exp2/gap/waymo/{windows.csv, gt_poses.npz (keys <key>_RS), clips_RS.csv, prep_info.json}
  data/windows_waymo/<segment>/<wid:04d>_RS/       16 symlinks per window
GT camera pose at the image's centre readout time (DEVIATIONS D15): the vehicle pose is extrapolated from
pose_timestamp with its global-frame velocity, then camera = vehicle . extrinsic . (OpenCV -> Waymo camera axes).

usage (env ~/rs3d/envs/waymo): PYTHONPATH=~/rs3d/ext/waymo_pb python waymo_prep.py
"""
import glob
import json
import os
import struct
import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from waymo_open_dataset import dataset_pb2

ROOT = os.path.expanduser("~/rs3d")
N_FRAMES, STRIDE, STEP, MIN_PATH_M = 16, 1, 8, 0.20
FRONT = dataset_pb2.CameraName.FRONT
# a vector in OpenCV camera axes (x right, y down, z forward) expressed in Waymo camera axes (x fwd, y left, z up)
CV_TO_WAYMO = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
DIRS = {0: "unknown", 1: "tb", 2: "lr", 3: "bt", 4: "rl", 5: "global"}


def records(path):
    """Raw TFRecord reader: uint64 length, uint32 crc, payload, uint32 crc."""
    with open(path, "rb") as f:
        while True:
            h = f.read(12)
            if len(h) < 12:
                return
            n = struct.unpack("<Q", h[:8])[0]
            data = f.read(n)
            f.read(4)
            yield data


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def segment(path, out_img):
    os.makedirs(out_img, exist_ok=True)
    rows, calib = [], None
    for i, data in enumerate(records(path)):
        fr = dataset_pb2.Frame()
        fr.ParseFromString(data)
        if calib is None:
            c = next(c for c in fr.context.camera_calibrations if c.name == FRONT)
            calib = dict(intrinsic=list(c.intrinsic), extrinsic=list(c.extrinsic.transform), width=c.width,
                         height=c.height, rs_direction=DIRS.get(c.rolling_shutter_direction, "unknown"))
        im = next(m for m in fr.images if m.name == FRONT)
        p = f"{out_img}/{i:04d}.jpg"
        if not os.path.exists(p):
            with open(p, "wb") as f:
                f.write(im.image)
        v = im.velocity
        readout = im.camera_readout_done_time - im.camera_trigger_time - im.shutter
        rows.append(dict(i=i, pose_ts=im.pose_timestamp, trigger=im.camera_trigger_time,
                         readout_done=im.camera_readout_done_time, shutter=im.shutter, readout=readout,
                         t_center=im.camera_trigger_time + 0.5 * im.shutter + 0.5 * readout,
                         vx=v.v_x, vy=v.v_y, vz=v.v_z, wx=v.w_x, wy=v.w_y, wz=v.w_z,
                         **{f"T{k}": x for k, x in enumerate(im.pose.transform)}))
    return pd.DataFrame(rows), calib


def camera_poses(df, calib):
    """Cam-to-world (OpenCV axes) at each image's centre readout time."""
    T_vc = np.array(calib["extrinsic"]).reshape(4, 4)
    M = np.eye(4)
    M[:3, :3] = CV_TO_WAYMO
    out = []
    for r in df.itertuples():
        T_wv = np.array([getattr(r, f"T{k}") for k in range(16)]).reshape(4, 4)
        dt = r.t_center - r.pose_ts
        w, v = np.array([r.wx, r.wy, r.wz]), np.array([r.vx, r.vy, r.vz])
        T = T_wv.copy()
        T[:3, :3] = Rotation.from_rotvec(w * dt).as_matrix() @ T_wv[:3, :3]  # dR/dt = [w]x R, w in the global frame
        T[:3, 3] = T_wv[:3, 3] + v * dt
        out.append(T @ T_vc @ M)
    return np.stack(out)


def main():
    res = f"{ROOT}/results/exp2/gap/waymo"
    os.makedirs(res, exist_ok=True)
    win_rows, gt, info = [], {}, {}
    for path in sorted(glob.glob(f"{ROOT}/data/waymo/raw/*.tfrecord")):
        seg = os.path.basename(path).split("_with_camera_labels")[0].replace("segment-", "s")
        img_dir = f"{ROOT}/data/waymo/front/{seg}"
        df, calib = segment(path, img_dir)
        df.to_csv(f"{res}/frames_{seg}.csv", index=False)
        P = camera_poses(df, calib)
        info[seg] = dict(calib, n_frames=len(df), readout_ms_median=float(df.readout.median() * 1e3),
                         shutter_ms_median=float(df.shutter.median() * 1e3),
                         pose_to_center_ms_median=float((df.t_center - df.pose_ts).median() * 1e3))
        f_px = calib["intrinsic"][0]
        span = (N_FRAMES - 1) * STRIDE
        n_keep = 0
        for wid, st in enumerate(range(0, len(df) - span, STEP)):
            idx = st + STRIDE * np.arange(N_FRAMES)
            T = P[idx]
            d = df.iloc[idx]
            path_m = float(np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1).sum())
            om = float(np.linalg.norm(d[["wx", "wy", "wz"]].values, axis=1).mean())
            dur = float(d.t_center.iloc[-1] - d.t_center.iloc[0])
            key = f"{seg}_w{wid:04d}"
            keep = path_m >= MIN_PATH_M
            win_rows.append(dict(seq=seg, window_id=wid, key=key, start_frame=st, keep=keep, gt_path_m=path_m,
                                 omega_mean=om, v_mean=path_m / dur, readout_ms=float(d.readout.median() * 1e3),
                                 rs_direction=calib["rs_direction"],
                                 rs_shift_px=om * float(d.readout.median()) * f_px))
            if keep:
                n_keep += 1
                gt[f"{key}_RS"] = T
                wd = f"{ROOT}/data/windows_waymo/{seg}/{wid:04d}_RS"
                os.makedirs(wd, exist_ok=True)
                for k, i in enumerate(idx):
                    lnk = f"{wd}/{k:02d}_{i:04d}.jpg"
                    if not os.path.lexists(lnk):
                        os.symlink(f"{img_dir}/{i:04d}.jpg", lnk)
        print(seg, len(df), "frames,", n_keep, "kept windows, readout %.1f ms, %s" %
              (info[seg]["readout_ms_median"], calib["rs_direction"]), flush=True)
    w = pd.DataFrame(win_rows)
    w.to_csv(f"{res}/windows.csv", index=False)
    np.savez(f"{res}/gt_poses.npz", **gt)
    k = w[w.keep]
    pd.DataFrame(dict(key=[f"{x}_RS" for x in k.key],
                      dir=[f"{ROOT}/data/windows_waymo/{s}/{i:04d}_RS" for s, i in zip(k.seq, k.window_id)])
                 ).to_csv(f"{res}/clips_RS.csv", index=False)
    with open(f"{res}/prep_info.json", "w") as f:
        json.dump(dict(n_frames=N_FRAMES, stride=STRIDE, step=STEP, min_path_m=MIN_PATH_M, segments=info,
                       n_windows=len(w), n_kept=int(w.keep.sum())), f, indent=1)
    print("TOTAL windows", len(w), "kept", int(w.keep.sum()), flush=True)


if __name__ == "__main__":
    sys.exit(main())
