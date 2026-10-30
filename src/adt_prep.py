"""EXP2 Aria Digital Twin (ADT): RGB rolling-shutter frames, GT at the centre-row time, EXP0-style windows (D18).

Per selected sequence (data/adt/selected.txt: every 6th by name, fixed before download), reads video.vrs and
aria_trajectory.csv and writes
  data/frames_adt/<seq>/<idx:05d>.png                       RGB undistorted to a 512x512 pinhole (f = 256 px) and
                                                            rotated 90 deg clockwise to upright
  data/windows_adt/<seq>/<wid:04d>_RS/<k:02d>_<idx:05d>.png 16 symlinks per kept window
  results/exp2/gap/adt/{windows.csv, gt_poses.npz (keys <key>_RS), clips_RS.csv, prep_info.json}
  results/exp2/gap/adt/frames/<seq>.csv                     per written frame: capture time, exposure, |w|
Timing (Aria docs): t(row) = t_image + (row/H - 0.5) * readout, readout 5 ms at 1408x1408, so the capture timestamp
is the centre-row time and the GT is interpolated there. Speed axis from the GT orientation: the file's
angular_velocity columns disagree with the gyroscope (D18).

usage (env ~/rs3d/envs/aria): python adt_prep.py [--workers 6] [--seqs a,b] [--check]
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

ROOT = os.path.expanduser("~/rs3d")
RAW = f"{ROOT}/data/adt/raw"
RES = f"{ROOT}/results/exp2/gap/adt"
N_FRAMES, STRIDE, STEP, MIN_PATH_M = 16, 4, 72, 0.20
W, F = 512, 256.0  # output pinhole: 512x512, f = 256 px (90 deg FOV)
READOUT = 0.005  # s, RGB 1408x1408 (Aria docs)
MAX_GAP = 0.040  # s, GT is ~30 Hz
# upright = raw pinhole rotated 90 deg clockwise: pixel (u', v') = (W-1-v, u), camera axes x' = -y, y' = x, z' = z
R_OLD_NEW = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float)  # new-camera vector -> raw-camera vector


def gt_track(seq_dir):
    tr = pd.read_csv(f"{seq_dir}/aria_trajectory.csv")
    t = tr.tracking_timestamp_us.values * 1e-6
    o = np.argsort(t)
    tr, t = tr.iloc[o], t[o]
    R = Rotation.from_quat(tr[["qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]].values)
    p = tr[["tx_world_device", "ty_world_device", "tz_world_device"]].values
    w_body = (R[:-1].inv() * R[1:]).as_rotvec() / np.diff(t)[:, None]  # device-frame rate per GT interval
    g = tr[["gravity_x_world", "gravity_y_world", "gravity_z_world"]].values[0]
    return dict(t=t, R=R, p=p, slerp=Slerp(t, R), w=np.linalg.norm(w_body, axis=1), tw=0.5 * (t[:-1] + t[1:]),
                up=-g / np.linalg.norm(g), q=tr.quality_score.values)


def gt_valid(g, tq):
    i = np.searchsorted(g["t"], tq)
    ok = (i > 0) & (i < len(g["t"]))
    i = np.clip(i, 1, len(g["t"]) - 1)
    return ok & (tq - g["t"][i - 1] <= MAX_GAP) & (g["t"][i] - tq <= MAX_GAP)


def device_poses(g, tq):
    tq = np.clip(tq, g["t"][0], g["t"][-1])
    T = np.tile(np.eye(4), (len(tq), 1, 1))
    T[:, :3, :3] = g["slerp"](tq).as_matrix()
    T[:, :3, 3] = np.stack([np.interp(tq, g["t"], g["p"][:, k]) for k in range(3)], -1)
    return T


def omega_span(g, t0, t1):
    m = (g["tw"] >= t0) & (g["tw"] <= t1)
    return float(g["w"][m].mean()) if m.any() else np.nan


def save_png(path, arr):
    tmp = path[:-4] + ".tmp.png"
    Image.fromarray(arr).save(tmp, compress_level=1)
    os.replace(tmp, path)


def do_seq(seq, check=False):
    from projectaria_tools.core import calibration, data_provider
    from projectaria_tools.core.sensor_data import TimeDomain

    sd = f"{RAW}/{seq}"
    p = data_provider.create_vrs_data_provider(f"{sd}/video.vrs")
    rgb = p.get_stream_id_from_label("camera-rgb")
    src = p.get_device_calibration().get_camera_calib("camera-rgb")
    T_dc = src.get_transform_device_camera().to_matrix()
    dst = calibration.get_linear_camera_calibration(W, W, F, "camera-rgb", src.get_transform_device_camera())
    M = np.eye(4)
    M[:3, :3] = R_OLD_NEW
    ts = np.array(p.get_timestamps_ns(rgb, TimeDomain.DEVICE_TIME)) * 1e-9
    g = gt_track(sd)
    T_cam = device_poses(g, ts) @ T_dc @ M  # upright camera-to-world at each frame's capture (centre-row) time
    ok_f = gt_valid(g, ts)
    up = np.einsum("nji,j->ni", T_cam[ok_f, :3, :3], g["up"])  # world-up in the upright camera frame
    span = (N_FRAMES - 1) * STRIDE
    rows, gt, need = [], {}, set()
    for wid, st in enumerate(range(0, len(ts) - span, STEP)):
        idx = st + STRIDE * np.arange(N_FRAMES)
        valid = bool(ok_f[idx].all())
        T = T_cam[idx]
        path_m = float(np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1).sum())
        dur = float(ts[idx[-1]] - ts[idx[0]])
        om = omega_span(g, ts[idx[0]], ts[idx[-1]]) if valid else np.nan
        key = f"{seq}_w{wid:04d}"
        keep = valid and path_m >= MIN_PATH_M
        rows.append(dict(seq=seq, window_id=wid, key=key, start_frame=int(st), gt_valid=valid, keep=keep,
                         gt_path_m=path_m if valid else np.nan, omega_mean=om, v_mean=path_m / dur if valid else np.nan,
                         readout_ms=READOUT * 1e3, rs_direction="rl (UNVERIFIED sign)",
                         rs_shift_px=om * READOUT * F if valid else np.nan))
        if keep:
            gt[f"{key}_RS"] = T
            need.update(int(i) for i in idx)
    out_f = f"{ROOT}/data/frames_adt/{seq}"
    os.makedirs(out_f, exist_ok=True)
    frames = []
    for i in sorted(need):
        img, rec = p.get_image_data_by_index(rgb, i)
        fp = f"{out_f}/{i:05d}.png"
        if not os.path.exists(fp):
            und = calibration.distort_by_calibration(img.to_numpy_array(), dst, src)  # default: bilinear
            save_png(fp, np.ascontiguousarray(np.rot90(und, k=-1)))
        frames.append(dict(idx=i, t_capture_ns=int(rec.capture_timestamp_ns), exposure_ms=rec.exposure_duration * 1e3,
                           gain=rec.gain, omega=np.interp(ts[i], g["tw"], g["w"])))
        if check and len(frames) >= 32:
            break
    fr = pd.DataFrame(frames, columns=["idx", "t_capture_ns", "exposure_ms", "gain", "omega"])
    ex = fr.set_index("idx").exposure_ms if len(fr) else pd.Series(dtype=float)
    for r in rows:
        if r["keep"]:
            ii = r["start_frame"] + STRIDE * np.arange(N_FRAMES)
            e = ex.reindex(ii)
            r["exposure_ms_mean"] = float(e.mean())
            r["blur_px"] = r["omega_mean"] * float(e.mean()) * 1e-3 * F
            r["gt_quality_min"] = float(g["q"][(g["t"] >= ts[ii[0]] - MAX_GAP) & (g["t"] <= ts[ii[-1]] + MAX_GAP)].min())
            wd = f"{ROOT}/data/windows_adt/{seq}/{r['window_id']:04d}_RS"
            os.makedirs(wd, exist_ok=True)
            for k, i in enumerate(ii):
                lnk = f"{wd}/{k:02d}_{i:05d}.png"
                if not os.path.lexists(lnk):
                    os.symlink(f"{out_f}/{i:05d}.png", lnk)
    info = dict(n_frames=len(ts), n_frames_with_gt=int(ok_f.sum()), fps=float(1 / np.median(np.diff(ts))),
                n_windows=len(rows), n_kept=int(sum(r["keep"] for r in rows)),
                up_in_camera_mean=up.mean(0).round(3).tolist(),
                upright_ok=bool(up[:, 1].mean() < -0.5),  # image up is -y
                exposure_ms_pct_5_50_95=np.percentile(fr.exposure_ms, [5, 50, 95]).round(2).tolist() if len(fr) else [],
                src_calib=str(src.get_model_name()), src_focal=float(src.get_focal_lengths()[0]))
    return seq, pd.DataFrame(rows), gt, fr, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seqs", default="")
    ap.add_argument("--check", action="store_true", help="one sequence, 32 frames, print and exit")
    a = ap.parse_args()
    seqs = a.seqs.split(",") if a.seqs else open(f"{ROOT}/data/adt/selected.txt").read().split()
    have = [s for s in seqs if os.path.exists(f"{RAW}/{s}/video.vrs") and os.path.exists(f"{RAW}/{s}/aria_trajectory.csv")]
    print(f"{len(have)} of {len(seqs)} sequences downloaded", flush=True)
    if a.check:
        seq, w, gt, fr, info = do_seq(have[0], check=True)
        print(json.dumps(info, indent=1))
        print(w[w.keep].head().to_string())
        return
    os.makedirs(f"{RES}/frames", exist_ok=True)
    wins, gts, infos = [], {}, {}
    with Pool(a.workers) as pool:
        for seq, w, gt, fr, info in pool.imap_unordered(do_seq, have):
            wins.append(w)
            gts.update(gt)
            infos[seq] = info
            fr.to_csv(f"{RES}/frames/{seq}.csv", index=False)
            print(seq, info["n_frames"], "frames,", info["n_kept"], "kept windows, up", info["up_in_camera_mean"],
                  flush=True)
    w = pd.concat(wins).sort_values(["seq", "window_id"])
    w.to_csv(f"{RES}/windows.csv", index=False)
    np.savez(f"{RES}/gt_poses.npz", **gts)
    k = w[w.keep]
    pd.DataFrame(dict(key=[f"{x}_RS" for x in k.key],
                      dir=[f"{ROOT}/data/windows_adt/{s}/{i:04d}_RS" for s, i in zip(k.seq, k.window_id)])
                 ).to_csv(f"{RES}/clips_RS.csv", index=False)
    with open(f"{RES}/prep_info.json", "w") as f:
        json.dump(dict(n_frames=N_FRAMES, stride=STRIDE, step=STEP, min_path_m=MIN_PATH_M, out_size=W, focal=F,
                       readout_s=READOUT, max_gap_s=MAX_GAP, sequences=infos, n_windows=len(w),
                       n_kept=int(w.keep.sum()), all_upright_ok=all(v["upright_ok"] for v in infos.values())),
                  f, indent=1)
    print("TOTAL windows", len(w), "kept", int(w.keep.sum()), "sequences", len(infos), flush=True)


if __name__ == "__main__":
    sys.exit(main())
