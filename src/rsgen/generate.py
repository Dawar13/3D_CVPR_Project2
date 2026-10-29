"""Generate a synthetic RS dataset split from a YAML config. Every clip is one 16-frame window (EXP2 P10).

Per clip, in data/synth/<name>/<split>/<scene>/<clip>/:
  rs/{k:02d}.png  gs/{k:02d}.png      RS frames and their global-shutter twins (same centre-line instant)
  depth.npz       rs, gs: (16, H, W) float16 z-depth (0 = no hit); RS depth is per-line-camera z
  traj.npz        dense trajectory (t, quat xyzw, c) at 1 kHz: line poses are traj(line_t) for any line
                  T_mid (16, 4, 4) cam-to-world at frame times; w, v (16, 3) true camera-frame twist at frame times
  meta.json       camera, shutter, motion settings, frame times, measured speeds
An index CSV (data/synth/<name>/<split>.csv) lists every clip with its settings and measured speeds.

usage: python -m rsgen.generate --config configs/synth_core.yaml --split test [--workers 6] [--limit N]
"""
import argparse
import json
import os
import time
from multiprocessing import get_context

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation

from .core import Camera, Shutter, render
from .motion import sample_clip_trajectory
from .scenes import Scene

_SCENES = {}


def _draw(spec, rng):
    """A value from a config entry: a scalar, a list (uniform choice) or {dist: uniform|loguniform, lo, hi}."""
    if isinstance(spec, dict):
        lo, hi = spec["lo"], spec["hi"]
        if spec.get("dist", "uniform") == "loguniform":
            return float(np.exp(rng.uniform(np.log(max(lo, 1e-3)), np.log(hi))))
        return float(rng.uniform(lo, hi))
    if isinstance(spec, list):
        return spec[int(rng.integers(len(spec)))]
    return spec


def _scene(path, scale, nthreads):
    key = (path, scale)
    if key not in _SCENES:
        _SCENES.clear()  # one scene per worker at a time keeps memory bounded
        _SCENES[key] = Scene.from_ply(path, scale=scale, nthreads=nthreads)
    return _SCENES[key]


def make_clip(job):
    cfg, split, scene_name, clip_id, seed = job
    out = os.path.join(cfg["out_dir"], split, scene_name, f"{clip_id:05d}")
    if os.path.exists(f"{out}/meta.json"):
        with open(f"{out}/meta.json") as f:
            return json.load(f)["index_row"]
    rng = np.random.default_rng(seed)
    scene = _scene(os.path.join(cfg["scene_dir"], cfg["scene_files"][scene_name]), cfg.get("scene_scale", 1.0),
                   cfg.get("threads_per_worker", 2))
    m, s = cfg["motion"], cfg["shutter"]
    kind = str(rng.choice(m["kinds"], p=m.get("kind_probs")))
    omega = np.radians(_draw(m["omega_dps"], rng)) if kind != "translation" else 0.0
    speed = _draw(m["speed_mps"], rng) if kind != "rotation" else 0.0
    sh = Shutter(readout=_draw(s["readout_ms"], rng) / 1e3, direction=_draw(s["direction"], rng),
                 exposure=_draw(s.get("exposure_ms", 0), rng) / 1e3, n_exp=int(s.get("n_exp", 1)))
    if sh.exposure == 0:
        sh.n_exp = 1
    n, dtf = cfg["n_frames"], cfg["stride"] / cfg["fps"]
    t_frames = np.arange(n) * dtf
    pad = 0.1 + sh.readout + sh.exposure
    traj, info = sample_clip_trajectory(scene, kind, omega, speed, -pad, t_frames[-1] + pad, rng, up=cfg["up"])
    cam = Camera.from_hfov(cfg["camera"]["W"], cfg["camera"]["H"], cfg["camera"]["hfov"])
    ssaa = cfg.get("render", {}).get("ssaa", 2)
    os.makedirs(f"{out}/rs", exist_ok=True)
    os.makedirs(f"{out}/gs", exist_ok=True)
    d_rs, d_gs = [], []
    gs_sh = Shutter(0.0, sh.direction, sh.exposure, sh.n_exp)
    for k, t in enumerate(t_frames):
        r = render(scene, traj, cam, sh, t, ssaa=ssaa)
        g = render(scene, traj, cam, gs_sh, t, ssaa=ssaa)
        cv2.imwrite(f"{out}/rs/{k:02d}.png", r["rgb"][..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1])
        cv2.imwrite(f"{out}/gs/{k:02d}.png", g["rgb"][..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1])
        d_rs.append(r["depth"])
        d_gs.append(g["depth"])
    np.savez_compressed(f"{out}/depth.npz", rs=np.stack(d_rs).astype(np.float16), gs=np.stack(d_gs).astype(np.float16))
    w, v = traj.twist(t_frames)
    np.savez_compressed(f"{out}/traj.npz", t=traj.t, quat=traj.rot.as_quat(), c=traj.c, T_mid=traj.T(t_frames),
                        w=w, v=v, t_frames=t_frames, K=cam.K)
    # window speed as in EXP0: mean |w| from the first frame to the last frame's end of readout
    tt = np.linspace(t_frames[0], t_frames[-1] + sh.readout, 400)
    ww, vv = traj.twist(tt)
    row = dict(split=split, scene=scene_name, clip=clip_id, key=f"{scene_name}_{clip_id:05d}", seed=int(seed),
               kind=kind, omega_target=float(omega), speed_target=float(speed), readout_ms=sh.readout * 1e3,
               direction=sh.direction, exposure_ms=sh.exposure * 1e3, n_exp=sh.n_exp,
               omega_mean=float(np.linalg.norm(ww, axis=1).mean()), omega_max=float(np.linalg.norm(ww, axis=1).max()),
               v_mean=float(np.linalg.norm(vv, axis=1).mean()),
               rs_shift_px=float(np.linalg.norm(ww, axis=1).mean() * sh.readout * cam.fx),
               depth_median=float(np.median(np.stack(d_gs)[np.stack(d_gs) > 0])))
    meta = dict(index_row=row, camera=dict(cfg["camera"], fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy),
                shutter=dict(readout=sh.readout, direction=sh.direction, exposure=sh.exposure, n_exp=sh.n_exp, band=1),
                motion=info, t_frames=t_frames.tolist(), ssaa=ssaa, config=cfg["name"])
    with open(f"{out}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    for k in ("scene_dir", "out_dir"):
        cfg[k] = os.path.expanduser(cfg[k])
    cfg["out_dir"] = os.path.join(cfg["out_dir"], cfg["name"])
    os.makedirs(cfg["out_dir"], exist_ok=True)
    scenes = cfg["splits"][a.split]
    n_clips = cfg["clips_per_scene"][a.split]
    ss = np.random.SeedSequence([cfg["seed"], sum(map(ord, a.split))])
    seeds = ss.generate_state(len(scenes) * n_clips)
    jobs = [(cfg, a.split, sc, i, int(seeds[j * n_clips + i])) for j, sc in enumerate(scenes) for i in range(n_clips)]
    if a.limit:
        jobs = jobs[:a.limit]
    t0 = time.time()
    rows = []
    with get_context("spawn").Pool(a.workers) as pool:
        for n, r in enumerate(pool.imap(make_clip, jobs, chunksize=1)):
            rows.append(r)
            if (n + 1) % 20 == 0 or n + 1 == len(jobs):
                print(f"{n + 1}/{len(jobs)} clips, {(time.time() - t0) / (n + 1):.1f} s/clip", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(cfg["out_dir"], f"{a.split}.csv"), index=False)
    print("GENERATE_DONE", a.split, len(rows), flush=True)


if __name__ == "__main__":
    main()
