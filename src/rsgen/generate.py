"""Generate a synthetic RS dataset split from a YAML config. Every clip is one window of n_frames (EXP2 P10).

Per clip, in data/synth/<name>/<split>/<scene>/<clip>/:
  rs/{k:02d}.png  gs/{k:02d}.png      RS frames and their global-shutter twins (same centre-line instant)
  rs_nb/{k:02d}_m1.png, _p1.png       (neighbors: true, clip id < neighbors_max_clip) RS frames one video
                                      frame (1/fps) before / after
  depth.npz       rs, gs: (n, H, W) float16 z-depth (0 = no hit); RS depth is per-line-camera z;
                  rs_obj: (n, H, W) int8 moving-object index per RS pixel (-1 = static), when objects exist
  traj.npz        dense trajectory (t, quat xyzw, c) at 1 kHz: line poses are traj(line_t) for any line
                  T_mid (n, 4, 4) cam-to-world at frame times; w, v (n, 3) true camera-frame twist at frame times
  meta.json       camera, shutter, motion settings, frame times, measured speeds
An index CSV (data/synth/<name>/<split>.csv) lists every clip with its settings and measured speeds.

Seeds are per (config seed, split, scene, clip id) and each part of a clip draws from its own random stream
(motion, shutter, objects). Two configs that share the seed, the motion settings and the scene list therefore
produce the same trajectories, whatever their shutter, object or clip-count settings: stress sets are paired
with the core test set clip by clip.

usage: python -m rsgen.generate --config configs/synth_core.yaml --split test [--workers 12] [--limit N]
"""
import argparse
import json
import os
import time
import zlib
from multiprocessing import get_context

import cv2
import numpy as np
import pandas as pd
import yaml

from .core import Camera, Shutter, render
from .motion import sample_clip_trajectory
from .scenes import Scene, moving_objects

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


def clip_seed(seed, split, scene, clip_id):
    return int(np.random.SeedSequence([seed, zlib.crc32(split.encode()), zlib.crc32(scene.encode()),
                                       clip_id]).generate_state(1)[0])


def _scene(path, scale, nthreads, contrast):
    key = (path, scale, contrast)
    if key not in _SCENES:
        _SCENES.clear()  # one scene per worker at a time keeps memory bounded
        sc = Scene.from_ply(path, scale=scale, nthreads=nthreads)
        if contrast != 1:
            sc.set_contrast(contrast)
        _SCENES[key] = sc
    return _SCENES[key]


def _write(path, rgb):
    cv2.imwrite(path, rgb[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1])


def make_clip(job):
    cfg, split, scene_name, clip_id = job
    out = os.path.join(cfg["out_dir"], split, scene_name, f"{clip_id:05d}")
    if os.path.exists(f"{out}/meta.json"):
        with open(f"{out}/meta.json") as f:
            return json.load(f)["index_row"]
    seed = clip_seed(cfg["seed"], split, scene_name, clip_id)
    rng_m, rng_s, rng_o = (np.random.default_rng([seed, k]) for k in range(3))
    contrast = cfg.get("scene_mod", {}).get("contrast", 1.0)
    scene = _scene(os.path.join(cfg["scene_dir"], cfg["scene_files"].format(scene=scene_name)),
                   cfg.get("scene_scale", 1.0), cfg.get("threads_per_worker", 2), contrast)
    m, s = cfg["motion"], cfg["shutter"]
    kind = str(rng_m.choice(m["kinds"], p=m.get("kind_probs")))
    omega_draw, speed_draw = np.radians(_draw(m["omega_dps"], rng_m)), _draw(m["speed_mps"], rng_m)
    omega = 0.0 if kind == "translation" else omega_draw
    speed = 0.0 if kind == "rotation" else speed_draw
    sh = Shutter(readout=_draw(s["readout_ms"], rng_s) / 1e3, direction=_draw(s["direction"], rng_s),
                 exposure=_draw(s.get("exposure_ms", 0), rng_s) / 1e3, n_exp=int(s.get("n_exp", 1)))
    if sh.exposure == 0:
        sh.n_exp = 1
    n, fps = cfg["n_frames"], cfg["fps"]
    t_frames = np.arange(n) * cfg["stride"] / fps
    nb = bool(cfg.get("neighbors", False)) and clip_id < cfg.get("neighbors_max_clip", 10 ** 9)
    # fixed padding (covers readout + exposure <= 0.1 s and the neighbour frames), so the trajectory, which is
    # normalised over its time span, does not depend on the shutter knobs: paired sets share it exactly
    pad = cfg.get("pad_s", 0.25)
    assert sh.readout / 2 + sh.exposure / 2 + (1 / fps if nb else 0) < pad, "pad_s too small"
    traj, info = sample_clip_trajectory(scene, kind, omega, speed, -pad, t_frames[-1] + pad, rng_m, up=cfg["up"])
    oc = cfg.get("objects", {})
    n_obj = int(_draw(oc.get("n", 0), rng_o))
    objs = moving_objects(traj, t_frames[n // 2], -pad, t_frames[-1] + pad, rng_o, n_obj,
                          _draw(oc.get("speed_mps", 1.0), rng_o), up=cfg["up"]) if n_obj else ()
    cam = Camera.from_hfov(cfg["camera"]["W"], cfg["camera"]["H"], cfg["camera"]["hfov"])
    ssaa = cfg.get("render", {}).get("ssaa", 2)
    for d in ["rs", "gs"] + (["rs_nb"] if nb else []):
        os.makedirs(f"{out}/{d}", exist_ok=True)
    d_rs, d_gs, o_rs = [], [], []
    gs_sh = Shutter(0.0, sh.direction, sh.exposure, sh.n_exp)
    for k, t in enumerate(t_frames):
        r = render(scene, traj, cam, sh, t, ssaa=ssaa, objects=objs)
        g = r if sh.readout == 0 else render(scene, traj, cam, gs_sh, t, ssaa=ssaa, objects=objs)
        _write(f"{out}/rs/{k:02d}.png", r["rgb"])
        _write(f"{out}/gs/{k:02d}.png", g["rgb"])
        if nb:
            for tag, dt in (("m1", -1 / fps), ("p1", 1 / fps)):
                _write(f"{out}/rs_nb/{k:02d}_{tag}.png", render(scene, traj, cam, sh, t + dt, ssaa=ssaa, objects=objs)["rgb"])
        d_rs.append(r["depth"])
        d_gs.append(g["depth"])
        o_rs.append(r["obj"])
    dep = dict(rs=np.stack(d_rs).astype(np.float16), gs=np.stack(d_gs).astype(np.float16))
    if n_obj:
        dep["rs_obj"] = np.stack(o_rs)
    np.savez_compressed(f"{out}/depth.npz", **dep)
    w, v = traj.twist(t_frames)
    np.savez_compressed(f"{out}/traj.npz", t=traj.t, quat=traj.rot.as_quat(), c=traj.c, T_mid=traj.T(t_frames),
                        w=w, v=v, t_frames=t_frames, K=cam.K)
    # window speed as in EXP0: mean |w| from the first frame to the last frame's end of readout
    tt = np.linspace(t_frames[0], t_frames[-1] + sh.readout, 400)
    ww, vv = traj.twist(tt)
    dg = np.stack(d_gs)
    row = dict(split=split, scene=scene_name, clip=clip_id, key=f"{scene_name}_{clip_id:05d}", seed=seed,
               kind=kind, omega_target=float(omega), speed_target=float(speed), readout_ms=sh.readout * 1e3,
               direction=sh.direction, exposure_ms=sh.exposure * 1e3, n_exp=sh.n_exp, n_objects=n_obj,
               contrast=contrast, n_frames=n, stride=cfg["stride"],
               omega_mean=float(np.linalg.norm(ww, axis=1).mean()), omega_max=float(np.linalg.norm(ww, axis=1).max()),
               v_mean=float(np.linalg.norm(vv, axis=1).mean()),
               rs_shift_px=float(np.linalg.norm(ww, axis=1).mean() * sh.readout * cam.fx),
               depth_median=float(np.median(dg[dg > 0])) if (dg > 0).any() else 0.0,
               obj_frac=float((np.stack(o_rs) >= 0).mean()))
    meta = dict(index_row=row, camera=dict(cfg["camera"], fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy),
                shutter=dict(readout=sh.readout, direction=sh.direction, exposure=sh.exposure, n_exp=sh.n_exp, band=1),
                motion=info, t_frames=t_frames.tolist(), fps=fps, neighbors=nb, ssaa=ssaa, config=cfg["name"],
                objects=[dict(t=o.traj.t[[0, -1]].tolist(), c0=o.traj.c[0].tolist(), c1=o.traj.c[-1].tolist())
                         for o in objs])
    with open(f"{out}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    return row


def _safe(job):
    try:
        return make_clip(job)
    except RuntimeError as e:  # no collision-free trajectory: recorded, never silently dropped
        return dict(split=job[1], scene=job[2], clip=job[3], key=f"{job[2]}_{job[3]:05d}", failed=str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    for k in ("scene_dir", "out_dir"):
        cfg[k] = os.path.expanduser(cfg[k])
    cfg["out_dir"] = os.path.join(cfg["out_dir"], cfg["name"])
    os.makedirs(cfg["out_dir"], exist_ok=True)
    with open(os.path.join(cfg["out_dir"], f"config_{a.split}.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    scenes = cfg["splits"][a.split]
    n_clips = cfg["clips_per_scene"][a.split]
    # interleave scenes so a partial run still covers every scene
    jobs = [(cfg, a.split, sc, i) for i in range(n_clips) for sc in scenes]
    if a.limit:
        jobs = jobs[:a.limit]
    t0 = time.time()
    rows = []
    with get_context("spawn").Pool(a.workers) as pool:
        for n, r in enumerate(pool.imap(_safe, jobs, chunksize=1)):
            rows.append(r)
            if (n + 1) % 20 == 0 or n + 1 == len(jobs):
                print(f"{n + 1}/{len(jobs)} clips, {(time.time() - t0) / (n + 1):.1f} s/clip", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg["out_dir"], f"{a.split}.csv"), index=False)
    n_fail = int(df["failed"].notna().sum()) if "failed" in df else 0
    print("GENERATE_DONE", a.split, len(rows), "failed", n_fail, flush=True)


if __name__ == "__main__":
    main()
