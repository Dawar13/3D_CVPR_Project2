"""EXP2 synthetic conditions: straightened versions of each clip's RS frames, built on the GPU (gpu_unroll.py).

  v3                 rotation only, true w at each frame time (oracle, EXP2 §7 V3)
  v4                 true w, true v and the true RS depth (oracle V4)
  v3_r<f>, v4_r<f>   the unroller assumes readout x f (f in 0.5, 0.75, 1.25, 1.5): oracle sensitivity
  v3_dir, v4_dir     the unroller assumes the wrong scan direction (tb <-> bt; lr -> tb)
  se2p-<m>           training-free self-estimate, rotation only: w from model m's own pass-1 poses on rs/
  se2pt-<m>          ... plus v from the pass-1 camera centres and m's own pass-1 depth (model units cancel in v/Z)
  se2p-<m>-shuf      as se2p, but the pass-1 run (<m>_rsshuf.npz, keys <key>_rsshuf) saw the frames in a shuffled
                     order, and finite differences are taken in that order with a uniform frame gap (an unordered
                     photo set has no times)
Writes <clip>/<cond>/{k:02d}.png (8-bit RGB, same size and intrinsics as rs/).

usage: python synth_conditions.py --set ~/rs3d/data/synth/core_test --conds v3,v4 [--preds <dir>]
"""
import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import gpu_unroll as G
from dense_utils import Clip, list_clips, pmap_of, to_orig_grid
from infer2 import frame_order
from tum_conditions import twist_from_poses

DEV = "cuda" if torch.cuda.is_available() else "cpu"
WRONG_DIR = dict(tb="bt", bt="tb", lr="tb")


def parse(cond):
    m = re.fullmatch(r"(v3|v4)(?:_r([0-9.]+))?(_dir)?", cond)
    if m:
        return dict(kind=m.group(1), factor=float(m.group(2) or 1.0), wrong_dir=bool(m.group(3)))
    m = re.fullmatch(r"(se2pt?)-(\w+?)(-shuf)?", cond)
    if m:
        return dict(kind=m.group(1), model=m.group(2), shuf=bool(m.group(3)), factor=1.0, wrong_dir=False)
    raise ValueError(cond)


def unroll(cl, imgs, w, v=None, depth=None, factor=1.0, direction=None):
    """imgs (n, H, W, 3) uint8 RS frames -> straightened uint8 frames, one constant twist per frame."""
    d = direction or cl.shutter.direction
    L = cl.W if d == "lr" else cl.H
    scan = G.ScanModel(cl.shutter.readout * factor / L, cl.H, cl.W, d)
    K = cl.K
    un = G.Unroller(K, (cl.H, cl.W), lambda X: G.pinhole_project(X, K), scan, device=DEV)
    mu, mv = un.maps(np.asarray(w), None if v is None else np.asarray(v), depth)
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(DEV).float()
    out = G.Unroller.remap(x, mu.float(), mv.float())
    return (out + 0.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def read_rgb(p):
    return cv2.imread(p, cv2.IMREAD_COLOR)[..., ::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="generated set dir, e.g. ~/rs3d/data/synth/core_test")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conds", required=True)
    ap.add_argument("--preds", default="", help="pass-1 predictions dir (for se2p*)")
    a = ap.parse_args()
    clips, n_fail = list_clips(os.path.expanduser(a.set), a.split)
    print(f"{len(clips)} clips ({n_fail} failed at generation)", flush=True)
    io = ThreadPoolExecutor(12)
    for cond in a.conds.split(","):
        p = parse(cond)
        if p["kind"].startswith("se2p"):
            tag = "rsshuf" if p["shuf"] else "rs"  # pass 1: plain RS frames, sorted or shuffled
            P1 = np.load(f"{a.preds}/{p['model']}_{tag}.npz")
        n_done = 0
        for r in clips.itertuples():
            out = f"{r.dir}/{cond}"
            if os.path.isdir(out) and len(os.listdir(out)) == r.n_frames:
                continue
            cl = Clip(r.dir)
            n = len(cl.t_frames)
            imgs = np.stack(list(io.map(read_rgb, [f"{r.dir}/rs/{k:02d}.png" for k in range(n)])))
            v = depth = None
            direction = WRONG_DIR[cl.shutter.direction] if p["wrong_dir"] else None
            if p["kind"] in ("v3", "v4"):
                w = cl.w
                if p["kind"] == "v4":
                    v, depth = cl.v, cl.depth("rs")
            else:
                P = P1[f"{cl.key}_{tag}"]
                if p["shuf"]:
                    order = frame_order(f"{cl.key}_{tag}", n, "shuffled")  # the order pass 1 saw
                    ws, vs = twist_from_poses(P[order], np.arange(n) * (cl.t_frames[1] - cl.t_frames[0]))
                    w, v = np.empty_like(ws), np.empty_like(vs)
                    w[order], v[order] = ws, vs
                else:
                    w, v = twist_from_poses(P, cl.t_frames)
                if p["kind"] == "se2pt":
                    dz = np.load(f"{a.preds}/{p['model']}_rs_dense/{cl.key}_rs.npz")
                    depth = to_orig_grid(dz["local_pts"][..., 2].astype(np.float32), pmap_of(dz))
                else:
                    v = None
            ims = unroll(cl, imgs, w, v, depth, p["factor"], direction)
            os.makedirs(out, exist_ok=True)
            list(io.map(lambda kv: cv2.imwrite(f"{out}/{kv[0]:02d}.png", kv[1][..., ::-1],
                                               [cv2.IMWRITE_PNG_COMPRESSION, 1]), enumerate(ims)))
            n_done += 1
        print(f"{cond}: {n_done} clips written", flush=True)
    print("SYNTH_COND_DONE", flush=True)


if __name__ == "__main__":
    main()
