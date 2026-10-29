"""EXP2 inference on any clip list: poses always, dense output on request. Images only, no priors, bf16.

clips CSV columns: key, dir   (dir holds the clip's frames; they are used in sorted-name order)
                  optional name, dense, order: per-row output file, dense flag and frame order (else the CLI's)
writes <out>/<model>_<name>.npz                 key -> (N, 4, 4) cam-to-world (OpenCV), in sorted-frame order
       <out>/<model>_<name>_run.json            wall clock, peak memory, checkpoint, pixel map
       <out>/<model>_<name>_dense/<key>.npz     (with --dense) local_pts (N, h, w, 3) f16 camera-frame points,
                                                conf (N, h, w) f16, order (N,) frame order the model saw
The pixel map (sx, ox, sy, oy) gives original pixel = s * model pixel + o (pixel centres). It is measured by
passing black, white and ramp images through the model's own loader, so resizes and crops are exact.

--order shuffled feeds each clip's frames in a fixed random order (seed from the key); poses are returned to
sorted order before saving, so scoring is unchanged. --views N keeps the first N frames (stress test).

usage: python infer2.py --model pi3 --clips clips.csv --out <dir> --name rs [--dense] [--order shuffled]
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.expanduser("~/rs3d/third_party/pi3"))

DTYPE = torch.bfloat16
DEV = "cuda"


def free_vram_gb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return float(out.split()[0]) / 1024


def intr_unproject(depth, K):
    """depth (N, h, w), K (N, 3, 3) in model pixels -> camera-frame points (N, h, w, 3)."""
    N, h, w = depth.shape
    v, u = torch.meshgrid(torch.arange(h, device=depth.device, dtype=depth.dtype),
                          torch.arange(w, device=depth.device, dtype=depth.dtype), indexing="ij")
    x = (u[None] - K[:, 0, 2, None, None]) / K[:, 0, 0, None, None]
    y = (v[None] - K[:, 1, 2, None, None]) / K[:, 1, 1, None, None]
    return torch.stack([x * depth, y * depth, depth], -1)


class Pi3Runner:
    ckpt = "yyfz233/Pi3"
    PIXEL_LIMIT = 255000

    def __init__(self):
        from pi3.models.pi3 import Pi3
        self.model = Pi3.from_pretrained(self.ckpt).to(DEV).eval()

    def load(self, paths):
        """pi3.utils.basic.load_images_as_tensor, but for an explicit path list (same size rule, LANCZOS)."""
        ims = [Image.open(p).convert("RGB") for p in paths]
        W0, H0 = ims[0].size
        scale = np.sqrt(self.PIXEL_LIMIT / (W0 * H0))
        Wt, Ht = W0 * scale, H0 * scale
        k, m = round(Wt / 14), round(Ht / 14)
        while (k * 14) * (m * 14) > self.PIXEL_LIMIT:
            if k / m > Wt / Ht:
                k -= 1
            else:
                m -= 1
        size = (max(1, k) * 14, max(1, m) * 14)
        arr = np.stack([np.asarray(im.resize(size, Image.Resampling.LANCZOS)) for im in ims])
        return torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255

    def channel0(self, x):
        return x[:, 0].numpy()

    @torch.no_grad()
    def run(self, x, dense):
        with torch.amp.autocast("cuda", dtype=DTYPE):
            res = self.model(x.to(DEV)[None])
        out = dict(poses=res["camera_poses"][0].float().cpu().numpy())
        if dense:
            out["local_pts"] = res["local_points"][0].float().cpu().numpy()
            out["conf"] = torch.sigmoid(res["conf"][0, ..., 0].float()).cpu().numpy()
        return out


class VGGTRunner:
    ckpt = "facebook/VGGT-1B"

    def __init__(self):
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        self.lp, self.dec = load_and_preprocess_images, pose_encoding_to_extri_intri
        self.model = VGGT.from_pretrained(self.ckpt).to(DEV).eval()

    def load(self, paths):
        return self.lp(paths)  # default mode="crop": width 518

    def channel0(self, x):
        return x[:, 0].numpy()

    @torch.no_grad()
    def run(self, x, dense):
        imgs = x.to(DEV)[None]
        with torch.autocast("cuda", dtype=DTYPE):  # as in the VGGT README detailed usage
            toks, ps_idx = self.model.aggregator(imgs)
        pose_enc = self.model.camera_head(toks)[-1]
        extr, intr = self.dec(pose_enc, imgs.shape[-2:])  # (1, N, 3, 4) world-to-camera
        E = np.tile(np.eye(4), (extr.shape[1], 1, 1))
        E[:, :3, :] = extr[0].float().cpu().numpy()
        out = dict(poses=np.linalg.inv(E))
        if dense:
            with torch.autocast("cuda", dtype=DTYPE):
                depth, dconf = self.model.depth_head(toks, imgs, ps_idx)
            out["local_pts"] = intr_unproject(depth[0, ..., 0].float(), intr[0].float()).cpu().numpy()
            out["conf"] = dconf[0].float().cpu().numpy()
        return out


class MapAnythingRunner:
    ckpt = "facebook/map-anything-apache"

    def __init__(self):
        from mapanything.models import MapAnything
        from mapanything.utils.image import load_images
        self.load_images = load_images
        self.model = MapAnything.from_pretrained(self.ckpt).to(DEV).eval()

    def load(self, paths):
        return self.load_images(paths)

    def channel0(self, views):
        return torch.cat([v["img"] for v in views], 0)[:, 0].numpy()

    @torch.no_grad()
    def run(self, views, dense):
        preds = self.model.infer(views, memory_efficient_inference=False, use_amp=True, amp_dtype="bf16",
                                 apply_mask=False, mask_edges=False, apply_confidence_mask=False)
        out = dict(poses=torch.cat([p["camera_poses"] for p in preds], 0).float().cpu().numpy())
        if dense:
            out["local_pts"] = torch.cat([p["pts3d_cam"] for p in preds], 0).float().cpu().numpy()
            out["conf"] = torch.cat([p["conf"] for p in preds], 0).float().cpu().numpy()
        return out


RUNNERS = dict(pi3=Pi3Runner, vggt=VGGTRunner, mapanything=MapAnythingRunner)
try:  # extra models (DA3, streaming, VGGT-Omega) live in their own module and may need their own env
    from models_extra import EXTRA_RUNNERS
    RUNNERS.update(EXTRA_RUNNERS)
except ImportError:
    pass


def pixel_map(runner, W, H):
    """(sx, ox, sy, oy): original pixel = s * model pixel + o, measured through the runner's own loader."""
    # 8-bit ramps quantise into a staircase that biases the fit by up to ~0.6 px; random dither makes the rounding
    # error zero-mean, so the fit is unbiased (< 0.04 px at the image edges, checked against exact PIL resizes).
    # The ramp stays inside [8, 247] so the dither is never clipped.
    d = tempfile.mkdtemp(prefix="pmap_")
    rng = np.random.default_rng(0)
    lo_v, hi_v = 8.0, 247.0
    xs = np.linspace(lo_v, hi_v, W)[None].repeat(H, 0) + rng.uniform(-0.5, 0.5, (H, W))
    ys = np.linspace(lo_v, hi_v, H)[:, None].repeat(W, 1) + rng.uniform(-0.5, 0.5, (H, W))
    ims = dict(a_black=np.full((H, W), lo_v), b_white=np.full((H, W), hi_v), c_rx=xs, d_ry=ys)
    paths = []
    for k, im in ims.items():
        paths.append(f"{d}/{k}.png")
        cv2.imwrite(paths[-1], np.round(im).astype(np.uint8))
    c = runner.channel0(runner.load(paths))
    lo, hi = c[0].mean(), c[1].mean()
    fx = (c[2] - lo) / (hi - lo) * (W - 1)  # original x at each model pixel
    fy = (c[3] - lo) / (hi - lo) * (H - 1)
    h, w = fx.shape
    m = 6  # stay clear of resampling ringing at the borders
    um = np.arange(w)[m:w - m]
    vm = np.arange(h)[m:h - m]
    sx, ox = np.polyfit(um, fx[m:h - m, m:w - m].mean(0), 1)
    sy, oy = np.polyfit(vm, fy[m:h - m, m:w - m].mean(1), 1)
    for p in paths:
        os.remove(p)
    os.rmdir(d)
    return dict(sx=float(sx), ox=float(ox), sy=float(sy), oy=float(oy), w=int(w), h=int(h), W=int(W), H=int(H))


def frame_order(key, n, how):
    if how == "sorted":
        return np.arange(n)
    seed = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).permutation(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(RUNNERS))
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--order", default="sorted", choices=["sorted", "shuffled"])
    ap.add_argument("--views", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min_free_gb", type=float, default=12)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    clips = pd.read_csv(a.clips)
    # optional per-row routing: name (output file), dense (0/1), order (sorted/shuffled); CLI values otherwise
    clips["name"] = clips["name"] if "name" in clips else a.name
    clips["dense"] = clips["dense"].astype(bool) if "dense" in clips else a.dense
    clips["order"] = clips["order"] if "order" in clips else a.order
    tag = f"{a.model}_{a.name}"
    done = {}
    for nm in clips.name.unique():
        p = f"{a.out}/{a.model}_{nm}.npz"
        done[nm] = dict(np.load(p)) if os.path.exists(p) else {}
        if clips[clips.name == nm].dense.any():
            os.makedirs(f"{a.out}/{a.model}_{nm}_dense", exist_ok=True)
    jobs = []
    for r in clips.itertuples():
        if r.key in done[r.name]:
            continue
        paths = sorted(glob.glob(f"{r.dir}/*.png") + glob.glob(f"{r.dir}/*.jpg"))
        if a.views:
            paths = paths[:a.views]
        order = frame_order(r.key, len(paths), r.order)
        jobs.append((r.key, paths, order, r.name, r.dense))
    if a.limit:
        jobs = jobs[:a.limit]
    n_done = sum(len(d) for d in done.values())

    fv = free_vram_gb()
    print(f"[{tag}] free VRAM before load: {fv:.1f} GB; {len(jobs)} clips to run ({n_done} already done)", flush=True)
    while fv < a.min_free_gb:  # shared GPU rule: never launch below the limit; wait for room instead
        print(f"[{tag}] free VRAM {fv:.1f} GB < {a.min_free_gb} GB; waiting", flush=True)
        time.sleep(60)
        fv = free_vram_gb()
    if not jobs:
        print("INFER_DONE", flush=True)
        return

    t_start = time.time()
    runner = RUNNERS[a.model]()
    W0, H0 = Image.open(jobs[0][1][0]).size
    pmap = pixel_map(runner, W0, H0)
    print(f"[{tag}] pixel map {pmap}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t_loaded = time.time()

    def load(job):
        return runner.load([job[1][i] for i in job[2]])

    pool = ThreadPoolExecutor(4)
    futs = [pool.submit(load, j) for j in jobs[:6]]
    def save_all():
        for nm, d in done.items():
            if d:
                np.savez(f"{a.out}/{a.model}_{nm}.npz", **d)

    for n, (key, paths, order, nm, dense) in enumerate(jobs):
        x = futs[n].result()
        futs[n] = None
        if n + 6 < len(jobs):
            futs.append(pool.submit(load, jobs[n + 6]))
        res = runner.run(x, dense)
        inv = np.argsort(order)  # back to sorted-frame order
        done[nm][key] = res["poses"][inv].astype(np.float64)
        if dense:
            np.savez(f"{a.out}/{a.model}_{nm}_dense/{key}.npz", local_pts=res["local_pts"][inv].astype(np.float16),
                     conf=res["conf"][inv].astype(np.float16), order=order, **{f"map_{k}": v for k, v in pmap.items()})
        if (n + 1) % 100 == 0 or n + 1 == len(jobs):
            save_all()
            el = time.time() - t_loaded
            print(f"[{tag}] {n + 1}/{len(jobs)} clips, {el / (n + 1):.3f} s/clip, "
                  f"peak alloc {torch.cuda.max_memory_allocated() / 2**30:.1f} GB, free VRAM {free_vram_gb():.1f} GB",
                  flush=True)

    info = dict(model=a.model, ckpt=runner.ckpt, clips=a.clips, names=list(done), views=a.views,
                n_clips_this_run=len(jobs), n_total=sum(len(d) for d in done.values()), load_s=t_loaded - t_start,
                infer_s=time.time() - t_loaded, peak_alloc_gb=torch.cuda.max_memory_allocated() / 2**30,
                torch=torch.__version__, dtype=str(DTYPE), pixel_map=pmap)
    with open(f"{a.out}/{tag}_run.json", "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info), flush=True)
    print("INFER_DONE", flush=True)


if __name__ == "__main__":
    main()
