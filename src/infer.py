"""EXP0 inference: one model, every kept window, GS and RS. Images only, no priors. bf16, one clip at a time.

usage: python infer.py --model {mapanything,vggt,pi3} [--limit N]
writes results/exp0/preds/<model>.npz  (key s{seq}_w{wid}_{GS|RS} -> (16, 4, 4) cam-to-world, OpenCV)
       results/exp0/preds/<model>_run.json  (wall clock, peak GPU memory, versions)
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.expanduser("~/rs3d/third_party/pi3"))
from common import RES, WIN_DIR, set_suffix

DTYPE = torch.bfloat16
DEV = "cuda"


def free_vram_gb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return float(out.split()[0]) / 1024


class MapAnythingRunner:
    ckpt = "facebook/map-anything-apache"

    def __init__(self):
        from mapanything.models import MapAnything
        from mapanything.utils.image import load_images
        self.load_images = load_images
        self.model = MapAnything.from_pretrained(self.ckpt).to(DEV).eval()

    def load(self, paths):
        return self.load_images(paths)

    @torch.no_grad()
    def run(self, views):
        # masking only post-processes dense outputs; camera poses are unaffected, so it is off for speed
        preds = self.model.infer(views, memory_efficient_inference=False, use_amp=True, amp_dtype="bf16",
                                 apply_mask=False, mask_edges=False, apply_confidence_mask=False)
        return torch.cat([p["camera_poses"] for p in preds], 0).float().cpu().numpy()


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

    @torch.no_grad()
    def run(self, imgs):
        imgs = imgs.to(DEV)[None]
        with torch.autocast("cuda", dtype=DTYPE):  # as in the VGGT README detailed usage
            toks, _ = self.model.aggregator(imgs)
        pose_enc = self.model.camera_head(toks)[-1]
        extr, _ = self.dec(pose_enc, imgs.shape[-2:])  # (1, N, 3, 4) world-to-camera
        E = np.tile(np.eye(4), (extr.shape[1], 1, 1))
        E[:, :3, :] = extr[0].float().cpu().numpy()
        return np.linalg.inv(E)


class Pi3Runner:
    ckpt = "yyfz233/Pi3"

    def __init__(self):
        from pi3.models.pi3 import Pi3
        from pi3.utils.basic import load_images_as_tensor
        self.lit = load_images_as_tensor
        self.model = Pi3.from_pretrained(self.ckpt).to(DEV).eval()

    def load(self, paths):
        return self.lit(os.path.dirname(paths[0]), interval=1, verbose=False)  # PIXEL_LIMIT default 255000

    @torch.no_grad()
    def run(self, imgs):
        with torch.amp.autocast("cuda", dtype=DTYPE):
            res = self.model(imgs.to(DEV)[None])
        return res["camera_poses"][0].float().cpu().numpy()


RUNNERS = dict(mapanything=MapAnythingRunner, vggt=VGGTRunner, pi3=Pi3Runner)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(RUNNERS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--set", default="main", choices=["main", "control"])
    ap.add_argument("--cams", default="GS,RS", help="window folders to run, e.g. RSU for EXP1 unrolled RS")
    ap.add_argument("--tag", default="", help="suffix for the output file, e.g. _e1rot")
    a = ap.parse_args()
    sfx = set_suffix(a.set) + a.tag
    cams = a.cams.split(",")

    os.makedirs(f"{RES}/preds", exist_ok=True)
    out_path = f"{RES}/preds/{a.model}{sfx}.npz"
    done = dict(np.load(out_path)) if os.path.exists(out_path) else {}

    df = pd.read_csv(f"{RES}/windows{set_suffix(a.set)}.csv")
    df = df[df.keep]
    jobs = [(f"{r.key}_{cam}", sorted(glob.glob(f"{WIN_DIR}/seq{r.seq}/{r.window_id:04d}_{cam}/*.png")))
            for r in df.itertuples() for cam in cams]
    jobs = [j for j in jobs if j[0] not in done]
    if a.limit:
        jobs = jobs[:a.limit]

    fv = free_vram_gb()
    print(f"[{a.model}] free VRAM before load: {fv:.1f} GB; {len(jobs)} clips to run ({len(done)} already done)",
          flush=True)
    if fv < 12:
        sys.exit(f"free VRAM {fv:.1f} GB < 12 GB; not launching (shared GPU rule)")

    t_start = time.time()
    runner = RUNNERS[a.model]()
    torch.cuda.reset_peak_memory_stats()
    t_loaded = time.time()

    pool = ThreadPoolExecutor(4)
    futs = [pool.submit(runner.load, p) for _, p in jobs[:6]]
    for n, (key, paths) in enumerate(jobs):
        assert len(paths) == 16, (key, len(paths))
        x = futs[n].result()
        futs[n] = None  # drop the finished future so its image tensor is freed
        if n + 6 < len(jobs):
            futs.append(pool.submit(runner.load, jobs[n + 6][1]))
        done[key] = runner.run(x).astype(np.float64)
        if (n + 1) % 100 == 0 or n + 1 == len(jobs):
            np.savez(out_path, **done)
            el = time.time() - t_loaded
            print(f"[{a.model}] {n + 1}/{len(jobs)} clips, {el / (n + 1):.3f} s/clip, "
                  f"peak alloc {torch.cuda.max_memory_allocated() / 2**30:.1f} GB, free VRAM {free_vram_gb():.1f} GB",
                  flush=True)

    info = dict(model=a.model, ckpt=runner.ckpt, n_clips_this_run=len(jobs), n_total=len(done),
                load_s=t_loaded - t_start, infer_s=time.time() - t_loaded,
                peak_alloc_gb=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gb=torch.cuda.max_memory_reserved() / 2**30,
                torch=torch.__version__, dtype=str(DTYPE))
    with open(f"{RES}/preds/{a.model}{sfx}_run.json", "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info), flush=True)
    print("INFER_DONE", flush=True)


if __name__ == "__main__":
    main()
