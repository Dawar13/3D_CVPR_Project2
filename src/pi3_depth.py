"""EXP1b helper: metric z-depth for every RS frame, for the rotation+translation unrolling.

pi3 runs on the original RS clips; its depth is scaled to metres with the Sim(3) scale between its camera
centres and GT (oracle setting). Writes data/pi3_depth/seq{s}.npz  {timestamp: (112, 140) float16 depth in m}.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from common import RES, WIN_DIR, umeyama_sim3
from infer import DEV, DTYPE, Pi3Runner, free_vram_gb
from unroll import DEPTH_DIR


def main():
    os.makedirs(DEPTH_DIR, exist_ok=True)
    win = pd.read_csv(f"{RES}/windows.csv")
    win = win[win.keep]
    gt = np.load(f"{RES}/gt_poses.npz")
    print(f"free VRAM {free_vram_gb():.1f} GB", flush=True)
    runner = Pi3Runner()
    store, n_run = {}, 0
    for r in win.itertuples():
        paths = sorted(glob.glob(f"{WIN_DIR}/seq{r.seq}/{r.window_id:04d}_RS/*.png"))
        ts = [int(os.path.basename(p)[3:-4]) for p in paths]
        have = store.setdefault(r.seq, {})
        if all(t in have for t in ts):
            continue
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
            res = runner.model(runner.load(paths).to(DEV)[None])
        P = res["camera_poses"][0].float().cpu().numpy()
        s, _, _ = umeyama_sim3(P[:, :3, 3], gt[f"{r.key}_RS"][:, :3, 3])
        z = F.avg_pool2d(res["local_points"][0, ..., 2].float()[:, None], 4)[:, 0].cpu().numpy() * s
        for k, t in enumerate(ts):
            if t not in have:
                have[t] = np.clip(z[k], 0.05, 50).astype(np.float16)
        n_run += 1
        if n_run % 100 == 0:
            print(f"{n_run} clips run, {sum(len(v) for v in store.values())} frames with depth", flush=True)
    for s, d in store.items():
        np.savez(f"{DEPTH_DIR}/seq{s}.npz", **{str(t): v for t, v in d.items()})
        print(f"seq{s}: {len(d)} frames, median depth {np.median([np.median(v) for v in d.values()]):.2f} m",
              flush=True)
    print("DEPTH_DONE", flush=True)


if __name__ == "__main__":
    main()
