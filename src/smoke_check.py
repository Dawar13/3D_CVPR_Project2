"""Smoke check: metrics for whatever predictions exist (pose-convention sanity)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import RES, pose_metrics

gt = np.load(f"{RES}/gt_poses.npz")
for m in ["mapanything", "vggt", "pi3"]:
    p = f"{RES}/preds/{m}.npz"
    if not os.path.exists(p):
        continue
    pr = np.load(p)
    for k in list(pr.keys())[:4]:
        r = pose_metrics(pr[k], gt[k])
        print(m, k, {a: round(b, 4) for a, b in r.items()})
