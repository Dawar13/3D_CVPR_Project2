"""Code-path smoke test of VGGTOmegaRunner WITHOUT the gated weights (random init; numbers are meaningless).

Checks the loader (default balanced 512), the pixel map through infer2's own pixel_map(), and that run() returns
the expected shapes. It does NOT validate conventions; that needs the real checkpoint (~/rs3d/.hf/token).
"""
import glob
import importlib.util
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/rs3d/ext/models"))
import models_extra as me  # noqa: E402

r = me.VGGTOmegaRunner.__new__(me.VGGTOmegaRunner)  # skip __init__ (it refuses to run without the token)
me._add_path(f"{me.TP}/vggt_omega")
from vggt_omega.models import VGGTOmega  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402

r.lp, r.dec = load_and_preprocess_images, encoding_to_camera
paths = sorted(glob.glob(os.path.expanduser("~/rs3d/data/windows/seq1/0001_GS/*.png")))
x = r.load(paths)
print("load", tuple(x.shape), x.dtype, float(x.min()), float(x.max()))
spec = importlib.util.spec_from_file_location("infer2_mod", os.path.expanduser("~/rs3d/src/infer2.py"))
inf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inf)  # defines functions only (main() is guarded)
print("pixel map", inf.pixel_map(r, 1280, 1024))
r.model = VGGTOmega().cuda().eval()
torch.cuda.reset_peak_memory_stats()
t = time.time()
out = r.run(x, True)
torch.cuda.synchronize()
print("random-init run ok:", {k: v.shape for k, v in out.items()}, f"{time.time() - t:.2f}s",
      f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
print("poses finite", bool(np.isfinite(out["poses"]).all()), "last row", out["poses"][0, 3])
