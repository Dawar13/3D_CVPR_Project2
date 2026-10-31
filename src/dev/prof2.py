import sys, time
import numpy as np
sys.path.insert(0, r"C:\Users\djtde\Downloads\3D_CVPR_Exp_2_Rolling_Shutter\src")
from rsgen.scenes import test_room
import open3d as o3d
sc = test_room(); rng = np.random.default_rng(0)
N = 1228800
o = np.tile([[3.0, 2.5, 1.5]], (N, 1)); d = rng.normal(size=(N, 3))
T = {}
def tic(k, t0): T[k] = T.get(k, 0) + time.time() - t0; return time.time()
for _ in range(3):
    t = time.time()
    rays = o3d.core.Tensor(np.concatenate([o, d], 1).astype(np.float32)); t = tic("make tensor", t)
    ans = sc.rc.cast_rays(rays, nthreads=12); t = tic("cast", t)
    th = ans["t_hit"].numpy().astype(np.float64); t = tic("t_hit numpy", t)
    hit = np.isfinite(th)
    pid = ans["primitive_ids"].numpy()[hit].astype(np.int64); uv = ans["primitive_uvs"].numpy()[hit]; t = tic("pid uv", t)
    tri = sc.F[pid]; t = tic("tri gather", t)
    C = sc.C[tri]; t = tic("color gather", t)
    w = np.stack([1 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1]], 1)
    rgb = np.einsum("nk,nkc->nc", w, C); t = tic("einsum", t)
print({k: round(v / 3, 3) for k, v in T.items()})
