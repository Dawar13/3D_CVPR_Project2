import sys, time, cProfile, pstats
import numpy as np
sys.path.insert(0, r"C:\Users\djtde\Downloads\3D_CVPR_Exp_2_Rolling_Shutter\src")
from rsgen.core import Camera, Shutter, render
from rsgen.scenes import test_room
from rsgen.motion import sample_clip_trajectory
import open3d as o3d
sc = test_room(); rng = np.random.default_rng(0)
cam = Camera.from_hfov(640, 480, 70)
traj, info = sample_clip_trajectory(sc, "handheld", np.radians(150), 1.0, -0.1, 2.4, rng)
N = 1228800
o = np.tile([[3.0, 2.5, 1.5]], (N, 1)).astype(np.float32); d = rng.normal(size=(N, 3)).astype(np.float32)
r = o3d.core.Tensor(np.concatenate([o, d], 1))
for nt in [0, 12]:
    t = time.time(); a = sc.rc.cast_rays(r, nthreads=nt); a["t_hit"].numpy(); print("cast 1.2M rays nthreads", nt, f"{time.time()-t:.3f}s")
cProfile.run("render(sc, traj, cam, Shutter(0.03), 1.0, ssaa=2)", "p.out")
pstats.Stats("p.out").sort_stats("cumtime").print_stats(12)
