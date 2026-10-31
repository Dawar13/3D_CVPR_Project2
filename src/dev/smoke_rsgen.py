import sys, time
import numpy as np, cv2
sys.path.insert(0, r"C:\Users\djtde\Downloads\3D_CVPR_Exp_2_Rolling_Shutter\src")
from rsgen.core import Camera, Shutter, render
from rsgen.scenes import test_room
from rsgen.motion import sample_clip_trajectory
import open3d as o3d
t = time.time(); sc = test_room(); print("room", len(sc.V), "verts", len(sc.F), "tris", f"{time.time()-t:.1f}s")
# barycentric convention check
rng = np.random.default_rng(0)
o = np.tile([[3.0, 2.5, 1.5]], (2000, 1)); d = rng.normal(size=(2000, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
ans = sc.rc.cast_rays(o3d.core.Tensor(np.concatenate([o, d], 1).astype(np.float32)))
th = ans["t_hit"].numpy(); pid = ans["primitive_ids"].numpy().astype(np.int64); uv = ans["primitive_uvs"].numpy()
tri = sc.F[pid]; Vt = sc.V[tri]
for name, w in [("(1-u-v),u,v", np.stack([1-uv[:,0]-uv[:,1], uv[:,0], uv[:,1]],1)), ("u,v,(1-u-v)", np.stack([uv[:,0], uv[:,1], 1-uv[:,0]-uv[:,1]],1))]:
    P = np.einsum("nk,nkc->nc", w, Vt); err = np.linalg.norm(P - (o + th[:,None]*d), axis=1)
    print("barycentric", name, "max err m", float(err.max()))
cam = Camera.from_hfov(640, 480, 70)
traj, info = sample_clip_trajectory(sc, "handheld", np.radians(150), 1.0, -0.1, 2.4, rng)
w, v = traj.twist(np.linspace(0, 2.25, 50)); print("mean |w| deg/s", np.degrees(np.linalg.norm(w,axis=1).mean()), "mean |v|", np.linalg.norm(v,axis=1).mean())
for sh in [Shutter(0.0), Shutter(0.030)]:
    t = time.time(); out = render(sc, traj, cam, sh, 1.0, ssaa=2); dt_ = time.time()-t
    print("readout", sh.readout, f"render {dt_:.2f}s", "depth range", float(out["depth"][out["depth"]>0].min()), float(out["depth"].max()), "holes", int((out["depth"]==0).sum()))
    cv2.imwrite(rf"C:\Users\djtde\AppData\Local\Temp\claude\c--Users-djtde-Downloads-3D-CVPR-Exp-2-Rolling-Shutter\270d52c8-a58d-416d-8845-c86ba3700ab9\scratchpad\smoke_{int(sh.readout*1000)}ms.png", out["rgb"][..., ::-1])
