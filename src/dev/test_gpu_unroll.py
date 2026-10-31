"""Equivalence test: gpu_unroll (torch) vs unroll.py (EXP1 numpy) on TUM's real cam1 calibration values."""
import sys, time
import numpy as np, torch, cv2
sys.path.insert(0, r"C:\Users\djtde\Downloads\3D_CVPR_Exp_2_Rolling_Shutter\src")
import unroll, gpu_unroll as G
from common import W, H, ROW_TIME, RS_MID_OFFSET
import json
info = json.load(open(r"C:\Users\djtde\Downloads\3D_CVPR_Exp_2_Rolling_Shutter\results\exp0\prep_info.json"))
c1 = info["calib"]["cam1"]
K, D, T = np.array(c1["K"]), np.array(c1["D"]), np.array(c1["T_cam_imu"])
unroll.load_calib = lambda: {"cam1": dict(K=K, D=D, T_cam_imu=T)}
K_new = np.array(info["K_new"])
ref = unroll.Unroller(K_new)
scan = G.ScanModel(ROW_TIME, H, W, "tb")
g = G.Unroller(K_new, (H, W), lambda X: G.fisheye_project(X, K, D), scan, device="cpu")
mx, my = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_new, (W, H), cv2.CV_32FC1)
print("zero-motion vs cv2 map, max px:", float(np.abs(g.u0.numpy() - mx).max()), float(np.abs(g.v0.numpy() - my).max()))
rng = np.random.default_rng(0)
cases = [(np.array([0.3, -2.5, 0.4]), None, None),
         (np.array([1.2, 0.8, -0.6]), np.array([0.6, -0.2, 0.9]), "depth")]
for w_c, v_c, dep in cases:
    depth = None
    if dep:
        yy, xx = np.mgrid[0:112, 0:140]
        depth = (1.5 + 0.8 * np.sin(xx / 20.0) + 0.3 * np.cos(yy / 15.0)).astype(np.float32)
    t0 = time.time()
    if depth is None:
        a, b = ref.maps(w_c)
    else:
        a, b = ref.maps(w_c, v_c, cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR))
    t1 = time.time()
    ga, gb = g.maps(w_c[None], None if v_c is None else v_c[None], None if depth is None else depth[None])
    t2 = time.time()
    da, db = np.abs(ga[0].numpy() - a), np.abs(gb[0].numpy() - b)
    sh = np.hypot(a - mx, b - my)
    print(f"case {'rot' if dep is None else 'rotrans'}: max shift applied {sh.max():.1f}px; numpy vs torch max |du| "
          f"{da.max():.2e} |dv| {db.max():.2e} px; numpy {t1-t0:.1f}s torch-cpu {t2-t1:.1f}s")
# image remap check: same maps, cv2.remap vs grid_sample
raw = rng.random((H, W)).astype(np.float32)
raw = cv2.GaussianBlur(raw, (0, 0), 3)
a, b = ref.maps(cases[0][0])
r_cv = cv2.remap(raw, a, b, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
r_t = G.Unroller.remap(torch.from_numpy(raw)[None, None], torch.from_numpy(a.astype(np.float64))[None], torch.from_numpy(b.astype(np.float64))[None])[0, 0].numpy()
print("cv2.remap vs grid_sample on a smooth image: mean |diff| %.2e, max %.2e (image range %.2f)" % (np.abs(r_cv - r_t).mean(), np.abs(r_cv - r_t).max(), raw.max() - raw.min()))
u8a = np.clip(255.0 * np.power(np.clip(r_cv, 0, 1), 1/2.2) + 0.5, 0, 255).astype(np.uint8)
u8b = G.tone_map_u8(torch.from_numpy(r_cv)).numpy()
print("tone map identical:", bool((u8a == u8b).all()))
