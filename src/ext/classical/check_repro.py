"""Checks on the Barath repo's own TUM-RS run (seq 1, strides 10/20, cached RoMa ACs; CPU only):
 1. Which camera the repo loader reads: tum_rs.py takes mav0/cam0 and calls it the rolling-shutter camera. On TUM RS-VIO
    cam0 is the global-shutter camera (Schubert et al. 2019; our EXP0 notes). So the estimated RS motion should be ~0.
 2. Re-solve 7AC on the cached ACs (same config) and record the per-frame RS angular motion w1, w2 (rad per readout)
    against the loader's GT omega (mocap rotation over the readout), and against predicting zero.
 3. Fraction of ACs whose affine part is exactly the identity (symmetric-warp lookup in rs_ac_extractor).
usage: python check_repro.py
"""
import os
import sys
from argparse import Namespace

import h5py
import numpy as np
import pandas as pd

REPO = os.path.expanduser("~/rs3d/third_party/rolling_shutter_made_practical")
TESTS = f"{REPO}/external/superansac/tests"
sys.path.insert(0, f"{TESTS}/essential_matrix")
sys.path.insert(0, f"{TESTS}/datasets")
sys.path.insert(0, TESTS)
import tester_rs_superansac as T  # noqa: E402
from tum_rs import TumRS  # noqa: E402

OUT = os.path.expanduser("~/rs3d/results/exp2/baselines/classical/barath/repro")
CFG = Namespace(min_iterations=50, max_iterations=1000, sampler="PROSAC", scoring="MSAC", lo="NestedRANSAC",
                fo="LSQ", threshold=1.0, ac_weight=0.0, nonmin_solver="pc")

ds = TumRS(os.path.expanduser("~/rs3d/ext/classical/tum_rs_link"), sequences=[1], strides=[10, 20], max_pairs=100)
print("loader image dir example:", ds.data[0]["img_path1"])
h5 = h5py.File([f"{OUT}/{x}" for x in os.listdir(OUT) if x.endswith(".h5")][0], "r")
K = ds.K_undist
H, W = ds.height, ds.width
rows = []
nI = nT = 0
for i, d in enumerate(ds.data):
    acs = h5["acs"][str(i)][:]
    J = acs[:, [6, 7, 9, 10]]
    nI += int((np.abs(J - np.array([1, 0, 0, 1])) < 1e-9).all(1).sum())
    nT += len(acs)
    cfg = T.build_config(CFG, threshold=CFG.threshold / (0.5 * (K[0, 0] + K[1, 1])))
    model, inl, ms = T.run_rs_ac_7direct(acs, K[1, 1] / H, [W, H, W, H], cfg, 0.0, "pc")
    if model is None or len(inl) < 7:
        continue
    _, _, w1, v1, w2, v2 = T.extract_pose_from_rs_model(np.asarray(model))
    for w, g in ((w1, d["omega1_gt"]), (w2, d["omega2_gt"])):
        rows.append(dict(pair=i, stride=d["stride"], gt_norm=np.linalg.norm(g), est_norm=np.linalg.norm(w),
                         err=np.linalg.norm(w - g), err_flipped=np.linalg.norm(-w - g),
                         cos=float(w @ g / (np.linalg.norm(w) * np.linalg.norm(g) + 1e-12))))
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/check_omega_cam0.csv", index=False)
fast = df[df.gt_norm > np.median(df.gt_norm)]
summ = dict(n_frames=len(df), gt_norm_med=df.gt_norm.median(), est_norm_med=df.est_norm.median(),
            err_med=df.err.median(), zero_pred_err_med=df.gt_norm.median(), err_flipped_med=df.err_flipped.median(),
            cos_med_upper_half=fast.cos.median(), frac_identity_affine=nI / max(nT, 1))
print(summ)
pd.Series(summ).to_csv(f"{OUT}/check_omega_cam0_summary.csv")
r = pd.read_csv(f"{OUT}/results_rs_tum_seq1.csv")
print("gs_5pt pairs with R_err > 170 deg:", int((r[r.method == "gs_5pt"].R_err_deg > 170).sum()), "of",
      int((r.method == "gs_5pt").sum()))
