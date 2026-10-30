"""Barath (ECCV 2026) RS-aware relative pose (7-AC solver, SupeRANSAC) on EXP2 TUM windows, plus its GS-5PC variant
on the same correspondences.

Settings are the repo's real-data defaults (examples/run_real_data.py -> tester_rs_superansac.py):
  RoMa outdoor dense matcher, 500 sampled affine correspondences per pair (tester default --roma_samples 500),
  threshold 1 px (RS: / mean focal), MSAC scoring, PROSAC sampler, NestedRANSAC LO, LSQ final opt (3 iters each),
  50..1000 iterations, confidence 0.99, ac_weight 0, non-minimal solver 'pc'.
Images: our prepped 8-bit pinhole frames (K_new, same as the 3D models see), not the repo's own /256 + undistortion.
RoMa goes through roma_compat (drops romatch 0.1.x's batch dim so the repo's extractor runs unchanged).

Pairs: (k, k+gap) inside each window, gap in --gaps. Frame pairs shared by overlapping windows are solved once.
Outputs (under --out):
  acs/{cond}/seq{s}/{ta}_{tb}.npy   RoMa ACs per frame pair (12-col RS format); atomic per-pair files
  pairs_{cond}.csv            one row per (pair, method): R (row-major 9), t (3), omega1, v1, omega2, v2 (raw model
                              units: per full-frame tau in [-0.5, 0.5]), n_acs, n_inliers, time_ms

usage: python barath_windows.py --conds RS --gaps 1,2 --workers 4 [--keys ...] [--limit_windows N]
"""
import argparse
import csv
import os
import sys
import threading
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor

import cv2
import h5py
import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/rs3d")
REPO = f"{ROOT}/third_party/rolling_shutter_made_practical"
TESTS = f"{REPO}/external/superansac/tests"
sys.path.insert(0, f"{TESTS}/essential_matrix")
sys.path.insert(0, TESTS)
import tester_rs_superansac as T  # noqa: E402  (repo's validated tester: build_config, run_* helpers)
from rs_ac_extractor import extract_affine_correspondences_roma  # noqa: E402

WIN_DIR = f"{ROOT}/data/windows"
RES_TUM = f"{ROOT}/results/exp2/gap/tum"
OUT = f"{ROOT}/results/exp2/baselines/classical/barath"
N_FRAMES = 16

# tester defaults as used by examples/run_real_data.py
CFG = Namespace(min_iterations=50, max_iterations=1000, sampler="PROSAC", scoring="MSAC", lo="NestedRANSAC",
                fo="LSQ", threshold=1.0, ac_weight=0.0, nonmin_solver="pc", roma_samples=500)
METHODS = ("ac_rs_7direct", "gs_5pt")


def load_K():
    import json
    with open(f"{RES_TUM}/prep_info.json") as f:
        return np.array(json.load(f)["K_new"], dtype=np.float64)


def frame_list(seq, wid, cond):
    d = f"{WIN_DIR}/seq{seq}/{wid:04d}_{cond}"
    names = sorted(n for n in os.listdir(d) if n.endswith(".png"))
    assert len(names) == N_FRAMES
    return d, names


def solve_pair(acs, K, H, W):
    """Both methods on one pair's ACs. Returns list of dict rows."""
    fy_over_h = K[1, 1] / H
    sizes = [float(W), float(H), float(W), float(H)]
    rs_thr = CFG.threshold / (0.5 * (K[0, 0] + K[1, 1]))
    cfg_rs = T.build_config(CFG, threshold=rs_thr)
    cfg_gs = T.build_config(CFG, threshold=CFG.threshold)
    rows = []
    for m in METHODS:
        r = dict(method=m, n_acs=int(acs.shape[0]), n_inliers=0, time_ms=np.nan, ok=False)
        try:
            if m == "ac_rs_7direct":
                model, inl, ms = T.run_rs_ac_7direct(acs, fy_over_h, sizes, cfg_rs, CFG.ac_weight, CFG.nonmin_solver)
                r.update(n_inliers=len(inl), time_ms=ms)
                if model is not None and len(inl) >= 7:
                    Rm, t, w1, v1, w2, v2 = T.extract_pose_from_rs_model(np.asarray(model))
                    r.update(ok=True, R=Rm.ravel().tolist(), t=t.tolist(), w1=w1.tolist(), v1=v1.tolist(),
                             w2=w2.tolist(), v2=v2.tolist())
            else:
                _, Rm, t, ms, inl = T.run_gs_5pt(acs, K, K, sizes, cfg_gs)
                r.update(n_inliers=len(inl), time_ms=ms)
                if Rm is not None and len(inl) >= 5:
                    r.update(ok=True, R=np.asarray(Rm).ravel().tolist(), t=np.asarray(t).ravel().tolist())
        except Exception as e:
            r.update(err=f"{type(e).__name__}: {e}")
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="RS")
    ap.add_argument("--gaps", default="1,2")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keys", default="")
    ap.add_argument("--limit_windows", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    gaps = [int(g) for g in a.gaps.split(",")]
    win = pd.read_csv(f"{RES_TUM}/windows.csv")
    win = win[win.keep]
    if a.keys:
        win = win[win.key.isin(a.keys.split(","))]
    # fixed random window order (seed 0, as run_sfm3.py), then interleaved across speed bins (slow, medium, fast, ...)
    # so that a partial run (RoMa costs ~4 s per pair on the shared GPU) covers complete windows in every bin
    win = win.iloc[np.random.default_rng(0).permutation(len(win))].copy()
    win["bin"] = pd.cut(win.omega_mean, [0, 0.5, 1.5, np.inf], labels=["slow", "medium", "fast"], right=False)
    win["rank"] = win.groupby("bin", observed=True).cumcount()
    win = win.sort_values(["rank", "bin"], kind="stable")
    if a.limit_windows:
        win = win.iloc[:a.limit_windows]
    K = load_K()
    os.makedirs(f"{a.out}/acs", exist_ok=True)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from roma_compat import roma_outdoor_unbatched as roma_outdoor  # romatch 0.1.x batched-output shim
    roma = None
    lock = threading.Lock()
    for cond in a.conds.split(","):
        # unique frame pairs per seq
        pairs = {}
        for r in win.itertuples():
            d, names = frame_list(r.seq, r.window_id, cond)
            for g in gaps:
                for k in range(N_FRAMES - g):
                    ta, tb = names[k].split("_")[1][:-4], names[k + g].split("_")[1][:-4]
                    pairs.setdefault((int(r.seq), ta, tb), (f"{d}/{names[k]}", f"{d}/{names[k + g]}", g))
        csv_path = f"{a.out}/pairs_{cond}.csv"
        done = set()
        if os.path.exists(csv_path):
            p = pd.read_csv(csv_path, usecols=["seq", "ta", "tb"], dtype=str)
            done = set(zip(p.seq.astype(int), p.ta, p.tb))
        todo = [k for k in pairs if k not in done]  # dict insertion order = window order
        print(f"[{cond}] {len(win)} windows, {len(pairs)} unique pairs, {len(todo)} to do", flush=True)
        if not todo:
            continue
        fields = ["cond", "seq", "ta", "tb", "gap", "method", "ok", "n_acs", "n_inliers", "time_ms", "t_roma_s",
                  "R", "t", "w1", "v1", "w2", "v2", "err"]
        new = not os.path.exists(csv_path)
        fcsv = open(csv_path, "a", newline="")
        wr = csv.DictWriter(fcsv, fieldnames=fields, extrasaction="ignore")
        if new:
            wr.writeheader()
        ex = ThreadPoolExecutor(a.workers)
        futs = []
        t0 = time.time()
        h5s = {}

        def job(key, acs, g, t_roma):
            rows = solve_pair(acs, K, H, W)
            with lock:
                for r in rows:
                    for f in ("R", "t", "w1", "v1", "w2", "v2"):
                        if f in r:
                            r[f] = " ".join(f"{x:.9g}" for x in r[f])
                    wr.writerow(dict(cond=cond, seq=key[0], ta=key[1], tb=key[2], gap=g, t_roma_s=round(t_roma, 3),
                                     **r))
                fcsv.flush()

        for i, key in enumerate(todo):
            s, ta, tb = key
            pa, pb, g = pairs[key]
            adir = f"{a.out}/acs/{cond}/seq{s}"
            os.makedirs(adir, exist_ok=True)
            apath = f"{adir}/{ta}_{tb}.npy"
            img1 = cv2.imread(pa, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(pb, cv2.IMREAD_GRAYSCALE)
            H, W = img1.shape
            t1 = time.time()
            if os.path.exists(apath):
                acs = np.load(apath)
            else:
                if roma is None:
                    roma = roma_outdoor(device=a.device)
                try:
                    acs = extract_affine_correspondences_roma(img1, img2, K, K, height=H, width=W, roma_model=roma,
                                                              num_samples=CFG.roma_samples, device=a.device)
                except Exception as e:
                    print(f"  RoMa failed {key}: {e}", flush=True)
                    acs = np.zeros((0, 12))
                np.save(apath + ".tmp.npy", acs)
                os.replace(apath + ".tmp.npy", apath)
            futs.append(ex.submit(job, key, acs, g, time.time() - t1))
            if (i + 1) % 200 == 0:
                print(f"  [{cond} {i + 1}/{len(todo)} {time.time() - t0:.0f}s] last n_acs={acs.shape[0]}", flush=True)
        for fu in futs:
            fu.result()
        ex.shutdown()
        fcsv.close()
        print(f"[{cond}] done in {time.time() - t0:.0f}s", flush=True)
    print("BARATH_DONE", flush=True)


if __name__ == "__main__":
    main()
