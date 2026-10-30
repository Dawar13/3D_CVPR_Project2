"""CPU-only COLMAP / GLOMAP on the EXP2 TUM windows (final run). Same settings as run_sfm.py, but SIFT runs on the
CPU and is computed once per frame (frames are shared by overlapping windows) instead of once per window.

Per frame (cond, seq, t): single-image database _work/framefeat/{cond}/seq{s}/{t}.db with SIFT (CPU, default
options, max 8192 features) on the prepped pinhole frame data/prep/seq{s}/cam{0|1}/{t}.png and a PINHOLE camera
with K_new. CPU SIFT is deterministic, so these features equal per-window extraction.
Per window: the 16 frame databases are merged (pycolmap.Database.merge; each frame keeps its own camera entry, all
with the identical fixed K_new, so geometry is the same as one shared camera because no intrinsics are refined),
then exhaustive matching + default two-view verification (CPU, seed 0), then the COLMAP incremental mapper and the
GLOMAP global mapper (seed 0, no focal / principal point / extra refinement). Largest model kept.

Outputs: {mapper}/recon/seq{s}/{wid:04d}_{cond}/ (binary model), jobs_cpu.jsonl (one record per window).
usage: python run_sfm3.py --procs 7
"""
import argparse
import json
import multiprocessing
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_sfm import MAPPERS, N_FRAMES, OUT, RES_TUM, WIN_DIR, _largest, load_K  # noqa: E402

PREP = os.path.expanduser("~/rs3d/data/prep")
CAM = {"GS": "cam0", "RS": "cam1"}


def frame_db(out, cond, s, t):
    """Single-image SIFT database for one frame; computed once, shared by all windows (atomic rename)."""
    import pycolmap
    d = f"{out}/_work/framefeat/{cond}/seq{s}"
    db = f"{d}/{t}.db"
    if os.path.exists(db):
        return db
    os.makedirs(d, exist_ok=True)
    tmp = f"{d}/{t}.{os.getpid()}.tmp.db"
    fx, fy, cx, cy = load_K()
    reader = pycolmap.ImageReaderOptions(camera_model="PINHOLE", camera_params=f"{fx},{fy},{cx},{cy}")
    ext = pycolmap.FeatureExtractionOptions()
    ext.num_threads = 1
    ext.use_gpu = False
    pycolmap.extract_features(tmp, f"{PREP}/seq{s}/{CAM[cond]}", image_names=[f"{t}.png"],
                              camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader, extraction_options=ext,
                              device=pycolmap.Device.cpu)
    x = pycolmap.Database.open(tmp)
    n = [x.num_keypoints_for_image(i.image_id) for i in x.read_all_images()]
    x.close()
    if len(n) != 1 or n[0] == 0:
        os.remove(tmp)
        raise RuntimeError(f"feature extraction failed for {cond} seq{s} {t}: {n}")
    os.replace(tmp, db)
    return db


def window_job(args):
    import pycolmap
    pycolmap.logging.minloglevel = 2
    s, wid, key, cond, out = args
    tag = f"{wid:04d}_{cond}"
    src = f"{WIN_DIR}/seq{s}/{tag}"
    work = f"{out}/_work/win/seq{s}_{tag}"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    t0 = time.time()
    res = {}
    try:
        names = sorted(x for x in os.listdir(src) if x.endswith(".png"))
        assert len(names) == N_FRAMES
        ts = [n.split("_")[1][:-4] for n in names]
        k_of = {f"{t}.png": k for k, t in enumerate(ts)}
        dbs = [frame_db(out, cond, s, t) for t in ts]
        t_feat = time.time() - t0
        # merge the 16 frame databases
        cur = dbs[0]
        for i, d in enumerate(dbs[1:], 1):
            nxt = f"{work}/merge{i}.db"
            a, b, m = pycolmap.Database.open(cur), pycolmap.Database.open(d), pycolmap.Database.open(nxt)
            pycolmap.Database.merge(a, b, m)
            a.close(), b.close(), m.close()
            if i > 1:
                os.remove(cur)
            cur = nxt
        db = f"{work}/database.db"
        os.replace(cur, db)
        t1 = time.time()
        mo = pycolmap.FeatureMatchingOptions()
        mo.num_threads = 1
        mo.use_gpu = False
        tv = pycolmap.TwoViewGeometryOptions()
        tv.ransac.random_seed = 0
        pycolmap.match_exhaustive(db, matching_options=mo, verification_options=tv, device=pycolmap.Device.cpu)
        t_match = time.time() - t1
        x = pycolmap.Database.open(db)
        n_kp = [x.num_keypoints_for_image(i.image_id) for i in x.read_all_images()]
        n_img, n_cam = x.num_images(), x.num_cameras()
        n_pairs_verified = int(x.num_verified_image_pairs())
        x.close()
        assert n_img == N_FRAMES and min(n_kp) > 0, (n_img, n_kp)
        img_dir = f"{PREP}/seq{s}/{CAM[cond]}"
        for mapper in MAPPERS:
            t1 = time.time()
            dbm = f"{work}/db_{mapper}.db"
            shutil.copy(db, dbm)
            sp = f"{work}/sparse_{mapper}"
            os.makedirs(sp)
            err = ""
            try:
                if mapper == "colmap":
                    o = pycolmap.IncrementalPipelineOptions()
                    o.num_threads = 1
                    o.random_seed = 0
                    o.ba_refine_focal_length = False
                    o.ba_refine_principal_point = False
                    o.ba_refine_extra_params = False
                    o.mapper.abs_pose_refine_focal_length = False
                    o.mapper.abs_pose_refine_extra_params = False
                    o.mapper.num_threads = 1
                    o.mapper.random_seed = 0
                    o.triangulation.random_seed = 0
                    recs = pycolmap.incremental_mapping(dbm, img_dir, sp, options=o)
                else:
                    o = pycolmap.GlobalPipelineOptions()
                    o.num_threads = 1
                    o.random_seed = 0
                    m = o.mapper
                    m.num_threads = 1
                    m.random_seed = 0
                    m.rotation_averaging.random_seed = 0
                    m.global_positioning.use_gpu = False
                    m.global_positioning.random_seed = 0
                    m.bundle_adjustment.refine_focal_length = False
                    m.bundle_adjustment.refine_principal_point = False
                    m.bundle_adjustment.refine_extra_params = False
                    m.bundle_adjustment.ceres.use_gpu = False
                    recs = pycolmap.global_mapping(dbm, img_dir, sp, options=o)
            except Exception as e:
                recs, err = {}, f"{type(e).__name__}: {e}"
            rec = _largest(recs)
            P = np.full((N_FRAMES, 4, 4), np.nan)
            if rec is not None:
                for img in rec.images.values():
                    if img.has_pose:
                        T = np.eye(4)
                        T[:3, :4] = img.cam_from_world().inverse().matrix()
                        P[k_of[img.name]] = T
                dst = f"{out}/{mapper}/recon/seq{s}/{tag}"
                os.makedirs(dst, exist_ok=True)
                rec.write_binary(dst)
            res[mapper] = dict(n_models=len(recs), n_reg=int(np.isfinite(P[:, 0, 0]).sum()), err=err,
                               n_points3D=int(rec.num_points3D()) if rec is not None else 0,
                               mean_reproj=float(rec.compute_mean_reprojection_error()) if rec is not None else np.nan,
                               t_map=time.time() - t1, poses=P.tolist())
        status = "ok"
        info = dict(t_feat=t_feat, t_match=t_match, n_kp_median=float(np.median(n_kp)), n_kp=n_kp,
                    n_cameras=n_cam, n_pairs_verified=n_pairs_verified)
    except Exception:
        status, info = "error", dict(trace=traceback.format_exc())
    shutil.rmtree(work, ignore_errors=True)
    return dict(seq=int(s), window_id=int(wid), key=key, cond=cond, status=status, t_total=time.time() - t0,
                sift="cpu", pycolmap=pycolmap.__version__, **info, mappers=res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="GS,RS")
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--keys", default="")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    win = pd.read_csv(f"{RES_TUM}/windows.csv")
    win = win[win.keep]
    if a.keys:
        win = win[win.key.isin(a.keys.split(","))]
    # windows in a fixed random order (seed 0), both conditions of a window back to back, so a partial run is a
    # random subset of windows with GS and RS paired (the same order is used by barath_windows.py)
    rows = list(win.itertuples())
    rows = [rows[i] for i in np.random.default_rng(0).permutation(len(rows))]
    jobs = [(int(r.seq), int(r.window_id), r.key, c) for r in rows for c in a.conds.split(",")]
    log = f"{a.out}/jobs_cpu.jsonl"
    done = set()
    if os.path.exists(log):
        for line in open(log):
            j = json.loads(line)
            if j["status"] == "ok":
                done.add((j["key"], j["cond"]))
    todo = [(s, w, k, c, a.out) for s, w, k, c in jobs if (k, c) not in done]
    print(f"{len(jobs)} jobs, {len(done)} done, {len(todo)} to run, procs={a.procs}", flush=True)
    ctx = multiprocessing.get_context("spawn")
    t0, n = time.time(), 0
    with ProcessPoolExecutor(a.procs, mp_context=ctx) as ex, open(log, "a") as f:
        futs = [ex.submit(window_job, j) for j in todo]
        for fu in as_completed(futs):
            r = fu.result()
            f.write(json.dumps(r) + "\n")
            f.flush()
            n += 1
            m = r.get("mappers", {})
            print(f"[{n}/{len(todo)} {time.time() - t0:.0f}s] {r['key']}_{r['cond']} {r['status']} t={r['t_total']:.1f}s "
                  + " ".join(f"{k}:{v['n_reg']}" for k, v in m.items()), flush=True)
    print("SFM_DONE", flush=True)


if __name__ == "__main__":
    main()
