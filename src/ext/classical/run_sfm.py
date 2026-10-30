"""EXP2 classical GS-model SfM reference on TUM RS-VIO windows: COLMAP (incremental) and GLOMAP (global), both
through pycolmap 4.2.1 (COLMAP 4.2.1, which ships GLOMAP as `global_mapping`).

Per window folder (16 undistorted pinhole frames): SIFT (CPU) -> exhaustive matching + two-view verification ->
  * incremental mapper  (COLMAP)
  * global mapper       (GLOMAP)
One shared PINHOLE camera with the known K_new, held fixed in every BA (no focal / pp / extra refinement).
If a mapper yields several models, the one with the most registered images is kept.

Outputs (under --out):
  {mapper}/recon/seq{s}/{wid:04d}_{cond}/   COLMAP binary model (raw output; kept)
  {mapper}/jobs.jsonl                        one line per window: status, n_reg, timing, poses (16x4x4, NaN = unreg.)
The npz files are assembled by collect_sfm.py.

usage: python run_sfm.py --conds GS,RS --procs 5 --threads 2 [--gpu 1] [--limit N] [--keys s1_w0001,...]
SIFT runs on the GPU with --gpu 1 (pycolmap-cuda12 4.2.1, env ~/rs3d/envs/classical_cuda); the mappers are CPU.
"""
import argparse
import json
import os
import shutil
import sys
import time
import traceback
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/rs3d")
WIN_DIR = f"{ROOT}/data/windows"
RES_TUM = f"{ROOT}/results/exp2/gap/tum"
OUT = f"{ROOT}/results/exp2/baselines/classical"
N_FRAMES = 16
MAPPERS = ("colmap", "glomap")
MIN_FREE_MB = 12000  # shared-GPU rule: start GPU work only with >= 12 GB free


def load_K():
    with open(f"{RES_TUM}/prep_info.json") as f:
        K = np.array(json.load(f)["K_new"])
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def _largest(recs):
    if not recs:
        return None
    return max(recs.values(), key=lambda r: r.num_reg_images())


def _poses(rec):
    P = np.full((N_FRAMES, 4, 4), np.nan)
    if rec is None:
        return P
    for img in rec.images.values():
        if not img.has_pose:
            continue
        k = int(img.name.split("_")[0])
        T = np.eye(4)
        T[:3, :4] = img.cam_from_world().inverse().matrix()  # cam-to-world, OpenCV convention (COLMAP)
        P[k] = T
    return P


def gpu_free_mb():
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return int(out.split()[0])
    except Exception:
        return -1


def run_job(job):
    import pycolmap
    pycolmap.logging.minloglevel = 2  # warnings and errors only
    s, wid, key, cond, threads, out, gpu = job
    dev = pycolmap.Device.cuda if gpu else pycolmap.Device.cpu
    tag = f"{wid:04d}_{cond}"
    src = f"{WIN_DIR}/seq{s}/{tag}"
    work = f"{out}/_work/seq{s}_{tag}"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    res = {}
    t0 = time.time()
    if gpu:  # SiftGPU corrupted its CUDA context under memory pressure; wait for headroom
        while gpu_free_mb() < MIN_FREE_MB:
            time.sleep(15)
    t_wait = time.time() - t0
    try:
        names = sorted(n for n in os.listdir(src) if n.endswith(".png"))
        assert len(names) == N_FRAMES, f"{src}: {len(names)} frames"
        fx, fy, cx, cy = load_K()
        db = f"{work}/database.db"
        reader = pycolmap.ImageReaderOptions(camera_model="PINHOLE", camera_params=f"{fx},{fy},{cx},{cy}")
        ext = pycolmap.FeatureExtractionOptions()
        ext.num_threads = threads
        ext.use_gpu = gpu
        pycolmap.extract_features(db, src, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=reader, extraction_options=ext, device=dev)
        mo = pycolmap.FeatureMatchingOptions()
        mo.num_threads = threads
        mo.use_gpu = gpu
        tv = pycolmap.TwoViewGeometryOptions()
        tv.ransac.random_seed = 0
        pycolmap.match_exhaustive(db, matching_options=mo, verification_options=tv, device=dev)
        t_feat = time.time() - t0
        d = pycolmap.Database.open(db)
        n_kp = [d.num_keypoints_for_image(i.image_id) for i in d.read_all_images()]
        n_pairs_verified = int(d.num_verified_image_pairs())
        d.close()
        if len(n_kp) != N_FRAMES or min(n_kp) == 0:  # GPU extraction failure -> retry later, never score it
            raise RuntimeError(f"FEATURE_FAILURE n_images={len(n_kp)} n_kp={n_kp}")
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
                    o.num_threads = threads
                    o.random_seed = 0
                    o.ba_refine_focal_length = False
                    o.ba_refine_principal_point = False
                    o.ba_refine_extra_params = False
                    o.mapper.abs_pose_refine_focal_length = False
                    o.mapper.abs_pose_refine_extra_params = False
                    o.mapper.num_threads = threads
                    o.mapper.random_seed = 0
                    o.triangulation.random_seed = 0
                    recs = pycolmap.incremental_mapping(dbm, src, sp, options=o)
                else:
                    o = pycolmap.GlobalPipelineOptions()
                    o.num_threads = threads
                    o.random_seed = 0
                    m = o.mapper
                    m.num_threads = threads
                    m.random_seed = 0
                    m.rotation_averaging.random_seed = 0
                    m.global_positioning.use_gpu = False
                    m.global_positioning.random_seed = 0
                    m.bundle_adjustment.refine_focal_length = False
                    m.bundle_adjustment.refine_principal_point = False
                    m.bundle_adjustment.refine_extra_params = False
                    m.bundle_adjustment.ceres.use_gpu = False
                    recs = pycolmap.global_mapping(dbm, src, sp, options=o)
            except Exception as e:  # mapper crash counts as a failure, recorded verbatim
                recs, err = {}, f"{type(e).__name__}: {e}"
            rec = _largest(recs)
            P = _poses(rec)
            n_reg = int(np.isfinite(P[:, 0, 0]).sum())
            dst = f"{out}/{mapper}/recon/seq{s}/{tag}"
            if rec is not None:
                os.makedirs(dst, exist_ok=True)
                rec.write_binary(dst)
            res[mapper] = dict(n_models=len(recs), n_reg=n_reg, err=err,
                               n_points3D=int(rec.num_points3D()) if rec is not None else 0,
                               mean_reproj=float(rec.compute_mean_reprojection_error()) if rec is not None else np.nan,
                               t_map=time.time() - t1, poses=P.tolist())
        status = "ok"
        info = dict(t_feat=t_feat, t_wait_gpu=t_wait, n_kp_median=float(np.median(n_kp)), n_kp=n_kp,
                    n_pairs_verified=n_pairs_verified)
    except Exception:
        status = "error"
        info = dict(trace=traceback.format_exc())
    shutil.rmtree(work, ignore_errors=True)
    return dict(seq=int(s), window_id=int(wid), key=key, cond=cond, status=status, t_total=time.time() - t0,
                gpu_sift=bool(gpu), pycolmap=pycolmap.__version__, **info, mappers=res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="GS,RS")
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keys", default="")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--gpu", type=int, default=0, help="1 = SIFT extraction + matching on the GPU (pycolmap-cuda12)")
    ap.add_argument("--shuffle", type=int, default=1, help="process windows in a fixed random order (seed 0)")
    a = ap.parse_args()
    win = pd.read_csv(f"{RES_TUM}/windows.csv")
    win = win[win.keep]
    if a.keys:
        win = win[win.key.isin(a.keys.split(","))]
    jobs = [(int(r.seq), int(r.window_id), r.key, c) for r in win.itertuples() for c in a.conds.split(",")]
    if a.shuffle:
        jobs = [jobs[i] for i in np.random.default_rng(0).permutation(len(jobs))]
    if a.limit:
        jobs = jobs[:a.limit]
    log = f"{a.out}/jobs.jsonl"
    os.makedirs(a.out, exist_ok=True)
    done = set()
    if os.path.exists(log):
        with open(log) as f:
            for line in f:
                j = json.loads(line)
                if j["status"] == "ok":
                    done.add((j["key"], j["cond"]))
    todo = [(s, w, k, c, a.threads, a.out, bool(a.gpu)) for s, w, k, c in jobs if (k, c) not in done]
    print(f"{len(jobs)} jobs, {len(done)} done, {len(todo)} to run; procs={a.procs} threads={a.threads} "
          f"gpu={a.gpu}", flush=True)
    t0 = time.time()
    n = 0
    # one fresh process per job: a CUDA error inside SiftGPU is sticky for the whole process (seen 2026-10-29)
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(a.procs, mp_context=ctx, max_tasks_per_child=1) as ex, open(log, "a") as f:
        futs = [ex.submit(run_job, j) for j in todo]
        for fu in as_completed(futs):
            r = fu.result()
            f.write(json.dumps(r) + "\n")
            f.flush()
            n += 1
            m = r.get("mappers", {})
            print(f"[{n}/{len(todo)} {time.time() - t0:.0f}s] {r['key']}_{r['cond']} {r['status']} "
                  f"t={r['t_total']:.1f}s " + " ".join(f"{k}:{v['n_reg']}" for k, v in m.items()), flush=True)
    print("SFM_DONE", flush=True)


if __name__ == "__main__":
    main()
