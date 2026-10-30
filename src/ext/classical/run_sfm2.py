"""Two-stage version of run_sfm.py (same settings, same outputs) that keeps GPU use to ONE process:

  stage feat : one process, sequential over windows: SIFT extraction + exhaustive matching on the GPU
               (pycolmap-cuda12 4.2.1). Writes _work/feat/seq{s}_{wid}_{cond}.db once every image has keypoints.
               A SiftGPU CUDA failure is sticky for the process, so on any image with 0 keypoints the process
               exits with code 3 and the wrapper (run_sfm2_feat.sh) restarts it through gpu_slot.sh.
  stage map  : CPU pool. For each ready database: two-view geometric verification (CPU) + COLMAP incremental
               mapper + GLOMAP global mapper, record to jobs.jsonl, delete the database.

Settings identical to run_sfm.py: PINHOLE K_new shared & fixed, default SIFT (max 8192 features), exhaustive
matching, default two-view verification (RANSAC seed 0), mappers with seed 0 and no intrinsics refinement,
largest model kept.

usage: python run_sfm2.py feat [--conds GS,RS]
       python run_sfm2.py map --procs 7
"""
import argparse
import json
import multiprocessing
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_sfm import MAPPERS, N_FRAMES, OUT, RES_TUM, WIN_DIR, _largest, _poses, load_K  # noqa: E402

FEAT = f"{OUT}/_work/feat"


def all_jobs(conds):
    win = pd.read_csv(f"{RES_TUM}/windows.csv")
    win = win[win.keep]
    jobs = [(int(r.seq), int(r.window_id), r.key, c) for r in win.itertuples() for c in conds]
    return [jobs[i] for i in np.random.default_rng(0).permutation(len(jobs))]


def done_set(out):
    done = set()
    p = f"{out}/jobs.jsonl"
    if os.path.exists(p):
        for line in open(p):
            j = json.loads(line)
            if j["status"] == "ok":
                done.add((j["key"], j["cond"]))
    return done


def tag_of(s, wid, cond):
    return f"seq{s}_{wid:04d}_{cond}"


def stage_feat(a):
    import pycolmap
    pycolmap.logging.minloglevel = 2
    os.makedirs(FEAT, exist_ok=True)
    fails = {}
    fl = f"{FEAT}/feat_failures.jsonl"
    if os.path.exists(fl):
        for line in open(fl):
            t = json.loads(line)["tag"]
            fails[t] = fails.get(t, 0) + 1
    done = done_set(a.out)
    fx, fy, cx, cy = load_K()
    jobs = all_jobs(a.conds.split(","))
    n = 0
    t0 = time.time()
    for s, wid, key, cond in jobs:
        tag = tag_of(s, wid, cond)
        db = f"{FEAT}/{tag}.db"
        if ((key, cond) in done or any(os.path.exists(db + x) for x in ("", ".mapping", ".maperror"))
                or fails.get(tag, 0) >= 3):
            continue
        src = f"{WIN_DIR}/seq{s}/{wid:04d}_{cond}"
        names = sorted(x for x in os.listdir(src) if x.endswith(".png"))
        tmp = f"{FEAT}/{tag}.tmp.db"
        if os.path.exists(tmp):
            os.remove(tmp)
        t1 = time.time()
        reader = pycolmap.ImageReaderOptions(camera_model="PINHOLE", camera_params=f"{fx},{fy},{cx},{cy}")
        ext = pycolmap.FeatureExtractionOptions()
        ext.num_threads = 1
        ext.use_gpu = True
        pycolmap.extract_features(tmp, src, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=reader, extraction_options=ext, device=pycolmap.Device.cuda)
        mo = pycolmap.FeatureMatchingOptions()
        mo.num_threads = 1
        mo.use_gpu = True
        mo.skip_geometric_verification = True  # verification runs in the CPU stage
        pycolmap.match_exhaustive(tmp, matching_options=mo, device=pycolmap.Device.cuda)
        d = pycolmap.Database.open(tmp)
        n_kp = [d.num_keypoints_for_image(i.image_id) for i in d.read_all_images()]
        n_matched = int(d.num_matched_image_pairs())
        d.close()
        if len(n_kp) != N_FRAMES or min(n_kp) == 0:
            with open(fl, "a") as f:
                f.write(json.dumps(dict(tag=tag, n_kp=n_kp, time=time.time())) + "\n")
            os.remove(tmp)
            print(f"FEATURE_FAILURE {tag} n_kp={n_kp} -> exit 3 (fresh CUDA context)", flush=True)
            sys.exit(3)
        with open(f"{FEAT}/{tag}.json", "w") as f:
            json.dump(dict(t_feat_gpu=time.time() - t1, n_kp=n_kp, n_matched_pairs=n_matched), f)
        os.rename(tmp, db)
        n += 1
        if a.limit and n >= a.limit:
            break
        if n % 50 == 0:
            print(f"[feat {n} {time.time() - t0:.0f}s] {tag} n_kp_med={np.median(n_kp)}", flush=True)
    open(f"{FEAT}/ALL_DONE", "w").close()
    print("FEAT_DONE", flush=True)


def map_job(args):
    import pycolmap
    pycolmap.logging.minloglevel = 2
    s, wid, key, cond, out = args
    FEAT = f"{out}/_work/feat"
    tag = tag_of(s, wid, cond)
    src = f"{WIN_DIR}/seq{s}/{wid:04d}_{cond}"
    work = f"{out}/_work/map/{tag}"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    t0 = time.time()
    res = {}
    try:
        info = json.load(open(f"{FEAT}/{tag}.json"))
        db = f"{work}/database.db"
        shutil.copy(f"{FEAT}/{tag}.db.mapping", db)
        names = sorted(x for x in os.listdir(src) if x.endswith(".png"))
        with open(f"{work}/pairs.txt", "w") as f:
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    f.write(f"{names[i]} {names[j]}\n")
        tv = pycolmap.TwoViewGeometryOptions()
        tv.ransac.random_seed = 0
        t1 = time.time()
        pycolmap.verify_matches(db, f"{work}/pairs.txt", tv)
        t_verify = time.time() - t1
        d = pycolmap.Database.open(db)
        n_pairs_verified = int(d.num_verified_image_pairs())
        d.close()
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
                    recs = pycolmap.incremental_mapping(dbm, src, sp, options=o)
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
                    recs = pycolmap.global_mapping(dbm, src, sp, options=o)
            except Exception as e:
                recs, err = {}, f"{type(e).__name__}: {e}"
            rec = _largest(recs)
            P = _poses(rec)
            dst = f"{out}/{mapper}/recon/seq{s}/{wid:04d}_{cond}"
            if rec is not None:
                os.makedirs(dst, exist_ok=True)
                rec.write_binary(dst)
            res[mapper] = dict(n_models=len(recs), n_reg=int(np.isfinite(P[:, 0, 0]).sum()), err=err,
                               n_points3D=int(rec.num_points3D()) if rec is not None else 0,
                               mean_reproj=float(rec.compute_mean_reprojection_error()) if rec is not None else np.nan,
                               t_map=time.time() - t1, poses=P.tolist())
        status = "ok"
        info = dict(t_feat=info["t_feat_gpu"] + t_verify, t_verify=t_verify, n_kp_median=float(np.median(info["n_kp"])),
                    n_kp=info["n_kp"], n_pairs_verified=n_pairs_verified)
    except Exception:
        status, info = "error", dict(trace=traceback.format_exc())
    shutil.rmtree(work, ignore_errors=True)
    return dict(seq=int(s), window_id=int(wid), key=key, cond=cond, status=status, t_total=time.time() - t0,
                gpu_sift=True, two_stage=True, pycolmap=pycolmap.__version__, **info, mappers=res)


def stage_map(a):
    log = f"{a.out}/jobs.jsonl"
    meta = {tag_of(s, w, c): (s, w, k, c) for s, w, k, c in all_jobs(a.conds.split(","))}
    ctx = multiprocessing.get_context("spawn")
    ex = ProcessPoolExecutor(a.procs, mp_context=ctx)
    running = {}
    n, t0 = 0, time.time()
    with open(log, "a") as f:
        while True:
            ready = sorted(x[:-3] for x in os.listdir(FEAT) if x.endswith(".db") and not x.endswith(".tmp.db"))
            for tag in ready:
                if len(running) >= 2 * a.procs:
                    break
                os.rename(f"{FEAT}/{tag}.db", f"{FEAT}/{tag}.db.mapping")  # claimed
                running[tag] = ex.submit(map_job, (*meta[tag], a.out))
            for tag, fu in list(running.items()):
                if fu.done():
                    r = fu.result()
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                    if r["status"] == "ok":
                        os.remove(f"{FEAT}/{tag}.db.mapping")
                    else:
                        os.rename(f"{FEAT}/{tag}.db.mapping", f"{FEAT}/{tag}.db.maperror")
                    del running[tag]
                    n += 1
                    m = r.get("mappers", {})
                    print(f"[map {n} {time.time() - t0:.0f}s] {tag} {r['status']} t={r['t_total']:.1f}s "
                          + " ".join(f"{k}:{v['n_reg']}" for k, v in m.items()), flush=True)
            if not running and os.path.exists(f"{FEAT}/ALL_DONE") and not any(
                    x.endswith(".db") and not x.endswith(".tmp.db") for x in os.listdir(FEAT)):
                break
            time.sleep(2)
    ex.shutdown()
    print("MAP_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["feat", "map"])
    ap.add_argument("--conds", default="GS,RS")
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--limit", type=int, default=0, help="feat: stop after N windows (testing)")
    a = ap.parse_args()
    global FEAT
    FEAT = f"{a.out}/_work/feat"
    os.makedirs(FEAT, exist_ok=True)
    stage_feat(a) if a.stage == "feat" else stage_map(a)


if __name__ == "__main__":
    main()
