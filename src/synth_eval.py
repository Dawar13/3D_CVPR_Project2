"""EXP2 scoring on a synthetic set: pose metrics for every condition, dense metrics where dense output exists, and
V5 (the model's own RS-input points rebuilt row by row with the true motion).

Clips with < 0.20 m of camera travel (EXP0's static filter) keep rotation and dense metrics only.
Conditions are clip sub-folders (rs, gs, v3, v4, se2p-pi3, u2d-..., ...). Poses are scored against the frame
(centre-line) poses T_mid. Dense GT is the true surface point at each model pixel: for RS input the pixel's own
line pose and the RS depth, for the GS twin and for straightened images the frame pose and the GS depth.

Dense metrics, after one Sim(3) taking the predicted points onto their pixel-matched GT points:
  acc / comp / chamfer   mean nearest-neighbour distances (m), pred->GT, GT->pred, their mean
  corr_err               median distance between a pixel's aligned predicted point and its GT point (m)
  absrel / delta1        per-view median-scaled depth (pred local z vs GT z)
  plan_rms               mean RMS out-of-plane deviation (m) of the predicted points on the 3 largest GT planes
V5 rebuilds each RS-input point as T_pred · (T_mid^-1 T_line)_true · x_local, with the true translation converted to
model units by the Sim(3) scale of the model's own RS points. V5 changes points only, never poses (brief §7), so it has no pose metrics.

usage: python synth_eval.py --set ~/rs3d/data/synth/core_test --preds <dir> --models pi3,vggt --conds rs,gs,v3,v4
       [--dense] [--v5] [--views N] --out <csv>
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import MIN_PATH_M, SPEED_BINS, pose_metrics, umeyama_sim3
from dense_utils import Clip, list_clips, model_pixels_orig, pmap_of
from metrics2 import chamfer, depth_metrics, planarity, pose_auc

BINS = ["slow", "medium", "fast"]
STRAIGHTENED = ("gs", "v3", "v4", "se2p", "u2d", "cls")
TRANS_METRICS = ("ATE", "RPE_trans", "RTA5", "AUC30", "sim3_scale")
POSE_METRICS = TRANS_METRICS + ("RPE_rot", "RRA5")


def rs_geometry(cond):
    return not cond.startswith(STRAIGHTENED)


def dense_metrics(cl, P, dz, rs_gt, v5=False, s_pose=1.0, step=2, seed=0):
    rng = np.random.default_rng(seed)
    pm = pmap_of(dz)
    u, v = model_pixels_orig(pm)
    u, v = u[::step, ::step], v[::step, ::step]
    lp = dz["local_pts"][:, ::step, ::step].astype(np.float64)
    Xp, Xg, zp, zg, ok = [], [], [], [], []
    for i in range(len(P)):
        g, valid = cl.gt_points(i, u, v, rs_gt)
        x = lp[i]
        if v5:
            Rr, tr = cl.row_motion(i, u, v)
            x = np.einsum("hwij,hwj->hwi", Rr, x) + tr / s_pose
        Xp.append(x @ P[i, :3, :3].T + P[i, :3, 3])
        Xg.append(g)
        zp.append(lp[i][..., 2])
        D = cl.depth("rs" if rs_gt else "gs")[i]
        zg.append(D[np.clip(np.round(v), 0, cl.H - 1).astype(int), np.clip(np.round(u), 0, cl.W - 1).astype(int)])
        ok.append(valid & np.isfinite(Xp[-1]).all(-1))
    ok = np.stack(ok)
    out = depth_metrics(np.stack(zp), np.stack(zg), ok)
    Xp, Xg = np.stack(Xp)[ok], np.stack(Xg)[ok]
    if len(Xp) < 1000:
        return dict(out, n_pts=int(len(Xp)))
    s, R, t = _align(Xp, Xg, rng)
    Xa = s * Xp @ R.T + t
    out.update(chamfer(Xa, Xg, rng=rng))
    out["corr_err"] = float(np.median(np.linalg.norm(Xa - Xg, axis=1)))
    sub = rng.choice(len(Xg), min(150000, len(Xg)), replace=False)
    pl = planarity(Xa[sub], Xg[sub], rng=rng)
    out["plan_rms"] = float(np.mean([p["rms_dev"] for p in pl])) if pl else np.nan
    out["n_planes"] = len(pl)
    out["n_pts"] = int(len(Xp))
    out["pts_scale"] = float(s)  # metres per model unit
    return out


def _align(Xp, Xg, rng):
    idx = rng.choice(len(Xp), min(50000, len(Xp)), replace=False)
    return umeyama_sim3(Xp[idx], Xg[idx])


def score_clip(args):
    row, preds, models, conds, dense, v5, views = args
    cl = Clip(row["dir"])
    G = cl.T_mid
    out = []
    for m in models:
        for c in conds:
            pk = preds.get((m, c))
            k = f"{cl.key}_{c}"
            if pk is None or k not in pk:
                continue
            P = pk[k]
            g = G[:len(P)]
            rec = dict(model=m, cond=c, **{x: row[x] for x in ("key", "scene", "kind", "omega_mean", "v_mean",
                                                                 "readout_ms", "direction", "rs_shift_px",
                                                                 "depth_median")})
            rec.update(pose_metrics(P, g))
            rec["AUC30"] = pose_auc(P, g)
            rec["n_views"] = len(P)
            # EXP0's static filter, applied uniformly: with < 0.20 m of camera travel, Sim(3) ATE and translation
            # directions are undefined (pure rotation collapses the alignment), so only rotation metrics are kept
            rec["gt_path_m"] = float(np.linalg.norm(np.diff(g[:, :3, 3], axis=0), axis=1).sum())
            if rec["gt_path_m"] < MIN_PATH_M:
                for kk in TRANS_METRICS:
                    rec[kk] = np.nan
            dpath = f"{preds['_dir']}/{m}_{c}_dense/{k}.npz"
            if dense and os.path.exists(dpath):
                dz = np.load(dpath)
                base = dict(rec)
                rec.update(dense_metrics(cl, P, dz, rs_geometry(c)))
                if v5 and c == "rs" and "pts_scale" in rec:
                    r5 = dict(base, cond="v5")
                    for kk in POSE_METRICS:
                        r5[kk] = np.nan  # V5 does not change poses
                    # the true line motion (metres) enters in model units via the RS points' own Sim(3) scale
                    r5.update(dense_metrics(cl, P, dz, True, v5=True, s_pose=rec["pts_scale"]))
                    out.append(r5)
            out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--models", default="pi3,vggt")
    ap.add_argument("--conds", default="rs,gs")
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--v5", action="store_true")
    ap.add_argument("--views", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    clips, n_fail = list_clips(os.path.expanduser(a.set), a.split)
    models, conds = a.models.split(","), a.conds.split(",")
    preds = {"_dir": a.preds}
    for m in models:
        for c in conds:
            p = f"{a.preds}/{m}_{c}.npz"
            if os.path.exists(p):
                preds[(m, c)] = dict(np.load(p))
            else:
                print("missing", p, flush=True)
    jobs = [(r._asdict(), preds, models, conds, a.dense, a.v5, a.views) for r in clips.itertuples(index=False)]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for res in ex.map(score_clip, jobs, chunksize=2):
            rows += res
    df = pd.DataFrame(rows)
    df["bin"] = pd.cut(df.omega_mean, [0, *SPEED_BINS, np.inf], labels=BINS, right=False).astype(str)
    df["n_gen_failed"] = n_fail
    df.to_csv(a.out, index=False)
    cols = [c for c in ("ATE", "AUC30", "chamfer", "corr_err", "absrel", "plan_rms") if c in df]
    print(df.groupby(["model", "cond", "bin"])[cols].median().round(4).to_string(), flush=True)
    print("SYNTH_EVAL_DONE", len(df), flush=True)


if __name__ == "__main__":
    main()
