"""Convention checks for a new runner on a few TUM windows (run infer2.py --dense on them first).

1. poses are cam-to-world: all-pairs relative-rotation error vs GT and vs pi3 (deg), and Sim(3) ATE, both for the
   poses as saved and for their inverses (if the saved ones were world-to-camera, the inverse would win).
2. local_pts are camera-frame and consistent with the pose scale: median z > 0; frame-0 points moved into frame k
   with the predicted relative pose (translation scaled by s) and projected with frame k's intrinsics (fitted from
   its own local_pts) must land on frame k's depth. We report the median relative depth error at s = 1 and the s
   that minimises it (should be ~1).
3. pixel map from <out>/<model>_<name>_run.json: sx ~ W/w, sy ~ H/h, offsets ~ half-pixel conventions.

usage: python validate.py --model da3 --out ~/rs3d/ext/models/val --name val
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/rs3d/src"))
from common import pose_metrics, rot_angle_deg  # noqa: E402  (read-only use of the project's metric code)

RES = os.path.expanduser("~/rs3d/results/exp2/gap/tum")
OLD = os.path.expanduser("~/rs3d/results/exp0")


def rel_rot_err(P, G):
    i, j = np.triu_indices(len(P), 1)
    inv = np.linalg.inv
    rp = (inv(P[i]) @ P[j])[:, :3, :3]
    rg = (inv(G[i]) @ G[j])[:, :3, :3]
    e = rot_angle_deg(np.swapaxes(rg, 1, 2) @ rp)
    return float(np.mean(e)), float(np.max(e))


def fit_intrinsics(X):
    """X (h, w, 3) camera-frame points -> fx, fy, cx, cy with u = fx X/Z + cx (least squares, z > 0)."""
    h, w, _ = X.shape
    v, u = np.mgrid[:h, :w]
    ok = X[..., 2] > 1e-6
    a = X[..., 0][ok] / X[..., 2][ok]
    b = X[..., 1][ok] / X[..., 2][ok]
    fx, cx = np.polyfit(a, u[ok], 1)
    fy, cy = np.polyfit(b, v[ok], 1)
    return fx, fy, cx, cy


def cross_frame(L, C, P, i, k, scales):
    """median |z_proj - z_k| / z_k for frame-i points moved to frame k, for each translation scale."""
    h, w, _ = L[k].shape
    fx, fy, cx, cy = fit_intrinsics(L[k])
    Xi = L[i].reshape(-1, 3)
    ci = C[i].reshape(-1)
    keep = (Xi[:, 2] > 0) & (ci >= np.quantile(ci, 0.5))
    Xi = Xi[keep][::7]
    Rel = np.linalg.inv(P[k]) @ P[i]  # camera i -> camera k
    out, fracs = [], []
    for s in scales:
        Xk = Xi @ Rel[:3, :3].T + s * Rel[:3, 3]
        z = Xk[:, 2]
        good = z > 1e-6
        u = np.round(fx * Xk[good, 0] / z[good] + cx).astype(int)
        v = np.round(fy * Xk[good, 1] / z[good] + cy).astype(int)
        inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        fracs.append(float(inb.sum()) / max(1, len(Xi)))
        if inb.sum() < 100:
            out.append(np.nan)
            continue
        zk = L[k][v[inb], u[inb], 2]
        out.append(float(np.median(np.abs(z[good][inb] - zk) / np.maximum(zk, 1e-6))))
    return np.array(out), fracs[int(np.argmin(np.abs(np.asarray(scales) - 1)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="val")
    a = ap.parse_args()
    tag = f"{a.model}_{a.name}"
    pr = np.load(f"{a.out}/{tag}.npz")
    run = json.load(open(f"{a.out}/{tag}_run.json"))
    gt = np.load(f"{RES}/gt_poses.npz")
    pi3 = [np.load(p) for p in (f"{RES}/preds/pi3.npz", f"{OLD}/preds/pi3.npz") if os.path.exists(p)]

    pm = run["pixel_map"]
    print(f"== {tag}: ckpt {run['ckpt']}, model input {pm['w']}x{pm['h']}, peak alloc {run['peak_alloc_gb']:.2f} GB, "
          f"{run['infer_s'] / max(1, run['n_clips_this_run']):.3f} s/clip")
    print(f"pixel map: sx={pm['sx']:.4f} (W/w={pm['W'] / pm['w']:.4f}) ox={pm['ox']:.3f} | "
          f"sy={pm['sy']:.4f} (H/h={pm['H'] / pm['h']:.4f}) oy={pm['oy']:.3f}")
    scales = np.geomspace(0.25, 4, 49)
    for k in sorted(pr.keys()):
        P = pr[k]
        cond = k.rsplit("_", 1)[1]
        G = gt[k if cond == "GS" else k.rsplit("_", 1)[0] + "_RS"]
        Pinv = np.linalg.inv(P)
        e_gt, m_gt = rel_rot_err(P, G)
        e_gt_inv, _ = rel_rot_err(Pinv, G)
        ate = pose_metrics(P, G)["ATE"]
        ate_inv = pose_metrics(Pinv, G)["ATE"]
        line = (f"{k}: relrot vs GT mean {e_gt:.2f} max {m_gt:.2f} deg (inverted poses: {e_gt_inv:.2f}) | "
                f"ATE {ate * 100:.2f} cm (inverted: {ate_inv * 100:.2f} cm)")
        ref = next((p[k] for p in pi3 if k in p), None)
        if ref is not None:
            e_pi, m_pi = rel_rot_err(P, ref)
            e_pigt, _ = rel_rot_err(ref, G)
            line += f" | vs pi3 mean {e_pi:.2f} max {m_pi:.2f} deg (pi3 vs GT {e_pigt:.2f}, ATE {pose_metrics(ref, G)['ATE'] * 100:.2f} cm)"
        print(line)
        dp = f"{a.out}/{tag}_dense/{k}.npz"
        if os.path.exists(dp):
            d = np.load(dp)
            L = d["local_pts"].astype(np.float32)
            C = d["conf"].astype(np.float32)
            z = L[..., 2]
            fx, fy, cx, cy = fit_intrinsics(L[0])
            n = len(L)
            res = []
            for kk in (n // 3, 2 * n // 3, n - 1):
                errs, frac = cross_frame(L, C, P, 0, kk, scales)
                if np.all(np.isnan(errs)):
                    res.append(f"0->{kk}: no overlap")
                    continue
                j = int(np.nanargmin(errs))
                e = {t: errs[np.argmin(np.abs(scales - t))] * 100 for t in (0.5, 1, 2)}
                res.append(f"0->{kk}: err@s=1 {e[1]:.1f}% (s=0.5 {e[0.5]:.1f}%, s=2 {e[2]:.1f}%), best s "
                           f"{scales[j]:.2f} ({errs[j] * 100:.1f}%), in-view {frac:.2f}")
            print(f"   dense: shape {L.shape}, median z {np.median(z):.3f}, z>0 {np.mean(z > 0):.3f}, "
                  f"fitted K f=({fx:.1f},{fy:.1f}) c=({cx:.1f},{cy:.1f}); " + "; ".join(res))


if __name__ == "__main__":
    main()
