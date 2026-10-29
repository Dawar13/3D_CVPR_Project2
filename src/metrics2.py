"""EXP2 metrics beyond EXP0's pose_metrics: pose AUC@30, dense point-map accuracy / completeness / Chamfer, depth
AbsRel and delta<1.25, and wall planarity. Conventions follow the 3D foundation-model papers (see LIT_REVIEW.md)."""
import numpy as np
from scipy.spatial import cKDTree

from common import rot_angle_deg, umeyama_sim3


def pair_errors(P, G):
    """All-pairs relative rotation error and translation-direction error (deg). P, G: (N, 4, 4) cam-to-world."""
    inv = np.linalg.inv
    i, j = np.triu_indices(len(P), 1)
    rp, rg = inv(P[i]) @ P[j], inv(G[i]) @ G[j]
    rerr = rot_angle_deg(np.swapaxes(rg[:, :3, :3], 1, 2) @ rp[:, :3, :3])
    tp = rp[:, :3, 3] / (np.linalg.norm(rp[:, :3, 3], axis=1, keepdims=True) + 1e-12)
    tg = rg[:, :3, 3] / (np.linalg.norm(rg[:, :3, 3], axis=1, keepdims=True) + 1e-12)
    terr = np.degrees(np.arccos(np.clip((tp * tg).sum(1), -1, 1)))
    return rerr, terr


def pose_auc(P, G, max_threshold=30):
    """VGGT-style AUC: per pair err = max(rotation err, translation-direction err); mean accuracy over 1..max deg."""
    rerr, terr = pair_errors(P, G)
    err = np.maximum(rerr, terr)
    thr = np.arange(1, max_threshold + 1)
    return float(np.mean([(err < t).mean() for t in thr]))


def align_points_sim3(P, G, max_n=50000, rng=None):
    """Sim(3) taking predicted points P onto pixel-corresponding GT points G (both (N, 3), finite)."""
    rng = rng or np.random.default_rng(0)
    idx = rng.choice(len(P), min(max_n, len(P)), replace=False)
    s, R, t = umeyama_sim3(P[idx], G[idx])
    return s, R, t


def chamfer(P, G, max_n=50000, rng=None):
    """Accuracy (pred -> GT), completeness (GT -> pred) and their mean, as mean nearest-neighbour distances."""
    rng = rng or np.random.default_rng(0)
    P = P[rng.choice(len(P), min(max_n, len(P)), replace=False)]
    G = G[rng.choice(len(G), min(max_n, len(G)), replace=False)]
    acc = cKDTree(G).query(P, workers=4)[0]
    comp = cKDTree(P).query(G, workers=4)[0]
    return dict(acc=float(acc.mean()), comp=float(comp.mean()), chamfer=float((acc.mean() + comp.mean()) / 2),
                acc_med=float(np.median(acc)), comp_med=float(np.median(comp)))


def depth_metrics(pred, gt, valid):
    """Per-frame median scaling, then AbsRel and delta < 1.25 over valid pixels. pred, gt, valid: (N, h, w)."""
    absrel, d1 = [], []
    for p, g, m in zip(pred, gt, valid):
        m = m & (p > 0) & (g > 0)
        if m.sum() < 100:
            continue
        p = p[m] * np.median(g[m]) / np.median(p[m])
        absrel.append(np.mean(np.abs(p - g[m]) / g[m]))
        d1.append(np.mean(np.maximum(p / g[m], g[m] / p) < 1.25))
    return dict(absrel=float(np.mean(absrel)) if absrel else np.nan, delta1=float(np.mean(d1)) if d1 else np.nan)


def fit_planes(G, n_planes=3, thresh=0.01, min_frac=0.05, iters=300, rng=None):
    """Greedy RANSAC planes on GT points G (N, 3). Returns a list of index arrays (inliers per plane)."""
    rng = rng or np.random.default_rng(0)
    remaining = np.arange(len(G))
    planes = []
    for _ in range(n_planes):
        if len(remaining) < 1000:
            break
        Gr = G[remaining]
        best = None
        for _ in range(iters):
            a, b, c = Gr[rng.choice(len(Gr), 3, replace=False)]
            n = np.cross(b - a, c - a)
            if np.linalg.norm(n) < 1e-9:
                continue
            n /= np.linalg.norm(n)
            inl = np.abs((Gr - a) @ n) < thresh
            if best is None or inl.sum() > best.sum():
                best = inl
        if best is None or best.sum() < min_frac * len(G):
            break
        planes.append(remaining[best])
        remaining = remaining[~best]
    return planes


def planarity(P_aligned, G, n_planes=3, rng=None):
    """For each large GT plane, the RMS out-of-plane deviation of the corresponding predicted points (PCA)."""
    out = []
    for idx in fit_planes(G, n_planes, rng=rng):
        X = P_aligned[idx] - P_aligned[idx].mean(0)
        ev = np.linalg.eigvalsh(X.T @ X / len(X))
        out.append(dict(n=int(len(idx)), rms_dev=float(np.sqrt(max(ev[0], 0))),
                        extent=float(np.sqrt(ev[-1]))))
    return out
