"""Helpers shared by the EXP2 synthetic evaluation and the self-estimate baselines: model-grid <-> image-grid
resampling (via infer2's measured pixel map), synthetic clip loading, and per-row true poses."""
import json
import os
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from rsgen.core import Shutter, Trajectory


def pmap_of(d):
    """Pixel map from a dense npz (keys map_sx, ...) as a dict."""
    return {k[4:]: float(d[k]) for k in d.files if k.startswith("map_")}


def model_pixels_orig(pm):
    """Original-image coordinates (u, v) of every model pixel centre, each (h, w)."""
    vm, um = np.mgrid[0:int(pm["h"]), 0:int(pm["w"])].astype(np.float64)
    return pm["sx"] * um + pm["ox"], pm["sy"] * vm + pm["oy"]


def to_orig_grid(x, pm, interp=cv2.INTER_LINEAR):
    """Resample model-grid maps x (N, h, w) to the original image grid (N, H, W); outside the model's crop the
    nearest edge value is used."""
    W, H = int(pm["W"]), int(pm["H"])
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    mx = (u - pm["ox"]) / pm["sx"]
    my = (v - pm["oy"]) / pm["sy"]
    return np.stack([cv2.remap(np.ascontiguousarray(a, np.float32), mx.astype(np.float32), my.astype(np.float32),
                               interp, borderMode=cv2.BORDER_REPLICATE) for a in x])


class Clip:
    """One synthetic clip folder written by rsgen.generate."""

    def __init__(self, d):
        self.dir = d
        with open(f"{d}/meta.json") as f:
            self.meta = json.load(f)
        z = np.load(f"{d}/traj.npz")
        self.traj = Trajectory(z["t"], Rotation.from_quat(z["quat"]).as_matrix(), z["c"])
        self.T_mid, self.w, self.v, self.t_frames, self.K = z["T_mid"], z["w"], z["v"], z["t_frames"], z["K"]
        s = self.meta["shutter"]
        self.shutter = Shutter(s["readout"], s["direction"], s["exposure"], s["n_exp"], s.get("band", 1))
        c = self.meta["camera"]
        self.W, self.H = c["W"], c["H"]
        self.key = self.meta["index_row"]["key"]
        self._depth = None

    def depth(self, which):
        if self._depth is None:
            self._depth = dict(np.load(f"{self.dir}/depth.npz"))
        return self._depth[which].astype(np.float32)

    def line_dt_at(self, u, v):
        """Capture time (s, relative to the frame time) of original pixels (u, v), from their readout line."""
        sh, cam = self.shutter, SimpleNamespace(W=self.W, H=self.H)
        if sh.direction == "lr":
            idx = np.clip(np.round(u), 0, self.W - 1).astype(int)
        else:
            idx = np.clip(np.round(v), 0, self.H - 1).astype(int)
        return sh.line_dt(sh.line_index(cam)[idx], cam)

    def gt_points(self, i, u, v, rs):
        """True world points seen at original pixels (u, v) of frame i: depth sampled at the nearest pixel, ray at
        (u, v), posed with the pixel's own line pose (rs=True, RS frame) or the frame pose (GS frame / twin).
        Returns X (..., 3) and valid (...)."""
        D = self.depth("rs" if rs else "gs")[i]
        ui = np.clip(np.round(u), 0, self.W - 1).astype(int)
        vi = np.clip(np.round(v), 0, self.H - 1).astype(int)
        Z = D[vi, ui]
        edge = cv2.dilate((np.abs(cv2.Laplacian(D, cv2.CV_32F)) > 0.05 * np.maximum(D, 1e-3)).astype(np.uint8),
                          np.ones((3, 3), np.uint8))
        valid = (Z > 0) & (edge[vi, ui] == 0)
        Kinv = np.linalg.inv(self.K)
        ray = np.stack([u, v, np.ones_like(u)], -1) @ Kinv.T
        Xc = ray * Z[..., None]
        if rs and self.shutter.readout > 0:
            R, c = self.traj.poses(self.t_frames[i] + self.line_dt_at(u, v).reshape(-1))
            X = np.einsum("nij,nj->ni", R, Xc.reshape(-1, 3)) + c
            X = X.reshape(Xc.shape)
        else:
            T = self.T_mid[i]
            X = Xc @ T[:3, :3].T + T[:3, 3]
        return X, valid

    def row_motion(self, i, u, v):
        """True motion from the frame pose to each original pixel's line pose: (R, t) with T_mid^-1 T_line,
        each (..., 3, 3) and (..., 3) in metres."""
        R, c = self.traj.poses(self.t_frames[i] + self.line_dt_at(u, v).reshape(-1))
        T = self.T_mid[i]
        Rr = np.einsum("ji,njk->nik", T[:3, :3], R)
        tr = (c - T[:3, 3]) @ T[:3, :3]
        return Rr.reshape(u.shape + (3, 3)), tr.reshape(u.shape + (3,))


def list_clips(set_dir, split="test"):
    """Index rows of a generated set (failed clips dropped and counted)."""
    import pandas as pd
    df = pd.read_csv(os.path.join(set_dir, f"{split}.csv"))
    n_fail = int(df["failed"].notna().sum()) if "failed" in df else 0
    if n_fail:
        df = df[df["failed"].isna()]
    df = df.copy()
    df["dir"] = [os.path.join(set_dir, split, r.scene, f"{int(r.clip):05d}") for r in df.itertuples()]
    return df, n_fail
