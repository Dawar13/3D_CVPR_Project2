"""Post-capture stress variants (EXP2 §9 items 7-10, image side) of finished clips, applied identically to the RS
frames and their GS twins so every comparison stays paired. Written as new condition folders in each clip:

  {rs,gs}_noise     Poisson-Gaussian sensor noise: 400 photo-electrons at white, read noise 2 e-
  {rs,gs}_lowlight  the same sensor at 1/10 of the light (40 e- at white), then 10x digital gain
  {rs,gs}_gamma     tone curve exponent 1/1.6 instead of 1/2.2 (flatter, brighter)
  {rs,gs}_crop      centre crop to rows 64..447 (640x384, 5:3), no resize; principal point shifts by 64 rows
  {rs,gs}_cropzoom  the same crop, then resized to 480x288 (0.75x)
  v3_crop           crop of the full-frame V3 output: straightening that follows the crop (row timing kept)
  v3_cropnaive      V3 applied to rs_crop assuming the cropped rows span the whole readout (a method that
                    ignores the crop); readout 30 ms over 384 rows instead of 384/512 of it
  {rs,gs}_eis       simulated electronic stabilisation: each frame is rotated (homography K R_v^T R_a K^-1) to a
                    virtual camera whose orientation is the true orientation smoothed over +-0.3 s, then zoomed
                    1.1x to hide borders. gt_T.npy holds the virtual cameras' poses (smoothed rotation, true
                    centre); rows no longer map to readout lines
Noise is seeded per clip and condition, so RS and GS get the same noise field statistics, not the same sample.

usage: python -m rsgen.post --set ~/rs3d/data/synth/core_test --max_clip 14
"""
import argparse
import os
import sys
import zlib

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .core import SRGB_GAMMA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_unroll as G  # noqa: E402
from dense_utils import Clip, list_clips  # noqa: E402

CROP = (64, 448)  # rows kept
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def to_lin(im8):
    return np.power(im8.astype(np.float32) / 255, SRGB_GAMMA)


def to_u8(lin, gamma=SRGB_GAMMA):
    return (np.power(np.clip(lin, 0, 1), 1 / gamma) * 255 + 0.5).astype(np.uint8)


def sensor(lin, rng, full_well, read=2.0, gain=1.0):
    e = rng.poisson(np.clip(lin, 0, 1) * full_well).astype(np.float32) + rng.normal(0, read, lin.shape)
    return e / full_well * gain


def eis_homographies(cl, smooth_s=0.3):
    """Per-frame rotation-only warps to a virtual camera with smoothed orientation; returns H (n,3,3), gt_T."""
    Z = np.diag([1.1, 1.1, 1.0])
    Z[0, 2], Z[1, 2] = -0.1 * (cl.W - 1) / 2, -0.1 * (cl.H - 1) / 2  # zoom about the image centre
    ts = cl.traj.t
    dt = ts[1] - ts[0]
    Hs, Ts = [], []
    for i, t in enumerate(cl.t_frames):
        m = np.abs(ts - t) <= smooth_s
        w = np.exp(-0.5 * ((ts[m] - t) / (smooth_s / 2)) ** 2)
        Rv = Rotation.from_matrix(cl.traj.poses(ts[m])[0]).mean(weights=w).as_matrix()
        Ra = cl.T_mid[i, :3, :3]
        Hs.append(Z @ cl.K @ Rv.T @ Ra @ np.linalg.inv(cl.K))
        T = cl.T_mid[i].copy()
        T[:3, :3] = Rv
        Ts.append(T)
    assert dt > 0
    return np.stack(Hs), np.stack(Ts)


def naive_crop_unroll(cl, imgs, w):
    """V3 on cropped frames, assuming the cropped rows span the full readout (ignores the crop)."""
    h = CROP[1] - CROP[0]
    K = cl.K.copy()
    K[1, 2] -= CROP[0]
    scan = G.ScanModel(cl.shutter.readout / h, h, cl.W, cl.shutter.direction)
    un = G.Unroller(K, (h, cl.W), lambda X: G.pinhole_project(X, K), scan, device=DEV)
    mu, mv = un.maps(np.asarray(w))
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(DEV).float()
    out = G.Unroller.remap(x, mu.float(), mv.float())
    return (out + 0.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def read(p):
    return cv2.imread(p, cv2.IMREAD_COLOR)[..., ::-1]


def write(d, k, im):
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(f"{d}/{k:02d}.png", np.ascontiguousarray(im[..., ::-1]), [cv2.IMWRITE_PNG_COMPRESSION, 1])


def process(r):
    cl = Clip(r.dir)
    n = len(cl.t_frames)
    if os.path.isdir(f"{r.dir}/gs_eis") and len(os.listdir(f"{r.dir}/gs_eis")) == n + 1:
        return
    Hs, Tv = eis_homographies(cl)
    for src in ("rs", "gs"):
        ims = [read(f"{r.dir}/{src}/{k:02d}.png") for k in range(n)]
        rng = np.random.default_rng([r.seed, zlib.crc32(src.encode())])
        for k, im in enumerate(ims):
            lin = to_lin(im)
            write(f"{r.dir}/{src}_noise", k, to_u8(sensor(lin, rng, 400)))
            write(f"{r.dir}/{src}_lowlight", k, to_u8(sensor(lin * 0.1, rng, 400, gain=10.0)))
            write(f"{r.dir}/{src}_gamma", k, to_u8(lin, gamma=1.6))
            c = im[CROP[0]:CROP[1]]
            write(f"{r.dir}/{src}_crop", k, c)
            write(f"{r.dir}/{src}_cropzoom", k, cv2.resize(c, (480, 288), interpolation=cv2.INTER_AREA))
            write(f"{r.dir}/{src}_eis", k, cv2.warpPerspective(im, Hs[k], (cl.W, cl.H), flags=cv2.INTER_LINEAR))
        np.save(f"{r.dir}/{src}_eis/gt_T.npy", Tv)
    if os.path.isdir(f"{r.dir}/v3"):
        for k in range(n):
            write(f"{r.dir}/v3_crop", k, read(f"{r.dir}/v3/{k:02d}.png")[CROP[0]:CROP[1]])
        crop = np.stack([read(f"{r.dir}/rs_crop/{k:02d}.png") for k in range(n)])
        for k, im in enumerate(naive_crop_unroll(cl, crop, cl.w)):
            write(f"{r.dir}/v3_cropnaive", k, im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_clip", type=int, default=14)
    a = ap.parse_args()
    clips, _ = list_clips(os.path.expanduser(a.set), a.split)
    clips = clips[clips.clip <= a.max_clip]
    for r in clips.itertuples():
        process(r)
    print("POST_DONE", len(clips), flush=True)


if __name__ == "__main__":
    main()
