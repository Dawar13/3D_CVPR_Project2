"""Generator validation (EXP2 §6). Every check must pass before rsgen data is used.

  1 readout = 0 gives exactly the global-shutter render, for every band size
  2 known 3D points re-cast through their predicted RS pixel (fixed point on the line time) hit the same point
  3 unrolling the RS render with the true motion matches the GS twin
      a exact: per-line true poses and true GS depth (tests the renderer)
      b gpu_unroll with the true (w, v) at the frame time and the true RS depth: the V4 oracle
      c gpu_unroll, rotation only: the V3 oracle
  4 convergence: halving the band size (doubling the band count) changes pixels by < 0.5 px
  5 visual: a fast pan, GS | RS | exact unroll | |GS - unrolled|
  6 motion generator: measured mean speeds match their targets
Directions tb, bt and lr are all exercised by checks 1-3.

usage: python -m rsgen.validate --scene testroom|<mesh.ply> [--up 2] [--out results/exp2/generator]
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

from .core import Camera, Shutter, Trajectory, render
from .motion import look_rotation, sample_clip_trajectory
from .scenes import Scene, sample_start, test_room

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import gpu_unroll as G  # noqa: E402


def line_dt_cont(u, v, cam, sh):
    """Continuous line time (no row quantisation), for the fixed-point start."""
    L = sh.n_lines(cam)
    l = {"tb": v, "bt": cam.H - 1 - v, "lr": u}[sh.direction]
    l = np.clip(np.nan_to_num(l, nan=(L - 1) / 2), 0, L - 1)  # off-image points (masked later) stay in range
    return (l - (L - 1) / 2) * sh.readout / L


def line_dt_disc(u, v, cam, sh):
    """The renderer's line time for the pixel that contains (u, v)."""
    pix_line = sh.line_index(cam)
    L = sh.n_lines(cam)
    if sh.direction == "lr":
        idx = np.clip(np.round(u), 0, cam.W - 1).astype(int)
    else:
        idx = np.clip(np.round(v), 0, cam.H - 1).astype(int)
    return sh.line_dt(pix_line[idx], cam) if L else 0.0


def project_at(X, traj, t, cam):
    R, c = traj.poses(t)
    Xc = np.einsum("nji,nj->ni", R, X - c)
    u, v = cam.project(Xc)
    return u, v, Xc[:, 2]


def rs_pixel(X, traj, cam, sh, t_frame, n_cont=15, n_disc=4):
    """Where world points X (N, 3) appear in the RS frame: fixed point of pixel -> line time -> projection."""
    u, v, _ = project_at(X, traj, np.full(len(X), t_frame), cam)
    for _ in range(n_cont):
        u, v, _ = project_at(X, traj, t_frame + line_dt_cont(u, v, cam, sh), cam)
    for _ in range(n_disc):
        u, v, z = project_at(X, traj, t_frame + line_dt_disc(u, v, cam, sh), cam)
    return u, v, z


def backproject(depth, traj, cam, t_line):
    """World points of a depth map; t_line: (H, W) capture time of each pixel."""
    H, W = depth.shape
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    R, c = traj.poses(t_line.reshape(-1))
    d = cam.rays_cam(uu, vv).reshape(-1, 3) * depth.reshape(-1, 1)
    return (np.einsum("nij,nj->ni", R, d) + c).reshape(H, W, 3)


def pixel_times(cam, sh, t_frame):
    lt = sh.line_dt(sh.line_index(cam), cam)
    return t_frame + (np.broadcast_to(lt[None, :], (cam.H, cam.W)) if sh.direction == "lr"
                      else np.broadcast_to(lt[:, None], (cam.H, cam.W)))


def psnr(a, b, m):
    e = ((a.astype(np.float64) - b.astype(np.float64)) ** 2)[m].mean()
    return 10 * np.log10(255 ** 2 / max(e, 1e-12))


def unroll_exact(rs, gs_depth, traj, cam, sh, t_frame):
    """GS-twin view rebuilt from the RS render with the true per-line poses and true GS depth."""
    X = backproject(gs_depth, traj, cam, np.full(gs_depth.shape, t_frame)).reshape(-1, 3)
    u, v, z = rs_pixel(X, traj, cam, sh, t_frame)
    u, v, z = (a.reshape(gs_depth.shape) for a in (u, v, z))
    out = cv2.remap(rs, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out, u, v, z


def visible_mask(u, v, z, rs_depth, gs_depth, cam, margin=2):
    inside = (u >= margin) & (u <= cam.W - 1 - margin) & (v >= margin) & (v <= cam.H - 1 - margin)
    ui = np.clip(np.round(u), 0, cam.W - 1).astype(int)
    vi = np.clip(np.round(v), 0, cam.H - 1).astype(int)
    dz = np.abs(rs_depth[vi, ui] - z)
    edge = cv2.dilate((cv2.Laplacian(gs_depth, cv2.CV_32F) > 0.05).astype(np.uint8), np.ones((5, 5)))
    return inside & (gs_depth > 0) & (dz < 0.01 * z + 0.01) & (edge == 0)


def unroll_gpu(rs, rs_depth, traj, cam, sh, t_frame, rotation_only):
    w, v = traj.twist(np.array([t_frame]))
    scan = G.ScanModel(sh.readout / sh.n_lines(cam), cam.H, cam.W, sh.direction)
    K = cam.K
    un = G.Unroller(K, (cam.H, cam.W), lambda X: G.pinhole_project(X, K), scan, device="cpu")
    if rotation_only:
        mu, mv = un.maps(w)
    else:
        mu, mv = un.maps(w, v, rs_depth[None])
    img = torch.from_numpy(rs.astype(np.float32)).permute(2, 0, 1)[None]
    out = G.Unroller.remap(img, mu, mv)[0].permute(1, 2, 0).numpy()
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def constant_pan(scene, rng, omega_dps, speed, up=2):
    """Constant-rate yaw pan while walking straight: the visual check's motion."""
    p0 = sample_start(scene, rng, up=up, clearance=1.0)
    up_vec = np.eye(3)[up]
    h = np.zeros(3)
    h[[k for k in range(3) if k != up][0]] = 1
    t = np.linspace(-0.2, 0.2, 801)
    R0 = look_rotation(h, up_vec, pitch=np.radians(-10))
    from scipy.spatial.transform import Rotation
    R = R0[None] @ Rotation.from_rotvec(np.outer(t, [0, np.radians(omega_dps), 0])).as_matrix()
    c = p0[None] + np.outer(t, speed * (R0 @ np.array([1.0, 0, 0])))  # sideways walk (camera x)
    return Trajectory(t, R, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="testroom")
    ap.add_argument("--up", type=int, default=2)
    ap.add_argument("--out", default="results/exp2/generator")
    ap.add_argument("--W", type=int, default=640)
    ap.add_argument("--H", type=int, default=480)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    name = "testroom" if a.scene == "testroom" else os.path.basename(os.path.dirname(a.scene))
    rng = np.random.default_rng(a.seed)
    scene = test_room() if a.scene == "testroom" else Scene.from_ply(a.scene)
    cam = Camera.from_hfov(a.W, a.H, 70)
    R = dict(scene=a.scene, camera=dict(W=a.W, H=a.H, fx=cam.fx), checks={})
    t_frame = 1.0
    T0 = time.time()

    # a demanding but realistic clip: handheld, ~150 deg/s turning, 1.5 m/s walking
    traj, info = sample_clip_trajectory(scene, "handheld", np.radians(150), 1.5, 0.5, 1.5, rng, up=a.up)
    wv = traj.twist(np.array([t_frame]))
    R["clip"] = dict(info, omega_at_frame_dps=float(np.degrees(np.linalg.norm(wv[0]))),
                     speed_at_frame=float(np.linalg.norm(wv[1])))

    # 1. readout 0 == GS, for several bands
    gs = render(scene, traj, cam, Shutter(0.0), t_frame)
    same = all(np.array_equal(render(scene, traj, cam, Shutter(0.0, d, band=b), t_frame)["rgb"], gs["rgb"])
               for d in ("tb", "bt", "lr") for b in (1, 8))
    R["checks"]["1_readout0_equals_GS"] = dict(identical=bool(same), passed=bool(same))

    res2, res3 = {}, {}
    for d in ("tb", "bt", "lr"):
        sh = Shutter(0.030, d)
        rs = render(scene, traj, cam, sh, t_frame)
        # 2. known points: GS-depth points, predicted RS pixel, re-cast at that pixel's line time
        m = gs["depth"] > 0
        idx = rng.choice(np.flatnonzero(m), 3000, replace=False)
        X = backproject(gs["depth"], traj, cam, np.full(m.shape, t_frame)).reshape(-1, 3)[idx]
        u, v, z = rs_pixel(X, traj, cam, sh, t_frame)
        ok = (u > 1) & (u < cam.W - 2) & (v > 1) & (v < cam.H - 2)
        u, v, z, X = u[ok], v[ok], z[ok], X[ok]
        t_l = t_frame + line_dt_disc(u, v, cam, sh)
        Rl, cl = traj.poses(t_l)
        dw = np.einsum("nij,nj->ni", Rl, cam.rays_cam(u, v))
        th, _ = scene.cast(cl, dw)
        hitX = cl + th[:, None] * dw
        uh, vh, _ = project_at(hitX, traj, t_l, cam)
        err_px = np.hypot(uh - u, vh - v)
        unocc = np.abs(th - z) < 0.01 * z + 0.005
        shift = np.hypot(*(np.array(project_at(X, traj, np.full(len(X), t_frame), cam)[:2]) - np.array([u, v])))
        res2[d] = dict(n=int(unocc.sum()), occluded_frac=float(1 - unocc.mean()),
                       median_px=float(np.median(err_px[unocc])), p99_px=float(np.percentile(err_px[unocc], 99)),
                       rs_shift_median_px=float(np.median(shift)), rs_shift_max_px=float(shift.max()),
                       passed=bool(np.percentile(err_px[unocc], 99) < 0.1))
        # 3. unrolling vs the GS twin
        ex, uu, vv, zz = unroll_exact(rs["rgb"], gs["depth"], traj, cam, sh, t_frame)
        vis = visible_mask(uu, vv, zz, rs["depth"], gs["depth"], cam)
        v4 = unroll_gpu(rs["rgb"], rs["depth"], traj, cam, sh, t_frame, rotation_only=False)
        v3 = unroll_gpu(rs["rgb"], rs["depth"], traj, cam, sh, t_frame, rotation_only=True)
        res3[d] = dict(visible_frac=float(vis.mean()),
                       psnr_raw_rs_vs_gs=float(psnr(rs["rgb"], gs["rgb"], vis)),
                       psnr_exact_unroll_vs_gs=float(psnr(ex, gs["rgb"], vis)),
                       psnr_V4_gpu_unroll_vs_gs=float(psnr(v4, gs["rgb"], vis)),
                       psnr_V3_rot_only_vs_gs=float(psnr(v3, gs["rgb"], vis)),
                       mae_exact_unroll=float(np.abs(ex.astype(float) - gs["rgb"])[vis].mean()))
        res3[d]["passed"] = bool(res3[d]["psnr_exact_unroll_vs_gs"] > res3[d]["psnr_raw_rs_vs_gs"] + 6
                                 and res3[d]["psnr_exact_unroll_vs_gs"] > 30)
        if d == "tb":
            keep = (rs, ex)
    R["checks"]["2_known_point_reprojection"] = res2
    R["checks"]["3_unroll_matches_GS_twin"] = res3

    # 4. convergence over band size (tb, 30 ms)
    conv = {}
    prev = None
    for b in (32, 16, 8, 4, 2, 1):
        sh = Shutter(0.030, "tb", band=b)
        r = render(scene, traj, cam, sh, t_frame)
        Xw = backproject(r["depth"], traj, cam, pixel_times(cam, sh, t_frame))
        if prev is not None:
            # where the previous (coarser) render's surface point at each pixel lands in this render's cameras
            tl = pixel_times(cam, sh, t_frame).reshape(-1)
            up_, vp_, _ = project_at(prev.reshape(-1, 3), traj, tl, cam)
            uu, vv = np.meshgrid(np.arange(cam.W), np.arange(cam.H))
            dpx = np.hypot(up_ - uu.reshape(-1), vp_ - vv.reshape(-1)).reshape(cam.H, cam.W)
            ok = (r["depth"] > 0) & (np.linalg.norm(prev - Xw, axis=-1) < 0.05)
            conv[f"band {2 * b}->{b} rows"] = dict(median_px=float(np.median(dpx[ok])),
                                                   p99_px=float(np.percentile(dpx[ok], 99)),
                                                   max_px=float(dpx[ok].max()))
        prev = Xw
    last = conv["band 2->1 rows"]
    R["checks"]["4_band_convergence"] = dict(conv, passed=bool(last["p99_px"] < 0.5))

    # 5. visual: constant 200 deg/s pan while walking 1 m/s sideways
    pan = constant_pan(scene, rng, 200, 1.0, up=a.up)
    sh = Shutter(0.030, "tb")
    g5, r5 = render(scene, pan, cam, Shutter(0.0), 0.0), render(scene, pan, cam, sh, 0.0)
    ex5, *_ = unroll_exact(r5["rgb"], g5["depth"], pan, cam, sh, 0.0)
    diff = np.clip(np.abs(ex5.astype(int) - g5["rgb"]).sum(-1) * 2, 0, 255).astype(np.uint8)
    sep = np.full((cam.H, 8, 3), 255, np.uint8)
    strip = np.concatenate([g5["rgb"], sep, r5["rgb"], sep, ex5, sep, np.repeat(diff[..., None], 3, -1)], 1)
    cv2.imwrite(f"{a.out}/validation_fast_pan_{name}.jpg", cv2.resize(strip, None, fx=0.5, fy=0.5)[..., ::-1],
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    R["checks"]["5_visual_fast_pan"] = dict(file=f"{a.out}/validation_fast_pan_{name}.jpg",
                                            note="GS | RS (30 ms, tb) | exact unroll | |GS - unrolled| x2")

    # 6. motion generator hits its speed targets
    rows = []
    for kind in ("handheld", "drone", "rotation", "translation", "mixed"):
        for om, spd in ((30, 0.5), (120, 1.0), (200, 2.0)):
            om_eff = 0 if kind == "translation" else om
            spd_eff = 0 if kind == "rotation" else spd
            tr, _ = sample_clip_trajectory(scene, kind, np.radians(om_eff), spd_eff, -0.1, 2.4, rng, up=a.up)
            ww, vv_ = tr.twist(np.linspace(-0.1, 2.4, 200))
            rows.append(dict(kind=kind, omega_target_dps=om_eff, omega_meas_dps=float(np.degrees(
                np.linalg.norm(ww, axis=1).mean())), speed_target=spd_eff,
                speed_meas=float(np.linalg.norm(vv_, axis=1).mean())))
    errs = [abs(r_["omega_meas_dps"] - r_["omega_target_dps"]) / max(r_["omega_target_dps"], 1) for r_ in rows] + \
           [abs(r_["speed_meas"] - r_["speed_target"]) / max(r_["speed_target"], 0.1) for r_ in rows]
    R["checks"]["6_motion_targets"] = dict(rows=rows, max_rel_err=float(max(errs)), passed=bool(max(errs) < 0.05))

    R["all_passed"] = bool(all(v.get("passed", True) if "passed" in v else all(x["passed"] for x in v.values())
                               for k, v in R["checks"].items() if k != "5_visual_fast_pan"))
    R["wall_s"] = time.time() - T0
    with open(f"{a.out}/validation_{name}.json", "w") as f:
        json.dump(R, f, indent=1)
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
