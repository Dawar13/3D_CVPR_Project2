"""Rolling-shutter ray-casting renderer (Open3D / Embree, CPU).

Every ray carries its own capture time, so a rolling-shutter frame is rendered exactly: each readout line is cast
from the camera pose at that line's time (band = 1 line). Occlusions come from the ray caster, which warping a
single image cannot do. Coarser bands exist only for the convergence test.

Conventions
  camera   OpenCV (x right, y down, z forward); poses are camera-to-world T_wc; pixel centres at integer coords
  timing   line l of L lines is read at  t_frame + (l - (L-1)/2) * readout / L,  so the frame timestamp is the
           centre-line time (EXP0's RS GT convention) and readout = 0 is a global shutter
  exposure every line integrates over [t_line - exposure/2, t_line + exposure/2] (n_exp stratified time samples),
           averaged in linear light
  depth    z-depth in the camera of the pixel's own line, from one ray through the pixel centre (0 = no hit)
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class Camera:
    W: int
    H: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_hfov(cls, W, H, hfov_deg):
        f = W / (2 * np.tan(np.radians(hfov_deg) / 2))
        return cls(W, H, f, f, (W - 1) / 2, (H - 1) / 2)

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def rays_cam(self, u, v):
        """Camera-frame ray directions with z = 1 for pixel coords u, v (any shape) -> (..., 3)."""
        return np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u, dtype=np.float64)], -1)

    def project(self, X):
        z = X[..., 2]
        return self.fx * X[..., 0] / z + self.cx, self.fy * X[..., 1] / z + self.cy


@dataclass
class Shutter:
    readout: float = 0.0  # s; L * line_time, so first to last line is readout * (L-1)/L
    direction: str = "tb"  # tb: top->bottom rows, bt: bottom->top rows, lr: left->right columns (portrait phone)
    exposure: float = 0.0  # s
    n_exp: int = 1  # time samples per exposure (motion blur); 1 = instantaneous
    band: int = 1  # lines per time band; 1 = exact per-line timing

    def n_lines(self, cam):
        return cam.W if self.direction == "lr" else cam.H

    def line_index(self, cam):
        """Readout line index of every pixel row (tb/bt) or column (lr)."""
        L = self.n_lines(cam)
        return np.arange(L)[::-1].copy() if self.direction == "bt" else np.arange(L)

    def line_dt(self, lines, cam):
        """Capture time of readout lines relative to the frame (centre-line) time, with band quantisation."""
        L = self.n_lines(cam)
        lq = (lines // self.band) * self.band + (min(self.band, L) - 1) / 2
        return (lq - (L - 1) / 2) * self.readout / L

    def exp_offsets(self):
        n = max(1, self.n_exp)
        return self.exposure * ((np.arange(n) + 0.5) / n - 0.5)


class Trajectory:
    """Camera-to-world poses sampled densely in time; positions are interpolated linearly, rotations by slerp."""

    def __init__(self, t, R, c):
        self.t = np.asarray(t, np.float64)
        self.rot = Rotation.from_matrix(R)
        self.c = np.asarray(c, np.float64)
        self._slerp = Slerp(self.t, self.rot)

    def poses(self, tq):
        tq = np.clip(np.asarray(tq, np.float64), self.t[0], self.t[-1])
        R = self._slerp(tq).as_matrix()
        c = np.stack([np.interp(tq, self.t, self.c[:, k]) for k in range(3)], -1)
        return R, c

    def T(self, tq):
        R, c = self.poses(np.atleast_1d(tq))
        T = np.tile(np.eye(4), (len(R), 1, 1))
        T[:, :3, :3], T[:, :3, 3] = R, c
        return T

    def twist(self, tq, h=1e-3):
        """Camera-frame angular velocity (rad/s) and linear velocity (m/s) by central differences."""
        tq = np.atleast_1d(np.asarray(tq, np.float64))
        Ra, ca = self.poses(tq - h)
        Rb, cb = self.poses(tq + h)
        R0, _ = self.poses(tq)
        w = Rotation.from_matrix(np.swapaxes(Ra, 1, 2) @ Rb).as_rotvec() / (2 * h)
        v = np.einsum("nji,nj->ni", R0, (cb - ca) / (2 * h))
        return w, v


SRGB_GAMMA = 2.2


def to_linear(c):
    return np.power(np.clip(c, 0, 1), SRGB_GAMMA)


def to_display(c):
    return np.power(np.clip(c, 0, 1), 1 / SRGB_GAMMA)


def _world_rays(R, c, d_cam, by_column):
    """Rotate camera rays (h, w, 3) by per-row (or per-column) rotations R (n, 3, 3); origins c (n, 3)."""
    R, c = R.astype(np.float32), c.astype(np.float32)
    if by_column:
        d_w = np.matmul(d_cam.transpose(1, 0, 2), R.transpose(0, 2, 1)).transpose(1, 0, 2)
        o = np.broadcast_to(c[None], d_w.shape)
    else:
        d_w = np.matmul(d_cam, R.transpose(0, 2, 1))
        o = np.broadcast_to(c[:, None], d_w.shape)
    return o.reshape(-1, 3), d_w.reshape(-1, 3)


def render(scene, traj, cam, shutter, t_frame, ssaa=2, objects=(), background=0.0):
    """Render one frame. Returns dict(rgb uint8 (H,W,3), depth float32 (H,W), line_T (L,4,4), line_t (L,),
    T_mid (4,4)). line_T are the poses at each readout line's exposure centre, in sensor-line order."""
    H, W = cam.H, cam.W
    L = shutter.n_lines(cam)
    line_dt = shutter.line_dt(np.arange(L), cam)  # time per readout line
    pix_line = shutter.line_index(cam)  # readout line of each pixel row (or column for lr)
    by_col = shutter.direction == "lr"
    offs = (np.arange(ssaa) + 0.5) / ssaa - 0.5
    us = (np.arange(W)[:, None] + offs[None]).reshape(-1)
    vs = (np.arange(H)[:, None] + offs[None]).reshape(-1)
    d_cam = cam.rays_cam(*np.meshgrid(us, vs)).astype(np.float32)  # (H*s, W*s, 3)
    n_along = (W if by_col else H) * ssaa
    samp_line = pix_line[np.repeat(np.arange(n_along // ssaa), ssaa)]  # every subsample shares its pixel's line
    acc = np.zeros((d_cam.shape[0] * d_cam.shape[1], 3), np.float32)
    for e in shutter.exp_offsets():
        ts = t_frame + line_dt[samp_line] + e
        o, d = _world_rays(*traj.poses(ts), d_cam, by_col)
        tq = np.broadcast_to(ts[None, :] if by_col else ts[:, None], d_cam.shape[:2]).reshape(-1)
        acc += cast_all(scene, objects, o, d, tq, background)[1]
    lin = (acc / max(1, shutter.n_exp)).reshape(H, ssaa, W, ssaa, 3).mean((1, 3))
    rgb8 = (to_display(lin) * 255 + 0.5).astype(np.uint8)

    # depth: one ray per pixel centre at the line's exposure-centre time
    d_c = cam.rays_cam(*np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64)))
    ts = t_frame + line_dt[pix_line]
    o, d = _world_rays(*traj.poses(ts), d_c.astype(np.float32), by_col)
    tq = np.broadcast_to(ts[None, :] if by_col else ts[:, None], (H, W)).reshape(-1)
    th = cast_all(scene, objects, o, d, tq, background)[0]
    depth = np.where(np.isfinite(th), th, 0).reshape(H, W).astype(np.float32)

    line_t = t_frame + line_dt
    return dict(rgb=rgb8, depth=depth, line_T=traj.T(line_t).astype(np.float32), line_t=line_t,
                T_mid=traj.T(t_frame)[0])


def cast_all(scene, objects, o, d, tq, background):
    """Nearest hit over the static scene and moving objects; colours in linear light. With d_cam.z = 1, the ray
    parameter t of o + t d is the z-depth in the camera that cast the ray."""
    th, rgb = scene.cast(o, d)
    for ob in objects:
        Ro, po = ob.poses(tq)  # object-to-world at each ray's time
        o_l = np.einsum("nji,nj->ni", Ro, o - po)
        d_l = np.einsum("nji,nj->ni", Ro, d)
        th2, rgb2 = ob.scene.cast(o_l, d_l)
        closer = th2 < th
        th = np.where(closer, th2, th)
        rgb = np.where(closer[:, None], rgb2, rgb)
    rgb = np.where(np.isfinite(th)[:, None], rgb, np.float32(background) ** SRGB_GAMMA)
    return th, rgb


def shutter_dict(sh):
    return asdict(sh)
