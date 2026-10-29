"""EXP2 rolling-shutter unrolling in torch (GPU), batched over frames. A port of unroll.py (EXP1), generalised.

Every output pixel is re-rendered at the reference time, which is the centre readout line, as in EXP1:
  R_wc(t) = R_wc(t_ref) exp([w_c]x dt),   c(t) = c_ref + v_w dt,   dt = t_line - t_ref.
The reference-camera ray d (z = 1) at depth Z was seen by line-camera r as X_r = exp(-[w_c]x dt) (Z d - v_c dt).
The source line depends on the answer, so it is found by fixed-point iteration (N_ITER steps, as in EXP1).

New in EXP2:
- The projection is pluggable: Kalibr equidistant fisheye (TUM raw frames) or plain pinhole (synthetic frames).
- ScanModel adds a scan direction ('tb' top->bottom, 'bt' bottom->top, 'lr' left->right) and any line time.
  That covers the readout-error and wrong-direction runs of the oracle sensitivity test.
- torch.grid_sample replaces cv2.remap. align_corners=True keeps cv2's pixel-centre convention, and
  padding_mode='border' is cv2.BORDER_REPLICATE, as in the EXP1 run.
"""
import torch
import torch.nn.functional as F

N_ITER = 4
GAMMA = 1 / 2.2  # prep.py's tone curve for TUM's linear 16-bit frames


def fisheye_project(X, K, D):
    """Kalibr pinhole-equidistant (= OpenCV fisheye) projection of camera-frame points (..., 3) to raw pixels."""
    z = X[..., 2].clamp_min(1e-6)
    a, b = X[..., 0] / z, X[..., 1] / z
    r = torch.sqrt(a * a + b * b)
    th = torch.atan(r)
    t2 = th * th
    thd = th * (1 + D[0] * t2 + D[1] * t2 ** 2 + D[2] * t2 ** 3 + D[3] * t2 ** 4)
    sc = torch.where(r > 1e-9, thd / r.clamp_min(1e-9), torch.ones_like(r))
    return K[0][0] * a * sc + K[0][2], K[1][1] * b * sc + K[1][2]


def pinhole_project(X, K):
    z = X[..., 2].clamp_min(1e-6)
    return K[0][0] * X[..., 0] / z + K[0][2], K[1][1] * X[..., 1] / z + K[1][2]


def rotate(X, axis, ang):
    """Rodrigues rotation of X (B, H, W, 3) about per-frame unit axes (B, 3) by per-pixel angles (B, H, W)."""
    c, s = torch.cos(ang)[..., None], torch.sin(ang)[..., None]
    k = axis[:, None, None, :].expand_as(X)
    kdX = (X * k).sum(-1, keepdim=True)
    return X * c + torch.cross(k, X, dim=-1) * s + k * kdX * (1 - c)


class ScanModel:
    """Readout time of a raw sensor pixel relative to the reference (centre) line.

    For TUM, tb with line_time = ROW_TIME and n_rows = 1024 gives dt = v * ROW_TIME - RS_MID_OFFSET, as in EXP1.
    """

    def __init__(self, line_time, n_rows, n_cols, direction="tb"):
        assert direction in ("tb", "bt", "lr", "rl")
        self.line_time, self.n_rows, self.n_cols, self.direction = line_time, n_rows, n_cols, direction

    def dt(self, u, v):
        if self.direction == "tb":
            return (v - (self.n_rows - 1) / 2) * self.line_time
        if self.direction == "bt":
            return ((self.n_rows - 1) / 2 - v) * self.line_time
        if self.direction == "rl":
            return ((self.n_cols - 1) / 2 - u) * self.line_time
        return (u - (self.n_cols - 1) / 2) * self.line_time


def _sample(img, u, v, mode="bilinear"):
    """Sample img (B, C, h, w) at pixel coordinates u, v (B, H, W), cv2.remap style. Returns (B, C, H, W)."""
    h, w = img.shape[-2:]
    grid = torch.stack([2 * u / (w - 1) - 1, 2 * v / (h - 1) - 1], -1).to(img.dtype)
    return F.grid_sample(img, grid, mode=mode, padding_mode="border", align_corners=True)


class Unroller:
    """Maps from the straightened output image (pinhole K_out, size out_hw) to raw sensor pixels.

    project: callable mapping camera-frame points (..., 3) to raw pixel coordinates (u, v).
    """

    def __init__(self, K_out, out_hw, project, scan, device="cuda", dtype=torch.float64):
        H, W = out_hw
        self.K = torch.as_tensor(K_out, dtype=dtype, device=device)
        v, u = torch.meshgrid(torch.arange(H, dtype=dtype, device=device),
                              torch.arange(W, dtype=dtype, device=device), indexing="ij")
        self.rays = torch.stack([(u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1],
                                 torch.ones_like(u)], -1)  # z = 1, so Z * ray is the point at z-depth Z
        self.project, self.scan = project, scan
        self.u0, self.v0 = project(self.rays)  # zero motion: plain undistortion map
        self.device, self.dtype = device, dtype

    def maps(self, w_c, v_c=None, depth=None):
        """w_c, v_c: (B, 3) camera-frame angular (rad/s) and linear (m/s) velocity.

        depth is None (rotation only) or (B, h, w) metric z-depth on the output pinhole grid, at any resolution.
        Low-res depth is upsampled like cv2.resize INTER_LINEAR, as EXP1 did.
        """
        w_c = torch.as_tensor(w_c, dtype=self.dtype, device=self.device)
        B = w_c.shape[0]
        H, W = self.rays.shape[:2]
        th = w_c.norm(dim=-1)
        z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=self.dtype, device=self.device)
        axis = torch.where(th[:, None] > 1e-12, w_c / th.clamp_min(1e-12)[:, None], z_axis)
        rays = self.rays.expand(B, H, W, 3)
        u_raw, v_raw = self.u0.expand(B, H, W), self.v0.expand(B, H, W)
        if depth is not None:
            depth = torch.as_tensor(depth, device=self.device).to(self.dtype)[:, None]
            if depth.shape[-2:] != (H, W):
                depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)
            v_c = torch.as_tensor(v_c, dtype=self.dtype, device=self.device)[:, None, None, :]
        X_r = None
        for _ in range(N_ITER):
            dt = self.scan.dt(u_raw, v_raw)
            ang = -th[:, None, None] * dt
            if depth is None:
                X_r = rotate(rays, axis, ang)
            else:
                # depth is read where the source point currently lands in the (RS) pinhole image
                if X_r is None:
                    xs, ys = rays[..., 0], rays[..., 1]
                else:
                    xs, ys = X_r[..., 0] / X_r[..., 2], X_r[..., 1] / X_r[..., 2]
                Z = _sample(depth, xs * self.K[0, 0] + self.K[0, 2], ys * self.K[1, 1] + self.K[1, 2])[:, 0]
                X_r = rotate(rays * Z[..., None] - v_c * dt[..., None], axis, ang)
            u_raw, v_raw = self.project(X_r)
        return u_raw, v_raw

    @staticmethod
    def remap(raw, u_raw, v_raw):
        """raw: (B, C, h, w) float images; returns (B, C, H, W) sampled with bilinear + border replicate."""
        return _sample(raw, u_raw, v_raw)


def tone_map_u8(x, gamma=GAMMA):
    """prep.py's tone curve: linear [0, 1] -> 8-bit, round half up."""
    return (255.0 * x.clamp(0, 1).pow(gamma) + 0.5).clamp(0, 255).to(torch.uint8)
