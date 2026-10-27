"""Shared paths, calibration, GT interpolation and pose metrics for RS-3D EXP0."""
import os
import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

ROOT = os.path.expanduser("~/rs3d")
DATA = f"{ROOT}/data/tumrs"
PREP = f"{ROOT}/data/prep"
WIN_DIR = f"{ROOT}/data/windows"
RES = f"{ROOT}/results/exp0"
CALIB_YAML = f"{DATA}/camchain-calibration-equidistant4_camimu_dataset-calib-imu1.yaml"

SEQS = list(range(1, 11))
CAMS = {"GS": "cam0", "RS": "cam1"}  # Schubert et al. 2019: left = global shutter, right = rolling shutter
W, H = 1280, 1024
ROW_TIME = 29.4737e-6  # s per row (dataset page)
READOUT = ROW_TIME * H  # ~30.18 ms
RS_MID_OFFSET = ROW_TIME * (H - 1) / 2  # RS timestamps refer to the first row -> centre row is ~15.08 ms later

N_FRAMES = 16
STRIDE = 3
WIN_STEP = 8
MIN_PATH_M = 0.20  # static filter: drop windows whose GT camera-centre path length is below this
SPEED_BINS = [0.5, 1.5]  # rad/s cut points: slow < 0.5 <= medium < 1.5 <= fast


def seq_dir(s):
    # "cal" = dataset-calib-imu1, where BOTH cameras ran in global-shutter mode (camera-identity control)
    name = "dataset-calib-imu1" if s == "cal" else f"dataset-seq{s}"
    return f"{DATA}/{name}/{name}/mav0"


def set_suffix(which):
    return "_control" if which == "control" else ""


def load_calib():
    with open(CALIB_YAML) as f:
        c = yaml.safe_load(f)
    out = {}
    for cam in ["cam0", "cam1"]:
        fx, fy, cx, cy = c[cam]["intrinsics"]
        out[cam] = dict(
            K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]),
            D=np.array(c[cam]["distortion_coeffs"], dtype=np.float64),
            T_cam_imu=np.array(c[cam]["T_cam_imu"], dtype=np.float64),
            model=c[cam]["distortion_model"],
            resolution=c[cam]["resolution"],
        )
    return out


def load_frames(s, cam):
    a = np.loadtxt(f"{seq_dir(s)}/{cam}/data.csv", delimiter=",", dtype=np.int64, usecols=0, comments="#")
    return a


def load_imu(s):
    a = np.loadtxt(f"{seq_dir(s)}/imu0/data.csv", delimiter=",", comments="#")
    return a[:, 0].astype(np.int64), a[:, 1:4]


class GTInterp:
    """Interpolates mocap T_world_imu (120 Hz) at query times (ns). Linear position, slerp rotation."""

    def __init__(self, s, max_gap_s=0.025):
        a = np.loadtxt(f"{seq_dir(s)}/mocap0/data.csv", delimiter=",", comments="#")
        self.t = a[:, 0].astype(np.int64)
        self.p = a[:, 1:4]
        q = a[:, 4:8]  # w, x, y, z
        self.rot = Rotation.from_quat(q[:, [1, 2, 3, 0]])
        self.slerp = Slerp((self.t - self.t[0]) * 1e-9, self.rot)
        self.max_gap = max_gap_s

    def valid(self, tq):
        tq = np.atleast_1d(tq)
        i = np.searchsorted(self.t, tq)
        ok = (i > 0) & (i < len(self.t))
        i = np.clip(i, 1, len(self.t) - 1)
        gap = (self.t[i] - self.t[i - 1]) * 1e-9
        return ok & (gap <= self.max_gap)

    def __call__(self, tq):
        tq = np.atleast_1d(tq)
        tt = (self.t - self.t[0]) * 1e-9
        ts = np.clip((tq - self.t[0]) * 1e-9, 0, tt[-1])  # out-of-range frames are flagged by valid()
        p = np.stack([np.interp(ts, tt, self.p[:, k]) for k in range(3)], -1)
        R = self.slerp(ts).as_matrix()
        T = np.tile(np.eye(4), (len(tq), 1, 1))
        T[:, :3, :3] = R
        T[:, :3, 3] = p
        return T


# ---------------------------------------------------------------- metrics

def umeyama_sim3(src, dst):
    """Find s, R, t minimising ||dst - (s R src + t)||^2. src, dst: (N, 3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    Dm = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Dm[2, 2] = -1
    R = U @ Dm @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = np.trace(np.diag(S) @ Dm) / var_s if var_s > 1e-12 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def rot_angle_deg(R):
    c = (np.trace(R, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def pose_metrics(P, G):
    """P, G: (N, 4, 4) cam-to-world poses (pred, GT). Returns dict of window metrics."""
    s, R, t = umeyama_sim3(P[:, :3, 3], G[:, :3, 3])
    c_al = (s * (R @ P[:, :3, 3].T)).T + t
    ate = float(np.sqrt(((c_al - G[:, :3, 3]) ** 2).sum(1).mean()))

    Ps = P.copy()
    Ps[:, :3, 3] *= s  # scale-align translations; the world rotation cancels in relative poses
    inv = np.linalg.inv
    rel_p = inv(Ps[:-1]) @ Ps[1:]
    rel_g = inv(G[:-1]) @ G[1:]
    E = inv(rel_g) @ rel_p
    rpe_rot = float(rot_angle_deg(E[:, :3, :3]).mean())
    rpe_trans = float(np.linalg.norm(E[:, :3, 3], axis=1).mean())

    # all-pairs relative rotation / translation-direction errors (VGGT-style RRA / RTA)
    i, j = np.triu_indices(len(P), 1)
    rp = inv(P[i]) @ P[j]
    rg = inv(G[i]) @ G[j]
    rerr = rot_angle_deg(np.swapaxes(rg[:, :3, :3], 1, 2) @ rp[:, :3, :3])
    tp = rp[:, :3, 3] / (np.linalg.norm(rp[:, :3, 3], axis=1, keepdims=True) + 1e-12)
    tg = rg[:, :3, 3] / (np.linalg.norm(rg[:, :3, 3], axis=1, keepdims=True) + 1e-12)
    terr = np.degrees(np.arccos(np.clip((tp * tg).sum(1), -1, 1)))
    return dict(
        ATE=ate, RPE_rot=rpe_rot, RPE_trans=rpe_trans,
        RRA5=float((rerr < 5).mean()), RTA5=float((terr < 5).mean()), sim3_scale=float(s),
    )
