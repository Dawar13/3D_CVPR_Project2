"""Camera trajectories with independently controlled turning speed and walking speed.

Kinds (EXP2 §6)
  handheld     pans and tilts (camera-frame yaw > pitch > roll) plus hand shake, while walking with a head bob
  drone        smooth yaw-dominant turning, smooth 3D flight
  rotation     handheld turning, no translation
  translation  walking, fixed orientation
  mixed        random axis weights and a random translation style

Speeds: omega is the target mean |angular velocity| (rad/s) and speed the target mean |velocity| (m/s), both over
the clip. Both profiles vary smoothly in time, as real motion does. Each window is binned later by its measured
speed, as in EXP0. The walker steers away from walls and a clip is rejected if it comes within 0.25 m of geometry.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .core import Trajectory

KINDS = {
    "handheld": dict(w_axes=(0.5, 1.0, 0.25), trans="walk", jitter=0.12),
    "drone": dict(w_axes=(0.25, 1.0, 0.08), trans="fly", jitter=0.02),
    "rotation": dict(w_axes=(0.5, 1.0, 0.25), trans="none", jitter=0.12),
    "translation": dict(w_axes=(0.0, 0.0, 0.0), trans="walk", jitter=0.0),
    "mixed": dict(w_axes=None, trans=None, jitter=0.08),
}


def _smooth(t, rng, n=3, fmin=0.15, fmax=1.2):
    """Sum of n sinusoids with random frequencies and phases, unit RMS."""
    f = rng.uniform(fmin, fmax, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    a = rng.uniform(0.5, 1.0, n)
    s = (a[:, None] * np.sin(2 * np.pi * f[:, None] * t[None] + ph[:, None])).sum(0)
    return s / (np.sqrt((s ** 2).mean()) + 1e-12)


def look_rotation(forward, up_vec, pitch=0.0, roll=0.0):
    """Camera-to-world rotation (OpenCV axes) looking along `forward`, then pitched and rolled in the camera frame."""
    z = forward / np.linalg.norm(forward)
    x = np.cross(-up_vec, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], 1)
    return R @ Rotation.from_euler("xz", [pitch, roll]).as_matrix()


def make_trajectory(scene, kind, omega, speed, t0, t1, rng, start, up=2, fs=1000, margin=0.25):
    """Simulate [t0, t1] at fs Hz from `start`. Returns (Trajectory, info) or raises RuntimeError on a collision."""
    cfg = dict(KINDS[kind])
    if cfg["w_axes"] is None:
        cfg["w_axes"] = tuple(rng.uniform(0, 1, 3))
        cfg["trans"] = rng.choice(["walk", "fly"])
    t = np.arange(t0, t1 + 1.0 / fs, 1.0 / fs)
    n = len(t)
    up_vec = np.eye(3)[up]
    horiz = [k for k in range(3) if k != up]

    # angular velocity in the camera frame
    w = np.stack([cfg["w_axes"][k] * _smooth(t, rng) for k in range(3)], 1)
    if cfg["jitter"] > 0:
        w += cfg["jitter"] * np.stack([_smooth(t, rng, 4, 4.0, 9.0) for _ in range(3)], 1)
    mag = np.linalg.norm(w, axis=1).mean()
    w = w * (omega / mag) if omega > 0 and mag > 0 else np.zeros_like(w)

    # translation: heading wanders slowly, the walker steers toward open space
    ang0 = rng.uniform(0, 2 * np.pi)
    dirs24 = np.zeros((24, 3))
    a24 = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    dirs24[:, horiz[0]], dirs24[:, horiz[1]] = np.cos(a24), np.sin(a24)
    free = scene.free_dist(start, dirs24)
    ang0 = a24[int(np.argmax(np.where(np.isfinite(free), free, 0)))] + rng.normal(0, 0.3)
    wander = np.radians(25) * _smooth(t, rng, 2, 0.1, 0.5)  # heading rate, rad/s
    sp = 1 + 0.2 * _smooth(t, rng, 2, 0.2, 1.0)
    vert = 0.15 * _smooth(t, rng, 2, 0.1, 0.6) if cfg["trans"] == "fly" else 0.02 * 2 * np.pi * 1.8 * np.cos(
        2 * np.pi * 1.8 * t + rng.uniform(0, 2 * np.pi))  # walk: 2 cm head bob at 1.8 Hz
    mean_sp = None
    if cfg["trans"] != "none" and speed > 0:
        mean_sp = np.mean(np.sqrt(sp ** 2 + vert ** 2))

    p = np.zeros((n, 3))
    p[0] = start
    heading = ang0
    look = max(0.8, 1.0 * speed)
    for k in range(n - 1):
        dt = t[k + 1] - t[k]
        v = np.zeros(3)
        if mean_sp is not None:
            if k % 10 == 0:  # steer every 10 ms
                hd = np.zeros(3)
                hd[horiz[0]], hd[horiz[1]] = np.cos(heading), np.sin(heading)
                ahead = scene.free_dist(p[k], hd[None])[0]
                if ahead < look:
                    free = scene.free_dist(p[k], dirs24)
                    score = np.where(np.isfinite(free), free, 50) + 0.3 * np.cos(a24 - heading)
                    target = a24[int(np.argmax(score))]
                    turn = np.angle(np.exp(1j * (target - heading)))
                    heading += np.clip(turn, -np.radians(240) * 0.01, np.radians(240) * 0.01)
            heading += wander[k] * dt
            v[horiz[0]], v[horiz[1]] = np.cos(heading) * sp[k], np.sin(heading) * sp[k]
            v[up] = vert[k]
            v *= speed / mean_sp
        p[k + 1] = p[k] + v * dt
        if k % 20 == 0 and scene.distance(p[k + 1])[0] < margin:
            raise RuntimeError("collision")

    # orientation: integrate body-frame angular velocity from a look-ahead start pose
    hd0 = np.zeros(3)
    hd0[horiz[0]], hd0[horiz[1]] = np.cos(ang0), np.sin(ang0)
    R = np.empty((n, 3, 3))
    R[0] = look_rotation(hd0, up_vec, pitch=np.radians(rng.uniform(-20, 5)), roll=np.radians(rng.normal(0, 2)))
    dR = Rotation.from_rotvec(w[:-1] * np.diff(t)[:, None]).as_matrix()
    for k in range(n - 1):
        R[k + 1] = R[k] @ dR[k]
    if scene.distance(p).min() < margin:
        raise RuntimeError("collision")
    traj = Trajectory(t, R, p)
    info = dict(kind=kind, omega_target=float(omega), speed_target=float(speed), w_axes=list(map(float, cfg["w_axes"])),
                trans=str(cfg["trans"]), jitter=float(cfg["jitter"]))
    return traj, info


def sample_clip_trajectory(scene, kind, omega, speed, t0, t1, rng, up=2, tries=50, **kw):
    """Retry start points / seeds until a collision-free clip is found."""
    from .scenes import sample_start
    for _ in range(tries):
        start = sample_start(scene, rng, up=up)
        try:
            return make_trajectory(scene, kind, omega, speed, t0, t1, rng, start, up=up, **kw)
        except RuntimeError:
            continue
    raise RuntimeError(f"no collision-free {kind} clip at omega={omega}, speed={speed}")
