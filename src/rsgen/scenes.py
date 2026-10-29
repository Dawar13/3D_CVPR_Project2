"""Meshes for rsgen: Replica-style vertex-coloured PLYs, a procedural test room, and free-space queries."""
import numpy as np
import open3d as o3d

from .core import SRGB_GAMMA

NTHREADS = 12  # the GPU box is shared: leave CPUs for the other job


_PLY_TYPES = dict(char="i1", uchar="u1", int8="i1", uint8="u1", short="i2", ushort="u2", int16="i2", uint16="u2",
                  int="i4", uint="u4", int32="i4", uint32="u4", float="f4", float32="f4", double="f8", float64="f8")


def read_ply(path):
    """Binary little-endian PLY with per-vertex RGB and polygon faces -> vertices, triangles, colours in [0, 1].

    Open3D rejects Replica's quad meshes ("a polygon could not be decomposed"), so faces are fan-triangulated here.
    """
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        body = f.read()
    assert "format binary_little_endian 1.0" in header, f"{path}: only binary little-endian PLY is supported"
    elems, cur = [], None
    for line in header:
        p = line.split()
        if p[0] == "element":
            cur = dict(name=p[1], n=int(p[2]), props=[])
            elems.append(cur)
        elif p[0] == "property":
            cur["props"].append(p[1:])
    off, V, C, F = 0, None, None, None
    for e in elems:
        if e["name"] == "vertex":
            dt = np.dtype([(p[-1], "<" + _PLY_TYPES[p[0]]) for p in e["props"]])
            a = np.frombuffer(body, dt, e["n"], off)
            off += dt.itemsize * e["n"]
            V = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
            C = np.stack([a["red"], a["green"], a["blue"]], 1).astype(np.float32) / 255
        elif e["name"] == "face":
            (_, ct, it, _), = e["props"]
            ct, it = np.dtype("<" + _PLY_TYPES[ct]), np.dtype("<" + _PLY_TYPES[it])
            k = int(np.frombuffer(body, ct, 1, off)[0])
            dt = np.dtype([("n", ct), ("idx", it, (k,))])
            a = np.frombuffer(body, dt, e["n"], off)
            assert (a["n"] == k).all(), f"{path}: mixed polygon sizes are not supported"
            off += dt.itemsize * e["n"]
            q = a["idx"].astype(np.int64)
            F = np.concatenate([q[:, [0, j, j + 1]] for j in range(1, k - 1)])
        else:
            raise ValueError(f"{path}: unexpected element {e['name']}")
    return V, F, C


class Scene:
    """A static triangle mesh with per-vertex colours (display-referred, [0, 1]) behind an Embree ray caster.

    Colours are interpolated in linear light: corner colours are linearised once and stored per triangle.
    """

    def __init__(self, vertices, triangles, colors, scale=1.0, nthreads=NTHREADS):
        self.V = np.asarray(vertices, np.float32) * np.float32(scale)
        self.F = np.asarray(triangles, np.int64)
        self.C = np.asarray(colors, np.float32)
        self.FC = np.power(np.clip(self.C, 0, 1), SRGB_GAMMA)[self.F]  # (F, 3 corners, 3) linear
        self.rc = o3d.t.geometry.RaycastingScene()
        self.rc.add_triangles(o3d.core.Tensor(self.V), o3d.core.Tensor(self.F.astype(np.uint32)))
        self.nthreads = nthreads
        self.lo, self.hi = self.V.min(0), self.V.max(0)
        self.rc.compute_distance(o3d.core.Tensor(self.V[:1]))  # builds the BVH now, not inside the first render

    @classmethod
    def from_ply(cls, path, scale=1.0, nthreads=NTHREADS):
        V, F, C = read_ply(path)
        return cls(V, F, C, scale, nthreads)

    def cast(self, o, d):
        """Rays o + t d, (N, 3) each. Returns t_hit (float32, inf on a miss) and linear colour (N, 3)."""
        rays = np.empty((len(d), 6), np.float32)
        rays[:, :3], rays[:, 3:] = o, d
        ans = self.rc.cast_rays(o3d.core.Tensor(rays), nthreads=self.nthreads)
        th = ans["t_hit"].numpy()
        hit = np.isfinite(th)
        all_hit = bool(hit.all())
        pid = ans["primitive_ids"].numpy()
        uv = ans["primitive_uvs"].numpy()
        if not all_hit:
            pid, uv = pid[hit], uv[hit]
        w = np.empty((len(uv), 3), np.float32)  # Embree barycentrics: p = (1-u-v) v0 + u v1 + v v2 (checked)
        w[:, 1:] = uv
        w[:, 0] = 1 - uv[:, 0] - uv[:, 1]
        col = np.einsum("nk,nkc->nc", w, self.FC[pid])
        if all_hit:
            return th, col
        rgb = np.zeros((len(th), 3), np.float32)
        rgb[hit] = col
        return th, rgb

    def distance(self, p):
        q = o3d.core.Tensor(np.asarray(p, np.float32).reshape(-1, 3))
        return self.rc.compute_distance(q, nthreads=self.nthreads).numpy()

    def free_dist(self, p, dirs):
        """Distance to the first hit from point p along each unit direction (inf if open)."""
        o = np.broadcast_to(np.asarray(p, np.float64), dirs.shape)
        th, _ = self.cast(o, dirs)
        return th

    def enclosed(self, p):
        """True if rays in the 6 axis directions from p all hit geometry (p is inside the room, not outside)."""
        d = np.concatenate([np.eye(3), -np.eye(3)])
        return bool(np.isfinite(self.free_dist(p, d)).all())


class MovingObject:
    """A rigid mesh moving along its own trajectory (object-to-world poses from core.Trajectory)."""

    def __init__(self, scene, traj):
        self.scene, self.traj = scene, traj

    def poses(self, tq):
        return self.traj.poses(tq)


def _grid_quad(origin, eu, ev, nu, nv):
    """Vertices and triangles of a subdivided planar quad origin + a eu + b ev, a, b in [0, 1]."""
    a, b = np.meshgrid(np.linspace(0, 1, nu + 1), np.linspace(0, 1, nv + 1))
    V = origin + a[..., None] * eu + b[..., None] * ev
    idx = np.arange((nu + 1) * (nv + 1)).reshape(nv + 1, nu + 1)
    q = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
    F = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    return V.reshape(-1, 3), F


def _texture(P, rng, n_blobs=60):
    """Procedural albedo: a 25 cm checker, coloured blobs ('posters') and per-vertex noise."""
    chk = ((np.floor(P[:, 0] / 0.25) + np.floor(P[:, 1] / 0.25) + np.floor(P[:, 2] / 0.25)) % 2)
    col = 0.35 + 0.25 * chk[:, None] * np.array([1.0, 0.9, 0.8])
    for _ in range(n_blobs):
        c = rng.uniform(P.min(0), P.max(0))
        r = rng.uniform(0.15, 0.6)
        m = np.linalg.norm(P - c, axis=1) < r
        col[m] = rng.uniform(0.05, 0.95, 3)
    col += rng.normal(0, 0.04, col.shape)
    return np.clip(col, 0, 1)


def test_room(size=(6.0, 5.0, 3.0), res=0.03, n_boxes=8, seed=0):
    """Closed box room (z up) with textured walls and furniture boxes. For local tests of the renderer."""
    rng = np.random.default_rng(seed)
    X, Y, Z = size
    Vs, Fs = [], []

    def add(o, eu, ev):
        nu = max(1, int(np.linalg.norm(eu) / res))
        nv = max(1, int(np.linalg.norm(ev) / res))
        V, F = _grid_quad(np.asarray(o, float), np.asarray(eu, float), np.asarray(ev, float), nu, nv)
        Fs.append(F + sum(len(v) for v in Vs))
        Vs.append(V)

    def box(lo, hi):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        dx, dy, dz = hi - lo
        add(lo, [dx, 0, 0], [0, dy, 0]), add(lo + [0, 0, dz], [dx, 0, 0], [0, dy, 0])
        add(lo, [dx, 0, 0], [0, 0, dz]), add(lo + [0, dy, 0], [dx, 0, 0], [0, 0, dz])
        add(lo, [0, dy, 0], [0, 0, dz]), add(lo + [dx, 0, 0], [0, dy, 0], [0, 0, dz])

    box([0, 0, 0], [X, Y, Z])
    for _ in range(n_boxes):
        w = rng.uniform([0.4, 0.4, 0.4], [1.2, 1.2, 1.6])
        lo = rng.uniform([0, 0, 0], [X - w[0], Y - w[1], 0])
        lo[2] = 0
        box(lo, lo + w)
    V = np.concatenate(Vs)
    return Scene(V, np.concatenate(Fs), _texture(V, rng))


def sample_start(scene, rng, z_range=(1.0, 1.8), clearance=0.6, up=2, tries=2000):
    """A camera start point inside the room with at least `clearance` metres to any surface."""
    lo, hi = scene.lo.astype(float), scene.hi.astype(float)
    for _ in range(tries):
        p = rng.uniform(lo, hi)
        p[up] = lo[up] + rng.uniform(*z_range)
        if scene.enclosed(p) and scene.distance(p)[0] >= clearance:
            return p
    raise RuntimeError("no free start point found")
