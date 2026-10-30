"""EXP2 §8.6 LoRA training controls (6a: synthetic RS train split, 6b: its GS twins) -- shared pieces.

- LoRA (peft) on the linear layers of the multi-view transformer only; checkpoint save / load
- training forwards: the models' own forward code, with per-block gradient checkpointing
- training samples: one clip, a random subset of its frames (temporal order kept), GT on the model's pixel grid
- validation through the evaluation runners (infer2.Pi3Runner / VGGTRunner .load/.run), so val uses the test protocol

GT (brief §8.6): camera pose = T_mid[k] (the frame's centre-line pose, one rigid pose per image, no motion modelling);
per-pixel GT point in the frame's camera = K^-1 [u, v, 1] * depth[k][v, u], depth from the frame's own depth map
(depth.npz 'rs' for RS frames, 'gs' for the GS twins). On the model grid: ray through the exact original position of
each model pixel centre (infer2's measured pixel map), depth from the nearest original pixel; invalid where there is
no hit or at a depth edge (same edge rule as dense_utils.Clip.gt_points).
"""
import glob
import math
import os
import subprocess
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

HOME = os.path.expanduser("~")
ROOT = f"{HOME}/rs3d"
SRC = f"{ROOT}/src"
PI3_DIR = f"{ROOT}/third_party/pi3"
VGGT_TRAIN = f"{ROOT}/third_party/vggt/training"
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, PI3_DIR, SRC):  # same precedence as infer2 (src first, then pi3)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# peft imports transformers, which imports torchaudio; the venv's torchaudio build needs libcudart.so.13 and fails to
# load. Nothing here uses audio: a None entry makes importlib.util.find_spec return None, so transformers skips it.
if "torchaudio" not in sys.modules:
    sys.modules["torchaudio"] = None

LORA_TARGETS = {  # peft matches a string target with re.fullmatch on the module name
    "pi3": r"decoder\.\d+\.(attn\.qkv|attn\.proj|mlp\.fc1|mlp\.fc2)",  # the 36 alternating frame/global blocks
    "vggt": r"aggregator\.(frame_blocks|global_blocks)\.\d+\.(attn\.qkv|attn\.proj|mlp\.fc1|mlp\.fc2)",  # 24 + 24
}
N_TARGETS = {"pi3": 36 * 4, "vggt": 48 * 4}
DATASET_NAME = "ReplicaRS"


def git_commit():
    return subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def free_vram_gb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return float(out.split()[0]) / 1024


def wait_for_vram(need_gb, log=print, poll_s=60):
    """Shared-GPU rule: launch only with >= need_gb free (the caller adds the 12 GB margin)."""
    while True:
        fv = free_vram_gb()
        if fv >= need_gb:
            log(f"free VRAM {fv:.1f} GB >= {need_gb:.1f} GB: go")
            return fv
        log(f"free VRAM {fv:.1f} GB < {need_gb:.1f} GB: waiting")
        time.sleep(poll_s)


# ---------------------------------------------------------------- runners (the evaluation code path)

def infer2_module():
    """The running infer2 (when we are imported by it as an extra-runner module) or a fresh import of it."""
    m = sys.modules.get("__main__")
    if m is not None and hasattr(m, "Pi3Runner") and hasattr(m, "pixel_map"):
        return m
    import infer2
    return infer2


def make_runner(name):
    I = infer2_module()
    return {"pi3": I.Pi3Runner, "vggt": I.VGGTRunner}[name]()


def pixel_map(runner, W=640, H=512):
    return infer2_module().pixel_map(runner, W, H)


# ---------------------------------------------------------------- LoRA

def add_lora(model, name, rank, alpha, dropout=0.0):
    """Inject LoRA (A: kaiming-uniform, B: zeros, so the model is unchanged at step 0) and freeze everything else."""
    from peft import LoraConfig, inject_adapter_in_model
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none", target_modules=LORA_TARGETS[name])
    model = inject_adapter_in_model(cfg, model)
    wrapped = [n for n, m in model.named_modules() if hasattr(m, "lora_A") and len(m.lora_A) > 0]
    assert len(wrapped) == N_TARGETS[name], (len(wrapped), N_TARGETS[name])
    for k, p in model.named_parameters():
        p.requires_grad_("lora_" in k)
    return model


def lora_state(model):
    return {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}


def save_lora(path, model, meta):
    tmp = path + ".tmp"
    torch.save(dict(lora=lora_state(model), meta=meta), tmp)
    os.replace(tmp, path)


def load_lora(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    add_lora(model, meta["model"], meta["rank"], meta["alpha"], meta.get("dropout", 0.0))
    res = model.load_state_dict(ck["lora"], strict=False)
    miss = [k for k in res.missing_keys if "lora_" in k]
    assert not miss and not res.unexpected_keys, (miss[:3], res.unexpected_keys[:3])
    assert len(ck["lora"]) == 2 * N_TARGETS[meta["model"]]
    return meta


def grad_checkpoint_blocks(blocks):
    """Activation checkpointing per transformer block whenever autograd is recording (training); a plain call
    otherwise, so validation / inference run the untouched forward. Module names (and LoRA keys) are unchanged."""
    from torch.utils.checkpoint import checkpoint
    for blk in blocks:
        if getattr(blk, "_rs3d_ckpt", False):
            continue
        f = blk.forward

        def fwd(*a, _f=f, **k):
            if torch.is_grad_enabled():
                return checkpoint(_f, *a, use_reentrant=False, **k)
            return _f(*a, **k)
        blk.forward = fwd
        blk._rs3d_ckpt = True


def prepare_for_training(name, model):
    model.eval()  # no dropout / batch-norm anywhere; checkpointing is keyed on grad mode, not train()
    if name == "pi3":
        blocks = list(model.decoder) + [b for d in (model.point_decoder, model.conf_decoder, model.camera_decoder)
                                         for b in d.blocks]
    else:
        blocks = list(model.aggregator.frame_blocks) + list(model.aggregator.global_blocks)
    grad_checkpoint_blocks(blocks)
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------- training forwards

def pi3_forward(model, imgs, need_conf):
    """Pi3.forward (third_party/pi3/pi3/models/pi3.py @ 9fa3ddb) line for line; the confidence branch only when the
    confidence loss is on. Call under bf16 autocast (as infer2.Pi3Runner does)."""
    imgs = (imgs - model.image_mean) / model.image_std
    B, N, _, H, W = imgs.shape
    patch_h, patch_w = H // 14, W // 14
    imgs = imgs.reshape(B * N, _, H, W)
    with torch.no_grad():  # frozen DINO encoder, no LoRA: no graph needed
        hidden = model.encoder(imgs, is_training=True)
    if isinstance(hidden, dict):
        hidden = hidden["x_norm_patchtokens"]
    hidden, pos = model.decode(hidden, N, H, W)
    point_hidden = model.point_decoder(hidden, xpos=pos)
    conf_hidden = model.conf_decoder(hidden, xpos=pos) if need_conf else None
    camera_hidden = model.camera_decoder(hidden, xpos=pos)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        point_hidden = point_hidden.float()
        ret = model.point_head([point_hidden[:, model.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
        xy, z = ret.split([2, 1], dim=-1)
        z = torch.exp(z)
        local_points = torch.cat([xy * z, z], dim=-1)
        conf = None
        if need_conf:
            conf_hidden = conf_hidden.float()
            conf = model.conf_head([conf_hidden[:, model.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
        camera_hidden = camera_hidden.float()
        camera_poses = model.camera_head(camera_hidden[:, model.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)
    return dict(local_points=local_points, conf=conf, camera_poses=camera_poses)


def vggt_forward(model, imgs):
    """VGGT.forward (third_party/vggt/vggt/models/vggt.py) for the camera and depth heads (the point and track heads
    are not used by VGGT's default training loss): aggregator under the caller's bf16 autocast, heads in fp32."""
    toks, ps_idx = model.aggregator(imgs)
    with torch.amp.autocast("cuda", enabled=False):
        pose_enc_list = model.camera_head(toks)
        depth, depth_conf = model.depth_head(toks, images=imgs, patch_start_idx=ps_idx)
    return dict(pose_enc=pose_enc_list[-1], pose_enc_list=pose_enc_list, depth=depth, depth_conf=depth_conf)


# ---------------------------------------------------------------- losses

def pi3_loss_fn(conf):
    from vendor_pi3.loss import Pi3Loss
    return Pi3Loss(train_conf=conf)


def pi3_loss(loss_fn, pred, b):
    """Pi3Loss (pi3 training branch) on our batch: GT world points = T_mid . local GT points."""
    B, N = b["local"].shape[:2]
    c2w = b["c2w"]
    world = torch.einsum("bnij,bnhwj->bnhwi", c2w[..., :3, :3], b["local"]) + c2w[:, :, None, None, :3, 3]
    views = [dict(pts3d=world[:, i], valid_mask=b["valid"][:, i], camera_pose=c2w[:, i], img=b["imgs"][:, i],
                  dataset=[DATASET_NAME] * B) for i in range(N)]
    loss, det = loss_fn(pred, views)
    return loss, {k: float(v) for k, v in det.items()}


_VGGT_LOSS = None


def vggt_train_modules():
    """VGGT's own training loss and batch normalisation (third_party/vggt/training/{loss.py,train_utils/})."""
    global _VGGT_LOSS
    if _VGGT_LOSS is None:
        if "iopath" not in sys.modules:  # train_utils.general imports iopath for checkpoint I/O only (unused here)
            for nm in ("iopath", "iopath.common", "iopath.common.file_io"):
                sys.modules[nm] = types.ModuleType(nm)
            sys.modules["iopath.common.file_io"].g_pathmgr = None
        if VGGT_TRAIN not in sys.path:
            sys.path.append(VGGT_TRAIN)
        import importlib.util
        spec = importlib.util.spec_from_file_location("vggt_training_loss", f"{VGGT_TRAIN}/loss.py")
        L = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(L)
        from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
        _VGGT_LOSS = (L, normalize_camera_extrinsics_and_points_batch)
    return _VGGT_LOSS


def vggt_batch(b, dev):
    """VGGT trainer._process_batch: extrinsics (world-to-camera) relative to the first camera; scene scaled so the
    valid GT points (in the first camera's frame) have unit mean distance; depths scaled alike."""
    _, normalize = vggt_train_modules()
    c2w = b["c2w"].double()
    w2c = torch.linalg.inv(c2w)[..., :3, :].float()
    loc = b["local"]
    world = (torch.einsum("bnij,bnhwj->bnhwi", c2w[..., :3, :3].float(), loc)
             + c2w[:, :, None, None, :3, 3].float())
    ext, cam_pts, world_pts, depths = normalize(extrinsics=w2c, cam_points=loc, world_points=world,
                                                depths=b["depth"], point_masks=b["valid"])
    B, N = loc.shape[:2]
    return dict(images=b["imgs"].to(dev), extrinsics=ext.to(dev), intrinsics=b["K"][:, None].expand(B, N, 3, 3).to(dev),
                depths=depths.to(dev), point_masks=b["valid"].to(dev), world_points=world_pts.to(dev),
                cam_points=cam_pts.to(dev))


def vggt_loss(pred, batch, conf):
    """training/config/default.yaml: camera (L1 on the pose encoding, 4 refinement stages, gamma 0.6, focal weight
    0.5) x 5 + depth x 1 = conf term (gamma 1, alpha 0.2) + plain regression + gradient loss ("grad"), outlier
    quantile 0.98. conf=False drops the confidence term only (the default "grad" gradient loss is not conf-weighted)."""
    L, _ = vggt_train_modules()
    cam = L.compute_camera_loss(pred, batch, loss_type="l1", gamma=0.6, pose_encoding_type="absT_quaR_FoV",
                                weight_trans=1.0, weight_rot=1.0, weight_focal=0.5)
    dep = L.compute_depth_loss(pred, batch, gamma=1.0, alpha=0.2, gradient_loss_fn="grad", valid_range=0.98)
    depth_loss = dep["loss_reg_depth"] + dep["loss_grad_depth"] + (dep["loss_conf_depth"] if conf else 0.0)
    total = 5.0 * cam["loss_camera"] + 1.0 * depth_loss
    det = {**{k: float(v) for k, v in cam.items()}, **{k: float(v) for k, v in dep.items()}}
    return total, det


# ---------------------------------------------------------------- data

def grid_gt(D, K, pm):
    """D (n, H, W) metric z-depth (0 = no hit), K (3, 3) original intrinsics, pm infer2 pixel map.
    -> local points (n, h, w, 3) in the frame camera, z-depth (n, h, w), valid (n, h, w) on the model grid."""
    from dense_utils import model_pixels_orig
    n, H, W = D.shape
    u, v = model_pixels_orig(pm)
    ui = np.clip(np.round(u), 0, W - 1).astype(int)
    vi = np.clip(np.round(v), 0, H - 1).astype(int)
    Z = D[:, vi, ui]
    edge = np.stack([cv2.dilate((np.abs(cv2.Laplacian(d, cv2.CV_32F)) > 0.05 * np.maximum(d, 1e-3)).astype(np.uint8),
                                np.ones((3, 3), np.uint8)) for d in D])
    valid = (Z > 0) & (edge[:, vi, ui] == 0)
    ray = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T
    pts = ray[None] * Z[..., None]
    return pts.astype(np.float32), Z.astype(np.float32), valid


def model_K(K, pm):
    """Intrinsics on the model grid (original pixel = s * model pixel + o)."""
    return np.array([[K[0, 0] / pm["sx"], 0, (K[0, 2] - pm["ox"]) / pm["sx"]],
                     [0, K[1, 1] / pm["sy"], (K[1, 2] - pm["oy"]) / pm["sy"]],
                     [0, 0, 1.0]])


def load_clip_gt(d, cond, frames, pm):
    dep = np.load(f"{d}/depth.npz")[cond][frames].astype(np.float32)
    tr = np.load(f"{d}/traj.npz")
    K = tr["K"].astype(np.float64)
    pts, Z, valid = grid_gt(dep, K, pm)
    return dict(local=pts, depth=Z, valid=valid, c2w=tr["T_mid"][frames].astype(np.float64), K=model_K(K, pm))


class ClipFrames(torch.utils.data.Dataset):
    """Sample i: one train clip (epoch-wise permutations from the run seed) and a random subset of n in
    [frames[0], frames[1]] of its 16 frames, temporal order kept, from condition folder `cond` (rs | gs)."""

    def __init__(self, dirs, cond, seed, length, frames, load_fn, pm, n_total=16):
        rng = np.random.default_rng(seed)
        reps = length // len(dirs) + 1
        order = np.concatenate([rng.permutation(len(dirs)) for _ in range(reps)])[:length]
        self.items = []
        for ci in order:
            n = int(rng.integers(frames[0], frames[1] + 1))
            self.items.append((dirs[ci], np.sort(rng.choice(n_total, n, replace=False))))
        self.cond, self.load_fn, self.pm = cond, load_fn, pm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d, fr = self.items[i]
        x = self.load_fn([f"{d}/{self.cond}/{k:02d}.png" for k in fr])
        g = load_clip_gt(d, self.cond, fr, self.pm)
        return dict(imgs=x.float(), local=torch.from_numpy(g["local"]), depth=torch.from_numpy(g["depth"]),
                    valid=torch.from_numpy(g["valid"]), c2w=torch.from_numpy(g["c2w"]).float(),
                    K=torch.from_numpy(g["K"]).float(), clip=os.path.basename(os.path.dirname(d)) + "/"
                    + os.path.basename(d), frames=torch.from_numpy(fr))


# ---------------------------------------------------------------- validation

def _prefetch(fn, items, n_threads=4, depth=6):
    with ThreadPoolExecutor(n_threads) as ex:
        futs = [ex.submit(fn, it) for it in items[:depth]]
        for i in range(len(items)):
            r = futs[i].result()
            futs[i] = None
            if i + depth < len(items):
                futs.append(ex.submit(fn, items[i + depth]))
            yield items[i], r


@torch.no_grad()
def validate(runner, dirs, conds, pm):
    """Every val clip, all 16 frames in order, through runner.load / runner.run (the infer2 test path).
    Per-clip rows: Sim(3) ATE etc. (common.pose_metrics vs T_mid; translation metrics NaN below EXP0's 0.20 m path
    filter, as synth_eval), AUC30, depth AbsRel / delta1 (per-view median scaling, metrics2.depth_metrics)."""
    from common import MIN_PATH_M, pose_metrics
    from metrics2 import depth_metrics, pose_auc
    rows = []
    for cond in conds:
        items = [(d, sorted(glob.glob(f"{d}/{cond}/*.png"))) for d in dirs]
        for (d, paths), x in _prefetch(lambda it: runner.load(it[1]), items):
            for attempt in range(30):  # shared GPU: wait out brief memory dips
                try:
                    out = runner.run(x, dense=True)
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"val OOM (attempt {attempt + 1}), retry in 60 s", flush=True)
                    time.sleep(60)
            else:
                raise RuntimeError("validation: 30 OOM retries")
            P = out["poses"].astype(np.float64)
            g = load_clip_gt(d, cond, np.arange(len(paths)), pm)
            G = g["c2w"]
            r = dict(cond=cond, clip=os.path.basename(os.path.dirname(d)) + "/" + os.path.basename(d))
            r.update(pose_metrics(P, G))
            r["AUC30"] = pose_auc(P, G)
            r["gt_path_m"] = float(np.linalg.norm(np.diff(G[:, :3, 3], axis=0), axis=1).sum())
            if r["gt_path_m"] < MIN_PATH_M:
                for k in ("ATE", "RPE_trans", "RTA5", "AUC30", "sim3_scale"):
                    r[k] = np.nan
            r.update(depth_metrics(out["local_pts"][..., 2], g["depth"], g["valid"]))
            rows.append(r)
    return rows


def summarize_val(rows, cond):
    import pandas as pd
    df = pd.DataFrame(rows)
    df = df[df.cond == cond]
    return dict(ATE_mean=float(df.ATE.mean()), ATE_median=float(df.ATE.median()), AUC30=float(df.AUC30.mean()),
                RRA5=float(df.RRA5.mean()), RPE_rot=float(df.RPE_rot.mean()), absrel=float(df.absrel.mean()),
                delta1=float(df.delta1.mean()), n=int(len(df)), n_moving=int(df.ATE.notna().sum()))


def lr_lambda(warmup, max_steps):
    def f(s):
        if s < warmup:
            return (s + 1) / warmup
        p = min(1.0, (s - warmup) / max(1, max_steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * p))
    return f
