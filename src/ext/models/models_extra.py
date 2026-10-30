"""Extra EXP2 runners for src/infer2.py: Depth Anything 3, StreamVGGT (causal streaming), VGGT-Omega.

Each class mirrors infer2.Pi3Runner:
  ckpt                      HF repo id (pinned revision in `rev`)
  __init__()                model on cuda, eval mode
  load(paths) -> x          the model's OWN default preprocessing, frames in the given order (CPU tensor)
  channel0(x) -> (N, h, w)  first channel of the preprocessed tensor (affine in input intensity)
  run(x, dense) -> dict     poses (N, 4, 4) cam-to-world OpenCV; with dense: local_pts (N, h, w, 3) camera frame,
                            conf (N, h, w)
Images only, no priors. Mixed precision exactly where the authors' own inference code puts it (see each class).

Import is cheap (no model packages at module level), so infer2.py can import this module from any env.
Each model runs in its own env (~/rs3d/envs/<name>); see ENV.md next to this file.
"""
import os
import sys

import numpy as np
import torch

TP = os.path.expanduser("~/rs3d/third_party")
DEV = "cuda"
DTYPE = torch.bfloat16  # A100: bf16 supported, which is what every author's dtype rule picks


def _add_path(p):
    if p not in sys.path:
        sys.path.append(p)  # append, so nothing in the third-party tree shadows ~/rs3d/src modules


def _c2w_from_w2c(E):
    """E (N, 3, 4) or (N, 4, 4) world-to-camera -> (N, 4, 4) camera-to-world (float64)."""
    E = np.asarray(E, dtype=np.float64)
    M = np.tile(np.eye(4), (E.shape[0], 1, 1))
    M[:, :3, :] = E[:, :3, :]
    return np.linalg.inv(M)


def _unproject(depth, K):
    """depth (N, h, w), K (N, 3, 3) in model pixels -> camera-frame points (N, h, w, 3). Same as infer2."""
    N, h, w = depth.shape
    v, u = torch.meshgrid(torch.arange(h, device=depth.device, dtype=depth.dtype),
                          torch.arange(w, device=depth.device, dtype=depth.dtype), indexing="ij")
    x = (u[None] - K[:, 0, 2, None, None]) / K[:, 0, 0, None, None]
    y = (v[None] - K[:, 1, 2, None, None]) / K[:, 1, 1, None, None]
    return torch.stack([x * depth, y * depth, depth], -1)


def _squeeze_hw(t, N):
    """(1, N, h, w[, 1]) / (N, h, w[, 1]) -> (N, h, w)."""
    t = t.reshape(N, *t.shape[-3:]) if t.shape[-1] == 1 else t.reshape(N, *t.shape[-2:])
    return t[..., 0] if t.ndim == 4 else t


# ---------------------------------------------------------------- Depth Anything 3

class DA3Runner:
    """Depth Anything 3 any-view model (ByteDance-Seed/Depth-Anything-3).

    Preprocessing: DepthAnything3.inference defaults, process_res=504, process_res_method="upper_bound_resize"
    (InputProcessor: longest side -> 504 with cv2 INTER_AREA, then each side to the nearest multiple of 14,
    ImageNet normalisation). 1280x1024 -> 504x406.
    Forward: DepthAnything3.forward (the API's own autocast: bf16 if supported) with no extrinsics/intrinsics,
    default ref_view_strategy="saddle_balanced", camera-decoder poses (use_ray_pose=False, the API default).
    Output extrinsics are world-to-camera (da3.py: extrinsics = affine_inverse(c2w)); we invert them.
    """
    ckpt = "depth-anything/DA3-GIANT-1.1"
    rev = "72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19"
    PROCESS_RES = 504
    PROCESS_METHOD = "upper_bound_resize"
    REF_VIEW = "saddle_balanced"

    def __init__(self):
        _add_path(f"{TP}/da3/src")
        from depth_anything_3.api import DepthAnything3
        self.model = DepthAnything3.from_pretrained(self.ckpt, revision=self.rev).to(DEV).eval()

    def load(self, paths):
        x, _, _ = self.model.input_processor(list(paths), None, None, self.PROCESS_RES, self.PROCESS_METHOD,
                                             num_workers=1, sequential=True)
        return x  # (N, 3, h, w) float, ImageNet-normalised

    def channel0(self, x):
        return x[:, 0].numpy()

    @torch.no_grad()
    def run(self, x, dense):
        N = x.shape[0]
        imgs = x.to(DEV)[None].float()
        out = self.model.forward(imgs, None, None, [], False, False, self.REF_VIEW)  # autocast inside
        res = dict(poses=_c2w_from_w2c(out["extrinsics"][0].float().cpu().numpy()))
        if dense:
            depth = _squeeze_hw(out["depth"].float(), N)
            K = out["intrinsics"][0].float()
            res["local_pts"] = _unproject(depth, K).cpu().numpy()
            res["conf"] = _squeeze_hw(out["depth_conf"].float(), N).cpu().numpy()
        return res


class DA3NestedRunner(DA3Runner):
    """DA3 Nested Giant-Large 1.1: any-view Giant branch (poses, depth) + metric Large branch (scale)."""
    ckpt = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    rev = "b2359bdf726fb44ef62acca04d629dcf158053e7"


# ---------------------------------------------------------------- StreamVGGT

class StreamVGGTRunner:
    """StreamVGGT (wzzheng/StreamVGGT), causal streaming with cached memory tokens.

    Preprocessing: streamvggt.utils.load_fn.load_and_preprocess_images default mode="crop" (width 518,
    height rounded to /14, BICUBIC; 1280x1024 -> 518x420, no crop needed), values in [0, 1].
    Forward: StreamVGGT.inference(frames) exactly as demo_gradio.py / eval: frames fed one at a time in the given
    order, KV cache for aggregator and camera head, under torch.cuda.amp.autocast(bf16) (demo rule: bf16 if
    compute capability >= 8). Heads run in fp32 inside the model.
    Per-frame outputs are the ones produced when that frame arrived (frame i sees frames 0..i only). The model
    never revisits earlier frames, so these are also its final outputs after the last frame.
    Pose encoding -> (extrinsic w2c, intrinsic) with streamvggt.utils.pose_enc.pose_encoding_to_extri_intri
    (VGGT convention); we invert to cam-to-world.
    """
    ckpt = "lch01/StreamVGGT"
    rev = "f9ba55b1955bc4f34337c51142158a9aa2862c7f"
    fname = "checkpoints.pth"

    def __init__(self):
        _add_path(f"{TP}/streamvggt/src")
        from huggingface_hub import hf_hub_download
        from streamvggt.models.streamvggt import StreamVGGT
        from streamvggt.utils.load_fn import load_and_preprocess_images
        from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
        self.lp, self.dec = load_and_preprocess_images, pose_encoding_to_extri_intri
        path = hf_hub_download(self.ckpt, self.fname, revision=self.rev)
        m = StreamVGGT()
        sd = torch.load(path, map_location="cpu")
        m.load_state_dict(sd, strict=True)
        del sd
        self.model = m.to(DEV).eval()

    def load(self, paths):
        return self.lp(list(paths))  # default mode="crop"

    def channel0(self, x):
        return x[:, 0].numpy()

    @torch.no_grad()
    def run(self, x, dense):
        N = x.shape[0]
        frames = [{"img": x[i:i + 1].to(DEV)} for i in range(N)]
        with torch.autocast("cuda", dtype=DTYPE):
            out = self.model.inference(frames)
        pose_enc = torch.stack([r["camera_pose"].reshape(-1) for r in out.ress])[None].float()  # (1, N, 9)
        extr, intr = self.dec(pose_enc, x.shape[-2:])
        res = dict(poses=_c2w_from_w2c(extr[0].cpu().numpy()))
        if dense:
            depth = torch.stack([_squeeze_hw(r["depth"].float(), 1)[0] for r in out.ress])
            conf = torch.stack([_squeeze_hw(r["depth_conf"].float(), 1)[0] for r in out.ress])
            res["local_pts"] = _unproject(depth, intr[0].float()).cpu().numpy()
            res["conf"] = conf.cpu().numpy()
        return res


# ---------------------------------------------------------------- VGGT-Omega

class VGGTOmegaRunner:
    """VGGT-Omega 1B (facebookresearch/vggt-omega), checkpoint vggt_omega_1b_512.pt (README quick start).

    GATED weights (facebook/VGGT-Omega, FAIR Noncommercial Research License). Loaded only if the user has put a
    token in ~/rs3d/.hf/token (we never log in or accept the licence ourselves).
    Preprocessing: vggt_omega.utils.load_fn.load_and_preprocess_images(image_resolution=512) with its default
    mode="balanced", patch 16 (1280x1024 -> 576x464), BICUBIC, values in [0, 1].
    Forward: VGGTOmega() default autocast=True, i.e. the model itself runs the aggregator under bf16 autocast
    (bf16 if supported) and the heads in fp32; called under torch.inference_mode() as in the README.
    encoding_to_camera gives [R|T] world-to-camera (eval/metrics.py builds relative poses as T_j @ inv(T_i));
    we invert to cam-to-world.
    """
    ckpt = "facebook/VGGT-Omega"
    rev = "1041e80fc0e911235d3426b0a3d9a81075111a53"
    fname = "vggt_omega_1b_512.pt"
    RES = 512

    def __init__(self):
        tok = os.path.expanduser("~/rs3d/.hf/token")
        if not os.path.exists(tok):
            raise RuntimeError("facebook/VGGT-Omega is gated and ~/rs3d/.hf/token does not exist; not downloading")
        _add_path(f"{TP}/vggt_omega")
        from huggingface_hub import hf_hub_download
        from vggt_omega.models import VGGTOmega
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        from vggt_omega.utils.pose_enc import encoding_to_camera
        self.lp, self.dec = load_and_preprocess_images, encoding_to_camera
        path = hf_hub_download(self.ckpt, self.fname, revision=self.rev, token=open(tok).read().strip())
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(sd, dict):
            sd = sd.get("model", sd.get("state_dict", sd))
        m = VGGTOmega()
        m.load_state_dict(sd, strict=True)
        del sd
        self.model = m.to(DEV).eval()

    def load(self, paths):
        return self.lp(list(paths), image_resolution=self.RES)  # default mode="balanced", patch_size=16

    def channel0(self, x):
        return x[:, 0].numpy()

    @torch.no_grad()
    def run(self, x, dense):
        N = x.shape[0]
        with torch.inference_mode():
            pred = self.model(x.to(DEV))
        extr, intr = self.dec(pred["pose_enc"].float(), pred["images"].shape[-2:])
        res = dict(poses=_c2w_from_w2c(extr[0].cpu().numpy()))
        if dense:
            depth = _squeeze_hw(pred["depth"].float(), N)
            res["local_pts"] = _unproject(depth, intr[0].float()).cpu().numpy()
            res["conf"] = _squeeze_hw(pred["depth_conf"].float(), N).cpu().numpy()
        return res


EXTRA_RUNNERS = dict(da3=DA3Runner, da3nested=DA3NestedRunner, streamvggt=StreamVGGTRunner,
                     vggt_omega=VGGTOmegaRunner)
