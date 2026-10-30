"""Uniform inference wrappers around the authors' released 2D RS-correction networks (their code, their weights).

Each wrapper exposes
  nb        neighbour offsets fed to the net, in input order, relative to the frame being corrected (offset 0)
  mult      spatial size multiple the net needs
  target    which instant the output GS image corresponds to: 'mid' = middle scanline of frame 0,
            'first' = first scanline of frame 0
  __call__(x) with x (B, N, 3, h, w) float32 RGB in [0, 1] on CUDA -> (B, 3, h, w) float32 (not clamped)

Only the authors' network code is used (imported from ~/rs3d/third_party/<repo>, unmodified); unavailable compiled
ops are replaced by pure-PyTorch shims in ./shims (correlation, forward splatting, detectron2 DCN, registries).
One process per method (the repos share module names such as package_core / model / dataloader).

Optional readout adaptation (OUR modification, not the authors'): `gamma` = readout / frame interval assumed by the
net. Default = the training value (1.0 for Fastec-RS / Carla-RS, where the authors set gamma = 1). For nets whose
code has an explicit gamma (JAMNet, LBCNet) we can set it; for nets with a time-map input (DFRSC, AdaRSC) the time
map is scaled by gamma / gamma_train. Variants built this way are labelled "-g0.60" and reported separately.
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

TP = os.path.expanduser("~/rs3d/third_party")
WT = os.path.expanduser("~/rs3d/weights")
SHIMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shims")


def _paths(*ps):
    for p in reversed(ps):
        if p not in sys.path:
            sys.path.insert(0, p)


def time_map(H, W, ref_row, scale=1.0):
    """distortion_map() of DFRSC/BSRSC/RSCD dataset/augment.py: (row - ref_row) / (H - 1), shape (1, 1, H, W)."""
    rows = torch.arange(H, dtype=torch.float32) / (H - 1)
    rf = math.floor(ref_row)
    m = rows - rows[rf] - (ref_row - rf) * (1.0 / (H - 1))
    return (m * scale).view(1, 1, H, 1).expand(1, 1, H, W).contiguous()


class _NoCache(dict):
    def get(self, k, d=None):
        return None


def _load(path, key=None):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if key is not None:
        sd = sd[key]
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


class DFRSC:
    """Cao et al. CVPR 2024. 3F: frames (t-1, t, t+1) -> GS of frame t, middle scanline (configs: pos: middle,
    sampling n_c). 2F: the released configs cover 3F only; the 2F frame order is determined empirically on the
    Fastec-RS test set (check_repro.py) and passed in as `order2f` ('prev' = (t-1, t), 'next' = (t, t+1))."""
    mult = 16
    target = "mid"

    def __init__(self, weights, frames=3, gamma=None, order2f="prev"):
        _paths(SHIMS, f"{TP}/DFRSC")
        from model.dfrsc import DFRSC as Net
        self.net = Net(in_channels=3, inner_channels=[16, 28, 40, 64, 96], out_channels=3, num_frames=frames,
                       num_flows=4, num_blocks=3).cuda().eval()
        self.ckpt = f"{WT}/dfrsc/dfrsc_{frames}f_{weights}.pth"
        self.net.load_state_dict(_load(self.ckpt, "params_ema"), strict=True)  # test configs: param_key_g params_ema
        # FlowAttention caches its coordinate grid keyed by (H, W) only, expanded to the first batch size it saw, so
        # a later call with another batch size fails. Disable the cache (grid is rebuilt each call; same values).
        for mod in self.net.modules():
            if hasattr(mod, "grids"):
                mod.grids = _NoCache()
        self.nb = [-1, 0, 1] if frames == 3 else ([-1, 0] if order2f == "prev" else [0, 1])
        g_train = 1.0 if weights in ("fastecrs", "carlars") else None
        self.tscale = 1.0 if gamma is None else gamma / (g_train or 1.0)  # unknown training gamma (BS-RSC*): taken as 1, as in finetune_dfrsc.py

    @torch.no_grad()
    def __call__(self, x):
        B, N, C, H, W = x.shape
        tm = time_map(H, W, (H - 1) / 2, self.tscale).to(x)[:, None].expand(B, N, 1, H, W)
        out, _ = self.net(torch.cat([x, tm], 2))
        return out


def _patch_adawarp_grid():
    """Speed only: AdaWarp.feats_sampling rebuilds its integer pixel grid on the CPU and copies it to the GPU on
    every call (81 times per frame), which dominated runtime on the loaded shared CPU (7.9 s/frame). Same maths with
    the grid built once on the GPU; outputs verified bit-identical (check in ENV.md)."""
    import model.rscnet as rn
    cache = {}

    def feats_sampling(self, x, flow, interpolation='bilinear', padding_mode='zeros', align_corners=True):
        if x.size()[-2:] != flow.size()[1:3]:
            raise ValueError("spatial sizes of input and flow differ")
        h, w = x.shape[-2:]
        key = (h, w, x.device, x.dtype)
        if key not in cache:
            gy, gx = torch.meshgrid(torch.arange(0, h, device=x.device), torch.arange(0, w, device=x.device),
                                    indexing="ij")
            cache[key] = torch.stack((gx, gy), 2).type_as(x)
        grid_flow = cache[key] + flow
        grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
        grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
        return F.grid_sample(x, grid_flow, mode=interpolation, padding_mode=padding_mode, align_corners=align_corners)
    rn.AdaWarp._orig_feats_sampling = rn.AdaWarp.feats_sampling
    rn.AdaWarp.feats_sampling = feats_sampling


class AdaRSC:
    """Cao et al. CVPR 2022 (BS-RSC). Frames (t-1, t, t+1) + time map -> GS of frame t, middle scanline."""
    mult = 4
    target = "mid"
    nb = [-1, 0, 1]

    def __init__(self, weights, gamma=None):
        _paths(SHIMS, f"{TP}/BSRSC")
        from model.rscnet import AdaRSCNet
        # configs/adarsc_{bsrsc,fastec}.yaml: in 3, inner 32, out 3, num_frames 3, num_flow 9, multi_scale_loss False
        self.net = AdaRSCNet(in_channels=3, inner_channels=32, out_channels=3, num_flow=9, num_frames=3,
                             multi_scale_loss=False).cuda().eval()
        self.ckpt = f"{WT}/adarsc/adarsc_{weights}.pth"
        self.net.load_state_dict(_load(self.ckpt, "model"), strict=True)
        _patch_adawarp_grid()
        g_train = 1.0 if weights == "fastecrs" else None
        self.tscale = 1.0 if gamma is None else gamma / (g_train or 1.0)  # unknown training gamma (BS-RSC*): taken as 1, as in finetune_dfrsc.py

    @torch.no_grad()
    def __call__(self, x):
        B, N, C, H, W = x.shape
        tm = time_map(H, W, (H - 1) / 2, self.tscale).to(x)[:, None].expand(B, N, 1, H, W)
        return self.net(torch.cat([x, tm], 2))[0][0]  # RSCArch.postprocess: outputs[0][0][0] (batch item 0)


class JAMNet:
    """Fan et al. CVPR 2023. Frames (t-1, t) -> GS of frame t (the second input), middle scanline (demo.sh /
    README); explicit gamma = 1.0 in add_temporal_offset."""
    mult = 16
    target = "mid"
    nb = [-1, 0]

    def __init__(self, weights, gamma=None):
        _paths(SHIMS, f"{TP}/JAMNet/deep_unroll_net", f"{TP}/JAMNet/package_core")
        import network_JAM
        self.net = network_JAM.JAMNet().cuda().eval()
        self.ckpt = f"{TP}/JAMNet/deep_unroll_weights/model_weights/{weights}/pre_net_flow.pth"
        self.net.load_state_dict(_load(self.ckpt), strict=True)
        if gamma is not None:
            g = float(gamma)

            def add_temporal_offset(self_, F_pred, H, W):  # = authors' method with gamma exposed
                grid_rows = network_JAM.generate_2D_grid(H, W)[1]
                t_flow_offset = grid_rows.unsqueeze(0).unsqueeze(0)
                tau = g * (t_flow_offset - H // 2) / H + 0.0001
                return (1.0 - tau) * F_pred[:, 0:2], (-tau) * F_pred[:, 2:4]
            n = 0
            for cls in vars(network_JAM).values():
                if isinstance(cls, type) and "add_temporal_offset" in vars(cls):
                    cls.add_temporal_offset = add_temporal_offset
                    n += 1
            assert n > 0

    @torch.no_grad()
    def __call__(self, x):
        return self.net(torch.cat([x[:, 0], x[:, 1]], 1))[0]


class LBCNet:
    """Fan et al. TPAMI 2024. Frames (t-1, t) -> GS at any time `time` in frame-interval units with t = 0 at the
    middle scanline of frame t-1: time 1.0 = middle scanline of frame t (BS-RSC weights were trained only at 1.0),
    time 1 - gamma/2 = first scanline of frame t. Explicit gamma = 1.0 in add_(initial_)temporal_offset."""
    mult = 32
    nb = [-1, 0]

    def __init__(self, weights, gamma=None, target="mid"):
        _paths(SHIMS, f"{TP}/LBCNet/deep_unroll_net", f"{TP}/LBCNet/package_core")
        import BUF_Net
        self.net = BUF_Net.BUFNet().cuda().eval()
        self.ckpt = f"{TP}/LBCNet/deep_unroll_weights/model_weights/{weights}/pre_net_flow.pth"
        self.net.load_state_dict(_load(self.ckpt), strict=True)
        g = 1.0 if gamma is None else float(gamma)
        self.target = target
        self.time = 1.0 if target == "mid" else 1.0 - g / 2
        if gamma is not None:
            def add_temporal_offset(F_pred, H, W, t):
                grid_rows = BUF_Net.generate_2D_grid(H, W)[1]
                tau = g * (grid_rows.unsqueeze(0).unsqueeze(0) - H // 2) / H + 0.0001
                return (t - tau) * F_pred[:, 0:2], (t - tau - 1.0) * F_pred[:, 2:4]

            def add_initial_temporal_offset(H, W, t):
                grid_rows = BUF_Net.generate_2D_grid(H, W)[1]
                tau = g * (grid_rows.unsqueeze(0).unsqueeze(0) - H // 2) / H + 0.0001
                return (t - tau), (t - tau - 1.0)
            BUF_Net.add_temporal_offset = add_temporal_offset
            BUF_Net.add_initial_temporal_offset = add_initial_temporal_offset

    @torch.no_grad()
    def __call__(self, x):
        return self.net(torch.cat([x[:, 0], x[:, 1]], 1), self.time)[0]


class DeepUnrollNet:
    """Liu et al. CVPR 2020. Frames (t-1, t) -> GS of frame t. The released models are run with
    --net_type netMiddle (demo.sh / inference.sh), which sets pred_mid_gs=True: middle scanline (verified: the
    repo's demo gs_1.png is byte-identical to JAMNet's demo gs_1_m.png for the same Fastec frame)."""
    mult = 8
    target = "mid"
    nb = [-1, 0]

    def __init__(self, weights, gamma=None):
        assert gamma is None
        _paths(SHIMS, f"{TP}/DeepUnrollNet/deep_unroll_net", f"{TP}/DeepUnrollNet/package_core")
        from net_unroll import Net_flow_unroll
        # model_unroll.ModelUnroll, netMiddle: (n_chan 3, n_init_feat 32, n_chan 3, True, seq_len 2, est_vel=True,
        # share_encoder=True, md=[4,4,4], pred_middle_gs=True)
        self.net = Net_flow_unroll(3, 32, 3, True, 2, est_vel=True, share_encoder=True, md=[4, 4, 4],
                                   pred_middle_gs=True).cuda().eval()
        self.ckpt = f"{WT}/deepunrollnet/pretrained_{weights}_net_G.pth"
        self.net.load_state_dict(_load(self.ckpt), strict=True)

    @torch.no_grad()
    def __call__(self, x):
        im = self.net(torch.cat([x[:, 0], x[:, 1]], 1))[0]
        return im[0] if isinstance(im, (list, tuple)) else im


class JCD:
    """Zhong et al. CVPR 2021 (joint RS correction + deblurring). Frames (t-1, t, t+1), each with the middle-row
    time map (video_inference.py: distortion_map(h, w, (h-1)/2)) -> GS of frame t, middle scanline.
    Inputs BGR / 255 (cv2 frames, no colour conversion in video_inference.py); first output = full resolution.
    The time map multiplies the predicted velocities (est_vel), so gamma scaling is a time-map scale."""
    mult = 8
    target = "mid"
    nb = [-1, 0, 1]

    def __init__(self, weights, gamma=None):
        _paths(SHIMS, f"{TP}/RSCD", f"{TP}/RSCD/packages_from_deepunrollnet/package_core")
        from model.JCD import Model
        self.net = Model().cuda().eval()
        self.ckpt = f"{WT}/jcd/checkpoints/JCD_{'BS-RSCD' if weights == 'bsrscd' else 'Fastec-RS'}.tar"
        sd = _load(self.ckpt, "state_dict")
        sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
        self.net.load_state_dict(sd, strict=True)
        g_train = 1.0 if weights == "fastec" else None
        self.tscale = 1.0 if gamma is None else gamma / (g_train or 1.0)  # unknown training gamma (BS-RSC*): taken as 1, as in finetune_dfrsc.py

    @torch.no_grad()
    def __call__(self, x):
        B, N, C, H, W = x.shape
        tm = time_map(H, W, (H - 1) / 2, self.tscale).to(x)[:, None].expand(B, N, 1, H, W)
        out = self.net(torch.cat([x.flip(2), tm], 2))[0]
        out = out[0] if isinstance(out, (list, tuple)) else out
        return out.flip(1)  # BGR -> RGB


METHODS = dict(dfrsc3f=lambda w, g=None: DFRSC(w, 3, g), dfrsc2f=lambda w, g=None: DFRSC(w, 2, g),
               adarsc=AdaRSC, jamnet=JAMNet, lbcnet=LBCNet, dun=DeepUnrollNet, jcd=JCD)
# released weights per method (names as in the repos / releases)
WEIGHTS = dict(dfrsc3f=["bsrsc", "fastecrs", "carlars"], dfrsc2f=["bsrsc", "fastecrs", "carlars"],
               adarsc=["bsrsc", "fastecrs"], jamnet=["bsrsc", "fastec", "carla"], lbcnet=["bsrsc", "fastec", "carla"],
               dun=["fastec", "carla"], jcd=["bsrscd", "fastec"])
# input scale used on TUM (1280x1024): match the training resolution (Fastec 640x480, Carla 640x448 -> 0.5,
# i.e. 640x512; BS-RSC 1024x768 -> 0.75, i.e. 960x768). Both are exact, aspect-preserving scales.
SCALE = dict(bsrsc=0.75, fastecrs=0.5, carlars=0.5, fastec=0.5, carla=0.5, bsrscd=0.5)  # BS-RSCD: 640x480 half-res


def build(method, weights, gamma=None, **kw):
    if method.startswith("dfrsc"):
        return DFRSC(weights, int(method[5]), gamma, **kw)
    return METHODS[method](weights, gamma, **kw) if kw else METHODS[method](weights, gamma)
