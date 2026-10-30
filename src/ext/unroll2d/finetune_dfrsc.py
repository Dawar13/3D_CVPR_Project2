"""Fine-tune DFRSC (Cao et al. CVPR 2024, authors' network code) on our synthetic RS clips, early-stopped on val.

Data (main agent's generator): <set>/{train,val}/<scene>/<clip>/
  rs/{k:02d}.png, rs_nb/{k:02d}_m1.png, rs_nb/{k:02d}_p1.png (RS video neighbours 1/fps before/after),
  gs/{k:02d}.png (GS twin at the RS centre-row instant = training target), meta.json (shutter readout, direction, fps,
  "neighbors": true only for clips with rs_nb). Only clips with neighbours are used. The test split is never read.

Sample = (clip, k): input frames (k-1/fps, k, k+1/fps) [3F] or (k-1/fps, k) [2F], target gs/k.
Scan direction is canonicalised to top->bottom (bt: vertical flip, lr: transpose) for input and target alike.
Readout conditioning (our change, matching run_tum --gamma): DFRSC's time-map input (row offset from the centre
row / (H-1)) is multiplied by gamma = readout * fps, i.e. it becomes the per-row time offset in frame intervals.
The released Fastec/Carla weights were trained with gamma = 1 (time map unscaled), so initialising from them is
consistent; readout 0 gives an all-zero time map. At TUM test time use gamma = 0.6.

Training recipe follows the authors' configs/DFRSC_3F_*.yml where possible: Charbonnier pixel loss + 0.05 x
distortion-flow loss (DFRscModel.optimize_parameters: RS warped by each level's flow vs GS, masked), Adam
betas (0.9, 0.99), EMA 0.999, crops of 512 (rows) x 256 (cols) with random horizontal flip (authors: full-height
480x256 crops + hflip). Changes: 30% grey samples (TUM is grey), lr 1e-4 (fine-tuning, authors 4e-4 from scratch), no VGG perceptual term by default
(--perceptual enables the authors' conv5_4 L1 x 1e-3), early stopping on val PSNR (EMA weights), max --iters.

Outputs ~/rs3d/weights/dfrsc_ft/<name>/{best.pth (params_ema), last.pth, log.csv, config.json}

usage: python finetune_dfrsc.py --init fastecrs --frames 3 --name dfrsc3f-ft [--iters 30000] [--val_every 1000]
"""
import argparse
import glob
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import methods  # noqa: E402

WT = os.path.expanduser("~/rs3d/weights")


def list_samples(split_dir, frames):
    out = []
    for m in sorted(glob.glob(f"{split_dir}/*/*/meta.json")):
        meta = json.load(open(m))
        if not meta.get("neighbors", False):
            continue
        d = os.path.dirname(m)
        g = meta["shutter"]["readout"] * meta["fps"]
        for p in sorted(glob.glob(f"{d}/rs/*.png")):
            k = int(os.path.basename(p)[:2])
            ins = [f"{d}/rs_nb/{k:02d}_m1.png", p] + ([f"{d}/rs_nb/{k:02d}_p1.png"] if frames == 3 else [])
            if all(os.path.exists(q) for q in ins) and os.path.exists(f"{d}/gs/{k:02d}.png"):
                out.append((ins, f"{d}/gs/{k:02d}.png", meta["shutter"]["direction"], g))
    return out


def canon(a, d):
    """a (..., H, W, C) numpy -> top->bottom scan orientation."""
    if d == "bt":
        return a[..., ::-1, :, :]
    if d == "lr":
        return np.swapaxes(a, -3, -2)
    return a


class Clips(Dataset):
    def __init__(self, samples, crop=None, hflip=False, gray_p=0.0):
        self.s, self.crop, self.hflip, self.gray_p = samples, crop, hflip, gray_p

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        ins, gt, d, g = self.s[i]
        x = np.stack([cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in ins])  # (N, H, W, 3)
        y = cv2.cvtColor(cv2.imread(gt), cv2.COLOR_BGR2RGB)
        x, y = canon(x, d), canon(y, d)
        H, W = y.shape[:2]
        tm = methods.time_map(H, W, (H - 1) / 2, g)[0, 0].numpy()  # (H, W) time offset in frame intervals
        if self.crop:
            ch, cw = self.crop
            t, l = random.randint(0, H - ch), random.randint(0, W - cw)
            x, y, tm = x[:, t:t + ch, l:l + cw], y[t:t + ch, l:l + cw], tm[t:t + ch, l:l + cw]
        if self.hflip and random.random() < 0.5:
            x, y, tm = x[:, :, ::-1], y[:, ::-1], tm[:, ::-1]
        if self.gray_p and random.random() < self.gray_p:  # TUM is grey: same luminance for inputs and target
            lum = np.array([0.299, 0.587, 0.114], np.float32)
            x = np.repeat((x.astype(np.float32) @ lum)[..., None], 3, -1)
            y = np.repeat((y.astype(np.float32) @ lum)[..., None], 3, -1)
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float() / 255
        tm = torch.from_numpy(np.ascontiguousarray(tm)).float()[None, None].expand(x.shape[0], 1, -1, -1)
        y = torch.from_numpy(np.ascontiguousarray(y)).permute(2, 0, 1).float() / 255
        return torch.cat([x, tm], 1), y


def charbonnier(a, b, eps=1e-12):  # basicsr CharbonnierLoss default eps
    return torch.sqrt((a - b) ** 2 + eps).mean()


def backward_warp_mask(img, flow):  # DFRscModel.backward_warp
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W, device=img.device).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H, device=img.device).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([xx, yy], 1).to(img)
    flow_ = torch.cat([flow[:, 0:1] / ((W - 1.0) / 2.0), flow[:, 1:2] / ((H - 1.0) / 2.0)], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    img = torch.cat([img, torch.ones_like(img[:, :1])], 1)
    out = F.grid_sample(img, grid_, mode="bilinear", padding_mode="border", align_corners=True)
    mask = (out[:, -1:] > 0.999).float()
    return out[:, :-1], mask


def psnr_t(a, b):
    a = (a.clamp(0, 1) * 255).round()
    b = (b * 255).round()
    mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(255.0 ** 2 / mse)


@torch.no_grad()
def evaluate(net, loader):
    net.eval()
    ps = []
    for x, y in loader:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        H, W = y.shape[-2:]
        ph, pw = (-H) % 16, (-W) % 16
        if ph or pw:
            B, N = x.shape[:2]
            x = F.pad(x.flatten(0, 1), (0, pw, 0, ph), mode="replicate").view(B, N, 4, H + ph, W + pw)
        out = net(x)[0][:, :, :H, :W]
        ps.append(psnr_t(out, y).cpu())
    net.train()
    return float(torch.cat(ps).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="~/rs3d/data/synth/core_trainval")
    ap.add_argument("--init", default="fastecrs", help="released DFRSC weights to start from")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--name", default="dfrsc3f-ft")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=6, help="evals without val improvement before stopping")
    ap.add_argument("--val_max", type=int, default=480, help="val frames used (fixed subset)")
    ap.add_argument("--perceptual", action="store_true")
    ap.add_argument("--gray_p", type=float, default=0.3, help="prob. of a grey (luminance x3) training sample")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed), np.random.seed(a.seed), torch.manual_seed(a.seed)
    root = os.path.expanduser(a.set)
    out = f"{WT}/dfrsc_ft/{a.name}"
    os.makedirs(out, exist_ok=True)

    tr = list_samples(f"{root}/train", a.frames)
    va = list_samples(f"{root}/val", a.frames)
    rng = random.Random(1)
    va = rng.sample(va, min(a.val_max, len(va)))
    print(f"train samples {len(tr)}, val samples {len(va)}", flush=True)
    tl = DataLoader(Clips(tr, crop=(512, 256), hflip=True, gray_p=a.gray_p), batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    drop_last=True, pin_memory=True, persistent_workers=True)
    # val: full frames; lr clips are 640x512 after transposition, so batch 1
    vl = DataLoader(Clips(va), batch_size=1, shuffle=False, num_workers=4)

    m = methods.DFRSC(a.init, a.frames)
    net = m.net.train()
    ema = methods.DFRSC(a.init, a.frames).net.eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, betas=(0.9, 0.99))
    vgg = None
    if a.perceptual:
        import torchvision
        vgg = torchvision.models.vgg19(weights="IMAGENET1K_V1").features[:35].cuda().eval()  # up to conv5_4
        for p in vgg.parameters():
            p.requires_grad_(False)
        mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    cfg = dict(vars(a), init_ckpt=m.ckpt, n_train=len(tr), n_val=len(va), torch=torch.__version__)
    json.dump(cfg, open(f"{out}/config.json", "w"), indent=1)
    v0 = evaluate(ema, vl)
    print(f"iter 0 val PSNR (released {a.init}, gamma-scaled time map) {v0:.3f}", flush=True)
    log = [dict(it=0, val_psnr=v0, loss=np.nan)]
    best, bad, it, t0, run_loss = v0, 0, 0, time.time(), []
    torch.save({"params_ema": ema.state_dict(), "iter": 0, "val_psnr": v0}, f"{out}/best.pth")
    while it < a.iters and bad < a.patience:
        for x, y in tl:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            img, flows = net(x)
            loss = charbonnier(img, y)
            if vgg is not None:
                loss = loss + 1e-3 * F.l1_loss(vgg((img - mean) / std), vgg((y - mean) / std))
            lf = 0.0
            for i in range(x.shape[1]):
                for fl in flows:
                    sf = fl.shape[-1] / y.shape[-1]
                    rs_i = F.interpolate(x[:, i, :3], scale_factor=sf, mode="bilinear", align_corners=False)
                    gs_i = F.interpolate(y, scale_factor=sf, mode="bilinear", align_corners=False)
                    w, msk = backward_warp_mask(rs_i, fl[:, i])
                    lf = lf + charbonnier(w * msk, gs_i * msk)
            loss = loss + 0.05 * lf
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.mul_(0.999).add_(pn.detach(), alpha=0.001)
                for be, bn in zip(ema.buffers(), net.buffers()):
                    be.copy_(bn)
            run_loss.append(float(loss))
            it += 1
            if it % 100 == 0:
                print(f"iter {it} loss {np.mean(run_loss[-100:]):.4f} {(time.time() - t0) / it:.3f} s/it", flush=True)
            if it % a.val_every == 0:
                v = evaluate(ema, vl)
                log.append(dict(it=it, val_psnr=v, loss=float(np.mean(run_loss[-a.val_every:]))))
                if v > best + 0.01:
                    best, bad = v, 0
                    torch.save({"params_ema": ema.state_dict(), "iter": it, "val_psnr": v}, f"{out}/best.pth")
                else:
                    bad += 1
                torch.save({"params_ema": ema.state_dict(), "params": net.state_dict(), "iter": it}, f"{out}/last.pth")
                import pandas as pd
                pd.DataFrame(log).to_csv(f"{out}/log.csv", index=False)
                print(f"iter {it} val PSNR {v:.3f} (best {best:.3f}, bad {bad})", flush=True)
            if it >= a.iters or bad >= a.patience:
                break
    print(f"FT_DONE best val PSNR {best:.3f} (start {v0:.3f}) after {it} iters", flush=True)


if __name__ == "__main__":
    main()
