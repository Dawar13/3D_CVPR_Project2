"""Run one 2D RS-correction net on a generated synthetic set (RGB 640x512, rs/ + rs_nb/ video neighbours).

For clip frame k the net gets rs_nb/{k}_m1.png, rs/{k}.png, rs_nb/{k}_p1.png (RS frames exactly 1/fps before /
after) in the order it expects; output -> <clip>/u2d-<method>-<weights>[-g..]/{k:02d}.png (640x512 RGB).
Resolution: same rule as TUM, the net runs at its training height (Fastec/Carla: 512 -> scale 1.0; BS-RSC: 768 ->
scale 1.5, i.e. 960x768), exact scale, antialiased bilinear down / bicubic up, align_corners=False (K preserved).
Scan direction: the nets assume top->bottom. Clips with direction 'bt' are flipped vertically and 'lr' clips are
transposed before the net and mapped back afterwards (core_test is all 'tb', so this only matters for other sets).
gamma: --gamma_from_meta feeds readout*fps as the readout ratio (only for nets that accept it: DFRSC/AdaRSC time-map
scale, JAMNet/LBCNet explicit gamma); default = released behaviour.
--ckpt loads a fine-tuned checkpoint (params_ema / state dict of the same architecture) under --name.
Also writes per-frame PSNR of RS and of the output vs the GS twin (gs/{k}.png) to
~/rs3d/results/exp2/baselines/unroll2d/synth/<set>_<split>_<name>.csv

usage: python run_synth.py --set ~/rs3d/data/synth/core_test --method dfrsc3f --weights fastecrs [--split test]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/rs3d/src"))
from dense_utils import list_clips  # noqa: E402

import methods  # noqa: E402

OUTRES = os.path.expanduser("~/rs3d/results/exp2/baselines/unroll2d/synth")
NET_H = dict(bsrsc=768, fastecrs=512, carlars=512, fastec=512, carla=512, bsrscd=512)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-10)))


def to_tb(x, d):
    """x (..., H, W) -> top->bottom scan orientation."""
    if d == "bt":
        return x.flip(-2)
    if d == "lr":
        return x.transpose(-1, -2)
    return x


def from_tb(x, d):
    if d == "bt":
        return x.flip(-2)
    if d == "lr":
        return x.transpose(-1, -2)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--method", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--gamma_from_meta", action="store_true")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--neighbours_only", action="store_true", help="skip clips without rs_nb (meta neighbors)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--net_h", type=int, default=0, help="net input height (default: training height of --weights)")
    a = ap.parse_args()
    set_dir = os.path.expanduser(a.set)
    name = a.name or (f"u2d-{a.method}-{a.weights}" + (f"-g{a.gamma:.2f}" if a.gamma else "") +
                      ("-gmeta" if a.gamma_from_meta else ""))
    clips, n_fail = list_clips(set_dir, a.split)
    if a.limit:
        clips = clips.head(a.limit)

    nets = {}

    def get_net(g):
        key = None if g is None else round(float(g), 4)
        if key not in nets:
            nets[key] = methods.build(a.method, a.weights, key)
            if a.ckpt:
                sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
                sd = sd.get("params_ema", sd.get("params", sd))
                nets[key].net.load_state_dict(sd, strict=True)
                nets[key].ckpt = a.ckpt
        return nets[key]

    pool = ThreadPoolExecutor(8)
    rows, t0, n_frames = [], time.time(), 0
    for r in clips.itertuples():
        meta = json.load(open(f"{r.dir}/meta.json"))
        if not meta.get("neighbors", False):
            if a.neighbours_only:
                continue
            sys.exit(f"{r.dir} has no rs_nb neighbours")
        d = meta["shutter"]["direction"]
        g = meta["shutter"]["readout"] * meta["fps"] if a.gamma_from_meta else a.gamma
        net = get_net(g)
        out_dir = f"{r.dir}/{name}"
        os.makedirs(out_dir, exist_ok=True)
        ks = sorted(int(f[:2]) for f in os.listdir(f"{r.dir}/rs") if f.endswith(".png"))

        def paths(k):
            return [f"{r.dir}/rs/{k:02d}.png" if o == 0 else f"{r.dir}/rs_nb/{k:02d}_{'m' if o < 0 else 'p'}{abs(o)}.png"
                    for o in net.nb]
        ims = list(pool.map(lambda k: np.stack([cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths(k)]),
                            ks))
        arr = np.stack(ims)  # (K, N, H, W, 3)
        K, N, H, W, _ = arr.shape
        with torch.no_grad():
            x = torch.from_numpy(arr).cuda().permute(0, 1, 4, 2, 3).float().div_(255)  # (K, N, 3, H, W)
            x = to_tb(x, d)
            Ht, Wt = x.shape[-2:]
            s = (a.net_h or NET_H[a.weights]) / 512.0
            h, w = int(round(Ht * s)), int(round(Wt * s))
            xs = x.flatten(0, 1)
            if s != 1.0:
                xs = F.interpolate(xs, size=(h, w), mode="bilinear", antialias=s < 1, align_corners=False)
            ph, pw = (-h) % net.mult, (-w) % net.mult
            if ph or pw:
                xs = F.pad(xs, (0, pw, 0, ph), mode="replicate")
            xs = xs.reshape(K, N, 3, h + ph, w + pw)
            outs = []
            for i in range(0, K, 4):
                outs.append(net(xs[i:i + 4]).float()[:, :, :h, :w])
            y = torch.cat(outs)
            if s != 1.0:
                y = F.interpolate(y, size=(Ht, Wt), mode="bicubic", antialias=s > 1, align_corners=False)
            y = from_tb(y, d)
            y = (y.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        for j, k in enumerate(ks):
            cv2.imwrite(f"{out_dir}/{k:02d}.png", cv2.cvtColor(y[j], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
            gsp = f"{r.dir}/gs/{k:02d}.png"
            if os.path.exists(gsp):
                gs = cv2.cvtColor(cv2.imread(gsp), cv2.COLOR_BGR2RGB)
                rows.append(dict(key=r.key, scene=r.scene, clip=r.clip, k=k, direction=d,
                                 readout_ms=meta["shutter"]["readout"] * 1e3, omega_mean=meta["index_row"]["omega_mean"],
                                 psnr_rs=psnr(arr[j, net.nb.index(0)], gs), psnr_out=psnr(y[j], gs)))
        n_frames += len(ks)
    os.makedirs(OUTRES, exist_ok=True)
    tag = f"{os.path.basename(set_dir.rstrip('/'))}_{a.split}_{name}"
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTRES}/{tag}.csv", index=False)
    info = dict(name=name, method=a.method, weights=a.weights, ckpt=list(nets.values())[0].ckpt if nets else None,
                n_clips=int(clips.shape[0]), n_frames=n_frames, s_per_frame=(time.time() - t0) / max(n_frames, 1),
                psnr_rs_mean=float(df.psnr_rs.mean()) if len(df) else None,
                psnr_out_mean=float(df.psnr_out.mean()) if len(df) else None)
    json.dump(info, open(f"{OUTRES}/{tag}.json", "w"), indent=1)
    print(json.dumps(info), flush=True)
    print("SYNTH_DONE", flush=True)


if __name__ == "__main__":
    main()
