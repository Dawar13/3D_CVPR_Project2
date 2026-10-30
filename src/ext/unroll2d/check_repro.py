"""Reproduction check of each released 2D RS-correction model on the authors' own sample data.

  (a) repo demo pairs (JAMNet / LBCNet / DeepUnrollNet `demo/` folders: rs_0, rs_1 + GT GS of frame 1)
  (b) Fastec-RS test set (JCD's public LMDB mirror; GT = *_global_middle.png, stored BGR), all frames with the
      needed neighbours, for weights trained on Fastec-RS
  (c) BS-RSC test set if present under ~/rs3d/data/rsc2d/BS-RSC/test (download was quota-blocked on 2026-10-29)
PSNR on uint8 RGB (as the authors' metric code). Writes results/exp2/baselines/unroll2d/repro/<tag>.json.

usage: python check_repro.py --method jamnet --weights fastec [--order2f prev|next] [--gamma 0.6]
"""
import argparse
import glob
import json
import os
import pickle
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import methods

TP = os.path.expanduser("~/rs3d/third_party")
DATA = os.path.expanduser("~/rs3d/data/rsc2d")
OUT = os.path.expanduser("~/rs3d/results/exp2/baselines/unroll2d/repro")


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-10)))


def rgb(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im.ndim == 2:
        im = np.repeat(im[..., None], 3, 2)
    return cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2RGB)


def run(net, frames):
    """frames: list of HxWx3 uint8 RGB in net.nb order -> HxWx3 uint8 RGB."""
    x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div(255)[None].cuda()
    H, W = x.shape[-2:]
    m = net.mult
    ph, pw = (-H) % m, (-W) % m
    if ph or pw:  # replicate-pad bottom/right to the size multiple, crop back afterwards
        x = F.pad(x.flatten(0, 1), (0, pw, 0, ph), mode="replicate").view(1, len(frames), 3, H + ph, W + pw)
    y = net(x)[0, :, :H, :W]
    return (y.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()


def demo_sets(method, weights):
    dom = dict(bsrsc="BSRSC", fastec="Fastec", fastecrs="Fastec", carla="Carla", carlars="Carla")[weights]
    sets = []
    for repo in ["JAMNet", "LBCNet", "DeepUnrollNet"]:
        for d in sorted(glob.glob(f"{TP}/{repo}/demo/{dom}/seq_*")):
            gt = f"{d}/gs_1_m.png" if os.path.exists(f"{d}/gs_1_m.png") else f"{d}/gs_1.png"
            sets.append((f"{repo}/{dom}/{os.path.basename(d)}", f"{d}/rs_0.png", f"{d}/rs_1.png", gt))
    return sets


def fastec_test():
    import lmdb
    root = f"{DATA}/fastec_rs_lmdb"
    info = pickle.load(open(f"{root}/fastec_rs_info_test.pkl", "rb"))
    e_in = lmdb.open(f"{root}/fastec_rs_test", readonly=True, lock=False).begin()
    e_gt = lmdb.open(f"{root}/fastec_rs_test_gt", readonly=True, lock=False).begin()

    def get(txn, s, f):
        b = txn.get(("%03d_%08d" % (s, f)).encode())
        return cv2.cvtColor(np.frombuffer(b, np.uint8).reshape(480, 640, 3), cv2.COLOR_BGR2RGB)
    return info, e_in, e_gt, get


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--order2f", default="prev")
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--max_fastec", type=int, default=0, help="0 = all test frames")
    a = ap.parse_args()
    kw = dict(order2f=a.order2f) if a.method == "dfrsc2f" else {}
    net = methods.build(a.method, a.weights, a.gamma, **kw)
    tag = f"{a.method}_{a.weights}" + (f"_{a.order2f}" if a.method == "dfrsc2f" else "") + \
          (f"_g{a.gamma:.2f}" if a.gamma else "")
    os.makedirs(f"{OUT}/{tag}", exist_ok=True)
    res = dict(method=a.method, weights=a.weights, ckpt=net.ckpt, nb=net.nb, gamma=a.gamma, demo=[], fastec=None)

    # (a) repo demo pairs (2-frame nets only; the demos have no third frame)
    if len(net.nb) == 2 and net.nb == [-1, 0]:
        for name, p0, p1, pg in demo_sets(a.method, a.weights):
            r0, r1, gt = rgb(p0), rgb(p1), rgb(pg)
            out = run(net, [r0, r1])
            res["demo"].append(dict(set=name, psnr_out=psnr(out, gt), psnr_rs=psnr(r1, gt)))
            cv2.imwrite(f"{OUT}/{tag}/{name.replace('/', '_')}.png",
                        cv2.cvtColor(np.concatenate([r1, out, gt], 1), cv2.COLOR_RGB2BGR))
    elif len(net.nb) == 2:  # 'next' order: GT of frame 0 of the pair = gs_0_m
        for name, p0, p1, pg in demo_sets(a.method, a.weights):
            pg0 = pg.replace("gs_1", "gs_0")
            if not os.path.exists(pg0):
                continue
            r0, r1, gt = rgb(p0), rgb(p1), rgb(pg0)
            out = run(net, [r0, r1])
            res["demo"].append(dict(set=name, psnr_out=psnr(out, gt), psnr_rs=psnr(r0, gt)))

    # (b) Fastec-RS test set
    if a.weights in ("fastec", "fastecrs") and os.path.isdir(f"{DATA}/fastec_rs_lmdb"):
        info, e_in, e_gt, get = fastec_test()
        lo, hi = -min(net.nb), max(net.nb)
        ps, pr, t0, n = [], [], time.time(), 0
        for s in range(info["num"]):
            L = info[s]["length"]
            for f in range(lo, L - hi):
                ins = [get(e_in, s, f + o) for o in net.nb]
                gt = get(e_gt, s, f)
                out = run(net, ins)
                ps.append(psnr(out, gt))
                pr.append(psnr(get(e_in, s, f), gt))
                if n == 0:
                    cv2.imwrite(f"{OUT}/{tag}/fastec_s{s:02d}_f{f:03d}.png",
                                cv2.cvtColor(np.concatenate([ins[net.nb.index(0)], out, gt], 1), cv2.COLOR_RGB2BGR))
                n += 1
                if a.max_fastec and n >= a.max_fastec:
                    break
            if a.max_fastec and n >= a.max_fastec:
                break
        res["fastec"] = dict(n=n, psnr_out_mean=float(np.mean(ps)), psnr_rs_mean=float(np.mean(pr)),
                             s_per_frame=(time.time() - t0) / max(n, 1))
    with open(f"{OUT}/{tag}.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(dict(tag=tag, demo=[(d["set"], round(d["psnr_rs"], 2), round(d["psnr_out"], 2))
                                         for d in res["demo"]], fastec=res["fastec"])), flush=True)


if __name__ == "__main__":
    main()
