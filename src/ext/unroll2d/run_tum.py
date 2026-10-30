"""Run one released 2D RS-correction net on every frame of the kept TUM RS-VIO windows.

Inputs: the prepped RS frame (data/prep/seq{s}/cam1/{t}.png, the file the RS window symlinks point to) plus its
adjacent raw cam1 frames at 20 fps (data/prep_nb, same prep recipe), in the order the net expects (methods.py nb).

Geometry / resolution (documented in ENV.md):
  1280x1024 8-bit gray (K_new pinhole) -> float [0,1] -> exact downscale by s (antialiased bilinear,
  align_corners=False) to the net's training scale (s = 0.5 -> 640x512 for Fastec/Carla weights, s = 0.75 -> 960x768
  for BS-RSC weights) -> gray replicated to 3 channels -> net -> mean of the 3 output channels -> exact upscale by
  1/s back to 1280x1024 (bicubic, align_corners=False) -> clip, round to uint8 gray PNG.
  Pixel centres map as x_full = (x_net + 0.5) / s - 0.5 in both directions, so K_new is preserved exactly; nothing
  is cropped (if a size multiple were needed we replicate-pad bottom/right and crop the pad off again).
  If a neighbour frame does not exist (sequence start/end: 3 of 9248 frames), the frame itself is used instead.

Outputs
  ~/rs3d/data/u2d/<COND>/seq{s}/{t}.png                   one straightened frame per unique RS frame
  ~/rs3d/data/windows/seq{s}/{wid:04d}_<COND>/{k:02d}_{t}.png  symlinks to the above, same names as the RS folder
  ~/rs3d/results/exp2/baselines/unroll2d/clips/<COND>.csv  key,dir list for src/infer2.py (key = {key}_<COND>)
  ~/rs3d/results/exp2/baselines/unroll2d/runs/<COND>.json  run info (ckpt, scale, timing, missing neighbours)
  COND = U2D-<method>-<weights>[-g<gamma>][-first]

usage: python run_tum.py --method dfrsc3f --weights fastecrs [--gamma 0.6] [--windows 20] [--batch 8] [--amp]
"""
import argparse
import glob
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/rs3d/src"))
from common import WIN_DIR, load_frames  # noqa: E402

import methods  # noqa: E402

ROOT = os.path.expanduser("~/rs3d")
RES = f"{ROOT}/results/exp2/gap/tum"
PREP = f"{ROOT}/data/prep"
PREP_NB = f"{ROOT}/data/prep_nb"
STORE = f"{ROOT}/data/u2d"
OUTRES = f"{ROOT}/results/exp2/baselines/unroll2d"
H0, W0 = 1024, 1280


def free_vram_gb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return float(out.split()[0]) / 1024


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def select_windows(win, n):
    """n windows spread evenly over the kept windows sorted by omega_mean (so slow, medium and fast are covered)."""
    w = win.sort_values("omega_mean").reset_index(drop=True)
    idx = np.unique(np.round(np.linspace(0, len(w) - 1, n)).astype(int))
    return w.iloc[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gamma", type=float, default=None, help="readout ratio fed to the net (our variant)")
    ap.add_argument("--target", default=None, help="lbcnet only: mid (default) or first")
    ap.add_argument("--windows", type=int, default=0, help="0 = all kept windows; N = smoke subset")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (only if it reproduces, see ENV.md)")
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--min_free_gb", type=float, default=12)
    ap.add_argument("--io_threads", type=int, default=8)
    ap.add_argument("--ckpt", default="", help="fine-tuned checkpoint (params_ema) for the same architecture")
    ap.add_argument("--cond", default="", help="condition name override (e.g. U2D-dfrsc3f-ft)")
    a = ap.parse_args()

    cond = a.cond or (f"U2D-{a.method}-{a.weights}" + (f"-g{a.gamma:.2f}" if a.gamma else "") +
                      ("-first" if a.target == "first" else ""))
    win = pd.read_csv(f"{RES}/windows.csv")
    win = win[win.keep].copy()
    if a.windows:
        win = select_windows(win, a.windows)

    fv = free_vram_gb()
    print(f"[{cond}] free VRAM {fv:.1f} GB, {len(win)} windows", flush=True)
    if fv < a.min_free_gb:
        sys.exit(f"free VRAM {fv:.1f} GB < {a.min_free_gb} GB; not launching (shared GPU rule)")

    kw = dict(target=a.target) if a.target else {}
    net = methods.build(a.method, a.weights, a.gamma, **kw)
    if a.ckpt:
        sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        net.net.load_state_dict(sd.get("params_ema", sd), strict=True)
        net.ckpt = a.ckpt
    s = a.scale or methods.SCALE[a.weights]
    h, w = int(round(H0 * s)), int(round(W0 * s))
    assert abs(h / H0 - s) < 1e-9 and abs(w / W0 - s) < 1e-9, (h, w, s)
    m = net.mult
    ph, pw = (-h) % m, (-w) % m

    # unique frames and window links
    frames, links, missing = {}, [], 0
    tpos = {}
    for r in win.itertuples():
        if r.seq not in tpos:
            t1 = load_frames(r.seq, "cam1")
            tpos[r.seq] = (t1, {int(t): i for i, t in enumerate(t1)})
        t1, pos = tpos[r.seq]
        for p in sorted(glob.glob(f"{WIN_DIR}/seq{r.seq}/{r.window_id:04d}_RS/*.png")):
            name = os.path.basename(p)
            t = int(name[3:-4])
            assert os.path.realpath(p) == os.path.realpath(f"{PREP}/seq{r.seq}/cam1/{t}.png"), p
            links.append((r.seq, r.window_id, name, t))
            if (r.seq, t) in frames:
                continue
            i = pos[t]
            ins = []
            for o in net.nb:
                if o == 0:
                    ins.append(f"{PREP}/seq{r.seq}/cam1/{t}.png")
                elif 0 <= i + o < len(t1):
                    ins.append(f"{PREP_NB}/seq{r.seq}/cam1/{int(t1[i + o])}.png")
                else:
                    ins.append(f"{PREP}/seq{r.seq}/cam1/{t}.png")
                    missing += 1
            frames[(r.seq, t)] = ins
    todo = [(k, v) for k, v in frames.items() if not os.path.exists(f"{STORE}/{cond}/seq{k[0]}/{k[1]}.png")]
    for sq in {k[0] for k in frames}:
        os.makedirs(f"{STORE}/{cond}/seq{sq}", exist_ok=True)
    print(f"[{cond}] {len(frames)} unique frames ({len(todo)} to run), {len(links)} window slots, "
          f"net input {w}x{h} (+pad {pw},{ph}), nb {net.nb}, target {net.target}, missing nb {missing}", flush=True)

    # threaded I/O: readers fill a queue of batches, writers drain an output queue
    def read_one(paths):
        return np.stack([cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths])

    rpool = ThreadPoolExecutor(a.io_threads)
    wq = queue.Queue(maxsize=64)

    def writer():
        while True:
            item = wq.get()
            if item is None:
                return
            path, img = item
            tmp = path[:-4] + ".tmp.png"
            cv2.imwrite(tmp, img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            os.replace(tmp, path)

    writers = [threading.Thread(target=writer, daemon=True) for _ in range(4)]
    for t_ in writers:
        t_.start()

    def forward(arr):
        with torch.no_grad():
            x = torch.from_numpy(arr).cuda().float().div_(255)  # (B, N, H0, W0)
            B, N = x.shape[:2]
            x = F.interpolate(x.view(B * N, 1, H0, W0), size=(h, w), mode="bilinear", antialias=True,
                              align_corners=False)
            if ph or pw:
                x = F.pad(x, (0, pw, 0, ph), mode="replicate")
            x = x.expand(-1, 3, -1, -1).reshape(B, N, 3, h + ph, w + pw).contiguous()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp):
                y = net(x)
            y = y.float()[:, :, :h, :w].mean(1, keepdim=True)
            y = F.interpolate(y, size=(H0, W0), mode="bicubic", align_corners=False)
            return (y.clamp_(0, 1) * 255).round_().to(torch.uint8)[:, 0].cpu().numpy()

    def run_chunks(arr, tries=0):
        """Shared GPU: on OOM free the cache, wait, and retry in halves (results are per-frame, so identical)."""
        try:
            return forward(arr)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if tries > 20:
                raise
            print(f"[{cond}] OOM at batch {len(arr)}; retrying smaller in 30 s", flush=True)
            time.sleep(30)
            if len(arr) == 1:
                return run_chunks(arr, tries + 1)
            k = len(arr) // 2
            return np.concatenate([run_chunks(arr[:k], tries + 1), run_chunks(arr[k:], tries + 1)])

    batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    futs = [[rpool.submit(read_one, v) for _, v in b] for b in batches[:3]]
    t_start, t_gpu, n_done = time.time(), 0.0, 0
    torch.cuda.reset_peak_memory_stats()
    for bi, b in enumerate(batches):
        arr = np.stack([f.result() for f in futs[bi]])  # (B, N, H0, W0) uint8
        futs[bi] = None
        if bi + 3 < len(batches):
            futs.append([rpool.submit(read_one, v) for _, v in batches[bi + 3]])
        torch.cuda.synchronize()
        tg = time.time()
        y = run_chunks(arr)
        torch.cuda.synchronize()
        t_gpu += time.time() - tg
        for (k, _), img in zip(b, y):
            wq.put((f"{STORE}/{cond}/seq{k[0]}/{k[1]}.png", img))
        n_done += len(b)
        if (bi + 1) % 50 == 0 or bi + 1 == len(batches):
            el = time.time() - t_start
            print(f"[{cond}] {n_done}/{len(todo)} frames, wall {el / n_done:.3f} s/frame, "
                  f"gpu {t_gpu / n_done:.3f} s/frame, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB",
                  flush=True)
    for _ in writers:
        wq.put(None)
    for t_ in writers:
        t_.join()
    wall = time.time() - t_start

    # window folders (symlinks to the unique straightened frames) + clip list for infer2
    rows = []
    for sq, wid, name, t in links:
        d = f"{WIN_DIR}/seq{sq}/{wid:04d}_{cond}"
        os.makedirs(d, exist_ok=True)
        lnk = f"{d}/{name}"
        if not os.path.lexists(lnk):
            os.symlink(f"{STORE}/{cond}/seq{sq}/{t}.png", lnk)
    for r in win.itertuples():
        d = f"{WIN_DIR}/seq{r.seq}/{r.window_id:04d}_{cond}"
        assert len(glob.glob(f"{d}/*.png")) == len(glob.glob(f"{WIN_DIR}/seq{r.seq}/{r.window_id:04d}_RS/*.png"))
        rows.append(dict(key=f"{r.key}_{cond}", dir=d))
    os.makedirs(f"{OUTRES}/clips", exist_ok=True)
    os.makedirs(f"{OUTRES}/runs", exist_ok=True)
    clip_csv = f"{OUTRES}/clips/{cond}{'_smoke' if a.windows else ''}.csv"
    pd.DataFrame(rows).to_csv(clip_csv, index=False)
    info = dict(cond=cond, method=a.method, weights=a.weights, gamma=a.gamma, target=net.target, ckpt=net.ckpt,
                ckpt_sha256=sha256(net.ckpt), nb=net.nb, scale=s, net_hw=[h, w], pad_hw=[ph, pw], amp=a.amp,
                batch=a.batch, n_windows=len(win), n_unique_frames=len(frames), n_run=len(todo),
                n_missing_neighbours=missing, wall_s=wall, wall_s_per_frame=wall / max(len(todo), 1),
                gpu_s_per_frame=t_gpu / max(len(todo), 1), peak_alloc_gb=torch.cuda.max_memory_allocated() / 2**30,
                torch=torch.__version__, clips=clip_csv)
    with open(f"{OUTRES}/runs/{cond}{'_smoke' if a.windows else ''}.json", "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info), flush=True)
    print("U2D_DONE", flush=True)


if __name__ == "__main__":
    main()
