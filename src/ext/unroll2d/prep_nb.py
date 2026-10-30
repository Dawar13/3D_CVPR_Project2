"""Prep the adjacent raw cam1 (RS) frames that the 2D RS-correction nets need, with exactly the src/prep.py recipe.

For every kept window frame (cam1 index i, from the RS window folder names) we prep cam1 frames i-R..i+R (R = 2 by
default, covering 2- and 3-frame nets with margin) into ~/rs3d/data/prep_nb/seq{s}/cam1/{t}.png using
prep.shared_new_K / prep._init_worker / prep._prep_one (imported, not copied or edited). Also:
  - asserts K_new equals results/exp2/gap/tum/prep_info.json
  - reports the inter-frame interval statistics of the needed pairs (dropped frames would break "adjacent")
  - verifies the recipe: every prep_nb frame that also exists in data/prep is compared pixel-for-pixel
Writes ~/rs3d/data/prep_nb/prep_nb_info.json.  Run with the main env: ~/rs3d/.venv/bin/python prep_nb.py
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

SRC = os.path.expanduser("~/rs3d/src")
sys.path.insert(0, SRC)
import prep  # noqa: E402  (src/prep.py, unmodified)
from common import WIN_DIR, load_calib, load_frames, seq_dir  # noqa: E402

RES = os.path.expanduser("~/rs3d/results/exp2/gap/tum")
OUT = os.path.expanduser("~/rs3d/data/prep_nb")
PREP = os.path.expanduser("~/rs3d/data/prep")
R = int(os.environ.get("U2D_NB_RADIUS", 2))


def _cmp(args):
    a, b = args
    x, y = cv2.imread(a, cv2.IMREAD_UNCHANGED), cv2.imread(b, cv2.IMREAD_UNCHANGED)
    return int(np.abs(x.astype(np.int16) - y.astype(np.int16)).max())


def main():
    info = json.load(open(f"{RES}/prep_info.json"))
    K_new, _ = prep.shared_new_K(load_calib())
    assert np.allclose(K_new, np.array(info["K_new"]), atol=1e-9), (K_new, info["K_new"])
    win = pd.read_csv(f"{RES}/windows.csv")
    win = win[win.keep]
    jobs, stats, need_all = [], {}, {}
    for s in sorted(win.seq.unique()):
        t1 = load_frames(s, "cam1")
        pos = {int(t): i for i, t in enumerate(t1)}
        need = set()
        centre = []
        for r in win[win.seq == s].itertuples():
            for p in sorted(glob.glob(f"{WIN_DIR}/seq{s}/{r.window_id:04d}_RS/*.png")):
                t = int(os.path.basename(p)[3:-4])
                i = pos[t]
                centre.append(i)
                for o in range(-R, R + 1):
                    if 0 <= i + o < len(t1):
                        need.add(i + o)
        centre = np.array(sorted(set(centre)))
        dt_prev = np.array([(t1[i] - t1[i - 1]) * 1e-6 for i in centre if i - 1 >= 0])
        dt_next = np.array([(t1[i + 1] - t1[i]) * 1e-6 for i in centre if i + 1 < len(t1)])
        stats[int(s)] = dict(n_centre=int(len(centre)), n_needed=int(len(need)),
                             missing_prev=int((centre - 1 < 0).sum()), missing_next=int((centre + 1 >= len(t1)).sum()),
                             dt_prev_ms=[float(dt_prev.min()), float(np.median(dt_prev)), float(dt_prev.max())],
                             dt_next_ms=[float(dt_next.min()), float(np.median(dt_next)), float(dt_next.max())],
                             n_gap_gt_75ms=int((dt_prev > 75).sum() + (dt_next > 75).sum()))
        os.makedirs(f"{OUT}/seq{s}/cam1", exist_ok=True)
        for i in sorted(need):
            t = int(t1[i])
            jobs.append((f"{seq_dir(s)}/cam1/data/{t}.png", f"{OUT}/seq{s}/cam1/{t}.png", "cam1"))
        need_all[int(s)] = len(need)
        print(f"seq{s}: {stats[int(s)]}", flush=True)
    print(f"{len(jobs)} frames to prep into {OUT}", flush=True)
    with ProcessPoolExecutor(12, initializer=prep._init_worker, initargs=(K_new,)) as ex:
        list(ex.map(prep._prep_one, jobs, chunksize=16))
    pairs = [(d, d.replace(OUT, PREP)) for _, d, _ in jobs if os.path.exists(d.replace(OUT, PREP))]
    with ProcessPoolExecutor(12) as ex:
        diffs = np.array(list(ex.map(_cmp, pairs, chunksize=32)))
    out = dict(radius=R, K_new=K_new.tolist(), n_prepped=len(jobs), per_seq=stats,
               recipe_check=dict(n_compared=int(len(pairs)), max_abs_diff=int(diffs.max()) if len(diffs) else None,
                                 n_identical=int((diffs == 0).sum())))
    json.dump(out, open(f"{OUT}/prep_nb_info.json", "w"), indent=1)
    print(json.dumps(out["recipe_check"]), flush=True)
    print("PREP_NB_DONE", flush=True)


if __name__ == "__main__":
    main()
