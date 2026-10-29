"""Write an infer2 clips CSV (key, dir, name, dense, order) for conditions of a generated synthetic set.

Each --conds item is  name[=folder][:flags]  with flags d (dense output) and s (shuffled frame order); the folder
defaults to the name. Keys are <clip key>_<name>; predictions go to <model>_<name>.npz.
e.g.  --conds rs:d,gs:d,v3:d,v4:d,v3_r0.5,rsshuf=rs:s

usage: python synth_clips.py --set ~/rs3d/data/synth/core_test --conds rs:d,gs:d --out clips.csv [--limit N]
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from dense_utils import list_clips


def parse_item(item):
    spec, _, flags = item.partition(":")
    name, _, folder = spec.partition("=")
    return name, folder or name, "d" in flags, "shuffled" if "s" in flags else "sorted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--conds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    clips, n_fail = list_clips(os.path.expanduser(a.set), a.split)
    if a.limit:
        clips = clips.head(a.limit)
    rows = []
    for item in a.conds.split(","):
        name, folder, dense, order = parse_item(item)
        rows += [dict(key=f"{r.key}_{name}", dir=f"{r.dir}/{folder}", name=name, dense=int(dense), order=order)
                 for r in clips.itertuples()]
    missing = [r["dir"] for r in rows if not os.path.isdir(r["dir"])]
    if missing:
        sys.exit(f"{len(missing)} condition folders missing, e.g. {missing[0]}")
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"{len(rows)} rows -> {a.out} ({n_fail} clips failed at generation)")


if __name__ == "__main__":
    main()
