"""Write an infer2 clips CSV (key, dir) for conditions of a generated synthetic set: key = <clip key>_<cond>.

usage: python synth_clips.py --set ~/rs3d/data/synth/core_test --conds rs,gs --out clips.csv [--limit N]
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from dense_utils import list_clips


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
    rows = [dict(key=f"{r.key}_{c}", dir=f"{r.dir}/{c}") for c in a.conds.split(",") for r in clips.itertuples()]
    missing = [r["dir"] for r in rows if not os.path.isdir(r["dir"])]
    if missing:
        sys.exit(f"{len(missing)} condition folders missing, e.g. {missing[0]}")
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"{len(rows)} rows -> {a.out} ({n_fail} clips failed at generation)")


if __name__ == "__main__":
    main()
