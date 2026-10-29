"""Thin a clips CSV by window index (DEVIATIONS D14): keep every k-th window of each sequence, in window order.

usage: python thin_clips.py <clips.csv> <k> <out.csv>
"""
import sys

import pandas as pd

src, k, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
df = pd.read_csv(src)
seq = df.key.str.rsplit("_w", n=1).str[0]
wid = df.key.str.extract(r"_w(\d+)_")[0].astype(int)
df = df.assign(_s=seq, _w=wid).sort_values(["_s", "_w"])
df = df[df.groupby("_s").cumcount() % k == 0].drop(columns=["_s", "_w"])
df.to_csv(out, index=False)
print(f"{src}: kept {len(df)} rows (every {k}th window per sequence) -> {out}")
