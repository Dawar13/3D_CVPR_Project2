"""Union of two infer2.py outputs for the same clip list run from both ends (forward + reversed shard).
preds/<m>_<name>.npz (forward) + preds/<m>_<name>_b.npz (reversed) -> <dst>; shard files are kept.
Overlapping keys (both shards ran the clip) keep the forward result; their max abs pose difference is printed.
usage: python merge_shards.py <preds_dir> <model> <name> <clips.csv> <dst.npz>"""
import sys
import numpy as np
import pandas as pd
d, m, name, clips, dst = sys.argv[1:6]
a = dict(np.load(f"{d}/{m}_{name}.npz"))
b = dict(np.load(f"{d}/{m}_{name}_b.npz"))
keys = list(pd.read_csv(clips).key)
ov = sorted(set(a) & set(b))
if ov:
    print("overlap", len(ov), "max |pose diff|", max(float(np.abs(a[k] - b[k]).max()) for k in ov))
out = {k: (a[k] if k in a else b[k]) for k in keys if k in a or k in b}
missing = [k for k in keys if k not in out]
print("merged", len(out), "of", len(keys), "missing", len(missing))
if not missing:
    np.savez(dst, **out)
    print("wrote", dst)
