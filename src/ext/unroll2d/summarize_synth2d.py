"""2D image quality of every U2D condition on a synthetic set: PSNR of RS and of the straightened frame vs the GS
twin (centre-row instant), per speed bin of the clip's omega_mean (same rad/s cut points as TUM).
Reads results/exp2/baselines/unroll2d/synth/<set>_<split>_<name>.csv (written by run_synth.py), writes
results/exp2/baselines/unroll2d/synth2d_<set>_<split>.{csv,md}.
usage: python summarize_synth2d.py [--set core_test] [--split test]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

D = os.path.expanduser("~/rs3d/results/exp2/baselines/unroll2d")
BINS = ["slow", "medium", "fast", "all"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="core_test")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(f"{D}/synth/{a.set}_{a.split}_*.csv")):
        name = os.path.basename(p)[len(f"{a.set}_{a.split}_"):-4]
        df = pd.read_csv(p)
        df["bin"] = pd.cut(df.omega_mean, [0, 0.5, 1.5, np.inf], labels=BINS[:3], right=False).astype(str)
        for b in BINS:
            d = df if b == "all" else df[df.bin == b]
            if len(d) == 0:
                continue
            rows.append(dict(name=name, bin=b, n_frames=len(d), n_clips=d.key.nunique(), psnr_rs=d.psnr_rs.mean(),
                             psnr_out=d.psnr_out.mean(), gain_db=(d.psnr_out - d.psnr_rs).mean(),
                             frac_frames_improved=float((d.psnr_out > d.psnr_rs).mean())))
    s = pd.DataFrame(rows)
    s.to_csv(f"{D}/synth2d_{a.set}_{a.split}.csv", index=False)
    with open(f"{D}/synth2d_{a.set}_{a.split}.md", "w") as f:
        f.write(f"PSNR (dB) vs the GS twin on {a.set}/{a.split}, mean over frames; RS -> straightened (gain).\n\n")
        f.write("| condition | " + " | ".join(BINS) + " | frac frames improved (all) |\n|---|" + "---|" * 5 + "\n")
        for n in s.name.unique():
            sc = s[s.name == n].set_index("bin")
            cells = [f"{sc.loc[b, 'psnr_rs']:.2f} -> {sc.loc[b, 'psnr_out']:.2f} ({sc.loc[b, 'gain_db']:+.2f}; "
                     f"{int(sc.loc[b, 'n_clips'])} clips)" if b in sc.index else "-" for b in BINS]
            f.write(f"| {n} | " + " | ".join(cells) + f" | {sc.loc['all', 'frac_frames_improved']:.2f} |\n")
    print(open(f"{D}/synth2d_{a.set}_{a.split}.md").read())


if __name__ == "__main__":
    main()
