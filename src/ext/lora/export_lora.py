"""Put the LoRA evaluations into the layout analyze_exp2.py reads (one row per window/clip, model = base model name,
cond = lora6a_s<seed> / lora6b_s<seed>):
  results/exp2/baselines/lora/tum_per_window_lora.csv      TUM RS windows (eval subset: every 4th kept window)
  results/exp2/baselines/lora/synth_core_test_lora.csv     synthetic core test, RS clips
  results/exp2/baselines/lora/noharm_lora.csv              GS input: LoRA runs and the base model (cond gs) on the
                                                           same GS windows / clips, column dataset = tum_cam0 /
                                                           synthetic_gs (the no-harm check, brief section 9.11)
usage: python export_lora.py
"""
import glob
import json
import os
import re

import pandas as pd

ROOT = os.path.expanduser("~/rs3d/results/exp2")
E = f"{ROOT}/training/eval"
OUT = f"{ROOT}/baselines/lora"


def main():
    os.makedirs(OUT, exist_ok=True)
    tum, syn, nh = [], [], []
    for d in sorted(glob.glob(f"{E}/*/")):
        tag = os.path.basename(d.rstrip("/"))
        m = re.fullmatch(r"(pi3|vggt)_(6a|6b)_s(\d)", tag)
        base = re.fullmatch(r"base_(pi3|vggt)", tag)
        if not (m or base):
            continue
        model = (m or base).group(1)
        cond = f"lora{m.group(2)}_s{m.group(3)}" if m else "gs"
        t, s = f"{d}/tum_windows.csv", f"{d}/synth_eval.csv"
        if os.path.exists(t):
            x = pd.read_csv(t).assign(model=model)
            if m:
                tum.append(x[x.cam == "RS"].assign(cond=cond))
            nh.append(x[x.cam == "GS"].assign(cond=cond, dataset="tum_cam0"))
        if os.path.exists(s):
            x = pd.read_csv(s).assign(model=model)
            if m:
                syn.append(x[x.cond == "rs"].assign(cond=cond))
            nh.append(x[x.cond == "gs"].assign(cond=cond, dataset="synthetic_gs"))
    for rows, name in ((tum, "tum_per_window_lora.csv"), (syn, "synth_core_test_lora.csv"), (nh, "noharm_lora.csv")):
        if rows:
            pd.concat(rows).to_csv(f"{OUT}/{name}", index=False)
            print(name, sum(len(r) for r in rows), "rows")
    with open(f"{OUT}/export_info.json", "w") as f:
        json.dump(dict(eval_dirs=sorted(os.path.basename(p.rstrip("/")) for p in glob.glob(f"{E}/*/"))), f, indent=1)


if __name__ == "__main__":
    main()
