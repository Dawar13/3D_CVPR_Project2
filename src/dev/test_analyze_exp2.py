"""Smoke test for src/analyze_exp2.py on fake data (no VM needed): ADT slopes + exposure context, TUM ladder and Q2
with classical SfM as model-independent methods, D13 model filter. Planted effect: log ATE slope 0.2 per rad/s on
ADT and no exposure interaction, so adt_exposure_context should recover b_omega ~ 0.2 and b_omega_x_exposure ~ 0.

usage: python src/dev/test_analyze_exp2.py <empty output dir>
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd


def main():
    S = os.path.abspath(sys.argv[1])
    rng = np.random.default_rng(0)
    for d in ("gap/adt", "gap/tum", "baselines/classical/sfm_cpu"):
        os.makedirs(f"{S}/{d}", exist_ok=True)
    rows, win = [], []
    for s in range(8):
        for w in range(30):
            key, om, ex = f"seqA{s}_w{w:04d}", rng.uniform(0.1, 3), rng.uniform(8, 14)
            win.append(dict(key=key, exposure_ms_mean=ex))
            for m in ("pi3", "vggt", "da3"):
                rows.append(dict(seq=f"seqA{s}", window_id=w, key=key, model=m, cond="RS", omega_mean=om, v_mean=0.5,
                                 rs_shift_px=om * 1.28, bin="fast" if om > 1.5 else "slow",
                                 ATE=np.exp(-3 + 0.2 * om + rng.normal(0, 0.3))))
    pd.DataFrame(rows).to_csv(f"{S}/gap/adt/per_window.csv", index=False)
    pd.DataFrame(win).to_csv(f"{S}/gap/adt/windows.csv", index=False)
    rows = []
    for s in range(1, 5):
        for w in range(40):
            key, om = f"s{s}_w{w:04d}", rng.uniform(0.1, 3)
            for m in ("pi3", "vggt", "mapanything", "da3"):
                g = np.exp(-3 + rng.normal(0, 0.3))
                for c, f in (("GS", 1), ("RS", 1 + om), ("V3", 1.1), ("V4", 1.0), ("SE2P_" + m, 1 + om / 2)):
                    rows.append(dict(seq=s, window_id=w, key=key, model=m, cond=c, omega_mean=om, v_mean=0.5,
                                     rs_shift_px=1, bin="fast" if om > 1.5 else ("slow" if om < 0.5 else "medium"),
                                     ATE=g * f, AUC30=50))
    pd.DataFrame(rows).to_csv(f"{S}/gap/tum/per_window_all.csv", index=False)
    pd.DataFrame([dict(key=f"s{s}_w{w:04d}", cond=c, method=mt, ATE=rng.uniform(0.01, 0.2))
                  for s in range(1, 5) for w in range(40) for c in ("GS", "RS") for mt in ("colmap", "glomap")]
                 ).to_csv(f"{S}/baselines/classical/sfm_cpu/per_window_sfm.csv", index=False)

    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analyze_exp2.py")
    r = subprocess.run([sys.executable, src, "--root", S], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    ctx = pd.read_csv(f"{S}/gap/adt_exposure_context.csv")
    assert ctx.b_omega.between(0.1, 0.3).all() and (ctx.b_omega_x_exposure.abs() < 0.03).all(), ctx
    rec = pd.read_csv(f"{S}/baselines/tum_recovery.csv")
    assert "da3" not in set(rec.model), "D13: extra models must not enter Q2"
    assert {"colmap", "glomap"} <= set(rec.method), "classical SfM must enter every Q2 pair"
    sl = pd.read_csv(f"{S}/gap/slopes.csv")
    assert "adt" in set(sl.dataset)
    dec = json.load(open(f"{S}/decision.json"))
    print("OK", dec["Q1"]["answer"], dec["Q1_main_models_only"]["answer"], dec["Q2"]["answer"])


if __name__ == "__main__":
    main()
