"""Clip lists (key, dir, name) for src/infer2.py from the TUM windows in results/exp2/gap/tum.

  python make_clips.py --cams GS,RS --name gsrs --out X.csv              all kept main windows, both cameras
  python make_clips.py --cams GS,RS --name control --set control --out X  the GS-GS control set
  python make_clips.py --cams V3,V4 --out X.csv                          oracle conditions (name = condition)
  python make_clips.py --cams GS,RS --name gsrs --thin 2 --out X.csv     every 2nd kept window per sequence (same
                                                                         windows for every camera)
Window folders are data/windows/seq<seq>/<wid:04d>_<cam> (prep.py and tum_conditions.py).
"""
import argparse
import os

import pandas as pd

ROOT = os.path.expanduser("~/rs3d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--set", default="main", choices=["main", "control"])
    ap.add_argument("--thin", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sfx = "_control" if a.set == "control" else ""
    w = pd.read_csv(f"{ROOT}/results/exp2/gap/tum/windows{sfx}.csv")
    w = w[w.keep].sort_values(["seq", "window_id"])
    if a.thin > 1:
        w = w.groupby("seq", group_keys=False).apply(lambda g: g.iloc[::a.thin])
    rows = []
    for cam in a.cams.split(","):
        for r in w.itertuples():
            d = f"{ROOT}/data/windows/seq{r.seq}/{r.window_id:04d}_{cam}"
            rows.append(dict(key=f"{r.key}_{cam}", dir=d, name=a.name or cam))
    missing = [r["dir"] for r in rows if not os.path.isdir(r["dir"])]
    assert not missing, f"{len(missing)} missing window folders, e.g. {missing[:2]}"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"{len(rows)} rows ({len(w)} windows) -> {a.out}")


if __name__ == "__main__":
    main()
