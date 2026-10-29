"""Writes every EXP2 synthetic-set config (YAML) from one base, so paired sets differ only in the knob under test.

Scene split (fixed 2026-10-29, before any synthetic data was generated; see DEVIATIONS.md D4):
  train  frl_apartment_0..5 (one physical apartment in 6 arrangements, so kept together), apartment_0, apartment_1,
         office_0, office_1, room_1, hotel_0
  val    room_2, office_2
  test   room_0, office_3, office_4, apartment_2
Stress sets use the test scenes and clip ids 0..14 of each, with the core test seed, so their trajectories are the
core test clips 0..14 exactly (the motion stream does not depend on the knob). Sets that change the motion itself
(view count, frame spacing) are paired only within the set.

usage: python configs/synth/make_configs.py   (writes configs/synth/*.yaml next to this file)
"""
import copy
import os

import yaml

SPLITS = dict(
    train=["frl_apartment_0", "frl_apartment_1", "frl_apartment_2", "frl_apartment_3", "frl_apartment_4",
           "frl_apartment_5", "apartment_0", "apartment_1", "office_0", "office_1", "room_1", "hotel_0"],
    val=["room_2", "office_2"],
    test=["room_0", "office_3", "office_4", "apartment_2"],
)

BASE = dict(
    scene_dir="~/rs3d/data/replica", scene_files="{scene}/mesh.ply", out_dir="~/rs3d/data/synth", up=2,
    camera=dict(W=640, H=512, hfov=90),  # 5:4 and ~90 deg like the undistorted TUM frames
    fps=20, stride=3, n_frames=16, seed=2027, render=dict(ssaa=2), threads_per_worker=2,
    motion=dict(kinds=["handheld", "drone", "rotation", "translation", "mixed"],
                kind_probs=[0.4, 0.15, 0.15, 0.15, 0.15],
                omega_dps=dict(lo=0, hi=220), speed_mps=dict(lo=0, hi=2.0)),
    shutter=dict(readout_ms=30, direction="tb", exposure_ms=0, n_exp=1),
    splits=SPLITS,
)


def cfg(name, splits, clips, **over):
    c = copy.deepcopy(BASE)
    c["name"] = name
    c["splits"] = {k: SPLITS[k] for k in splits}
    c["clips_per_scene"] = clips
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(c.get(k), dict):
            c[k].update(v)
        else:
            c[k] = v
    return c


SETS = [
    # training data for LoRA controls (6a RS, 6b its GS twins) and unroller fine-tuning: broad shutter knobs
    cfg("core_trainval", ["train", "val"], dict(train=100, val=30), neighbors=True, neighbors_max_clip=40,
        shutter=dict(readout_ms=[0, 5, 10, 20, 30, 40], direction=["tb", "tb", "tb", "bt", "lr"])),
    # main test set: TUM-like 30 ms top->bottom, no blur; gap (2D), oracle ladder (2E), baselines (2F)
    cfg("core_test", ["test"], dict(test=60), neighbors=True),
]
for r in (0, 5, 10, 20, 40):  # 2G-1 readout sweep (30 ms = core_test clips 0..14)
    SETS.append(cfg(f"st_readout{r}", ["test"], dict(test=15), shutter=dict(readout_ms=r)))
for d in ("bt", "lr"):  # 2G-2 scan direction
    SETS.append(cfg(f"st_dir_{d}", ["test"], dict(test=15), shutter=dict(direction=d)))
for ro, ex in ((0, 16), (30, 8), (30, 16)):  # 2G-7 blur alone and with RS (ssaa 1 x 8 time samples)
    SETS.append(cfg(f"st_blur_r{ro}_e{ex}", ["test"], dict(test=15), render=dict(ssaa=1),
                    shutter=dict(readout_ms=ro, exposure_ms=ex, n_exp=8)))
for ro in (0, 30):  # 2G-6 moving objects, with and without RS
    SETS.append(cfg(f"st_objects_r{ro}", ["test"], dict(test=15), shutter=dict(readout_ms=ro),
                    objects=dict(n=2, speed_mps=dict(lo=0.5, hi=2.0))))
SETS.append(cfg("st_lowtex", ["test"], dict(test=15), scene_mod=dict(contrast=0.25)))  # 2G-10 low texture
SETS.append(cfg("st_views64", ["test"], dict(test=15), n_frames=64,  # 2G-4 views: first N of 64 frames
                motion=dict(omega_dps=dict(lo=20, hi=160), speed_mps=dict(lo=0, hi=1.0))))
for st in (1, 6, 12):  # 2G-5 frame spacing (stride 3 = core_test); unordered sets use --order shuffled
    SETS.append(cfg(f"st_stride{st}", ["test"], dict(test=15), stride=st))

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    for c in SETS:
        with open(os.path.join(here, f"{c['name']}.yaml"), "w") as f:
            yaml.safe_dump(c, f, sort_keys=False)
    print(len(SETS), "configs:", " ".join(c["name"] for c in SETS))
