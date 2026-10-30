"""Write the JSON configs of the EXP2 §8.6 LoRA runs to ~/rs3d/results/exp2/training/configs/.
  python make_configs.py pilot               confidence-loss pilots (pi3 6a seed 0, 400 steps, with / without;
                                             VGGT likewise as an extra check)
  python make_configs.py main --conf 0|1     the 12 runs {pi3, vggt} x {6a (rs), 6b (gs)} x seeds {0, 1, 2}
"""
import argparse
import json
import os

D = os.path.expanduser("~/rs3d/results/exp2/training/configs")
BASE = dict(
    train_set="~/rs3d/data/synth/core_trainval",
    rank=16, alpha=32, dropout=0.0,                    # LoRA on qkv/proj/fc1/fc2 of the multi-view transformer
    lr=1e-4, betas=[0.9, 0.999], weight_decay=0.05,   # AdamW; wd 0.05 as in both models' own training configs
    warmup=100, max_steps=3000, grad_clip=1.0,        # linear warm-up, cosine decay to 0 at max_steps
    frames=[8, 16],                                   # frames per sample (random subset, temporal order kept)
    conf_loss=False,
    val_every=250, patience=4, val_conds=["rs", "gs"], select_metric="ATE_mean",
    workers=5, vram_need_gb=30, vram_margin_gb=5,     # start only with >= 35 GB free (coordinator rule)
)


def write(c):
    os.makedirs(D, exist_ok=True)
    with open(f"{D}/{c['run']}.json", "w") as f:
        json.dump(c, f, indent=1)
    print(c["run"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["pilot", "main"])
    ap.add_argument("--conf", type=int, default=0)
    a = ap.parse_args()
    if a.what == "pilot":
        for m in ("pi3", "vggt"):
            for conf in (0, 1):
                write(dict(BASE, run=f"pilot_{m}_6a_s0_{'conf' if conf else 'noconf'}", model=m, train_cond="rs",
                           select_cond="rs", seed=0, conf_loss=bool(conf), stop_after=400, val_every=100,
                           patience=99))
    else:
        for seed in (0, 1, 2):
            for m in ("pi3", "vggt"):
                for arm, cond in (("6a", "rs"), ("6b", "gs")):
                    # each arm early-stops on its own domain's val split (6b never sees RS data, not even in
                    # model selection); both conditions are logged for every run
                    write(dict(BASE, run=f"{m}_{arm}_s{seed}", model=m, train_cond=cond, select_cond=cond, seed=seed,
                               conf_loss=bool(a.conf)))


if __name__ == "__main__":
    main()
