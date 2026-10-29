# EXP2 environment

All values below were measured on the VM on 2026-10-29.

## Machine

- **Hardware:** gpu.net VM, 1× NVIDIA A100-SXM4-80GB (driver 550.144.03, CUDA 12.4 driver API), 24 vCPU, 113 GB RAM.
- **OS:** Ubuntu 22.04.5 LTS.
- **Disks:**
  - 92 GB root disk;
  - a 2.1 TB data disk (`/dev/vdc`), formatted ext4 and mounted at `/data`. `~/rs3d` → `/data/rs3d` (DEVIATIONS D3).
- **Shared GPU:** the other project's experiment runs on the same GPU (tmux `e2eth3d`, `~/3D-CVPR_Exp`). Our jobs never touch it.
  - Since 2026-10-29 ~14:40 UTC, each of our GPU steps goes through `src/gpu_slot.sh`: at most 3 concurrent jobs of ours, each started only when ≥ 14 GB is free.
  - Inference scripts also wait (they no longer exit) while < 12 GB is free.

## Main Python environment (`~/rs3d/.venv`)

- **Python:** 3.11.18, built with `uv`, which is installed at `~/rs3d/bin/uv`.
- **Steps:** `src/setup_env.sh` installs the following, then three manual fixes are applied:
  - map-anything (editable), vggt (editable, `--no-deps`), π³ (by path), and the common packages;
  - torch 2.7.1+cu128 from the PyTorch cu128 index;
  - opencv-python-headless 4.10 only (opencv-python 5, which vggt pulled in, was removed);
  - pandas 2.3.3, because pandas 3.1 was broken in this env (DEVIATIONS D8).
- **Full freeze:** `~/rs3d/logs/pip_freeze_main.txt` on the VM.

| Package | Version |
|---|---|
| torch / torchvision | 2.7.1+cu128 / 0.22.1+cu128 |
| numpy | 2.4.6 |
| pandas | 2.3.3 |
| scipy | 1.17.1 |
| opencv-python-headless | 4.10.0.84 |
| open3d | 0.20.0 (Embree ray casting for `rsgen`) |
| peft | 0.21.2 (LoRA controls) |
| timm / einops | 1.0.30 / 0.8.2 |
| huggingface-hub / safetensors | 2.3.0 / 0.8.0 |
| uniception | 0.1.7 (MapAnything) |
| pillow / matplotlib / pyyaml | 13.0.0 / 3.11.2 / 6.0.3 |

## Pinned models and code

| Model | HF checkpoint @ revision | Code @ commit | Licence (weights) |
|---|---|---|---|
| π³ | `yyfz233/Pi3` @ `ae722e7039287d0c8fde9f11f197f804f44b510c` | github.com/yyfz/Pi3 @ `9fa3ddb3f8d53041f8b2738df404f62223bbaa7b` | CC BY-NC 4.0 per README (the HF tag says bsd-2-clause: conflict, treated as non-commercial) |
| VGGT-1B | `facebook/VGGT-1B` @ `860abec7937da0a4c03c41d3c269c366e82abdf9` | github.com/facebookresearch/vggt @ `a288dd0f14786c93483e45524328726ab7b1b4ce` (last commit on or before 2026-10-27; D2) | CC BY-NC 4.0 |
| MapAnything (Apache) | `facebook/map-anything-apache` @ `00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a` | github.com/facebookresearch/map-anything @ `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9` | Apache-2.0 |

- These are the same commits as EXP0. The reproduction check matched EXP0 bit for bit: 4,496 window × model × camera ATE values.
- **Settings for all three models:**
  - images only, no priors;
  - bf16 autocast;
  - each model's own default preprocessing;
  - input sizes for 1280×1024 TUM frames (measured by `infer2.py`'s pixel map):

| Model | Input | Preprocessing |
|---|---|---|
| π³ | 560×448 | plain resize |
| VGGT | 518×420 | plain resize |
| MapAnything | 518×392 | resize, then ~28 rows cropped top and bottom |

## Extra environments (helper agents)

Each has its own `ENV.md` with install commands, commits and weight hashes. They are copied into the repo at the end of EXP2 (`src/ext/`).

| Env | Used for | Notes |
|---|---|---|
| `~/rs3d/envs/da3`, `streamvggt`, `vggt_omega` | DA3, StreamVGGT, VGGT-Ω runners (`~/rs3d/ext/models/`) | VGGT-Ω weights are gated: waiting on the user's token (HUMAN_TODO 4) |
| `~/rs3d/envs/u2d` | 2D rolling-shutter unrollers (`~/rs3d/ext/unroll2d/`) | |
| `~/rs3d/envs/classical`, `classical_cuda`, `rsmp` | COLMAP/GLOMAP and Barath's RS relative pose (`~/rs3d/ext/classical/`) | |
| main env + peft | LoRA controls (`~/rs3d/ext/lora/`) | |

## Reproducing a run

Every EXP2 driver is a resumable bash script in `src/`, launched in tmux on the VM:
- `run_exp2_stage1.sh`: TUM prep, reproduction, GS-GS control.
- `run_exp2_tum2.sh`: TUM oracle ladder, sensitivity, self-estimate.
- `run_synth_gen.sh`: all synthetic sets.
- `run_synth_core.sh`: synthetic gap, oracle, self-estimate.
- `run_real_gap.sh`: ADVIO and ETH3D.
- `run_stress.sh`: stress tests.
- `analyze_exp2.py`: tables and the decision.

Configs are in `configs/synth/`. Seeds are per clip (`rsgen.generate.clip_seed`). Logs are in `~/rs3d/logs/`.
