# Extra EXP2 models: environments, pins, licences

Code: `~/rs3d/ext/models/` (nothing under `~/rs3d/src` was changed).

- `models_extra.py`: `EXTRA_RUNNERS = {da3, da3nested, streamvggt, vggt_omega}`, imported by `src/infer2.py` when `PYTHONPATH=~/rs3d/ext/models`.
- `download.py`: pinned HF snapshots (ungated repos only).
- `validate.py`: convention checks (c2w poses, camera-frame points, pose and depth scale, pixel map).
- `run_full.sh <model> <env>`: GS+RS main set, then the control set. A file `skip_<model>` makes it a no-op.
- `chain.sh [start_index]`: the sequential GPU queue (scope = DEVIATIONS D16). It runs one GPU job at a time. Each job takes our own single slot: `env RS3D_LOCK_DIR=~/rs3d/locks_models RS3D_GPU_SLOTS=1 bash src/gpu_slot.sh ...`, which also waits for >= 14 GB free. The jobs, in order:
  1. da3 on ADVIO RS (`gap/advio/clips_RS_thin.csv`, 1641 clips).
  2. da3 on ETH3D GS (`gap/eth3d/clips_GS_thin.csv`, 1062 clips, every 3rd window, made with `src/thin_clips.py`).
  3. da3 on synthetic core_test rs/gs, dense.
  4. streamvggt on TUM, every 2nd window (`clips/tum_gsrs_thin2.csv`). This keeps the 100 clips already done, 50 of which are in the thinned set.
  5. streamvggt on the TUM control set (all windows).
  6. streamvggt on synthetic core_test rs/gs, dense.

  StreamVGGT is skipped on ADVIO and ETH3D, and DA3-nested is skipped. It runs in tmux `rs3d:mdl-chain2` and logs to `logs/models_chain.log`; per-job logs are `logs/models_<model>_<set>.log`.
- `clips/tum_GS.csv`, `clips/tum_RS.csv` and their `_thin2` versions: `thin_clips.py <csv> 2`, run separately for GS and RS so both keep the same 564 windows. `clips/tum_gsrs_thin2.csv` is their union (1128 rows).
- `clips/synth_core_test.csv`: written by `src/synth_clips.py --set ~/rs3d/data/synth/core_test --conds rs:d,gs:d` (480 rows).
- `summarize.py`: per-bin medians from the evaluate2 CSV; control-set scoring via EXP0 `evaluate.compute_per_window("control")`.
- `smoke_vggt_omega.py`: runs the VGGT-Omega code path with random weights (no gated download).
- `clips/gsrs.csv` (2248 rows = 1124 windows x GS,RS), `clips/control.csv` (224 rows = 112 windows x GS,RS).
- `val/`: 3-window validation runs (`--dense`).

All envs were built with `~/rs3d/bin/uv` (`UV_CACHE_DIR=~/rs3d/.uv-cache`) on CPython 3.11.18 (uv-managed).
Each env uses torch 2.7.1+cu128 and torchvision 0.22.1+cu128 from `https://download.pytorch.org/whl/cu128`, the same build as the main `.venv`.
Each has its own env because DA3 and VGGT-Omega require `numpy<2`, while the main `.venv` has numpy 2.4.6.
The main `.venv` was not modified.

## Install commands (as run; log in `~/rs3d/logs/models_install.log`)

```bash
cd ~/rs3d; export UV_CACHE_DIR=~/rs3d/.uv-cache
for e in da3 streamvggt vggt_omega; do
  bin/uv venv envs/$e --python 3.11
  bin/uv pip install --python envs/$e/bin/python torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
done
# DA3: the pyproject deps without the GUI/server extras; headless OpenCV in place of opencv-python
bin/uv pip install --python envs/da3/bin/python "numpy<2" pandas==2.3.3 opencv-python-headless==4.10.0.84 pillow einops \
    huggingface_hub safetensors omegaconf addict trimesh imageio "moviepy==1.0.3" plyfile pillow_heif evo e3nn pycolmap \
    tqdm scipy matplotlib typer requests pyyaml
bin/uv pip install --python envs/da3/bin/python xformers==0.0.31.post1 torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
# StreamVGGT: requirements.txt pins torch 2.3.1 / numpy 1.26.1; we use torch 2.7.1 / numpy 1.26.4 and only the inference deps
bin/uv pip install --python envs/streamvggt/bin/python numpy==1.26.4 pandas==2.3.3 opencv-python-headless==4.10.0.84 pillow \
    einops huggingface_hub safetensors transformers matplotlib scipy        # transformers 5.21.1 (ModelOutput import only)
bin/uv pip install --python envs/streamvggt/bin/python py-spy              # profiling only
# VGGT-Omega: requirements.txt (torch>=2.6, numpy<2, Pillow, einops, safetensors, opencv) + infer2 needs
bin/uv pip install --python envs/vggt_omega/bin/python numpy==1.26.4 pandas==2.3.3 opencv-python-headless==4.10.0.84 pillow \
    einops huggingface_hub safetensors scipy pyyaml
```

The third-party packages are not pip-installed.
`models_extra.py` appends `third_party/da3/src`, `third_party/streamvggt/src` or `third_party/vggt_omega` to `sys.path`.
Key resolved versions:

| env | numpy | xformers | other |
|---|---|---|---|
| da3 | 1.26.4 | 0.0.31.post1 | omegaconf 2.4.0, scipy 1.17.1 |
| streamvggt | 1.26.4 | — | transformers 5.21.1 |
| vggt_omega | 1.26.4 | — | — |

## Pins

| runner | repo (clone) | commit | HF repo | HF revision | file(s) | code licence | weights licence |
|---|---|---|---|---|---|---|---|
| `da3` | github.com/ByteDance-Seed/Depth-Anything-3 → `third_party/da3` | `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` (2026-07-27) | `depth-anything/DA3-GIANT-1.1` | `72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19` | config.json, model.safetensors (5.1 GB) | Apache-2.0 | CC BY-NC 4.0 (HF card) |
| `da3nested` | same | same | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | `b2359bdf726fb44ef62acca04d629dcf158053e7` | config.json, model.safetensors (6.3 GB) | Apache-2.0 | CC BY-NC 4.0 (HF card) |
| `streamvggt` | github.com/wzzheng/StreamVGGT → `third_party/streamvggt` | `05682bcb05683c4c71522e9cb618fc8ca349063c` (2025-10-22) | `lch01/StreamVGGT` | `f9ba55b1955bc4f34337c51142158a9aa2862c7f` | checkpoints.pth (loaded, strict, as in demo_gradio.py); model.safetensors also in the snapshot | CC BY-NC-SA 4.0 (LICENSE.txt) | CC BY-NC 4.0 (HF card tag) |
| `vggt_omega` | github.com/facebookresearch/vggt-omega → `third_party/vggt_omega` | `48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6` (2026-09-22) | `facebook/VGGT-Omega` (GATED) | `1041e80fc0e911235d3426b0a3d9a81075111a53` (from the verified models table) | vggt_omega_1b_512.pt | FAIR Noncommercial Research License v1 | FAIR Noncommercial Research License (gated, manual approval) |

The HF snapshots are in `~/rs3d/.hf/hub` (`HF_HOME=~/rs3d/.hf`).
Runs use `HF_HUB_OFFLINE=1`.
VGGT-Omega was NOT downloaded because `~/rs3d/.hf/token` does not exist.
The runner refuses to start without that file.

## Runner settings (each model's own defaults)

| runner | preprocessing | input for 1280x1024 | precision | poses |
|---|---|---|---|---|
| `da3`, `da3nested` | `InputProcessor`, process_res=504, `upper_bound_resize` (cv2 INTER_AREA, then nearest /14 by resize), ImageNet norm | 504x406 | `DepthAnything3.forward` own autocast: bf16 (bf16 supported) | camera decoder (`use_ray_pose=False`), `ref_view_strategy="saddle_balanced"` (API defaults); extrinsics w2c → inverted |
| `streamvggt` | `load_and_preprocess_images` mode="crop" (width 518, BICUBIC), [0,1] | 518x420 | `torch.autocast(bf16)` around `model.inference` as in demo_gradio.py; heads fp32 inside | `model.inference(frames)`: frames streamed one at a time in the given order with KV cache; each frame's output is the one produced on arrival (sees frames 0..i), which is also the final output since past frames are never revised; pose enc → w2c → inverted |
| `vggt_omega` | `load_and_preprocess_images(image_resolution=512)`, default mode="balanced", patch 16, BICUBIC, [0,1] | 576x464 | model's own `autocast=True` (bf16 aggregator, fp32 heads) under `torch.inference_mode()` | `encoding_to_camera` w2c → inverted |

Dense outputs: `local_pts` = unprojection of the depth head with the predicted intrinsics (as `infer2.VGGTRunner` does), and `conf` = depth confidence.

## Run history

- Validation runs (`val/`, logs `logs/models_val_*.log` and `logs/models_validate_*.log`) used `src/infer2.py` at commit c4e1941. That version had an undithered pixel map.
- The full runs used commit 111a8c0, which added a dithered pixel map. The sx/sy values agree; ox/oy moved by up to 0.17 px.
- 14:01 UTC: DA3 and StreamVGGT were started in parallel.
- 14:37 UTC: the main agent asked for one GPU process at a time after an OOM. StreamVGGT was stopped with 100 clips saved and re-queued behind DA3; `infer2.py` resumes from `preds/streamvggt_gsrs.npz`.
- 15:37 UTC: DA3 TUM gsrs and control finished.
- 16:14–17:05 UTC: a first `chain.sh` waited in the shared `gpu_slot` slots and never started a job.
- 17:05 UTC: the chain was restarted on its own slot with the D16 scope.

## Timing on the shared GPU

At 100% utilisation from other jobs:
- DA3: ~2.2 s/clip.
- StreamVGGT `inference()`: 10–15 s/clip. In a profile of one clip, the aggregator took 8.5 s of 9.9 s: 16 sequential single-frame passes, which suffer most from GPU time-slicing.

StreamVGGT note: the batch `forward()` is also causal (masked), but it is NOT identical to `inference()`.
On 2 clips, per-frame rotations differed by up to 0.9 deg.
The reason is that the streaming camera head appends its KV cache at each of its 4 refinement iterations.
We use the official streaming `inference()`.

## Validation

Three windows: s1_w0001 (slowest, 0.11 rad/s), s3_w0003 (slow, 0.40 rad/s) and s5_w0063 (medium, 0.82 rad/s), each GS and RS.
Logs: `logs/models_validate_<model>.log`. VGGT-1B was run through the same checks as a reference (`logs/models_validate_vggt_ref.log`).

| check | da3 | da3nested | streamvggt | vggt (ref) |
|---|---|---|---|---|
| all-pairs rel. rotation error vs GT, mean (deg), 6 clips | 0.41–0.69 | 0.42–0.72 | 0.34–1.03 | 0.33–0.88 |
| same with the poses inverted (w2c hypothesis) | 5.4–77 | 5.4–77 | 5.0–76 | 5.3–77 |
| vs pi3, mean (deg) | 0.38–0.86 | 0.34–0.82 | 0.38–1.19 | 0.52–1.04 |
| Sim(3) ATE as saved < ATE inverted | 6/6 | 6/6 | 6/6 | 6/6 |
| local_pts median z (all > 0, z>0 frac 1.000) | 0.88–1.12 | 2.6–2.9 (metric-scaled) | 0.80–0.86 | 0.76–0.91 |
| frame 0→k depth reprojection error at s=1 (pairs with overlap) | 0.2–3.5 % | 0.2–4.3 % | 1.0–9.6 % | 0.1–2.1 % |
| best translation scale s (overlapping pairs) | 0.94–1.12 | 0.89–1.50 | 0.25–2.5 | 0.94–1.12 |
| pixel map sx vs W/w; sy vs H/h | 2.5395/2.5397; 2.5218/2.5222 | same | 2.4711/2.4710; 2.4385/2.4381 | — |

StreamVGGT's depth head is less consistent with its own camera translations than VGGT's. On the slowest window the best s is at 0.25–0.56. Its rotations and frame conventions are correct. This looks like a property of the model, not a wrapper error. ATE is Sim(3)-aligned and uses the poses only.
