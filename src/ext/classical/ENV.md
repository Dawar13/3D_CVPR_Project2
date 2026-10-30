# Classical baselines (EXP2): environment record

All code: `~/rs3d/ext/classical/`. Results: `~/rs3d/results/exp2/baselines/classical/`. Logs: `~/rs3d/logs/classical_*.log`.

## System packages (apt, 2026-10-29; needed only for the Barath build)
```
sudo apt-get update
sudo apt-get install -y cmake libeigen3-dev libboost-all-dev libopencv-dev libopencv-contrib-dev
```
Installed: cmake 3.22.1-1ubuntu1.22.04.2, libeigen3-dev 3.4.0-2ubuntu2, libboost-all-dev 1.74.0.3ubuntu7,
libopencv-dev / libopencv-contrib-dev 4.5.4+dfsg-9ubuntu4. Already present: build-essential (gcc 11.4.0), git, python3-dev.

## Python envs (uv 0.13.6, `UV_CACHE_DIR=~/rs3d/.uv-cache`, CPython 3.11.18)

### ~/rs3d/envs/classical (CPU pycolmap; only used for the first, abandoned CPU-SIFT attempt)
```
~/rs3d/bin/uv venv --python 3.11 ~/rs3d/envs/classical
~/rs3d/bin/uv pip install pycolmap==4.2.1 numpy==2.4.6 scipy pandas==2.3.3 pyyaml opencv-python-headless==4.10.0.84
```

### ~/rs3d/envs/classical_cuda (COLMAP + GLOMAP runs)
```
~/rs3d/bin/uv venv --python 3.11 ~/rs3d/envs/classical_cuda
~/rs3d/bin/uv pip install --python ~/rs3d/envs/classical_cuda/bin/python pycolmap-cuda12==4.2.1 numpy==2.4.6 scipy pandas==2.3.3 pyyaml opencv-python-headless==4.10.0.84
~/rs3d/bin/uv pip install --python ~/rs3d/envs/classical_cuda/bin/python tabulate
```
pycolmap-cuda12 4.2.1 = COLMAP 4.2.1, has_cuda=True (pulls cuda-toolkit 12.9.2.0 / nvidia-cuda-runtime-cu12 12.9.79 wheels).
"GLOMAP" here = the global SfM pipeline built into COLMAP 4.2.1 (`pycolmap.global_mapping`, docstring
"Recover 3D points and camera poses using global SfM (GLOMAP)"). The standalone GLOMAP repo was not built.

### ~/rs3d/envs/rsmp (Barath RS relative pose)
```
~/rs3d/bin/uv venv --python 3.11 ~/rs3d/envs/rsmp
~/rs3d/bin/uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
~/rs3d/bin/uv pip install kornia kornia_moons h5py pandas matplotlib tqdm romatch pybind11 numpy scipy opencv-python-headless pyyaml setuptools wheel cmake
~/rs3d/bin/uv pip install --python ~/rs3d/envs/rsmp/bin/python numpy==2.2.6   # repo tests/evaluation.py calls np.trapz (gone in numpy 2.4)
~/rs3d/bin/uv pip install --python ~/rs3d/envs/rsmp/bin/python tabulate
```
Resolved: torch 2.7.1+cu128, torchvision 0.22.1+cu128, kornia 0.9.0, kornia-moons 0.3.0, romatch 0.1.2, timm 1.0.30,
h5py 3.16.0, pybind11 3.1.0, numpy 2.2.6, pandas 3.1.0, opencv-python(-headless) 5.0.0.93 (pulled in by romatch).

## Clones (third_party)
- `~/rs3d/third_party/rolling_shutter_made_practical`: https://github.com/danini/rolling_shutter_made_practical
  commit **d8340562be6a84a879e13ae61548f96cbdf807dd** (2026-07-14), MIT.
  - submodule `external/superansac`: https://github.com/danini/superansac commit
    **a3015a3dbebe4695c188eed42609a744d0ee4768** (pinned by the repo), with `rs_overlay/apply_overlay.sh` applied
    (copies the RS files into the submodule, as the repo intends).
  - Build: `ext/classical/build_rsmp_py.sh` follows the repo's `setup.sh --with-python` steps, but uses `-j3` instead of
    `-j$(nproc)`. The Python module goes to `build_py/lib` and is exposed to the rsmp env through
    `pysuperansac_local.pth`. (The first script, `build_rsmp.sh`, built the C++ benchmark first; it was stopped so the
    Python module could be built first.)
  - Dataset layout for the repo loader is a symlink tree, with no re-download:
    `ext/classical/tum_rs_link/euroc/dataset-seqN -> ~/rs3d/data/tumrs/dataset-seqN/dataset-seqN`,
    `ext/classical/tum_rs_link/calibration/<calib yaml>`.
- Extra install (romatch 0.1.2 `roma_outdoor()` defaults to `use_custom_corr=True` on Linux and needs the `local_corr`
  module; without it every RoMa call failed with "No module named 'local_corr'"):
  ```
  ~/rs3d/bin/uv pip install --python ~/rs3d/envs/rsmp/bin/python fused-local-corr==0.3.27 numpy==2.2.6 torch==2.7.1 \
      --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match --only-binary fused-local-corr
  ```
- romatch API shim (`ext/classical/roma_compat.py`): romatch 0.1.x `match()` only returns batched output
  (warp (1,H,2W,4)); the repo's `rs_ac_extractor.extract_affine_correspondences_roma` indexes it as the old unbatched
  (H,2W,4) and raised IndexError on every pair. The shim drops the batch dim; no third-party file is edited.
  `ext/classical/run_repro.py` runs the repo tester with the shim and the run_real_data.py defaults (+ `--workers 4`).
- RoMa / DINOv2 weights: `TORCH_HOME=~/rs3d/cache/torch` (roma_outdoor.pth, dinov2_vitl14_pretrain.pth; public URLs).

## GPU discipline (from the main agent, 2026-10-29 14:35)
All GPU work runs in ONE process at a time through `bash ~/rs3d/src/gpu_slot.sh bash ~/rs3d/ext/classical/gpu_chain.sh`
(repro -> COLMAP SIFT feature stage -> Barath windows). COLMAP/GLOMAP mapping is CPU-only (`run_sfm2.py map --procs 7`).

## SfM run history
1. `run_sfm.py`, CPU SIFT, 8 then 7 procs: stopped after 30 windows (too slow under the shared CPU load). Raw records
   and models kept in `results/.../classical/_cpusift_partial/`. Not used.
2. `run_sfm.py --gpu 1`, 7 procs each with its own SiftGPU context: after ~225 windows one process hit
   "an illegal memory access" (GPU had ~5 GB free) and its CUDA context stayed broken, so every later job it took
   produced 0 keypoints. Unfiltered records: `jobs_raw_run2_gpu_with_cuda_errors.jsonl`. `filter_jobs.py` kept the
   206 records with valid features and 16/16 for both mappers; the other 19 valid-feature windows were rerun (same
   result) and their old models moved to `_gpuerr_suspect/`. Raw log: `logs/classical_sfm.log.gz`.
3. `run_sfm2.py` (two-stage): one GPU feature process (keypoint check; exit + restart on any 0-keypoint image) and
   a CPU pool for verification + mapping. Same settings; records append to the same `jobs.jsonl`.
4. FINAL: `run_sfm3.py --procs 7 --out results/.../classical/sfm_cpu` (16:50 onward): CPU-only SIFT, computed once per
   frame (`sfm_cpu/_work/framefeat/`), 16 frame databases merged per window, CPU matching + verification + both
   mappers. No GPU. Windows in a fixed random order (seed 0) with GS and RS back to back, so a partial run is a random
   window subset with paired conditions. Records: `sfm_cpu/jobs_cpu.jsonl`; models `sfm_cpu/{colmap,glomap}/recon/`.
   Check: on 126 (window, cond) jobs done by both the GPU-SIFT run 2 and this CPU run, median per-window ATE ratio
   CPU/GPU = 0.97-1.005 and identical registration rates (`sift_gpu_vs_cpu_overlap.csv`).
   (`run_sfm2.py` two-stage GPU version was written but never ran: no GPU slot for 2 h.)

## Barath runs
- Repro: `gpu_chain.sh` -> `run_repro.py --sequences 1` (repo defaults: RoMa, strides 10 20, 100 pairs/stride,
  MSAC, PROSAC, 1 px, 1000 iters) -> `barath/repro/results_rs_tum_seq1.csv`; `check_repro.py` re-solves 7AC on the
  cached ACs (CPU) to inspect the RS motion estimates and the AC affine parts.
- Windows: `barath_windows.py --conds RS --gaps 1 --workers 2` (same settings; our prepped 8-bit pinhole frames;
  per-pair AC cache `barath/acs/RS/seqN/{ta}_{tb}.npy`; results `barath/pairs_RS.csv`). Windows in random order
  interleaved across speed bins. First attempt's HDF5 AC caches were corrupted by the kill and are kept in
  `barath/acs/_corrupt_h5_after_kill/` (the pairs they produced are in pairs_RS.csv and were not recomputed).
- GPU: `env RS3D_LOCK_DIR=$HOME/rs3d/locks_classical RS3D_GPU_SLOTS=1 bash ~/rs3d/src/gpu_slot.sh bash gpu_chain.sh`.
- Eval: `eval_barath.py --cond RS` -> `barath/per_pair_RS.csv`, `summary_pairs_RS.{csv,md}`,
  `omega_vs_gyro_RS.csv`, `summary_omega_RS.{csv,md}`, `barath_omega.npz`.
