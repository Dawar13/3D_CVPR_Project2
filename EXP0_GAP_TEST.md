# EXP0 — Gap Test: do feed-forward 3D models break on rolling-shutter input?

Read `PROJECT_CONTEXT.md` first.
Budget: **~1–2 h wall clock**. Inference only. No training.

## 0. Question this experiment answers

Same scenes, same instant, two cameras: one global shutter (GS), one rolling shutter (RS).

Does MapAnything / VGGT / π³ pose error on RS exceed GS, and does the gap grow with camera rotation speed?

## 1. Shared-GPU safety rules (IMPORTANT)

Another experiment (MapAnything corrupted-priors baseline) is running on this machine.

- Remote: `ssh gpunet@216.81.248.38`.
- The startup script may still be running. Check `journalctl -u init-script -f` and wait until it finishes.
- Work ONLY inside `~/rs3d/`.
- Use a separate venv: `python -m venv ~/rs3d/.venv`.
- Do not modify, kill, or pip-install into the other experiment's environment or folders.
  - You MAY read its MapAnything checkout to copy install steps.
- Before each run: `nvidia-smi`. Note the free VRAM.
  - Use bf16/fp16.
  - Process one clip at a time.
  - If free VRAM < 12 GB, wait or run models sequentially. Never preempt the other job.
- Run inside `tmux new -s rs3d`. Log everything to `~/rs3d/logs/`.

## 2. Setup (~20–30 min)

```
~/rs3d/
  data/tumrs/        # dataset
  src/               # scripts
  results/exp0/      # outputs
  logs/
```

1. **Dataset: TUM Rolling-Shutter Visual-Inertial Odometry dataset** (Schubert et al., IROS 2019).
   - Page: https://cvg.cit.tum.de/data/datasets/rolling-shutter-dataset
   - 10 sequences. Download the **calibrated** versions. Start with all 10 if bandwidth allows, else seq 1–4 first.
   - Inspect the folder layout. Do NOT assume it. Identify:
     - (a) GS camera image folder + timestamps;
     - (b) RS camera image folder + timestamps;
     - (c) IMU (gyro) data;
     - (d) ground-truth poses;
     - (e) calibration file: intrinsics, distortion, camera↔IMU extrinsics, RS line delay / readout time (expected ~29.5 µs per row — read the real value).
   - Note from the dataset page: GT poses are given in the **IMU frame** and are time-synchronized. Convert to each camera frame with that camera's extrinsic: `T_world_cam = T_world_imu · T_imu_cam`.
   - Write the discovered layout into `results/exp0/DATA_NOTES.md`.
2. **Models** (official checkpoints only):
   - MapAnything: `facebookresearch/map-anything` (Apache 2.0 checkpoint) — first priority.
   - VGGT: `facebookresearch/vggt` (VGGT-1B).
   - π³: `yyfz233/Pi3`.
   - Images only, no priors (no poses/intrinsics/depth given to MapAnything).
3. **Preprocessing — identical for GS and RS:**
   - Undistort with each camera's calibration.
   - If grayscale, replicate to 3 channels.
   - Resize to each model's default input (e.g. 518 long side).

## 3. Clip construction (~15 min)

- For each sequence, make sliding windows of **N = 16 frames** with a frame stride giving visible parallax (start with stride 3; adjust so per-window GT translation is ≳ 0.2 m). Window step ≈ 8 frames.
- **Match GS and RS windows by timestamp** (nearest frame, tolerance < 5 ms). Same windows for every model.
- **Per window, record:**
  - `omega_mean` = mean ‖gyro‖ over the window (rad/s) — the primary speed axis;
  - `rs_shift_px` = omega_mean × readout_time × focal_px — an approximate rolling-shutter pixel shift (secondary axis);
  - GT translation length (to drop near-static windows).
- **Speed bins:** slow < 0.5, medium 0.5–1.5, fast > 1.5 rad/s. If any bin has < 15 windows, use terciles of `omega_mean` instead and report the cut points.

## 4. Metrics

Per window, per model, per camera (GS / RS), against that camera's GT:

- **ATE RMSE** (m) after Sim(3) Umeyama alignment of predicted to GT camera centres.
- **RPE-rot** (deg): mean over consecutive frame pairs of the relative rotation angle error.
- **RPE-trans** (scale-aligned, m).
- Optional: RRA@5° / RTA@5° over all pairs (VGGT-style AUC).

## 5. Analysis

1. Per bin, report **median** GS error, median RS error, and **ratio = median_RS / median_GS** per metric. Add bootstrap 95% CIs (1000 resamples).
2. **Paired test:** Wilcoxon signed-rank on (RS − GS) per window, per bin.
3. **Plot:** error vs `omega_mean` (scatter + binned median line), GS vs RS, one panel per model. Save PNG.
4. **Confound check:** the slow-bin ratio should be ≈ 1. If it is > 1.3, RS/GS differ for non-motion reasons (exposure, noise, camera position). Flag it and report the fast-bin ratio divided by the slow-bin ratio as a corrected gap.

## 6. GO / STOP verdict

Main metric is ATE; RPE-rot is supporting. Decide per model, then overall.

- **GO:** fast-bin ratio ≥ 1.5× AND ratio increases slow→fast AND slow-bin ratio ≤ 1.2×, for MapAnything plus at least one other model.
- **GRAY:** fast-bin ratio 1.2–1.5×. Proceed to EXP1 (oracle headroom) and decide there.
- **STOP:** fast-bin ratio < 1.2× for all models.

## 7. Deliverables (`~/rs3d/results/exp0/`)

- `DATA_NOTES.md` — dataset layout, calibration values used, readout time.
- `per_window.csv` — seq, window_id, model, camera, omega_mean, rs_shift_px, ATE, RPE_rot, RPE_trans.
- `bins_summary.csv` — per model × bin × metric: median GS, median RS, ratio, CI, p-value, n.
- `err_vs_speed_<model>.png`.
- 2–3 qualitative examples: the fastest windows, GS vs RS reconstructions side by side (point-cloud screenshots or top-down trajectory plots).
- `summary.md` containing:
  - the GO / GRAY / STOP verdict;
  - the key table;
  - any confounds;
  - anything that deviated from this plan;
  - wall-clock time and GPU memory used.
  - Label every number as **measured**.

## 8. Do NOT

- Train or fine-tune anything.
- Give MapAnything any GT priors (that is a different experiment).
- Cherry-pick windows. Report all windows that pass the static-filter rule, and state the rule.
- Touch the other experiment.
