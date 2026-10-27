# EXP0 — Gap test summary (GS vs RS on TUM RS-VIO)

Run date: 2026-10-27. **Every number in this file is measured.** Data details are in `DATA_NOTES.md`.

## Verdict: **GRAY** (pre-registered rule) → proceed to EXP1

| Model | Fast-bin ATE ratio RS/GS | Slow → medium → fast (ATE) | Slow-bin ≤ 1.2? | Rule result |
|---|---|---|---|---|
| MapAnything | **1.16×** [1.13, 1.21] | 1.14 → 1.06 → 1.16 | yes (1.14) | **STOP** (fast < 1.2) |
| VGGT | **1.47×** [1.32, 1.57] | 1.73 → 1.69 → 1.47 (falls) | no (1.73) | **GRAY** |
| π³ | **2.61×** [2.44, 2.82] | 1.70 → 2.19 → 2.61 (rises, monotone) | no (1.70) | **GRAY** |

- GO needs MapAnything plus one other model to pass, and MapAnything fails. STOP needs every model below 1.2×, and VGGT and π³ are well above it. The overall result is therefore **GRAY**.
- In plain words:
  - **There is a real and large RS penalty for VGGT and π³.** RS is worse than GS in every speed bin, with Wilcoxon p < 1e-9 in every bin.
  - **For π³ the penalty grows strongly with rotation speed.** In finer bins the ATE ratio goes from 1.6× at ω ≤ 0.3 rad/s to 3.9× at ω > 2.5 rad/s, and the absolute penalty goes from +0.4 cm to +5.2 cm.
  - **MapAnything is the least affected**, at ~1.05–1.3×. It is also the least accurate on clean GS input, so its own error may hide the RS effect.
  - **π³ misses GO only on the slow-bin check.** Reasons to think the slow bin is not RS-free are in "Confounds" below.

## Key table (ATE and RPE-rot; full table in `key_table.md` and `bins_summary.csv`)

Bins use the gyro mean ‖ω‖ per window: slow < 0.5, medium 0.5–1.5, fast ≥ 1.5 rad/s. n = 56 / 648 / 420 windows, the same windows for every model. Ratio = median RS / median GS, with a 95% paired bootstrap CI (1000 resamples). p is a Wilcoxon signed-rank test on per-window RS vs GS.

| Model | Metric | Bin | Median GS | Median RS | Ratio [95% CI] | p |
|---|---|---|---|---|---|---|
| MapAnything | ATE | slow | 3.42 cm | 3.89 cm | 1.14 [1.06, 1.27] | 8e-04 |
| | | medium | 6.60 cm | 7.01 cm | 1.06 [1.03, 1.08] | 7e-14 |
| | | fast | 9.33 cm | 10.85 cm | 1.16 [1.13, 1.21] | 3e-40 |
| | RPE-rot | slow / med / fast | 0.37° / 0.56° / 1.23° | 0.38° / 0.62° / 1.54° | 1.01 / 1.10 / **1.26** | — / 5e-62 / 7e-47 |
| VGGT | ATE | slow | 1.36 cm | 2.35 cm | 1.73 [1.53, 1.97] | 9e-11 |
| | | medium | 2.29 cm | 3.86 cm | 1.69 [1.58, 1.77] | 7e-106 |
| | | fast | 6.60 cm | 9.72 cm | 1.47 [1.32, 1.57] | 2e-59 |
| | RPE-rot | slow / med / fast | 0.22° / 0.30° / 0.67° | 0.32° / 0.46° / 1.14° | 1.47 / 1.54 / **1.68** | 8e-11 / 2e-107 / 3e-69 |
| π³ | ATE | slow | 0.77 cm | 1.32 cm | 1.70 [1.43, 1.89] | 2e-10 |
| | | medium | 1.03 cm | 2.26 cm | 2.19 [2.01, 2.30] | 1e-106 |
| | | fast | 1.95 cm | 5.10 cm | **2.61 [2.44, 2.82]** | 5e-56 |
| | RPE-rot | slow / med / fast | 0.18° / 0.23° / 0.42° | 0.24° / 0.36° / 0.77° | 1.30 / 1.61 / **1.80** | 5e-09 / 2e-106 / 4e-68 |

The RPE-rot ratio rises slow → fast for **all three** models. The ATE ratio rises only for π³ (and weakly for MapAnything).

**Figures:**
- `ratio_vs_speed.png` (all models, sextile bins with CIs);
- `err_vs_speed_<model>.png`;
- `qual_traj_s9_w0077/0078/0089.png` (the 3 fastest windows, ω ≈ 3.5 rad/s, top-down trajectories). In s9_w0077, π³ has ATE 2.8 cm on GS vs 8.2 cm on RS.
- `check_fastest_pair.jpg` shows the GS and RS frames side by side.

## Confounds

1. **Different physical cameras: measured with an extra GS-vs-GS control (not in the plan).**
   - In `dataset-calib-imu1` **both** cameras ran in global-shutter mode. We ran the same pipeline on it: 112 windows, ω terciles.
   - cam1/cam0 ATE ratio, all windows:

     | Model | cam1/cam0 ATE ratio [95% CI] |
     |---|---|
     | MapAnything | 1.00 [0.89, 1.16] |
     | VGGT | 1.18 [1.09, 1.26] |
     | π³ | 1.12 [0.90, 1.27] |

   - So camera identity explains **at most ~1.1–1.2×**.
   - Fast-bin ratio ÷ control ratio:

     | Model | Fast ÷ control |
     |---|---|
     | MapAnything | 1.17 |
     | VGGT | 1.25 |
     | π³ | **2.33** |

   - Caveat: the control scene (a calibration target, 1 ms exposure) differs from seq 1–10.
2. **Slow-bin ratio > 1.3 for VGGT (1.73) and π³ (1.70): flagged as the plan requires.**
   - The corrected gap (fast ÷ slow) is MapAnything 1.02, VGGT 0.85, **π³ 1.53**.
   - The plan's check assumes the slow bin has ≈ no rolling-shutter effect. **On this rig that assumption does not hold:**
     - the slow bin still has a median rotation-induced RS shift of **7.5 px**;
     - its median linear speed is **0.61 m/s**, i.e. ~1.8 cm of translation during the 30 ms readout in a small room.
   - VGGT and π³ have sub-cm GS error, so a few pixels of RS distortion is already a large *relative* error.
   - Even the 28 slow windows with v < 0.6 m/s show ratios of 2.0 (VGGT) and 1.9 (π³). That residual is not fully explained by either the control or motion, and it is **unresolved**.
   - EXP1 (oracle unrolling) tests this directly: if unrolling with GT motion closes the slow-bin gap too, the gap is RS.
3. **Rotation vs translation.**
   - ω and v correlate at 0.40 across the kept windows.
   - An OLS of log(RS/GS) on ω and v (`regression_logratio.csv`) gives a positive ω coefficient for π³ (ATE: 0.21 ± 0.02 per rad/s) and MapAnything (0.05 ± 0.01).
   - VGGT's ATE coefficient is negative (−0.10 ± 0.01), though its RPE-rot coefficient is positive.
4. **Things that are not confounds:**
   - Exposure is a fixed 2.0 ms on both cameras in all frames, so blur is matched.
   - Frames are paired exactly by timestamp (0.0 ms offset).
   - Identical undistortion to one shared K and identical tone mapping are applied to both cameras.
   - **RS GT timing:** RS GT is taken at the centre-row time (t + 15.08 ms). Using the first-row time instead raises the median RS ATE by +1.6% (MapAnything), +2.6% (VGGT) and +10% (π³). The conclusions are unchanged, and the centre-row choice is the conservative one.
5. **Static-filter rule:** windows were dropped if their GT camera path was < 0.20 m or any frame lacked valid GT.
   - 1124 of 1139 windows were kept, and no others were excluded.
   - Median GT path per window is 2.04 m.

## Deviations from the plan

- **π³ repo:** `github.com/yyfz233/Pi3` returns 404. We used the official code at `github.com/yyfz/Pi3` (commit 9fa3ddb) with the HF weights `yyfz233/Pi3`.
- **Venv:** made with `uv` (the box has no `ensurepip`, so `python -m venv` fails), at `~/rs3d/.venv` with Python 3.11 and torch 2.7.1+cu128.
- **Concurrency:** the three models ran **concurrently** (each one clip at a time, bf16). Free VRAM never went below ~42 GB.
- **MapAnything settings:** `memory_efficient_inference=False, apply_mask=False, mask_edges=False`. These only post-process dense outputs; camera poses are unaffected. The change was for a 3× speed-up.
- **Memory leak:** a leak in my prefetch loop (host RAM, not GPU) was caught at 400/200/<200 clips. The run was stopped and resumed from checkpoints, and the predictions are unaffected.
- **Extras not in the plan:**
  - the GS-vs-GS control set;
  - a ω/v regression and fine ω bins (`ate_fine_bins.csv`);
  - an RS-GT timing sensitivity check;
  - RRA@5° / RTA@5° (in `per_window.csv` and `bins_summary.csv`).
- **Qualitative figures** are top-down trajectory plots, not point-cloud screenshots.
- **Leftover process:** one hung `git clone` of the dead π³ URL is still sitting idle in tmux window `rs3d:setup`. It uses no CPU or GPU. Killing processes on the shared box was blocked, so it needs to be closed by hand (`tmux kill-window -t rs3d:setup`).

## Wall clock and GPU

- **Total wall clock:** ~1 h 50 min, from first login at 14:13 UTC to final evaluation at ~16:03 UTC. Setup and downloads took ~25 min, preprocessing ~5 min, inference ~70 min (including the control) and analysis ~5 min.
- **Inference time** (2248 clips each = 1124 windows × 2 cameras, running concurrently):

  | Model | Inference time | Peak GPU allocated (reserved) |
  |---|---|---|
  | MapAnything | 58 min | 13.5 GB (18.2 GB) |
  | VGGT | ~70 min | 7.5 GB (7.7 GB) |
  | π³ | ~78 min | 6.1 GB (6.4 GB) |

  The control set (224 clips per model) took 2–6 min each.
- **Box-level GPU memory peak:** 38.7 GB of 80 GB. No other process used the GPU during the run.

## What this means for the plan

- **Proceed to EXP1 (oracle headroom).** That is what the plan says for GRAY, and EXP1 also resolves the slow-bin question.
- **Consider π³ (then VGGT) as the primary backbone instead of MapAnything.**
  - π³ shows the largest, speed-dependent gap (2.6× fast-bin ATE).
  - MapAnything shows only ~1.2× here, and its clean-GS error is 2–6× higher than the others, which would cap any visible gain.
  - Before writing MapAnything off, check that its image-only setup matches the other team's (e.g. input resolution).
- **Unrolling in EXP1:** the plan's rotation-only homography will not remove translation-induced RS. The rig moves at ~0.6–1.1 m/s, so add a variant that also uses GT translation (with model depth), or report rotation-only as a lower bound.
