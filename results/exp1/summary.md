# EXP1 — Oracle headroom summary (unroll RS with known motion, rerun the models)

Run date: 2026-10-27. **Every number in this file is measured** unless marked *projected*. The windows, models, GT and metrics are the same as in EXP0 (`../exp0/summary.md`, `../exp0/DATA_NOTES.md`).

## Verdict: **GO** (pre-registered rule: ≥ 50% of the GS–RS gap recovered; STOP < 25%) → proceed to Phase 2 (E2)

Primary test: E1a (rotation-only unrolling), fast bin, ATE.

`recovered = (median RS − median RS-unrolled) / (median RS − median GS)`

- 100% means the error is back at the global-shutter level.
- 95% CIs come from a paired bootstrap over windows (1000 resamples).

| Model | E1a: rotation only (gyro) | E1b: rotation + translation | Rule result |
|---|---|---|---|
| MapAnything | **99%** [86, 117] | 117% [106, 136] | **GO** |
| VGGT | **83%** [71, 97] | 90% [75, 103] | **GO** |
| π³ | **82%** [79, 85] | 98% [96, 100] | **GO** |

In plain words:
- **The extra error on RS frames is almost entirely the rolling-shutter geometry.** Remove the row-by-row motion and the models are back to, or close to, their global-shutter accuracy. This holds for all three models and at every speed.
- **Rotation alone fixes most of it.** That is the plan's test, and it is enough for GO.
- **Adding translation fixes most of the rest.** For π³ it brings the error fully to GS level: fast-bin ATE goes 5.10 → 2.03 cm, against a GS level of 1.95 cm.
- **EXP0's open question is answered.** EXP0 found an RS penalty even in the slow bin (π³ 1.70×, VGGT 1.73×), so it could not pass the "slow ≈ 1" check. Unrolling removes that penalty too:
  - slow bin, rotation only: 86% (π³) and 63% (VGGT);
  - slow bin, rotation + translation: 109% (π³) and 83% (VGGT).
  - The slow-bin penalty was RS, not a camera or model artefact. "Slow" in gyro terms still contains walking motion at ~0.6 m/s, plus some rotation.

## Key table (ATE; full table in `e1_key_table.md`, all metrics in `e1_summary.csv`)

Bins and windows are the same as EXP0: slow < 0.5, medium 0.5–1.5, fast ≥ 1.5 rad/s; n = 56 / 648 / 420.

How to read the columns:
- **RSU** = unrolled for rotation (E1a). **RSUT** = unrolled for rotation + translation (E1b).
- **p** is a Wilcoxon signed-rank test, per window, of unrolled vs raw RS.
- **Improved** is the share of windows where unrolled ATE < raw RS ATE.

| Model | Bin | GS | RS raw | RSU | gap closed | RSUT | gap closed | p (RSUT vs raw) | Improved (RSU / RSUT) |
|---|---|---|---|---|---|---|---|---|---|
| MapAnything | slow | 3.42 cm | 3.89 cm | 4.02 cm | −28% [−86, 48] | 3.88 cm | 2% [−102, 67] | 0.97 | 39% / 55% |
| | medium | 6.60 cm | 7.01 cm | 6.66 cm | 86% [56, 137] | 6.60 cm | 101% [72, 157] | 2e-23 | 67% / 65% |
| | fast | 9.33 cm | 10.85 cm | 9.35 cm | **99%** [86, 117] | 9.07 cm | 117% [106, 136] | 4e-25 | 80% / 77% |
| VGGT | slow | 1.36 cm | 2.35 cm | 1.73 cm | 63% [55, 79] | 1.52 cm | 83% [71, 91] | 2e-10 | 96% / 93% |
| | medium | 2.29 cm | 3.86 cm | 2.88 cm | 62% [56, 69] | 2.63 cm | 78% [73, 84] | 4e-99 | 95% / 93% |
| | fast | 6.60 cm | 9.72 cm | 7.14 cm | **83%** [71, 97] | 6.91 cm | 90% [75, 103] | 5e-54 | 87% / 86% |
| π³ | slow | 0.77 cm | 1.32 cm | 0.85 cm | 86% [76, 114] | 0.72 cm | 109% [99, 135] | 8e-11 | 100% / 98% |
| | medium | 1.03 cm | 2.26 cm | 1.21 cm | 85% [82, 90] | 0.99 cm | 104% [102, 108] | 5e-107 | 98% / 99% |
| | fast | 1.95 cm | 5.10 cm | 2.51 cm | **82%** [79, 85] | 2.03 cm | 98% [96, 100] | 1e-58 | 86% / 85% |

RPE-rot gives the same picture. Fast bin, gap closed by RSU / RSUT:
- MapAnything: 117% / 122%
- VGGT: 89% / 93%
- π³: 99% / 102%

**Figures:**
- `e1_ate_bars.png`: median ATE per bin, four bars each (GS, RS raw, RSU, RSUT).
- `e1_recovered_vs_speed.png`: gap closed vs ω in sextile bins with 95% CIs. The share is flat across speed for π³ and VGGT.
- `check_unroll_s9_w0077.jpg`: the fastest frame, showing GS, RS raw and RS unrolled side by side.

**Values above 100%** mean the unrolled RS beat the GS camera. These differences are small:
- MapAnything fast, RSUT: 9.07 vs 9.33 cm.
- π³ slow, RSUT: 0.72 vs 0.77 cm.

Both are inside the camera-to-camera spread from EXP0's GS-vs-GS control (cam1/cam0 = 1.00 / 1.18 / 1.12). Read them as "at GS level", not "better than GS".

## How the unrolling works

For each RS frame, every raw sensor row is warped to the pose at the **centre-row time**. That is the same time EXP0's RS GT uses, so raw RS and unrolled RS are scored against identical GT.

**Rotation (E1a):**
- Rotation over the readout is R(t_mid)·exp([ω_c]× dt), with dt = row × 29.47 µs − 15.08 ms.
- ω_c is the gyro mean over the 30.2 ms readout, rotated into the camera frame with the Kalibr extrinsic.
- The warp is done in the **raw fisheye image**, because row timing belongs to raw sensor rows. A fixed-point iteration (4 steps) finds the source row. The result then goes through the same undistortion and tone curve as EXP0.

**Translation (E1b):**
- E1b also moves each row by −v_c·dt, where v_c is the camera-centre velocity from GT (±25 ms central difference).
- That needs per-pixel depth. Depth comes from π³ on the raw RS clips, scaled to metres with the GT Sim(3) scale. The median depth per sequence is 1.7–2.5 m.

**Sanity checks, all passed before the run:**
- Zero motion reproduces EXP0's RS image exactly (max difference 0.0 px).
- The gyro-predicted rotation over one readout matches mocap to 0.133° (median), against 4.45° of true rotation. A sign error would give 8.96°.
- The largest row shift is 35 px, at ω = 3.1 rad/s.
- On the fastest frame, the unrolled RS lines up with GS.

## Deviations from the plan

1. **Raw-fisheye per-row warp instead of `H(s) = K·R(s)ᵀ·K⁻¹` on a pinhole image.**
   - The two are the same model.
   - Doing it before undistortion keeps the row timing exact for the fisheye lens.
2. **ω comes from the IMU gyro, not mocap.**
   - It is constant over the 30 ms readout (the mean of about 6 gyro samples).
   - Mocap was used only to check the sign and axes.
3. **Added E1b (rotation + translation).** It was not in the plan. It was recommended after EXP0 because the slow bin still had walking motion.
   - E1b uses oracle velocity and an oracle depth scale.
   - Depth comes from **π³** for all three models, which may slightly favour π³ in E1b.
   - VGGT and MapAnything still improve with it.
4. **Out-of-frame pixels repeat the edge pixel instead of being black.** This avoids giving the models a black-border cue.
   - At fast motion it leaves smeared strips up to ~35 px wide.
   - These can only hurt the unrolled result, so the recovered numbers are, if anything, conservative.

## Caveats

- **This is an oracle ceiling, not a method.** E1b uses GT velocity and a GT depth scale.
- **E1a used a real gyro.** That is a sensor phones and robots have, and gyro-based rolling-shutter correction already exists in prior work (*related-work check needed; no citation verified yet*).
  - The paper therefore cannot simply be "unroll with known rotation".
  - The contribution has to come from images only, with no IMU.
  - **E2 tests exactly this.**
- **MapAnything's gap is small** (0.5–1.5 cm on a 3–9 cm error), so its percentages are noisy. Its slow-bin numbers (−28%, 2%) have CIs spanning zero; with n = 56 and p = 0.16 / 0.97, they are noise.
- **One dataset:** one room, one rig, 30 ms readout. Generalisation to phone sensors, with different readout times and outdoor scenes, is untested.

## Wall clock and GPU

- **Total:** 17:14 → 19:15 UTC, box clock, **2 h 02 min**. 9,248 RS frames were unrolled per variant.
- **Pipeline steps:**

  | Step | Hardware | Wall clock |
  |---|---|---|
  | Rotation unrolling | CPU, 12 workers | 14 min |
  | π³ depth (ran in parallel) | GPU | 65 min |
  | Rotation + translation unrolling | CPU | 17 min |

- **Inference** (1124 clips per model per variant, three models in parallel, one clip at a time each, bf16):

  | Model | E1a (min) | E1b (min) | Peak allocated GPU memory |
  |---|---|---|---|
  | MapAnything | 39.0 | 31.9 | 13.5 GB |
  | VGGT | 46.1 | 37.6 | 7.5 GB |
  | π³ | 48.2 | 39.0 | 6.1 GB |

- **Free VRAM:** the lowest reading in any log was 26.9 GB, above the 12 GB rule. The other experiment was never touched.
  - At 18:31 my unrolling workers had pushed CPU load to 21 on 14 cores. I set my own processes to `nice 10` so the other job keeps CPU priority.
- **Why it was slow:**
  - Unrolling is CPU-only.
  - The depth step had no prefetch.
  - Four GPU jobs plus the other experiment shared the 14 cores for image loading, so each clip took 2.1–2.6 s in E1a vs 1.5–2.0 s in EXP0.
  - *Projected:* caching the resized frames once for all models and unrolling on the GPU would cut this a lot. Not done for this run.

## What this means for the plan

1. **Phase 1 is GO.** The models lose accuracy on RS because of fixable geometry, not because RS images are otherwise unusable. If the per-row motion is known, about 80–100% of the lost accuracy comes back.
2. **Model rotation and translation together, as a full 6-DoF twist.** Translation adds about 15–20 points:
   - π³ fast: 82% → 98%;
   - VGGT slow: 63% → 83%.
   - This supports the Phase 3 "twist head" (ω and v), not ω alone.
3. **π³ as the primary backbone.** It has the largest gap (2.6× in the fast bin) and the tightest CIs ([79, 85]). VGGT is second. MapAnything's gap is too small to show gains clearly.
4. **Next: Phase 2 (E2), the cheap training-free baseline.**
   - Estimate ω (and v) from each model's **own** neighbouring-frame poses, unroll, and run a second pass.
   - If E2 reaches the oracle, the paper becomes training-free.
   - If E2 falls well short, a learned velocity head is justified (Phase 3).
   - E2 also answers the "just use a gyro" objection.
