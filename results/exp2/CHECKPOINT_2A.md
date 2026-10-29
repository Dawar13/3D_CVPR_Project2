# Checkpoint 2A: literature and data (2026-10-29)

Details are in `LIT_REVIEW.md`, `related_work.csv`, `DATA_AUDIT.md` and `datasets.csv`.

## Novelty verdict: OPEN, 80% confidence (scoop.md §2)

**STOP rule: not triggered.** The rule is "a paper already does our exact idea: RS motion inside a feed-forward multi-view 3D FM, images only". scoop.md §1: "No paper found that does our exact idea."
- 0 of 40 arXiv RS papers since 2025 combine RS with a feed-forward or foundation 3D model.
- 0 RS hits among 3,386 papers that cite VGGT, DUSt3R, MASt3R, π³ or MapAnything.
- None in the CVPR, ICCV, ECCV, 3DV or WACV title lists, or on OpenReview.

**Nearest papers.** Each misses at least one of: RS, inside a multi-view feed-forward FM, images only.
- **Foundation4D:** RS only in label generation and rendering; one pose per image.
- **Image as an IMU:** single image, motion blur, "we do not model for rolling shutter".
- **Toronto RS-as-IMU report:** single image, two-stage, not peer-reviewed.
- **Interp3R:** needs events; time is between frames.
- **Anchor Points:** 2D de-rolling, then monocular depth.
- **Point4Cast:** frame time for dynamic scenes.

**Residual risk:** concurrent CVPR 2027 work; RS mentioned only in body text (as in Foundation4D); journals not scanned. Re-search weekly.

**Safe claim:** "first to model intra-frame (row-time) camera motion inside a feed-forward multi-view 3D foundation model, from images only".
**Do not claim:** first velocity-from-images (SMARSC 2019; Image as an IMU), first time-conditioned pointmap, or first FM to consider RS.

## Final baselines

- **2D unroll → 3D model.** Each is run zero-shot and fine-tuned on our synthetic RS, because the released weights assume γ = 1 and TUM's ratio is ~0.60.
  - **DFRSC** (CVPR 2024): github.com/ljzycmd/DFRSC, no LICENSE file.
  - **JAMNet** (CVPR 2023): github.com/GitCVfb/JAMNet, research-only.
  - **AdaRSC / BS-RSC** (CVPR 2022): github.com/ljzycmd/BSRSC, MIT.
  - **LBCNet** (TPAMI 2024; fallbacks RSSR, CVR): github.com/GitCVfb/LBCNet, research-only.
  - **DeepUnrollNet** (CVPR 2020): github.com/ethliup/DeepUnrollNet, MIT.
  - **JCD** (CVPR 2021, blurry clips only): github.com/zzh-tech/RSCD, MIT.
- **Classical.**
  - **COLMAP / GLOMAP** as the GS-model reference (no RS support; BSD / BSD-style).
  - **Barath, rolling_shutter_made_practical** (ECCV 2026; TUM RS scripts): github.com/danini/rolling_shutter_made_practical, MIT.
- **Ours and controls.**
  - **Self-estimate two-pass** (training-free; `src/tum_conditions.py`, `src/synth_conditions.py`).
  - **Gyro straightening** (EXP1 E1a, Karpenko-style, privileged). It closes 82–99% of the fast-clip ATE gap (measured).
  - **LoRA-only control** (same data, no twist head, 3 seeds).

## Final datasets

| Dataset | Role | Status |
|---|---|---|
| TUM RS-VIO | main real RS test + GS–GS control (calib-imu1) | **in hand, prepped:** 10 sequences + calib-imu1, ~22 GB, 1124 windows, identical to EXP0 (measured) |
| ADVIO | real RS stress test (coarse GT, readout unknown) | **in hand:** 23 sequences, CC BY-NC 4.0 (measured) |
| Replica | synthetic RS, dense GT | **in hand:** 18 scenes; vertex-coloured quad-mesh PLYs, e.g. room_0 954,492 vertices (measured) |
| ETH3D SLAM | GS no-harm control (mocap) | **in hand:** 61 training sequences, mono + IMU, CC BY-NC-SA 4.0 (measured) |
| TartanAir V2 | synthetic RS (1000 Hz video) | approved by the user (CC BY 4.0); not yet downloaded |
| Waymo, Aria Digital Twin, ScanNet++ | real RS + 3D GT | waiting on the user's sign-up (HUMAN_TODO 1–3); VGGT-Ω weights too (item 4) |

**Do not count as GS controls:** CO3Dv2, RealEstate10K, ScanNet / ScanNet-1500 and ScanNet++ iPhone; and, because their shutter is unknown, 7-Scenes, TUM RGB-D and Bonn.

## Consequences for EXP2

1. **"Use the gyroscope" is published** (Karpenko 2011 onward), so E1a is a baseline, not a contribution.
2. **Carry each model's resize and crop into the row-time map.** MapAnything and CUT3R crop rows.
3. **Report pure-translation segments separately** (SMARSC's degeneracy).
4. **Before running ADVIO and ETH3D SLAM:**
   - verify their timestamp reference from the files;
   - log their strides in `DEVIATIONS.md` (P10).
