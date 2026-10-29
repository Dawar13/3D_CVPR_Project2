# EXP2: deviations and pre-registration clarifications

Rule 3 of the EXP2 brief says that if a rule is unclear or wrong, the fix is written down here **before** running.

- Every entry below the "Pre-registration" line was written on **2026-10-29, before any EXP2 result existed**.
- Later entries carry their own date and say whether they were written before or after the relevant results.

---

## Pre-registration (2026-10-29, before any EXP2 result)

### P1. The GPU is shared, not "ours alone" (brief §2)
- **What the brief says:** "This GPU is ours alone unless the user says otherwise."
- **What the user said** (2026-10-29): another experiment, "the Exp 1 of the 3d project", will run on the same GPU at the same time.
- **So the EXP0/EXP1 shared-GPU rules apply again:**
  - work only in `~/rs3d/`;
  - use our own venv and HF cache;
  - never touch the other job's files or processes;
  - check `nvidia-smi` before every launch;
  - never preempt the other job;
  - wait if free VRAM is under 12 GB.
- **Effect on EXP2:** wall clock only. VRAM-heavy jobs (LoRA training) are scheduled around the other job's memory use.

### P2. The primary metric for every decision is ATE
- ATE is Sim(3)-aligned, as in EXP0/EXP1. RPE-rot, RPE-trans, AUC@30, RRA/RTA and the dense metrics are all reported.
- A decision in §10 uses a secondary metric only where the rule names one. Q3 names one: dense geometry.

### P3. Q1: what "fairness-adjusted" and "significant slope" mean
1. **Fairness-adjusted ratio, where a GS twin exists:**
   - adjusted ratio = (RS/GS median ATE ratio in the fast bin) ÷ (the same ratio from a GS-vs-GS control).
   - **TUM RS-VIO:** the control is the `dataset-calib-imu1` sequence, where both cameras ran global shutter, the same as in EXP0.
   - **Synthetic:** the GS twin is the identical virtual camera, so the factor is 1.
   - "Fast" means ω ≥ 1.5 rad/s, as in EXP0. On synthetic data the same cut is applied to the true ω.
2. **Slope test, where no GS twin exists:**
   - Fit log(ATE) on the per-window angular speed ω using OLS.
   - Get the 95% CI with a 1000-sample bootstrap, resampling windows *within sequences*, then sequences.
   - The p-value is one-sided (slope > 0), from the bootstrap.
3. **Holm family for Q1:** all (dataset × model) slope or ratio tests on ATE.
4. **Counting toward "≥ 2 real datasets, ≥ 2 models":**
   - a (dataset, model) pair counts if its adjusted ratio is ≥ 1.3× **or** its Holm-corrected slope p is < 0.05;
   - Q1 is YES if at least 2 real datasets and synthetic each have at least 2 models that count.

### P4. Q2: how headroom is computed
- For each (dataset, model) pair:
  - **oracle gain** = med ATE(V2) − med ATE(V4);
  - **recovery** of a method M = (med ATE(V2) − med ATE(M)) / (oracle gain).
- **Which pairs enter Q2:** only pairs whose oracle gain is significant (paired Wilcoxon V2 vs V4, Holm-corrected p < 0.05). Pairs left out are listed with the reason.
- **Best existing method:** chosen per pair, as the one with the highest recovery among:
  - 2D unrollers (authors' weights and fine-tuned);
  - classical RS-BA/SfM;
  - the training-free self-estimate two-pass;
  - LoRA-only 6a (mean over 3 seeds).
- **Decision:** Q2 = YES if the best existing method's recovery is < 60% on at least half of the eligible pairs.

### P5. Q3: what a "significant margin" for V5 means
- V5 "beats V3/V4 on dense geometry" if **both** hold:
  1. its median Chamfer distance is at least 10% lower than the better of V3 and V4;
  2. the paired Wilcoxon test is significant at Holm-corrected p < 0.05, with the Holm family taken across models.
- This is on synthetic data only, because no other dataset has both dense GT and a known motion.
- Pose metrics are reported for V5 but **not** used for Q3. V5 cannot change the model's poses (brief §7).

### P6. Q5: what "±25% readout error costs > 20% of the oracle gain" means
- **lost share** = (med ATE(V4 with readout × (1 ± 0.25)) − med ATE(V4 correct)) / oracle gain.
- Take the worse of +25% and −25%.
- Q5 = YES if the lost share is > 0.20 on at least half of the eligible pairs.
- If V4 is not possible for a pair, the same rule is applied to V3.

### P7. V4 on TUM uses estimated depth, not true depth
- TUM RS-VIO has no dense depth. V4 on TUM therefore uses the EXP1-b recipe: π³ depth scaled to metres with the GT Sim(3) scale.
- So the TUM "oracle" is slightly weaker than a true oracle.
- On synthetic data, V4 uses the true rendered depth.

### P8. V5 only where dense GT exists
- V5 (row-time unprojection of the model's own output, using the true motion) changes only 3D points.
- It is therefore run and scored only on synthetic data.
- On TUM there is no dense GT to score it against.

### P9. Statistics conventions
- Medians use 1000-sample bootstrap 95% CIs.
- Paired Wilcoxon tests are run wherever pairs exist (same window, two conditions).
- Holm–Bonferroni correction is applied inside each question's family.
- n is reported everywhere.
- Training results use 3 seeds, reported as mean ± std.
- These settings follow the brief.

### P10. Windows and clip protocol
- **Real data:** EXP0 windows, i.e. 16 frames, stride 3, step 8, the static filter of < 0.20 m path, and GT at the centre row.
- **Synthetic:** clips are generated directly as 16-frame windows, at a 20 fps base rate with stride 3 (150 ms between frames), to match TUM.
- Any dataset that needs a different stride because of its frame rate is recorded here before it is run.

---

## Deviations logged during EXP2

| # | Date | Before/after results | What | Why |
|---|---|---|---|---|
| D1 | 2026-10-29 | before | The local repo was committed (`1a41da4`) and will be **copied** to the VM, not pushed to GitHub. | The brief allows "push or copy". Copying avoids publishing to the remote without asking. |
| D2 | 2026-10-29 | before | The VGGT commit used in EXP0 was not recorded. The reproduction check pins the last `vggt` commit on or before 2026-10-27 (resolved on the VM with `git rev-list -1 --before=2026-10-27T23:59 main`). | EXP0's notes give MapAnything `3d10cf7` and π³ `9fa3ddb`, but not VGGT. |
| D3 | 2026-10-29 | before | The 2.1 TB disk was attached but unformatted (`/dev/vdc`, checked blank). It was formatted ext4 and mounted at `/data`; `~/rs3d` is a symlink to `/data/rs3d`. The other project's files stay in `~` and are not touched. | The 92 GB root disk is too small for EXP2 data. |
| D4 | 2026-10-29 | before (no synthetic data yet) | **Synthetic scene split, fixed now.** Train: frl_apartment_0–5, apartment_0, apartment_1, office_0, office_1, room_1, hotel_0 (12). Val: room_2, office_2 (2). Test: room_0, office_3, office_4, apartment_2 (4). | The six frl_apartment scenes are one physical apartment in different arrangements, so they go to one split to avoid leakage. Test covers a living room, two offices and a multi-room apartment. |
| D5 | 2026-10-29 | before | **Synthetic camera:** 640×512 (5:4) with 90° horizontal FOV, about the aspect and field of view of the undistorted TUM frames. 20 fps base rate, stride 3, 16 frames (P10). Core test readout 30 ms, top→bottom, no blur. The train/val split samples readout {0,5,10,20,30,40} ms and direction tb/tb/tb/bt/lr. All sets are written by `configs/synth/make_configs.py`. | Matches TUM so the models see the same image geometry. Stress sets vary one knob at a time. |
| D6 | 2026-10-29 | before | **Paired stress sets:** a per-clip seed with separate motion/shutter/object random streams, and fixed trajectory padding (0.25 s), so each stress set's clips 0–14 per test scene have exactly the core test trajectories (checked numerically). The view-count and frame-spacing sets change the motion itself, so they are paired only within the set. | Lets every stress comparison be a paired test. |
| D7 | 2026-10-29 | before | **Replica appearance:** we render the mesh's per-vertex colours, not the HDR PTex textures, because the PTex renderer is a separate SDK. Surfaces are therefore a little blurrier than in Replica's own renders. Open3D could not read the quad meshes, so `rsgen/scenes.py` has its own PLY reader that splits each quad into two triangles. | Recorded as a realism limitation of the synthetic data. |
| D8 | 2026-10-29 | before | **Environment:** π³'s `requirements.txt` (torch 2.5.1, opencv-python 5) is not installed. The env uses torch 2.7.1+cu128, opencv-python-headless 4.10, numpy 2.4.6 and pandas 2.3.3; pandas 3.1 was broken in this env. Exact versions are in `logs/pip_freeze_main.txt` on the VM. | One env for π³, VGGT and MapAnything; the reproduction check (§2.5) tests whether this matters. |
| D9 | 2026-10-29 | before (no ADVIO result yet) | **ADVIO protocol:**<br>• 60 fps → stride 9 (150 ms), 16 frames, a window every 72 frames (1.2 s, not 0.4 s, so about 23 sequences × several minutes stays tractable), static filter < 0.20 m.<br>• GT pose interpolated at each frame's stamp; where that stamp falls within the RS readout is undocumented (UNVERIFIED).<br>• Speed axis = mean gyroscope \|ω\| over the window.<br>• **Only ATE is used** (and reported) for ADVIO. Rotation metrics are not used. | The README does not give the quaternion order of `ground-truth/pose.csv`. A vector test (GT body rates vs gyroscope, best fixed rotation) was inconclusive for both orders (residual 50–100 % of \|ω\|), so GT orientation cannot be trusted at window scale. ATE uses camera positions only. Readout is unknown (calibration line delay = 0): ADVIO is the "unknown readout" real dataset. |
| D10 | 2026-10-29 | before (no ETH3D result yet) | **ETH3D SLAM protocol (GS no-harm check):**<br>• ~27.1 Hz → stride 4 (148 ms), a window every 11 frames (~0.4 s), static filter < 0.20 m.<br>• GT (TUM format) interpolated at the RGB stamps; gaps > 20 ms on either side drop the window.<br>• Speed axis from GT orientation.<br>• Whether `groundtruth.txt` is the RGB camera pose or the rig pose could not be re-checked (the site timed out): UNVERIFIED. | A lever arm would add the same small offset for every method; the no-harm check compares methods on the same windows. |
| D11 | 2026-10-29 | before (Q1 not yet computed for EXP2) | **P3 fairness factor = the control's ratio over all its windows** (cam1/cam0 median ATE on `dataset-calib-imu1`), not its "fast" bin. This is EXP0's convention (`evaluate.py`, `verdict.json`). | The calibration sequence never reaches the fast range: its fastest tertile is 0.89–1.72 rad/s, against ≥ 1.5 rad/s in the main set, so a same-bin control ratio cannot be formed. Camera identity (exposure, position) does not depend on speed. Using the fast-tertile control ratio instead would raise MapAnything's adjusted ratio, so this choice is the conservative one for Q1. |
| D12 | 2026-10-29 | before | **Q1 applies P3 exactly as written:** twin datasets (TUM, synthetic) use the adjusted-ratio criterion; no-twin datasets (ADVIO and any later real RS set) use the slope criterion. A log(RS/GS)-vs-ω slope on TUM is reported as an extra analysis only, labelled not pre-registered, and is not counted. | Keeps the pre-registered rule unchanged after EXP0's numbers were known. |
| D13 | 2026-10-29 | before (no V5 or Q2–Q5 result yet) | **How per-model tests combine into a decision:**<br>• **Q3:** YES if the P5 test holds for both main models (π³ and VGGT), MIXED if for one, NO if for neither. Secondary dense metrics (corr_err, plan_rms) are reported, not decisive.<br>• **Q2/Q5 pairs** are (dataset × model) with V4 available: TUM and synthetic core_test, for π³, VGGT and MapAnything.<br>• **"Best existing method" (P4)** is chosen among *ordered-video* methods. The shuffled (unordered) variants are reported separately, because the main setting is video.<br>• **Classical SfM** (COLMAP/GLOMAP) is model-independent: its windows enter every model's pair.<br>• **LoRA 6a recovery** = mean of the 3 seeds' recoveries. | P4/P5 define per-pair tests but not how they are aggregated. |
| D14 | 2026-10-29 | before (no ADVIO, ETH3D or stress result yet) | **GPU budget: secondary runs are thinned, decision runs are not.**<br>• ADVIO gap test: every 2nd kept window per sequence (windows 2.4 s apart, no overlap).<br>• ETH3D no-harm reference: every 3rd kept window.<br>• ADVIO gyroscope/self-estimate readout sweep: every 6th kept window (a subset of the gap test's windows, so every comparison stays paired).<br>• Stress tests (2G): π³ and VGGT only; MapAnything is the secondary model and stays in the gap, oracle and core-test runs.<br>• Unchanged, all windows: TUM (gap, oracle ladder, sensitivity, self-estimate), synthetic core test, training controls, unrollers. | The single A100 is shared by our runs, 4 helper jobs and the other project, and runs at 97–100 % utilisation. The full queue was ~160 k clip inferences (> 1 day). Thinning is fixed by window index, not by results. Consecutive ADVIO/ETH3D windows overlap, and the slope test resamples whole sequences, so the decision-relevant power is kept. |
| D15 | 2026-10-29 | before (no Waymo result yet) | **Waymo Open Dataset v1.4.3 protocol:**<br>• Validation split; every 6th segment in sorted name order (34 segments), fixed before any download.<br>• FRONT camera; JPEGs used as stored (not undistorted, images only).<br>• 10 Hz → stride 1 (100 ms between frames; 16 frames = 1.5 s), a window every 8 frames, static filter < 0.20 m.<br>• GT camera pose at each image's **centre readout time**: t_c = trigger + shutter/2 + readout/2, with readout = readout_done − trigger − shutter (devkit `camera_model.cc`). The image's vehicle pose is extrapolated from `pose_timestamp` to t_c with its global-frame velocity (v, ω), as `dataset.proto` documents. The camera pose then comes from the calibration extrinsic plus the fixed Waymo-camera → OpenCV axis change.<br>• Speed axis: mean \|ω\| of the window's images.<br>• Q1 criterion: the P3 slope test (no GS twin). | 10 Hz is the dataset's rate. Stride 1 vs 2 are equally far from TUM's 150 ms; stride 1 keeps windows short (~15 m at urban speed). The centre-time GT matches EXP0's TUM convention. |
| D16 | 2026-10-29 | before (no StreamVGGT result seen; DA3 TUM result already reported by the helper) | **Extra-model coverage under the GPU budget:**<br>• DA3-Giant-1.1: TUM (all windows + control), ADVIO and ETH3D (the D14 thinned windows), synthetic core test (rs, gs).<br>• StreamVGGT: official streaming inference runs at 10–15 s per window. It is run on TUM on every 2nd window per sequence (same windows for GS and RS), on the TUM control set, and on the synthetic core test. It is not run on ADVIO, ETH3D or Waymo.<br>• DA3-Nested is not run (DA3-Giant covers the DA3 family). | StreamVGGT on every set would take ~30 h of the shared GPU. TUM + synthetic give it a twin-based Q1 test on real and synthetic data. |
| D17 | 2026-10-29 | after its partial results (stopped for GPU time, not because of them) | **Barath RS relative pose stopped at 81 TUM windows** (27 per speed bin, 1,215 consecutive pairs). The windows were processed in a fixed random order (seed 0), so they are a random subset. COLMAP/GLOMAP (CPU only) keep running to all 1,124 windows. | RoMa matching takes ~4 s per pair on the shared GPU; all 9,216 pairs would take ~10 h of GPU needed by the decision runs. Barath gives pairwise metrics only (no window ATE), so it does not enter the Q2 recovery computation. |
| D18 | 2026-10-30 | before (no ADT model result yet; only the data checks below) | **Aria Digital Twin (ADT) protocol:**<br>• **Selection:** the 235 sequences in the user's `ADT_download_urls.json` (the 10 s sample excluded), sorted by name, every 6th from the first: 40 sequences (31 Apartment, 9 Lite office), fixed before any download (`data/adt/selected.txt`).<br>• **Images:** RGB camera (rolling shutter, 1408×1408, 30 fps, Fisheye624), undistorted with projectaria_tools to a 512×512 pinhole camera with f = 256 px (90° FOV, as the synthetic camera), then rotated 90° clockwise to upright. The rotation was chosen from GT gravity: world-up is image-left in the raw frame (mean −0.92 along x on seq131). After the rotation the sensor rows are image columns, so the scan runs horizontally (right→left if the sensor reads raw row 0 first; that sign is UNVERIFIED). Only V2 (plain RS) is run, so the direction is not used.<br>• **Timing and GT:** Aria docs give t(p) = t_image + (row(p)/H − 0.5)·readout with readout = 5 ms at 1408×1408, so the capture timestamp is the centre-row time. GT (`aria_trajectory.csv`, ~30 Hz, device time) is interpolated at each frame's capture timestamp (linear position, slerp). Camera = T_world_device · T_device_camera-rgb (VRS factory calibration) · upright rotation. This is the same centre-row convention as TUM and Waymo.<br>• **Windows:** 30 fps → stride 4 (133 ms; stride 5 at 167 ms is equally far from 150 ms, and stride 4 keeps windows shorter), 16 frames (2.0 s). A window starts every 72 frames (2.4 s, no overlap; the D14 ADVIO spacing), with the static filter < 0.20 m. A window is kept only if every frame has GT on both sides within 40 ms.<br>• **Speed axis:** mean \|ω\| over the window from the **GT orientation** (body-frame finite differences). The file's `angular_velocity_*` columns are not used: on seq131 they disagree with the GT orientation and with the factory-calibrated gyroscope (\|ω\| ratio 0.48, corr 0.76), whereas GT-orientation rates match the gyro (ratio 0.996, corr 0.997, median residual 0.04 rad/s against a median \|ω\| of 0.42).<br>• **Models and Q1:** π³, VGGT, MapAnything, every kept window. Q1 uses the P3 slope test (no GS twin), as for ADVIO and Waymo. ATE is primary; RPE, AUC@30, RRA and RTA are reported.<br>• **Known confound (written down before results):** ADT RGB exposure is 8–14 ms (seq131: 5th pct 8.3, median 9.4, max 14.0 ms), longer than the 5 ms readout, so motion blur grows with ω as fast as the RS shift does. An ADT slope therefore measures RS **and** blur together and cannot be attributed to RS alone. Context only, not counted: a log ATE regression on ω and on ω × mean exposure. The report must state this confound next to every ADT number.<br>• **Dense GT (secondary, reported, not decisive):** per-frame depth AbsRel and δ<1.25 against ADT depth. Use `depth_images_with_skeleton.vrs` when present (other people appear in the RGB), else `depth_images.vrs`. The depth is warped to the same 512×512 upright pinhole camera by nearest neighbour; only pixels with depth > 0 count. Model depth comes from its dense output with per-frame median scaling, and the slope of log AbsRel on ω is reported. Whether ADT depth is rendered RS-aware is UNVERIFIED; at 5 ms readout the mismatch is at most ω·2.5 ms·f (≈ 2 px at 3 rad/s).<br>• **Not run:** the GS SLAM cameras are grayscale 640×480 fisheye with different view directions, so they are not a twin. No extra models (D16 does not cover ADT).<br>• **Observed, not acted on:** gyro vs GT-orientation residual is smallest with a 5–10 ms shift (0.045 → 0.036 rad/s). The docs report sub-ms timecode sync, and gyro filter latency would give the same effect; GT stays at the documented capture time. | ADT is the only real RS dataset with a documented readout, mocap GT and dense depth (DATA_AUDIT). The fisheye must be undistorted for the pinhole models. Upright frames match the models' training data. The velocity-column problem and the blur confound were found in the data checks above, before any model was run on ADT. |
