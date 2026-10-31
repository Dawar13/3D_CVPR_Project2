# EXP2 data audit

**Date:** 2026-10-29
**Sources:**
- `results/exp2/lit_raw/datasets.md` and `datasets.csv` (36 datasets, audited 2026-10-29). Every fact there comes from a page, paper, devkit file or licence file that the audit agent opened.
- The 2D RS dataset notes in `rs2d_nerf_blur.md` §4.
- Status facts measured on our VM on 2026-10-29, as reported by the main agent.

**Machine-readable version:** `results/exp2/datasets.csv` (36 rows, one per dataset). It adds `our_status` and `frame_to_gt_timing` to the raw fields.

**Rules**
- No agent registered, signed or accepted any licence.
- Licences listed as approved below were approved by the user on 2026-10-29 (see `HUMAN_TODO.md`).
- Anything not confirmed from an official source is marked **UNVERIFIED**.
- Anything that has to be read from the downloaded files themselves is marked **"to verify from files (main agent)"**.

---

## 1. What we have now (measured on our VM, 2026-10-29)

| Dataset | Status | Facts (measured) | Licence and approval |
|---|---|---|---|
| **TUM RS-VIO** | re-downloaded and prepped | 10 sequences + `dataset-calib-imu1` (both cameras GS: the GS–GS control), ~22 GB extracted. Prep gives **1124 kept windows, identical to EXP0**. | Not stated on the dataset page (UNVERIFIED). Public download, no sign-up. Confirm before redistributing anything. |
| **Replica** | downloaded | 18 scenes. `mesh.ply` is a binary PLY with per-vertex xyz, normals and uchar RGB colours, and polygon (quad) faces. Example: `room_0` has 954,492 vertices and 953,647 faces. | *Replica Dataset Research Terms* (non-commercial research/education). **Approved by the user.** |
| **ADVIO** | downloaded | 23 sequences | CC BY-NC 4.0. **Approved by the user** for research use. |
| **ETH3D SLAM** (training set) | downloaded | 61 sequences, mono + IMU. This matches the audit's training split: 56 mocap + 5 SfM-GT training sequences. | CC BY-NC-SA 4.0. **Approved by the user** for research use. |
| **Waymo Open Dataset** | waiting | — | Needs the user's Google sign-in and the *Waymo Dataset License Agreement for Non-Commercial Use* (HUMAN_TODO item 1) |
| **Aria Digital Twin** | waiting | — | Needs the user's email form and the ADT licence (HUMAN_TODO item 2) |
| **ScanNet++** | waiting | — | Needs the user's account and application (HUMAN_TODO item 3) |
| VGGT-Ω weights (a model, not a dataset) | waiting | — | Gated on Hugging Face, FAIR Noncommercial (HUMAN_TODO item 4) |
| **TartanAir V2** | approved, not in the 2026-10-29 download list | — | CC BY 4.0. **Approved by the user.** No sign-up (`pip install tartanair`). |

The user approved the Creative Commons datasets **ETH3D, ADVIO and TartanAir V2** for research use. The approval names ETH3D as a whole. The ETH3D MVS part has the same licence (CC BY-NC-SA 4.0) and has not been downloaded.

---

## 2. Shortlist and the role of each dataset

| Role | Dataset | Why | Main weakness |
|---|---|---|---|
| Main real RS test | **TUM RS-VIO** | Real RS/GS twin on one rig, mocap GT, known readout (30.18 ms), already in our pipeline | 10 indoor sequences, grayscale, licence unstated |
| GS–GS control | TUM `dataset-calib-imu1` | Both cameras in GS mode: measures pure camera difference | one sequence |
| Real RS stress test | **ADVIO** | Real iPhone 6s RS, 60 fps, indoor + outdoor, dynamic people | GT is INS + manual fixes (too coarse for RS-sized errors); readout unknown; no GS twin, so Q1 uses the slope test (pre-registration P3) |
| Synthetic RS with dense GT | **Replica** | Best photometric quality; vertex-coloured meshes in hand; we choose readout and scan direction | Indoor only; full appearance needs the PTex HDR textures |
| Synthetic RS (alternative) | **TartanAir V2** | CC BY 4.0, 65 environments, 1000 Hz RGB video for row-stitching, GT depth and pose | 1000 Hz pose sampling UNVERIFIED; no meshes |
| GS no-harm control | **ETH3D SLAM** | Verified GS rig ("All cameras on this rig use a global shutter and are synchronized"), mocap GT | Timestamp convention not stated |
| Real RS outdoor | **Waymo** (pending) | Per-image trigger, readout and shutter times, scan direction, velocity; LiDAR GT; RS-aware devkit projection | Sign-up; vehicle pose, not camera pose |
| Real RS indoor, dense GT | **Aria Digital Twin** (pending) | Documented readout (5 ms in 1408×1408 mode), mocap synced <10 µs, dense GT depth, meshes | 5 ms readout gives a small RS effect except in fast head turns |
| Real phone RS + laser GT | **ScanNet++** (pending) | Sub-mm laser GT; MapAnything's own test set | RS not documented, so readout must be estimated; pseudo-GT is one pose per frame |
| Optional GS controls | ETH3D MVS, DTU, Argoverse 2 | Verified GS or static capture; Argoverse 2 is the GS driving counterpart to Waymo | Not downloaded; Argoverse 2 is terms-by-use, so a human decides |

---

## 3. All 36 datasets

Generated from `datasets.csv`. Long fields are cut at the first " - " or " ("; the CSV has the full text.

{{DSTABLES}}

---

## 4. Frame-to-GT timing per dataset

**Why this matters.** An RS frame has no single capture time. The GT pose must be looked up at the reference row the method assumes, or a constant timing error is added.

### TUM RS-VIO (in hand)
- **Dataset page (verbatim):** "Note that for the calibration sequences, both cameras were operating in global-shutter mode. This means the timestamps for the rolling-shutter images refer to the first row. In general, timestamps denote the middle of the exposure interval."
- The audit calls this wording ambiguous. The "first row" sentence follows a sentence about the calibration sequences.
- **Our choice (as in EXP0):**
  - RS GT is taken at the **centre-row time** = first-row stamp + 511.5 × 29.4737 µs = **first-row stamp + 15.08 ms** (`src/common.py`, `RS_MID_OFFSET`).
  - **Sensitivity check:** RS GT at the first-row stamp.
  - In EXP0, the first-row choice raised the median RS ATE by +1.6% (MapAnything), +2.6% (VGGT) and +10% (π³) (measured). The conclusions were unchanged.
- **GS cam0:** GT at the stamp (middle of exposure).
- **Readout:** 1024 rows × 29.4737 µs = 30.18 ms. The row time comes from the dataset page, which says it is "very well approximated by the step size of the exposure time".
- **Scan direction:** not stated by the dataset. Our pipeline assumes row 0 is read first (top-to-bottom), as in EXP0/EXP1.
  - The EXP1 checks agree with this. The gyro-predicted rotation matched the true rotation to within 0.13° of 4.45°, and the unrolled fastest frame lines up with the GS frame (measured).
  - But it is our assumption, not a documented fact.

### ADVIO (in hand)
- **What the raw audit says:**
  - all sensors are synced;
  - iPhone stamps are converted to "time interval since device boot";
  - the exposure reference point is not stated;
  - readout is unknown and scan direction is unknown;
  - video is 1280×720 portrait at 60 fps;
  - GT is at 100 Hz (INS + manual position fixes).
- **To verify from files (main agent):**
  - which row (first, middle) or which exposure point the frame stamp refers to;
  - the frame timestamps and the GT timestamps share a clock (check the files);
  - the sensor readout direction in the **stored** portrait frame. A portrait video may have been rotated relative to the sensor, so readout could run along stored columns. This is my inference from "portrait"; it must be checked;
  - a readout estimate, e.g. from a per-device line-delay fit or a known iPhone 6s value. No value is in the raw files.
- **Stride.** Pre-registration P10 requires the stride for a 60 fps dataset to be logged in `DEVIATIONS.md` before running.

### ETH3D SLAM (in hand)
- **Global shutter** ("All cameras on this rig use a global shutter and are synchronized"), so there is no row timing.
- **What the raw audit says:** the timestamp convention is not stated; camera and IMU nanosecond stamps are converted to mocap time via `sequence_calibration.txt`.
- **To verify from files (main agent):**
  - whether image stamps mark exposure start or middle;
  - exactly how `sequence_calibration.txt` maps camera time to mocap time.
- **Frame rate.** About 27.1 Hz; the `sfm_` sequences run at 13.6 Hz. The stride must be logged in `DEVIATIONS.md` before running (P10).

### Replica and TartanAir V2 (synthetic; we render)
- **Our generator** (`src/rsgen/core.py`, `generate.py`):
  - the frame time is the **centre-line time**;
  - the GS twin is rendered at the same centre-line instant;
  - readout = 0 gives a GS frame;
  - we choose readout and scan direction.
- **Measured checks** (`results/exp2/generator/validation_testroom.json`):
  - readout 0 reproduces GS exactly;
  - known-point reprojection has median error ~6e-14 px for the tb, bt and lr scans.
- **TartanAir V2:** RS would be made by row-stitching the 1000 Hz `lcam_front` video at 1 ms resolution. How poses are given for the 1000 Hz stream is UNVERIFIED.

### Pending datasets (for when access arrives)
- **Waymo (devkit, verbatim logic):**
  - readout = `camera_readout_done_time − camera_trigger_time − shutter`;
  - t(row) = `camera_trigger_time + 0.5·shutter + readout·row/H` for TOP_TO_BOTTOM;
  - other directions follow the `rolling_shutter_direction` enum;
  - the proto calls the image centre time the "middle of exposure of middle row";
  - `pose` is the vehicle pose at `pose_timestamp`, so the camera pose needs the calibration extrinsic;
  - sync error is bounded in [−6 ms, 7 ms] at 99.7% (paper).
- **Aria Digital Twin:**
  - t(p) = t_image + (row(p)/H − 0.5)·readout, so the image stamp is the middle row;
  - readout is 5 ms in the 1408×1408 mode;
  - sensor-row orientation vs the stored image must be checked;
  - whether the GT depth is rendered RS-aware is UNVERIFIED.
- **ScanNet++:** timing is not documented. Readout must be estimated, and the pseudo-GT is one pose per frame.

---

## 5. Do not count as GS controls

These benchmarks are used by VGGT, π³ or MapAnything but are **not** RS-free. Their "GT" poses were estimated from the same frames, so they already absorb RS (datasets.md §1D).

| Dataset | Used by | Capture | Why not a GS control |
|---|---|---|---|
| CO3Dv2 | VGGT, π³ (pose AUC) | crowd-sourced smartphone turn-table videos | RS presumed (phones); COLMAP GT on the same frames |
| RealEstate10K | VGGT, π³ (pose AUC) | ~10k YouTube videos | shutter unknown, likely mostly RS; SLAM + BA GT on the same frames |
| ScanNet / ScanNet-1500 | VGGT (matching), π³ (pose), MapAnything (Robust-MVD) | iPad Air2 colour + Structure sensor | RS presumed, not documented; BundleFusion GT |
| ScanNet++ iPhone | MapAnything | iPhone 13 Pro | RS presumed, not documented; one pseudo-GT pose per frame |
| 7-Scenes | π³ (point maps) | handheld Kinect | shutter unknown |
| TUM RGB-D (TUM-dynamics) | π³ (pose) | Kinect | shutter unknown |
| Bonn RGB-D Dynamic | π³ (video depth) | RGB-D sensor, not named | shutter unknown |

**Conditional (use only after a human checks):**
- **EuRoC:** GS only per secondary sources; the IJRR paper returned 403.
- **KITTI:** the official page does not state the shutter type.

**Verified GS or static:** ETH3D low-res rig and ETH3D SLAM; TUM RS-VIO cam0; Argoverse 2; NRGBD, Sintel and TartanAir (rendered); and, because of static capture, DTU, ETH3D high-res and Matterport3D.

---

## 6. Licences and access

**Agents accepted no licence.** The user's decisions (2026-10-29, `HUMAN_TODO.md`):
- **Approved:** Replica (Research Terms); ETH3D (CC BY-NC-SA 4.0); ADVIO (CC BY-NC 4.0); TartanAir V2 (CC BY 4.0).
- **The user will sign up:** Waymo, Aria Digital Twin, ScanNet++ and the VGGT-Ω weights.
- **Not needed:** HM3D, Matterport3D and 3D-FRONT (Replica covers the mesh need); ARKitScenes (ScanNet++ covers it).

**Licence notes**
- **TUM RS-VIO has no licence statement.** The TUM RGB-D CC BY 4.0 notice does not obviously cover it. Ask before redistributing derived data.
- **All approved licences are non-commercial or attribution licences.** Record attribution in the paper: CC BY for TartanAir V2; CC BY-NC for ADVIO; CC BY-NC-SA for ETH3D, which is also share-alike; the Research Terms for Replica.
- **Terms-by-use datasets need a human decision before download** (downloading counts as acceptance): ARKitScenes, 7-Scenes, Argoverse 2 and Replica (Replica is now approved).

---

## 7. 2D RS datasets (for the 2D-unroller baselines only)

From rs2d_nerf_blur.md §4. Only Carla-RS (per-pixel depth and velocity, via its loader) and Unreal-RS (GT poses) carry any 3D or pose GT. Link status was checked on 2026-10-29.

| Dataset | Real / synthetic | GT | Link status (2026-10-29) | Licence |
|---|---|---|---|---|
| Fastec-RS | real GS video → synthetic RS; γ = 1 | GS first and middle scanline; no depth or pose | **original GDrive 404**; mirrors: JCD repo `fastec_rs_lmdb.zip` (resolves), HF `yunfanlu/UNIINR-Fastec-Simulated-Dataset` (68 GB, not inspected) | not stated |
| Carla-RS | synthetic (CARLA) | GS first and middle; per-pixel RS depth, flow, occlusion mask, GT velocity (loader) | **original GDrive 404; no mirror found** | not stated |
| BS-RSC | real, beam splitter; 1024×768, 25 fps, 1 ms exposure | GS aligned to the RS middle scanline | GDrive `BS-RSC.zip` resolves | not stated (code MIT) |
| BS-RSCD | real, beam splitter, RS + blur (16 ms RS vs 2 ms GS exposure) | sharp GS at the middle of the RS exposure | GDrive resolves | not stated (code MIT) |
| RS-GOPRO | synthetic dual reversed; row exposure 1.0 ms, row readout 87 µs | 9 GS frames per pair | GDrive resolves | not stated |
| Gev-RS / Gev-RS-Real | synthetic RS + events / real RS + events | GS (synthetic only) | Baidu, unreachable | "research and education only" |
| DRE | real RS + events | no GS GT | GDrive folder resolves | not stated |
| RS-Real (RS-Diffusion) | real phone, gyro-made GS (rotation only) | GS | HF `Yzl-code/RS-Diffusion` public | none on the HF card |
| realBR | real GS-blur + RS + high-speed GT | sharp frames | GDrive resolves | not stated (code MIT) |
| RSGR-GS | real RSGR/GS pairs | GS | **link 404** | CC BY-NC-ND 4.0 badge |
| GS on the Move smartphone set | real phones (iPhone 15 Pro, Galaxy S20 FE, Pixel 5) | no GT (COLMAP + VIO) | Zenodo 10848124, direct | CC BY-SA 4.0 |
| Unreal-RS (USB-NeRF) | synthetic; GT poses | GS first and middle row | SharePoint redirects to a login | not stated |

**The readout ratio matters.** RSSR/TPAMI sets γ = 1 for Carla-RS and Fastec-RS. TUM RS-VIO's ratio is ~0.60 (30.18 ms over a 50 ms frame interval; my arithmetic). So 2D nets trained on these sets need fine-tuning on our synthetic RS, or a zero-shot caveat.

---

## 8. Open items

**To verify from files (main agent)**
1. ADVIO: frame-stamp reference row or exposure point; clock alignment of frames and GT; sensor readout direction in the stored portrait frame; a readout estimate.
2. ETH3D SLAM: exposure reference point of the image stamps; the `sequence_calibration.txt` mapping to mocap time.
3. Replica: whether the PTex HDR textures (`textures/*-color-ptex.hdr`) are present in the download. The raw audit says full-quality appearance needs them and the habitat-sim or ReplicaSDK PTex renderer.
4. TUM RS-VIO: the downloaded README's timestamp wording, against the page quote above.

**UNVERIFIED in the raw audit**
- **TUM RS-VIO:** licence; scan direction.
- **EuRoC:** GS, sequence count, licence and timestamps.
- **KITTI:** shutter, licence and download.
- **Shutter and readout not documented:** ScanNet++ iPhone, ARKitScenes, ScanNet iPad, CO3D, RE10K, 7-Scenes, TUM RGB-D and Bonn.
- **ZJU-SenseTime ISMAR19:** licence, sequence count and RS statement.
- **100-Phones:** GT method, and whether the Apache-2.0 notice covers the data.
- **PennCOSYVIO:** licence and download.
- **LaMAR:** licence and GT accuracy.
- **ADT:** licence text; whether GT depth is RS-aware; total size; 200 vs 236 sequences.
- **ASE:** whether RS is simulated.
- **Availability:** WHU-RSVI (URL 404) and the Kerl et al. data (no download found).
- **TartanAir V2:** pose sampling for the 1000 Hz stream, and whether meshes are released.
- **Mesh appearance:** ScanNet++ (vertex colour vs texture) and Matterport3D (textures).
- **3D-FRONT:** Terms of Use text.
- **Waymo:** size, current version and per-image readout values.
- **Citations not opened:** 7-Scenes, TUM RGB-D, KITTI and ZJU-SenseTime; DOIs for ADVIO and 100-Phones.

**Cross-slice corrections applied here**
- **Fastec-RS / Carla-RS download.** The datasets slice listed "Google Drive links in README"; the rs2d slice found all four archives return HTTP 404. The rs2d finding is used.
- **BS-RSC details.** These were missing in the datasets slice and were filled from the rs2d slice (paper values). The readout is still unknown.
- **GS on the Move venue.** The datasets slice saw no venue on arXiv; rs2d and classical give ECCV 2024.
