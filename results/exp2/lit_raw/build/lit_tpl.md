# EXP2 literature review: rolling shutter and feed-forward 3D foundation models

**Date:** 2026-10-29
**Project:** RS-3D, rolling-shutter-aware feed-forward 3D (CVPR 2027 track)
**Companion files:** `related_work.csv` (152 unique papers), `DATA_AUDIT.md` and `datasets.csv` (36 datasets), `CHECKPOINT_2A.md` (one-page decision).

## How this document was made

- **Sources.** Six raw files in `results/exp2/lit_raw/`, written by earlier search agents on 2026-10-29:
  - `scoop.md/.csv` (topic 1);
  - `genre_models.md/.csv` and `models.csv` (topics 2 and 12, plus the models we run);
  - `rs2d_nerf_blur.md/.csv` (topics 3, 6, 9);
  - `classical_slam_gyro.md/.csv` (topics 4, 5, 7, 8);
  - `datasets.md/.csv` (topic 10);
  - `vggt_omega_claim.md` (topic 11).
- **No new papers.** Every paper here is in a raw file, and every page behind a claim is in the raw CSVs (`url_opened`). This document only synthesises them.
- **Merging.** The four paper CSVs hold 178 rows. Merging by normalised title gives **152 unique papers** (22 papers appeared in 2 or 3 slices). Where slices disagreed, the choice is listed in "Cross-slice contradictions" below.
- **Tables.** The per-topic citation tables are generated from `related_work.csv`. Titles, authors, venues and IDs are therefore exactly as in the raw files. Each paper appears in one table only, under its main topic.
- **Labels.**
  - *(measured)*: a number we measured in EXP0, EXP1 or on our VM.
  - *(reported)*: a number a paper or repo states. We did not reproduce it.
  - *(projected)*: a plan or a guess.
  - **UNVERIFIED**: the raw slice could not confirm it from a primary page.
- **Threat level.** The highest level any slice gave the paper. No slice rated any paper "high".
- **Code column in the tables.** "Y" + repo URL + the first licence named in the raw code field. That is usually the code licence; weights can differ. The full field is in `related_work.csv`.

---

## 1. Scoop check

**Question.** Has anyone already put rolling-shutter (RS) camera motion inside a feed-forward multi-view 3D foundation model (FM), from images only?

**Answer: no paper found** (scoop.md §1–2).

**What was searched** (scoop.md §3):
- **arXiv API.** Every paper with "rolling shutter" in its title or abstract since 2025-01-01 (40 papers), plus 52 adversarial queries such as `shutter AND VGGT`, `readout AND pointmap` and `"intra-frame" AND pose`.
- **Semantic Scholar.** Every abstract that cites VGGT, DUSt3R, MASt3R, π³ or MapAnything: 3,386 unique papers, **0** RS hits.
- **OpenReview API.** 8 search terms, 2,912 unique forums.
- **Venue title lists.** CVPR 2025, ICCV 2025, CVPR 2026, WACV 2026, ECCV 2026, 3DV 2025 and 3DV 2026. NeurIPS 2025/2026 and ICLR 2026/2027 were covered through OpenReview.
- **Web.** 37 web searches, plus full-text checks of 14 arXiv HTML pages.

**Findings**
1. **None of the 40 arXiv RS papers since 2025 combines RS with a feed-forward or foundation 3D model.**
   - The 5 papers from the last 60 days (v1 on or after 2026-08-30) are all irrelevant: forensics, registration theory, a pointing dataset and HDR imaging.
2. **2025–26 RS work falls into four groups, none of them feed-forward 3D:**
   - classical geometry (three ECCV 2026 papers);
   - per-scene optimisation;
   - 2D restoration;
   - forensics and astronomy.
3. **The closest work comes from three directions.** Each one misses at least one of our three ingredients: (i) RS, (ii) inside a multi-view feed-forward FM, (iii) images only.
   - **Images-only camera velocity from one image:** Image as an IMU, the Toronto RS course report, and SMARSC.
   - **Time-conditioned pointmaps, but for time between frames:**
     - Interp3R needs an event camera;
     - Point4Cast, Uni4R and V-DPM model scene dynamics.
   - **An FM that touches RS only outside the network:** Foundation4D (NVIDIA, arXiv 2610.30340, 21 Oct 2026). This is inside our 60-day window.
4. **RS plus a depth FM appears only as "correct the image, then run the FM"** (Anchor Points, UF 2026).
5. **No paper measures how much RS degrades feed-forward 3D FMs.**
   - Our EXP0 does: π³ fast-bin ATE ratio RS/GS = 2.61× (measured).
   - The nearest measurement studies are Verhoeven (ISPRS 2026, Agisoft Metashape) and a UAV SLAM benchmark that tests DUSt3R and MASt3R under blur, not RS.
6. **Abstract filters miss papers whose RS content is only in the body text.** The arXiv abstract scan missed Foundation4D; a full-text web search found it. This is the main residual risk.

**Positioning advice** (scoop.md §1.6)
- **Safe claim:** "first to model intra-frame (row-time) camera motion inside a feed-forward multi-view 3D foundation model, from images only".
- **Claims to avoid:**
  - "first to estimate camera velocity from images" (SMARSC 2019; Image as an IMU);
  - "first time-conditioned pointmap" (Interp3R; Point4Cast);
  - "first FM to consider RS" (Foundation4D).
- **Cite SMARSC for the pure-translation degeneracy**, next to our ω–v collinearity note.

{{TABLE:1}}

---

## 2. Same genre: adapting 3D FMs to non-pinhole cameras

**Findings** (genre_models.md §1 and §3)
- **No adaptation paper mentions RS.** The full texts of all 19 Topic-2 papers were searched for "rolling shutter": zero hits.
- **Accepted precedents in 2025–26:**
  - **RayTun3R** (NeurIPS 2026) keeps DA3, VGGT or π³ frozen. It fits a 10,752-parameter positional-encoding adapter online, per sequence. It reports 2–12× lower rotation error than the unadapted model and beating LoRA with ~14× fewer parameters (reported).
  - **Fisheye3R** (ECCV 2026) adds calibration tokens and masked attention to frozen VGGT, π³ and MapAnything. It trains on synthetically distorted perspective data.
  - **Wid3R** (arXiv) adds a camera-model token and fully fine-tunes π³. Its repo title claims ECCV 2026, which is **UNVERIFIED**.
  - **MEOW** (arXiv) fine-tunes MapAnything on a synthetic camera-manifold data engine, with no architecture change.
  - **Metadata conditioning** is established: Pow3R, MapAnything, OmniVGGT, WorldMirror, Pi3X, Rig3R (which even conditions on per-frame timestamps) and UniScale.
- **All of these model a static camera geometry or given metadata. None models motion within a frame.** Our per-frame (ω, v) with per-row unprojection is therefore new in this genre.
- **Rig3R is the closest "time-conditioned" FM.** But its time is per frame, and it is an input rather than a prediction.
- **VGGT-Ω §5:** auxiliary inputs hurt when added during pre-training but help when added only during fine-tuning (genre_models.md §3). This supports adding our motion state at the LoRA stage.
- **Baselines reviewers will ask for:**
  1. **A static PE or token adapter trained on RS** (RayTun3R or Fisheye3R style). Neither has released code: RayTun3R "will be made publicly available"; Fisheye3R's implementation is "Coming soon".
  2. **"Just fine-tune on synthetic RS"** (the MEOW stance). This is our LoRA-only control.
- **The genre also raises scoop risk.** It is the reason "RS next" is an obvious move for other groups.

### 2b. The backbones we run (frozen, images only)

From genre_models.md §4 and `models.csv`. The SHAs are the HF `sha` fields fetched on 2026-10-29. Input sizes are computed from each repo's own resize code for our 1280×1024 TUM frames.

| Model | HF id | Code licence | Weights licence | Input size for 1280×1024 | Rows cropped? | Fine-tune code |
|---|---|---|---|---|---|---|
| π³ | `yyfz233/Pi3` | BSD-3-Clause | **Conflict:** CC BY-NC 4.0 (README) vs bsd-2-clause (HF tag); treat as non-commercial | 560×448 (isotropic ×0.4375) | no | Y (`training` branch) |
| Pi3X | `yyfz233/Pi3X` | BSD-3-Clause | CC BY-NC 4.0 | 560×448 | no | partial |
| VGGT-1B | `facebook/VGGT-1B` | VGGT License | CC-BY-NC-4.0 | 518×420 (x ×0.4047, y ×0.4102) | no | Y (aggregator frozen by default) |
| VGGT-1B-Commercial | `facebook/VGGT-1B-Commercial` (gated) | VGGT License | commercial (gated) | 518×420 | no | Y |
| MapAnything | `facebook/map-anything` | Apache-2.0 | CC-BY-NC 4.0 | 518×392 | **yes:** 11 resized rows (≈27 original) top and bottom | Y (also wraps VGGT and π³ fine-tuning) |
| MapAnything-Apache | `facebook/map-anything-apache` | Apache-2.0 | Apache-2.0 | 518×392 | **yes** | Y |
| DA3 Nested G-L 1.1 / Giant 1.1 | `depth-anything/DA3NESTED-GIANT-LARGE-1.1`, `DA3-GIANT-1.1` | Apache-2.0 | CC BY-NC 4.0 | 504×406 | no | N (eval only) |
| DA3 Large 1.1 | `depth-anything/DA3-LARGE-1.1` | Apache-2.0 | **Conflict:** CC BY-NC 4.0 (README) vs apache-2.0 (HF tag); treat as non-commercial | 504×406 | no | N |
| DA3 Base / Small | `depth-anything/DA3-BASE`, `DA3-SMALL` | Apache-2.0 | Apache-2.0 | 504×406 | no | N |
| VGGT-Ω 1B | `facebook/VGGT-Omega` (gated) | FAIR Noncommercial | FAIR Noncommercial | 576×464 (balanced 512) | no | Y (released 2026-09-08) |
| CUT3R | none official (Google Drive) | CC BY-NC-SA 4.0 | not separately stated (UNVERIFIED) | 512×400 | **yes:** 5 resized rows (≈12.5 original) top and bottom | Y |
| StreamVGGT | `lch01/StreamVGGT` | CC BY-NC-SA 4.0 | HF tag cc-by-nc-4.0 | 518×420 | no | Y |

**What this means for us**
- **Row-time unprojection must carry each model's exact resize and crop.** MapAnything and CUT3R crop rows, so a row offset must be added before computing row time. The cheapest safeguard is to pre-resize frames ourselves to a size that each loader passes through unchanged.
- **Neither π³ nor VGGT ships LoRA code.** We add it ourselves (e.g. with `peft`). MapAnything's training harness already wraps VGGT and π³ fine-tuning, so it is the lowest-effort shared harness.
- **Commercially permissive weights** exist only for VGGT-1B-Commercial (gated), `facebook/map-anything-apache`, and DA3-BASE / DA3-SMALL (plus the monocular DA3 checkpoints). Everything else is non-commercial.
- **VGGT-Ω weights are gated.** Accepting the gate is a human step (HUMAN_TODO.md item 4).

{{TABLE:2}}

---

## 3. Learned 2D RS correction and unrolling

**Findings** (rs2d_nerf_blur.md §1–3)
- **No direct threat.** No learned 2D method models RS inside a feed-forward multi-view 3D model.
- **This is a mature, saturated 2D field (2017–2026).**
  - Most methods are consecutive-frame networks that take 2–3 neighbouring RS frames: DeepUnrollNet, SUNet, RSSR, CVR, AdaRSC, JAMNet, QRSC, DFRSC, LBCNet and SelfRSSplat.
  - They train on Carla-RS, Fastec-RS or BS-RSC.
  - They output one GS image at the first or middle scanline, or a GS video at any time.
- **Since 2022, new work needs special capture:** dual reversed-scan cameras, global-reset sensors, event cameras, or synchronised GS+RS rigs. TUM RS-VIO has none of these, so these methods cannot run on it.
- **Learned single-image methods are few:** Rengarajan 2017, Zhuang 2019, Deep Homography Mixture 2023 and RS-Diffusion (AAAI 2025). They are limited to rotation-only, homography or row-wise motion models.
- **2D networks bake in the training readout.**
  - RSSR/TPAMI: "the readout time ratio γ is set to 1 in the Carla-RS and Fastec-RS datasets".
  - IFED: "the generic RSC problem cannot be solved by using only consecutive frames".
  - TUM RS-VIO's ratio is about 0.60 (30.18 ms readout over a 50 ms frame interval at 20 Hz; my arithmetic from the raw facts). So zero-shot use risks mis-scaled unrolling.
- **Protocol for the 2D baselines.** This is the rs2d slice's own suggestion, not taken from a paper.
  - Run each 2D net in three modes:
    1. zero-shot, with its released weights;
    2. fine-tuned on our synthetic RS rendered with TUM-like readout and motion;
    3. with the GS target row matched to the 3D model's pose time.
  - Report how many neighbour frames each baseline consumed. Our method uses none.
- **Motivation quote.** RS-NeRF (ECCV 2024) says the 2D-correct-then-NeRF pipeline "can not fully explore the 3D scene's geometry, leading to inaccuracies in correspondences".

**Runnable 2D unrollers, ranked by the rs2d slice (§2).** Every weight link was resolved without sign-in on 2026-10-29.

| Rank | Method | Frames in | Output time | Weights | Licence | Caveats |
|---|---|---|---|---|---|---|
| 1 | DFRSC (CVPR 2024) | 2 or 3 | middle (`pos: middle`; `first` option) | GitHub release `ckpts`: `dfrsc_{2f,3f}_{bsrsc,fastecrs,carlars}.pth`, ~25 MB | none (no LICENSE file) | BasicSR, YAML configs. The README's Carla link is a typo; the real file is in the release. Torch not pinned. |
| 2 | JAMNet (CVPR 2023) | 2 | middle scanline of frame 2 | in repo: Carla-RS, Fastec-RS, BS-RSC, ~19 MB | "research purpose only", no licence | PyTorch 1.9.1 / CUDA 11.4 + detectron2. Real-time: 28 ms at 640×480 on an RTX 3090 (reported). |
| 3 | AdaRSC / BS-RSC (CVPR 2022) | 3 | centre frame, middle scanline | release v1.0.0: `adarsc_bsrsc.pth`, `adarsc_fastecrs.pth`, ~17 MB | **MIT** | Trained on real beam-splitter data. Needs the SimDeblur framework and a custom CUDA correlation layer. No Carla checkpoint. |
| 4 | LBCNet (TPAMI 2024 per project page); fallbacks RSSR (ICCV 2021), CVR (CVPR 2022) | 2 | any scanline time | in repo (bsrsc/carla/fastec); RSSR and CVR in public Google Drive folders | research-only, no licence | Constant-velocity model. RSSR/CVR need old custom CUDA ops on PyTorch 1.6. |
| 5 | DeepUnrollNet (CVPR 2020) | 2 | latest frame; code default first scanline (middle optional) | `download_pretrained_models.sh`: Carla and Fastec, ~15.6 MB | **MIT** | Legacy PyTorch 1.1.0 / CUDA 9.0 with three custom CUDA packages. Its dataset links are dead (404). |
| RS+blur | JCD (CVPR 2021) | 3 | sharp GS | `checkpoints.zip` (`JCD_BS-RSCD.tar`) | **MIT** | Only a BS-RSCD checkpoint is named; a Fastec one was not found. |
| optional | SUNet (ICCV 2021) | 2 | first scanline of frame 2 (loader default) | in repo, ~48 MB | no licence | Superseded by JAMNet and LBCNet. |

**Not runnable in our setting** (rs2d §2):
- QRSC: no weights.
- EvShutter and Neural Global Shutter: no code.
- RS-Diffusion: its weight repo is empty.
- UniINR: weights are behind a Microsoft login.
- SelfRSSplat: weights are on Baidu only.
- Deep Homography Mixture: `gen_bases` is missing from the repo.
- All event-based and dual-reversed methods: TUM RS-VIO has no events and no reversed-scan pair.

**Single-image 2D options with code and weights** (scoop slice):
- StableMotion: MIT, Hugging Face weights.
- Anchor Points / Stable-Derolling: Apache-2.0, Dropbox checkpoint.

{{TABLE:3}}

---

## 4. Classical RS geometry

**Findings** (classical_slam_gyro.md §1 and §4; scoop.md)
- **RS bundle adjustment is mature, and all of it is per-scene optimisation.**
  - Papers: Hedborg CVPR 2012, Saurer CVPR 2016, NW-RSBA, RSL-BA (ECCV 2024) and RS Camera Self-Calibration (ECCV 2026).
  - NW-RSBA is **CVPR 2023, not ICCV 2023** as the seed list said.
  - The NW-RSBA and RSL-BA releases are MATLAB demos on synthetic scenes.
- **Minimal solvers.**
  - Absolute pose: R6P (CVPR 2015), linear R6P (ACCV 2018) and R7Pf/R7Pfr (ECCV 2020). All are in `CenekAlbl/RnP` (BSD-3).
  - Relative pose: generalized epipolar geometry (CVPR 2016), single-scanline (ICCV 2025), asynchronous tracks (ICCV 2025 and 3DV 2026), and Barath (ECCV 2026).
- **Barath, "Rolling Shutter Relative Pose Estimation Made Practical", is our classical baseline.**
  - It solves for relative pose plus RS motion from 7 affine correspondences, with a ~1.2 ms solver inside RANSAC (reported).
  - It reports the best accuracy on the TUM RS benchmark (reported).
  - It is the only method in this list with a TUM RS run script (MIT). It is two-view only.
- **Degeneracies a reviewer will raise:**
  - Albl et al. ECCV 2016: parallel readout directions are critical.
  - Ito & Okatani CVPR 2017: critical motion sequences.
  - Zhuang et al. CVPR 2019: RS two-view geometry is degenerate under pure translation.
  - Handheld video has parallel readout and often pure translation. Our multi-view model inherits this in principle, so we should report pure-translation segments separately.
- **Off-the-shelf SfM has no RS support.** COLMAP (commit `98f3154`), GLOMAP (`99806d0`), TheiaSfM and OpenMVG all gave 0 grep matches.
  - RS-aware multi-view SfM exists only in commercial tools: Metashape Pro ("Enable rolling shutter compensation") and Pix4Dmapper ("Linear Rolling Shutter").
  - Kalibr calibrates RS line delay from a target video. It is not SfM.
- **The same message, for classical SfM.** Verhoeven (ISPRS 2026) measures that RS degrades exterior orientation and surfaces in Metashape, and that compensation helps. We make the same point for feed-forward FMs.
- **Theory that one RS image carries motion information:** Single-View RS SfM (arXiv 2026) and Order-One RS Cameras.

{{TABLE:4}}

---

## 5. RS in SLAM, VO and VIO

**TUM RS-VIO** (classical_slam_gyro.md §1; datasets.md). Citation: {{CITE:Rolling-Shutter Modelling for Direct Visual-Inertial Odometry}}.
- **Rig.** Two IDS uEye UI-3241LE-M-GL cameras. The left runs in GS mode and the right in RS mode, triggered together at 20 Hz. Images are 1280×1024 grayscale, 16-bit.
- **Readout.** The row time is 29.4737 µs (dataset page), so a frame reads out in 1024 × 29.4737 µs = 30.18 ms.
- **Other sensors.** Bosch BMI160 IMU at 200 Hz; OptiTrack Flex13 mocap at 120 Hz.
- **Sequences.** 10 sequences. The calibration sequences were recorded with both cameras in GS mode.
- **Timestamps.** The dataset page says "timestamps denote the middle of the exposure interval" and "the timestamps for the rolling-shutter images refer to the first row". datasets.md calls this wording ambiguous. We keep EXP0's choice: GT at the centre row, with first-row GT as the sensitivity check (DATA_AUDIT.md).
- **Licence.** Not stated (UNVERIFIED).

**Classical RS-aware odometry**
- Spline Fusion (BMVC 2013) and its IJCV 2015 version.
- Kerl et al. (ICCV 2015), RGB-D.
- RS-DSO (ECCV 2018, inferred from its DOI) and RS direct VIO (IROS 2019).
- Lee & Yoon RS-VO (arXiv 2017).
- Ctrl-VIO (RA-L 2022), with a TUM-RSVI config.
- Readout self-calibration inside VIO: Yang et al. 2022; Huai et al. 2021.
- Yang et al. (RA-L 2018) show that RS hurts monocular VO.

**Newer work (2024–26) needs an IMU or per-scene optimisation:** GS on the Move, TRGS-SLAM, Breaking Time, and Semantic SLAM with RS + INS.

**No learned feed-forward VO/SLAM that models RS from images alone was found.**

**Open-source VIO**
- VINS-Mono has a first-order RS option: `rolling_shutter: 1`, `rolling_shutter_tr`. It is used only when `estimate_td: 1`, and it needs an IMU.
- OpenVINS and Basalt have no RS readout model.

{{TABLE:5}}

---

## 6. RS in NeRF and 3DGS: per-scene optimisation vs our feed-forward model

**The precise difference**
- **Every RS-aware NeRF/3DGS method found is per-scene optimisation.** Each one needs:
  - many views of one static scene;
  - initial poses (COLMAP, or VIO with IMU velocities);
  - usually a known readout time, plus the exposure time for joint blur;
  - 20k–200k optimisation steps per scene (reported).
- **Our method is feed-forward** (projected design, not yet built):
  - a few unordered RGB frames in;
  - no poses, gyroscope, depth or video neighbours;
  - one forward pass at test time;
  - the LoRA is trained once, offline, not per scene.

| | Gaussian Splatting on the Move (ECCV 2024) | USB-NeRF (ICLR 2024) | URS-NeRF (ECCV 2024) | RS-NeRF (ECCV 2024) | **Ours (planned)** |
|---|---|---|---|---|---|
| Motion model | per-frame linear + angular velocity; per-pixel time from row *y* | continuous-time cubic B-spline in SE(3) | per-image pose + velocities | first- and last-row poses, linear interpolation in se(3), plus a smoothness term | per-frame (ω, v), predicted feed-forward |
| Runtime (reported) | per scene, 20k iterations; ≈5× slower than the Splatfacto baseline (A100) | per scene, 200K steps (RTX 3090) | per scene, coarse-to-fine (time not in abstract) | per scene, 200K iterations (RTX 4090) | one forward pass (projected) |
| Inputs | video or image set of one static scene; COLMAP (or VIO) poses and intrinsics; joint blur + RS needs known T_e and T_ro | ordered RS sequence; per-row capture time (e.g. 100 µs synthetic, ≈29.47 µs TUM-RS) | unordered RS images | forward-facing RS images, top-to-bottom scan | a few unordered RGB frames |
| Pose initialisation | yes: COLMAP keyframes, VIO | yes: COLMAP on RS images (failed on Fastec-RS) | yes, tolerant of inaccurate poses | yes; "may fail with completely random initialization" | **none** |
| Output | 3DGS + refined poses and velocities | NeRF + trajectory + GS video | NeRF + poses and velocities | NeRF + trajectory | pointmaps + poses (+ ω, v) |

**Other RS-aware 3D work** (all per-scene or per-subject, with known or initialised poses):
- 3DGUT (CVPR 2025): RS support in a 3DGS renderer. Apache-2.0; usable for rendering our synthetic RS data.
- NeuRAD and SplatAD: RS for driving sensors. Their venues are UNVERIFIED.
- Photoreal Scene Reconstruction from an Egocentric Device (SIGGRAPH 2025): calibrates RS with VIBA and models it in 3DGS.
- Light-field RS 2DGS (arXiv 2024).
- TRGS-SLAM (arXiv 2026).
- RS-Avatar (arXiv 2026): per-subject.
- 4D-GRT (NeurIPS 2025 workshop) and XSIM (arXiv 2026): RS data synthesis.

{{TABLE:6}}

---

## 7. Camera velocity from a single RS image, and SMARSC exactly

**Classical precedents** (classical_slam_gyro.md §1). These need a known 3D model or a specific scene structure.
- Ait-Aider et al. ECCV 2006: object pose and velocity from one RS view.
- Ait-Aider, Bartoli & Andreff CVPR 2007: kinematics from lines. The venue comes from the group page only.
- Magerand et al. ECCV 2012.
- Purkait et al. ICCV 2017: Manhattan world.
- Lao & Ait-Aider CVPR 2018: lines.
- Xue et al. ECCV 2026: velocity and spin of a patterned sphere (object, not camera).

**Learned**
- Rengarajan et al. CVPR 2017: 2-DOF row-wise motion (in §3).
- **SMARSC**, Zhuang et al. CVPR 2019: details below.
- Mo et al. CoRL 2021: single image plus IMU.
- **Image as an IMU** (ICCV 2025 Oral, MIT code): velocity from motion blur. Its paper says "we do not model for rolling shutter or non-uniform motion within exposure".
- **Rolling Shutter Image as an IMU** (Toronto course report, 2025): the same recipe for RS images. Not peer-reviewed; no code found.
- Anchor Points (UF 2026): fine-tunes a Marigold-based depth model under strong RS. Depth only.
- Roll2Depth (NeurIPS 2026): metric depth from RS stripe patterns under flickering light. It does not estimate camera motion.

**None of these is multi-view, and none outputs poses plus a global pointmap.**

### SMARSC exactly

Read from the CVF PDF by the classical slice (§2).

Citation: {{CITE:Learning Structure-And-Motion-Aware}}. No arXiv version found; no official code found.

- **Input:**
  - one RS image from an intrinsically calibrated camera;
  - real test images (Logitech C920) "resized and cropped to match the image size and intrinsic parameters of training data";
  - no IMU, and no readout time as input.
- **Predicts:**
  - Depth-Net (DispNet, regresses inverse depth): an RS depth map;
  - Velocity-Net (ResNet-34 plus conv layers, then a 1×1 conv to 6 outputs): **one constant 6-DOF velocity per image**, v ∈ R³ and ω ∈ so(3), in per-scanline units, with P_i = [exp((i−1)ω) | −(i−1)v].
- **Output:** a rectified image on the first-scanline GS canvas. Depth and velocity are used only to warp pixels.
- **Assumptions:**
  - constant velocity during the exposure;
  - static scene;
  - the training camera's row timing is built into the per-scanline units, so transfer to another readout needs rescaling (the slice's inference; the paper does not discuss it);
  - an ambiguity between pitch and vertical resizing, so images must be cropped, not resized.
- **Training:**
  - synthetic RS from KITTI Raw (stereo depth, sampled 6-DOF velocities, small random homographies);
  - L1 losses on inverse depth, v and ω, plus a photometric term;
  - the two networks are trained separately;
  - no fine-tuning on real data.
- **Evaluation:**
  - undistortion-flow error on 200 synthetic KITTI-derived images;
  - qualitative results on real C920 images;
  - a two-step SfM test (SMARSC, then VisualSFM) on 47 real RS images.
- **Theory:** RS two-view geometry is degenerate under pure translation.
- **Stated future work:** "exploit multiple views and consider other artifacts such as motion blurs".

### What RS-3D adds beyond SMARSC

The method items below are our plan, not results.
1. **Multi-view and feed-forward.** N frames are processed jointly, so cross-view parallax constrains (ω, v). This is the "multiple views" step that SMARSC names as future work.
2. **Geometry out, not a rectified image.** We output a global pointmap and a pose for every frame. SMARSC gives no inter-view poses and no shared 3D frame.
3. **Row-time-aware unprojection inside the 3D head.** There is no image resampling or inpainting, so there are no occlusion holes from a GS warp.
4. **Built on a pretrained 3D FM** (π³/VGGT class) with a light LoRA, not a CNN trained from scratch on synthetic KITTI. The backbone handles intrinsics, so we do not need to force the training crop.
5. **Real-data metric evaluation.** We score pose ATE/RPE and (ω, v) against mocap and gyro on TUM RS-VIO, with the GS-twin control. SMARSC's real-data results are qualitative only.
6. **Readout handled explicitly** (given or estimated), not baked into per-scanline units.
7. **Risk we inherit:** the pure-translation degeneracy. We report pure-translation segments separately.

**We do not claim** "first network to predict RS velocity". SMARSC (2019) did that for single images.

{{TABLE:7}}

---

## 8. Gyroscope-based RS correction and phone stabilisation

**"Use the gyroscope" has been published since 2010–2011** (classical_slam_gyro.md §1).
- **Karpenko et al., Stanford CSTR 2011-03** (read via Wayback). Citation: {{CITE:Digital Video Stabilization and Rolling Shutter Correction using Gyroscopes}}.
  - It models camera motion as rotation only: "We ignore translations because they are difficult to measure accurately using IMUs".
  - It auto-calibrates gyro drift and delay, focal length and RS speed from one capture.
  - It warps each row to a synthetic smooth GS camera, using a mesh with about 10 vertical subdivisions on the GPU, then crops.
- **Others follow the same pattern:**
  - Forssén & Ringaby (CVPR 2010) and Ringaby & Forssén (IJCV 2012);
  - Bell et al. (ECCV 2014);
  - the Pixel 2 "Fused Video Stabilization" blog (2017), which uses gyro rotation plus OIS lens motion, a mesh, and "multiple transformations for each frame";
  - Deep-FVS (WACV 2022).
- **Gyro-aided learned work:**
  - DeepOIS (TCSVT 2022);
  - RS-Diffusion, which uses the gyro only to build its GS ground truth;
  - Mo et al., which uses the IMU in pose refinement;
  - gyro- and inertial-aided RS relative pose (Lee et al. 2017 and 2019).
- **Gyroflow** is GPL-3.0. Its RS mechanism is undocumented (its docs page returned 404).

**Consequence for us.** EXP1 E1a is a Karpenko-style rotation-only per-row unroll driven by the gyro. It closes 82–99% of the fast-clip ATE gap (measured). It is therefore a **privileged baseline**, not a contribution. Our angle is images only.

**Stabilised phone video.** The points below are the classical slice's inference from these sources; no single source states them.
1. **EIS output is not global-shutter.** Rotation-induced RS is mostly removed. Translation-induced, depth-dependent RS stays.
2. **The effective intrinsics change per frame.** Each frame gets a mesh warp and crop toward a smoothed virtual rotation, plus a lens shift with OIS. The output rotation is the virtual one, not the true one.
3. **Row index no longer equals readout time** after the warp.
4. **For RS-3D:**
   - on stabilised footage, expect ω ≈ 0 apart from residual smoothing error, while v still matters;
   - evaluate on unstabilised capture (as in TUM RS-VIO), or list stabilisation as a separate condition.

{{TABLE:8}}

---

## 9. RS and motion blur

**One shared per-row model** (rs2d_nerf_blur.md §1 and §6). Every paper read uses it: row *y* integrates the scene over an exposure window T_e centred at t₀ + (y/H − ½)·T_ro.
- GS on the Move, Eq. 8: Δt_k(y) = ((k−1)/(N−1) − ½)·T_e + (y/H − ½)·T_ro.
- JCD: the virtual GS row time is t − t_m + i·t_r.
- IFED (RS-GOPRO): row exposure 1.0 ms, row readout 87 µs (reported).

**Regimes**
- **T_e ≪ T_ro: pure RS.** Examples: BS-RSC (1 ms exposure) and RS-NeRF's "infinitesimal exposure time for each row".
- **T_e comparable to T_ro: the two effects are coupled.** Example: BS-RSCD (16 ms RS exposure vs 2 ms GS).
- **TUM RS-VIO is in the pure-RS regime:** 2 ms exposure vs 30.18 ms readout. This is my inference from the raw facts and EXPERIMENTS_SO_FAR.md.

**2D methods**
- Su & Heidrich (CVPR 2015).
- Mohan et al. (ICCV 2017).
- JCD (CVPR 2021), the first learned RS+blur method, trained on BS-RSCD.
- With events: EvShutter (CVPR 2023), UniINR (ECCV 2024) and Neural Image Re-Exposure (arXiv 2023).
- Cross-shutter: dualBR (CVPR 2024) and Moment-Reenacting (TPAMI 2026).
- Wang et al., IEEE SPL 2024: UNVERIFIED.

**3D methods (all per-scene)**
- GS on the Move is the only RGB method found that handles both effects in 3D. Joint mode needs known T_e and T_ro.
- TRGS-SLAM is thermal and needs an IMU.
- BAD-NeRF models blur only.
- USB-NeRF and RS-NeRF model RS only.

**Blur-only work near FMs** (scoop): Image as an IMU, Unblur-SLAM (CVPR 2026, sub-frame poses for blur), RoMo-GS, DeblurSplat and AsyncEvGS.

**Implications**
- No feed-forward 3D model handles either effect.
- A blur extension for us would add exposure integration over T_e on top of the row time t(y). This is future work (projected).
- JCD is the blur baseline if we evaluate blurry sequences.

{{TABLE:9}}

---

## 10. Dataset survey (summary)

The full audit is in `DATA_AUDIT.md` and `datasets.csv` (36 datasets).

**Real RS data with true camera paths**
- **TUM RS-VIO** is still the only real dataset with an RS/GS twin on one rig and mocap GT. It is in hand and prepped (measured).
- **Aria Digital Twin** has:
  - real RS RGB, with 5 ms readout in the 1408×1408 mode and 16.26 ms at 2880×2880;
  - mocap synced to under 10 µs;
  - dense GT depth.
  - It needs a sign-up.
- **ADVIO** is real iPhone 6s RS video, but its GT is INS plus manual fixes, too coarse for RS-sized errors. It is in hand (measured) and serves as a stress test.

**Real RS data with true 3D**
- **Waymo** has the best-documented RS timing: per-image trigger, readout and shutter, scan direction, velocity, plus LiDAR. It needs a sign-up.
- **ScanNet++** has iPhone RS (presumed, not documented) and sub-mm laser scans. It needs an application.

**Synthetic RS sources:** Replica is in hand (measured) and TartanAir V2 is approved.

**GS "no harm" controls:** ETH3D SLAM is verified GS with mocap and is in hand (measured).

**Benchmarks the models use are often not GS.**
- Several benchmarks that VGGT, π³ and MapAnything use are phone or consumer video: CO3Dv2, RealEstate10K, ScanNet and ScanNet++.
- Their "GT" was computed from the same RS frames, so they cannot be GS controls. This is worth saying in the paper.

**2D RS datasets**
- The original Fastec-RS and Carla-RS links return 404. Fastec mirrors exist; no Carla mirror was found.
- BS-RSC, BS-RSCD and RS-GOPRO resolve.
- Most 2D RS datasets state no licence.

---

## 11. Claim verification: VGGT-Ω filters out severe RS

**Status: VERIFIED** (vggt_omega_claim.md).

**Source**
- The local `VGGT-Ω.pdf`, extracted with pypdf on 2026-10-29. PDF metadata: arXiv 2605.15195v1, DOI 10.48550/arXiv.2605.15195.
- Authors: Jianyuan Wang, Minghao Chen, Shangzhan Zhang, Nikita Karaev, Johannes Schönberger, Patrick Labatut, Piotr Bojanowski, David Novotny, Andrea Vedaldi, Christian Rupprecht (VGG Oxford and Meta AI).
- Venue: not stated in the PDF. The arXiv comments field says CVPR 2026 Oral (genre slice).

**Quote 1** (Appendix A.3 "VLM Prompt"). The STEP 1 header is on **PDF page 24** (printed page 24); item 3 is on **PDF page 25**.

> **A.3. VLM Prompt.** We use a VLM to discard videos not suitable for multi-view geometry. We prompt the VLM as follows: …
> STEP 1: CHECK FOR HARD REJECTION FACTORS ‘‘Assess the following. If ANY are present, the Classification is ‘REJECT HARD’’’: …
> 3. Extreme Visual Failure: Severe motion blur (edges lost), severe rolling shutter (wobble/jello effect), or corruption/glitching.

- **Same prompt, other items.** It also hard-rejects uncalibrated fisheye and 360° footage (item 5). It soft-rejects rotation-only clips.
- **Versions.** The genre slice confirms the text is in arXiv v1 and v2.
- **What it supports.** VGGT-Ω's video training data was filtered to remove *severe* RS, so the model was not trained to handle strong RS. Mild RS is not filtered.
- **What it does not support.** It does not show that VGGT-Ω fails on RS. It is a VLM judgement with no readout threshold, so the failure must be measured. That needs the gated weights (HUMAN_TODO.md item 4).
- **Our earlier note is now settled.** PROJECT_CONTEXT.md said "p. 25, re-verify the exact wording". The wording is verified: the header is on p. 24 and the item on p. 25.

**Quote 2** (PDF page 13), used for the EXP2 fine-tuning rules:

> Second, we found that the aleatoric uncertainty (confidence) loss can be unstable during fine-tuning. We therefore recommend initially removing this loss, particularly when the fine-tuning dataset is very small. If the downstream task requires confidence estimates, the uncertainty head can be fine-tuned separately.

---

## 12. Benchmarks and metrics used by the strongest 3D FM papers

From genre_models.md §5. These are the protocols as the papers report them.

| Paper | Datasets | Frames per test | Pose metrics | Geometry metrics | Alignment before scoring |
|---|---|---|---|---|---|
| VGGT (CVPR 2025) | Co3Dv2, RealEstate10K (pose); DTU (MVS); ETH3D (point map) | 10 random frames per scene | RRA, RTA per pair; **AUC@30** of min(RRA, RTA) | Accuracy, Completeness, Overall (Chamfer) | ETH3D: Umeyama Sim(3), official masks; DTU: MASt3R protocol |
| π³ (ICLR 2026) | RE10K, Co3Dv2; Sintel, TUM-dynamics, ScanNet; 7-Scenes, NRGBD, DTU, ETH3D; Sintel, Bonn, KITTI video depth; NYU-v2 | pose AUC: 10 random images, all pairs; trajectories follow CUT3R | RRA@30, RTA@30, AUC@30; **ATE, RPE-t, RPE-r** | Acc, Comp, NC (mean, median); AbsRel, δ<1.25 | trajectories Sim(3); point maps Umeyama Sim(3) + ICP; video depth per-sequence scale; mono depth per-frame scale |
| MapAnything (3DV 2026) | ETH3D, ScanNet++ v2, TartanAirV2-WB (+ two-view, calibration, RobustMVD) | 2–100 views, one covisible component, fixed seed | ATE RMSE, **AUC@5**, ray-direction error | rel (AbsRel), τ inlier ratio at 1.03% | both normalised by mean point distance in the view-0 frame; ATE after Horn rigid alignment; metric scale scored separately |
| DA3 (ICLR 2026 Oral) | HiRoom, ETH3D, DTU, 7Scenes, ScanNet++ | all images, or 100 random (fixed seed) | **AUC@3, AUC@30** | Acc, Comp, Chamfer, F1@d (0.05 m; 0.25 m for ETH3D) | RANSAC over evo/Umeyama pose alignment, then TSDF fusion |
| VGGT-Ω (CVPR 2026 Oral) | 7 Scenes, NRGBD, ETH3D; DyCheck, Sintel, TUM-Dynamic | 10 random frames per scene | **AUC@3, AUC@30** | AbsRel, δ1.25 | depth: per-frame median-based scale + shift; pose AUC needs no alignment |
| CUT3R (CVPR 2025 Oral) | KITTI, Sintel, Bonn, NYU-v2; TUM-dynamics, ScanNet; 7-Scenes, NRGBD | pose: 90-frame TUM/ScanNet clips; recon: 3–5 (7-Scenes) and 2–4 (NRGBD) frames | ATE, RPE trans, RPE rot | AbsRel, δ<1.25; Acc, Comp, NC | pose Sim(3); mono depth per-frame median; video depth per-sequence scale + metric |
| StreamVGGT (ICLR 2026) | 7-Scenes, NRGBD, ETH3D, Sintel, Bonn, KITTI, Co3D | CUT3R / VGGT conventions | Co3D pose AUC | Acc, Comp, NC; AbsRel, δ<1.25 | CUT3R / VGGT conventions |

**Protocol for our paper** (genre_models.md §1 and §5)
- **Trajectories.** ATE RMSE plus RPE-t and RPE-r after Sim(3) Umeyama, as in π³ and CUT3R. This matches our EXP0/EXP1 and pre-registration P2, where ATE is primary.
- **Pose AUC.**
  - Compute all-pairs RRA/RTA from the same N frames for every model.
  - Report AUC@30 for comparability.
  - Report AUC@3 (DA3, VGGT-Ω) or MapAnything's AUC@5 for sensitivity, since RS errors are small angles.
- **Frame counts.** Use both a 10-frame sparse setting (VGGT, π³, VGGT-Ω) and our contiguous 16-frame windows at the velocity bins.
- **Point maps** (synthetic data with GT): Acc, Comp and Chamfer after Umeyama Sim(3), plus ICP as in π³.
- **Depth.** AbsRel and δ<1.25, with per-frame median scaling or per-sequence scale. State which one we use.

**Caveat.** Several of these benchmarks are phone or consumer video, so their pose "GT" is not RS-free (§10; DATA_AUDIT.md).

{{TABLE:12}}

---

## Search log statistics

**Counts per slice**, as reported in the raw files.

| Slice | Topics | Web searches | arXiv API queries | Other searches |
|---|---|---|---|---|
| scoop | 1 | **37** | **63** (6 RS-scan + 5 index checks + 52 adversarial) | Semantic Scholar citation scans of 5 base models (2,117 + 2,103 + 1,293 + 387 + 399 citing papers; 3,386 unique); OpenReview API, 8 terms (2,912 unique forums); CVF title lists for CVPR 2025 (2,871), ICCV 2025 (2,701), CVPR 2026 (4,042) and WACV 2026 (831); ECCV 2026 accepted and virtual lists; 3DV 2025 (~142 rows) and 3DV 2026 (~177 rows); 14 arXiv HTML full-text checks; a partial OpenAlex pass |
| genre_models | 2, 12, models | **26** (T2: 13, models: 7, T12: 6) | **16** logged queries (several hit HTTP 429 and were retried) | HF API on 20 model ids plus 2 searches; 13 GitHub repo pages plus raw source files |
| rs2d_nerf_blur | 3, 6, 9, 2D datasets | **44** (T3: 20, 2D datasets: 8, T6: 9, T9: 7) | **12** (q11 not parsed: local disk full) | 30 GitHub repos read (READMEs, configs, loaders, trees); HF API 3; Zenodo 1; 5 project pages |
| classical_slam_gyro | 4, 5, 7, 8 | **50** (T4: 16, T5: 13, T7: 12, T8: 9) | **13** | GitHub search API, 4 queries; 8 shallow clones grepped; 2 raw-file checks; 15 README checks |
| datasets | 10 | **16** | **2** searches (51 and 46 hits) + 1 metadata lookup of 20 papers | pages for about 30 dataset groups; 2 Waymo devkit source files |
| vggt_omega_claim | 11 | 0 | 0 | pypdf text extraction from the local PDF |

**Counts per topic.** Web searches come straight from the logs. Which arXiv queries count toward which topic is my mapping.

| Topic | Web searches | arXiv API queries |
|---|---|---|
| 1 Scoop | 37 | 63, plus Semantic Scholar, OpenReview and venue lists |
| 2 Same genre | 13 | about 15 (genre log) |
| 3 2D RS correction | 20 (+ 8 on 2D RS datasets) | rs2d q1, q5–q8, q11–q12 |
| 4 Classical geometry | 16 | classical q1–q4 (+ q8, q10 general) |
| 5 SLAM/VO/VIO | 13 | classical q5 |
| 6 NeRF/3DGS | 9 | rs2d q2, q9, q10 |
| 7 Velocity from one image | 12 | classical q6, q9, q13 (+ q12 FM check) |
| 8 Gyro / stabilisation | 9 | classical q7, q11 |
| 9 RS + blur | 7 | rs2d q3, q4 |
| 10 Datasets | 16 (+ 8 in rs2d) | 2 + 1 metadata lookup |
| 11 Claim check | 0 | 0 (local PDF) |
| 12 Benchmarks | 6 (+ 7 on model releases) | 0 |
| **Total** | **173** | **~106** logged arXiv API queries (a few are metadata lookups) |

**Search limits the raw files report**
- The arXiv API and the arXiv website both return only 150 RS papers in total and 40 since 2025.
- About 2–6% of citing papers have no abstract in Semantic Scholar.
- Abstract filters cannot see body text (the Foundation4D case).
- Journals (TPAMI, TIP, RA-L, IJCV) were not scanned systematically, and Google Scholar was not accessible.
- Several venue web pages were truncated, blocked or rate-limited.

---

## UNVERIFIED items flagged in the raw files

**scoop**
- *Anchor Points* (UF 2026): no venue and no arXiv ID.
- *Rolling Shutter Image as an IMU*: a course report. Its PDF text was extracted crudely, and only the opening pages were readable.
- ICLR 2027 thermal RS submission: title and abstract come from the OpenReview API record only.
- XSIM (2602.05617): arXiv API metadata only.
- Point4Cast: full PDF text not checked.
- Venues taken from arXiv comments rather than proceedings: EAG3R, Fin3R, Uni4R, ProMo3D, Event3R, DeblurSplat, Lv et al., 4D-GRT and Unblur-SLAM.
- Image-as-IMU weights: whether the Google Drive link holds the full model is unclear.
- EAG3R: no licence, and weights come via a script.
- Code not checked: Fin3R, ProMo3D, Lv et al., XSIM and the UAV-SLAM benchmark.
- The OpenAlex pass is incomplete.

**genre_models**
- Wid3R's ECCV 2026 claim (repo title only).
- Weight-licence conflicts: π³ and DA3-LARGE-1.1. Both are treated as non-commercial.
- Licences not confirmed: CUT3R weights, OmniVGGT and Wid3R.
- VRAM not stated: π³/Pi3X, DA3 offline models, CUT3R and StreamVGGT.
- bf16 inference not confirmed: VGGT-Ω, CUT3R and StreamVGGT.
- VGGT's memory table predates the May 2026 memory fix.
- Code not found: RayTun3R, PanoVGGT, MEOW, CAM3R, VGGT-360, X-Lens, Rig3R, UniScale and LoRA3D.
- π³'s 7-Scenes/NRGBD strides: the values come from a secondary paper (DAGE, 2603.03744) and are not confirmed for π³.

**rs2d_nerf_blur**
- Wang et al., IEEE SPL 2024: authors and DOI.
- Venues from non-publisher pages: LBCNet TPAMI (project page), SelfRSSplat TCSVT (BibTeX), SelfUnroll IJCV (project page) and the URS-NeRF DOI (search result).
- NeuRAD and SplatAD are left as arXiv.
- RS-Diffusion: whether the 1.28 GB `RS_Real.pt` is a checkpoint.
- Links not testable: Baidu assets, UniINR weights and Unreal-RS.
- Carla-RS: no download found, and its resolution is unknown.
- Repos not opened: Naor, Mo, Praveen K, Lu (TVCG), Photoreal Egocentric, BAD-NeRF and 4DGRT.
- Output GS time not stated: Deep HM and RS-Diffusion.
- Not checked: whether JCD has a Fastec checkpoint, and whether the Deep HM code is complete.
- USB-NeRF's ≈29.47 µs/row figure for TUM-RS (frame rate and row count not checked there).

**classical_slam_gyro**
- Kim/Cadena/Reid ICRA 2016 and Li/Kim/Mourikis ICRA 2013 (title not confirmed).
- Venues from secondary pages: Ait-Aider CVPR 2007 (group page), RS Self-Calibration ECCV 2026 (group page), Barath ECCV 2026 (Google page) and RS-DSO ECCV 2018 (DOI-inferred).
- Aggregator only: Hedborg 2012 and Forssén 2010.
- Journal venues not confirmed: Huai 2021 and Yang 2022.
- Metashape mode names (from an unofficial manual).
- Code not found or not checked: SMARSC, Rengarajan, Purkait and Lao.
- Gyroflow's RS mechanism.
- Not checked: whether AsyncTrack needs known rotation, and what `single_scanline` contains.

**datasets** (details in DATA_AUDIT.md)
- TUM RS-VIO: licence, scan direction and timestamp wording.
- EuRoC: GS, sequence count, licence and timestamps.
- KITTI: shutter, licence and download.
- Shutter and readout not documented: ScanNet++ iPhone, ARKitScenes, ScanNet, CO3D, RE10K, 7-Scenes, TUM RGB-D and Bonn.
- ZJU-SenseTime, 100-Phones, PennCOSYVIO and LaMAR.
- ADT: licence text, whether its GT depth is RS-aware, total size, and 200 vs 236 sequences.
- ASE: whether RS is simulated.
- WHU-RSVI availability and the Kerl data.
- TartanAir V2: pose sampling for the 1000 Hz stream, and whether meshes are released.
- Mesh appearance: ScanNet++ and Matterport3D.
- 3D-FRONT terms.
- Waymo: size, version and per-image readout values.
- Several citations and DOIs.

---

## Cross-slice contradictions and how they were resolved

1. **Mo et al., *IMU-Assisted Learning of Single-View Rolling Shutter Correction*.**
   - Venue: scoop and rs2d say arXiv (2020); classical opened the PMLR page and says CoRL 2021 (PMLR vol 164, 2022). → **CoRL 2021 used.**
   - Code: scoop and rs2d say none; classical found `IRVLab/unrolling` (no LICENSE file). → **Y.**
2. **RS-Diffusion, *Single Image Rolling Shutter Removal with Diffusion Models*.**
   - Venue: rs2d opened the AAAI OJS page and says AAAI 2025; classical did not and left it as arXiv 2024. → **AAAI 2025.**
   - Code: rs2d found the repo; classical says none. → **Y, but the weights repo is empty.**
3. **Order-One RS Cameras.** CVPR 2025 per the CVF listing (scoop) vs arXiv 2024 (classical). → **CVPR 2025.**
4. **Su et al., Linear N-Point solver.** arXiv (scoop) vs ICCV 2025 per the CVF page (classical). → **ICCV 2025.**
5. **Hruby & Pollefeys, asynchronous-track minimal solvers.** 3DV 2026 per OpenReview (scoop) vs arXiv (classical). → **3DV 2026.**
6. **Xue et al., velocity and spin of spheres.** ECCV 2026 per the accepted list (scoop) vs arXiv (classical). → **ECCV 2026.**
7. **TRGS-SLAM code.** rs2d opened `umautobots/trgs_slam` (MIT); classical says the project page lists no repo; scoop says UNVERIFIED. → **Y (MIT).**
8. **Single-View RS SfM and Single-Scanline code.** classical found repos; scoop and rs2d saw no link on the arXiv page. → **Y**, with the notes kept in the CSV.
9. **Threat levels differ** for Barath, RS Self-Calibration, SMARSC, Mo, RS-Diffusion and GS on the Move. → **The higher level (med) is used.**
10. **RnGCam.** The rs2d CSV says arXiv; the scoop venue check lists it among ICCV 2025 CVF titles. → **Kept as arXiv, venue flagged UNVERIFIED.**
11. **Fastec-RS / Carla-RS download.** datasets.csv says "Google Drive links in README"; rs2d re-checked all four archives and they return HTTP 404. → **rs2d's finding used** (DATA_AUDIT.md).
12. **BS-RSC.** datasets.csv lists sensors, size and readout as "not on repo"; rs2d gives the paper's values (FLIR cameras, 1024×768, 25 fps, 80 × 50 frames). → **Paper values added, with attribution; readout still unknown.**
13. **GS on the Move venue.** datasets.md says no venue on the arXiv record; rs2d and classical give ECCV 2024 (ML Anthology and DOI). → **ECCV 2024.**
14. **VGGT-Ω venue.** vggt_omega_claim.md says the PDF states no venue; genre says CVPR 2026 Oral from the arXiv comments field. → **CVPR 2026 Oral (arXiv comments).**
15. **TUM RS-VIO timestamps.** classical presents the two dataset-page sentences as the convention; datasets.md calls them ambiguous. → **EXP0 choice kept** (centre row, with first-row sensitivity).
16. **Within one slice:**
    - Fastec-RS sequence counts: 56/20 (RSSR) vs 54/21 (SelfUnroll).
    - ADT sequence count: 200 vs 236.
    - The weight-licence conflicts for π³ and DA3-LARGE-1.1.
17. **Consistency checks that passed.**
    - TUM readout: 30.18 ms (datasets) ≈ 30.2 ms (classical) ≈ 1024 × 29.47 µs (USB-NeRF via rs2d).
    - ETH3D SLAM: our 61 downloaded training sequences match the audit's 56 mocap + 5 SfM-GT training sequences.
    - Replica: our measurement of `mesh.ply` confirms the audit's "quad mesh with vertex colours".
    - VGGT-Ω page numbers: PROJECT_CONTEXT's "p. 25" is now confirmed (header p. 24, item p. 25).

---

## Novelty verdict

**OPEN, 80% confidence** (scoop.md §2). No paper puts RS motion inside a feed-forward multi-view 3D FM from images only.

**The 20% residual risk:**
1. Unpublished concurrent work aimed at CVPR 2027. The obvious groups are Gallego (Interp3R), Clark (Image as an IMU), Hruby/Pollefeys/Kohn (RS geometry) and NVIDIA (Foundation4D).
2. Abstract filters miss RS mentioned only in body text, as Foundation4D shows.
3. The arXiv index looked small (150 RS papers in total).
4. Journals were not scanned.

**STOP rule** ("a paper already does our exact idea: RS motion inside a feed-forward multi-view 3D FM, images only"): **not triggered.** scoop.md §1 finding 1: "No paper found that does our exact idea."

**Nearest papers and exactly how we differ**

| # | Paper (exact citation from the raw files) | What it shares with us | Exactly how we differ |
|---|---|---|---|
| 1 | {{CITE:Foundation4D}} | Feed-forward multi-view FM; time-varying ego trajectory; RS exposure boundaries used to query poses | RS appears only in label generation and downstream 3DGS rendering. The network still predicts one pose and a GS depth/pointmap per image, for a calibrated driving rig. We put row time into the pointmap itself (per-row unprojection with per-frame ω, v), for generic handheld cameras, images only. |
| 2 | {{CITE:Image as an IMU: Estimating Camera Motion from a Single Motion}} | Images-only instantaneous ω, v of the camera | Single image; the signal comes from motion blur; it states it does not model RS; it outputs velocity only. We use RS row timing across many views inside a 3D FM and feed ω, v back into the geometry. |
| 3 | {{CITE:Rolling Shutter Image as an IMU}} | Images-only ω, v from RS distortion | Single image; two stages (RS→GS flow + mono depth, then least squares); needs the readout time; not peer-reviewed. We are end-to-end inside π³/VGGT with LoRA, with multi-view pose output. |
| 4 | {{CITE:Interp3R}} | Pointmap FM queried at an arbitrary time; joint depth + pose | Needs an event camera; its time is between frames, not within a frame's readout; zero RS mentions in its full text. We use images only and intra-frame row time. |
| 5 | {{CITE:Anchor Points}} | RS + a foundation model; a reference ("anchor") row time | Image-space de-rolling (Marigold U-Net) before a monocular depth FM; single view; no pose or velocity; needs RS-GS paired training data. We never undistort images, and we output multi-view poses. |
| 6 | {{CITE:Point4Cast}} | Time-conditioned pointmaps and cameras from a feed-forward model | Time indexes frames of a dynamic scene, and each frame is still treated as GS. We index rows inside a frame to absorb camera motion during readout. |

**Also cite and contrast in the same paragraph**
- **SMARSC**, the classical images-only "network predicts RS motion" prior and the pure-translation degeneracy (§7).
- **RayTun3R and Fisheye3R**, the same-genre static-camera adapters (§2).
- **Barath (ECCV 2026)**, the classical RS baseline on TUM RS (§4).

---

## Final baseline list

These are methods with working code that we will run. They come from the raw rankings (rs2d §2, classical §4, genre §3) plus our own EXP1/EXP2 code.

**A. 2D unroll, then run the 3D model.** Each is run zero-shot and fine-tuned on our synthetic RS, with the GS target row matched to the pose time.

| # | Method | Citation | Code | Licence |
|---|---|---|---|---|
| A1 | DFRSC | {{CITE:Rolling Shutter Correction with Intermediate Distortion Flow}} | https://github.com/ljzycmd/DFRSC (release `ckpts`) | none (no LICENSE file); research baseline only, ask before redistributing |
| A2 | JAMNet | {{CITE:Joint Appearance and Motion Learning}} | https://github.com/GitCVfb/JAMNet (weights in repo) | "research purpose only" statement, no licence |
| A3 | AdaRSC (BS-RSC) | {{CITE:Learning Adaptive Warping}} | https://github.com/ljzycmd/BSRSC (release v1.0.0) | MIT |
| A4 | LBCNet (fallbacks RSSR, CVR) | {{CITE:Learning Bilateral Cost Volume}}. Fallbacks: {{CITE:Inverting a Rolling Shutter Camera}}; {{CITE:Context-Aware Video Reconstruction}} | https://github.com/GitCVfb/LBCNet ; https://github.com/GitCVfb/RSSR ; https://github.com/GitCVfb/CVR | research-only statements, no licence |
| A5 | DeepUnrollNet | {{CITE:Deep Shutter Unrolling Network}} | https://github.com/ethliup/DeepUnrollNet (`download_pretrained_models.sh`) | MIT |
| A6 | JCD (blurry sequences only) | {{CITE:Towards Rolling Shutter Correction and Deblurring}} | https://github.com/zzh-tech/RSCD (`checkpoints.zip`) | MIT |

**B. Classical**

| # | Method | Citation | Code | Licence |
|---|---|---|---|---|
| B1 | COLMAP and/or GLOMAP on the RS frames (and on GS cam0) as the non-RS SfM reference | no paper in the raw files | repo URLs are not recorded in the raw slice; it cloned COLMAP at `98f3154` (2026-10-28) and GLOMAP at `99806d0` (2026-01-30). Neither has any RS support (0 grep matches) | COLMAP: BSD (new BSD); GLOMAP: BSD-style (ETH 2024) |
| B2 | Barath 7-AC RS relative pose | {{CITE:Rolling Shutter Relative Pose Estimation Made Practical}} | https://github.com/danini/rolling_shutter_made_practical (`examples/download_tum_rs.sh`, `examples/run_real_data.py`; the real-data path needs CUDA, torch, kornia, RoMa) | MIT |

**C. Ours and controls**

| # | Method | What it is | Code | Licence |
|---|---|---|---|---|
| C1 | Self-estimate two-pass (training-free) | Run the model on raw RS, estimate ω (and v) from its own neighbouring-frame poses, unroll, then run again | our repo (`src/tum_conditions.py` for TUM, `src/synth_conditions.py` for synthetic; GPU unroller `src/gpu_unroll.py`) | ours |
| C2 | Gyro straightening (privileged: needs a gyro) | EXP1 E1a: rotation-only per-row unroll from the gyro, Karpenko-style. It closes 82–99% of the fast-clip ATE gap (measured). Classical form: {{CITE:Digital Video Stabilization and Rolling Shutter Correction using Gyroscopes}} | our repo (`src/unroll.py`, `src/gpu_unroll.py`) | ours |
| C3 | LoRA-only control | Same backbone, same synthetic RS data, same LoRA, but no twist head and no row-time unprojection. Answers the MEOW-style "just fine-tune" objection. 3 seeds (pre-registration P9) | ours. Neither π³ nor VGGT ships LoRA code, so we add it (e.g. `peft`) | ours |

**Optional references, if time allows** (from the raw recommendations; not in the core set)
- Image as an IMU, per-frame ω/v comparison: MIT, https://github.com/jerredchen/image-as-an-imu.
- RnP R6P absolute pose + velocity against a GS-twin map: BSD-3, https://github.com/CenekAlbl/RnP.
- IMU-privileged RS-VIO:
  - Ctrl-VIO: GPL-3.0 per its README, no LICENSE file; https://github.com/APRIL-ZJU/Ctrl-VIO.
  - VINS-Mono with `rolling_shutter: 1`: GPL-3.0.
- Single-image 2D unrollers:
  - Stable-Derolling (Anchor Points): Apache-2.0, https://github.com/JacobACarter/Stable-Derolling.
  - StableMotion: MIT, https://github.com/ivowang/StableMotion.
- Metashape RS compensation: needs a commercial licence.
- RayTun3R- or Fisheye3R-style static adapter: no released code, so it would have to be reimplemented.

---

## Final dataset shortlist

Details, licences and timing are in `DATA_AUDIT.md`. Citations are copied from datasets.md §6. "In hand" facts were measured on our VM on 2026-10-29.

| Role | Dataset | Citation (datasets.md §6) | Status |
|---|---|---|---|
| Main real RS test (RS/GS twin, mocap) | **TUM RS-VIO** | Rolling-Shutter Modelling for Direct Visual-Inertial Odometry. D. Schubert, N. Demmel, L. von Stumberg, V. Usenko, D. Cremers. IROS 2019. arXiv:1911.01015; DOI 10.1109/IROS40897.2019.8968539 | **In hand + prepped:** 10 sequences + `dataset-calib-imu1`, ~22 GB extracted, 1124 kept windows, identical to EXP0 |
| GS–GS camera-difference control | TUM RS-VIO `dataset-calib-imu1` (both cameras GS) | as above | in hand |
| Real RS stress test (no GS twin, coarse GT) | **ADVIO** | ADVIO: An authentic dataset for visual-inertial odometry. S. Cortés, A. Solin, E. Rahtu, J. Kannala. ECCV 2018. arXiv:1807.09828 | **In hand:** 23 sequences; CC BY-NC 4.0 approved by the user |
| Synthetic RS with dense GT (meshes) | **Replica** | The Replica Dataset: A Digital Replica of Indoor Spaces. J. Straub, T. Whelan, L. Ma, Y. Chen, et al. (30). arXiv 2019. arXiv:1906.05797 | **In hand:** 18 scenes; vertex-coloured quad-mesh PLYs; approved under the Replica Dataset Research Terms |
| Synthetic RS (1000 Hz video, indoor + outdoor) | **TartanAir V2** | no formal citation on tartanair.org (credits W. Wang, Y. Hu, Y. Qiu, S. Shen, Y. Shaoul, CMU 2023). Related: TartanAir: A Dataset to Push the Limits of Visual SLAM. W. Wang, D. Zhu, X. Wang, Y. Hu, et al. (9). IROS 2020. arXiv:2003.14338 | approved by the user (CC BY 4.0); not in the 2026-10-29 download list |
| GS no-harm control (mocap) | **ETH3D SLAM** | BAD SLAM: Bundle Adjusted Direct RGB-D SLAM. T. Schöps, T. Sattler, M. Pollefeys. CVPR 2019 | **In hand:** 61 training sequences (mono + IMU); CC BY-NC-SA 4.0 approved |
| GS no-harm control (laser GT, used by all three models) | ETH3D MVS (optional) | A Multi-View Stereo Benchmark with High-Resolution Images and Multi-Camera Videos. T. Schöps, J. L. Schönberger, S. Galliani, T. Sattler, K. Schindler, M. Pollefeys, A. Geiger. CVPR 2017 | licence approved by the user; not in the 2026-10-29 download list |
| Real RS outdoor, per-image RS timing, LiDAR | **Waymo Open Dataset** | Scalability in Perception for Autonomous Driving: Waymo Open Dataset. P. Sun, H. Kretzschmar, X. Dotiwalla, A. Chouard, et al. (25). CVPR 2020. arXiv:1912.04838 | waiting on the user's sign-up (HUMAN_TODO item 1) |
| Real RS indoor, 5 ms readout, mocap, dense GT depth | **Aria Digital Twin** | Aria Digital Twin: A New Benchmark Dataset for Egocentric 3D Machine Perception. X. Pan, N. Charron, Y. Yang, S. Peters, T. Whelan, C. Kong, O. Parkhi, R. Newcombe, C. Y. Ren. ICCV 2023 (venue per projectaria.com link). arXiv:2306.06362 | waiting on the user's sign-up (HUMAN_TODO item 2) |
| Real phone RS + laser GT (readout to estimate) | **ScanNet++** | ScanNet++: A High-Fidelity Dataset of 3D Indoor Scenes. C. Yeshwanth, Y.-C. Liu, M. Nießner, A. Dai. ICCV 2023. arXiv:2308.11417 | waiting on the user's application (HUMAN_TODO item 3) |
| Optional GS controls | DTU (static capture); Argoverse 2 (GS driving counterpart to Waymo; terms-by-use, so a human decides) | Large Scale Multi-view Stereopsis Evaluation. Jensen et al. CVPR 2014; Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting. B. Wilson, W. Qi, T. Agarwal, J. Lambert, et al. (13). NeurIPS Datasets & Benchmarks Track (year not in record). arXiv:2301.00493 | not in the 2026-10-29 download list |

**Do not count as GS controls:** CO3Dv2, RealEstate10K, ScanNet / ScanNet-1500 and ScanNet++ iPhone; and, because their shutter is unknown, 7-Scenes, TUM RGB-D and Bonn.
