# RS-3D lit review, slice `rs2d_nerf_blur`

Slice scope: Topic 3 (learned 2D rolling-shutter correction and unrolling) plus 2D RS datasets, Topic 6 (RS in NeRF / Gaussian splatting, per-scene) and Topic 9 (joint RS and motion blur).
Compiled 2026-10-29. Companion table: `rs2d_nerf_blur.csv` (50 rows, one per paper, written with Python `csv`).

How the pages were checked:
- **arXiv pages** were opened with WebFetch. Exact metadata (authors, comments field, journal-ref) was cross-checked against the arXiv API.
- **CVF open-access pages** returned HTTP 403 to WebFetch, so they were opened with Python `urllib` (HTTP 200). Title, authors, pages and abstract were parsed from the HTML. The CSV `verified` column marks these rows.
- **GitHub repos** were opened in three ways: WebFetch of the repo page, raw README/LICENSE/config files from `raw.githubusercontent.com`, and the GitHub API (license, tree) until the API hit its rate limit.
- **Weight and dataset links** were resolved with HTTP requests on 2026-10-29. Status codes are reported below.
- A venue is given only where the arXiv comments field or a venue page states it. The source of each venue claim is in the CSV `venue` column.

---

## 1. Plain-English findings

1. **No direct threat found in this slice.** These searches found no learned 2D or NeRF/3DGS paper that models RS inside a feed-forward multi-view 3D model.
   - The closest work falls into two groups.
   - The first group is per-scene optimisers that model per-frame or per-row motion: USB-NeRF (ICLR 2024), URS-NeRF (ECCV 2024), RS-NeRF (ECCV 2024) and Gaussian Splatting on the Move (ECCV 2024). These need many views of one static scene, pose initialisation (COLMAP and/or VIO), and minutes to hours of optimisation per scene.
   - The second group is single-image 2D CNNs that predict depth plus inter-scanline camera motion: Zhuang et al. CVPR 2019, and Mo et al. 2020, which also uses an IMU. These are the closest conceptual precursors to "predict per-frame ω, v from images". They output a corrected 2D image, not multi-view pointmaps and poses.
   - Our extra query (`abs:"rolling shutter" AND (abs:depth OR abs:pointmap OR abs:"feed-forward")`, and a web search on "rolling shutter DUSt3R VGGT") returned nothing RS-aware in the DUSt3R/VGGT family.
2. **Learned 2D RS correction is a mature, saturated 2D field (2017–2026).**
   - Most methods are *consecutive-frame* networks (2–3 neighbouring RS frames). They are trained on Carla-RS, Fastec-RS or BS-RSC, and output one GS image at the first or middle scanline, or a GS video at arbitrary time. Examples: DeepUnrollNet, SUNet, RSSR, CVR, AdaRSC, JAMNet, QRSC, DFRSC, LBCNet and SelfRSSplat.
   - Since 2022, new work has moved to *special capture*: dual reversed-scan cameras (IFED, SelfDRSC, SelfDRSC++, HaRP), global-reset sensors (Neural Global Shutter), event cameras (EvUnroll, EvShutter, SelfUnroll, UniINR), or synchronised GS+RS rigs (dualBR, Moment-Reenacting).
   - *Learned single-image* methods are few (Rengarajan 2017, Zhuang 2019, Deep Homography Mixture 2023, RS-Diffusion AAAI 2025). They are limited to rotation-only, homography or row-wise motion models.
3. **2D networks bake in the training readout.**
   - The RSSR/TPAMI paper states "the readout time ratio γ is set to 1 in the Carla-RS and Fastec-RS datasets".
   - IFED argues that "the generic RSC problem cannot be solved by using only consecutive frames". Two cameras with different readout times can produce identical RS views ("correction ambiguity").
   - So running these nets zero-shot on TUM RS-VIO, whose readout ratio is below 1, risks mis-scaled unrolling. USB-NeRF's text gives ≈29.47 µs/row for TUM-RS; I did not verify the frame rate or row count here. For a fair "2D unroll → 3D" baseline, fine-tune the 2D nets on our synthetic data with the TUM readout, or at least report both zero-shot and fine-tuned results.
4. **Usable baselines exist** (ranked in §2). DFRSC, JAMNet, AdaRSC, LBCNet/RSSR/CVR and DeepUnrollNet all have code and downloadable weights. JCD is the RS+blur option.
   - Several popular methods are **not** runnable. QRSC has no weights. EvShutter and Neural Global Shutter have no code. RS-Diffusion's weight repo is empty. UniINR's weights are behind an institutional OneDrive login. SelfRSSplat's weights are Baidu-only.
   - All event-based and dual-reversed methods need hardware that TUM RS-VIO lacks.
5. **Dataset availability is a real problem.**
   - The original Fastec-RS and Carla-RS Google Drive links in the DeepUnrollNet README return **HTTP 404**. All four archives were re-checked in both `open?id=` and `file/d/` form.
   - Fastec-RS survives as an LMDB mirror in the JCD repo, and as a 68 GB Hugging Face backup posted by the UniINR author.
   - I found **no working Carla-RS mirror**. That matters because Carla-RS is the only 2D RS set with per-pixel depth, GT velocity, flow and occlusion masks.
   - BS-RSC, BS-RSCD and RS-GOPRO links work. None of the 2D RS datasets states a data license, except EvUnroll ("research and education only") and RSGR-GS (CC BY-NC-ND 4.0 badge, link dead).
6. **Per-scene vs our feed-forward setting.** See §5. In short, per-scene RS methods need:
   - a dense image set of one static scene,
   - initial poses (COLMAP, or VIO + IMU velocities),
   - usually a known readout time (and exposure time for joint blur),
   - 20k–200k optimisation steps per scene.

   Our method needs a few unordered RGB frames, with no poses, gyroscope, depth or video neighbours, and one forward pass.
7. **Joint RS + blur (Topic 9).**
   - **Common model.** Every paper read uses the same per-row model: row *y* integrates the scene over an exposure window T_e centred at t₀ + (y/H − ½)·T_ro.
     - Gaussian Splatting on the Move, Eq. 8: Δt_k(y) = ((k−1)/(N−1) − ½)·T_e + (y/H − ½)·T_ro.
     - JCD: the virtual GS row time is t − t_m + i·t_r.
     - IFED (RS-GOPRO): row exposure 1.0 ms, row readout 87 µs.
   - **Regimes.**
     - When T_e ≪ T_ro, the image is pure RS: BS-RSC (1 ms exposure), and RS-NeRF's "infinitesimal exposure time for each row".
     - When T_e is comparable to T_ro, the two effects are coupled: BS-RSCD has a 16 ms RS exposure against a 2 ms GS exposure.
   - **In 3D, only per-scene methods model both effects.** Gaussian Splatting on the Move needs known T_e and T_ro for simultaneous compensation. TRGS-SLAM is thermal and needs an IMU.
   - **Learned 2D joint methods** either need neighbouring frames (JCD) or extra sensors: events (UniINR, EvShutter) or a GS+RS pair (dualBR, Moment-Reenacting).
   - No feed-forward 3D model handles either effect, as far as these searches found.

---

## 2. Ranked runnable baselines (2D unroll → 3D model), plus fine-tuning on our synthetic data

Every candidate below has public code and weights. I resolved each weight link myself without signing in on 2026-10-29.

| Rank | Method | Why this rank | Caveats |
|---|---|---|---|
| **1** | **DFRSC** (Cao et al., CVPR 2024), https://github.com/ljzycmd/DFRSC | Newest consecutive-frame state of the art with a **complete** checkpoint set. GitHub release `ckpts` holds `dfrsc_{2f,3f}_{bsrsc,fastecrs,carlars}.pth`: direct download, HTTP 200, ~25 MB, no sign-in. The 2-frame variant suits short clips. It is built on BasicSR and YAML-config driven, so fine-tuning on our synthetic RS data only needs a dataset class. The target row is configurable (`pos: middle  # first`). | No LICENSE file: fine for research baselines, but ask before redistribution. The README's Carla link points to the Fastec file, a typo; the real `dfrsc_3f_carlars.pth` is in the release. Torch version not pinned. |
| **2** | **JAMNet** (Fan et al., CVPR 2023), https://github.com/GitCVfb/JAMNet | Weights for **Carla-RS, Fastec-RS and BS-RSC are inside the repo** (`deep_unroll_weights/model_weights/*/pre_net_flow.pth`, ~19 MB), so a `git clone` is enough. Needs only **2 frames**, the smallest neighbour requirement among strong methods. Outputs the middle-scanline GS of frame 2. Real-time (28 ms at 640×480, RTX 3090). | "Research purpose only" statement, no license. Pinned to PyTorch 1.9.1 / CUDA 11.4 and needs detectron2. |
| **3** | **AdaRSC / BS-RSC** (Cao et al., CVPR 2022, seed), https://github.com/ljzycmd/BSRSC | **MIT license**, the cleanest legally. Release v1.0.0 has `adarsc_bsrsc.pth` and `adarsc_fastecrs.pth`, direct download, ~17 MB. Trained on *real* beam-splitter data. 3 frames in, centre-frame middle-scanline GS out. README documents how to plug in custom data. | Needs 2 neighbours. Depends on the SimDeblur framework and a custom CUDA correlation layer. No Carla checkpoint. |
| **4** | **LBCNet** (Fan et al., TPAMI 2024 per project page), https://github.com/GitCVfb/LBCNet. Fallbacks: **RSSR** (ICCV 2021) and **CVR** (CVPR 2022) | Produces GS at **any scanline time** from 2 frames. We can match whatever time convention our pose GT uses (first row vs mid-exposure), and can unroll to several row times. LBCNet weights for bsrsc/carla/fastec are in the repo. RSSR and CVR weights are in public Google Drive folders (both resolve). | Explicit constant-velocity model. Research-only statement, no license. RSSR/CVR need old custom CUDA ops (correlation, forward-warp, cupy 8.2) on PyTorch 1.6. |
| **5** | **DeepUnrollNet** (Liu et al., CVPR 2020, seed), https://github.com/ethliup/DeepUnrollNet | The canonical reference baseline, MIT-licensed. `download_pretrained_models.sh` pulls Carla and Fastec weights via direct Google Drive `uc` links (both HTTP 200, ~15.6 MB, no sign-in). | Legacy stack: PyTorch 1.1.0 / CUDA 9.0 with three custom CUDA packages, so expect porting work. Its dataset links are dead (404). Code default target is the *first* scanline GS of the latest frame (`load_middle_gs=False`). |
| (RS+blur) | **JCD** (Zhong et al., CVPR 2021, seed), https://github.com/zzh-tech/RSCD | Use if we evaluate blurry sequences. MIT. `checkpoints.zip` (contains `JCD_BS-RSCD.tar`) is a public Google Drive file. Its `video_inference.py` runs on a raw video. | 3 frames. Only a BS-RSCD checkpoint is named in the README; a Fastec checkpoint was not found. |
| (optional) | **SUNet** (Fan et al., ICCV 2021, seed), https://github.com/GitCVfb/SUNet | Runnable: weights in repo for Carla and Fastec (~48 MB). | Superseded by JAMNet and LBCNet from the same group. PyTorch 1.6 with custom CUDA ops. No license. |

**Not runnable in our setting:**
- **QRSC**: no checkpoints on master, and the referenced old commit has no checkpoint folder.
- **EvShutter**: the repo has only a README.
- **Neural Global Shutter**: README-only repo, and the dataset link is 404.
- **RS-Diffusion**: the weight repo `Lhaippp/RS-Diffusion` on Hugging Face has no checkpoint.
- **UniINR**: weights sit on an HKUST-GZ SharePoint that redirects to a Microsoft login.
- **SelfRSSplat**: Baidu only, unreachable from here.
- **Deep Homography Mixture**: weights are available, but `model_8block.py` imports `gen_bases`, which is absent from the repo tree.
- **All event-based and dual-reversed/RSGR methods**: TUM RS-VIO has no events and no reversed-scan pair.

**Protocol suggestion (my opinion, not from a paper):**
- Evaluate each 2D net in three modes: (a) zero-shot with its released weights, (b) fine-tuned on our synthetic RS set rendered with TUM-like readout and motion, and (c) with the GS target row matched to the 3D model's pose timestamp.
- Report how many neighbour frames each baseline consumed, since our method uses none.

---

## 3. Topic 3 per-method details (inputs, weights, readout/scan, output time)

"Weights trained on" lists only checkpoints I saw in the repo, the release or the README.

| Method | Input | Released weights trained on | Readout / scan assumptions | Output time | Code / weights / license / framework |
|---|---|---|---|---|---|
| Rengarajan CVPR'17 | 1 RS image | n/a (no code found) | row-wise camera motion; top-to-bottom implied | **first row** ("back to the time of first-row exposure") | none found |
| Zhuang CVPR'19 | 1 RS image | n/a | learns depth + inter-scanline motion; training data synthesised geometrically | corrected GS (time not checked) | none found by search |
| Mo et al. arXiv'20 | 1 RS image **+ IMU** | n/a | row-wise pose, IMU refinement | not checked | none linked |
| Praveen K et al. arXiv'21 | 1 RS image | n/a | per-row pose, cubic polynomial trajectory | not checked | none linked |
| Deep HM ICCV'23 | 1 RS image | RS-Homo (synthetic) per README | homography mixture over Gaussian-weighted scanline blocks; motion bases from natural videos | UNVERIFIED | GDrive weights resolve; no license; code possibly incomplete |
| RS-Diffusion AAAI'25 | 1 RS image | RS-Real (weights not found) | **rotation-only**; readout not modelled; gyro needed only to build GT | not stated in paper (checked v2 HTML) | Apache-2.0; HF weight repo empty |
| DeepUnrollNet CVPR'20 | **2 consecutive** RS frames | Carla-RS, Fastec-RS | learned; datasets have γ = 1, top-to-bottom | GS of **latest** frame; code default **first** scanline (middle optional) | MIT; GDrive direct; PyTorch 1.1.0 |
| SUNet ICCV'21 | 2 consecutive | Carla-RS, Fastec-RS (in repo) | learned; γ = 1 training data | "intermediate time of these two frames"; loader default `load_middle_gs=False`, i.e. first scanline of frame 2 | no license; PyTorch 1.6.0 |
| RSSR ICCV'21 / TPAMI'23 | 2 consecutive | Carla-RS, Fastec-RS (GDrive) | **constant velocity** (TPAMI adds constant acceleration); γ = 1 | any scanline; supervised by middle-scanline GS | research-only; PyTorch 1.6.0 |
| CVR CVPR'22 | 2 consecutive | Carla-RS, Fastec-RS (GDrive; RSSR, CVR, CVR*) | bilateral motion model (approx. constant velocity) | any t in (0,1); evaluation at t = 0.5 and 1.0 | research-only; PyTorch 1.6.0 |
| AdaRSC CVPR'22 | **3 consecutive** | BS-RSC, Fastec-RS | learned; BS-RSC real FLIR readout (value not stated) | centre frame, **middle** scanline | MIT; GitHub release; SimDeblur |
| JAMNet CVPR'23 | 2 consecutive | Carla-RS, Fastec-RS, BS-RSC (in repo) | learned; "vertical flip + inverse order" augmentation | **middle** scanline of frame 2 | research-only; PyTorch 1.9.1 + detectron2 |
| QRSC ICCV'23 | multi-frame (`n_inputs`) | none released | **quadratic** (constant-acceleration) per-pixel solver; uses RAFT/GMA flow | middle (`load_middle_gs: true`) | no weights; MMFlow |
| DFRSC CVPR'24 | 2 or 3 consecutive | BS-RSC, Fastec-RS, Carla-RS (2F and 3F) | learned non-linear distortion flow (GS→RS, backward warp) | **middle** (`pos: middle`; `first` option) | no license; GitHub release; BasicSR |
| LBCNet TPAMI'24 | 2 consecutive | BS-RSC, Carla-RS, Fastec-RS (in repo) | symmetric and asymmetric bilateral motion model | arbitrary time | research-only; PyTorch 1.9.1 |
| SelfRSSplat TCSVT'25 | 2 consecutive (trained on triplets, self-supervised) | Baidu only (not checked) | cycle-consistent re-synthesis; GMFlow + SoftSplat | arbitrary time (11-frame demo) | research-only; PyTorch 1.9.1 |
| Naor et al. ECCV'22 | whole video | n/a | test-time internal optimisation + external training on small synthetic RS/GS set | not checked | not checked |
| IFED ECCV'22 | **dual reversed** (t2b + b2t) simultaneous pair | RS-GOPRO (GDrive folder) | explicit readout coupling; trained at 87 µs/row | GS sequence (5 or 9 frames) | MIT; PyTorch 1.11 |
| Neural Global Shutter CVPR'22 | RSGR (global reset) video | none | RSGR sensor mode | GS video | no code |
| SelfDRSC ICCV'23 / ++ PR'26 | dual reversed pair | RS-GOPRO (self-supervised) | distortion time maps | arbitrary intermediate times | MIT / none; PyTorch ≥ 1.7 |
| EvUnroll CVPR'22 | 1 RS image **+ events** | Gev-RS (GDrive weights) | events bridge row times | consecutive GS frames | code no license; data research-only; torch 1.7.0 |
| EvShutter CVPR'23 | 1 RGB RS (± blur) **+ events** | none | no constant-velocity assumption | GS | no code |
| SelfUnroll (arXiv'23 / IJCV'25) | 1 RS (+events), or 2 RS + events | Fastec, Gev, Gev-real, DRE (GDrive) | line delay calibrated (Kalibr) for DRE | arbitrary timestamps | no license; PyTorch 1.11 |
| UniINR ECCV'24 | 1 RS-**blur** frame **+ events** | weights behind login | exposure time embedded | arbitrary frame rate | weights not anonymously downloadable |

---

## 4. 2D RS datasets

Link status was checked 2026-10-29. "Direct" means the public link resolves without sign-in. Google Drive folders resolve publicly but may need a browser or gdown to bulk-download.

| Dataset (paper) | Real / synthetic | Sensor | GT: GS / depth / pose / readout | Size | License | Download |
|---|---|---|---|---|---|---|
| **Fastec-RS** (DeepUnrollNet, CVPR'20) | real GS video → **synthetic RS** (rows copied from consecutive GS frames) | Fastec TS5 high-speed GS, 2,400 fps, 640×480 (per SelfUnroll paper) | GS **first and middle** scanline; no depth or pose; γ = 1 per RSSR-TPAMI | 56 train / 20 test seqs per RSSR; SelfUnroll's text says 54 / 21; ~2 GB train + ~930 MB test (DUN README) | not stated | **Original GDrive links 404.** Mirrors: JCD repo `fastec_rs_lmdb.zip` (GDrive, resolves); HF `yunfanlu/UNIINR-Fastec-Simulated-Dataset` (public, ungated, 68 GB, HF tag apache-2.0, labelled a backup by the UniINR README; contents not inspected) |
| **Carla-RS** (DeepUnrollNet, CVPR'20) | **synthetic** (CARLA / Unreal; 6-DoF motion close to constant velocity) | virtual RS camera | GS first and middle; **per-pixel RS depth (`_rs.pdepth`), optical flow, occlusion mask, GT velocity (`gt_vel.log`)** per the DUN loader; no per-row pose file seen; γ = 1 | 210 train / 40 test seqs; ~10 GB train + ~2 GB test | not stated | **Original GDrive links 404; no mirror found.** Synthesis code: github.com/ethliup/UnrealCV (not opened) |
| **BS-RSC** (Cao et al., CVPR'22) | **real**, beam splitter | RS: FLIR FL3-U3-13S2C (CMOS); GS: FLIR GS3-U3-28S4C (CCD); 1024×768, 25 fps, 1 ms exposure (no blur) | GS aligned to RS **middle scanline**; no depth, pose or readout value | 80 videos × 50 frames (50/15/15) | not stated (code MIT) | Direct: GDrive `BS-RSC.zip` resolves |
| **BS-RSCD** (JCD, CVPR'21) | **real**, beam splitter, **RS + blur** | same FLIR pair; 15 fps; 1280×920 → half-res (640×480 in table, 640×460 in text); RS exposure 16 ms, GS 2 ms | sharp GS aligned to the middle of the RS exposure; per-row offset t_r defined, no value given | 80 seqs × 50 frames (50/15/15) | not stated (code MIT) | Direct: GDrive `rscd.zip` and `rscd_lmdb.zip` resolve |
| **RS-GOPRO** (IFED, ECCV'22) | **synthetic**, dual reversed | GOPRO 240 fps + GoPro HERO9, interpolated to 15,360 fps (RIFE); 960×540 | each t2b/b2t pair has 9 consecutive GS frames; **row exposure 1.0 ms, row readout 87 µs** | 50/13/13 seqs; RS 3554×2 / 945×2 / 966×2 | not stated (contains GOPRO-derived data) | Direct: GDrive `RS-GOPRO_DS.zip` resolves; Baidu mirror |
| **Gev-RS** (EvUnroll, CVPR'22) | **synthetic** RS + simulated events (DVS-Voltmeter) | Phantom VEO 640 GS at 5,700 fps, 640×360 | GS frames; `rs_blur` and `rs_sharp` folders; no depth or pose | 29 seqs (20/9), 3,700 triplet clips | "research and education only, commercial use prohibited" | Baidu Pan (code `evun`). **Could not reach** from our network; Baidu usually needs an account (UNVERIFIED) |
| **Gev-RS-Real** (EvUnroll) | real RS + events | hybrid rig (DAVIS346 / Prophesee Gen4 per SelfUnroll) | **no GS GT** | 16 seqs (9/7) | as Gev-RS | Baidu |
| **DRE** (SelfUnroll) | real RS + events | DAVIS346, 346×260 | no GS GT; line delay calibrated | 100 seqs (75/25) | not stated | GDrive folder `DRE_dataset` resolves (README still says "coming soon") |
| **RS-ERGB** (EvShutter, CVPR'23) | real events + RS (optional blur); GS via simulator | not checked | GS | not checked | n/a | **Not released** (repo has README only) |
| **RS-Real** (RS-Diffusion, AAAI'25) | real phone (gyro via Android HAL) | not named | GS made from RS by gyro, **rotation-only** homographies | 40k train / 1k test | none on HF card (code Apache-2.0) | Direct: HF `Yzl-code/RS-Diffusion` (public, ungated, ~42k files) |
| **RS-Homo** (Deep HM, ICCV'23) | synthetic from natural videos | n/a | GS / homography | not checked | none | generation scripts in a GDrive folder (resolves) |
| **RSGR-GS** (Neural Global Shutter, CVPR'22) | real, paired RSGR/GS | custom optical system | GS | not checked | CC BY-NC-ND 4.0 (README badge) | **link 404** |
| **realBR** (dualBR CVPR'24 / Moment-Reenacting TPAMI'26) | real triaxial GS-blur + RS + high-speed GT | custom rig | high-speed sharp frames | not checked | not stated (code MIT) | Direct: GDrive `realBR.tgz` resolves |

RS datasets with 3D ground truth, from Topic 6 (useful for our evaluation or training):
- **Unreal-RS** (USB-NeRF). Synthetic. Provides RS images, GS at the first and middle row, GT poses and COLMAP poses. Hosted on Westlake SharePoint, which **redirected to a login**.
- **RS-NeRF synthetic and real** (forward-facing). The GDrive folder resolves.
- **Gaussian Splatting on the Move smartphone set.** Zenodo 10848124, **CC BY-SA 4.0**, 1.9 GB, direct download with no login. Includes Spectacular Rec raw data with VIO.
- **Gaussian Splatting on the Move synthetic set.** Zenodo 10847884; the README says CC BY-SA 4.0 / CC BY 4.0. I did not open that record.
- **WHU-RS** (URS-NeRF). Baidu only.
- **RS-ZJU** (avatars, 2026). Release not stated.

---

## 5. Topic 6: RS in NeRF / 3DGS, and how per-scene differs from our feed-forward setting

| | Gaussian Splatting on the Move (ECCV'24) | USB-NeRF (ICLR'24) | URS-NeRF (ECCV'24) | RS-NeRF (ECCV'24) | **Ours (planned)** |
|---|---|---|---|---|---|
| Motion model | per-frame linear + angular velocity; screen-space linear pixel velocity v′ = −J(ω×μ + v); per-pixel time from row *y* | continuous-time **cubic B-spline in SE(3)** (linear interpolation as ablation) | per-image pose + velocities | first and last row poses, linear interpolation in se(3), plus a smoothness regulariser | per-frame (ω, v) predicted feed-forward |
| Runtime | per-scene, 20k iterations; full method ≈ 5× slower than the Splatfacto baseline (A100) | per-scene, 200K steps (RTX 3090) | per-scene, coarse-to-fine (time not stated in the abstract) | per-scene, 200K iterations (RTX 4090) | one forward pass; LoRA trained once |
| Inputs needed | a video or image set of one static scene; COLMAP (or VIO) poses and intrinsics; VIO velocities optional (zero init); **joint blur + RS needs known T_e and T_ro** (README) | ordered RS sequence; per-row capture time (per-dataset readout values, e.g. 100 µs synthetic, ≈29.47 µs TUM-RS) | unordered RS images | forward-facing RS images; top-to-bottom scan; infinitesimal row exposure | a few unordered RGB frames; no gyro, depth or neighbours |
| Known poses / init | yes: COLMAP keyframe poses, VIO | yes: spline knots from **COLMAP on RS images** (COLMAP failed on Fastec-RS) | yes, but tolerant of inaccurate poses (epipolar checks) | yes: middle-row or COLMAP init; "may fail with completely random initialization" | **none** |
| Output | 3DGS + refined poses and velocities | NeRF + trajectory + GS video at any frame rate | NeRF + poses and velocities | NeRF + trajectory | pointmaps + poses (+ ω, v) |

Other RS-aware 3D work found:
- **3DGUT** (CVPR'25) supports RS in a 3DGS renderer.
- **NeuRAD** and **SplatAD** (arXiv; venues unverified) handle RS for driving sensors. SplatAD uses a custom gsplat fork.
- **Photoreal Scene Reconstruction from an Egocentric Device** (SIGGRAPH'25) calibrates RS with VIBA and models it in 3DGS.
- **Light-field RS 2DGS** (arXiv'24).
- **TRGS-SLAM** (arXiv'26) is thermal-inertial and handles RS and blur.
- **Scanline-Aware Gaussian Avatars** (arXiv'26) is per-subject.
- **4DGRT** (NeurIPS'25 workshop) generates RS and fisheye camera-effect data, which could help build our synthetic set.

All of these are per-scene or per-subject optimisation with known or initialised poses.

RS-NeRF's text makes the motivation for our work explicit. It says the 2D-correct-then-NeRF pipeline "can not fully explore the 3D scene's geometry, leading to inaccuracies in correspondences".

---

## 6. Topic 9: joint RS and motion blur (exposure time vs readout time)

**2D methods:**
- **Su & Heidrich (CVPR'15)** use a global camera trajectory, and each scanline is deblurred with its own motion segment. This is the classic coupled model.
- **Mohan et al. (ICCV'17)** do unconstrained RS blind deblurring for wide-angle lenses and irregular trajectories.
- **JCD (CVPR'21)** is the first learned RS+blur method, trained on BS-RSCD (RS exposure 16 ms vs GS 2 ms; virtual GS row time t − t_m + i·t_r).
- **EvShutter (CVPR'23)** deblurs first with a flow-based module, then corrects RS, using events.
- **UniINR (ECCV'24)** recovers sharp GS at any time from one RS-blur frame plus events, with exposure time embedded.
- **Neural Image Re-Exposure (arXiv'23)** re-exposes images using events.
- **dualBR (CVPR'24)** uses an RS image to disambiguate GS blur decomposition.
- **Moment-Reenacting (TPAMI'26)** builds dual-shutter blur-RS pairs, including stereo.
- **Wang et al., IEEE SPL'24**, does event-based unrolling and deblurring. UNVERIFIED: no publisher page could be opened.

**3D methods:**
- **Gaussian Splatting on the Move** is the only RGB method found that handles both effects in 3D: N_blur samples over T_e, shifted by (y/H − ½)·T_ro, with optimised velocities. It is per-scene and needs known T_e and T_ro for joint mode.
- **TRGS-SLAM** blends sharp renders with precomputed weights covering RS and the thermal time constant.
- **BAD-NeRF (CVPR'23)** models blur only, through the exposure trajectory.
- **USB-NeRF and RS-NeRF** model RS only, with instantaneous row exposure.

**Implication for us:** a blur extension would add exposure integration over T_e on top of per-row time t(y). Neither appears in any feed-forward 3D model found.

---

## 7. Per-paper notes

Full fields, one row per paper, are in the CSV. Key points not already covered above:
- **DeepUnrollNet**: not on arXiv. The CVF abstract says code and datasets are released; dataset links are now dead.
- **AdaRSC**: uses M = 9 displacement fields.
- **BS-RSC**: the GS pulse is delayed so the GS exposure matches the RS middle scanline.
- **JAMNet**: no arXiv version found. CVF abstract claims a "4.7 dB PSNR leap on real-world RSC".
- **QRSC**: reports +4.98 / +0.77 / +4.33 dB PSNR over prior work on Carla-RS / Fastec-RS / BS-RSC (CVF abstract).
- **DFRSC**: argues linear scaling of flow fails for non-linear motion and forward warping creates artifacts, so it estimates GS→RS distortion flow and warps backward.
- **IFED**: argues that readout cannot be resolved from consecutive frames, which supports a model that reasons about motion and readout jointly. Its real dual-RS rigs used FL3-U3-13S2C and BFS-U3-63S4C cameras, with no GT.
- **RS-Diffusion**: venue AAAI-25 (Vol. 39 No. 9, pp. 9373–9381; DOI 10.1609/aaai.v39i9.33015) from the AAAI OJS page. The arXiv page lists no venue.
- **SelfDRSC++**: arXiv comments say "Accepted by Pattern Recognition 2026". The README cites article 114636.
- **SelfUnroll**: arXiv title differs from the journal title ("Self-supervised Shutter Unrolling with Events", IJCV 2025 per project page, author order changed). Inference ≈ 68 ms per 640×480 GS frame (RTX 3090).
- **Gaussian Splatting on the Move**: v2 (2024-05) made VIO optional; velocities can start at zero. Without known T_e and T_ro it can run blur *or* RS mode from plain video, but not both. Real data came from a Samsung S20 FE, Pixel 5 and iPhone 15 Pro.
- **USB-NeRF**: real data came from a GoPro HERO6, Canon EOS M3, iPhone 14 Pro and TUM-RS.
- **URS-NeRF**: code is at https://github.com/ZiaoLiuS/URS-NERF (PyTorch 1.13.1, tiny-cuda-nn, Tri-MipRF base).
- **3DGUT / 3DGRUT**: Apache-2.0, could serve as an RS renderer for synthetic data.

---

## 8. What could not be verified

- **Wang et al., "Event-based Shutter Unrolling and Motion Deblurring in Dynamic Scenes" (IEEE SPL 2024)**: only aggregator snippets. The x-mol page was empty and the IEEE page did not load. Authors and DOI are unverified.
- **Venue claims taken from non-publisher pages:**
  - LBCNet, TPAMI 2024: project page only.
  - SelfRSSplat, TCSVT 2025: repo BibTeX only.
  - SelfUnroll, IJCV 2025: project page only.
  - URS-NeRF's Springer DOI: search result only; the ECCV poster page was opened.
- **Venue claims I did not accept:** NeuRAD and SplatAD are written as "arXiv" because their arXiv pages state no venue.
- **RS-Diffusion weights**: the Hugging Face model repo is empty. Whether the 1.28 GB `RS_Real.pt` in the dataset repo is a checkpoint was not checked.
- **Links I could not test:**
  - Baidu-hosted assets (Gev-RS, Gev-RS-Real, SelfRSSplat weights, WHU-RS) timed out from our network.
  - UniINR weights and USB-NeRF Unreal-RS data redirected to a Microsoft login.
- **Carla-RS**: no working download found. Its resolution was not verified.
- **Repos not opened:** Naor et al., Mo et al., Praveen K et al., Lu et al. TVCG, Photoreal Egocentric, BAD-NeRF, 4DGRT. Rengarajan and Zhuang: no code found, but not exhaustively searched.
- **Output GS time** for Deep HM and RS-Diffusion: not stated in what I read.
- **JCD**: whether `checkpoints.zip` also contains a Fastec model was not checked (the zip was not downloaded).
- **Deep HM**: code completeness is inferred from the repo tree; `gen_bases` is missing.

---

## 9. Query log

### WebSearch queries (44)

**Topic 3: learned 2D RS correction**
1. `deep learning single image rolling shutter correction network`
2. `rolling shutter correction two consecutive frames CNN global shutter`
3. `self-supervised rolling shutter correction`
4. `event camera guided rolling shutter correction unrolling`
5. `rolling shutter correction 2025 CVPR ICCV new method`
6. `rolling shutter to global shutter video temporal super-resolution learning`
7. `diffusion model rolling shutter removal single image`
8. `dual reversed rolling shutter distortion correction network`
9. `rolling shutter correction arXiv 2026` (extended)
10. `JAMNet joint appearance and motion learning rolling shutter correction github`
11. `rolling shutter correction transformer video 2025 arXiv learned undistortion flow nonlinear motion` (extended)
12. `rolling shutter gyroscope guided learning correction network GyroFlow RS`
13. `deep homography mixture single image rolling shutter correction ICCV 2023 github`
14. `EvShutter transforming events unconstrained rolling shutter correction`
15. `Bin Fan rolling shutter temporal super-resolution bilateral cost volume TPAMI`
16. `rolling shutter correction IEEE Transactions 2025 learning-based video RS correction network` (extended)
17. `EvShutter RS-ERGB dataset github code release Huawei`
18. `"Structure-And-Motion-Aware Rolling Shutter Correction" code github Zhuang SMARSC`
19. `"Event-based Shutter Unrolling and Motion Deblurring in Dynamic Scenes" IEEE Signal Processing Letters`
20. `rolling shutter DUSt3R VGGT feed-forward 3D reconstruction rolling shutter aware pointmap` (extended; direct-threat check)

**2D RS datasets**
21. `Fastec-RS Carla-RS dataset download rolling shutter`
22. `BS-RSC dataset beam splitter rolling shutter correction real-world`
23. `RS-GOPRO dataset rolling shutter`
24. `event camera rolling shutter dataset Gev-RS real paired global shutter`
25. `RSCD dataset rolling shutter blur correction BS-RSCD beam splitter`
26. `real-world rolling shutter benchmark dataset paired ground truth 2024 2025`
27. `synthetic rolling shutter dataset high-speed video generation training correction network`
28. `gyroscope rolling shutter correction deep learning dataset smartphone`

**Topic 6: RS NeRF / 3DGS**
29. `rolling shutter Gaussian splatting reconstruction`
30. `rolling shutter neural radiance fields bundle adjustment unrolling`
31. `rolling shutter aware Gaussian splatting SLAM`
32. `3DGS rolling shutter compensation camera trajectory per-row pose novel view synthesis 2025` (extended)
33. `NeRF rolling shutter smartphone video pose optimization continuous-time trajectory`
34. `"rolling shutter" splatting event camera or spline trajectory rendering`
35. `"URS-NeRF" unordered rolling shutter bundle adjustment ECCV 2024 ecva`
36. `"USB-NeRF" ICLR 2024 openreview rolling shutter`
37. `Gaussian splatting rolling shutter 2025 2026 arXiv per-scanline rendering handheld video` (extended)

**Topic 9: RS + blur**
38. `joint rolling shutter correction and deblurring exposure time readout time model`
39. `rolling shutter motion deblurring single image camera motion`
40. `event-guided rolling shutter deblurring joint network 2024`
41. `motion blur and rolling shutter NeRF deblur radiance field exposure integration`
42. `blurry rolling shutter video to sharp global shutter high frame rate` (returned only consumer camera articles)
43. `Moment-Reenacting inverse motion degradation cross-shutter guidance`
44. `"Going Unconstrained With Rolling Shutter Deblurring" ICCV 2017 Mohan Rengarajan Rajagopalan`

### arXiv API queries

All used `https://export.arxiv.org/api/query?search_query=…&sortBy=submittedDate&sortOrder=descending&max_results=200`. The first attempts returned HTTP 429; I retried with 20–25 s backoff.

| # | search_query | hits |
|---|---|---|
| q1 | `abs:"rolling shutter" AND abs:correction` | 33 |
| q2 | `abs:"rolling shutter" AND (abs:NeRF OR abs:"radiance field" OR abs:"gaussian splatting" OR abs:splatting)` | 14 |
| q3 | `abs:"rolling shutter" AND (abs:blur OR abs:deblurring OR abs:exposure)` | 41 |
| q4 | `abs:"rolling shutter" AND abs:event` | 12 |
| q5 | `abs:"rolling shutter" AND abs:dataset` | 51 |
| q6 | `abs:"rolling shutter" AND (abs:unrolling OR abs:undistortion OR abs:rectification)` | 16 |
| q7 | `ti:"rolling shutter"` | 62 |
| q8 | `all:"rolling shutter" AND (all:network OR all:learning OR all:diffusion)` | 47 |
| q9 | `abs:"rolling shutter" AND (abs:SLAM OR abs:reconstruction)` | 42 |
| q10 | `all:"rolling shutter" AND (all:gaussian OR all:nerf)` | 18 |
| q11 | `ti:rolling AND ti:shutter` | download truncated (local disk full), not parsed |
| q12 | `abs:"rolling shutter" AND (abs:depth OR abs:pointmap OR abs:"feed-forward")` | 16 (nothing feed-forward or RS-aware in 3D) |

### Pages and repos opened

The CSV `url_opened` column lists every page per paper. In addition:
- GitHub README, config, loader and tree files were read for:
  - **2D RS correction:** ethliup/DeepUnrollNet, GitCVfb/{SUNet, RSSR, CVR, JAMNet, LBCNet, SelfRSSplat}, zzh-tech/{RSCD, Dual-Reversed-RS}, ljzycmd/{BSRSC, DFRSC}, DelinQu/qrsc (master and the old commit 8739072), lightChaserX/neural-global-shutter, DavidYan2001/Deep_RS-HM, lhaippp/RS-Diffusion.
  - **Dual-reversed:** shangwei5/{SelfDRSC, SelfDRSC_plusplus}.
  - **Event-based:** zxyemo/EvUnroll, juliuserbach/EvShutter, w3un/selfunroll_code, yunfanLu/UniINR.
  - **Cross-shutter (RS + blur):** jixiang2016/{dualBR, stereoBR}.
  - **NeRF / 3DGS / SLAM:** SpectacularAI/3dgs-deblur, WU-CVGL/USB-NeRF, MyNiuuu/RS-NeRF, ZiaoLiuS/URS-NERF, umautobots/trgs_slam, nv-tlabs/3dgrut, georghess/neurad-studio.
- Hugging Face API: `Lhaippp/RS-Diffusion` (model), `Yzl-code/RS-Diffusion` (dataset), `yunfanlu/UNIINR-Fastec-Simulated-Dataset`.
- Zenodo record 10848124.
- Project pages: w3un.github.io/selfunroll, npucvr.github.io/LBCNet, jixiang2016.github.io/dualBR_site, boxulibrary.github.io/projects/URS-NeRF, umautobots.github.io/trgs_slam.

### Environment issues during the run

- CVF returned 403 to WebFetch, so I used urllib instead.
- The GitHub API hit its rate limit, so I switched to raw files and HTML pages.
- dblp (bot wall), Springer (auth redirect) and IEEE pages could not be read.
- Baidu Pan timed out.
- The local C: drive filled up mid-session (0 MB free; ~360 MB free again later). I did not delete anything; the cause was outside this slice's files.
