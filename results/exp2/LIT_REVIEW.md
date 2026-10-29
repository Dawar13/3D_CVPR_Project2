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

*Papers whose primary topic is 1 (17; generated from `related_work.csv`, fields as in the raw files).*

| # | Title | Authors | Venue (as recorded, with source) | Year | ID | Code | Threat |
|---|---|---|---|---|---|---|---|
| 1 | Anchor Points for Monocular Depth Estimation Under Severe Rolling Shutter Ego-motion | Jacob Carter, Vaishnav Ramesh, Md Jahidul Islam, Sanjeev J. Koppal | UNVERIFIED (lab page lists it under 2026; no venue or arXiv given) | 2026 | none found (no arXiv ID or venue listed on the project page or PDF) | Y https://github.com/JacobACarter/Stable-Derolling (Apache-2.0) | med |
| 2 | Foundation4D: Towards a unified perception and reconstruction model for autonomous vehicles | Sergio Agostinho, Maximilian Apodaca, Emmanuel Attia, Guillem Braso, George Dong, Ilya Fedorov, et al. (NVIDIA) | arXiv | 2026 | arXiv:2610.30340 | N | med |
| 3 | Interp3R: Continuous-time 3D Geometry Estimation with Frames and Events | Shuang Guo, Filbert Febryanto, Lei Sun, Luc Van Gool, Guillermo Gallego | arXiv | 2026 | arXiv:2603.14528 | N | med |
| 4 | AsyncEvGS: Asynchronous Event-Assisted Gaussian Splatting for Handheld Motion-Blurred Scenes | Jun Dai, Renbiao Jin, Bo Xu, Yutian Chen, Linning Xu, Mulin Yu, et al. | arXiv | 2026 | arXiv:2605.07192 | N | low |
| 5 | DynamicVGGT: Learning Dynamic Point Maps for 4D Scene Reconstruction in Autonomous Driving | Zhuolin He, Jing Li, Guanghao Li, Xiaolei Chen, Jiacheng Tang, Siyang Zhang, et al. | arXiv | 2026 | arXiv:2603.08254 | N | low |
| 6 | Event3R: Asynchronous-to-Global 3D Reconstruction from Event Camera via Spatial-Temporal Feature Aggregation | Jian Huang, Haotian Shen, Xinhao Lou, Chengrui Dong, Wenpu Li, Peidong Liu | IROS 2026 (arXiv comments) | 2026 | arXiv:2607.15727 | N | low |
| 7 | EventVGGT: Exploring Cross-Modal Distillation for Consistent Event-based Depth Estimation | Yinrui Ren, Jinjing Zhu, Kanghao Chen, Zhuoxiao Li, Jing Ou, Zidong Cao, et al. | ECCV 2026 (arXiv comments; ECCV 2026 accepted list) | 2026 | arXiv:2603.09385 | Y https://github.com/yinruiRen/EventVGGT (MIT) | low |
| 8 | Marrying Optimal Transport and ODEs for Unified Continuous-Time 4D Reconstruction and Tracking | Liying Yang, Hao Mo, Jialun Liu, Chen Liu, Xinxing Yu, Chenhao Guan, et al. | NeurIPS 2026 main track (arXiv comments field) | 2026 | arXiv:2608.09613 | N | low |
| 9 | Point4Cast: Streaming Dynamic Scene Reconstruction and Forecasting | Xinhang Liu, Pedro Miraldo, Suhas Lohit, Huaizu Jiang, Naoko Sawada, Yu-Wing Tai, et al. | CVPR 2026 (CVF open access, pp. 14600-14611) | 2026 | none found (no arXiv); MERL TR2026-077 | N | low |
| 10 | ProMo3D: Probing Motion Cues in Frozen 3D Foundation Models | Xiaoang Zhang, Mert Kiray, Yordanka Velikova, Manoj Biswanath, Benjamin Busam | NeurIPS 2026 Main Track (arXiv comments) | 2026 | arXiv:2610.33802 | UNVERIFIED | low |
| 11 | Robust Visual SLAM for UAV Navigation in GPS-Denied and Degraded Environments: A Multi-Paradigm Evaluation and Deployment Study | Prasoon Kumar, Akshay Deepak, Sandeep Kumar | arXiv | 2026 | arXiv:2605.03678 | UNVERIFIED | low |
| 12 | RoMo-GS: Efficient Online 3D Gaussian Splatting for Robot Mapping via Exposure-Integrated Motion Blur | Sangwoo Jung, Di Deng, Jiahui Fu, Laura Herlant, Ayoung Kim, Yafei Hu | arXiv | 2026 | arXiv:2610.24605 | N | low |
| 13 | Unblur-SLAM: Dense Neural SLAM for Blurry Inputs | Qi Zhang, Denis Rozumny, Francesco Girlanda, Sezer Karaoglu, Marc Pollefeys, Theo Gevers, et al. | CVPR 2026 (arXiv comments; CVF CVPR 2026 listing) | 2026 | arXiv:2603.26810 | N | low |
| 14 | V-DPM: 4D Video Reconstruction with Dynamic Point Maps | Edgar Sucar, Eldar Insafutdinov, Zihang Lai, Andrea Vedaldi | CVPR 2026 (CVF CVPR 2026 listing; arXiv comments list only project page) | 2026 | arXiv:2601.09499 | N | low |
| 15 | DeblurSplat: SfM-free 3D Gaussian Splatting with Event Camera for Robust Deblurring | Pengteng Li, Yunfan Lu, Pinhao Song, Weiyu Guo, Huizai Yao, F. Richard Yu, et al. | TMM 2026 (arXiv comments) | 2025 | arXiv:2509.18898 | N | low |
| 16 | EAG3R: Event-Augmented 3D Geometry Estimation for Dynamic and Extreme-Lighting Scenes | Xiaoshan Wu, Yifei Yu, Xiaoyang Lyu, Yihua Huang, Bo Wang, Baoheng Zhang, et al. | NeurIPS 2025 (spotlight) (arXiv comments; project page) | 2025 | arXiv:2512.00771 | Y https://github.com/Candy-Crusher/EAG3R (no licence) | low |
| 17 | Fin3R: Fine-tuning Feed-forward 3D Reconstruction Models via Monocular Knowledge Distillation | Weining Ren, Hongjun Wang, Xiao Tan, Kai Han | NeurIPS 2025 (arXiv comments) | 2025 | arXiv:2511.22429 | UNVERIFIED | low |

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

*Papers whose primary topic is 2 (19; generated from `related_work.csv`, fields as in the raw files).*

| # | Title | Authors | Venue (as recorded, with source) | Year | ID | Code | Threat |
|---|---|---|---|---|---|---|---|
| 1 | Fisheye3R: Adapting Unified 3D Feed-Forward Foundation Models to Fisheye Lenses | Ruxiao Duan, Erin Hong, Dongxu Zhao, Eric Turner, Alex Wong, Yunwen Zhou | ECCV 2026 (source: arXiv comments field 'European Conference on Computer Vision 2026') | 2026 | arXiv:2603.28896 | Y https://github.com/android-xr/fisheye3r (Apache-2.0) | med |
| 2 | RayTun3R: Online Camera Adaptation in 3D Foundation Models | Daniil Sinitsyn, Nikita Araslanov, Daniel Cremers | NeurIPS 2026 (poster; source: neurips.cc/virtual/2026/poster/151667; arXiv page has no venue comment) | 2026 | arXiv:2607.02711 | N | med |
| 3 | CAM3R: Camera-Agnostic Model for 3D Reconstruction | Namitha Guruprasad, Abhay Yadav, Cheng Peng, Rama Chellappa | arXiv | 2026 | arXiv:2603.22631 | N | low |
| 4 | Many Eyes, One World: Feed-Forward 3D Reconstruction from Mixed Cameras (MEOW) | Qiaoge Li, Yifan Zhan, Haijun Yang, Haiyang Liu, Yiyi Cai, Chenchi Luo | arXiv | 2026 | arXiv:2609.35658 | N | low |
| 5 | OmniRay: Camera-Agnostic Monocular 3D Reconstruction with Composite Ray Fields | Namitha Guruprasad, Abhay Yadav, Cheng Peng, Rama Chellappa, Anand Bhattad | arXiv | 2026 | arXiv:2610.31645 | N | low |
| 6 | PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery | Yijing Guo, Mengjun Chao, Luo Wang, Tianyang Zhao, Haizhao Dai, Yingliang Zhang et al. | CVPR 2026 (source: arXiv comments 'Accepted by CVPR 2026') | 2026 | arXiv:2603.17571 | N | low |
| 7 | RIGOR: Rig-Informed Geometry for Omnidirectional Reconstruction | Tingjun Huang, Dmitry Rudshin, Mathieu Meyer, Pietro Bonazzi, Marc Pollefeys, Emilia Szymanska | arXiv | 2026 | arXiv:2609.13504 | Y https://github.com/TangentH/RIGOR (Apache-2.0) | low |
| 8 | UniScale: Unified Scale-Aware 3D Reconstruction for Multi-View Understanding via Prior Injection for Robotic Perception | Mohammad Mahdavian, Gordon Tan, Binbin Xu, Yuan Ren, Dongfeng Bai, Bingbing Liu | arXiv | 2026 | arXiv:2602.23224 | N | low |
| 9 | VGGT-360: Geometry-Consistent Zero-Shot Panoramic Depth Estimation | Jiayi Yuan, Haobo Jiang, De Wen Soh, Na Zhao | arXiv | 2026 | arXiv:2603.18943 | N | low |
| 10 | Wid3R: Wide Field-of-View 3D Reconstruction via Camera Model Conditioning | Dongki Jung, Jaehoon Choi, Adil Qureshi, Somi Jeong, Dinesh Manocha, Suyong Yeon | arXiv (GitHub repo title says '[ECCV 2026]'; the arXiv page shows no venue) | 2026 | arXiv:2602.05321 | Y https://github.com/jdk9405/Wid3R | low |
| 11 | X-Lens: Real-Time Metric Depth Estimation with Heterogeneous Cameras | Heng Zhou, Shuhong Liu, Yonghao He, Bohao Zhang, Fa Fu, Chenhui Hou et al. | arXiv | 2026 | arXiv:2607.12993 | N | low |
| 12 | Cameras as Relative Positional Encoding | Ruilong Li, Brent Yi, Junchen Liu, Hang Gao, Yi Ma, Angjoo Kanazawa | NeurIPS 2025 (source: arXiv journal-ref) | 2025 | arXiv:2507.10496 | N | low |
| 13 | MapAnything: Universal Feed-Forward Metric 3D Reconstruction | Nikhil Keetha, Norman Muller, Johannes Schonberger, Lorenzo Porzi, Yuchen Zhang, Tobias Fischer et al. | 3DV 2026 (source: arXiv comments) | 2025 | arXiv:2509.13414 | Y https://github.com/facebookresearch/map-anything (Apache-2.0) | low |
| 14 | OmniVGGT: Omni-Modality Driven Visual Geometry Grounded Transformer | Haosong Peng, Hao Li, Yalun Dai, Yushi Lan, Yihang Luo, Tianyu Qi et al. | CVPR 2026 (source: cvpr.thecvf.com/virtual/2026/poster/39518; project page says 'CVPR 2026 Highlight') | 2025 | arXiv:2511.10560 | Y https://github.com/Livioni/OmniVGGT-official | low |
| 15 | Pow3R: Empowering Unconstrained 3D Reconstruction with Camera and Scene Priors | Wonbong Jang, Philippe Weinzaepfel, Vincent Leroy, Lourdes Agapito, Jerome Revaud | CVPR 2025 (source: arXiv comments) | 2025 | arXiv:2503.17316 | Y https://github.com/naver/pow3r | low |
| 16 | Rig3R: Rig-Aware Conditioning for Learned 3D Reconstruction | Samuel Li, Pujith Kachana, Prajwal Chidananda, Saurabh Nair, Yasutaka Furukawa, Matthew Brown | NeurIPS 2025 Spotlight (source: neurips.cc/virtual/2025/poster/115429, listed there as 'Rig3R: Rig-Aware Conditioning and Discovery for 3D Reconstruction') | 2025 | arXiv:2506.02265 | N | low |
| 17 | UniK3D: Universal Camera Monocular 3D Estimation | Luigi Piccinelli, Christos Sakaridis, Mattia Segu, Yung-Hsu Yang, Siyuan Li, Wim Abbeloos et al. | CVPR 2025 (source: GitHub README; arXiv page shows no venue) | 2025 | arXiv:2503.16591 | Y https://github.com/lpiccinelli-eth/unik3d (CC BY-NC 4.0) | low |
| 18 | WorldMirror: Universal 3D World Reconstruction with Any-Prior Prompting | Yifan Liu, Zhiyuan Min, Zhenwei Wang, Junta Wu, Tengfei Wang, Yixuan Yuan et al. | ICML 2026 (source: arXiv comments) | 2025 | arXiv:2510.10726 | Y https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror | low |
| 19 | LoRA3D: Low-Rank Self-Calibration of 3D Geometric Foundation Models | Ziqi Lu, Heng Yang, Danfei Xu, Boyi Li, Boris Ivanovic, Marco Pavone et al. | ICLR 2025 (source: research.nvidia.com publication page; project page says Spotlight) | 2024 | arXiv:2412.07746 | N | low |

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

*Papers whose primary topic is 3 (28; generated from `related_work.csv`, fields as in the raw files).*

| # | Title | Authors | Venue (as recorded, with source) | Year | ID | Code | Threat |
|---|---|---|---|---|---|---|---|
| 1 | Single Image Rolling Shutter Removal with Diffusion Models | Zhanglei Yang, Haipeng Li, Mingbo Hong, Chen-Lin Zhang, Jiajun Li, Shuaicheng Liu | AAAI 2025 (AAAI OJS page: Vol.39 No.9 pp.9373-9381; rs2d slice). Classical slice left it as arXiv 2024 because it did not open the AAAI page | 2025 | arXiv:2407.02906; DOI 10.1609/aaai.v39i9.33015 | Y https://github.com/lhaippp/RS-Diffusion (Apache-2.0) | med |
| 2 | HaRP: High Dynamic Range Photosequencing through Dual Reversed Shutter Scanning | Xiang Ji, Guixu Lin, Jiancheng Zhao, Zhengwei Yin, Yinqiang Zheng | arXiv | 2026 | arXiv:2609.28439 | N | low |
| 3 | SelfDRSC++: Self-Supervised Dual Reversed Rolling Shutter Correction via Video Interpolation | Wei Shang, Dongwei Ren, Wanying Zhang, Qilong Wang, Pengfei Zhu, Wangmeng Zuo | Pattern Recognition 2026 (arXiv comments) | 2026 | arXiv:2408.11411 | Y https://github.com/shangwei5/SelfDRSC_plusplus (no licence) | low |
| 4 | Thermal Sequence Reconstruction under Coupled Motion Blur and Rolling-Shutter Readout | Anonymous (ICLR 2027 submission) | ICLR 2027 Conference Submission (OpenReview, under review) | 2026 | OpenReview:BMYxcibovq | N | low |
| 5 | Self-Supervised Learning for Rolling Shutter Temporal Super-Resolution | Bin Fan, Ying Guo, Yuchao Dai, Chao Xu, Boxin Shi | IEEE TCSVT 2025, 35(1):769-782 (per repo README BibTeX; IEEE page not opened) | 2025 | UNVERIFIED DOI (IEEE page not opened) | Y https://github.com/GitCVfb/SelfRSSplat (research-only) | low |
| 6 | Self-supervised Learning of Event-guided Video Frame Interpolation for Rolling Shutter Frames | Yunfan Lu, Guoqiang Liang, Yiran Shen, Lin Wang | IEEE TVCG (arXiv comments: "accepted by IEEE TVCG") | 2025 | arXiv:2306.15507 | N | low |
| 7 | StableMotion: One-Step Motion Estimation with Diffusion Prior | Ziyi Wang, Haipeng Li, Lin Sui, Tianhao Zhou, Hai Jiang, Lang Nie, et al. | arXiv | 2025 | arXiv:2505.06668 | Y https://github.com/ivowang/StableMotion (MIT) | low |
| 8 | Learning Bilateral Cost Volume for Rolling Shutter Temporal Super-Resolution | Bin Fan, Yuchao Dai, Hongdong Li | IEEE TPAMI 2024 (per project page npucvr.github.io/LBCNet; IEEE page not opened) | 2024 | IEEE doc 10382595 (from project page) | Y https://github.com/GitCVfb/LBCNet (research-only) | low |
| 9 | Rolling Shutter Correction with Intermediate Distortion Flow Estimation | Mingdeng Cao, Sidi Yang, Yujiu Yang, Yinqiang Zheng | CVPR 2024 (arXiv comments + CVF page) | 2024 | arXiv:2404.06350; CVF CVPR2024 pp.25338-25347 | Y https://github.com/ljzycmd/DFRSC (no licence) | low |
| 10 | UniINR: Event-guided Unified Rolling Shutter Correction, Deblurring, and Interpolation | Yunfan Lu, Guoqiang Liang, Yusheng Wang, Lin Wang, Hui Xiong | ECCV 2024 (arXiv comments) | 2024 | arXiv:2305.15078 | Y https://github.com/yunfanLu/UniINR (no licence) | low |
| 11 | Deep Homography Mixture for Single Image Rolling Shutter Correction | Weilong Yan, Robby T. Tan, Bing Zeng, Shuaicheng Liu | ICCV 2023 (CVF page) | 2023 | CVF ICCV2023 pp.9868-9877 | Y https://github.com/DavidYan2001/Deep_RS-HM (no licence) | low |
| 12 | EvShutter: Transforming Events for Unconstrained Rolling Shutter Correction | Julius Erbach, Stepan Tulyakov, Patricia Vitoria, Alfredo Bochicchio, Yuanyou Li | CVPR 2023 (CVF page) | 2023 | CVF CVPR2023 pp.13904-13913 | N | low |
| 13 | Joint Appearance and Motion Learning for Efficient Rolling Shutter Correction | Bin Fan, Yuxin Mao, Yuchao Dai, Zhexiong Wan, Qi Liu | CVPR 2023 (CVF page) | 2023 | CVF CVPR2023 pp.5671-5681 (no arXiv found) | Y https://github.com/GitCVfb/JAMNet (research-only) | low |
| 14 | Rolling Shutter Inversion: Bring Rolling Shutter Images to High Framerate Global Shutter Video | Bin Fan, Yuchao Dai, Hongdong Li | IEEE TPAMI (arXiv comments: "Accepted by IEEE TPAMI"); repo bib: TPAMI 2023 45(5):6214-6230 | 2023 | arXiv:2210.03040 | Y https://github.com/GitCVfb/RSSR | low |
| 15 | Self-supervised Learning to Bring Dual Reversed Rolling Shutter Images Alive | Wei Shang, Dongwei Ren, Chaoyu Feng, Xiaotao Wang, Lei Lei, Wangmeng Zuo | ICCV 2023 (arXiv comments + CVF page) | 2023 | arXiv:2305.19862; CVF ICCV2023 pp.13086-13094 | Y https://github.com/shangwei5/SelfDRSC (MIT) | low |
| 16 | Self-Supervised Scene Dynamic Recovery from Rolling Shutter Images and Events | Yangguang Wang, Xiang Zhang, Mingyuan Lin, Lei Yu, Boxin Shi, Wen Yang et al. | arXiv (journal version "Self-supervised Shutter Unrolling with Events", IJCV 2025, DOI 10.1007/s11263-025-02364-z, per project page; Springer page not opened) | 2023 | arXiv:2304.06930 | Y https://github.com/w3un/selfunroll_code (no licence) | low |
| 17 | Towards Nonlinear-Motion-Aware and Occlusion-Robust Rolling Shutter Correction | Delin Qu, Yizhen Lao, Zhigang Wang, Dong Wang, Bin Zhao, Xuelong Li | ICCV 2023 (arXiv comments + CVF page) | 2023 | arXiv:2303.18125; CVF ICCV2023 pp.10680-10688 | Y https://github.com/DelinQu/qrsc (no licence) | low |
| 18 | Bringing Rolling Shutter Images Alive with Dual Reversed Distortion | Zhihang Zhong, Mingdeng Cao, Xiao Sun, Zhirong Wu, Zhongyi Zhou, Yinqiang Zheng et al. | ECCV 2022 Oral (arXiv comments) | 2022 | arXiv:2203.06451 | Y https://github.com/zzh-tech/Dual-Reversed-RS (MIT) | low |
| 19 | Combining Internal and External Constraints for Unrolling Shutter in Videos | Eyal Naor, Itai Antebi, Shai Bagon, Michal Irani | ECCV 2022 (arXiv comments) | 2022 | arXiv:2207.11725 | N | low |
| 20 | Context-Aware Video Reconstruction for Rolling Shutter Cameras | Bin Fan, Yuchao Dai, Zhiyuan Zhang, Qi Liu, Mingyi He | CVPR 2022 (arXiv comments + CVF page) | 2022 | arXiv:2205.12912; CVF CVPR2022 pp.17572-17582 | Y https://github.com/GitCVfb/CVR (research-only) | low |
| 21 | EvUnroll: Neuromorphic Events Based Rolling Shutter Image Correction | Xinyu Zhou, Peiqi Duan, Yi Ma, Boxin Shi | CVPR 2022 (CVF page) | 2022 | CVF CVPR2022 pp.17775-17784 | Y https://github.com/zxyemo/EvUnroll (no licence) | low |
| 22 | Learning Adaptive Warping for Real-World Rolling Shutter Correction | Mingdeng Cao, Zhihang Zhong, Jiahao Wang, Yinqiang Zheng, Yujiu Yang | CVPR 2022 (arXiv comments + CVF page) | 2022 | arXiv:2204.13886; CVF CVPR2022 pp.17785-17793 | Y https://github.com/ljzycmd/BSRSC (MIT) | low |
| 23 | Neural Global Shutter: Learn to Restore Video from a Rolling Shutter Camera with Global Reset Feature | Zhixiang Wang, Xiang Ji, Jia-Bin Huang, Shin'ichi Satoh, Xiao Zhou, Yinqiang Zheng | CVPR 2022 (arXiv comments + CVF page) | 2022 | arXiv:2204.00974; CVF CVPR2022 pp.17794-17803 | N | low |
| 24 | Deep network for rolling shutter rectification | Praveen K, Lokesh Kumar T, A. N. Rajagopalan | arXiv (no acceptance stated) | 2021 | arXiv:2112.06170 | N | low |
| 25 | Inverting a Rolling Shutter Camera: Bring Rolling Shutter Images to High Framerate Global Shutter Video | Bin Fan, Yuchao Dai | ICCV 2021 (CVF page) | 2021 | CVF ICCV2021 pp.4228-4237 | Y https://github.com/GitCVfb/RSSR (no licence) | low |
| 26 | SUNet: Symmetric Undistortion Network for Rolling Shutter Correction | Bin Fan, Yuchao Dai, Mingyi He | ICCV 2021 (arXiv comments + CVF page) | 2021 | arXiv:2108.04775; CVF ICCV2021 pp.4541-4550 | Y https://github.com/GitCVfb/SUNet (no licence) | low |
| 27 | Deep Shutter Unrolling Network | Peidong Liu, Zhaopeng Cui, Viktor Larsson, Marc Pollefeys | CVPR 2020 (CVF page) | 2020 | CVF CVPR2020 pp.5941-5949 (not on arXiv) | Y https://github.com/ethliup/DeepUnrollNet (MIT) | low |
| 28 | Unrolling the Shutter: CNN to Correct Motion Distortions | Vijay Rengarajan, Yogesh Balaji, A. N. Rajagopalan | CVPR 2017 (CVF page) | 2017 | CVF CVPR2017 pp.2291-2299 | N | low |

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

*Papers whose primary topic is 4 (23; generated from `related_work.csv`, fields as in the raw files).*

| # | Title | Authors | Venue (as recorded, with source) | Year | ID | Code | Threat |
|---|---|---|---|---|---|---|---|
| 1 | Rolling Shutter Camera Self-Calibration | Yongcong Zhang, Navid Rabbani, Bangyan Liao, Chengbo Wang, Yizhen Lao, Adrien Bartoli | ECCV 2026 per EnCoV group publications page (arXiv comments empty) | 2026 | arXiv:2608.01509 | N | med |
| 2 | Rolling Shutter Relative Pose Estimation Made Practical | Daniel Barath | ECCV 2026 (ECCV 2026 accepted-papers list and repo README per scoop slice; Google Research publication page per classical slice; arXiv comments field empty) | 2026 | arXiv:2606.26863 | Y https://github.com/danini/rolling_shutter_made_practical (MIT) | med |
| 3 | Single-View Rolling-Shutter SfM | Sofia Errazuriz Munoz, Kim Kiehn, Petr Hruby, Kathlen Kohn | arXiv (no comments) | 2026 | arXiv:2603.11888 | Y https://github.com/sofiaemd/Single-View-Rolling-Shutter-SfM (GPL-3.0) | low |
| 4 | The Fast and the Distorted - How Sensor Readout Shapes Image-based 3D Modelling | Geert J. Verhoeven | ISPRS Archives XLVIII-2/W12-2026 (3D-ARCH 2026), pp. 495-502 | 2026 | doi:10.5194/isprs-archives-XLVIII-2-W12-2026-495-2026 | N | low |
| 5 | A Linear N-Point Solver for Structure and Motion from Asynchronous Tracks | Hang Su, Yunlong Feng, Daniel Gehrig, Panfeng Jiang, Ling Gao, Xavier Lagorce, et al. | ICCV 2025 (CVF page; repo README says Highlight) | 2025 | arXiv:2507.22733 | Y https://github.com/suhang99/AsyncTrack-Motion-Solver (Apache-2.0) | low |
| 6 | Minimal Solvers for Full DoF Motion Estimation from Asynchronous Tracks | Petr Hruby, Marc Pollefeys | 3DV 2026 (OpenReview venue field "3DV 2026 Poster"; arXiv comments give only page count) | 2025 | arXiv:2508.17537 | N | low |
| 7 | Order-One Rolling Shutter Cameras | Marvin Anas Hahn, Kathlen Kohn, Orlando Marigliano, Tomas Pajdla | CVPR 2025 (CVF CVPR 2025 listing; scoop slice). Classical slice: arXiv 2024, no acceptance stated on the arXiv page | 2025 | arXiv:2403.11295 | N | low |
| 8 | Single-Scanline Relative Pose Estimation for Rolling Shutter Cameras | Petr Hruby, Marc Pollefeys | ICCV 2025 (arXiv comments; CVF ICCV 2025 listing) | 2025 | arXiv:2506.22069 | Y https://github.com/petrhruby97/single_scanline (no licence) | low |
| 9 | RSL-BA: Rolling Shutter Line Bundle Adjustment | Yongcong Zhang, Bangyan Liao, Yifei Xue, Chen Lu, Peidong Liu, Yizhen Lao | ECCV 2024 (ECVA page; repo README) | 2024 | arXiv:2408.05409 | Y https://github.com/Yongcong-Zhang/RSL-BA (no licence) | low |
| 10 | Revisiting Rolling Shutter Bundle Adjustment: Toward Accurate and Fast Solution | Bangyan Liao, Delin Qu, Yifei Xue, Huiqing Zhang, Yizhen Lao | CVPR 2023 (arXiv comments + CVF page pp 4863-4871). NOTE: seed list said ICCV 2023; pages say CVPR 2023. | 2023 | arXiv:2209.08503 | Y https://github.com/DelinQu/NW-RSBA (no licence) | low |
| 11 | Scanline Homographies for Rolling-Shutter Plane Absolute Pose | Fang Bai, Agniva Sengupta, Adrien Bartoli | CVPR 2022 (CVF page) | 2022 | CVF CVPR2022 | N | low |
| 12 | From two rolling shutters to one global shutter | Cenek Albl, Zuzana Kukelova, Viktor Larsson, Tomas Pajdla, Konrad Schindler | CVPR 2020 (arXiv comments) | 2020 | arXiv:2006.01964 | N | low |
| 13 | Minimal Rolling Shutter Absolute Pose with Unknown Focal Length and Radial Distortion | Zuzana Kukelova, Cenek Albl, Akihiro Sugimoto, Konrad Schindler, Tomas Pajdla | ECCV 2020 (ECVA page) | 2020 | arXiv:2004.14052 | Y https://github.com/CenekAlbl/RnP (BSD-3-Clause) | low |
| 14 | Linear solution to the minimal absolute pose rolling shutter problem | Zuzana Kukelova, Cenek Albl, Akihiro Sugimoto, Tomas Pajdla | ACCV 2018 (arXiv comments) | 2018 | arXiv:1812.11532 | N | low |
| 15 | Rolling-Shutter-Aware Differential SfM and Image Rectification | Bingbing Zhuang, Loong-Fah Cheong, Gim Hee Lee | ICCV 2017 (CVF page) | 2017 | arXiv:1903.03943 | Y https://github.com/ThomasZiegler/RS-aware-differential-SfM (GPL-3.0) | low |
| 16 | Self-Calibration-Based Approach to Critical Motion Sequences of Rolling-Shutter Structure From Motion | Eisuke Ito, Takayuki Okatani | CVPR 2017 (CVF page; arXiv comments empty) | 2017 | arXiv:1611.05476 | N | low |
| 17 | Degeneracies in Rolling Shutter SfM | Cenek Albl, Akihiro Sugimoto, Tomas Pajdla | ECCV 2016 (Springer LNCS 9909 pp 36-51) | 2016 | doi:10.1007/978-3-319-46454-1_3 | N | low |
| 18 | Rolling Shutter Camera Relative Pose: Generalized Epipolar Geometry | Yuchao Dai, Hongdong Li, Laurent Kneip | CVPR 2016 (CVF page; arXiv comments 'Accepted by CVPR 2016, this is an extended version') | 2016 | arXiv:1605.00475 | N | low |
| 19 | Sparse to Dense 3D Reconstruction From Rolling Shutter Images | Olivier Saurer, Marc Pollefeys, Gim Hee Lee | CVPR 2016 (CVF page) | 2016 | CVF CVPR2016 | N | low |
| 20 | R6P - Rolling Shutter Absolute Camera Pose | Cenek Albl, Zuzana Kukelova, Tomas Pajdla | CVPR 2015 (CVF page) | 2015 | CVF CVPR2015 | Y https://github.com/CenekAlbl/RnP (BSD-3-Clause) | low |
| 21 | Rolling Shutter Camera Calibration | Luc Oth, Paul Furgale, Laurent Kneip, Roland Siegwart | CVPR 2013 (CVF page) | 2013 | CVF CVPR2013 | Y https://github.com/ethz-asl/kalibr | low |
| 22 | Rolling Shutter Bundle Adjustment | Johan Hedborg, Per-Erik Forssén, Michael Felsberg, Erik Ringaby | CVPR 2012 (aggregator page; DOI 10.1109/CVPR.2012.6247831) | 2012 | doi:10.1109/CVPR.2012.6247831 | N | low |
| 23 | Geometric Models of Rolling-Shutter Cameras | Marci Meingast, Christopher Geyer, Shankar Sastry | arXiv (no acceptance stated on page) | 2005 | arXiv:cs/0503076 | N | low |

---

## 5. RS in SLAM, VO and VIO

**TUM RS-VIO** (classical_slam_gyro.md §1; datasets.md). Citation: *Rolling-Shutter Modelling for Direct Visual-Inertial Odometry*. David Schubert, Nikolaus Demmel, Lukas von Stumberg, Vladyslav Usenko, Daniel Cremers. IROS 2019 (DOI 10.1109/IROS40897.2019.8968539 on arXiv page). 2019. arXiv:1911.01015.
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

*Papers whose primary topic is 5 (17; generated from `related_work.csv`, fields as in the raw files).*

| # | Title | Authors | Venue (as recorded, with source) | Year | ID | Code | Threat |
|---|---|---|---|---|---|---|---|
| 1 | Breaking Time: A Fully Gaussian Framework for Distributed and Continuous-Time SLAM | Davide Ceriola, Simone Ferrari, Luca Di Giammarino, Leonardo Brizi, Giorgio Grisetti | RA-L (arXiv comments 'To be published in RA-L') | 2026 | arXiv:2606.06250 | Y https://github.com/rvp-group/gsolver (BSD-3-Clause) | low |
| 2 | Semantic SLAM with Rolling-Shutter Cameras and Low-Precision INS in Outdoor Environments | Yuchen Zhang, Miao Fan, Shengtong Xu, Xiangzeng Liu, Haoyi Xiong | IEEE IV 2025 (arXiv comments) | 2025 | arXiv:2504.01997 | N | low |
| 3 | Continuous-Time Spline Visual-Inertial Odometry | Jiawei Mo, Junaed Sattar | ICRA 2022 (arXiv comments) | 2022 | arXiv:2109.09035 | N | low |
| 4 | Ctrl-VIO: Continuous-Time Visual-Inertial Odometry for Rolling Shutter Cameras | Xiaolei Lang, Jiajun Lv, Jianxin Huang, Yukai Ma, Yong Liu, Xingxing Zuo | RA-L 2022 (arXiv journal-ref '2022 RAL') | 2022 | arXiv:2208.12008 | Y https://github.com/APRIL-ZJU/Ctrl-VIO (GPL-3.0) | low |
| 5 | Online Self-Calibration for Visual-Inertial Navigation Systems: Models, Analysis and Degeneracy | Yulin Yang, Patrick Geneva, Xingxing Zuo, Guoquan Huang | arXiv (comments do not state venue) | 2022 | arXiv:2201.09170 | N | low |
| 6 | Continuous-Time Spatiotemporal Calibration of a Rolling Shutter Camera-IMU System | Jianzhu Huai, Yuan Zhuang, Qicheng Yuan, Yukai Lin | arXiv (comments '11 pages, 9 figures') | 2021 | arXiv:2108.07200 | Y https://github.com/JzHuai0108/kalibr | low |
| 7 | Renormalization for Initialization of Rolling Shutter Visual-Inertial Odometry | Branislav Micusik, Georgios Evangelidis | arXiv (no comments) | 2020 | arXiv:2008.06399 | N | low |
| 8 | The WHU Rolling Shutter Visual-Inertial Dataset | Like Cao, Jie Ling, Xiaohui Xiao | IEEE Access 2020 (first page of author-hosted PDF) | 2020 | doi:10.1109/ACCESS.2020.2978589 | N | low |
| 9 | Rolling-Shutter Modelling for Direct Visual-Inertial Odometry | David Schubert, Nikolaus Demmel, Lukas von Stumberg, Vladyslav Usenko, Daniel Cremers | IROS 2019 (DOI 10.1109/IROS40897.2019.8968539 on arXiv page) | 2019 | arXiv:1911.01015 | N | low |
| 10 | Challenges in Monocular Visual Odometry: Photometric Calibration, Motion Bias and Rolling Shutter Effect | Nan Yang, Rui Wang, Xiang Gao, Daniel Cremers | RA-L 2018 (arXiv comments 'Accepted by IEEE RA-L, 2018') | 2018 | arXiv:1705.04300 | N | low |
| 11 | Direct Sparse Odometry with Rolling Shutter | David Schubert, Nikolaus Demmel, Vladyslav Usenko, Jörg Stückler, Daniel Cremers | ECCV 2018 (inferred from Springer DOI 10.1007/978-3-030-01237-3_42 shown on arXiv page; Springer page not opened) | 2018 | arXiv:1808.00558 | N | low |
| 12 | Monocular Visual Odometry with a Rolling Shutter Camera | Chang-Ryeol Lee, Kuk-Jin Yoon | arXiv (comments '14 pages, 8 figures') | 2017 | arXiv:1704.07163 | N | low |
| 13 | Direct semi-dense SLAM for rolling shutter cameras (title/venue from Schubert et al. 2019 reference list) | J.-H. Kim, C. Cadena, I. Reid | ICRA 2016 (UNVERIFIED; from citing paper's reference list) | 2016 | UNVERIFIED | N | low |
| 14 | A Spline-Based Trajectory Representation for Sensor Fusion and Rolling Shutter Cameras | Alonso Patron-Perez, Steven Lovegrove, Gabe Sibley | IJCV vol 113 pp 208-219 (Springer page) | 2015 | doi:10.1007/s11263-015-0811-3 | N | low |
| 15 | Dense Continuous-Time Tracking and Mapping with Rolling Shutter RGB-D Cameras | Christian Kerl, Jörg Stückler, Daniel Cremers | ICCV 2015 (CVF page) | 2015 | CVF ICCV2015 | N | low |
| 16 | Spline Fusion: A continuous-time representation for visual-inertial fusion with application to rolling shutter cameras | Steven Lovegrove, Alonso Patron-Perez, Gabe Sibley | BMVC 2013 (BMVA archive PDF) | 2013 | BMVC2013 paper0093 | N | low |
| 17 | UNVERIFIED: Li, Kim, Mourikis ICRA 2013 RS-camera VIO on a cellphone (exact title not confirmed) | M. Li, B. H. Kim, A. I. Mourikis (UNVERIFIED) | ICRA 2013 (UNVERIFIED) | 2013 | UNVERIFIED | N | low |

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

*Papers whose primary topic is 6 (14; generated from `related_work.csv`, fields as in the raw files).*

| # | Title | Authors | Venue (as recorded, with source) | Year | ID | Code | Threat |
|---|---|---|---|---|---|---|---|
| 1 | Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion | Otto Seiskari, Jerry Ylilammi, Valtteri Kaatrasalo, Pekka Rantalankila, Matias Turkulainen, Juho Kannala, et al. | ECCV 2024 (mlanthology + ECVA PDF URL; arXiv page states no venue) | 2024 | arXiv:2403.13327; DOI 10.1007/978-3-031-73209-6_10 | Y https://github.com/SpectacularAI/3dgs-deblur (Apache-2.0) | med |
| 2 | RS-NeRF: Neural Radiance Fields from Rolling Shutter Images | Muyao Niu, Tong Chen, Yifan Zhan, Zhuoxiao Li, Xiang Ji, Yinqiang Zheng | ECCV 2024 (arXiv comments) | 2024 | arXiv:2407.10267 | Y https://github.com/MyNiuuu/RS-NeRF (no licence) | med |
| 3 | URS-NeRF: Unordered Rolling Shutter Bundle Adjustment for Neural Radiance Fields | Bo Xu, Ziao Liu, Mengqi Guo, Jiancheng Li, Gim Hee Lee | ECCV 2024 (ECVA virtual poster page + project page) | 2024 | arXiv:2403.10119; DOI 10.1007/978-3-031-72761-0_26 (from search result) | Y https://github.com/ZiaoLiuS/URS-NERF (no licence) | med |
| 4 | USB-NeRF: Unrolling Shutter Bundle Adjusted Neural Radiance Fields | Moyang Li, Peng Wang, Lingzhe Zhao, Bangyan Liao, Peidong Liu | ICLR 2024 (proceedings.iclr.cc page) | 2024 | arXiv:2310.02687 | Y https://github.com/WU-CVGL/USB-NeRF (no licence) | med |
| 5 | Scanline-Aware Animatable Gaussian Avatars from Rolling-Shutter Videos | Youxiang Wang | arXiv | 2026 | arXiv:2608.17314 | N | low |
| 6 | TRGS-SLAM: IMU-Aided Gaussian Splatting SLAM for Blurry, Rolling Shutter, and Noisy Thermal Images | Spencer Carmichael, Katherine A. Skinner | arXiv (comments: project page only) | 2026 | arXiv:2603.20443 | Y https://github.com/umautobots/trgs_slam (MIT) | low |
| 7 | Unified Sensor Simulation for Autonomous Driving | Nikolay Patakin, Arsenii Shirokov, Anton Konushin, Dmitry Senushkin | arXiv | 2026 | arXiv:2602.05617 | UNVERIFIED | low |
| 8 | 3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting | Qi Wu, Janick Martinez Esturo, Ashkan Mirzaei, Nicolas Moenne-Loccoz, Zan Gojcic | CVPR 2025 (arXiv comments) | 2025 | arXiv:2412.12507 | Y https://github.com/nv-tlabs/3dgrut (Apache-2.0) | low |
| 9 | Every Camera Effect, Every Time, All at Once: 4D Gaussian Ray Tracing for Physics-based Camera Effect Data Generation | Yi-Ruei Liu, You-Zhe Xie, Yu-Hsiang Hsu, I-Sheng Fang, Yu-Lun Liu, Jun-Cheng Chen | NeurIPS 2025 Workshop SpaVLE (arXiv comments) | 2025 | arXiv:2509.10759 | N | low |
| 10 | Photoreal Scene Reconstruction from an Egocentric Device | Zhaoyang Lv, Maurizio Monge, Ka Chen, Yufeng Zhu, Michael Goesele, Jakob Engel, et al. | SIGGRAPH 2025 Conference Paper (arXiv comments) | 2025 | arXiv:2506.04444 | UNVERIFIED | low |
| 11 | Dense Scene Reconstruction from Light-Field Images Affected by Rolling Shutter | Hermes McGriff, Renato Martins, Nicolas Andreff, Cedric Demonceaux | arXiv | 2024 | arXiv:2412.03518 | N | low |
| 12 | SplatAD: Real-Time Lidar and Camera Rendering with 3D Gaussian Splatting for Autonomous Driving | Georg Hess, Carl Lindström, Maryam Fatemi, Christoffer Petersson, Lennart Svensson | arXiv (no venue on arXiv page; not verified) | 2024 | arXiv:2411.16816 | Y https://github.com/georghess/neurad-studio (Apache-2.0) | low |
| 13 | BAD-NeRF: Bundle Adjusted Deblur Neural Radiance Fields | Peng Wang, Lingzhe Zhao, Ruijie Ma, Peidong Liu | CVPR 2023 (CVF page) | 2023 | CVF CVPR2023 pp.4170-4179 | N | low |
| 14 | NeuRAD: Neural Rendering for Autonomous Driving | Adam Tonderski, Carl Lindström, Georg Hess, William Ljungbergh, Lennart Svensson, Christoffer Petersson | arXiv (no venue on arXiv page; not verified) | 2023 | arXiv:2311.15260 | Y https://github.com/georghess/neurad-studio (Apache-2.0) | low |

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

Citation: *Learning Structure-And-Motion-Aware Rolling Shutter Correction*. Bingbing Zhuang, Quoc-Huy Tran, Pan Ji, Loong-Fah Cheong, Manmohan Chandraker. CVPR 2019 (CVF page). 2019. CVF CVPR 2019 pp. 4551-4560 (DOI not recorded). No arXiv version found; no official code found.

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

*Papers whose primary topic is 7 (11; generated from `related_work.csv`, fields as in the raw files).*

| # | Title | Authors | Venue (as recorded, with source) | Year | ID | Code | Threat |
|---|---|---|---|---|---|---|---|
| 1 | Image as an IMU: Estimating Camera Motion from a Single Motion-Blurred Image | Jerred Chen, Ronald Clark | ICCV 2025 (Oral) [from project page + GitHub README; arXiv comments field lists only the project page] | 2025 | arXiv:2503.17358 | Y https://github.com/jerredchen/image-as-an-imu (MIT) | med |
| 2 | Rolling Shutter Image as an IMU: Estimating Camera Motion from a Single Rolling Shutter Image | Prashanth Shyamala | University of Toronto CSC2529 course project report (not peer reviewed; no arXiv found) | 2025 | none (course report; no arXiv ID or DOI found) | N | med |
| 3 | IMU-Assisted Learning of Single-View Rolling Shutter Correction | Jiawei Mo, Md Jahidul Islam, Junaed Sattar | CoRL 2021 = PMLR vol 164 (2022) pp 861-870 (PMLR page; classical slice). arXiv v1 2020; scoop/rs2d slices saw no venue on the arXiv page | 2022 | arXiv:2011.03106 | Y https://github.com/IRVLab/unrolling (no licence) | med |
| 4 | Learning Structure-And-Motion-Aware Rolling Shutter Correction | Bingbing Zhuang, Quoc-Huy Tran, Pan Ji, Loong-Fah Cheong, Manmohan Chandraker | CVPR 2019 (CVF page) | 2019 | CVF CVPR 2019 pp. 4551-4560 (DOI not recorded) | N | med |
| 5 | Estimating Velocity and Spin of Spherical Objects from Rolling-Shutter Image(s) | Wenjie Xue, Jun Yang, Jingmin Wang, Limin Shang | ECCV 2026 (ECCV 2026 accepted-papers list; arXiv has no comments) | 2026 | arXiv:2606.31760 | N | low |
| 6 | Roll2Depth: Zero-Shot Metric Depth Estimation Exploiting Camera's Rolling Shutter Effect | Rui Xiao, Jinming Xu, Jinsong Han | NeurIPS 2026 (poster page; track not stated) | 2026 | OpenReview OsxTtWnTJW | N | low |
| 7 | A Robust Method for Strong Rolling Shutter Effects Correction Using Lines With Automatic Feature Selection | Yizhen Lao, Omar Ait-Aider | CVPR 2018 (CVF page) | 2018 | CVF CVPR2018 | N | low |
| 8 | Rolling Shutter Correction in Manhattan World | Pulak Purkait, Christopher Zach, Ales Leonardis | ICCV 2017 (CVF page) | 2017 | CVF ICCV2017 | N | low |
| 9 | Global Optimization of Object Pose and Motion from a Single Rolling Shutter Image with Automatic 2D-3D Matching | Ludovic Magerand, Adrien Bartoli, Omar Ait-Aider, Daniel Pizarro | ECCV 2012 (Springer LNCS 7572 pp 456-469) | 2012 | doi:10.1007/978-3-642-33718-5_33 | N | low |
| 10 | Kinematics From Lines in a Single Rolling Shutter Image | Omar Ait-Aider, Adrien Bartoli, Nicolas Andreff | CVPR 2007 (listed on EnCoV group publications page only) | 2007 | (no DOI checked) | N | low |
| 11 | Simultaneous Object Pose and Velocity Computation Using a Single View from a Rolling Shutter Camera | Omar Ait-Aider, Nicolas Andreff, Jean Marc Lavest, Philippe Martinet | ECCV 2006 (Springer LNCS 3952 pp 56-68) | 2006 | doi:10.1007/11744047_5 | N | low |

---

## 8. Gyroscope-based RS correction and phone stabilisation

**"Use the gyroscope" has been published since 2010–2011** (classical_slam_gyro.md §1).
- **Karpenko et al., Stanford CSTR 2011-03** (read via Wayback). Citation: *Digital Video Stabilization and Rolling Shutter Correction using Gyroscopes*. Alexandre Karpenko, David Jacobs, Jongmin Baek, Marc Levoy. Stanford CS Tech Report CSTR 2011-03 (September 2011). 2011. CSTR 2011-03.
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

*Papers whose primary topic is 8 (9; generated from `related_work.csv`, fields as in the raw files).*

| # | Title | Authors | Venue (as recorded, with source) | Year | ID | Code | Threat |
|---|---|---|---|---|---|---|---|
| 1 | Deep Online Fused Video Stabilization | Zhenmei Shi, Fuhao Shi, Wei-Sheng Lai, Chia-Kai Liang, Yingyu Liang | WACV 2022 (project page + repo README) | 2022 | arXiv:2102.01279 | Y https://github.com/googleinterns/deep-stabilization (Apache-2.0) | low |
| 2 | DeepOIS: Gyroscope-Guided Deep Optical Image Stabilizer Compensation | Haipeng Li, Shuaicheng Liu, Jue Wang | IEEE TCSVT vol 32 no 5, May 2022 (arXiv journal-ref) | 2022 | arXiv:2101.11183 | Y https://github.com/lhaippp/DeepOIS (MIT) | low |
| 3 | Gyroscope-aided Relative Pose Estimation for Rolling Shutter Cameras | Chang-Ryeol Lee, Ju Hong Yoon, Min-Gyu Park, Kuk-Jin Yoon | arXiv (no comments) | 2019 | arXiv:1904.06770 | N | low |
| 4 | Fused Video Stabilization on the Pixel 2 and Pixel 2 XL (blog) | Chia-Kai Liang, Fuhao Shi | Google Research Blog, 10 Nov 2017 | 2017 | (blog) | N | low |
| 5 | Inertial-aided Rolling Shutter Relative Pose Estimation | Chang-Ryeol Lee, Kuk-Jin Yoon | arXiv (no comments) | 2017 | arXiv:1712.00184 | N | low |
| 6 | A Non-Linear Filter for Gyroscope-Based Video Stabilization | Steven Bell, Alejandro Troccoli, Kari Pulli | ECCV 2014 (Springer LNCS 8692 pp 294-308) | 2014 | doi:10.1007/978-3-319-10593-2_20 | N | low |
| 7 | Efficient Video Rectification and Stabilisation for Cell-Phones | Erik Ringaby, Per-Erik Forssén | IJCV vol 96 pp 335-352 (2012; published online 10 June 2011) (Springer page) | 2012 | doi:10.1007/s11263-011-0465-8 | N | low |
| 8 | Digital Video Stabilization and Rolling Shutter Correction using Gyroscopes | Alexandre Karpenko, David Jacobs, Jongmin Baek, Marc Levoy | Stanford CS Tech Report CSTR 2011-03 (September 2011) | 2011 | CSTR 2011-03 | Y https://github.com/alexgo1/Video-Stabilization (GPL-3.0) | low |
| 9 | Rectifying Rolling Shutter Video from Hand-Held Devices | Per-Erik Forssén, Erik Ringaby | CVPR 2010 (ML Anthology; DOI 10.1109/CVPR.2010.5540173) | 2010 | doi:10.1109/CVPR.2010.5540173 | N | low |

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

*Papers whose primary topic is 9 (8; generated from `related_work.csv`, fields as in the raw files).*

| # | Title | Authors | Venue (as recorded, with source) | Year | ID | Code | Threat |
|---|---|---|---|---|---|---|---|
| 1 | Moment-Reenacting: Inverse Motion Degradation with Cross-shutter Guidance | Xiang Ji, Guixu Lin, Zhengwei Yin, Jiancheng Zhao, Yinqiang Zheng | IEEE TPAMI (arXiv comments: "Accepted by TPAMI") | 2026 | arXiv:2605.22423 | Y https://github.com/jixiang2016/stereoBR (MIT) | low |
| 2 | RnGCam: High-speed video from rolling & global shutter measurements | Kevin Tandi, Xiang Dai, Chinmay Talegaonkar, Gal Mishne, Nick Antipa | arXiv (rs2d slice: no venue on arXiv page). The scoop slice lists RnGCam among ICCV 2025 CVF titles; venue UNVERIFIED until a proceedings page is opened | 2025 | arXiv:2509.18087 | N | low |
| 3 | Event-based Shutter Unrolling and Motion Deblurring in Dynamic Scenes | Yangguang Wang, Chenxu Jiang, Xu Jia, Yufei Guo, Lei Yu (from search snippet) | IEEE Signal Processing Letters (UNVERIFIED) | 2024 | DOI 10.1109/lsp.2024.3381894 (UNVERIFIED, from search snippet) | N | low |
| 4 | Motion Blur Decomposition with Cross-shutter Guidance | Xiang Ji, Haiyang Jiang, Yinqiang Zheng | CVPR 2024 (arXiv comments + CVF page) | 2024 | arXiv:2404.01120; CVF CVPR2024 pp.12534-12543 | Y https://github.com/jixiang2016/dualBR (MIT) | low |
| 5 | Neural Image Re-Exposure | Xinyu Zhang, Hefei Huang, Xu Jia, Dong Wang, Huchuan Lu | arXiv (no acceptance stated) | 2023 | arXiv:2305.13593 | N | low |
| 6 | Towards Rolling Shutter Correction and Deblurring in Dynamic Scenes | Zhihang Zhong, Yinqiang Zheng, Imari Sato | CVPR 2021 (arXiv comments + CVF page) | 2021 | arXiv:2104.01601; CVF CVPR2021 pp.9219-9228 | Y https://github.com/zzh-tech/RSCD (MIT) | low |
| 7 | Going Unconstrained With Rolling Shutter Deblurring | Mahesh Mohan M. R., A. N. Rajagopalan, Gunasekaran Seetharaman | ICCV 2017 (CVF page) | 2017 | CVF ICCV2017 pp.4010-4018 | N | low |
| 8 | Rolling Shutter Motion Deblurring | Shuochen Su, Wolfgang Heidrich | CVPR 2015 (CVF page) | 2015 | CVF CVPR2015 pp.1529-1537 | N | low |

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

*Papers whose primary topic is 12 (6; generated from `related_work.csv`, fields as in the raw files).*

| # | Title | Authors | Venue (as recorded, with source) | Year | ID | Code | Threat |
|---|---|---|---|---|---|---|---|
| 1 | VGGT-Omega (VGGT-Ω) | Jianyuan Wang, Minghao Chen, Shangzhan Zhang, Nikita Karaev, Johannes Schonberger, Patrick Labatut et al. | CVPR 2026 Oral (source: arXiv comments) | 2026 | arXiv:2605.15195 | Y https://github.com/facebookresearch/vggt-omega | low |
| 2 | Continuous 3D Perception Model with Persistent State (CUT3R) | Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A. Efros, Angjoo Kanazawa | CVPR 2025 Oral (source: project page cut3r.github.io; arXiv page shows no venue; CVF page returned HTTP 403) | 2025 | arXiv:2501.12387 | Y https://github.com/CUT3R/CUT3R (CC BY-NC-SA 4.0) | low |
| 3 | Depth Anything 3: Recovering the Visual Space from Any Views | Haotong Lin, Sili Chen, Junhao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi et al. | ICLR 2026 Oral (source: iclr.cc/virtual/2026/oral/10006532; arXiv page shows no venue) | 2025 | arXiv:2511.10647 | Y https://github.com/ByteDance-Seed/Depth-Anything-3 (Apache-2.0) | low |
| 4 | pi^3: Permutation-Equivariant Visual Geometry Learning | Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li et al. | ICLR 2026 (sources: GitHub repo description and mlanthology.org; OpenReview forum DTQIjngDta blocked by a browser check) | 2025 | arXiv:2507.13347 | Y https://github.com/yyfz/Pi3 (BSD-3-Clause) | low |
| 5 | Streaming 4D Visual Geometry Transformer (StreamVGGT) | Dong Zhuo, Wenzhao Zheng, Jiahe Guo, Yuqi Wu, Jie Zhou, Jiwen Lu | ICLR 2026 poster as 'Streaming Visual Geometry Transformer' (source: iclr.cc/virtual/2026/poster/10011495) | 2025 | arXiv:2507.11539 | Y https://github.com/wzzheng/StreamVGGT (CC BY-NC-SA 4.0) | low |
| 6 | VGGT: Visual Geometry Grounded Transformer | Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, David Novotny | CVPR 2025 (source: arXiv comments) | 2025 | arXiv:2503.11651 | Y https://github.com/facebookresearch/vggt | low |

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
| 1 | *Foundation4D: Towards a unified perception and reconstruction model for autonomous vehicles*. Sergio Agostinho, Maximilian Apodaca, Emmanuel Attia, Guillem Braso, George Dong, Ilya Fedorov, et al. (NVIDIA). arXiv. 2026. arXiv:2610.30340 | Feed-forward multi-view FM; time-varying ego trajectory; RS exposure boundaries used to query poses | RS appears only in label generation and downstream 3DGS rendering. The network still predicts one pose and a GS depth/pointmap per image, for a calibrated driving rig. We put row time into the pointmap itself (per-row unprojection with per-frame ω, v), for generic handheld cameras, images only. |
| 2 | *Image as an IMU: Estimating Camera Motion from a Single Motion-Blurred Image*. Jerred Chen, Ronald Clark. ICCV 2025 (Oral) [from project page + GitHub README; arXiv comments field lists only the project page]. 2025. arXiv:2503.17358 | Images-only instantaneous ω, v of the camera | Single image; the signal comes from motion blur; it states it does not model RS; it outputs velocity only. We use RS row timing across many views inside a 3D FM and feed ω, v back into the geometry. |
| 3 | *Rolling Shutter Image as an IMU: Estimating Camera Motion from a Single Rolling Shutter Image*. Prashanth Shyamala. University of Toronto CSC2529 course project report (not peer reviewed; no arXiv found). 2025. none (course report; no arXiv ID or DOI found) | Images-only ω, v from RS distortion | Single image; two stages (RS→GS flow + mono depth, then least squares); needs the readout time; not peer-reviewed. We are end-to-end inside π³/VGGT with LoRA, with multi-view pose output. |
| 4 | *Interp3R: Continuous-time 3D Geometry Estimation with Frames and Events*. Shuang Guo, Filbert Febryanto, Lei Sun, Luc Van Gool, Guillermo Gallego. arXiv. 2026. arXiv:2603.14528 | Pointmap FM queried at an arbitrary time; joint depth + pose | Needs an event camera; its time is between frames, not within a frame's readout; zero RS mentions in its full text. We use images only and intra-frame row time. |
| 5 | *Anchor Points for Monocular Depth Estimation Under Severe Rolling Shutter Ego-motion*. Jacob Carter, Vaishnav Ramesh, Md Jahidul Islam, Sanjeev J. Koppal. UNVERIFIED (lab page lists it under 2026; no venue or arXiv given). 2026. none found (no arXiv ID or venue listed on the project page or PDF) | RS + a foundation model; a reference ("anchor") row time | Image-space de-rolling (Marigold U-Net) before a monocular depth FM; single view; no pose or velocity; needs RS-GS paired training data. We never undistort images, and we output multi-view poses. |
| 6 | *Point4Cast: Streaming Dynamic Scene Reconstruction and Forecasting*. Xinhang Liu, Pedro Miraldo, Suhas Lohit, Huaizu Jiang, Naoko Sawada, Yu-Wing Tai, et al. CVPR 2026 (CVF open access, pp. 14600-14611). 2026. none found (no arXiv); MERL TR2026-077 | Time-conditioned pointmaps and cameras from a feed-forward model | Time indexes frames of a dynamic scene, and each frame is still treated as GS. We index rows inside a frame to absorb camera motion during readout. |

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
| A1 | DFRSC | *Rolling Shutter Correction with Intermediate Distortion Flow Estimation*. Mingdeng Cao, Sidi Yang, Yujiu Yang, Yinqiang Zheng. CVPR 2024 (arXiv comments + CVF page). 2024. arXiv:2404.06350; CVF CVPR2024 pp.25338-25347 | https://github.com/ljzycmd/DFRSC (release `ckpts`) | none (no LICENSE file); research baseline only, ask before redistributing |
| A2 | JAMNet | *Joint Appearance and Motion Learning for Efficient Rolling Shutter Correction*. Bin Fan, Yuxin Mao, Yuchao Dai, Zhexiong Wan, Qi Liu. CVPR 2023 (CVF page). 2023. CVF CVPR2023 pp.5671-5681 (no arXiv found) | https://github.com/GitCVfb/JAMNet (weights in repo) | "research purpose only" statement, no licence |
| A3 | AdaRSC (BS-RSC) | *Learning Adaptive Warping for Real-World Rolling Shutter Correction*. Mingdeng Cao, Zhihang Zhong, Jiahao Wang, Yinqiang Zheng, Yujiu Yang. CVPR 2022 (arXiv comments + CVF page). 2022. arXiv:2204.13886; CVF CVPR2022 pp.17785-17793 | https://github.com/ljzycmd/BSRSC (release v1.0.0) | MIT |
| A4 | LBCNet (fallbacks RSSR, CVR) | *Learning Bilateral Cost Volume for Rolling Shutter Temporal Super-Resolution*. Bin Fan, Yuchao Dai, Hongdong Li. IEEE TPAMI 2024 (per project page npucvr.github.io/LBCNet; IEEE page not opened). 2024. IEEE doc 10382595 (from project page). Fallbacks: *Inverting a Rolling Shutter Camera: Bring Rolling Shutter Images to High Framerate Global Shutter Video*. Bin Fan, Yuchao Dai. ICCV 2021 (CVF page). 2021. CVF ICCV2021 pp.4228-4237; *Context-Aware Video Reconstruction for Rolling Shutter Cameras*. Bin Fan, Yuchao Dai, Zhiyuan Zhang, Qi Liu, Mingyi He. CVPR 2022 (arXiv comments + CVF page). 2022. arXiv:2205.12912; CVF CVPR2022 pp.17572-17582 | https://github.com/GitCVfb/LBCNet ; https://github.com/GitCVfb/RSSR ; https://github.com/GitCVfb/CVR | research-only statements, no licence |
| A5 | DeepUnrollNet | *Deep Shutter Unrolling Network*. Peidong Liu, Zhaopeng Cui, Viktor Larsson, Marc Pollefeys. CVPR 2020 (CVF page). 2020. CVF CVPR2020 pp.5941-5949 (not on arXiv) | https://github.com/ethliup/DeepUnrollNet (`download_pretrained_models.sh`) | MIT |
| A6 | JCD (blurry sequences only) | *Towards Rolling Shutter Correction and Deblurring in Dynamic Scenes*. Zhihang Zhong, Yinqiang Zheng, Imari Sato. CVPR 2021 (arXiv comments + CVF page). 2021. arXiv:2104.01601; CVF CVPR2021 pp.9219-9228 | https://github.com/zzh-tech/RSCD (`checkpoints.zip`) | MIT |

**B. Classical**

| # | Method | Citation | Code | Licence |
|---|---|---|---|---|
| B1 | COLMAP and/or GLOMAP on the RS frames (and on GS cam0) as the non-RS SfM reference | no paper in the raw files | repo URLs are not recorded in the raw slice; it cloned COLMAP at `98f3154` (2026-10-28) and GLOMAP at `99806d0` (2026-01-30). Neither has any RS support (0 grep matches) | COLMAP: BSD (new BSD); GLOMAP: BSD-style (ETH 2024) |
| B2 | Barath 7-AC RS relative pose | *Rolling Shutter Relative Pose Estimation Made Practical*. Daniel Barath. ECCV 2026 (ECCV 2026 accepted-papers list and repo README per scoop slice; Google Research publication page per classical slice; arXiv comments field empty). 2026. arXiv:2606.26863 | https://github.com/danini/rolling_shutter_made_practical (`examples/download_tum_rs.sh`, `examples/run_real_data.py`; the real-data path needs CUDA, torch, kornia, RoMa) | MIT |

**C. Ours and controls**

| # | Method | What it is | Code | Licence |
|---|---|---|---|---|
| C1 | Self-estimate two-pass (training-free) | Run the model on raw RS, estimate ω (and v) from its own neighbouring-frame poses, unroll, then run again | our repo (`src/tum_conditions.py` for TUM, `src/synth_conditions.py` for synthetic; GPU unroller `src/gpu_unroll.py`) | ours |
| C2 | Gyro straightening (privileged: needs a gyro) | EXP1 E1a: rotation-only per-row unroll from the gyro, Karpenko-style. It closes 82–99% of the fast-clip ATE gap (measured). Classical form: *Digital Video Stabilization and Rolling Shutter Correction using Gyroscopes*. Alexandre Karpenko, David Jacobs, Jongmin Baek, Marc Levoy. Stanford CS Tech Report CSTR 2011-03 (September 2011). 2011. CSTR 2011-03 | our repo (`src/unroll.py`, `src/gpu_unroll.py`) | ours |
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
