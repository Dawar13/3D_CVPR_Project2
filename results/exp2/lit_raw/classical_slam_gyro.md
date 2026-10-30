# Lit review slice: classical_slam_gyro (Topics 4, 5, 7, 8)

Date of search: 2026-10-29. One row per paper is in `classical_slam_gyro.csv` (66 rows).

**How the papers were checked**
- Every listed paper's own page was opened: arXiv abs pages with WebFetch, and CVF, ECVA, Springer, ISPRS and Wayback pages with `curl` (CVF returns HTTP 403 to WebFetch).
- A few were checked only through an aggregator (ML Anthology) or the authors' group page. The CSV `verified` column says which.
- Venues are written as "arXiv" unless a page states acceptance. The venue column names the page the venue came from.
- Code facts (license, language, entry points) come from each repo's README, its LICENSE file, the GitHub API, or a grep of a shallow clone (commit hashes below).

---

## 1. Findings in plain English

### Topic 4: classical RS geometry

**Bundle adjustment (BA)**
- RS bundle adjustment is a mature line of work:
  - Hedborg et al., CVPR 2012;
  - Saurer et al., CVPR 2016;
  - NW-RSBA (Liao et al.). This is **CVPR 2023, not ICCV 2023** as the seed list said; both the CVF page (pp. 4863-4871) and the arXiv comment say CVPR 2023;
  - RSL-BA (ECCV 2024);
  - "Rolling Shutter Camera Self-Calibration" (Zhang, …, Bartoli), listed as ECCV 2026 on the authors' group page. It estimates intrinsics and the readout ratio from image sequences alone with a self-calibrating BA.
- All of these are per-scene optimisation methods.
- The NW-RSBA and RSL-BA releases are MATLAB demos on synthetic scenes, not SfM pipelines.

**Minimal solvers**
- Absolute pose:
  - R6P (CVPR 2015);
  - linear R6P variants (ACCV 2018);
  - R7Pf/R7Pfr with unknown focal length and radial distortion (ECCV 2020).
  - All are packaged in `CenekAlbl/RnP` (BSD-3).
- Relative pose:
  - generalized epipolar geometry (Dai et al., CVPR 2016: 20-point linear and 44-point uniform solvers);
  - Hruby & Pollefeys, single-scanline (ICCV 2025);
  - Su et al., asynchronous tracks (ICCV 2025);
  - **Barath, arXiv 2606.26863**. Its arXiv comments field is empty. The Google Research publication page lists it as **ECCV 2026**. It recovers relative pose plus RS motion from 7 affine correspondences, reports the best accuracy on the TUM RS benchmark, and is the only one in this list with a TUM RS run script (MIT).
- Theory:
  - order-one RS cameras (arXiv 2024);
  - Single-View RS SfM (arXiv 2026), which sets out which motion and structure parameters can be recovered from one RS image.

**Degeneracies**
- Albl et al. (ECCV 2016): common capture setups with parallel readout directions are critical, and BA near them gives deformed reconstructions.
- Ito & Okatani (CVPR 2017): critical motion sequences under a linearised rotation-only RS model.
- Zhuang et al. (CVPR 2019, SMARSC paper): **RS two-view geometry is degenerate under pure translation.**
- Together these are the main theory a reviewer will raise against "velocity from images". Normal handheld video has parallel readout directions, and pure translation is common.

**Off-the-shelf SfM**
- COLMAP, GLOMAP, Theia and OpenMVG have **no RS support at all** (grep results in Section 4).
- RS-aware multi-view SfM exists only in commercial tools:
  - Agisoft Metashape Pro: an "Enable rolling shutter compensation" checkbox;
  - Pix4Dmapper: the "Linear Rolling Shutter" shutter model.

### Topic 5: RS in SLAM, VO and VIO

**TUM RS-VIO dataset (Schubert et al., IROS 2019)**
- Facts confirmed from the paper PDF and the dataset page:
  - two IDS uEye UI-3241LE-M-GL cameras with Lensagon BM4018S118 lenses;
  - **left camera in global-shutter mode, right camera in rolling-shutter mode**, triggered together at 20 Hz;
  - 1280×1024 grayscale, 16-bit, linear response;
  - row time "approximately 29.47 µs". The dataset page gives 29.4737 µs, "very well approximated by the step size of the exposure time" because it "can't be read directly". 1024 rows × 29.4737 µs ≈ 30.2 ms readout (my arithmetic);
  - Bosch BMI160 accelerometer and gyroscope at 200 Hz, hardware-synchronised;
  - OptiTrack Flex13 motion capture at 120 Hz. Outliers are median-filtered and poses linearly interpolated. Time alignment is done by matching mocap and gyro rotational velocities;
  - 10 sequences (seq1–seq10, 30–57 m long). The calibration sequences were recorded with both cameras in GS mode.
- Timestamp convention, quoted from the dataset page: "timestamps denote the middle of the exposure interval" and "the timestamps for the rolling-shutter images refer to the first row".

**Classical RS-aware odometry**
- Spline Fusion (BMVC 2013) and its IJCV 2015 journal version; Kerl et al. (ICCV 2015, RGB-D).
- RS-DSO (ECCV 2018, inferred from the Springer DOI) and RS direct VIO (IROS 2019).
- Lee & Yoon RS-VO (arXiv 2017).
- Ctrl-VIO (RA-L 2022; code released, ROS, includes a TUM-RSVI config).
- VIO self-calibration of readout time (Yang et al., 2022; Huai et al., 2021).
- Newer work (2024–2026) is continuous-time back-ends and 3DGS/SLAM systems. All of it uses an IMU or optimises per scene:
  - GS on the Move (ECCV 2024, VIO velocities);
  - TRGS-SLAM (2026, thermal + IMU);
  - Breaking Time (RA-L 2026);
  - Semantic SLAM with an RS camera and INS (IV 2025).
- **I found no learned feed-forward VO/SLAM that models RS from images alone.**
- VINS-Mono has a first-order RS option. OpenVINS and Basalt have none (Section 4).

### Topic 7: camera velocity from a single RS image

**Classical precedents** (need a known 3D model or specific scene structure):
- Ait-Aider et al. (ECCV 2006): pose and velocity of an object from one RS image using 2D-3D matches;
- Ait-Aider, Bartoli & Andreff (CVPR 2007): kinematics from lines;
- Magerand et al. (ECCV 2012);
- Purkait et al. (ICCV 2017, Manhattan world);
- Lao & Ait-Aider (CVPR 2018, lines).

**Learned**
- Rengarajan et al. (CVPR 2017): 2-DOF row-wise motion.
- **SMARSC** (Zhuang et al., CVPR 2019). Details in Section 2.
- Mo et al. (CoRL 2021): single image plus IMU.
- The "image as IMU" idea is now in the air:
  - Chen & Clark, *Image as an IMU* (ICCV 2025 Oral, MIT code): angular and translational velocity from one motion-blurred image;
  - a 2025 U-Toronto course report, *Rolling Shutter Image as an IMU*, applies the same recipe to RS images (not peer-reviewed).
- Carter et al. (2026, venue not stated) fine-tune depth foundation models (Marigold-based) for monocular depth under strong RS. They argue that the time that minimises distortion is the mid-point of the exposure.
- None of these is multi-view, and none outputs poses plus a global pointmap.

### Topic 8: gyroscope-based RS correction

**Karpenko et al., Stanford CSTR 2011-03** (read via Wayback, because the live site refused connections):
- models camera motion **as rotation only**, and says so: "We ignore translations because they are difficult to measure accurately using IMUs";
- auto-calibrates gyro drift and delay, focal length and RS speed from one capture;
- builds a **synthetic smooth global-shutter camera** (t_s = 0, Gaussian-smoothed rotations);
- warps each input row to it, using a mesh with about 10 vertical subdivisions on the GPU;
- crops the result.

Bell et al. (ECCV 2014), Forssén & Ringaby (CVPR 2010; IJCV 2012) and the Pixel 2 "Fused Video Stabilization" blog (2017) follow the same pattern. The blog uses gyro rotation plus measured OIS x/y and focus-z lens motion, splits the frame into a mesh, and applies "multiple transformations for each frame" to correct RS. Deep-FVS (WACV 2022) warps the sensor-measured real motion toward an LSTM-predicted virtual pose and says this corrects RS without wobble.

Gyro-aided learned work: DeepOIS (TCSVT 2022), RS-Diffusion (arXiv 2024; gyro used to make the GS ground truth), Mo et al. (CoRL 2021, IMU in pose refinement), and gyro-aided RS relative-pose solvers (Lee et al. 2017 and 2019).

**What this means for stabilised phone video** (my inference from the sources above; not stated as such in any of them):
1. EIS output is **not global-shutter**. Rotation-induced RS is mostly removed, to the accuracy of the gyro sync, the readout calibration and the mesh resolution. **Translation-induced, depth-dependent RS stays**, because no system models translation.
2. Each output frame is a per-frame mesh/homography warp plus a crop toward a smoothed virtual rotation. With OIS, it also includes a lens shift. So the effective intrinsics and principal point change from frame to frame, and the output rotation is the *smoothed virtual* rotation, not the true one.
3. **Row-time geometry is reindexed.** After the warp, output row *y* no longer maps to a single sensor readout time; the time of a pixel is set by the inverse warp. Deep-FVS also notes that the OIS offset "is different at each scanline".
4. For RS-3D:
   - on stabilised footage, a per-row (ω, v) model should expect ω ≈ 0 apart from residual smoothing error, while v still matters;
   - "row index = time" does not strictly hold;
   - a fair evaluation needs unstabilised capture (as on TUM RS-VIO), or stabilisation must be listed as a separate condition.

---

## 2. SMARSC exactly (Zhuang, Tran, Ji, Cheong, Chandraker, CVPR 2019), and what we add

Read from the CVF PDF.

**Inputs**
- A **single RS image** from an **intrinsically calibrated** camera.
- Real test images (Logitech C920) were "resized and cropped to match the image size and intrinsic parameters of training data".
- No IMU, no readout time given as input.

**Predicts**
- **Depth-Net** (DispNet, regresses inverse depth): an RS depth map.
- **Velocity-Net** (ResNet-34 plus conv layers, then a 1×1 conv to 6 outputs): **one constant 6-DOF velocity per image**, a translation v ∈ R³ and a rotation ω ∈ so(3).
- Both are expressed **per scanline**, with per-scanline pose P_i = [exp((i−1)ω) | −(i−1)v].
- Depth and velocity are then used to project each pixel to the first-scanline GS canvas, which gives the rectified image.

**Assumptions**
- Constant velocity during the exposure.
- Static scene.
- Motion is in per-scanline units, so the training camera's row timing is built in. My inference: transferring to another readout needs rescaling; the paper does not discuss this.
- An identified ambiguity between pitch rotation and vertical resizing, so images must be cropped, not resized.

**Training**
- Synthetic RS images rendered from KITTI Raw: stereo depth, sampled 6-DOF velocities, plus small random homographies to add pitch and roll.
- L1 losses on inverse depth, v, ω and a photometric term.
- The two networks are trained separately.
- No fine-tuning on real data.

**Evaluation**
- Undistortion-flow error on 200 synthetic KITTI-derived images.
- Qualitative results on real C920 images.
- A two-step SfM test: SMARSC, then VisualSFM on 47 real RS images.

**Stated future work**
- "exploit multiple views and consider other artifacts such as motion blurs".

**Code**
- No official repository found.

**What RS-3D adds beyond SMARSC**
1. **Multi-view and feed-forward.** N frames are processed jointly. Cross-view parallax constrains (ω, v), instead of a single-view appearance prior. This is the "multiple views" extension that SMARSC itself names as future work.
2. **Outputs are geometry, not a rectified image.** We output a global pointmap and camera poses for every frame. SMARSC outputs per-image depth plus a velocity that it uses only to warp pixels; it gives no inter-view poses and no shared 3D frame.
3. **Row-time-aware unprojection inside the 3D head.** No image resampling or inpainting, so no occlusion holes from the GS warp.
4. **Built on a pretrained 3D foundation model** (π³/VGGT-class) with a light LoRA, rather than a CNN trained from scratch on synthetic KITTI. Unknown or different intrinsics are handled by the backbone, not by forcing the training crop.
5. **Real-data metric evaluation.** Pose ATE/RPE and (ω, v) against mocap and gyro ground truth on TUM RS-VIO, with GS-twin control. SMARSC's real-data results are qualitative only.
6. **Readout handled explicitly** (given or estimated) rather than baked into per-scanline units.
7. **A risk to address:** SMARSC's own result says RS two-view geometry is degenerate under pure translation. Our multi-view model inherits this ambiguity in principle, which matches our ω–v collinearity note from EXP0. We should report pure-translation segments separately.

---

## 3. Threats (relative to the planned method)

- **No direct threat found.**
  - arXiv API search over abstracts, `"rolling shutter"` AND {VGGT, DUSt3R, MASt3R, feed-forward, foundation, pointmap}, returned nothing relevant.
  - Two targeted web searches on feed-forward RS 3D (Topic 7 queries 6 and 8) also found nothing.
  - Caveat: the arXiv abstract search returned at most 150 entries for plain "rolling shutter", so coverage is not exhaustive. Other slices own the NeRF/3DGS and learned-correction literature.
- **Medium threats** (must cite, and compare where possible):
  - SMARSC: closest concept.
  - Barath, ECCV 2026: image-only RS pose plus velocity on TUM RS, with runnable code. This is the classical number we must beat or match on (ω, v).
  - RS Camera Self-Calibration, ECCV 2026: image-only readout and intrinsics from sequences.
  - Image as an IMU, ICCV 2025: same "velocity from a single image" framing.
  - The U-Toronto RS-as-IMU report: shows the RS version of that idea exists, but it is not peer-reviewed.
  - Carter et al. 2026: fine-tunes a foundation model under RS, but depth only.
- **Everything else is low:** IMU-based, per-scene optimisation, or two-view/minimal solvers.

---

## 4. Runnable classical baselines

### Repo grep results

Shallow clones taken 2026-10-29, each grepped for `rolling.?shutter|RollingShutter|readout|line.?delay`.

| Tool | Commit / date | RS support? | Exact file / flag | License | Needs |
|---|---|---|---|---|---|
| COLMAP | `98f3154` 2026-10-28 | **No** (0 matches) | — | BSD (new BSD) | C++ |
| GLOMAP | `99806d0` 2026-01-30 | **No** (0 matches) | — | BSD-style (ETH 2024) | C++ |
| TheiaSfM | `d2112f1` 2019-07-16 | **No** | — | BSD-style (UC Regents) | C++ |
| OpenMVG | `6887e73` 2026-08-30 | **No** | — | MPL-2.0 | C++ |
| Kalibr (ethz-asl) | `1f60227` 2024-03-08 | **Yes, calibration only** | `aslam_offline_calibration/kalibr/python/kalibr_calibrate_rs_cameras --model pinhole-radtan-rs\|pinhole-equi-rs --frame-rate N --target … --bag … --topic …`; `kalibr_rs_camera_calibration/RsCalibrator.py`; `aslam_cv/aslam_cameras/include/aslam/cameras/RollingShutter.hpp` (`lineDelay`). Implements Oth et al. CVPR 2013 (README, PR #65, Aug 2016). Needs a calibration-target video; this is not SfM. | BSD-style with an advertising clause | ROS1, C++/Python |
| Kalibr fork JzHuai0108/kalibr | README | Camera-IMU with line delay | set `line_delay_nanoseconds` (non-zero) in camchain YAML, then `kalibr_calibrate_imu_camera … --estimate-line-delay` | Kalibr LICENSE | ROS1 Melodic/Noetic |
| OpenVINS | `6948812` 2025-11-30 | **No** RS readout in estimator; only a citation in `ov_core/src/sim/BsplineSE3.h` | — | GPL-3.0 | ROS, C++ |
| Basalt | `0f3b2b5` 2026-03-22 | **No** (0 matches) | — | BSD-3-Clause | C++ |
| VINS-Mono | master | **Yes, first-order** | config `rolling_shutter: 1`, `rolling_shutter_tr: <s per frame>`. Used in `vins_estimator/src/factor/projection_td_factor.cpp` as `pts_i_td = pts_i - (td - td_i + TR/ROW*row_i)*velocity_i`, i.e. each feature is shifted by its image velocity × row time. This factor is only used when `estimate_td: 1`. Needs an IMU. | GPL-3.0 | ROS1, Ceres |

### RS-specific research code

1. **danini/rolling_shutter_made_practical** (Barath, ECCV 2026). MIT; C++ with pybind bindings on SupeRANSAC; needs OpenCV, Eigen and Boost.
   - Synthetic run: `bash examples/run_synthetic.sh 50`.
   - TUM RS run:
     ```
     bash examples/download_tum_rs.sh datasets/tum_rs 1
     python examples/run_real_data.py --dataset_root datasets/tum_rs --sequences 1
     ```
     It compares against GS-5PC, RS-20PC and RS-44PC, and the real-data path needs CUDA, torch, kornia and RoMa for affine-correspondence extraction.
   - **Best ready image-only RS baseline on our exact dataset.** It is two-view, so it gives relative pose and RS velocities per pair.
2. **CenekAlbl/RnP** (BSD-3). R6P, R7Pf, R7Pfr absolute pose plus velocity; C++, Python bindings, MATLAB. Use it to localise RS frames against a 3D map (for example one triangulated from the GS twin, or our own points). That gives a classical per-frame (ω, v) reference for our velocity head.
3. **APRIL-ZJU/Ctrl-VIO**. README states GPL-3.0, but no LICENSE file was found. ROS Melodic, Ceres 1.14, OpenCV 3.3, yaml-cpp, with `config/ct_odometry_tumrs.yaml` for TUM-RSVI. An IMU-privileged RS-VIO reference.
4. **suhang99/AsyncTrack-Motion-Solver** (Apache-2.0, C++, ICCV 2025). Linear structure plus constant-velocity motion from timestamped tracks (handles RS). I did not check whether rotation must be known.
5. **DelinQu/NW-RSBA** (no LICENSE file; MATLAB `rssfm/rssfm_main.mlx`; synthetic "general" and "degeneracy" scenes; six BA variants) and **Yongcong-Zhang/RSL-BA** (no license; MATLAB `main/RSL_BA_main.mlx`; three synthetic scenes). Not plug-and-play on TUM; you would need a front end and an initialisation.
6. **henrique/rsba** (Apache-2.0; C++, Ceres 1.9, Thrift; last push 2018). The README has build instructions only; status unknown.
7. **ThomasZiegler/RS-aware-differential-SfM** (GPL-3.0). A third-party C++ implementation of Zhuang ICCV 2017 two-frame differential RS-SfM.
8. **facebookresearch/visual_inertial_bundle_adjustment** (MIT, C++17). README: "RGB camera is uses a rolling shutter aware modelling (camera time offset and readout time are optional variables…)". Inputs are Aria VRS/MPS files, so it would not run on TUM without a converter.

### Commercial RS-aware multi-view SfM (image-only)

- **Agisoft Metashape Professional 2.2** (manual, 2025): "If the option **Enable rolling shutter compensation** is enabled in the Camera calibration dialog window before alignment step…". The manual also mentions a "Full model for rolling shutter". The model names "Regularized"/"Full" and the 6-parameters-per-frame figure come only from an unofficial community manual.
- **Pix4Dmapper** (support article): Image Properties Editor → Camera Model → Shutter Model = **"Linear Rolling Shutter"**, then re-run Initial Processing. Works only for landscape captures.
- Both are image-only, multi-view RS baselines, but they need licences.

### Recommended baseline set for the paper

| Kind | Baseline |
|---|---|
| Non-RS classical | COLMAP and/or GLOMAP on the RS images |
| Image-only classical RS, two-view | Barath 7AC |
| Image-only classical RS, multi-view | Metashape RS compensation, if a licence is available |
| Classical velocity reference | R6P/RnP against the GS-twin map |
| IMU-privileged | Ctrl-VIO, or VINS-Mono with `rolling_shutter: 1` |
| Gyro-oracle | Our EXP1 unrolling (Karpenko-style) |

---

## 5. Per-paper notes

One line each; the CSV has the full fields.

**Topic 4**
- Meingast et al., arXiv cs/0503076: early RS projection models.
- Hedborg et al., CVPR 2012: RSBA for video; 6 extra parameters; iPhone 4 data with a GS ground-truth camera.
- Saurer et al., CVPR 2016: RS BA with GPS/INS; RS plane sweep.
- Albl et al., ECCV 2016: degeneracies (parallel readout is critical).
- Ito & Okatani, CVPR 2017: critical motion sequences via self-calibration.
- Albl et al., CVPR 2015: R6P.
- Kukelova et al., ACCV 2018: linear R6P.
- Kukelova et al., ECCV 2020: R7Pf/R7Pfr.
- Dai et al., CVPR 2016: RS essential matrices.
- Zhuang et al., ICCV 2017: differential RS-SfM.
- Albl et al., CVPR 2020: two opposite RS cameras give one GS.
- Hahn et al., arXiv 2024: order-one RS cameras.
- Hruby & Pollefeys, ICCV 2025: single-scanline relative pose.
- Su et al., ICCV 2025: asynchronous-track linear solver.
- Hruby & Pollefeys, arXiv 2025: full-DoF velocity from asynchronous tracks.
- Errázuriz Muñoz et al., arXiv 2026: single-view RS SfM theory.
- Liao et al., CVPR 2023: NW-RSBA.
- Zhang et al., ECCV 2024: RSL-BA.
- Zhang et al., ECCV 2026 (group page): RS self-calibration.
- Barath, ECCV 2026 (Google page): 7-AC RS relative pose.
- Oth et al., CVPR 2013: RS calibration (in Kalibr).
- Bai et al., CVPR 2022: scanline homographies.
- Verhoeven, ISPRS 2026: readout measurement plus Metashape RS compensation.

**Topic 5**
- Lovegrove et al., BMVC 2013 and Patron-Perez et al., IJCV 2015: spline continuous-time trajectories.
- Kerl et al., ICCV 2015: RS RGB-D.
- Yang et al., RA-L 2018: RS hurts VO.
- Lee & Yoon, arXiv 2017: RS-VO with per-frame (ω, v).
- Schubert et al., ECCV 2018 (DOI-inferred): RS-DSO.
- Schubert et al., IROS 2019: RS direct VIO and the TUM RS-VIO dataset.
- Cao et al., IEEE Access 2020: WHU-RSVI (synthetic).
- Micusik & Evangelidis, arXiv 2020: RS-VIO initialisation.
- Mo & Sattar, ICRA 2022: continuous-time spline VIO.
- Yang et al., arXiv 2022: VINS self-calibration including readout.
- Huai et al., arXiv 2021: RS camera-IMU calibration (Kalibr fork).
- Lang et al., RA-L 2022: Ctrl-VIO.
- Seiskari et al., ECCV 2024: 3DGS with RS and blur using VIO velocities.
- Zhang et al., IV 2025: semantic SLAM with RS and INS.
- Carmichael & Skinner, arXiv 2026: TRGS-SLAM.
- Ceriola et al., RA-L 2026: Breaking Time (G-solver).
- Kim et al., ICRA 2016 and Li et al., ICRA 2013: **UNVERIFIED**.

**Topic 7**
- Ait-Aider et al., ECCV 2006: object pose and velocity from one RS view.
- Ait-Aider et al., CVPR 2007: kinematics from lines (venue from the group page only).
- Magerand et al., ECCV 2012: global optimisation of object pose and motion.
- Rengarajan et al., CVPR 2017: first CNN for single-image RS correction.
- Purkait et al., ICCV 2017: Manhattan-world correction.
- Lao & Ait-Aider, CVPR 2018: lines, angular velocity.
- **Zhuang et al., CVPR 2019: SMARSC** (Section 2).
- Liu et al., CVPR 2020: DeepUnrollNet (two-frame).
- Mo et al., CoRL 2021: IMU-assisted single view.
- Chen & Clark, ICCV 2025: Image as an IMU (blur).
- Shyamala 2025: RS-as-IMU course report.
- Carter et al. 2026: anchor points, foundation depth under RS.
- Xiao et al., NeurIPS 2026: Roll2Depth (flicker).
- Xue et al., arXiv 2026: sphere velocity from one RS frame.

**Topic 8**
- Forssén & Ringaby, CVPR 2010: rectification.
- Karpenko et al., CSTR 2011-03: gyro stabilisation plus RS correction.
- Ringaby & Forssén, IJCV 2012: phone rectification and stabilisation.
- Bell et al., ECCV 2014: non-linear gyro filter.
- Liang & Shi, Google blog 2017: Pixel 2 fused EIS/OIS.
- Shi et al., WACV 2022: Deep-FVS.
- Li et al., TCSVT 2022: DeepOIS.
- Yang et al., arXiv 2024: RS-Diffusion.
- Lee et al., arXiv 2019 and Lee & Yoon, arXiv 2017: gyro/inertial-aided RS relative pose.

---

## 6. Could not verify / caveats

**Venues and details not confirmed**
- Kim, Cadena & Reid, ICRA 2016, "Direct semi-dense SLAM for rolling shutter cameras": taken only from the reference list of arXiv 1911.01015.
- Li, Kim & Mourikis, ICRA 2013 phone RS-VIO: the PDF download failed and the exact title is not confirmed.
- Ait-Aider et al., CVPR 2007: venue comes only from the EnCoV group publication list; the paper page was not opened.
- RS-Diffusion: an AAAI link appeared in search results but was not opened, so the venue is left as arXiv.
- Rolling Shutter Camera Self-Calibration: ECCV 2026 comes from the authors' group page, not from arXiv or ECVA.
- Barath: ECCV 2026 comes from the Google Research page; arXiv states no venue.
- RS-DSO: ECCV 2018 is inferred from the Springer DOI shown on its arXiv page.
- Hedborg 2012 and Forssén 2010: confirmed only through ML Anthology (the liu.se host would not resolve and IEEE showed a bot wall).
- Huai et al. 2021 and Yang et al. 2022: journal venues not confirmed (left as arXiv).
- Metashape "Regularized/Full" mode names and parameter counts: from an unofficial manual only.

**Code status unchecked or unclear**
- SMARSC: no official code found.
- Rengarajan 2017, Purkait 2017, Lao 2018: code not looked for in depth.
- Gyroflow: its rolling-shutter correction mechanism is undocumented on the pages I could open (the docs RS page returned 404). License GPL-3.0 per the GitHub API.
- AsyncTrack solver: not checked whether rotation must be known.
- `petrhruby97/single_scanline`: contents not inspected.

**Search limits**
- The arXiv API abstract search may miss papers. The plain `"rolling shutter"` query returned at most 150 entries. The `abs:degenerac*` wildcard may not be honoured.
- GitHub search API results are partial, and later calls hit the unauthenticated rate limit.

---

## 7. Query log

### WebSearch

**Topic 4**
1. rolling shutter bundle adjustment structure from motion
2. Hedborg rolling shutter bundle adjustment CVPR 2012
3. Albl R6P rolling shutter absolute pose minimal solver
4. degeneracies in rolling shutter SfM Albl ECCV 2016
5. Barath "Rolling Shutter Relative Pose Estimation Made Practical"
6. COLMAP rolling shutter support camera model readout time
7. Liao "Revisiting Rolling Shutter Bundle Adjustment" code github
8. generalized epipolar geometry rolling shutter relative pose linear 44-point
9. rolling shutter structure from motion open source implementation OpenMVG Theia
10. "The fast and the distorted" sensor readout image-based 3D reconstruction
11. rolling shutter SfM 2025 2026 arXiv new method camera motion during exposure global optimization
12. Kalibr rolling shutter camera calibration line delay
13. github rolling shutter absolute pose solver code R6P R7Pf matlab C++ Albl Kukelova
14. Agisoft Metashape rolling shutter compensation option camera calibration user manual
15. Pix4D rolling shutter optimization camera model linear rolling shutter readout time
16. "Minimal Rolling Shutter Absolute Pose with Unknown Focal Length and Radial Distortion" ECCV 2020

**Topic 5**
1. TUM rolling shutter visual-inertial odometry dataset uEye global shutter rolling shutter twin camera
2. rolling shutter visual-inertial odometry 2023 2024 2025 continuous-time
3. rolling shutter aware ORB-SLAM feature-based monocular SLAM
4. OpenVINS rolling shutter readout time estimation MSCKF
5. learned deep visual odometry rolling shutter DROID-SLAM robustness
6. Gaussian splatting SLAM rolling shutter camera motion compensation 2025
7. "rolling shutter" anchor point visual odometry 2026
8. Basalt rolling shutter visual-inertial mapping Usenko Demmel
9. event camera vs rolling shutter visual odometry RS-aware VO smartphone ARCore rolling shutter compensation
10. rolling shutter direct sparse odometry RS-DSO code release
11. WHU-RSVI rolling shutter visual inertial dataset
12. "rolling shutter" visual SLAM 2025 OR 2026 arXiv deep learned odometry rolling shutter aware bundle adjustment
13. rolling shutter event camera SLAM continuous-time spline Lovegrove Patron-Perez spline fusion

**Topic 7**
1. Zhuang Learning structure-and-motion-aware rolling shutter correction CVPR 2019 SMARSC
2. single rolling shutter image depth and camera velocity estimation deep network
3. rolling shutter aware monocular depth estimation self-supervised
4. estimate ego-motion from one rolling shutter frame angular linear velocity learning
5. Deep Shutter Unrolling Network CVPR 2020 Liu velocity depth
6. rolling shutter pose estimation network unrolling multi-view foundation model DUSt3R VGGT rolling shutter
7. rolling shutter correction single image learned camera motion Rengarajan CVPR 2017 unrolling
8. "rolling shutter" feed-forward 3D reconstruction pointmap transformer 2026
9. Roll2Depth zero-shot metric depth rolling shutter effect NeurIPS 2026
10. "Rolling Shutter Image as an IMU" single image camera motion
11. SMARSC rolling shutter correction code github Zhuang Tran Chandraker
12. Ait-Aider Andreff Lavest Martinet "Simultaneous object pose and velocity computation using a single view from a rolling shutter camera" ECCV 2006

**Topic 8**
1. Karpenko Jacobs Baek Levoy digital video stabilization rolling shutter correction gyroscopes Stanford tech report
2. smartphone electronic image stabilization rolling shutter correction gyroscope warp per-row homography
3. Google Pixel fused video stabilization OIS EIS rolling shutter blog
4. gyroscope-aided deep rolling shutter correction network IMU guided
5. deep gyro video stabilization DeepOIS gyroflow network rolling shutter
6. Gyroflow open source rolling shutter correction lens profile gyro stabilization
7. Forssén Ringaby rectifying rolling shutter video from hand-held devices CVPR 2010
8. 3D reconstruction from stabilized smartphone video EIS breaks pinhole geometry intrinsics vary per frame SfM
9. Android camera EIS rolling shutter correction gyro per-scanline warp mesh

### arXiv API

All queries used `https://export.arxiv.org/api/query?search_query=…&sortBy=submittedDate&sortOrder=descending`.

| # | Query | Results |
|---|---|---|
| q1 | `abs:"rolling shutter" AND abs:bundle` | 12 |
| q2 | `abs:"rolling shutter" AND (abs:solver OR abs:minimal)` | 20 |
| q3 | `abs:"rolling shutter" AND (abs:"absolute pose" OR abs:"relative pose" OR abs:PnP)` | 11 |
| q4 | `abs:"rolling shutter" AND (abs:"structure from motion" OR abs:degenerac* OR abs:SfM)` | 11 |
| q5 | `abs:"rolling shutter" AND (abs:SLAM OR abs:odometry OR abs:VIO OR abs:inertial)` | 35 |
| q6 | `abs:"rolling shutter" AND (abs:velocity OR abs:"single image")` | 29 |
| q7 | `abs:"rolling shutter" AND (abs:gyro* OR abs:IMU OR abs:stabiliz*)` | 19 |
| q8 | `abs:"rolling shutter"` | 150 (API cap) |
| q9 | `abs:"rolling shutter" AND (abs:depth OR abs:monocular OR abs:network OR abs:learning)` | 54 |
| q10 | `ti:"rolling shutter" OR ti:"rolling-shutter"` | 62 |
| q11 | `abs:"video stabilization" AND (abs:gyroscope OR abs:gyro OR abs:OIS OR abs:EIS)` | 1 |
| q12 | `abs:"rolling shutter" AND (abs:VGGT OR abs:DUSt3R OR abs:MASt3R OR abs:"feed-forward" OR abs:foundation OR abs:pointmap)` | 2 (neither relevant) |
| q13 | `(abs:"rolling shutter" OR abs:"rolling-shutter") AND (abs:"angular velocity" OR abs:"ego-motion")` | 8 |

### GitHub repository search API

Queries: `rolling shutter absolute pose`, `rolling shutter bundle adjustment`, `rolling shutter SfM`, `rolling shutter relative pose`.

These found RnP, rsba, NW-RSBA, RSL-BA, URS-NeRF, Meta VIBA, RS-aware-differential-SfM, Single-View-RS-SfM and single_scanline.

### Repository checks

- Shallow clones and greps: COLMAP, GLOMAP, TheiaSfM, OpenMVG, Kalibr, NW-RSBA, OpenVINS (sparse), Basalt (sparse).
- Raw-file checks: VINS-Mono (`config/euroc/euroc_config.yaml`, `vins_estimator/src/parameters.cpp`, `vins_estimator/src/factor/projection_td_factor.cpp`) and OpenVINS (`VioManagerOptions.h`, euroc config).
- README checks: Barath, Ctrl-VIO, Kalibr fork, RnP, VIBA, rsba, RSL-BA, RS-aware-differential-SfM, Single-View-RS-SfM, single_scanline, AsyncTrack, deep-stabilization, IRVLab/unrolling, Stable-Derolling, image-as-an-imu.
