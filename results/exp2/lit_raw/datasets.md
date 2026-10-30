# RS-3D dataset survey (EXP2 literature, raw)

Audit date: 2026-10-29. Machine-readable version: `datasets.csv` in this folder (36 rows, one per dataset, with the URL behind every claim).

**Rules followed.**
- Every fact below comes from a page, paper, devkit source file or licence file that I opened. The URLs are in the CSV `url_opened` column and in section 8.
- Anything I could not confirm from an official source is marked **UNVERIFIED**.
- I did not register, sign or accept any licence. I downloaded no dataset content. I fetched only web pages, paper PDFs, licence/README files, directory listings, and two Waymo devkit source files.

**Legend for `needs_human`.**
- **YES**: needs registration, a signed form, or a click-through.
- **YES (terms-by-use)**: the download is public, but downloading counts as accepting a non-standard licence, so a human must decide.
- **NO**: plain open download.

---

## 1. Findings in plain English

### A. Real RS data with true camera paths

- **TUM RS-VIO is still the only real dataset that pairs RS and GS cameras on one rig with mocap ground truth.**
  - Rig: two IDS uEye UI-3241LE-M-GL cameras (1280x1024, 20 Hz). The left camera runs in GS mode and the right in RS mode (paper Sec. V).
  - Row time is 29.4737 µs, so a full frame reads out in about 30.2 ms.
  - Ground truth is OptiTrack Flex13 at 120 Hz, aligned to the IMU frame.
  - Size: 10 indoor sequences, about 20 GB in EuRoC format. Public HTTP download.
  - **No licence is stated** on the dataset page.
  - The timestamp wording is ambiguous. Verbatim: *"This means the timestamps for the rolling-shutter images refer to the first row. In general, timestamps denote the middle of the exposure interval."*
- **Aria Digital Twin (ADT) is the strongest new find.**
  - Camera: Project Aria's RGB camera is rolling shutter. The Aria docs give readout of **16.26 ms at 2880x2880** and **5 ms in the 1408x1408 binned mode**; ADT RGB is 1408x1408 (paper).
  - Image timestamp: the centre of exposure of the middle row. The docs give t(p) = t_image + (row/H − 0.5)·readout.
  - Ground truth:
    - device poses from OptiTrack, with timecode sync accuracy under 10 µs on average;
    - dense per-frame depth rendered from a digital twin;
    - object meshes and object 6DoF poses.
  - Content: dynamic objects and people.
  - Access: email form plus licence agreement, so needs_human = YES.
  - Caveat: the 5 ms readout is short, so the RS effect only shows during fast head turns. I could not check whether the GT depth is rendered RS-aware (UNVERIFIED).
- **ADVIO** is real iPhone 6s RS video (1280x720, 60 fps): 23 sequences, CC BY-NC 4.0, public on Zenodo.
  - Its ground truth is inertial navigation plus manual floor-plan position fixes, not mocap.
  - That is fine for long-range odometry but too coarse to measure RS-induced pose error. Readout is unknown.
- **WHU-RSVI is synthetic, not real.** It is POV-Ray renders with a GS twin at 640x480, 30 Hz, 69.44 µs row time (about 33.3 ms readout).
  - The official download page `aric.whu.edu.cn/rsvi-dataset.html` returns **HTTP 404**.
- **Phone + mocap leads:**
  - **ZJU-SenseTime ISMAR19** (Xiaomi MI8 and iPhone X with Vicon): RS is not stated, licence UNVERIFIED.
  - **100-Phones** (350 sequences from 100 phone models): GT method not stated on the repo; download is via Baidu Netdisk.
  - Neither gives readout times.
- **PennCOSYVIO** has RS GoPros and a Tango RGB camera next to GS cameras, but its GT is only about 10 cm accurate (AprilTags). Too coarse for us.
- **LaMAR** states that its iPad camera is rolling shutter, and its pseudo-GT comes from aligning device trajectories to NavVis laser scans. It needs registration.
- **"Gaussian Splatting on the Move"** (Zenodo, CC BY-SA 4.0, 1.9 GB) gives real Android RS **readout times** and exposure times. It has no GT poses (COLMAP plus VIO only).
  - Use it only as a source of realistic phone readout values.

### B. Real RS data with true 3D shape

- **Waymo Open Dataset has the best-documented RS timing of any real dataset.** I verified this from `dataset.proto` and the devkit `camera_model.cc` (section 3 has verbatim quotes).
  - Each camera has a `rolling_shutter_direction` enum.
  - Each image has `shutter`, `camera_trigger_time`, `camera_readout_done_time`, a vehicle `pose`, a `velocity` (linear and angular, global frame) and a `pose_timestamp`.
  - The devkit computes readout = readout_done − trigger − shutter and gives each row a time.
  - The paper says "the exact scanning mode can vary from scene to scene".
  - Ground truth: LiDAR plus 3D boxes. Outdoor and dynamic.
  - Access: Google sign-in plus the *Waymo Dataset License Agreement for Non-Commercial Use (March 2025)*, so needs_human = YES.
- **ScanNet++ (iPhone 13 Pro, 1920x1440 @ 60 FPS)** has sub-mm Faro laser scans.
  - Its iPhone poses come from COLMAP registration to laser-scan renders, refined photometrically.
  - **Neither the docs nor the paper mention rolling shutter, readout or exposure timing.** The readout would have to be estimated.
  - Access: account plus application (needs_human = YES).
- **ARKitScenes (2020 iPad Pro + Faro Focus S70)** has laser-registered GT poses and depth, and a public download script.
  - Its licence is Apple's custom non-commercial licence, where use counts as acceptance.
  - It has **no RS timing**.
- **ADT** (see A) is the only real dataset in this category that has both a documented readout and dense GT depth.
- **LaMAR** (see A): its iPad RS camera comes with laser scans.

### C. Mesh sources for synthetic RS rendering

- **Replica:** 18 scenes, 33.86 GB (17 GitHub release parts).
  - **Answer to the brief: both.** `mesh.ply` is a quad mesh *with vertex colours*, and full-quality appearance comes from separate HDR PTex textures (`textures/*-color-ptex.hdr`). The textures need the habitat-sim or ReplicaSDK PTex renderer.
  - Download: public `download.sh` script from GitHub releases.
  - Licence: *Replica Dataset Research Terms* (non-commercial; "By accessing ... Researcher hereby agrees"), so needs_human = YES (terms-by-use).
- **HM3D:** 1,000 textured scenes (OBJ/GLB plus JPG textures). Access needs a Matterport account and a Request Access form; licence is the *Matterport EULA for Academic Use of Model Data*.
- **Matterport3D:** 90 scenes; needs a signed Terms of Use form sent by email.
- **3D-FRONT:** 18,968 rooms with 13,151 textured furniture objects. You email a signed Terms of Use to Alibaba; the Tianchi page returned 404.
- **ScanNet++ meshes:** laser-derived Poisson meshes. The toolkit renders RGB from them, which implies vertex colour (UNVERIFIED). Needs an application.
- **TartanAir V2 is the most practical source with no sign-up.**
  - CC BY 4.0, 65 environments, 640x640 at 10 Hz, GT depth and pose.
  - It also has a **1000 Hz RGB MP4 of `lcam_front`**, so we can build RS by row-stitching at 1 ms resolution.
  - No meshes are released. The page does not say how poses for the 1000 Hz stream are provided (UNVERIFIED).
- **NRGBD .blend scenes** (BlendSwap, CC-BY or CC-0) form a small, permissively licensed mesh set.

### D. GS benchmarks for the "no harm" check

- **Confirmed GS from an official source:**
  - ETH3D low-res rig (paper: "four global-shutter cameras");
  - ETH3D SLAM ("All cameras on this rig use a global shutter");
  - TUM RS-VIO cam0;
  - Argoverse 2 ("The nine global shutter cameras");
  - synthetic NRGBD, Sintel, TartanAir.
- **Static capture, where RS cannot matter:** DTU (robot arm, fixed positions), ETH3D high-res (Nikon D3X on a tripod), Matterport3D.
- **EuRoC:** the official ETH page does *not* state the shutter type, and the IJRR paper returned 403. Secondary sources call it GS: Barath 2026's abstract says "global-shutter EuRoC MAV", and PennCOSYVIO says its matching VI-Sensor is GS. **Partially verified.**
- **KITTI:** the official setup page names Point Grey Flea 2 FL2-14S3C-C but does not state the shutter type (UNVERIFIED).

**Important: several benchmarks that VGGT, π³ and MapAnything use are phone or consumer video, not GS.**

| Benchmark | Used by (verified from paper) | Capture device (verified) | Shutter |
|---|---|---|---|
| CO3Dv2 | VGGT, π³ (pose AUC) | crowd-sourced "turn-table videos captured with smartphones" | RS presumed (phones) |
| RealEstate10K | VGGT, π³ (pose AUC) | ~10k YouTube videos | unknown, likely mostly RS |
| ScanNet / ScanNet-1500 | VGGT (matching), π³ (pose), MapAnything (Robust-MVD) | iPad Air2 colour camera + Structure sensor | RS presumed, not documented |
| ScanNet++ v2 | MapAnything | iPhone 13 Pro (+ static DSLR) | iPhone RS presumed |
| 7-Scenes | π³ (point maps) | handheld Kinect | not documented |
| TUM-dynamics (TUM RGB-D) | π³ (pose) | Kinect | not documented |
| Bonn | π³ (video depth) | RGB-D sensor (not named on page) | not documented |
| NYU-v2 | π³ (mono depth) | not audited | UNVERIFIED |
| IMC (phototourism) | VGGT | internet photos (not audited) | UNVERIFIED |
| ETH3D | VGGT, π³, MapAnything | GS rig / tripod DSLR | GS (verified) |
| DTU | VGGT, π³ | static robot positions | irrelevant (verified) |
| NRGBD | π³ | synthetic | GS (rendered) |
| Sintel | π³ | synthetic film | GS (rendered) |
| KITTI | π³, MapAnything (Robust-MVD) | Flea 2 cameras | GS UNVERIFIED |
| TartanAirV2-WB | MapAnything | synthetic | GS (rendered) |

The "ground truth" poses for CO3Dv2 (COLMAP), RealEstate10K (SLAM + BA) and ScanNet (BundleFusion) are themselves single-pose-per-frame estimates computed from the same RS frames. So these benchmarks cannot serve as RS-free controls, and their GT already absorbs some RS distortion. That point is worth making in the paper.

---

## 2. Recommended shortlist per category

**A. Real RS + GT camera path**
1. **TUM RS-VIO.** It has a real RS/GS twin, mocap GT, known readout (~30.2 ms) and a public download, and is already in our pipeline. Weaknesses: licence unstated, 10 sequences, grayscale.
2. **Aria Digital Twin.** Real RS RGB with documented 5 ms readout, mocap poses synced to under 10 µs, a centred timestamp convention, dense GT depth, and dynamic content. Needs sign-up.
3. **ZJU-SenseTime ISMAR19.** Phone RS with Vicon GT, the closest to an in-the-wild phone. Readout is unknown, so it must be calibrated, and the licence must be checked by a human.
4. *(stress test only)* **ADVIO**: long real phone sequences, but GT is too coarse for RS-sized errors.

**B. Real RS + GT 3D**
1. **Waymo Open Dataset.** The only real RS dataset with per-image trigger, readout and exposure times, scan direction, and velocity, plus official RS-aware projection code and LiDAR GT. Needs sign-up.
2. **Aria Digital Twin.** Indoor complement to Waymo: real RS, documented readout, dense GT depth and meshes.
3. **ScanNet++ v2 iPhone.** The highest-quality indoor laser GT, and MapAnything's own test set. Readout must be estimated (e.g. a per-device line-delay fit). Needs an application.
4. *(backup, public)* **ARKitScenes**: laser GT, public script, no RS timing; Apple non-commercial terms.

**C. Mesh / render sources for synthetic RS**
1. **Replica.** Best photometric quality (HDR PTex). Public download, research terms-by-access. 18 scenes, 33.9 GB.
2. **TartanAir V2.** CC BY 4.0 with no sign-up, 65 environments, and a 1000 Hz video that can be row-stitched into RS at chosen readouts, with GT depth and pose. Indoor and outdoor.
3. **HM3D.** Scale (1,000 textured real scans) for training or large-sample tests. Needs a Matterport account.
4. *(optional)* **ScanNet++ meshes**: renders RS twins of the exact scenes whose real iPhone RS frames we also test. That gives a matched real/synthetic pair.

**D. GS "no harm" benchmarks**
1. **ETH3D** (MVS high-res, low-res and SLAM). Verified GS or static capture, laser or mocap GT, CC BY-NC-SA 4.0, and used by all three model papers.
2. **TUM RS-VIO cam0.** The GS twin of exactly the same motion as our RS test, which makes it the cleanest paired control.
3. **DTU.** Static capture, so RS cannot matter; used by VGGT and π³.
4. **Argoverse 2.** Verified GS driving data with LiDAR, to separate "RS" from "driving domain" when comparing with Waymo.
5. *(conditional)* **EuRoC**: use once a human confirms the GS statement in the IJRR paper.
- **Do not count as GS:** CO3Dv2, RealEstate10K, ScanNet / ScanNet-1500, ScanNet++ iPhone, and (unknown shutter) 7-Scenes, TUM RGB-D, Bonn.

---

## 3. Waymo RS timing fields (verbatim from `dataset.proto` and the devkit)

Sources:
- `https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/master/src/waymo_open_dataset/dataset.proto` (also cross-checked on the GitHub blob page);
- `.../src/waymo_open_dataset/wdl_limited/camera/camera_model.cc` and `camera_model.h`.

```proto
message CameraCalibration {
  ...
  enum RollingShutterReadOutDirection {
    UNKNOWN = 0;
    TOP_TO_BOTTOM = 1;
    LEFT_TO_RIGHT = 2;
    BOTTOM_TO_TOP = 3;
    RIGHT_TO_LEFT = 4;
    GLOBAL_SHUTTER = 5;
  }
  optional RollingShutterReadOutDirection rolling_shutter_direction = 6;
}

// All timestamps in this proto are represented as seconds since Unix epoch.
message CameraImage {
  optional CameraName.Name name = 1;
  // JPEG image.
  optional bytes image = 2;
  // SDC pose.
  optional Transform pose = 3;
  // SDC velocity at 'pose_timestamp' below. The velocity value is represented
  // at *global* frame.
  // With this velocity, the pose can be extrapolated.
  // r(t+dt) = r(t) + dr/dt * dt where dr/dt = v_{x,y,z}.
  // dR(t)/dt = W*R(t) where W = SkewSymmetric(w_{x,y,z})
  // ...
  optional Velocity velocity = 4;
  // Timestamp of the `pose` above.
  optional double pose_timestamp = 5;

  // Rolling shutter params.
  // The following explanation assumes left->right rolling shutter.
  //
  // Rolling shutter cameras expose and read the image column by column, offset
  // by the read out time for each column. The desired timestamp for each column
  // is the middle of the exposure of that column as outlined below for an image
  // with 3 columns:
  // ...
  // ^trigger time                                ^readout end time
  //             ^time for row 1 (= middle of exposure of row 1)
  //                    ^time image center (= middle of exposure of middle row)
  // Shutter duration in seconds. Exposure time per column.
  optional double shutter = 6;
  // Time when the sensor was triggered and when last readout finished.
  // The difference between trigger time and readout done time includes
  // the exposure time and the actual sensor readout time.
  optional double camera_trigger_time = 7;
  optional double camera_readout_done_time = 8;
  ...
}

message Velocity {
  // Velocity in m/s.
  optional float v_x = 1; optional float v_y = 2; optional float v_z = 3;
  // Angular velocity in rad/s.
  optional double w_x = 4; optional double w_y = 5; optional double w_z = 6;
}
```

Devkit (`wdl_limited/camera/camera_model.cc`, lines 82–102 and 234–236):

```cpp
const double readout_duration =
    camera_readout_done_time - camera_trigger_time - shutter;
...
const double base_ts = camera_trigger_time + 0.5 * shutter;
switch (readout_direction) {
  // TOP_TO_BOTTOM:  return base_ts + readout_duration / image_height * y;
  // BOTTOM_TO_TOP:  return base_ts + readout_duration / image_height * (image_height - y);
  // LEFT_TO_RIGHT:  return base_ts + readout_duration / image_width * x;
  // RIGHT_TO_LEFT:  return base_ts + readout_duration / image_width * (image_width - x);
```

- `camera_model.h`: "It does rolling shutter projection if the camera is not a global shutter camera. To disable rolling shutter projection, override rolling_shutter_direction in the camera calibration."
- Paper (arXiv:1912.04838): "The camera images are captured with rolling shutter scanning, where the exact scanning mode can vary from scene to scene."
- Paper: "Rolling shutter timing information is provided with each image."
- Paper: sync error is bounded in [−6 ms, 7 ms] with 99.7% confidence.
- Note: `pose` is the vehicle (SDC) pose, not the camera pose. The camera pose needs `CameraCalibration.extrinsic`.
- Licence note: the camera model lives under `wdl_limited/`, which has its own LICENSE and PATENTS files (not read).

---

## 4. Datasets that need human sign-up

**Registration, form or signed agreement:**
- Waymo Open Dataset (Google sign-in plus licence)
- Aria Digital Twin and Aria Synthetic Environments (email form plus licence)
- ScanNet++ (account plus application)
- ScanNet (signed Terms of Use by email)
- LaMAR (registration form)
- HM3D (Matterport account plus Request Access)
- Matterport3D (signed form by email)
- 3D-FRONT (signed Terms of Use by email)
- 100-Phones (Baidu Netdisk, which likely needs an account; UNVERIFIED)

**Terms-by-use** (public download, but downloading counts as accepting a non-standard licence):
- Replica (Research Terms)
- ARKitScenes (Apple non-commercial licence)
- 7-Scenes (MSR non-commercial licence)
- Argoverse 2 (Terms of Use, CC BY-NC-SA 4.0)

**No sign-up:**
- TUM RS-VIO (licence unstated), ADVIO, TartanAir v1/V2, ETH3D, DTU, NRGBD, CO3Dv2, RealEstate10K poses (but scraping the YouTube videos is a human decision), TUM RGB-D, Bonn, BS-RSC, Fastec/Carla-RS, GS-on-the-Move.

---

## 5. What I could not verify

- **TUM RS-VIO licence.** Not stated on the dataset page or README. The TUM RGB-D CC BY 4.0 notice does not obviously cover it.
- **TUM RS-VIO scan direction** (top-to-bottom is not stated) and the exact meaning of the "first row" / "middle of exposure" sentence.
- **EuRoC global shutter.** Not on the official ETH page; the IJRR paper is behind a 403; the Research Collection page returned 500. Also unverified: EuRoC sequence count, licence and timestamp convention.
- **KITTI shutter type, licence and download terms.** The setup page names the cameras but not the shutter.
- **Rolling shutter or readout for ScanNet++ iPhone, ARKitScenes, ScanNet iPad, CO3D phones, RealEstate10K, 7-Scenes, TUM RGB-D and Bonn.** None is documented. "RS presumed" is my inference from the device class, not a verified fact.
- **ZJU-SenseTime ISMAR19:** licence, sequence count, RS statement. WebFetch failed on TLS, so the content was read with curl over http.
- **100-Phones:** GT method and whether the Apache-2.0 notice covers the data.
- **PennCOSYVIO:** licence and download.
- **LaMAR:** licence and exact GT accuracy.
- **ADT:** licence text; whether GT depth is rendered per row (RS-aware); total size. The two official pages disagree on sequence count (200 vs 236).
- **Aria Synthetic Environments:** whether RS is simulated.
- **WHU-RSVI:** current availability (official URL 404).
- **Kerl et al. ICCV 2015** real RS RGB-D sequences: no download found.
- **TartanAir V2:** pose sampling for the 1000 Hz stream; whether meshes or raw environments are released.
- **ScanNet++ mesh appearance:** vertex colour vs texture is not stated in the docs.
- **Matterport3D:** whether meshes are textured is not stated on the page.
- **3D-FRONT:** Terms-of-Use text. The Tianchi page returned 404 and the Aliyun terms page refused connection.
- **Waymo:** dataset size and current version or segment count (behind login); actual per-image readout values.
- **Citations** for 7-Scenes, TUM RGB-D, KITTI and ZJU-SenseTime (not opened).
- **Venues** for 3D-FRONT (arXiv lists none) and GS-on-the-Move (arXiv lists none).
- **DOIs** for ADVIO and 100-Phones: seen only in search-result URLs, not opened.

---

## 6. Citations

Each was verified from the arXiv record (API or abstract page), the dataset page, or the paper PDF. "et al. (N)" means N authors in the arXiv record.

| Dataset | Title | Authors | Venue, year | ID |
|---|---|---|---|---|
| TUM RS-VIO | Rolling-Shutter Modelling for Direct Visual-Inertial Odometry | D. Schubert, N. Demmel, L. von Stumberg, V. Usenko, D. Cremers | IROS 2019 | arXiv:1911.01015; DOI 10.1109/IROS40897.2019.8968539 (dataset page lists the title without "Direct") |
| ADVIO | ADVIO: An authentic dataset for visual-inertial odometry | S. Cortés, A. Solin, E. Rahtu, J. Kannala | ECCV 2018 | arXiv:1807.09828 |
| WHU-RSVI | The WHU Rolling Shutter Visual-Inertial Dataset | L. Cao, J. Ling, X. Xiao | IEEE Access 8:50771–50779, 2020 | DOI 10.1109/ACCESS.2020.2978589 |
| 100-Phones | 100-Phones: A Large VI-SLAM Dataset for Augmented Reality Towards Mass Deployment on Mobile Phones | G. Zhang, J. Yuan, H. Liu, Z. Peng, C. Li, Z. Wang, H. Bao | IEEE TVCG 2024 | (DOI not opened) |
| PennCOSYVIO | PennCOSYVIO: A challenging Visual Inertial Odometry benchmark | B. Pfrommer, N. Sanket, K. Daniilidis, J. Cleveland | ICRA 2017 | n/a |
| Aria Digital Twin | Aria Digital Twin: A New Benchmark Dataset for Egocentric 3D Machine Perception | X. Pan, N. Charron, Y. Yang, S. Peters, T. Whelan, C. Kong, O. Parkhi, R. Newcombe, C. Y. Ren | ICCV 2023 (venue per projectaria.com link) | arXiv:2306.06362 |
| LaMAR | LaMAR: Benchmarking Localization and Mapping for Augmented Reality | P.-E. Sarlin, M. Dusmanu, J. L. Schönberger, P. Speciale, L. Gruber, V. Larsson, O. Miksik, M. Pollefeys | ECCV 2022 | arXiv:2210.10770 |
| GS on the Move | Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion | O. Seiskari, J. Ylilammi, V. Kaatrasalo, P. Rantalankila, et al. (8) | venue not on arXiv record | arXiv:2403.13327; data DOI 10.5281/zenodo.10848124 |
| Waymo Open | Scalability in Perception for Autonomous Driving: Waymo Open Dataset | P. Sun, H. Kretzschmar, X. Dotiwalla, A. Chouard, et al. (25) | CVPR 2020 | arXiv:1912.04838 |
| ScanNet++ | ScanNet++: A High-Fidelity Dataset of 3D Indoor Scenes | C. Yeshwanth, Y.-C. Liu, M. Nießner, A. Dai | ICCV 2023 | arXiv:2308.11417 |
| ARKitScenes | ARKitScenes: A Diverse Real-World Dataset For 3D Indoor Scene Understanding Using Mobile RGB-D Data | G. Baruch, Z. Chen, A. Dehghan, T. Dimry, et al. (11) | NeurIPS 2021 Datasets & Benchmarks (per GitHub README) | arXiv:2111.08897 |
| Replica | The Replica Dataset: A Digital Replica of Indoor Spaces | J. Straub, T. Whelan, L. Ma, Y. Chen, et al. (30) | arXiv 2019 | arXiv:1906.05797 |
| HM3D | Habitat-Matterport 3D Dataset (HM3D): 1000 Large-scale 3D Environments for Embodied AI | S. K. Ramakrishnan, A. Gokaslan, E. Wijmans, O. Maksymets, et al. (13) | NeurIPS 2021 Datasets & Benchmarks (per aihabitat.org) | arXiv:2109.08238 |
| Matterport3D | Matterport3D: Learning from RGB-D Data in Indoor Environments | A. Chang, A. Dai, T. Funkhouser, M. Halber, M. Nießner, M. Savva, S. Song, A. Zeng, Y. Zhang | 3DV 2017 | arXiv:1709.06158 |
| 3D-FRONT | 3D-FRONT: 3D Furnished Rooms with layOuts and semaNTics | H. Fu, B. Cai, L. Gao, L. Zhang, J. Wang, C. Li, Z. Xun, C. Sun, R. Jia, B. Zhao, H. Zhang | venue not on arXiv record | arXiv:2011.09127 |
| TartanAir | TartanAir: A Dataset to Push the Limits of Visual SLAM | W. Wang, D. Zhu, X. Wang, Y. Hu, et al. (9) | IROS 2020 | arXiv:2003.14338 |
| TartanAir V2 | no formal citation on tartanair.org (credits W. Wang, Y. Hu, Y. Qiu, S. Shen, Y. Shaoul, CMU 2023) | | | |
| ETH3D MVS | A Multi-View Stereo Benchmark with High-Resolution Images and Multi-Camera Videos | T. Schöps, J. L. Schönberger, S. Galliani, T. Sattler, K. Schindler, M. Pollefeys, A. Geiger | CVPR 2017 | n/a |
| ETH3D SLAM | BAD SLAM: Bundle Adjusted Direct RGB-D SLAM | T. Schöps, T. Sattler, M. Pollefeys | CVPR 2019 | n/a |
| EuRoC | The EuRoC micro aerial vehicle datasets | M. Burri, J. Nikolic, P. Gohl, T. Schneider, J. Rehder, S. Omari, M. Achtelik, R. Siegwart | IJRR 2016 | DOI 10.1177/0278364915620033 |
| Argoverse 2 | Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting | B. Wilson, W. Qi, T. Agarwal, J. Lambert, et al. (13) | NeurIPS Datasets & Benchmarks Track (year not in record) | arXiv:2301.00493 |
| CO3D | Common Objects in 3D: Large-Scale Learning and Evaluation of Real-life 3D Category Reconstruction | J. Reizenstein, R. Shapovalov, P. Henzler, L. Sbordone, P. Labatut, D. Novotny | ICCV 2021 | arXiv:2109.00512 |
| RealEstate10K | Stereo Magnification: Learning View Synthesis using Multiplane Images | T. Zhou, R. Tucker, J. Flynn, G. Fyffe, et al. (5) | SIGGRAPH 2018 | arXiv:1805.09817 |
| NRGBD | Neural RGB-D Surface Reconstruction | D. Azinović, R. Martin-Brualla, D. B. Goldman, M. Nießner, J. Thies | CVPR 2022 | n/a |
| DTU (CVPR) | Large Scale Multi-view Stereopsis Evaluation | Jensen et al. | CVPR 2014 | n/a |
| DTU (IJCV) | Large-Scale Data for Multiple-View Stereopsis | Aanæs et al. | IJCV 2016 | n/a |
| ScanNet | ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes | A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, M. Nießner | CVPR 2017 | arXiv:1702.04405 |
| Sintel | A naturalistic open source movie for optical flow evaluation | D. J. Butler, J. Wulff, G. B. Stanley, M. J. Black | ECCV 2012 | n/a |
| Bonn RGB-D Dynamic | ReFusion: 3D Reconstruction in Dynamic Environments for RGB-D Cameras Exploiting Residuals | E. Palazzolo, J. Behley, P. Lottes, P. Giguère, C. Stachniss | IROS 2019 | arXiv:1905.02082 |
| BS-RSC | Learning Adaptive Warping for Real-World Rolling Shutter Correction | M. Cao, Z. Zhong, J. Wang, Y. Zheng, Y. Yang | CVPR 2022, pp. 17785–17793 | arXiv:2204.13886 |
| Fastec-RS / Carla-RS | Deep Shutter Unrolling Network | P. Liu, Z. Cui, V. Larsson, M. Pollefeys | CVPR 2020 | n/a |
| Kerl RS RGB-D (lead) | Dense Continuous-Time Tracking and Mapping with Rolling Shutter RGB-D Cameras | C. Kerl, J. Stückler, D. Cremers | ICCV 2015 | DOI 10.1109/ICCV.2015.261 (from search result) |

**Model papers whose evaluation sets were checked:**
- VGGT (local `VGGT.pdf`, §4): CO3Dv2, RealEstate10K, DTU, ETH3D, ScanNet-1500, IMC.
- MapAnything (local `Map Anything.pdf`): ETH3D, ScanNet++ v2, TartanAirV2-WB, Robust-MVD.
- π³ (arXiv:2507.13347, HTML §4): RealEstate10K, Co3Dv2, Sintel, TUM-dynamics, ScanNet, 7-Scenes, NRGBD, DTU, ETH3D, Bonn, KITTI, NYU-v2.

---

## 7. Query log

**WebSearch (16 queries):**
1. `rolling shutter dataset ground truth poses`
2. `smartphone visual inertial dataset rolling shutter ground truth motion capture`
3. `rolling shutter SLAM benchmark dataset`
4. `rolling shutter dataset depth ground truth real`
5. `rolling shutter event camera dataset global shutter pair`
6. `rolling shutter correction real dataset paired global shutter images beam splitter`
7. `mobile phone visual-inertial SLAM benchmark Vicon ground truth rolling shutter camera AR dataset`
8. `Project Aria RGB camera rolling shutter readout time calibration`
9. `rolling shutter structure from motion real dataset laser scan ground truth 3D reconstruction`
10. `rolling shutter NeRF Gaussian splatting real captured dataset ground truth poses smartphone`
11. `WHU-RSVI rolling shutter visual-inertial dataset download github`
12. `3D-FRONT dataset download application tianchi license 3D-FUTURE`
13. `EuRoC MAV dataset MT9V034 global shutter stereo VI-sensor Burri IJRR 2016`
14. `"EuRoC" "global shutter" "MT9V034" Burri Nikolic "EuRoC micro aerial vehicle datasets" pdf`
15. `Kerl Stückler Cremers "Dense Continuous-Time Tracking and Mapping with Rolling Shutter RGB-D Cameras" dataset download`
16. `Fastec-RS Carla-RS rolling shutter dataset download Liu Deep Shutter Unrolling Network github`

**arXiv API:**
- `search_query=abs:"rolling shutter" AND abs:dataset`, sorted by date, max 200: 51 hits.
- `search_query=abs:"rolling shutter" AND (abs:benchmark OR abs:"ground truth" OR abs:"real-world")`, max 200: 46 hits.
- `id_list=` metadata lookup for 20 dataset or model papers (citations in section 6).

**Notable arXiv hits not audited further (leads only, nothing claimed about them):**
- 2606.26863, Barath 2026, "Rolling Shutter Relative Pose Estimation Made Practical". Uses the TUM RS benchmark and calls EuRoC global-shutter.
- 2608.01509, "Rolling Shutter Camera Self-Calibration" (2026). Could estimate the readout for ScanNet++ / ARKitScenes phones. Seen in search results only.
- 2506.04444, "Photoreal Scene Reconstruction from an Egocentric Device" (Aria RS).
- 2509.10759 (4D Gaussian ray tracing for camera effects).
- 2602.05617 (unified AV sensor simulation).
- 2603.20443 (TRGS-SLAM, RS thermal).
- 2208.12008, Ctrl-VIO. Uses WHU-RSVI, TUM-RSVI and SenseTime-RSVI; no new dataset.

---

## 8. URLs opened (summary; per-row list is in `datasets.csv`)

| Group | URLs |
|---|---|
| TUM RS-VIO | cvg.cit.tum.de/data/datasets/rolling-shutter-dataset; cdn3.vision.in.tum.de/rolling/README.txt; …/rolling/exported/euroc/; …/rolling/calibrated/; arxiv.org/abs/1911.01015; arxiv.org/pdf/1911.01015 |
| ADVIO | github.com/AaltoVision/ADVIO; arxiv.org/abs/1807.09828; arxiv.org/pdf/1807.09828 |
| WHU-RSVI | mee-jieling.github.io/journals/2020-IEEE-ACCESS.pdf; aric.whu.edu.cn/rsvi-dataset.html (404); ieeexplore.ieee.org/document/9027901 (empty) |
| Waymo | raw dataset.proto and GitHub blob; wdl_limited/camera/camera_model.{cc,h}; waymo.com/open/terms; waymo.com/open; waymo.com/open/data/perception (login redirect); arxiv.org/abs/1912.04838; arxiv.org/pdf/1912.04838 |
| ScanNet++ | scannetpp.mlsg.cit.tum.de/scannetpp/ and /documentation; arxiv.org/pdf/2308.11417; scannetpp toolkit README |
| ARKitScenes | github.com/apple/ARKitScenes, DATA.md, LICENSE, download_data.py; arxiv.org/pdf/2111.08897 |
| Aria / ADT / ASE | projectaria_tools docs (ADT overview, ADT data format, temporal alignment, ASE data format); projectaria.com/datasets/adt and /ase; arxiv.org/abs/2306.06362; arxiv.org/pdf/2306.06362v2 |
| Replica | github.com/facebookresearch/Replica-Dataset; download.sh; LICENSE; GitHub releases API |
| HM3D | aihabitat.org/datasets/hm3d; matterport.com/habitat-matterport-3d-research-dataset; github.com/matterport/habitat-matterport-3dresearch |
| Matterport3D | niessner.github.io/Matterport |
| 3D-FRONT | arxiv.org/abs/2011.09127; tianchi page (404); BlenderProc front_3d README; Aliyun terms (refused) |
| TartanAir | theairlab.org/tartanair-dataset; tartanair.org; tartanair.org/modalities.html |
| EuRoC | projects.asl.ethz.ch/datasets/euroc-mav; old doku.php page; projects.asl.ethz.ch/datasets; doi.org/10.3929/ethz-b-000690084 (500); journals.sagepub.com (403); docs.ros.org ov_core (access denied) |
| ETH3D | eth3d.net, /overview, /datasets, /slam_overview, /slam_documentation, /slam_datasets; CVPR 2017 paper PDF |
| Argoverse 2 | argoverse.github.io sensor and getting_started pages; argoverse.org/about.html |
| KITTI | cvlibs.net/datasets/kitti/setup.php |
| TUM RGB-D | cvg.cit.tum.de rgbd-dataset, file_formats, intrinsic_calibration |
| CO3D | github.com/facebookresearch/co3d; arxiv.org/abs/2109.00512; ar5iv 2109.00512 |
| RealEstate10K | google.github.io/realestate10k and /download.html |
| ScanNet | github.com/ScanNet/ScanNet; arxiv.org/pdf/1702.04405 (www.scan-net.org had a TLS mismatch) |
| 7-Scenes | microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes |
| Bonn | ipb.uni-bonn.de/data/rgbd-dynamic-dataset |
| NRGBD | github.com/dazinovic/neural-rgbd-surface-reconstruction |
| DTU | roboimagedata.compute.dtu.dk/?page_id=36 |
| Sintel | sintel.is.tue.mpg.de |
| LaMAR | lamar.ethz.ch; ar5iv 2210.10770 |
| 100-Phones | github.com/zju3dv/100-Phones |
| ZJU-SenseTime | zjucvg.net/eval-vislam and /dataset (via curl) |
| PennCOSYVIO | daniilidis-group.github.io/penncosyvio |
| GS on the Move | zenodo.org/records/10848124 |
| BS-RSC | github.com/ljzycmd/BSRSC |
| Fastec / Carla-RS | github.com/ethliup/DeepUnrollNet |
| Other arXiv | abs 2507.13347 and HTML (π³); abs 2606.26863; abs 2208.12008 |
| Local PDFs | `VGGT.pdf`, `Map Anything.pdf` |
