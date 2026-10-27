# EXP0 — Data notes: TUM Rolling-Shutter VIO dataset

All values below were **measured** or **read from the files** on 2026-10-27 unless marked otherwise.

## Source

- Dataset page: https://cvg.cit.tum.de/data/datasets/rolling-shutter-dataset
- Paper: D. Schubert, N. Demmel, L. von Stumberg, V. Usenko, D. Cremers, "Rolling-Shutter Modelling for Direct Visual-Inertial Odometry", **IROS 2019**, arXiv 1911.01015. Verified from the PDF and the dataset page.
- Downloaded: the "calibrated" sequences in the EuRoC export, `https://cdn3.vision.in.tum.de/rolling/exported/euroc/dataset-seq{1..10}.tar` (21.4 GB total).
- Calibration: `https://cdn3.vision.in.tum.de/rolling/calibration/camchain-calibration-equidistant4_camimu_dataset-calib-imu1.yaml`. Each sequence's `dso/camchain.yaml` is identical.
- Control sequence: `dataset-calib-imu1.tar` (2.1 GB). The dataset page says both cameras ran in **global-shutter mode** for the calibration sequences.

## Layout (each tar extracts to `dataset-seqN/dataset-seqN/`)

```
mav0/cam0/data/<ts_ns>.png   + mav0/cam0/data.csv   (#timestamp [ns], filename)
mav0/cam1/data/<ts_ns>.png   + mav0/cam1/data.csv
mav0/imu0/data.csv           (#timestamp [ns], w_RS_S_x/y/z [rad/s], a_RS_S_x/y/z [m/s^2])  200 Hz
mav0/mocap0/data.csv         (#timestamp [ns], p_RS_R_x/y/z [m], q_RS_w/x/y/z)          120 Hz
dso/                         DSO-format copies (camera.txt, times.txt with exposure, gt_imu.csv = mocap0, imu.txt)
```

| Item | Location | Notes |
|---|---|---|
| (a) GS camera | `mav0/cam0` (left) | The paper says: "the left camera running in global-shutter mode and the right camera in rolling-shutter mode". The calibration agrees: cam1 sits +0.109 m along cam0's x-axis (to the right). |
| (b) RS camera | `mav0/cam1` (right) | |
| (c) Gyro | `mav0/imu0/data.csv`, columns 1–3 | Bosch BMI160, 200 Hz, hardware-synced. Bias and axis-scale are already corrected (README). |
| (d) GT poses | `mav0/mocap0/data.csv` | OptiTrack, 120 Hz. **T_world_imu**: already moved to the IMU frame and time-aligned (dataset page + README). |
| (e) Calibration | camchain YAML above | Kalibr, `pinhole` + `equidistant` (Kannala–Brandt 4) distortion. |

- Images: 1280×1024, **16-bit grayscale, linear response**.
- Exposure is a **fixed 2.0 ms on both cameras in every frame** of seq 1–10 (`dso/cam*/times.txt`). The calibration sequence uses 1.0 ms. Motion blur is therefore matched between GS and RS.
- cam0 and cam1 timestamps are **identical** (hardware trigger). Nearest-frame matching gives a 0.0 ms offset in all sequences.

| seq | frames | duration (s) | frames with valid GT |
|---|---|---|---|
| 1 | 808 | 40.4 | 807 |
| 2 | 535 | 26.7 | 534 |
| 3 | 1006 | 50.3 | 1006 |
| 4 | 761 | 38.0 | 761 |
| 5 | 1697 | 84.8 | 1696 |
| 6 | 867 | 43.3 | 865 |
| 7 | 787 | 39.3 | 785 |
| 8 | 1065 | 53.2 | 1064 |
| 9 | 904 | 45.2 | 896 |
| 10 | 1093 | 54.6 | 1084 |
| calib-imu1 (control) | 939 | 46.9 | 939 |

## Calibration values used

| | cam0 (GS) | cam1 (RS) |
|---|---|---|
| fx, fy, cx, cy | 743.429, 743.555, 618.719, 506.728 | 739.165, 739.144, 625.826, 517.337 |
| equidistant k1..k4 | 0.023584, −0.006764, 0.010259, −0.003756 | 0.019328, 0.006784, −0.008659, 0.005189 |
| T_cam_imu translation (m) | (0.0525, 0.0078, −0.0424) | (−0.0569, 0.0076, −0.0424) |

- The full `T_cam_imu` matrices are in `prep_info.json`.
- Camera-to-world GT is `T_world_cam = T_world_imu · inv(T_cam_imu)`, using Kalibr's convention that T_cam_imu maps IMU-frame points into the camera frame.

## Rolling-shutter timing

- Row time is **29.4737 µs**. The dataset page says it is "approximated by the step size of the exposure time". The paper says ≈29.47 µs.
- Full readout = 1024 × 29.4737 µs = **30.18 ms**.
- RS timestamps "refer to the first row" (dataset page), and timestamps in general denote the middle of the exposure.
  - GT for each RS frame is therefore interpolated at the **centre row**: t + 511.5 × 29.4737 µs = **t + 15.08 ms**.
  - A sensitivity run with GT at the first-row time t is also reported (`ATE_rs_firstrow_gt` in `per_window.csv`).
  - GS frames use GT at t.
  - In the calibration control, cam1 is global shutter, so its offset is 0.

## Preprocessing (identical for GS and RS)

1. **Tone map:** 16-bit linear → `uint8 = 255 · (x / 65535)^(1/2.2)`. This is a fixed curve with no per-image normalisation. Both cameras have similar raw levels (median ≈ 11–12k of 65535).
2. **Undistort** (`cv2.fisheye`, bilinear) each camera with its own K, D to **one shared pinhole** K_new:
   - f = 644.43 px, c = (639.5, 511.5), 1280×1024;
   - f was chosen as the smallest focal length at which every output pixel is valid for both cameras.
3. **Grayscale → 3 channels:** each model's own loader does `PIL .convert("RGB")`, which replicates the channel.
4. **Resize:** each model's official loader at its default input size:

   | Model | Loader | Input size |
   |---|---|---|
   | MapAnything | `load_images`, fixed_mapping 518 | — |
   | VGGT | `load_and_preprocess_images`, "crop" | 518×420 |
   | π³ | `load_images_as_tensor`, PIXEL_LIMIT 255000 | 560×448 |

## Windows

- N = 16 frames, frame stride 3 (2.25 s span), window step 8 frames.
- A window is used only if **every** frame has valid GT for both cameras: the bracketing mocap samples are ≤ 25 ms apart.
- **Static filter rule:** drop windows whose GT camera-centre path length (GS camera, summed over the 15 steps) is < 0.20 m.
  - Main set: **1139 windows, 1124 kept**.
  - Median GT path 2.04 m, median extent 0.78 m.
- `omega_mean`: mean ‖gyro‖ over [t_first, t_last + 30.18 ms].
  - It correlates 1.000 with the mocap-derived rotation speed.
  - Range 0.11–3.60 rad/s, median 1.19.
- `rs_shift_px` = omega_mean × 30.18 ms × 644.43 px. Median 23 px, max 70 px.
- Speed bins (fixed, every bin has at least 15 windows):

  | Bin | ω range (rad/s) | Windows |
  |---|---|---|
  | slow | < 0.5 | 56 |
  | medium | 0.5–1.5 | 648 |
  | fast | ≥ 1.5 | 420 |

- **Control set (calib-imu1, GS vs GS):** 112 windows, all kept.
  - ω 0.19–1.72 rad/s; fixed bins would leave 14 fast windows, so terciles are used.
  - The scene is a calibration target, so it differs from seq 1–10.

## Code and versions

| Component | Version |
|---|---|
| map-anything | `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9` (the commit the other team verified), checkpoint `facebook/map-anything-apache` |
| vggt | `a288dd0f14786c93483e45524328726ab7b1b4ce`, checkpoint `facebook/VGGT-1B` |
| π³ code | `github.com/yyfz/Pi3` @ `9fa3ddb3f8d53041f8b2738df404f62223bbaa7b`, checkpoint `yyfz233/Pi3` |
| Python / torch | 3.11 (uv venv `~/rs3d/.venv`), torch 2.7.1+cu128 |
| Inference | bf16 autocast |

- `github.com/yyfz233/Pi3`, as named in the plan, returns 404. The official code now lives at `github.com/yyfz/Pi3`; the HF weights are still under `yyfz233`.
