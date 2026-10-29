# EXP2 2C: synthetic rolling-shutter generator — validation

**Status: all checks passed on all 4 scenes tested** (measured, 2026-10-29). The scenes are a procedural test room plus three Replica scenes (office_3, apartment_2, room_0). Two of the Replica scenes (room_0, apartment_2) are test-split scenes. Validation measures the renderer only; no model was run on them for this.

- Code: `src/rsgen/` (`core.py` renderer, `motion.py` trajectories, `scenes.py` meshes, `generate.py` clip writer, `post.py` post-capture variants, `validate.py` this suite).
- Raw results: `generator/validation_<scene>.json`.
- Fast-pan pictures: `generator/validation_fast_pan_<scene>.jpg`.
- Command: `python -m rsgen.validate --scene <mesh.ply> --up 2`.

## How the generator works (plain words)

- **3D scenes, not warped photos.** Each picture is made by shooting a ray through every pixel into a 3D mesh. This uses Open3D / Embree ray casting on the CPU. Hidden surfaces are therefore handled correctly, which warping a single photo cannot do.
- **Rolling shutter.** Each image line has its own capture time, and its rays leave from the camera pose at exactly that time (1 line per time band).
  - Line *l* of *L* is read at `t_frame + (l − (L−1)/2)·readout/L`. The frame timestamp is the middle-line time, as EXP0's ground truth uses.
  - readout = 0 gives a normal global-shutter (GS) camera.
- **Directions:** top→bottom, bottom→top and left→right (a portrait phone).
- **Motion blur:** each line averages several time samples across its exposure, in linear light. Blur is set separately from readout.
- **Saved per clip:**
  - the RS frames and their GS twins at the same instant;
  - RS and GS depth (RS depth is per pixel, in that pixel's own line camera);
  - the dense camera trajectory at 1 kHz, so the pose of any line can be looked up;
  - the middle-line poses, the true twist (ω, v) at each frame, and the intrinsics;
  - the shutter settings;
  - when present: neighbour frames ±1/fps and a moving-object mask.
- **Meshes:**
  - Replica's per-vertex colours are rendered; the HDR PTex textures are not used (DEVIATIONS D7).
  - Quads are split into triangles by our own PLY reader.
  - z is up (checked: room heights 2.8–5.3 m).

## Checks (brief §6) and results

Each check uses a demanding handheld clip with these targets: ~150°/s turning, 1.5 m/s walking, 30 ms readout. The measured speed at the checked frame is 89–171°/s and 1.2–1.8 m/s. The camera is 640×480 with a 70° field of view.

### Check 1: readout = 0 gives exactly the GS render

Tested for all 3 directions × band sizes 1 and 8.

| Scene | Identical? |
|---|---|
| test room | yes |
| office_3 | yes |
| apartment_2 | yes |
| room_0 | yes |

### Check 2: a known 3D point lands at its predicted RS pixel

- **Method:** ~3,000 surface points per direction are taken from the GS depth.
  - Their RS pixel is predicted by a fixed-point iteration (pixel → line time → projection).
  - A ray is then re-cast through that pixel at that line's pose; it must hit the same point.
- **Result:** p99 reprojection error is **0.000 px** in every scene and direction.
  - Occluded points (0–0.1%) are excluded.
- **How much RS moves pixels:**

| Scene | Median shift | Max shift |
|---|---|---|
| test room | 4.9–5.2 px | 17.6–20.7 px |
| office_3 | 6.3–6.8 px | 23.1–24.8 px |
| apartment_2 | 11.8–13.2 px | 33.5–37.6 px |
| room_0 | 6.4–6.9 px | 22.9–24.6 px |

### Check 3: straightening the RS render with the true motion gives back the GS twin

- PSNR against the GS twin, over pixels visible in both images (89–99% of pixels).
- Each cell gives the range over tb / bt / lr.

| Scene | Raw RS | Exact unroll (true per-line poses + true depth) | V4 GPU unroller (true ω, v, RS depth) | V3 GPU unroller (rotation only) |
|---|---|---|---|---|
| test room | 22.0–22.9 dB | **50.2–51.1 dB** (MAE 0.36–0.40 /255) | 38.5–41.1 dB | 30.5–34.5 dB |
| office_3 | 24.5–24.8 dB | **53.9–54.7 dB** (MAE 0.13–0.15) | 43.7–44.2 dB | 38.0–43.6 dB |
| apartment_2 | 29.5–32.3 dB | **58.1–58.2 dB** (MAE 0.08–0.09) | 49.2–50.4 dB | 40.2–45.6 dB |
| room_0 | 26.9–29.0 dB | **51.7–52.8 dB** (MAE 0.28–0.31) | 40.6–43.0 dB | 35.5–35.9 dB |

- **Exact unroll:** residuals are below one grey level, i.e. resampling noise only. This confirms the renderer's per-line geometry.
- **The GPU unroller (`gpu_unroll.py`) is the oracle used for V3/V4 in EXP2.** It assumes one constant twist per frame, so it is a little less exact. That gap is part of what the oracle can and cannot do. It is not a renderer error.

### Check 4: thinner time bands converge

- **Test:** doubling the number of bands (halving rows per band) must move pixels by < 0.5 px.
- **Result:** p99 change going 2 → 1 rows per band:

| Scene | p99 change |
|---|---|
| test room | 0.035 px |
| office_3 | 0.044 px |
| apartment_2 | 0.078 px |
| room_0 | 0.045 px |

- The change halves with each halving of the band, as expected.
- All EXP2 data uses 1-line bands, i.e. exact per-line timing.

### Check 5: visual check of a fast pan

- **Setup:** 200°/s pan while walking 1 m/s sideways, 30 ms top→bottom readout.
- **Panels:** GS | RS | exact unroll | |GS − unrolled| ×2.
- **Result:** the RS frame visibly shears.
  - The unrolled frame matches the GS twin.
  - The difference is limited to depth edges and the strip the RS camera never saw.
- **Images:** `generator/validation_fast_pan_{testroom,office_3,apartment_2,room_0}.jpg`.

### Check 6: the motion generator hits its speed targets

- **Setup:** 15 trajectories: 5 kinds × 3 speed settings (30–200°/s, 0.5–2 m/s).
- **Result:** max relative error of mean |ω| and mean |v| vs target is **0.5–0.6%**.

## Other checks done while building

These were measured on the VM.
- **Paired stress sets:** a clip generated under two configs that differ only in shutter, object or neighbour settings has an identical trajectory. This was checked numerically (position and orientation arrays equal), so every stress-set comparison is paired clip by clip (DEVIATIONS D6).
- **V5 maths:** feeding the true row-camera points and true poses into the V5 rebuild (`synth_eval.dense_metrics`) gives exactly the true points.
  - Per-pixel error: 0.000 m, vs 0.024 m when every row uses the frame pose.
  - Wall-plane deviation: 3 mm, vs 20 mm.
  - Clip: apartment_2, |ω| = 2.25 rad/s.
- **Model pixel maps:** `infer2.py` measures how each model resizes or crops its input, using dithered ramps. Accuracy is < 0.04 px at the image edges against exact PIL resizes.
  - VGGT: 518×420.
  - π³: 560×448.
  - MapAnything: 518×392, with ~28 rows cropped top and bottom.
- **Speed:** one 640×512 frame with 2×2 samples per pixel takes 0.56 s on 2 CPU threads. The core test set (240 clips, RS + GS + 2 neighbours per frame) took ~25 min on 10 workers.

## Generated sets

All configs are in `configs/synth/`, written by `configs/synth/make_configs.py`. The scene split is fixed in DEVIATIONS D4.

| Set | Scenes | Clips | What changes |
|---|---|---|---|
| core_trainval | 12 train + 2 val | 100 / scene (train), 30 (val) | readout {0,5,10,20,30,40} ms; direction tb/tb/tb/bt/lr; neighbours for clip ids 0–39 |
| core_test | 4 test | 60 / scene = 240 | 30 ms top→bottom, no blur, neighbours |
| st_readout{0,5,10,20,40} | test | 15 / scene | readout (30 ms = core_test clips 0–14) |
| st_dir_{bt,lr} | test | 15 / scene | scan direction |
| st_blur_r0_e16, st_blur_r30_e8, st_blur_r30_e16 | test | 15 / scene | exposure blur alone and with RS (8 time samples, 1 sample per pixel) |
| st_objects_r{0,30} | test | 15 / scene | 2 moving boxes, 0.5–2 m/s, with and without RS |
| st_lowtex | test | 15 / scene | colours pulled to the mean (contrast × 0.25) |
| st_views64 | test | 15 / scene | 64-frame clips (motion ≤ 160°/s, ≤ 1 m/s); first N = 2…64 views |
| st_stride{1,6,12} | test | 15 / scene | frame spacing 50 / 300 / 600 ms |
| core_test post-capture variants (`rsgen/post.py`) | test | clips 0–14 | noise, low light, gamma, crop, crop + resize, V3 on crop with correct / naive timing, simulated EIS |

## Known limitations

- **Appearance:** vertex colours only, no PTex textures, no lighting changes. Replica surfaces are a bit blurrier than real photos.
- **Scenes:** all indoor. Depth is mostly 1–6 m, so "far" scenes (outdoor) are not covered.
- **Moving objects** are textured boxes on straight paths, not people or vehicles.
- **The EIS model** is a per-frame rotation plus 1.1× zoom. Real phones may also correct RS per line; that variant is not modelled.
