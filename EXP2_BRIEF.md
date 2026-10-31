<!-- The EXP2 brief as given by the team on 2026-10-29 (verbatim). Deviations from it are recorded, with dates,
     in results/exp2/DEVIATIONS.md. -->

# EXP2: Know the Field, Build the Benchmark, Measure the Headroom

Project: RS-3D (rolling-shutter-native feed-forward 3D). Read `PROJECT_CONTEXT.md` and `EXPERIMENTS_SO_FAR.md` first.
Target: an A* venue (CVPR / ICCV / ECCV 2027). **Rigor comes before speed.** There is no time budget: work as fast as is optimal, but never cut a corner listed here.

---

## 0. Why EXP2 exists

EXP0 and EXP1 showed two things on **one** dataset (TUM RS-VIO):
- rolling shutter (RS) hurts feed-forward 3D models;
- straightening the photos with the **true** motion (gyroscope plus ground truth) removes almost all of the damage.

That is not yet a paper. A top reviewer will ask:
1. Is this universal, or one room and one camera?
2. What has already been published?
3. How much do existing methods already fix, including simple fine-tuning?
4. Should the fix live *inside* the model, or in the photos?
5. What about unknown readout times, portrait phones, unordered photos, moving objects, blur, stabilized phone video and normal cameras?

EXP2 answers all of these **before** we build our method (EXP3).

The paper's goal: one method that works from images alone (no gyroscope, no given depth, no need for video neighbours), across models, cameras, readout times and scenes, and never hurts normal-camera photos. Weaker results on some datasets are fine and will be reported honestly.

---

## 1. Ground rules (non-negotiable)

1. **Honesty**
   - Every number is labelled *measured* or *projected*.
   - Never invent papers, numbers or dataset facts.
   - If something can't be verified, write "UNVERIFIED".
2. **Citations**
   - Every paper is verified (title, authors, venue, year, arXiv ID or DOI) from its actual page.
   - No citation from memory alone.
3. **Pre-registration**
   - The decision rules in §10 are fixed before results are seen.
   - If you think a rule is wrong, write down why in `DEVIATIONS.md` **before** running, never after.
4. **No cherry-picking**
   - Report every dataset × model × condition you ran, including bad ones.
   - Any filtering rule is stated and applied uniformly.
5. **Statistics**
   - Medians with 1000-sample bootstrap 95% CIs.
   - Paired Wilcoxon tests where pairs exist.
   - Holm–Bonferroni correction when many tests answer one question.
   - Report n everywhere.
   - Training runs: 3 seeds, report mean ± std.
6. **No leakage**
   - Train/val/test splits are by **scene**, never by frame.
   - Test scenes are never used for tuning.
7. **Reproducibility**
   - Every run has a config file, a git commit hash, a seed and a log.
   - Commit early and often.
   - Pin model checkpoints (hash or exact version).
8. **Licenses**
   - Record each dataset's and model's license.
   - **Never accept a license or terms of use on the user's behalf.** If a dataset needs registration or signing, add it to `HUMAN_TODO.md` and continue with everything else.
9. **Deviations**
   - Anything done differently from this file goes in `DEVIATIONS.md`, with the reason.

---

## 2. Machine and setup

New machine, from scratch. The previous VM was terminated, so no data or outputs survive on it.

**Machine**
- 1× A100 SXM4 80 GB, 24 vCPUs, 116 GB RAM, ~2.1 TB disk (gpu.net, Denver/Houston).
- SSH: `ssh <user>@<NEW_VM_IP>`. Ask the user for the IP if it is not given.
- This GPU is ours alone unless the user says otherwise. Still check `nvidia-smi` before big launches.

**Setup steps**
1. **Local repo first.**
   - EXP1 code was never committed: `src/unroll.py`, `pi3_depth.py`, `run_e1.sh`, `evaluate_e1.py`, the `infer.py` change, and `results/exp1/`.
   - **Commit everything locally first**, then push or copy the repo to the VM at `~/rs3d/`.
2. **Layout on the VM:**
   - `~/rs3d/{src,configs,data,cache,results/exp2,logs,report}`
3. **Environments.**
   - One Python env per model if dependencies conflict.
   - Document the exact install commands in `results/exp2/ENV.md`.
4. **Speed (lesson from EXP1).** CPU work was the bottleneck there.
   - Cache resized and undistorted images once, as uint8 tensors.
   - Do warping and unrolling on the GPU (`torch` `grid_sample`).
   - Parallelize data prep across the 24 vCPUs.
   - Overlap downloads, literature work and GPU jobs.
   - Use `tmux` and log everything.
5. **Reproduction check.**
   - Re-download TUM RS-VIO.
   - Re-run EXP0 on TUM for π³ and VGGT.
   - The numbers must match `results/exp0` within bootstrap CI. If they don't, stop and find out why before continuing.

---

## 3. Models (all frozen in EXP2, except the training controls in §8)

| Model | Role | Notes (verify each) |
|---|---|---|
| π³ (github.com/yyfz/Pi3) | Main model | Largest RS effect in EXP0/1 |
| VGGT-1B | Main | Non-commercial checkpoint, fine for research |
| MapAnything (Apache weights) | Secondary | Barely affected in EXP0. Explaining *why* is a finding. |
| Depth Anything 3 | Additional | Verify public weights and API |
| VGGT-Ω | Additional, if public | Verify weights exist |
| One streaming model (e.g. CUT3R or StreamVGGT) | Online / robot use | Pick whichever has public, working code |

Settings:
- images only (no priors given);
- bf16;
- each model's default resolution, recorded.

Add any newer strong model the literature review (§4) finds.

---

## 4. Part 2A: Literature review (start immediately, in parallel with downloads)

Use web search and web fetch. Read the actual abstract, and the method sections where needed. **Deliverables:**
- `results/exp2/LIT_REVIEW.md`, in plain English;
- `results/exp2/related_work.csv`, with columns: title, authors, venue, year, id, what it does, what it does NOT do, code available (Y/N plus link), how we differ, threat level (high/med/low).

### Topics (each needs ≥ 6 searches with different wording)

1. **Scoop check, the most important.** Rolling shutter combined with feed-forward / foundation 3D models: DUSt3R, MASt3R, VGGT, π³, MapAnything, DA3, CUT3R, Fast3R, Spann3R.
   - Search arXiv **last 60 days** explicitly, plus CVPR/ICCV/ECCV/NeurIPS/ICLR/3DV 2025–2026 and OpenReview.
   - Check papers citing VGGT, π³ and MapAnything for "rolling shutter".
   - Adversarial search: imagine the paper that would kill ours ("rolling-shutter-aware VGGT", "RS-DUSt3R", "readout-time-aware pointmap", "row-wise pose transformer") and search for exactly that.
2. **Same genre:** adapting 3D foundation models to non-pinhole cameras.
   - Seeds (verify): RayTun3R (arXiv 2607.02711), Fisheye3R.
   - Plus anything on camera-model tokens or intrinsics conditioning.
3. **Learned 2D rolling-shutter correction / unrolling.**
   - Seeds (verify): DeepUnrollNet (Liu et al., CVPR 2020); SUNet (Fan et al., ICCV 2021); JCD (Zhong et al., CVPR 2021); Cao et al. adaptive warping / BS-RSC (CVPR 2022); self-supervised and event-based RS correction; any 2024–2026 work.
   - Record code and weights availability for each.
4. **Classical RS geometry.**
   - RS bundle adjustment (seeds: Hedborg et al. CVPR 2012; Liao et al. ICCV 2023).
   - RS minimal solvers (seeds: Albl et al.; Barath arXiv 2606.26863).
   - RS-SfM degeneracies.
   - Record code availability.
5. **RS in SLAM / VO / VIO.** Seed: Schubert et al., IROS 2019, plus newer RS-aware SLAM.
6. **RS in NeRF / Gaussian splatting.**
   - Seeds: "Gaussian Splatting on the Move" (Seiskari et al., ECCV 2024), plus RS-NeRF variants.
   - These are per-scene optimization; we are feed-forward. State the difference precisely.
7. **Camera velocity from a single RS image.** Seed: SMARSC (Zhuang et al., CVPR 2019). This is our closest idea, so state exactly what we add.
8. **Gyroscope-based RS correction.** Seed: Karpenko et al., Stanford tech report 2011. Also phone video stabilization (EIS/OIS) and how it re-warps frames.
9. **RS + motion blur joint models** (exposure time vs readout time).
10. **Dataset survey:** every public RS dataset (see §5). For each: sensor, readout time, scan direction, ground truth available, size, license.
11. **Claim verification.** Find the VGGT-Ω supplement passage about rejecting rolling-shutter videos. Quote it exactly with its page, or mark it UNVERIFIED.
12. **Benchmarks and metrics** used by the strongest 3D foundation-model papers, so our protocol matches theirs:
    - pose AUC@30, RRA/RTA, ATE;
    - point-map accuracy / completeness / Chamfer;
    - depth AbsRel and δ<1.25.

### Literature-review output must end with
- **Novelty verdict:** OPEN / CROWDED / TAKEN, with confidence %, the 3–6 nearest papers, and exactly how we differ.
- **Final baseline list:** methods with working code that we will run.
- **Final dataset shortlist.**
- **STOP RULE:** if a paper already does our exact idea (RS motion inside a feed-forward multi-view 3D foundation model, images only), stop all non-literature work and report to the user with the paper and a pivot proposal.

---

## 5. Part 2B: Data audit

**Deliverable:** `results/exp2/DATA_AUDIT.md`, plus a `datasets.csv` table.

For each candidate, verify from the official source and the downloaded files:
- RS or GS;
- readout time (or whether it is unknown);
- scan direction;
- ground truth available: camera poses / depth / 3D scan / IMU / GS twin;
- indoor or outdoor;
- dynamic objects present;
- size;
- license;
- download method.

Drop or flag anything that fails.

| Category | Purpose | Candidates (all to verify; add those found in 2A) |
|---|---|---|
| Real RS + true camera path | Real-camera pose accuracy | TUM RS-VIO (done before; GS twin camera); ADVIO (Cortés et al., ECCV 2018, smartphone); WHU-RSVI; others from 2A |
| Real RS + true 3D shape | Real-camera geometry accuracy | ScanNet++ iPhone stream (laser-scanned scenes; license sign-up needed → HUMAN_TODO); Waymo Open Dataset (RS cameras + LiDAR; verify rolling-shutter timing fields; license → HUMAN_TODO) |
| Real RS image pairs (2D) | 2D unroller sanity | Fastec-RS, Carla-RS, BS-RSC (verify what GT each has) |
| Synthetic RS (we generate it) | Perfect GT, controlled knobs | See §6 |
| Normal camera (GS) | "No harm" check | ETH3D; TUM RS-VIO cam0; EuRoC (verify GS); a standard GS benchmark used by π³/VGGT papers |

Rules:
- Real data whose readout time is unknown is still valuable: it tests the "unknown readout" case. Mark it clearly.
- For each dataset, write the exact frame-to-ground-truth timing used. For RS, use the middle-row time as in EXP0, and record the sensitivity check.

---

## 6. Part 2C: Synthetic rolling-shutter generator (needed for 2D–2G and later EXP3)

**Deliverables:**
- `src/rsgen/`;
- `results/exp2/GENERATOR_VALIDATION.md`.

**Rendering**
- Render from **3D meshes** with a GPU renderer (nvdiffrast, PyTorch3D, Open3D raycasting, or similar).
- Do NOT warp single images: warping can't handle occlusions correctly.
- Rolling-shutter rendering: render the scene at many sub-times within the readout and take each row (or thin row band) from its own time.
  - Use enough bands that the result converges. Test this: doubling the band count must change pixels by < 0.5 px equivalent.
- **Scenes:** public meshes with research licenses. Candidates: Replica, ScanNet++ meshes (if licensed), others found in 2A. Record licenses.
  - Split scenes into train / val / test **before** generating anything.

**Trajectories:**
- realistic handheld walk-and-pan;
- drone-like;
- pure rotation;
- pure translation;
- mixed.
- Turning speed swept from 0 to ≥ 200°/s; walking speed from 0 to ~2 m/s.

**Knobs (each independently controllable):**
- readout time ∈ {0, 5, 10, 20, 30, 40} ms;
- scan direction ∈ {top→bottom, bottom→top, left→right (portrait phone)};
- exposure time / motion blur (separate from readout);
- noise and gamma;
- resolution and crop;
- optional moving objects;
- optional simulated electronic stabilization (a per-frame warp applied after capture).

**Ground truth saved per frame:**
- per-row camera pose;
- mid-row pose;
- global-shutter render of the same instant (the "twin");
- depth (RS and GS);
- intrinsics;
- the readout settings used.

**Validation (all must pass before use):**
- readout = 0 → identical to the GS render;
- reprojection of a known 3D point lands at the predicted RS pixel;
- unrolling the RS render with the true motion matches the GS twin (report residual);
- visual check of a fast pan.

---

## 7. Parts 2D and 2E

### Part 2D: Is the problem universal? (EXP0 repeated everywhere)

Run every model on every audited dataset.

- **Where a GS twin exists** (TUM, synthetic): measure the RS÷GS ratio by speed bin, exactly like EXP0, including the fairness check where available.
- **Where no GS twin exists** (e.g. ADVIO, Waymo, ScanNet++ iPhone): measure how error grows with motion speed. Fit error vs angular speed (and walking speed), using IMU or ground-truth motion for the x-axis. Report the slope with CI.
- **Metrics:**
  - ATE (Sim(3)-aligned), RPE-rot, RPE-trans, pose AUC@30 / RRA / RTA;
  - where dense ground truth exists: depth AbsRel, δ<1.25, point accuracy, completeness, Chamfer;
  - on synthetic: **wall planarity** (PCA residual of known planar surfaces).
- **Clip protocol:** same as EXP0 (16 frames, stride 3, step 8) unless a dataset's frame rate requires a change. Record any change.

### Part 2E: The oracle ladder (best case with the answer key)

Run on synthetic (perfect ground truth) and TUM (gyroscope as answer key).

| Version | What | Tells us |
|---|---|---|
| V1 | GS twin images | Ceiling |
| V2 | RS images, no fix | Floor |
| V3 | RS straightened for turning with true motion | Value of fixing rotation |
| V4 | RS straightened for turning + walking with true motion and true depth | Value of fixing both |
| V5 | RS images untouched; model's output geometry rebuilt row by row with the true motion (row-time unprojection inside the model's output) | Whether fixing *inside* the model beats fixing the photos |

**Honest caveat to record for V5.** At inference time, V5 can only correct the **3D points**, not the model's pose estimate. The model's internal features already saw bent photos. So compare V5 against V3/V4 mainly on **dense geometry**.
- Whether an in-model design also fixes poses can only be tested with training. Put that in EXP3.
- Do not over-interpret V5 on pose metrics.

**Sensitivity of the oracle:**
- run V3/V4 with the readout time wrong by ±25% and ±50%;
- run with the scan direction wrong.

This tells us whether our method must *estimate* readout time.

---

## 8. Part 2F: Existing methods (the baselines our method must beat)

Run on every dataset where they apply:

1. **Plain model** (V2): already done in 2D.
2. **Learned 2D unrollers → 3D model.** Every one with working code and weights from 2A.
   - Use the authors' weights.
   - Also fine-tune the best one on our synthetic train split, so the comparison is fair on our data distribution.
3. **Classical RS-aware optimization** (RS-BA / RS-SfM) where code exists. Also GS COLMAP/GLOMAP as a reference.
4. **Self-estimate two-pass, training-free:**
   - estimate motion from the model's own first-pass poses (finite differences over neighbours);
   - straighten the photos;
   - rerun.
   - Report it on **ordered video** and on **shuffled / unordered** photo sets (where it should fail).
5. **Gyroscope straightening** where IMU exists (TUM, ADVIO). This is the "why not use the gyro" reference: not image-only, so not a competitor, but reported.
6. **Training controls** (training is allowed and required here):
   - **6a. LoRA-only on RS data:** fine-tune each main model (π³, VGGT) with LoRA on the synthetic RS train split, with the original losses and **no motion modelling**. This is the most important control.
   - **6b. LoRA on GS data, same budget:** same recipe on the GS twins of the same scenes. This separates "more training on these scenes" from "seeing rolling shutter".
   - **Training rules for both:**
     - frozen backbone + LoRA;
     - check whether the confidence loss destabilizes small-data fine-tuning (VGGT-Ω reportedly warns about this — verify), and record the choice;
     - early stopping on the val split;
     - 3 seeds;
     - evaluate on test scenes, real RS datasets **and** GS benchmarks (check for regression).

---

## 9. Part 2G: Stress tests

Run on the oracle versions (V2–V5) and every baseline from §8. Use synthetic data unless noted.

1. Readout time sweep: 0–40 ms.
2. Scan direction: top-down, bottom-up, portrait (sideways).
3. Motion type: pure rotation, pure translation, mixed; near vs far scenes (depth range).
4. Number of views: 2, 4, 8, 16, 32, 64.
5. Frame spacing: dense video vs sparse; **unordered photo sets**.
6. Moving objects (synthetic; Waymo if available). Expected: camera-motion modelling won't fix object motion. Measure and report honestly.
7. Motion blur / long exposure, separately and together with RS.
8. Crop and resize. Row timing must follow the crop. Test what happens if a method ignores this.
9. Simulated electronic stabilization; real stabilized phone footage if any dataset has it.
10. Low texture and low light.
11. **GS input:** every method, every model, on GS benchmarks. Must show no harm.

---

## 10. Part 2H: Analysis and decision (rules fixed now)

**Q1. Universal?**
- **YES** if both hold:
  - RS degrades performance (fast-motion ratio ≥ 1.3× fairness-adjusted, or a significant positive error-vs-speed slope after Holm correction);
  - this holds on **≥ 2 real datasets** plus synthetic, for **≥ 2 models**.

**Q2. Headroom?** Define:
- the oracle gain as V4 versus V2;
- the best existing image-only method as the best of 2D unrollers (incl. fine-tuned), classical RS-BA, self-estimate two-pass, LoRA-only 6a.

Then:
- **YES** if the best existing method recovers **< 60%** of the oracle gain on at least half of the (dataset × model) pairs.
- If it recovers ≥ 60% almost everywhere, our contribution is small → **report and propose a pivot**.

**Q3. Design?**
- If V5 beats V3/V4 on dense geometry by a significant margin → EXP3 builds the in-model design.
- Otherwise EXP3 builds learned motion estimation that straightens internally, or a hybrid.
- Record the evidence either way.

**Q4. Novelty?** Must be OPEN, or CROWDED with a clear, defensible difference (from 2A).

**Q5. Must we estimate readout time?** Yes if the ±25% error in 2E costs > 20% of the oracle gain.

**Outcomes:**
- **GO to EXP3:** Q1 yes, Q2 yes, Q4 OK.
- **PIVOT:** Q2 no, or Q4 taken. Write a proposal.
- **STOP:** Q1 no.

---

## 11. Deliverables (all in `results/exp2/`)

**Documents**
- `LIT_REVIEW.md`, `related_work.csv`
- `DATA_AUDIT.md`, `datasets.csv`
- `GENERATOR_VALIDATION.md`
- `ENV.md`, `DEVIATIONS.md`, `HUMAN_TODO.md` (licenses or sign-ups only the user can do)

**Result tables (CSV)**
- `gap/`: per-window CSVs, summary tables, plots, one folder per dataset
- `oracle/`: V1–V5 tables and plots, plus the sensitivity results
- `baselines/`: every method × dataset × model
- `training/`: configs, logs, curves, results for 3 seeds
- `stress/`: one table and plot per stress test

**Final report and project updates**
- `EXP2_REPORT.md`, which:
  - **starts with a plain-English summary** (no jargon, a short glossary);
  - then gives answers to Q1–Q5 with evidence and the GO/PIVOT/STOP verdict;
  - then has full tables;
  - then lists what EXP2 could not prove.
  - Every number is labelled measured or projected.
- An updated `PROJECT_CONTEXT.md` phase plan reflecting the EXP2 outcome.
- An updated `EXPERIMENTS_SO_FAR.md`.

---

## 12. Suggested order (optimize freely; record what you did)

1. **Commit the local repo**, set up the VM, and start all downloads.
2. **2A literature review and 2B data audit**, in parallel with the downloads.
3. **Reproduction check on TUM** (§2 step 5), then **2D gap tests on TUM** with the new models.
4. **2C generator:** build, validate, generate splits.
5. **2D on all real and synthetic data**, then **2E oracle ladder**.
6. **2F baselines.** Start the LoRA training controls as soon as the synthetic train split exists. They run on the GPU while CPU-side baselines run.
7. **2G stress tests.**
8. **2H analysis and report.**

**At the end of 2A, write a short checkpoint note** in `results/exp2/CHECKPOINT_2A.md`, giving:
- the novelty verdict;
- the final baselines;
- the final datasets.

Then continue without waiting, **unless** the STOP RULE in §4 triggers.

## 13. Do NOT

- Accept dataset licenses or terms on the user's behalf.
- Tune anything on test scenes.
- Drop a dataset, model or condition because results look bad.
- Report a single seed for training results.
- Cite a paper you have not opened.
- Delete raw outputs. Keep them, or document how to regenerate them.
