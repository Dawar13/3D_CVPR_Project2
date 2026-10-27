# PROJECT CONTEXT — RS-3D: Rolling-Shutter-Native Feed-Forward 3D

Target: **CVPR 2027 main conference** (registration ~Nov 10, submission ~Nov 16, 2026 — verify on the CVPR site).
Fallback: CVPR 2027 workshop (≤4 pages, keeps the idea free for a later main-track submission).
Team: Dawar (lead), Shubh (3D), Sujal. Experiments run by Claude Code on rented GPUs.
Time estimates are in **hours**, not days.

---

## 1. The problem (plain words)

Phone, drone, action-cam and most robot cameras use a **rolling shutter (RS)**. They read the image row by row, top to bottom, over ~20–33 ms. If the camera moves during that readout, each row is seen from a slightly different pose. That is why buildings lean in phone videos ("jello effect").

Feed-forward 3D models assume **one rigid pose per image** — VGGT, VGGT-Ω, π³, MapAnything, DA3. When they turn depth into 3D points, row 1 and row H get the same rotation R and translation t. For a moving RS camera this is physically wrong: bent walls and trajectory drift.

Example: panning at 90°/s with a 30 ms readout rotates the camera ~2.7° between the top and bottom rows. That is tens of pixels of shift.

## 2. Evidence the problem is real

- **Schubert et al., "Rolling-Shutter Modelling for Direct Visual-Inertial Odometry", IROS 2019** (arXiv 1911.01015).
  - Ignoring RS severely degrades VIO.
  - Their TUM RS-VIO dataset has **time-synchronized global-shutter (GS) and RS images**, IMU, and mocap ground-truth poses for 10 sequences. This is our free, paired control.
  - Note: our older PDF wrongly calls this ICCV 2019.
- **VGGT-Ω supplement (p. 25, per our notes — re-verify the exact wording):** its data filter hard-rejects videos with severe rolling shutter.
- **Classical RS geometry is still active.** Barath, "Rolling Shutter Relative Pose Estimation Made Practical" (arXiv 2606.26863, ECCV 2026).
- **Same genre, accepted by reviewers.** These papers show 3D foundation models break under non-pinhole cameras and fix it with light adapters:
  - RayTun3R (arXiv 2607.02711), fisheye;
  - Fisheye3R (2026).
  - This also means RS is the obvious "next" one, so **scoop risk is real**.

## 3. Closest prior work and what is new

- **SMARSC** (Zhuang et al., CVPR 2019): a CNN predicts per-image camera velocity + depth from a single RS image, then corrects it. So "a network predicts velocity" is NOT new.
- **Classical RS-SfM / RS bundle adjustment / RS-VIO:** optimization-based, slow, needs features/IMU.
- **RS unrolling networks** (DeepUnrollNet, CVPR 2020, and successors): 2D image correction only.
- **NEW (our open leaf):** RS motion modelled *inside* a multi-view feed-forward 3D foundation model. Per-frame twist + row-wise Lie-group unprojection, trained with a light LoRA.
  - Quick novelty check: no paper found.
  - Confidence ~60–65% open. **Re-run the novelty search (last 60 days) before writing.**

## 4. Method (one paragraph + equations)

For each frame i, the camera head predicts the mid-frame pose T_i = (R_i, t_i) plus a body twist (ω_i, v_i):
- ω_i = angular velocity (main signal);
- v_i = linear velocity (poorly conditioned — Barath's v–t coupling — so regularize toward 0 / optional).

Row time: s = r · (row − H/2) / H, where r = readout_time / frame_interval.

Per-row pose: T_i(s) = T_i · exp(s · [ω_i, v_i]^). Small-motion form: R(s) ≈ R(I + s[ω]×), t(s) ≈ t + s·v.

Unprojection: X(u, s) = R_i(s) · (ray(u, s) · D(u, s)) + t_i(s).

If ω = v = 0 this reduces **exactly** to the original model, so clean/GS accuracy is preserved by design.

Training:
- frozen backbone + LoRA + velocity head;
- synthetic RS frames rendered from public datasets with depth + video-rate poses (ScanNet++, TartanAir), by per-row pose interpolation and depth-based warping;
- turn off the confidence loss when fine-tuning on small data (VGGT-Ω warning).

## 5. Phases, goals, GO / STOP rules

### Phase 0 — Gap test (E0) — ~1–2 h · NOW
- **Goal:** prove vanilla models get worse on RS than on GS, and that it worsens with camera speed.
- **Setup:**
  - TUM RS-VIO, matched GS vs RS clips;
  - models: MapAnything, VGGT, π³;
  - metrics: ATE (Sim3-aligned), RPE-rot, RPE-trans;
  - bins: angular speed.
- **GO:** fast-bin median error ratio RS/GS ≥ 1.5× AND the ratio grows with angular speed AND slow-bin ratio ≈ 1 (≤ 1.2×).
- **GRAY (1.2–1.5×):** run Phase 1 anyway; if oracle headroom is small → workshop.
- **STOP:** fast-bin ratio < 1.2× (models already robust, e.g. from phone-video training data like CO3D). Drop as main paper.
- Details: see `EXP0_GAP_TEST.md`.

### Phase 1 — Oracle headroom (E1) — ~2–3 h
- **Goal:** if motion were known exactly, how much of the gap closes?
- **Setup:**
  - unroll RS images with GT ω (from gyro/mocap);
  - rotation-only per-row homography H(s) = K · R(s)ᵀ · K⁻¹ (needs no depth);
  - rerun the models.
- **GO:** recovers ≥ 50% of the GS–RS gap. **STOP:** < 25%.

### Phase 2 — Killer cheap baseline (E2) — ~2–3 h
- **Goal:** check whether a training-free fix already works.
- **Setup:** estimate ω from the model's own neighbouring-frame poses (finite difference), unroll, run a second pass.
- **If ≈ oracle:** pivot to a training-free paper (cheaper, still publishable).
- **If ≪ oracle:** a learned velocity head is justified. Go to Phase 3.

### Phase 3 — Learned velocity head + LoRA (E3) — ~4–6 h first run
- Build the synthetic RS renderer, train LoRA + twist head, evaluate on held-out synthetic data and TUM RS-VIO.
- **GO:** clearly beats the **LoRA-only control** (same data, no twist head) AND no regression on GS/clean data.
- **STOP:** gain ≈ LoRA-only (the math is not the cause).

### Phase 4 — Full benchmark + ablations — remaining time before deadline
- **Backbones:** MapAnything + VGGT (+ π³ if time).
- **Baselines:**
  1. vanilla;
  2. unroll-then-reconstruct (DeepUnrollNet or newer);
  3. E2 training-free;
  4. LoRA-only;
  5. ours.
- **Metrics:** ATE / RPE on TUM RS-VIO; Points rel + inliers on synthetic-RS ScanNet++/ETH3D; GS clean-data check.
- **Target deltas (PROJECTIONS, not results):** fast motion ~30–50% lower ATE than vanilla; ≈0 change on slow/GS.

### Phase 5 — Writing — teaser figure
Real fast-pan scene: vanilla → bent wall; ours → straight wall.

## 6. Threats

1. Models may already be partly RS-robust. E0 tests this.
2. GS and RS cameras in TUM RS-VIO are different physical cameras (position, exposure). The slow-bin ratio is our confound check.
3. TUM RS-VIO has **poses only** (no dense GT, as far as we know) → dense metrics must come from synthetic RS.
4. v (linear velocity) is ill-conditioned → focus on ω.
5. Scoop risk (fisheye adapters already out) → move fast, re-search arXiv weekly.

## 7. Compute / infra

- GPU instance "3D-Exp-1": `ssh gpunet@216.81.248.38`, $1.38/hr.
- **Shared** with the MapAnything corrupted-priors E0 baseline run.
- Rules on the shared GPU:
  - own directory `~/rs3d/`;
  - own venv;
  - never touch the other run's files/processes;
  - check `nvidia-smi` before launching;
  - run in `tmux`, log to file.

## 8. Conventions

- Every number in notes is labelled **measured** or **projected**.
- Every citation is verified (title, authors, arXiv ID) before it enters the paper.
- Results go to `~/rs3d/results/<exp_id>/` with `summary.md`, CSVs and plots.
