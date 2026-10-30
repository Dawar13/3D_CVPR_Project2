# Scoop check (Topic 1): rolling shutter x feed-forward / foundation 3D models

Slice: `scoop`. Searches were run on 2026-10-29. Every paper listed below was opened in this session. The URL actually read is in `scoop.csv` (`url_opened`). Helper scripts and raw API dumps are in `results/exp2/lit_raw/scoop_work/`.

---

## 1. Plain-English findings

1. **No paper found that does our exact idea.** Our idea is to predict per-frame camera angular and linear velocity inside a feed-forward multi-view 3D foundation model (FM) from images alone, unproject the pointmap row by row using row time, and LoRA-tune the model. Searches covered:
   - arXiv: all RS papers since 2025-01-01, plus 52 adversarial API queries;
   - every abstract citing VGGT, DUSt3R, MASt3R, π³ and MapAnything (3,386 unique papers via Semantic Scholar);
   - the venues listed in Section 5 and OpenReview;
   - 37 web searches.

   None of these turned up a paper that puts RS motion inside DUSt3R, MASt3R, VGGT, π³, MapAnything, DA3, CUT3R, Fast3R, Spann3R, MonST3R or a similar model.

2. **2025–26 RS work falls into four groups, none of which is a feed-forward FM.**
   - Classical geometry. ECCV 2026 has three RS geometry papers: Barath's RS relative pose (evaluated on TUM RS), RS self-calibration, and velocity of spinning spheres.
   - Per-scene optimisation: RS-Avatar, Gaussian splatting on Project Aria data, TRGS-SLAM.
   - 2D restoration: flicker-banding, HDR, thermal.
   - Forensics and astronomy.

3. **The closest work comes from three directions. Each misses at least one of our three ingredients:** (i) RS, (ii) inside a multi-view feed-forward FM, (iii) images only.
   - **Images-only camera velocity from one image.**
     - *Image as an IMU* (ICCV 2025) gets velocity from motion blur. Its paper says explicitly: "we do not model for rolling shutter".
     - An unpublished University of Toronto course report extends that idea to a single RS image, using a two-stage flow + least-squares pipeline.
     - The classical CNNs of Zhuang et al. (CVPR 2019) and Mo et al. (2020, IMU-assisted) predict RS motion and depth from a single image.
   - **Time-conditioned pointmaps, but for time *between* frames.**
     - Interp3R needs an event camera.
     - Point4Cast (CVPR 2026), Uni4R (NeurIPS 2026) and V-DPM (CVPR 2026) model scene dynamics across frames, not camera motion during readout. Point4Cast does not mention RS in its abstract; Uni4R and V-DPM have zero RS mentions in their full text.
   - **A feed-forward FM that touches RS only outside the network.** *Foundation4D* (NVIDIA, arXiv 2610.30340, 21 Oct 2026, inside our 60-day window):
     - predicts a time-varying ego trajectory;
     - uses RS exposure boundaries when generating depth labels and when querying poses for downstream Gaussian rendering.

     Its network still outputs one pose and a global-shutter geometry per image. RS is not in its abstract, so the arXiv abstract scan missed it; a full-text web search found it.

4. **RS combined with a depth FM appears only as "correct the image, then run the FM."** UF's *Anchor Points* paper (2026) fine-tunes a Marigold U-Net to de-roll a single image, then runs a monocular depth FM. It estimates no pose and uses no multi-view geometry.

5. **No paper measures how much RS degrades feed-forward 3D FMs.** That is our EXP0 result: π³ pose error rises up to 2.6×.
   - The nearest measurement studies cover Agisoft Metashape (Verhoeven, ISPRS 2026).
   - A UAV SLAM benchmark tests DUSt3R and MASt3R under motion blur, not RS.

6. **Positioning advice for the paper.**
   - **Safe claim:** "first to model intra-frame (row-time) camera motion inside a feed-forward multi-view 3D foundation model, from images only".
   - **Claims to avoid:**
     - "first to estimate camera velocity from images" (Zhuang 2019, Image-as-IMU);
     - "first time-conditioned pointmap" (Interp3R, Point4Cast);
     - "first FM to consider RS" (Foundation4D uses RS in labels and rendering).
   - **Zhuang 2019 proves that RS two-view geometry is degenerate under pure translation.** That is directly relevant to our ω–v collinearity issue; cite it there.

---

## 2. Novelty verdict

**OPEN, 80% confidence.**

The 20% residual risk comes from four sources:

1. Unpublished concurrent work aimed at the CVPR 2027 deadline. The obvious candidate groups are:
   - Gallego (Interp3R, continuous-time pointmaps);
   - Clark (Image-as-IMU);
   - Hruby / Pollefeys / Kohn (RS geometry);
   - NVIDIA (Foundation4D already has a continuous ego trajectory and RS-aware rendering).
2. Abstract-level filters miss RS mentioned only in body text, as Foundation4D shows.
3. The arXiv index looked small in this environment (Section 7).
4. Journals (TPAMI, TIP, RA-L, IJCV) were not scanned systematically.

**Nearest papers and how we differ:**

| # | Paper | What it shares with us | Exactly how we differ |
|---|---|---|---|
| 1 | **Foundation4D** (arXiv 2610.30340, NVIDIA, 2026) | Feed-forward multi-view FM; time-varying ego trajectory; RS exposure boundaries used to query poses | RS appears only in label generation and downstream 3DGS rendering. The network predicts one pose and a global-shutter depth/pointmap per image, for a calibrated driving rig. We put row time into the pointmap itself (per-row unprojection with per-frame ω,v) for generic handheld cameras, images only. |
| 2 | **Image as an IMU** (Chen & Clark, ICCV 2025, arXiv 2503.17358) | Images-only instantaneous ω,v of the camera | Single image, signal from motion blur. States it does *not* model RS. Velocity only, no multi-view reconstruction. We use RS row timing across many views inside a 3D FM and feed ω,v back into geometry. |
| 3 | **Rolling Shutter Image as an IMU** (Shyamala, U. Toronto course report, 2025) | Images-only ω,v from RS distortion | Single image; two-stage (RS→GS flow + mono depth, then least squares); needs readout time; not peer-reviewed; no multi-view FM, no pointmap. We are end-to-end inside π³/VGGT with LoRA and multi-view pose output. |
| 4 | **Interp3R** (Guo et al., arXiv 2603.14528) | Pointmap FM queried at arbitrary time; joint depth + pose | Needs an event camera. Time is *between* frames, not within a frame's readout; zero RS mentions in full text. We use images only and intra-frame row time. |
| 5 | **Anchor Points … Severe RS Ego-motion** (Carter et al., UF, 2026) | RS + foundation model; reference ("anchor") row time | Image-space de-rolling (Marigold U-Net) before a monocular depth FM; single view; no pose or velocity; RS-GS paired training data. We never undistort images and we output multi-view poses. |
| 6 | **Point4Cast** (Liu et al., CVPR 2026) | Time-conditioned pointmaps and cameras from a feed-forward model | Time indexes frames of a dynamic scene; each frame is still treated as global-shutter. We index rows inside a frame to absorb ego-motion during readout. |

Classical prior to cite in the same paragraph: Zhuang et al., CVPR 2019. It is a single-RS-image CNN that predicts camera motion and depth for 2D correction, and it is the earliest images-only "network predicts RS motion" work.

---

## 3. Full query log

### 3a. arXiv API (export.arxiv.org)

Scripts: `scoop_work/arxiv_scan.py`, `arxiv_adv.py`, `arxiv_adv2.py`. Raw outputs: `arxiv_scan.json`, `arxiv_adv.txt`, `arxiv_adv2.txt`.

**RS scan**, sorted by `submittedDate` descending, with `max_results=200` and date filter `submittedDate:[202501010000 TO 202610302359]`:

| Query | Results |
|---|---|
| `abs:"rolling shutter"` | 40 |
| `ti:"rolling shutter"` | 10 |
| `abs:"rolling-shutter"` | 40 |
| `ti:"rolling-shutter"` | 10 |
| `abs:"rolling shutters"` | 40 |
| `abs:rolling AND abs:shutter` | 40 |

The union is **40** papers, and all 40 pass a strict regex `rolling[\s-]+shutter` on title or abstract.

**Index sanity checks** (no date filter):

| Query | Results |
|---|---|
| `abs:"rolling shutter"` | 150 total |
| `ti:"rolling shutter"` | 62 |
| `all:"rolling shutter"` | 150 |
| `all:rolling AND all:shutter` | 150 |
| `abs:"rolling shutter" AND cat:cs.CV` | 108 |

- The arXiv website search (`arxiv.org/search/?query="rolling shutter"&searchtype=all`) also reports **150 results**.

**Adversarial batch 1** (34 queries, `all:` fields, sorted by date). Hit counts are in brackets:

- RS combined with each FM name:
  - `shutter AND DUSt3R` [0]
  - `shutter AND VGGT` [0]
  - `shutter AND MASt3R` [0]
  - `shutter AND pointmap` [0]
  - `shutter AND "feed-forward"` [1, irrelevant]
  - `shutter AND "foundation model"` [1, irrelevant]
  - `shutter AND CUT3R` [0]
  - `shutter AND MonST3R` [0]
  - `shutter AND "Depth Anything"` [0]
  - `shutter AND MapAnything` [0]
- `jello` [4, all non-CV]
- Readout combined with FM names:
  - `readout AND pointmap` [0]
  - `readout AND VGGT` [2, irrelevant]
  - `readout AND DUSt3R` [0]
- Blur combined with FM names:
  - `"motion blur" AND DUSt3R` [1]
  - `"motion blur" AND VGGT` [1]
  - `"motion blur" AND pointmap` [0]
  - `blur AND MASt3R` [3]
- Events combined with FM names:
  - `event AND VGGT` [3]
  - `event AND DUSt3R` [4]
  - `"event camera" AND pointmap` [0]
- Velocity and scene flow:
  - `velocity AND pointmap` [1]
  - `velocity AND VGGT` [5]
  - `"scene flow" AND VGGT` [2]
- Intra-frame and row-wise pose:
  - `"intra-frame" AND pose` [23]
  - `"per-row" AND pose` [3]
  - `"row-wise" AND pose AND camera` [4]
- Exposure and continuous time:
  - `exposure AND VGGT` [3]
  - `"continuous-time" AND VGGT` [0]
  - `"continuous-time" AND DUSt3R` [1]
- RS combined with method terms:
  - `"rolling shutter" AND transformer` [8]
  - `"rolling shutter" AND learning AND pose` [7]
  - `"rolling shutter" AND depth` [17]
  - `"rolling shutter" AND reconstruction` [34]

**Adversarial batch 2** (18 queries):

- Timing terms:
  - `"readout time"` [125, mostly physics]
  - `"line delay" AND camera` [2]
  - `scanline AND pose AND learning` [1]
  - `"row-dependent"` [16]
  - `"sub-frame" AND pose` [6]
  - `"intra-frame motion"` [3]
- RS combined with other terms:
  - `"rolling shutter" AND "foundation"` [3]
  - `"rolling shutter" AND "ego-motion"` [2]
  - `"rolling shutter" AND velocity` [23]
  - `"rolling shutter" AND "multi-view"` [5]
  - `"rolling shutter" AND "pose estimation"` [11]
  - `"rolling shutter" AND SLAM` [10]
  - `"rolling shutter" AND "Gaussian splatting"` [9]
  - `"rolling shutter" AND "neural"` [20]
  - `"rolling shutter" AND "smartphone"` [7]
- Other:
  - `"motion blur" AND "feed-forward" AND pose` [2]
  - `"camera velocity" AND transformer` [2]
  - `"exposure time" AND pointmap` [0]

**No arXiv API query returned a paper combining RS with a feed-forward 3D FM.**

### 3b. Semantic Scholar Graph API

Scripts: `scoop_work/s2_cites.py` (run 1: VGGT), `s2_cites2.py` (run 2: the other four), `s2_meta.py`.

- The endpoint is `/paper/{arXiv:ID}/citations?fields=title,abstract,externalIds,year,venue,publicationDate,authors&limit=1000&offset=…`, paginated until `next` was absent.
- The filter regex was `rolling[\s-]*shutter|jello|read-?out`, applied to title and abstract.
- A second, wider regex `shutter|scanline|scan-line|row-wise pose|per-row|intra-frame|exposure time|during exposure` was applied to the union of all citing sets.

### 3c. OpenReview API v2

Script: `scoop_work/openreview_search.py`. Output: `openreview_rs.json`.

- The endpoint is `api2.openreview.net/notes/search`, with `type=terms, content=all, group=all, source=forum`, offsets 0–900.
- Terms searched:
  - `"rolling shutter"` (152 forums)
  - `"rolling-shutter"` (152)
  - `rolling shutter` (1322)
  - `jello` (4)
  - `readout pose` (≥1000)
  - `rolling shutter VGGT` (1536)
  - `rolling shutter DUSt3R` (1410)
  - `rolling shutter feed-forward` (≥1000)
- 2,912 unique forums were collected, then filtered by the RS regex on title and abstract.

### 3d. Venue listings

| Venue | How it was scanned |
|---|---|
| CVPR 2025, ICCV 2025, CVPR 2026, WACV 2026 | `openaccess.thecvf.com/<conf>?day=all` downloaded with curl; titles grepped with `scoop_work/cvf_grep.py` for `rolling\|shutter\|readout\|scanline\|jello\|wobble\|row-wise\|exposure`, plus a second grep of RS/blur/event titles that also contain 3D terms |
| ECCV 2026 | `eccv.ecva.net/Conferences/2026/AcceptedPapers` and `eccv.ecva.net/virtual/2026/papers.html` (curl + grep) |
| 3DV 2025 / 2026 | Schedule CSVs (`3dvconf.github.io/2025/schedule_updated2.csv`, `3dvconf.github.io/2026/schedule.csv`) read with WebFetch in three chunks each; curl returned empty bodies |
| NeurIPS 2025 / 2026, ICLR 2026 / 2027 | OpenReview search (3c). The neurips.cc and iclr.cc virtual pages downloaded truncated (320 KB / 128 KB / 16 KB), so they were only partly grepped. |

### 3e. Web searches (37)

**Direct scoop: RS combined with an FM**
1. `rolling shutter DUSt3R`
2. `rolling-shutter-aware VGGT`
3. `"RS-DUSt3R" OR "RS-VGGT" rolling shutter`
4. `rolling shutter feed-forward 3D reconstruction pointmap` (extended)
5. `rolling shutter MASt3R SLAM`
6. `rolling shutter VGGT fine-tuning LoRA row time`
7. `"rolling shutter" "feed-forward" Gaussian splatting reconstruction 2025`
8. `rolling shutter monocular video depth pose foundation model robustness benchmark DUSt3R MonST3R`
9. `"rolling shutter" CUT3R OR Spann3R OR Fast3R OR "Pi3" OR MapAnything`
10. `"rolling shutter" "VGGT"` (extended)
11. `"rolling shutter" "Depth Anything 3" OR "π³" OR "MapAnything" camera pose error smartphone`
12. `"rolling shutter" "foundation model" pose depth video 2026 arXiv unified` (extended). This is the query that surfaced Foundation4D.
13. `feed-forward driving scene reconstruction multi-camera "rolling shutter" ego trajectory Gaussian VGGT-style 2026` (extended)

**Adversarial wordings**

14. `rolling shutter pose estimation transformer per-row camera pose network`
15. `rolling shutter correction foundation model 3D geometry 2026`
16. `readout-time-aware pointmap rolling shutter unprojection row time`
17. `camera motion during exposure pointmap regression transformer velocity`
18. `rolling shutter generalizable novel view synthesis unposed images feed-forward network`
19. `jello effect wobble 3D reconstruction transformer smartphone video VGGT robustness`
20. `per-scanline pose regression rolling shutter transformer row embedding 3D`
21. `network predicts camera angular and linear velocity from images rolling shutter learned self-supervised depth`
22. `rolling shutter aware monocular depth and ego-motion learning 2025 2026 arXiv`. This is the query that surfaced UF Anchor Points.

**Adjacent work (blur, events, time, robustness)**

23. `event camera VGGT feed-forward 3D reconstruction pose transformer`
24. `motion blur aware feed-forward pointmap DUSt3R blurry images camera trajectory during exposure`
25. `continuous-time pointmap sub-frame timestamp feed-forward 3D reconstruction`
26. `robustness benchmark 3D foundation models corruptions motion blur DUSt3R VGGT image degradation`
27. `VGGT-SLAM OR MASt3R-SLAM handheld phone rolling shutter distortion fast motion failure`

**Venues**

28. `CVPR 2026 rolling shutter paper 3D reconstruction`
29. `ECCV 2026 rolling shutter accepted paper`
30. `NeurIPS 2026 OR ICLR 2027 rolling shutter 3D geometry transformer openreview`
31. `ICCV 2025 rolling shutter paper`
32. `3DV 2026 rolling shutter OR "rolling-shutter" paper`
33. `NeurIPS 2025 rolling shutter 3D reconstruction pose`

**Lookups**

34. `Point4Cast streaming 4D point map forecasting arXiv`
35. `Interp3R github continuous-time 3D geometry frames events code`
36. `EAG3R github event-augmented 3D geometry estimation code`
37. `V-DPM 4D Video Reconstruction with Dynamic Point Maps arXiv`

### 3f. Full-text checks

`scoop_work/html_grep.py` was run on arXiv HTML pages to count "rolling shutter" mentions in the body text:

| Paper | arXiv ID | RS mentions in body |
|---|---|---|
| Image as an IMU | 2503.17358 | 4. Includes the quote "we do not model for rolling shutter". |
| Foundation4D | 2610.30340 | 4 |
| RS-Avatar | 2608.17314 | Many |

These papers have **0** RS mentions in the body text:

- Interp3R (2603.14528)
- Uni4R (2608.09613)
- V-DPM (2601.09499)
- DynamicVGGT (2603.08254)
- Event3R (2607.15727)
- Fin3R (2511.22429)
- MCSplat (2610.25345)
- Drive2Gauss (2610.26585)
- FRUC (2605.29997)
- ReconDrive (2603.07552)
- ProMo3D (2610.33802)

---

## 4. arXiv scan: every RS paper since 2025-01-01

**Count: 40 papers** with "rolling shutter" or "rolling-shutter" in title or abstract and v1 submitted between 2025-01-01 and 2026-10-29: 17 in 2025 and 23 in 2026.

**0 of 40 combine RS with a feed-forward or foundation 3D model.** A keyword grep of all 40 abstracts for `dust3r|mast3r|vggt|feed-forward|foundation|pointmap|transformer|cut3r|monst3r|fast3r|spann3r|mapanything|depth anything` found only incidental uses of "foundation", in RIFLE and in the Single-Scanline paper's "foundational building block".

### 4a. Last 60 days (v1 submitted on or after 2026-08-30): 5 papers, all irrelevant to us

Each abstract page was opened with WebFetch.

| arXiv | Date | Title | Relevance note |
|---|---|---|---|
| 2610.29272 | 2026-10-21 | Certified Abstention for Provably-Unidentifiable Index-Phase Recovery in Image Registration | Irrelevant. Image-registration identifiability theory; RS is one of three example modalities. |
| 2610.22440 | 2026-10-15 | An Agentic ENF Framework for Forensic Video Authentication Against Generative AI | Irrelevant. Mains-frequency forensics from RS flicker. |
| 2610.20549 | 2026-10-14 | Frame Rate, Not the Grid: A Confound in Single-Clip Optical ENF Video Forensics | Irrelevant. ENF forensics letter. |
| 2610.03928 | 2026-10-02 | DABACO: A Multi-Camera Dataset and Benchmark for Screen Localization and Pointing Estimation | Irrelevant. Pointing dataset that includes RS camera modules. |
| 2609.28439 | 2026-09-23 | HaRP: High Dynamic Range Photosequencing through Dual Reversed Shutter Scanning | Irrelevant. 2D HDR imaging with dual reversed RS (also an ICLR 2027 submission). |

**Borderline and extra items for this window:**

- **2608.28304** (2026-08-28, two days before the window) corrects meteor velocity for RS detectors. Astronomy; irrelevant.
- **Older RS papers re-versioned inside the window:**
  - 2509.24644 RIFLE: v1 2025-09-29, latest version 2026-10-25. Flicker-banding; irrelevant.
  - 2506.19445 SloMoDeblur: latest version 2026-09-08. Deblurring dataset; irrelevant.
- **Inside the window but missed by the abstract scan** (RS only in body text):
  - **2610.30340 Foundation4D** (2026-10-21). **Relevant, med threat.** See Section 2.
- **Inside the window, adjacent, no RS:**
  - 2610.33802 ProMo3D (NeurIPS 2026);
  - 2610.24605 RoMo-GS (exposure-integrated blur);
  - 2610.25345 MCSplat and 2610.26585 Drive2Gauss (feed-forward driving 3DGS; zero RS mentions in full text).

### 4b. The other 35 (2025-01-01 to 2026-08-29), one line each

**2026**

- 2608.28304 meteor velocity RS correction: astro, irrelevant.
- 2608.17682 VoroTracing: ray tracer that supports RS rendering; irrelevant.
- 2608.17314 RS-Avatar: per-subject scanline-aware avatar optimisation; low.
- 2608.01509 RS self-calibration (ECCV 2026): low.
- 2606.31760 sphere velocity and spin from RS (ECCV 2026): low.
- 2606.29845 flicker-banding RAW dataset: irrelevant.
- 2606.26863 RS relative pose made practical (ECCV 2026): low; useful classical baseline on TUM RS.
- 2606.19067 quadruped SLAM sensor study: irrelevant.
- 2606.19007 X-ray DEPFET detector: irrelevant.
- 2606.06250 Gaussian-belief continuous-time SLAM: low.
- 2605.29118 sub-frame timing for aurora: irrelevant.
- 2605.22423 Moment-Reenacting (TPAMI): 2D blur/RS inversion; irrelevant.
- 2605.21079 VDFP deflicker: irrelevant.
- 2603.20443 TRGS-SLAM: IMU-aided thermal 3DGS; low.
- 2603.11888 single-view RS SfM: low.
- 2602.08474 optical camera communication: irrelevant.
- 2602.05617 XSIM driving sensor sim with RS: low.
- 2601.15222 MonoRace drone racing: irrelevant.

**2025**

- 2512.05268 CARD correlated-noise restoration: irrelevant.
- 2510.09996 BurstDeflicker (NeurIPS 2025 D&B): irrelevant.
- 2509.24644 RIFLE flicker-banding: irrelevant.
- 2509.18087 RnGCam (ICCV 2025): compressive video; irrelevant.
- 2509.10759 4D-GRT camera-effect data generation: low.
- 2508.17537 async-track minimal solvers (3DV 2026): low.
- 2507.22733 linear N-point async solver: low.
- 2507.16861 LiDAR-camera BEV alignment (CVPR 2026): irrelevant.
- 2507.12489 LiDAR resimulation: irrelevant.
- 2506.22069 single-scanline RS relative pose (ICCV 2025): low.
- 2506.19445 SloMoDeblur: irrelevant.
- 2506.04444 egocentric photoreal reconstruction (SIGGRAPH 2025): low.
- 2505.11562 space CMOS sensor: irrelevant.
- 2505.06668 StableMotion RS correction with SD prior: low.
- 2504.18500 Boxi sensor rig: irrelevant.
- 2504.11031 calibration image acquisition: irrelevant.
- 2504.01997 semantic SLAM with RS + INS: low.

---

## 5. Venue check (RS combined with a 3D FM)

| Venue | What was scanned | RS papers found | RS + 3D FM? |
|---|---|---|---|
| CVPR 2025 | 2,871 CVF titles | Order-One RS Cameras (theory) | No |
| ICCV 2025 | 2,701 CVF titles | Single-Scanline RS Relative Pose; RnGCam | No |
| CVPR 2026 | 4,042 CVF titles | **No RS-titled papers.** Adjacent: Point4Cast, V-DPM, Unblur-SLAM | No |
| ECCV 2026 | eccv.ecva.net accepted list + virtual list | RS Relative Pose Made Practical; RS Camera Self-Calibration; Velocity & Spin of Spheres from RS. Adjacent: EventVGGT, 4D-VGGT, OmniX | No |
| NeurIPS 2025 | OpenReview search; neurips.cc list (truncated) | BurstDeflicker (D&B, flicker). Adjacent: EAG3R (events), Fin3R (LoRA) | No |
| NeurIPS 2026 | OpenReview search; arXiv comments fields | None found. Adjacent: Uni4R, ProMo3D | No |
| ICLR 2026 | OpenReview search; iclr.cc list (truncated) | RIFLE (rejected); one withdrawn LiDAR paper | No |
| ICLR 2027 (submissions) | OpenReview search | Thermal Sequence Reconstruction under Coupled Motion Blur and RS Readout; HaRP | No (both 2D restoration) |
| 3DV 2025 | ~142 rows of schedule CSV (WebFetch; approximate count) | None | No |
| 3DV 2026 | ~177 rows of schedule CSV (WebFetch; approximate count) | None by title. Minimal Solvers … Asynchronous Tracks is 3DV 2026 per OpenReview. Adjacent: MapAnything, "Recovering 3D Shapes from Ultra-Fast Motion-Blurred Images" | No |
| WACV 2026 (extra) | 831 CVF titles | None | No |

---

## 6. Citation scans (Semantic Scholar, fully paginated)

| Base model | S2 ID used | Citing papers scanned | With abstract | "rolling shutter" / "rolling-shutter" / "jello" hits | "readout" hits |
|---|---|---|---|---|---|
| VGGT | arXiv:2503.11651 (S2 citationCount = 2117) | **2,117** | 1,989 | **0** | 16, all ML-sense false positives (e.g. "lightweight readout") |
| DUSt3R | arXiv:2312.14132 | **2,103** | 1,970 | **0** | 13, all false positives |
| MASt3R | arXiv:2406.09756 | **1,293** | 1,218 | **0** | 7, all false positives |
| π³ | arXiv:2507.13347 | **387** | 378 | **0** | 4, all false positives |
| MapAnything | arXiv:2509.13414 | **399** | 384 | **0** | 3, all false positives |
| **Union** | | **3,386 unique** | | **0** | |

- The wider union filter (`shutter|scanline|per-row|intra-frame|exposure time|during exposure`) matched 6 papers. All were false positives: "intra-frame attention" (ReDT, Good Token Hunting, State-Update Gate, FastViDAR, YouTube-Occ), plus the KITScenes dataset, which uses global-shutter cameras.
- **Caveat:** 2–6% of citing papers per model have no abstract in S2 (VGGT 128, DUSt3R 133, MASt3R 75, π³ 9, MapAnything 15), and the filter cannot see body text. Foundation4D is an example of a paper whose RS content lives only in the body.
- A supplementary OpenAlex pass (`openalex_cites.py`) returned only the first page of 200 per model before 401 and timeout errors. It found 0 RS hits but is incomplete, so we do not rely on it.

---

## 7. What could not be verified, and caveats

- **arXiv index size.** The arXiv API and the arXiv website search both report only 150 RS papers in total and 40 since 2025. That seemed low to me, but two independent arXiv endpoints and OpenReview (152 forums for the phrase) agree. Treat the count as "what arXiv search returns today."
- **Point4Cast:** no arXiv version found; the CVF page returned 403 to WebFetch but was read with curl.
- **Anchor Points (UF):** no venue and no arXiv ID; venue is UNVERIFIED.
- **Shyamala report:** a course report, not peer reviewed. The PDF text was extracted crudely and only the opening pages were readable.
- **ICLR 2027 thermal RS submission:** the forum page was behind a browser challenge, so title, venue and abstract come from the OpenReview API search record.
- **XSIM (2602.05617):** metadata from the arXiv API only; abs page not opened.
- **Venue claims taken from arXiv comments, not from proceedings pages:**
  - EAG3R (NeurIPS 2025)
  - Fin3R (NeurIPS 2025)
  - Uni4R (NeurIPS 2026)
  - ProMo3D (NeurIPS 2026)
  - Event3R (IROS 2026)
  - DeblurSplat (TMM 2026)
  - Lv et al. (SIGGRAPH 2025)
  - 4D-GRT (NeurIPS 2025 workshop)
  - Unblur-SLAM (CVPR 2026, also seen in the CVF listing)
- **Image as an IMU:** the ICCV 2025 venue comes from the project page and GitHub README.
- **Code columns marked UNVERIFIED:** Fin3R, ProMo3D, TRGS-SLAM, Lv et al., XSIM and the UAV-SLAM benchmark were not checked in their repos.
  - Image-as-IMU weights: the README links a Google Drive folder, but whether it holds the full model is unclear.
  - EAG3R: weights come via a script and no license is stated.
- **Not covered:** journals (TPAMI, TIP, RA-L, IJCV) were not scanned. Google Scholar is not accessible.

---

## 8. Per-paper notes (details in `scoop.csv`, 37 rows)

### Threat: med

- **Foundation4D (2610.30340):** NVIDIA unified driving FM.
  - The pose for image (t, c) is computed from the predicted ego trajectory at that image's capture time, T_ego(τ_t,c), times the camera's rig transform.
  - Full text: depth labels are projected "accounting for camera motion during rolling-shutter exposure", and rendering poses are queried "at the rolling-shutter exposure boundaries".
  - It is the closest thing to an RS-aware FM, but RS stays outside the network's geometry. **Cite and contrast.**
- **Image as an IMU (2503.17358, ICCV 2025 Oral):** blur → flow + depth → least squares for ω,v. It explicitly does not model RS.
  - Code: MIT.
  - **A natural baseline:** run it per frame and compare its ω,v with ours.
- **Rolling Shutter Image as an IMU (Toronto report):** a single-RS-image version of the above. It is not a publication, but a reviewer could know it.
- **Interp3R (2603.14528):** event-driven pointmap interpolation between frames (DUSt3R/MonST3R/Align3R).
  - It is the closest "pointmap at arbitrary time" model.
  - Its v2 and v3 revisions (late September 2026) show the authors are still active.
- **Anchor Points (UF 2026):**
  - Marigold U-Net de-rolling followed by a monocular depth FM.
  - Code (Stable-Derolling) and weights released under Apache-2.0.
  - Their "anchor point" is our reference row time.

### Threat: low (useful baselines or context)

- Point4Cast (CVPR 2026)
- Uni4R (NeurIPS 2026)
- V-DPM (CVPR 2026): shows VGGT adapts cheaply to time-dependent pointmaps
- DynamicVGGT
- Unblur-SLAM (CVPR 2026): models sub-frame poses for blur
- RoMo-GS
- EAG3R, Event3R, EventVGGT (event sensors)
- DeblurSplat and AsyncEvGS (DUSt3R/VGGT used off the shelf for initialisation)
- ProMo3D: motion is decodable from frozen 3D FM tokens, which supports our light-head design
- Fin3R: LoRA on 3D FMs, a method-ingredient precedent
- Classical RS geometry:
  - Barath ECCV 2026: MIT code, TUM RS evaluation; use as a classical baseline
  - Zhang et al. ECCV 2026 self-calibration: relevant if readout time is unknown
  - Xue et al. ECCV 2026
  - Hruby & Pollefeys ICCV 2025 and 3DV 2026
  - Su et al. 2025
  - Errázuriz Muñoz et al. 2026
  - Hahn et al. CVPR 2025
- Per-scene RS reconstruction: RS-Avatar, Lv et al. SIGGRAPH 2025, TRGS-SLAM
- RS data synthesis: 4D-GRT, XSIM
- 2D RS correction: StableMotion, ICLR 2027 thermal submission
- Measurement: Verhoeven ISPRS 2026, UAV SLAM benchmark
- Classical learned RS:
  - Zhuang et al. CVPR 2019: cite for the pure-translation degeneracy result
  - Mo et al. 2020: IMU-assisted
