# Lit slice `genre_models`: non-pinhole adaptation (T2), models we run (b), eval protocols (T12)

Compiled 2026-10-29. Every paper below had its arXiv abstract page opened with WebFetch, and venue pages where noted. Full texts were read from arXiv HTML (curl, then keyword search). Model facts come from the official GitHub README or source files (raw.githubusercontent), the Hugging Face API (`sha`, license tag), and model cards. Per-paper rows are in `genre_models.csv`; per-checkpoint rows are in `models.csv`.

---

## 1. Findings in plain English

1. **No non-pinhole or camera-conditioning paper mentions rolling shutter.**
   - I searched the full text of all 19 Topic-2 papers for "rolling shutter" and found zero hits. The papers: RayTun3R, Fisheye3R, Wid3R, PanoVGGT, CAM3R, VGGT-360, MEOW, RIGOR, X-Lens, UniK3D, OmniRay, Pow3R, MapAnything, OmniVGGT, Rig3R, WorldMirror, LoRA3D, UniScale and PRoPE.
   - arXiv abstract searches pairing "rolling shutter" with VGGT, DUSt3R, MASt3R, pointmap, fisheye, "foundation model" or "feed-forward" returned only two rendering papers (2608.17682, 2509.10759).
   - The one mention I found in our genre is in a 3D foundation-model paper, not an adaptation paper. VGGT-Ω (arXiv 2605.15195, App. A.3, present in v1 and v2) uses a VLM to filter training videos. Its prompt marks as `REJECT_HARD` any clip with *"Severe motion blur (edges lost), severe rolling shutter (wobble/jello effect), or corruption/glitching"* and any *"360° equirectangular or heavily distorted fisheye without calibration"*. So the newest strong model removes RS from its training data rather than modelling it. This is a quotable motivation line.
2. **What reviewers will accept as "same genre": adapting a frozen or lightly tuned 3D foundation model to a non-pinhole camera.** The 2026 precedents:
   - **RayTun3R** (NeurIPS 2026): tunes about 10.7k positional-encoding parameters, online per sequence, and reports beating LoRA with ~14× fewer parameters.
   - **Fisheye3R** (ECCV 2026): calibration tokens plus masked attention on a frozen backbone, trained on synthetically distorted perspective data.
   - **Wid3R** (repo says ECCV 2026): a camera-model token, with π³ fully fine-tuned.
   - **MEOW** (arXiv): MapAnything fine-tuned on a synthetic camera-manifold data engine, with no architecture change.
   - Conditioning on metadata is established by Pow3R, MapAnything, OmniVGGT, WorldMirror, Pi3X, Rig3R (which even conditions on per-frame timestamps) and UniScale.
   - All of these model a **static** camera geometry or **given** metadata. None models intra-frame motion. Our per-frame (ω, v) and per-row unprojection is therefore new in this genre.
   - The two baselines reviewers will ask for:
     - (i) a static PE or token adapter (RayTun3R or Fisheye3R style) trained on RS;
     - (ii) "just fine-tune on synthetic RS", the MEOW stance.
3. **Running models frozen, images only.**
   - All seven model families accept images only: π³/Pi3X, VGGT-1B, MapAnything, DA3, VGGT-Ω, CUT3R and StreamVGGT.
   - Licenses:
     - Commercially permissive weights exist only for VGGT-1B-Commercial (gated), `facebook/map-anything-apache`, and DA3-BASE / DA3-SMALL (plus the monocular DA3 checkpoints).
     - Everything else is non-commercial: π³, Pi3X, VGGT-1B, MapAnything CC-BY-NC, DA3 Giant/Large/Nested, VGGT-Ω (FAIR NC), CUT3R and StreamVGGT (both CC BY-NC-SA code).
4. **VGGT-Ω is public.**
   - Code: `facebookresearch/vggt-omega`. Weights are gated on HF as `facebook/VGGT-Omega`, with checkpoints at 512 px, 416 px (benchmark reproduction) and 256 px (text-aligned).
   - Training code was released 2026-09-08, under the FAIR Noncommercial Research License.
   - It is the newest strong multi-view model released in 2026 (arXiv May 2026, CVPR 2026 Oral). I found no later paper that claims to beat it as a general model.
5. **CUT3R and StreamVGGT both have working public code and weights.**
   - CUT3R: weights on Google Drive only, no official HF repo.
   - StreamVGGT: HF repo `lch01/StreamVGGT`.
   - Both are CC BY-NC-SA 4.0 and both stream.
6. **Resizing interacts with row time.** For our 1280×1024 TUM RS-VIO frames (W×H per `src/common.py`), computed from each repo's own resize code:

| model | default input size | rows cropped? | effect on row→time map |
|---|---|---|---|
| π³ / Pi3X | 560×448 | no | isotropic ×0.4375, row r' ↔ original r = r'/0.4375 |
| VGGT / StreamVGGT (crop) | 518×420 | no (420 < 518) | x ×0.4047, y ×0.4102 (slightly anisotropic) |
| MapAnything (fixed_mapping 518) | 518×392 | **yes**: 11 resized rows (≈27 original) top and bottom | row offset must be added before computing row time |
| DA3 (upper_bound_resize 504) | 504×406 | no | small anisotropic resize (403→406) |
| VGGT-Ω (balanced 512) | 576×464 | no | x ×0.45, y ×0.453 |
| CUT3R (512) | 512×400 | **yes**: 5 resized rows (≈12.5 original) top and bottom | row offset needed |

Our row-time-aware unprojection has to carry each model's exact resize and crop. The cheapest safeguard is to pre-resize frames ourselves to a size that each loader passes through unchanged.

7. **Protocol choice for our paper (see §5).**
   - Match π³ and CUT3R for trajectories: ATE / RPE-t / RPE-r after Sim(3) on TUM-style sequences.
   - Match VGGT, π³, DA3 and VGGT-Ω for pose AUC:
     - all pairs from N sampled frames;
     - AUC of min(RRA, RTA);
     - report @30 for comparability and @3 or @5 for sensitivity, since RS errors are small angles.
   - Point maps: Umeyama Sim(3) (+ ICP, as in π³), then Acc / Comp (+ Chamfer).
   - Depth: AbsRel and δ<1.25 with per-frame median scaling (CUT3R, π³), or median scale+shift (VGGT-Ω eval code). State which one we use.

---

## 2. Query log

### WebSearch queries (all mode=standard)

| # | topic | query |
|---|---|---|
| 1 | T2 | Fisheye3R fisheye 3D foundation model 2026 arXiv |
| 2 | T2 | RayTun3R online camera adaptation 3D foundation models |
| 3 | T2 | VGGT fisheye panoramic 360 camera adaptation feed-forward reconstruction |
| 4 | T2 | DUSt3R generic camera model non-pinhole fine-tuning pointmap |
| 5 | T2 | camera model token intrinsics conditioning ray map multi-view transformer reconstruction 2026 |
| 6 | T2 | Pow3R empowering unconstrained 3D reconstruction camera and scene priors |
| 7 | T2 | omnidirectional equirectangular DUSt3R MASt3R adaptation 360 images pose depth |
| 8 | T2 | "rolling shutter" fisheye VGGT OR DUSt3R OR "foundation model" camera distortion adaptation |
| 9 | T2 | OmniVGGT omni-modality visual geometry grounded transformer auxiliary inputs intrinsics depth pose |
| 10 | T2 | Rig3R rig-aware conditioning multi-camera 3D reconstruction metadata |
| 11 | T2 | WorldMirror universal 3D reconstruction any-prior prompting camera intrinsics poses |
| 12 | T2 | LoRA adapter calibration tokens wide-angle lens multi-view geometry transformer pinhole bias positional encoding |
| 13 | T2 | Wid3R github code release wide field-of-view reconstruction NAVER LABS |
| 14 | (b) | VGGT-Omega arXiv 2026 visual geometry grounded transformer next step |
| 15 | (b) | new feed-forward multi-view 3D reconstruction foundation model 2026 outperforms VGGT pi3 open weights |
| 16 | (b) | 2026 state-of-the-art pose depth point map transformer released weights github "Depth Anything 3" comparison |
| 17 | (b) | streaming online 3D reconstruction transformer 2026 causal VGGT successor StreamVGGT CUT3R comparison |
| 18 | (b) | facebook/map-anything-apache Apache 2.0 checkpoint huggingface MapAnything license training datasets |
| 19 | (b) | Pi3X yyfz233 huggingface conditional injection metric scale release model license |
| 20 | (b) | feed-forward 3D reconstruction model 2026 release beats VGGT-Omega DA3 multi-view pose depth github weights |
| 21 | T12 | "Depth Anything 3" Recovering the Visual Space from Any Views accepted conference 2026 oral |
| 22 | T12 | openreview ICLR 2026 "Permutation-Equivariant Visual Geometry Learning" OR "Streaming 4D Visual Geometry Transformer" |
| 23 | T12 | iclr.cc virtual 2026 poster "Streaming Visual Geometry Transformer" StreamVGGT |
| 24 | T12 | VGGT evaluation protocol CO3D RealEstate10K 10 random frames AUC@30 RRA RTA |
| 25 | T12 | point map evaluation Umeyama ICP accuracy completeness normal consistency 7-Scenes NRGBD sparse dense views feed-forward |
| 26 | T12 | video depth evaluation per-sequence scale alignment AbsRel delta 1.25 Sintel Bonn KITTI MonST3R protocol ATE RPE TUM-dynamics |

### arXiv API queries
These were run against `export.arxiv.org/api/query`, sorted by submittedDate descending.

| query | result |
|---|---|
| `abs:fisheye AND abs:VGGT` | RayTun3R, Fisheye3R |
| `abs:fisheye AND abs:DUSt3R` | RayTun3R |
| `abs:panoram* AND abs:VGGT` | VGGT-360, Pano3DComposer, CineScene |
| `abs:omnidirectional AND abs:"feed-forward" AND abs:reconstruction` | RIGOR, GTF, Matrix-3D, OmniSplat |
| `abs:"camera model" AND abs:"feed-forward" AND abs:"pointmap"` | MEOW (2609.35658), CAM3R |
| `abs:fisheye AND abs:"foundation model" AND abs:pose` | RayTun3R, Fisheye3R (+1 unrelated) |
| `abs:"wide field of view" AND abs:"feed-forward"` | PanoImager, Fisheye3R, Wid3R (+1 unrelated) |
| `abs:panoram* AND abs:"feed-forward" AND abs:pose` | HoloRecon, MvFFN, heterogeneous-camera NVS, MEOW, RIGOR, ORBIT++, Argus, PanoImager, Holo360D, CAM3R, PanoVGGT, … |
| `abs:"camera-agnostic" AND abs:reconstruction` | OmniRay, CAM3R (+1 unrelated) |
| `abs:"generic camera" AND abs:"feed-forward"` | HTTP 429 twice; on retry returned 2 unrelated papers (Lyra 2.0, fast 3DGS) |
| `abs:"rolling shutter" AND abs:fisheye` | HTTP 429; superseded by the combined query below |
| `abs:"rolling shutter" AND (abs:VGGT OR abs:DUSt3R OR abs:MASt3R OR abs:pointmap OR abs:fisheye)` | 2 rendering papers only (2608.17682, 2509.10759) |
| `abs:"rolling shutter" AND abs:"foundation model"` | 0 results (after an earlier 429) |
| `abs:"rolling shutter" AND abs:"feed-forward"` | 0 results |
| `abs:"VGGT-Omega"` (successor check) | follow-ups only (ReDT, Spark3R, RoMa-Ω, …); none claims a general model beating VGGT-Ω |
| `id_list=2607.02711`; `id_list=2507.10496,2610.31645,2606.30047`; `ti:UniK3D` | metadata checks |

### Hugging Face API
`curl https://huggingface.co/api/models/<id>` was run for:
- yyfz233/Pi3, yyfz233/Pi3X;
- facebook/VGGT-1B, facebook/VGGT-1B-Commercial, facebook/VGGT-Omega;
- facebook/map-anything{,-apache,-v1,-apache-v1};
- depth-anything/DA3-{GIANT,GIANT-1.1,LARGE,LARGE-1.1,BASE,SMALL}, DA3NESTED-GIANT-LARGE{,-1.1}, DA3METRIC-LARGE, DA3MONO-LARGE;
- lch01/StreamVGGT.

Searches run: `models?search=cut3r` (no official repo) and `models?author=facebook&search=vggt`.

### GitHub sources read
- Repo pages (WebFetch): yyfz/Pi3, facebookresearch/vggt, facebookresearch/map-anything, ByteDance-Seed/Depth-Anything-3, facebookresearch/vggt-omega, CUT3R/CUT3R, wzzheng/StreamVGGT, android-xr/fisheye3r, TangentH/RIGOR, jdk9405/Wid3R, lpiccinelli-eth/unik3d, naver/pow3r, Tencent-Hunyuan/HunyuanWorld-Mirror.
- Raw source files: preprocessing loaders, requirements and pyproject files, LICENSE files, training loss and config files, and DA3, MapAnything and VGGT-Ω evaluation code.
- The VGGT README history (GitHub API) was used to recover the removed runtime/memory table, from commit 44b3afbd18 (2025-10-11).

---

## 3. Topic 2: per-paper notes (how each adapts the model)

| paper (venue, source of venue) | base model(s) | adaptation type | what is conditioned on | RS? | code / weights |
|---|---|---|---|---|---|
| **RayTun3R** 2607.02711 (NeurIPS 2026, neurips.cc poster page) | DA3, VGGT, π³ (frozen) | **tiny adapter** on positional encodings: 10,752 params, absolute + RoPE residual LUTs, plus parameter-free tokenization / grid fixes; **test-time online** on a short segment with geometric losses | nothing; learns the lens from data | no | N ("will be made publicly available") |
| **Fisheye3R** 2603.28896 (ECCV 2026, arXiv comments) | VGGT, π³, MapAnything (frozen) | **learnable calibration tokens** in encoder, frame and global layers, with **masked attention** (off for perspective images) | camera type, via the mask | no | repo Apache-2.0, implementation "Coming soon"; no weights |
| **Wid3R** 2602.05321 (arXiv; repo title says ECCV 2026) | π³ init | **full fine-tune** + camera-model token + spherical-harmonic ray representation | discrete camera-model token per view | no | code + Google Drive weights; license not shown |
| **PanoVGGT** 2603.17571 (CVPR 2026, arXiv comments) | DINOv2, new aggregator | **trained model** (≈10 days on 8×A100) | spherical PE (panorama) | no | "will be released" |
| **CAM3R** 2603.22631 (arXiv) | UniK3D + DUSt3R init | **full training** of a 2-view model + ray-aware global alignment | none (predicts rays) | no | N |
| **VGGT-360** 2603.18943 (arXiv) | VGGT-style (frozen) | **training-free** slicing/reprojection wrapper | n/a | no | N |
| **MEOW** 2609.35658 (arXiv) | MapAnything (unchanged architecture) | **fine-tune on synthetic camera manifold** + aspect-ratio embedding + panorama flag | none supplied | no | "will be released" |
| **RIGOR** 2609.13504 (arXiv) | DA3 (frozen) | **wrapper**: panorama → 4-view virtual rig + Sim(3) pose graph | rig geometry | no | code Apache-2.0; uses upstream weights |
| **X-Lens** 2607.12993 (arXiv) | new compact model | calibration tokens + distortion bias in attention (architecture, not an adapter) | calibrated cameras | no | N |
| **UniK3D** 2503.16591 (CVPR 2025, GitHub README) | monocular | trained model with SH ray-bundle camera module | none | no | code + HF weights CC BY-NC 4.0 |
| **OmniRay** 2610.31645 (arXiv) | monocular | trained model with composite ray fields | none | no | project page only |
| **Pow3R** 2503.17316 (CVPR 2025, arXiv comments) | DUSt3R-style | trained with **optional prior inputs** (random subsets) | intrinsics, rel. pose, depth | no | code + checkpoint, NAVER license; training code TODO |
| **MapAnything** 2509.13414 (3DV 2026, arXiv comments) | own | trained with **optional inputs** (modality dropout) | rays/intrinsics, poses, depth, metric scale | no | code Apache-2.0; NC and Apache weights |
| **OmniVGGT** 2511.10560 (CVPR 2026, CVPR virtual page) | VGGT init | **GeoAdapter (zero-init conv) + fine-tune** | depth, intrinsics, extrinsics | no | code + HF `Livioni/OmniVGGT` |
| **Rig3R** 2506.02265 (NeurIPS 2025 Spotlight, neurips.cc) | DUSt3R-style | trained with **optional metadata embeddings** (50% dropout) | camera ID, **timestamp (s)**, rig raymap | no | N |
| **WorldMirror** 2510.10726 (ICML 2026, arXiv comments) | VGGT init | **prior tokens** (1 token for intrinsics, 1 for pose; depth added spatially) + training | pose, intrinsics, depth | no | code + HF weights (Tencent community license) |
| **LoRA3D** 2412.07746 (ICLR 2025, NVIDIA page) | DUSt3R/MASt3R | **per-scene LoRA** self-calibration from pseudo-labels (5 min, 18 MB per adapter; Replica/TUM/Waymo) | none | no | no code link found |
| **UniScale** 2602.23224 (arXiv) | VGGT | prior injection + metric-scale head | intrinsics, poses | no | "released upon acceptance" |
| **PRoPE** 2507.10496 (NeurIPS 2025, arXiv journal-ref) | generic multi-view transformers | camera frustum as **relative positional encoding** | intrinsics + extrinsics | no | project page |
| *(Pi3X, no paper)* | π³ | optional injection of pose, intrinsics and depth (GitHub README, 2025-12-28) | pose, K, depth | no | HF `yyfz233/Pi3X` |
| *(DA3, see T12)* | DA3 | pose-conditioning column ✅ in the model-zoo table | poses | no | — |

Takeaways for "same genre":
- PEFT on a frozen backbone (RayTun3R, Fisheye3R) and light fine-tunes with an added conditioning token (Wid3R, OmniVGGT, WorldMirror) are both accepted at top venues in 2025–26. A **LoRA plus a small motion head** is squarely in genre.
- RayTun3R's claim to beat LoRA is something we should either reproduce (a PE-only adapter on RS) or answer with an ablation.
- VGGT-Ω §5 says auxiliary inputs hurt when added during pre-training but help when added *only during fine-tuning*. This supports conditioning at the LoRA stage.

---

## 4. Part (b): models we run frozen (+ LoRA for π³ and VGGT)

All of these run on images only (no intrinsics, poses or depth needed). The SHAs are the HF `sha` fields fetched on 2026-10-29.

| model | repo | HF id @ sha | code license | weights license | default res. (our 1280×1024 →) | bf16 | VRAM note | streaming | train / fine-tune code |
|---|---|---|---|---|---|---|---|---|---|
| π³ | github.com/yyfz/Pi3 | `yyfz233/Pi3` @ ae722e7039287d0c8fde9f11f197f804f44b510c | BSD-3-Clause | **CC BY-NC 4.0** (README); HF card says bsd-2-clause, a **conflict** | ≤255k px, /14 (560×448) | Y (autocast bf16 if CC≥8) | not stated | N | Y, `training` branch: PointLoss (scale-aligned local pts, 1/z weights, normals, global pts) + CameraLoss + conf stage |
| Pi3X | same | `yyfz233/Pi3X` @ bb1deea4d7423de5b30691739cb451a3f57dc1d5 | BSD-3-Clause | CC BY-NC 4.0 | same loader (560×448) | Y | not stated | N | partial (`pi3x_dirty_training/pi3x_loss.py`) |
| VGGT-1B | github.com/facebookresearch/vggt | `facebook/VGGT-1B` @ 860abec7937da0a4c03c41d3c269c366e82abdf9 | VGGT License (commercial OK, no military) | **CC-BY-NC-4.0** | width 518, crop (518×420) | Y | H100 aggregator (old README): 10 fr 3.63 GB, 50 fr 11.41 GB, 100 fr 21.15 GB, 200 fr 40.63 GB; May 2026 fix gives 2–3× more frames | N | Y `training/`: camera (L1 T/R/FL, w=5) + depth (conf-weighted + gradient), optional point/track; default freezes aggregator, AMP bf16 |
| VGGT-1B-Commercial | same | `facebook/VGGT-1B-Commercial` @ ebb29a532abe92960eeb6903a5530f16990ef4ab | VGGT License | **commercial** (gated; `vggt-aup-license`) | 518 | Y | same | N | same |
| MapAnything | github.com/facebookresearch/map-anything | `facebook/map-anything` @ a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa | Apache-2.0 | CC-BY-NC 4.0 | 518 fixed-aspect map (518×392, centre crop) | Y (`use_amp=True, amp_dtype="bf16"`) | mem-efficient mode: "up to 2000 views on 140 GB" | N | Y `train.md` (+ fine-tune scripts for VGGT, π³, MoGe-2) |
| MapAnything-Apache | same | `facebook/map-anything-apache` @ 00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a | Apache-2.0 | **Apache-2.0** | same | Y | same | N | same |
| DA3 Nested G-L 1.1 | github.com/ByteDance-Seed/Depth-Anything-3 | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` @ b2359bdf726fb44ef62acca04d629dcf158053e7 | Apache-2.0 | CC BY-NC 4.0 | 504 longest side, /14 (504×406) | Y (bf16 if supported) | not stated; DA3-Streaming <12 GB | **Y** DA3-Streaming | **N** (eval/bench only) |
| DA3 Giant 1.1 | same | `depth-anything/DA3-GIANT-1.1` @ 72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19 | Apache-2.0 | CC BY-NC 4.0 | 504 | Y | — | Y | N |
| DA3 Large 1.1 | same | `depth-anything/DA3-LARGE-1.1` @ 0e109ae307c5982f319a67cf6f9f99ccdc0ec97c | Apache-2.0 | README CC BY-NC 4.0 vs HF apache-2.0: **conflict** | 504 | Y | — | Y | N |
| DA3 Base / Small | same | `DA3-BASE` @ f4a6c9b3c95e41c82048423d3493a81ec3fa810e; `DA3-SMALL` @ e08cab65ca0ec38e7826075418411ab90cab4da3 | Apache-2.0 | **Apache-2.0** | 504 | Y | — | Y | N |
| VGGT-Ω 1B | github.com/facebookresearch/vggt-omega | `facebook/VGGT-Omega` @ 1041e80fc0e911235d3426b0a3d9a81075111a53 (files `vggt_omega_1b_512.pt`, `_416_reproduce.pt`, `_256_text.pt`) | FAIR Noncommercial | FAIR Noncommercial (gated) | 512 balanced, patch 16 (576×464) | inference: not shown; training: AMP/FSDP bf16 | A100, 624×416, end-to-end: 10 fr 6.67 GB, 100 fr 13.37 GB, 200 fr 20.82 GB, 500 fr 43.15 GB | N | Y `training/` (2026-09-08): camera, depth, point, matching, consistency, gradient losses |
| CUT3R | github.com/CUT3R/CUT3R | none official (Google Drive `cut3r_512_dpt_4_64.pth`, `cut3r_224_linear_4.pth`) | CC BY-NC-SA 4.0 | not separately stated | 512 long edge, crop to /16 (512×400) | not stated | linear in #frames | **Y** recurrent | Y (`docs/train.md`) |
| StreamVGGT | github.com/wzzheng/StreamVGGT | `lch01/StreamVGGT` @ f9ba55b1955bc4f34337c51142158a9aa2862c7f | CC BY-NC-SA 4.0 | HF tag cc-by-nc-4.0 | 518 (VGGT loader, 518×420) | not stated | not stated (cache grows) | **Y** causal + memory tokens | Y: distillation from VGGT-1B (`DistillLoss`) + `finetune.py` |

**Install, as given in each README**
- **π³**:
  - `git clone https://github.com/yyfz/Pi3.git && cd Pi3 && pip install -r requirements.txt`
  - requirements: torch==2.5.1, torchvision==0.20.1, numpy==1.26.4
  - Python/CUDA not stated
- **VGGT**:
  - `pip install -r requirements.txt` (torch==2.3.1, torchvision==0.18.1)
  - or, per docs/package.md, `pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121` then `pip install -e .`
  - Python ≥3.10
- **MapAnything**:
  - `conda create -n mapanything python=3.12 -y`, then `pip install -e .` (or `".[all]"`)
  - torch/CUDA not pinned; Python ≥3.10
  - extras `.[vggt-omega]` and `.[depth-anything-3]` exist for the external-model wrappers
- **DA3**:
  - `pip install xformers torch>=2 torchvision`, then `pip install -e .`
  - Python ≥3.9 and ≤3.13
- **VGGT-Ω**:
  - `pip install -r requirements.txt` (torch>=2.6, torchvision>=0.21, numpy<2), then `pip install -e .`
  - Python ≥3.10
- **CUT3R**:
  - `conda create -n cut3r python=3.11 cmake=3.14.0`
  - `conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia`
  - `pip install -r requirements.txt`
  - `conda install 'llvm-openmp<16'`
  - build curope: `cd src/croco/models/curope && python setup.py build_ext --inplace`
- **StreamVGGT**:
  - `conda create -n StreamVGGT python=3.11 cmake=3.14.0`
  - `pip install -r requirements.txt` (torch==2.3.1)
  - `conda install 'llvm-openmp<16'`

**Preprocessing code, copied from the repos**

VGGT `vggt/utils/load_fn.py`, `load_and_preprocess_images(image_path_list, mode="crop")`, `target_size = 518`:
```python
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]
```
StreamVGGT `src/streamvggt/utils/load_fn.py` has the same function and the same 518 crop default.

π³ `pi3/utils/basic.py`, `load_images_as_tensor(path, interval=1, PIXEL_LIMIT=255000)`:
```python
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target:
            k -= 1
        else:
            m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    ...
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
```

MapAnything `mapanything/utils/image.py`, `load_images(folder_or_list, resize_mode="fixed_mapping", ..., norm_type="dinov2", patch_size=14, ..., resolution_set=518, ...)`:
```python
    if resize_mode == "fixed_mapping":
        # Resolution mappings are already compatible with their respective patch sizes
        # 518 mappings are divisible by 14, 512 mappings are divisible by 16
        target_width, target_height = find_closest_aspect_ratio(
            average_aspect_ratio, resolution_set
        )
```
The 518 mapping is `{1.000: (518, 518), 1.321: (518, 392), 1.542: (518, 336), 1.762: (518, 294), 2.056: (518, 252), 3.083: (518, 168), …}`. After that, `crop_resize_if_necessary` applies a Lanczos rescale by `max(output_resolution / image.size)` and then a centred crop (`left = (w - target_w) // 2; top = (h - target_h) // 2`).

DA3 `src/depth_anything_3/api.py`: `inference(..., process_res: int = 504, process_res_method: str = "upper_bound_resize", ...)`. In `utils/io/input_processor.py`, `_resize_longest_side` does `scale = target_size / float(longest); new_w = max(1, int(round(w * scale)))`, then `_make_divisible_by_resize` rounds each side to the nearest multiple of 14. The `*crop` methods floor each side by a centre crop instead.

VGGT-Ω `vggt_omega/utils/load_fn.py`, `load_and_preprocess_images(image_path_list, mode="balanced", image_resolution=512, patch_size=16)`:
- `_crop_to_supported_aspect_ratio(image, min_aspect_ratio=0.5, max_aspect_ratio=2.0)` centre-crops only extreme aspect ratios;
- balanced mode uses `token_number = (image_resolution // patch_size) ** 2; w_patches = np.sqrt(token_number / aspect_ratio); h_patches = token_number / w_patches`, then a BICUBIC resize.

CUT3R `src/dust3r/utils/image.py`, `load_images(folder_or_list, size)`:
- `_resize_pil_image(img, size)` sets the long edge with `int(round(x * long_edge_size / S))`;
- then `halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8; img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))`.

Notes for our LoRA plan:
- VGGT's own fine-tuning config freezes the aggregator by default, and it has no built-in LoRA. LoRA on π³ or VGGT must be added by us, e.g. with `peft`. Neither repo ships LoRA code. This is consistent with what I saw; I did not run a dedicated search for third-party LoRA forks.
- MapAnything already wraps fine-tuning for VGGT and π³ with its multi-dataset loader and losses. It is the lowest-effort shared training harness.
- VGGT-Ω's advice for fine-tuning: use a full warm-up plus cosine schedule, keep the peak learning rate small, and drop the confidence loss at first.

---

## 5. Part (c), Topic 12: evaluation protocols

| paper | datasets | frames per test | pose metrics | geometry metrics | alignment before scoring |
|---|---|---|---|---|---|
| **VGGT** (CVPR 2025) | Co3Dv2, RealEstate10K (pose); DTU (MVS depth); ETH3D (point map); ScanNet-1500 matching (not used here) | **10 random frames** per scene (pose and ETH3D) | RRA, RTA per pair; **AUC@30** of min(RRA, RTA) | DTU and ETH3D: Accuracy, Completeness, Overall (= Chamfer) | ETH3D: **Umeyama** Sim(3) to GT, invalid points removed with official masks; DTU: MASt3R protocol |
| **π³** (ICLR 2026) | RE10K, Co3Dv2; Sintel, TUM-dynamics, ScanNet; 7-Scenes, NRGBD (sparse and dense), DTU, ETH3D; video depth on Sintel, Bonn, KITTI; mono depth adds NYU-v2 | pose AUC: **10 random images**, all pairs; trajectories follow CUT3R; 7-Scenes/NRGBD at two sampling strides (values not given in the text I read) | RRA@30, RTA@30, **AUC@30**; **ATE, RPE-t, RPE-r** | Acc, Comp, NC (mean and median); AbsRel, δ<1.25 | trajectories: **Sim(3)**; point maps: **Umeyama Sim(3) + ICP**; video depth: **per-sequence scale**; mono depth: **per-frame scale** |
| **MapAnything** (3DV 2026) | ETH3D (undistorted), ScanNet++ v2, TartanAirV2-WB; also two-view, single-view calibration and RobustMVD benchmarks | **2–100 views**, randomly sampled to form one covisibility-connected component, constant seed | **ATE RMSE**, **AUC@5**, ray-direction angular error | pointmap/depth **rel** (AbsRel) and **τ** = inlier ratio at 1.03% | pred and GT each normalised by mean point distance (`norm_mode="avg_dis"`) in view-0 frame; ATE uses Horn rigid alignment on the normalised trajectories; metric scale scored separately (`metric_scale_abs_rel`) |
| **DA3** (ICLR 2026 Oral) | HiRoom, ETH3D (11 scenes), DTU (DTU-64 pose; DTU-49 recon, 22 scans), 7Scenes (every 11th frame), ScanNet++ (20 scenes, every 5th frame) | **all images, or 100 random** (fixed seed) | **AUC@3, AUC@30** (min RRA/RTA curve) | Acc, Comp, **Chamfer**, **F1@d** (d = 0.05 m for HiRoom, 7Scenes, ScanNet++; 0.25 for ETH3D); DTU "Overall" in mm | **RANSAC over evo/Umeyama** pose alignment (inliers = translation error below the median), then TSDF fusion of predicted depths |
| **VGGT-Ω** (CVPR 2026 Oral) | static: 7 Scenes, NRGBD, ETH3D; dynamic: DyCheck, Sintel, TUM-Dynamic | **10 random frames** per scene | **AUC@3, AUC@30** | depth **AbsRel, δ1.25** | depth: per-frame **median-based scale + shift** (released `eval/metrics.py`); pose AUC is alignment-free (relative pairs) |
| **CUT3R** (CVPR 2025 Oral) | mono depth: KITTI, Sintel, Bonn, NYU-v2; video depth: Sintel, Bonn, KITTI; pose: Sintel, TUM-dynamics, ScanNet; recon: 7-Scenes, NRGBD | pose: following MonST3R, the repo uses **90-frame** TUM and ScanNet clips (`rgb_90`, `color_90`) and a fixed list of Sintel sequences; recon: **3–5 frames** (7-Scenes), **2–4** (NRGBD) | **ATE, RPE trans, RPE rot** | AbsRel, δ<1.25; Acc, Comp, NC | pose: **Sim(3)**; mono depth: **per-frame median**; video depth: **per-sequence scale**, plus unaligned metric |
| **StreamVGGT** (ICLR 2026) | 7-Scenes, NRGBD, ETH3D, Sintel, Bonn, KITTI, Co3D | 3–5 / 2–4 frames (CUT3R); ETH3D follows VGGT | Co3D pose script (AUC) | Acc, Comp, NC; AbsRel, δ<1.25 | CUT3R/VGGT conventions; video depth per-sequence scale |

**Suggested protocol for TUM RS-VIO, matching the above:**
- **Pose**:
  - (a) trajectory ATE RMSE plus RPE-t/RPE-r after Sim(3) Umeyama, as in π³ and CUT3R. This matches their TUM-dynamics numbers and our EXP0.
  - (b) all-pairs RRA/RTA and AUC@30 (VGGT, π³) together with AUC@3 (DA3, VGGT-Ω). RS errors are small, so @3 (or MapAnything's @5) is the sensitive one.
  - Use the same N frames for every model.
  - Frame counts: 10 random frames (VGGT, π³, VGGT-Ω), up to 100 (DA3), or 90-frame contiguous clips (CUT3R TUM). I recommend both a 10-frame sparse setting and a contiguous-clip setting at our velocity bins.
- **Depth**: AbsRel and δ<1.25 with per-frame median scaling (CUT3R, π³ mono) or per-sequence scale (video depth). Report which one.
- **Point map**:
  - Acc, Comp and Chamfer after Umeyama Sim(3) (+ ICP, as in π³), as long as we have a GT point cloud;
  - otherwise depth plus pose only.

---

## 6. Could not verify / caveats

- **Venues not visible on the arXiv page.** Taken from secondary official pages instead:
  - RayTun3R: NeurIPS page.
  - π³: GitHub plus mlanthology. The OpenReview forum `DTQIjngDta` returned a browser check.
  - StreamVGGT: ICLR virtual page. OpenReview `5APgTKsnx8` was blocked.
  - DA3: ICLR page.
  - CUT3R: project page. The CVF open-access page returned HTTP 403.
  - OmniVGGT: CVPR virtual page.
  - Rig3R: NeurIPS page.
  - LoRA3D: NVIDIA page.
  - UniK3D: GitHub README.
  - Wid3R: the ECCV 2026 claim is only in its GitHub repo title. Treated as **UNVERIFIED**; the CSV gives the venue as arXiv.
- **License conflicts.**
  - π³ original weights: GitHub README says CC BY-NC 4.0, HF tag says bsd-2-clause.
  - DA3-LARGE-1.1: README says CC BY-NC 4.0, HF tag says apache-2.0.
  - Both are treated as non-commercial.
- **Licenses not confirmed.**
  - CUT3R weights: the repo license is CC BY-NC-SA 4.0, but the weights' own license is not stated.
  - OmniVGGT and Wid3R: licenses not shown.
- **VRAM not stated** for π³/Pi3X, DA3 offline models, CUT3R and StreamVGGT.
- **bf16 for inference not confirmed** for VGGT-Ω, CUT3R and StreamVGGT.
- **VGGT's current README no longer has the runtime/memory table.** The numbers above come from commit 44b3afbd18 (Oct 2025) and predate the May 2026 memory fix.
- **Code for RayTun3R, PanoVGGT, MEOW, CAM3R, VGGT-360, X-Lens, Rig3R, UniScale and LoRA3D** was not found. The papers say "will be released", or no link appears.
- **Minor facts left unconfirmed:**
  - Fisheye3R's weights: none released.
  - X-Lens's "FishRoPE / Jacobian distortion bias" names appear only in a search snippet. The CSV uses the abstract wording.
- **π³ protocol gaps:** the sparse/dense strides for 7-Scenes and NRGBD are not stated in the text I read. A secondary paper (DAGE, 2603.03744) reports strides of 200/500 (sparse) and 40/100 (dense), which is **UNVERIFIED** for π³ itself.
- **arXiv API rate limit:** several queries hit HTTP 429 and had to be retried. One combined rolling-shutter query replaced two that failed.
