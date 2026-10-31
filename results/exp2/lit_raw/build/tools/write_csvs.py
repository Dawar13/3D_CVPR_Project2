import csv, os, sys

OUT = r"C:\Users\djtde\Downloads\3D_CVPR_Exp_2_Rolling_Shutter\results\exp2\lit_raw"
os.makedirs(OUT, exist_ok=True)

COLS = ["topic", "title", "authors", "venue", "year", "id", "url_opened", "verified",
        "what_it_does", "what_it_does_not_do", "code", "how_we_differ", "threat"]

P = []
def add(**k):
    assert set(k) == set(COLS), set(COLS) ^ set(k)
    P.append(k)

# ---------------- Topic 2: non-pinhole / camera-conditioned 3D foundation models ----------------
add(topic="2", title="RayTun3R: Online Camera Adaptation in 3D Foundation Models",
    authors="Daniil Sinitsyn, Nikita Araslanov, Daniel Cremers",
    venue="NeurIPS 2026 (poster; source: neurips.cc/virtual/2026/poster/151667; arXiv page has no venue comment)",
    year="2026", id="arXiv:2607.02711", url_opened="https://arxiv.org/abs/2607.02711 ; https://neurips.cc/virtual/2026/poster/151667",
    verified="Y",
    what_it_does="Keeps the 3D foundation model frozen (experiments: Depth Anything 3, VGGT, pi3) and learns a 10,752-parameter adapter: residual radial/angular lookup tables for absolute and rotary positional encodings plus parameter-free tokenization and prediction-grid corrections that remove pinhole assumptions. Fitted online on a short temporal segment with geometric losses, then reused for the rest of the sequence. Fisheye FoV 110-200 deg; reports 2-12x lower rotation error than the unadapted model and beats LoRA with ~14x fewer parameters.",
    what_it_does_not_do="No motion or time model: the camera geometry it corrects is fixed for the sequence. Needs a test-time optimisation segment per camera/sequence. No rolling shutter anywhere in the full text.",
    code="N (paper: 'Our code will be made publicly available'; no repo link on arXiv page)",
    how_we_differ="RS distortion changes every frame with angular/linear velocity, so a static positional-encoding correction cannot express it. We predict per-frame (omega, v) inside the model and unproject each row at its own time, trained once offline (LoRA), no per-sequence test-time optimisation. RayTun3R-style PE tuning and its 'beats LoRA' claim are a natural ablation/baseline for us.",
    threat="med")

add(topic="2", title="Fisheye3R: Adapting Unified 3D Feed-Forward Foundation Models to Fisheye Lenses",
    authors="Ruxiao Duan, Erin Hong, Dongxu Zhao, Eric Turner, Alex Wong, Yunwen Zhou",
    venue="ECCV 2026 (source: arXiv comments field 'European Conference on Computer Vision 2026')",
    year="2026", id="arXiv:2603.28896", url_opened="https://arxiv.org/abs/2603.28896 ; https://github.com/android-xr/fisheye3r",
    verified="Y",
    what_it_does="Frozen backbone plus K learnable calibration tokens inserted into encoder, frame-wise and global attention layers (all but the first L0 layers); a masked-attention scheme switches the tokens off for perspective images. Trained by synthesising fisheye distortion on perspective datasets, either supervised (no fisheye data needed) or self-supervised by distilling the model's own perspective predictions. Applied to VGGT, pi3 and MapAnything; gains in pose, depth, point map and FoV.",
    what_it_does_not_do="Static lens model per camera type; no temporal or motion modelling; no rolling shutter (full text checked). Limitation stated: only works for transformer architectures because it is token-based.",
    code="Y https://github.com/android-xr/fisheye3r (weights: N, Apache-2.0; README Implementation section says 'Coming soon')",
    how_we_differ="Its 'synthesise the distortion on perspective data' recipe is the same idea as synthesising RS from global-shutter frames plus motion, but our distortion is motion-dependent and we model it explicitly (velocities + row time + row-wise unprojection) instead of with generic tokens. Calibration tokens / masked attention are a reasonable adapter baseline next to LoRA.",
    threat="med")

add(topic="2", title="Wid3R: Wide Field-of-View 3D Reconstruction via Camera Model Conditioning",
    authors="Dongki Jung, Jaehoon Choi, Adil Qureshi, Somi Jeong, Dinesh Manocha, Suyong Yeon",
    venue="arXiv (GitHub repo title says '[ECCV 2026]'; the arXiv page shows no venue)",
    year="2026", id="arXiv:2602.05321", url_opened="https://arxiv.org/abs/2602.05321 ; https://jdk9405.github.io/Wid3R/ ; https://github.com/jdk9405/Wid3R",
    verified="Y",
    what_it_does="Multi-frame feed-forward reconstruction for fisheye and 360 images without calibration or undistortion. Ray representation with spherical harmonics plus a per-view camera-model token as an explicit conditioning input. Initialises the pi3 encoder, aggregator and pose head and trains everything end-to-end (full fine-tune); 518x336 inputs.",
    what_it_does_not_do="Needs the camera type of each view (discrete token); no motion or time; no rolling shutter.",
    code="Y https://github.com/jdk9405/Wid3R (weights: Y Google Drive checkpoint wid3r.bin, license not shown on page)",
    how_we_differ="Our conditioning is a continuous per-frame motion state inferred from the images, not a discrete camera-class token, and we use a light LoRA rather than full fine-tuning.",
    threat="low")

add(topic="2", title="PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery",
    authors="Yijing Guo, Mengjun Chao, Luo Wang, Tianyang Zhao, Haizhao Dai, Yingliang Zhang et al.",
    venue="CVPR 2026 (source: arXiv comments 'Accepted by CVPR 2026')",
    year="2026", id="arXiv:2603.17571", url_opened="https://arxiv.org/abs/2603.17571",
    verified="Y",
    what_it_does="Permutation-equivariant transformer predicting poses, depth and point clouds from one or more panoramas in one pass; spherical-aware positional embeddings, three-axis SO(3) rotation augmentation, stochastic anchoring. Trained on panoramas (8 A100, ~10 days, 336x672); also retrains pi3 on panoramas as a baseline; releases PanoCity dataset.",
    what_it_does_not_do="Panorama-only; a newly trained model rather than an adapter; no rolling shutter.",
    code="N (abstract: 'Code and dataset will be released')",
    how_we_differ="Fixed spherical projection learned by re-training; we adapt existing pinhole models to a time-varying (motion-dependent) projection with a small LoRA.",
    threat="low")

add(topic="2", title="CAM3R: Camera-Agnostic Model for 3D Reconstruction",
    authors="Namitha Guruprasad, Abhay Yadav, Cheng Peng, Rama Chellappa",
    venue="arXiv", year="2026", id="arXiv:2603.22631", url_opened="https://arxiv.org/abs/2603.22631",
    verified="Y",
    what_it_does="Two-view network: a Ray Module (ViT initialised from UniK3D) predicts per-pixel rays, a Cross-view Module (initialised from DUSt3R) predicts radial distance, confidence, pointmaps and relative pose; a Ray-Aware Global Alignment merges pairs. Pinhole, fisheye and panoramic inputs.",
    what_it_does_not_do="Pairwise plus optimisation (not single-pass multi-view); static rays per image; no rolling shutter.",
    code="N (no repo link on arXiv page)",
    how_we_differ="We stay single-pass multi-view and model per-row pose change, not a static per-pixel ray field.",
    threat="low")

add(topic="2", title="VGGT-360: Geometry-Consistent Zero-Shot Panoramic Depth Estimation",
    authors="Jiayi Yuan, Haobo Jiang, De Wen Soh, Na Zhao",
    venue="arXiv", year="2026", id="arXiv:2603.18943", url_opened="https://arxiv.org/abs/2603.18943",
    verified="Y",
    what_it_does="Training-free: slices a panorama into perspective views, reconstructs them with a VGGT-style model, and reprojects to panoramic depth; plug-in uncertainty-guided slicing, structure-saliency attention and correlation-weighted 3D correction.",
    what_it_does_not_do="No training or adaptation; panoramic depth only; no rolling shutter.",
    code="N (no repo link on arXiv page)",
    how_we_differ="A geometric wrapper around a frozen model, like our EXP1 'unroll then run' oracle; we move the correction inside the network and need no known motion.",
    threat="low")

add(topic="2", title="Many Eyes, One World: Feed-Forward 3D Reconstruction from Mixed Cameras (MEOW)",
    authors="Qiaoge Li, Yifan Zhan, Haijun Yang, Haiyang Liu, Yiyi Cai, Chenchi Luo",
    venue="arXiv", year="2026", id="arXiv:2609.35658", url_opened="https://arxiv.org/abs/2609.35658",
    verified="Y",
    what_it_does="Keeps the MapAnything backbone unchanged and adapts it with synthetic tuples from a procedural camera-manifold data engine (Lenscope) with exact rays/depth and covisibility checks; adds full-FoV resizing, an aspect-ratio embedding and a panorama flag. One pass over mixed perspective/fisheye/panorama views with no camera info supplied; reports 80.4 mAA@30 vs 54.3 for Wid3R on 2D3DS tuples.",
    what_it_does_not_do="No motion or time; no rolling shutter. Explicitly treats the problem as data adaptation rather than architecture change.",
    code="N (abstract: data engine, benchmark and evaluation pipeline 'will be released')",
    how_we_differ="Its stance ('just fine-tune on the right synthetic data') is the strongest counter-argument to an architectural RS change; we must include a 'fine-tune on synthetic RS only, no row-time unprojection' ablation to show the explicit motion/row-time model matters.",
    threat="low")

add(topic="2", title="RIGOR: Rig-Informed Geometry for Omnidirectional Reconstruction",
    authors="Tingjun Huang, Dmitry Rudshin, Mathieu Meyer, Pietro Bonazzi, Marc Pollefeys, Emilia Szymanska",
    venue="arXiv", year="2026", id="arXiv:2609.13504", url_opened="https://arxiv.org/abs/2609.13504 ; https://github.com/TangentH/RIGOR",
    verified="Y",
    what_it_does="Pipeline for gravity-aligned 360 video: treats each panorama as a four-view virtual rig fed to a frozen perspective backbone (Depth Anything 3 per the repo), repairs inconsistent predictions, verifies loop closures and fixes drift with a Sim(3) pose graph.",
    what_it_does_not_do="No fine-tuning (all learned weights frozen); no rolling shutter.",
    code="Y https://github.com/TangentH/RIGOR (weights: N in repo, scripts fetch upstream DA3/PanoVGGT weights; code Apache-2.0)",
    how_we_differ="Wrapper around a frozen model; we change the model's geometry for intra-frame motion.",
    threat="low")

add(topic="2", title="X-Lens: Real-Time Metric Depth Estimation with Heterogeneous Cameras",
    authors="Heng Zhou, Shuhong Liu, Yonghao He, Bohao Zhang, Fa Fu, Chenhui Hou et al.",
    venue="arXiv", year="2026", id="arXiv:2607.12993", url_opened="https://arxiv.org/abs/2607.12993",
    verified="Y",
    what_it_does="Compact (0.04B) model, up to 41 FPS, metric depth from a varying number of calibrated fisheye and pinhole views; learnable calibration tokens align fisheye and pinhole projection spaces and a distortion-based bias in cross-attention keeps cameras consistent.",
    what_it_does_not_do="Requires calibration; depth only (no pose); not an adapter on a pretrained 3D foundation model per the abstract; no rolling shutter.",
    code="N (no repo link on arXiv page)",
    how_we_differ="We need no calibration or intrinsics and keep a pretrained multi-view foundation model.",
    threat="low")

add(topic="2", title="UniK3D: Universal Camera Monocular 3D Estimation",
    authors="Luigi Piccinelli, Christos Sakaridis, Mattia Segu, Yung-Hsu Yang, Siyuan Li, Wim Abbeloos et al.",
    venue="CVPR 2025 (source: GitHub README; arXiv page shows no venue)",
    year="2025", id="arXiv:2503.16591", url_opened="https://arxiv.org/abs/2503.16591 ; https://github.com/lpiccinelli-eth/unik3d",
    verified="Y",
    what_it_does="Monocular metric 3D for any camera: spherical 3D representation and a camera module that models the ray bundle with spherical harmonics (pinhole, fisheye, panoramic).",
    what_it_does_not_do="Single image; no multi-view pose; no rolling shutter.",
    code="Y https://github.com/lpiccinelli-eth/unik3d (weights: Y HF lpiccinelli/unik3d-vits|vitb|vitl, CC BY-NC 4.0)",
    how_we_differ="Monocular static camera model; our problem is multi-view and motion-dependent.",
    threat="low")

add(topic="2", title="OmniRay: Camera-Agnostic Monocular 3D Reconstruction with Composite Ray Fields",
    authors="Namitha Guruprasad, Abhay Yadav, Cheng Peng, Rama Chellappa, Anand Bhattad",
    venue="arXiv", year="2026", id="arXiv:2610.31645", url_opened="https://arxiv.org/abs/2610.31645",
    verified="Y",
    what_it_does="Single-image model predicting per-pixel ray direction and relative depth with a layered (composite) ray representation for pinhole, fisheye and panoramic lenses; no camera parameters at inference.",
    what_it_does_not_do="Monocular; no pose; no rolling shutter.",
    code="N (project page https://nam1410.github.io/omniray/ only)",
    how_we_differ="Per-pixel rays could in principle absorb a static warp but not a multi-view per-row pose; we are multi-view.",
    threat="low")

add(topic="2", title="Pow3R: Empowering Unconstrained 3D Reconstruction with Camera and Scene Priors",
    authors="Wonbong Jang, Philippe Weinzaepfel, Vincent Leroy, Lourdes Agapito, Jerome Revaud",
    venue="CVPR 2025 (source: arXiv comments)", year="2025", id="arXiv:2503.17316",
    url_opened="https://arxiv.org/abs/2503.17316 ; https://github.com/naver/pow3r",
    verified="Y",
    what_it_does="DUSt3R-style network that accepts any subset of intrinsics, relative pose and dense/sparse depth through a lightweight conditioning mechanism; random modality subsets during training; enables native-resolution inference and point-cloud completion.",
    what_it_does_not_do="Priors must be supplied (pinhole intrinsics); pairwise (+BA for multi-view); no rolling shutter.",
    code="Y https://github.com/naver/pow3r (weights: Y Pow3R_ViTLarge_BaseDecoder_512_linear.pth, NAVER custom license; training code listed as TODO)",
    how_we_differ="We take no metadata; we infer the motion that RS needs from images. Pow3R is the canonical 'condition on camera metadata' precedent.",
    threat="low")

add(topic="2;12", title="MapAnything: Universal Feed-Forward Metric 3D Reconstruction",
    authors="Nikhil Keetha, Norman Muller, Johannes Schonberger, Lorenzo Porzi, Yuchen Zhang, Tobias Fischer et al.",
    venue="3DV 2026 (source: arXiv comments)", year="2025", id="arXiv:2509.13414",
    url_opened="https://arxiv.org/abs/2509.13414 ; https://github.com/facebookresearch/map-anything",
    verified="Y",
    what_it_does="Factored output (per-view depth, local ray maps, camera poses, global metric scale); optional inputs: intrinsics/rays, poses, depth, partial reconstructions; trained with random modality dropout so one model covers 12+ problem configurations. Benchmarks dense N-view (2-100 views) on ETH3D, ScanNet++ v2, TartanAirV2-WB.",
    what_it_does_not_do="One rigid pose per image; a per-pixel ray input can describe a static lens but not a per-row pose; no rolling shutter (full text checked).",
    code="Y https://github.com/facebookresearch/map-anything (weights: Y; facebook/map-anything CC-BY-NC-4.0, facebook/map-anything-apache Apache-2.0; code Apache-2.0)",
    how_we_differ="Our row-time-aware unprojection extends its factored representation with a per-row pose from predicted (omega, v). Also our frozen baseline and a fine-tuning framework.",
    threat="low")

add(topic="2", title="OmniVGGT: Omni-Modality Driven Visual Geometry Grounded Transformer",
    authors="Haosong Peng, Hao Li, Yalun Dai, Yushi Lan, Yihang Luo, Tianyu Qi et al.",
    venue="CVPR 2026 (source: cvpr.thecvf.com/virtual/2026/poster/39518; project page says 'CVPR 2026 Highlight')",
    year="2025", id="arXiv:2511.10560", url_opened="https://arxiv.org/abs/2511.10560 ; https://cvpr.thecvf.com/virtual/2026/poster/39518 ; https://livioni.github.io/OmniVGGT-official/",
    verified="Y",
    what_it_does="GeoAdapter encodes depth and camera intrinsics/extrinsics and injects them via zero-initialised convolutions; stochastic multimodal fusion samples a subset of modalities per training instance; initialised from VGGT and fine-tuned.",
    what_it_does_not_do="Gains need supplied priors; one pose per image; no rolling shutter.",
    code="Y https://github.com/Livioni/OmniVGGT-official (weights: Y HF Livioni/OmniVGGT, license UNVERIFIED)",
    how_we_differ="Injects known geometry; we inject a predicted motion state. Zero-init injection is a good recipe for our velocity tokens.",
    threat="low")

add(topic="2", title="Rig3R: Rig-Aware Conditioning for Learned 3D Reconstruction",
    authors="Samuel Li, Pujith Kachana, Prajwal Chidananda, Saurabh Nair, Yasutaka Furukawa, Matthew Brown",
    venue="NeurIPS 2025 Spotlight (source: neurips.cc/virtual/2025/poster/115429, listed there as 'Rig3R: Rig-Aware Conditioning and Discovery for 3D Reconstruction')",
    year="2025", id="arXiv:2506.02265", url_opened="https://arxiv.org/abs/2506.02265 ; https://neurips.cc/virtual/2025/poster/115429",
    verified="Y",
    what_it_does="Multi-view model with optional per-image metadata: camera ID, continuous timestamp (seconds) and rig-relative raymap, each independently droppable; predicts pointmaps plus global pose raymap and rig raymap, so rig structure can be discovered from images when metadata is missing.",
    what_it_does_not_do="Time is per frame, not per row; needs rig metadata for best results; perspective rigs; no rolling shutter.",
    code="N (no repo link on arXiv page)",
    how_we_differ="Closest 'time-conditioned' 3D foundation model; we need intra-frame (row) time and we predict the motion rather than consume timestamps.",
    threat="low")

add(topic="2", title="WorldMirror: Universal 3D World Reconstruction with Any-Prior Prompting",
    authors="Yifan Liu, Zhiyuan Min, Zhenwei Wang, Junta Wu, Tengfei Wang, Yixuan Yuan et al.",
    venue="ICML 2026 (source: arXiv comments)", year="2025", id="arXiv:2510.10726",
    url_opened="https://arxiv.org/abs/2510.10726 ; https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror",
    verified="Y",
    what_it_does="Multi-modal tokenisation: normalised intrinsics (fx,fy,cx,cy) -> MLP -> one token, pose -> one token, depth -> spatial tokens added to image tokens; any subset of priors at inference; initialised from VGGT; predicts points, depth, cameras, normals and 3D Gaussians.",
    what_it_does_not_do="Priors are given, not inferred; one pose per image; no rolling shutter.",
    code="Y https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror (weights: Y HF tencent/HunyuanWorld-Mirror, Tencent HunyuanWorld-Mirror Community License)",
    how_we_differ="Its 'one token per camera quantity' design is the template for a per-frame velocity token, but we predict that token from images.",
    threat="low")

add(topic="2", title="LoRA3D: Low-Rank Self-Calibration of 3D Geometric Foundation Models",
    authors="Ziqi Lu, Heng Yang, Danfei Xu, Boyi Li, Boris Ivanovic, Marco Pavone et al.",
    venue="ICLR 2025 (source: research.nvidia.com publication page; project page says Spotlight)",
    year="2024", id="arXiv:2412.07746", url_opened="https://arxiv.org/abs/2412.07746 ; https://research.nvidia.com/labs/aspire/publication/lu.yang.etal.iclr2025/ ; https://520xyxyzq.github.io/lora3d/",
    verified="Y",
    what_it_does="Per-scene self-calibration of DUSt3R/MASt3R with LoRA, using pseudo-labels from the model's own multi-view predictions refined by robust optimisation with recalibrated confidence; ~5 min on one GPU, 18 MB per adapter; tested on 160+ scenes from Replica, TUM and Waymo.",
    what_it_does_not_do="Per-scene test-time fine-tune; no camera-model or motion change; no rolling shutter.",
    code="N (NVIDIA page says code is on the project page, but the project page shows no code link)",
    how_we_differ="Evidence that LoRA is enough to adapt a 3D foundation model; we train one LoRA offline for RS across scenes and add an explicit motion/row-time model.",
    threat="low")

add(topic="2", title="UniScale: Unified Scale-Aware 3D Reconstruction for Multi-View Understanding via Prior Injection for Robotic Perception",
    authors="Mohammad Mahdavian, Gordon Tan, Binbin Xu, Yuan Ren, Dongfeng Bai, Bingbing Liu",
    venue="arXiv", year="2026", id="arXiv:2602.23224", url_opened="https://arxiv.org/abs/2602.23224",
    verified="Y",
    what_it_does="Builds on VGGT: adds a metric-scale head and semantic-aware injection of optional intrinsics and poses into the corresponding tokens.",
    what_it_does_not_do="Priors are given; no rolling shutter.",
    code="N (abstract: implementation released upon acceptance)",
    how_we_differ="Prior injection vs. our inferred motion state.",
    threat="low")

add(topic="2", title="Cameras as Relative Positional Encoding",
    authors="Ruilong Li, Brent Yi, Junchen Liu, Hang Gao, Yi Ma, Angjoo Kanazawa",
    venue="NeurIPS 2025 (source: arXiv journal-ref)", year="2025", id="arXiv:2507.10496",
    url_opened="https://arxiv.org/abs/2507.10496",
    verified="Y",
    what_it_does="Compares token-level raymap conditioning with attention-level relative pose encodings for multi-view transformers and proposes PRoPE, a relative positional encoding of the full camera frustum (intrinsics + extrinsics).",
    what_it_does_not_do="Needs known cameras; one frustum per image; no rolling shutter.",
    code="N (project page https://www.liruilong.cn/prope/ ; no repo link on arXiv page, not checked further)",
    how_we_differ="A per-row frustum (pose at each row time) would be the RS generalisation of PRoPE; design option for our row-time-aware attention.",
    threat="low")

# ---------------- Topic 12: evaluation protocols of strong 3D foundation models ----------------
add(topic="12", title="VGGT: Visual Geometry Grounded Transformer",
    authors="Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, David Novotny",
    venue="CVPR 2025 (source: arXiv comments)", year="2025", id="arXiv:2503.11651",
    url_opened="https://arxiv.org/abs/2503.11651 ; https://github.com/facebookresearch/vggt",
    verified="Y",
    what_it_does="1B feed-forward transformer (alternating frame/global attention) predicting cameras, depth, point maps and tracks. Eval: Co3Dv2 + RealEstate10K, 10 random frames, AUC@30 of min(RRA,RTA) over all pairs; DTU MVS (Acc/Comp/Overall, MASt3R protocol); ETH3D point maps, 10 random frames, Umeyama alignment, official masks, Acc/Comp/Overall.",
    what_it_does_not_do="One rigid pose per image; no rolling shutter in paper.",
    code="Y https://github.com/facebookresearch/vggt (weights: Y; facebook/VGGT-1B CC-BY-NC-4.0, facebook/VGGT-1B-Commercial gated VGGT license; code VGGT License)",
    how_we_differ="Frozen baseline and LoRA target.",
    threat="low")

add(topic="12", title="pi^3: Permutation-Equivariant Visual Geometry Learning",
    authors="Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li et al.",
    venue="ICLR 2026 (sources: GitHub repo description and mlanthology.org; OpenReview forum DTQIjngDta blocked by a browser check)",
    year="2025", id="arXiv:2507.13347", url_opened="https://arxiv.org/abs/2507.13347 ; https://mlanthology.org/iclr/2026/wang2026iclr-permutationequivariant/ ; https://github.com/yyfz/Pi3",
    verified="Y",
    what_it_does="Reference-free, permutation-equivariant model predicting affine-invariant poses and scale-invariant local point maps. Eval: RE10K + Co3Dv2 (10 random images, all pairs, RRA/RTA/AUC@30); Sintel, TUM-dynamics, ScanNet (ATE, RPE-t, RPE-r after Sim(3)); point maps on 7-Scenes/NRGBD (sparse+dense strides), DTU, ETH3D with Umeyama Sim(3) + ICP, Acc/Comp/NC mean+median; video depth (per-sequence scale) and mono depth (per-frame scale) AbsRel and delta<1.25 on Sintel, Bonn, KITTI (+NYU-v2).",
    what_it_does_not_do="One rigid pose per image; no rolling shutter in paper.",
    code="Y https://github.com/yyfz/Pi3 (weights: Y; GitHub README says Pi3 and Pi3X weights CC BY-NC 4.0, but HF card yyfz233/Pi3 is tagged bsd-2-clause; code BSD-3-Clause)",
    how_we_differ="Our main frozen baseline (EXP0/EXP1) and LoRA target.",
    threat="low")

add(topic="12", title="Depth Anything 3: Recovering the Visual Space from Any Views",
    authors="Haotong Lin, Sili Chen, Junhao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi et al.",
    venue="ICLR 2026 Oral (source: iclr.cc/virtual/2026/oral/10006532; arXiv page shows no venue)",
    year="2025", id="arXiv:2511.10647", url_opened="https://arxiv.org/abs/2511.10647 ; https://iclr.cc/virtual/2026/oral/10006532 ; https://github.com/ByteDance-Seed/Depth-Anything-3",
    verified="Y",
    what_it_does="Plain DINOv2 transformer predicting depth + ray maps for any number of views (optional pose conditioning); teacher-student training. Benchmark: HiRoom, ETH3D, DTU, 7Scenes, ScanNet++; all images or 100 random (fixed seed); pose AUC@3 and AUC@30; reconstruction with predicted poses: RANSAC-robust evo/Umeyama pose alignment, TSDF fusion, Acc, Comp, Chamfer, F1 at dataset thresholds (0.05 m HiRoom/7Scenes/ScanNet++, 0.25 ETH3D).",
    what_it_does_not_do="One rigid pose per image; no rolling shutter in paper.",
    code="Y https://github.com/ByteDance-Seed/Depth-Anything-3 (weights: Y; Giant/Large/Nested CC BY-NC 4.0, Base/Small/Metric/Mono Apache-2.0 per README; code Apache-2.0)",
    how_we_differ="Frozen baseline; also a candidate backbone (RayTun3R and RIGOR already use it).",
    threat="low")

add(topic="12", title="VGGT-Omega (VGGT-Ω)",
    authors="Jianyuan Wang, Minghao Chen, Shangzhan Zhang, Nikita Karaev, Johannes Schonberger, Patrick Labatut et al.",
    venue="CVPR 2026 Oral (source: arXiv comments)", year="2026", id="arXiv:2605.15195",
    url_opened="https://arxiv.org/abs/2605.15195 ; https://vggt-omega.github.io/ ; https://github.com/facebookresearch/vggt-omega",
    verified="Y",
    what_it_does="Scaled successor of VGGT: single dense multi-task head, register attention limiting inter-frame exchange, self-supervised training, DINOv3 init, 1B and 10B variants, ~30% of predecessor's training memory. Eval: 7 Scenes, NRGBD, ETH3D, DyCheck, Sintel, TUM-Dynamic; 10 random frames per scene; pose AUC@3/@30; depth AbsRel and delta1.25 (released eval code: per-frame median-based scale+shift alignment).",
    what_it_does_not_do="Its training-data VLM filter (Appendix A.3) hard-rejects clips with 'severe rolling shutter (wobble/jello effect)' and '360 equirectangular or heavily distorted fisheye without calibration', i.e. RS is removed from training rather than modelled.",
    code="Y https://github.com/facebookresearch/vggt-omega (weights: Y gated HF facebook/VGGT-Omega, FAIR Noncommercial Research License; code same license)",
    how_we_differ="Newest strong baseline; its explicit RS filtering supports our motivation that foundation models are not trained to handle RS.",
    threat="low")

add(topic="12", title="Continuous 3D Perception Model with Persistent State (CUT3R)",
    authors="Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A. Efros, Angjoo Kanazawa",
    venue="CVPR 2025 Oral (source: project page cut3r.github.io; arXiv page shows no venue; CVF page returned HTTP 403)",
    year="2025", id="arXiv:2501.12387", url_opened="https://arxiv.org/abs/2501.12387 ; https://cut3r.github.io/ ; https://github.com/CUT3R/CUT3R",
    verified="Y",
    what_it_does="Recurrent stateful model for online metric pointmaps from streams or photo collections. Eval: mono depth (KITTI, Sintel, Bonn, NYU-v2; per-frame median scaling); video depth (Sintel, Bonn, KITTI; per-sequence scale and metric); pose on Sintel, TUM-dynamics, ScanNet (ATE, RPE trans, RPE rot after Sim(3); repo uses 90-frame TUM/ScanNet clips); 7-Scenes (3-5 frames) and NRGBD (2-4 frames) Acc/Comp/NC.",
    what_it_does_not_do="One rigid pose per image; no rolling shutter in paper.",
    code="Y https://github.com/CUT3R/CUT3R (weights: Y Google Drive cut3r_224_linear_4.pth / cut3r_512_dpt_4_64.pth, no official HF; repo license CC BY-NC-SA 4.0)",
    how_we_differ="Streaming baseline; its pose protocol (Sim(3) ATE/RPE on TUM) matches our TUM RS-VIO trajectory evaluation.",
    threat="low")

add(topic="12", title="Streaming 4D Visual Geometry Transformer (StreamVGGT)",
    authors="Dong Zhuo, Wenzhao Zheng, Jiahe Guo, Yuqi Wu, Jie Zhou, Jiwen Lu",
    venue="ICLR 2026 poster as 'Streaming Visual Geometry Transformer' (source: iclr.cc/virtual/2026/poster/10011495)",
    year="2025", id="arXiv:2507.11539", url_opened="https://arxiv.org/abs/2507.11539 ; https://iclr.cc/virtual/2026/poster/10011495 ; https://github.com/wzzheng/StreamVGGT",
    verified="Y",
    what_it_does="Causal VGGT with temporal causal attention and cached memory tokens, distilled from VGGT (DistillLoss). Eval follows CUT3R/VGGT: 7-Scenes (3-5) and NRGBD (2-4 frames) Acc/Comp/NC, ETH3D following VGGT, video depth with per-sequence scale, Co3D pose script.",
    what_it_does_not_do="One rigid pose per image; no rolling shutter in paper.",
    code="Y https://github.com/wzzheng/StreamVGGT (weights: Y HF lch01/StreamVGGT tagged cc-by-nc-4.0; repo license CC BY-NC-SA 4.0)",
    how_we_differ="Streaming baseline only.",
    threat="low")

with open(os.path.join(OUT, "genre_models.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=COLS, quoting=csv.QUOTE_MINIMAL)
    w.writeheader()
    for r in P:
        w.writerow(r)
print("papers:", len(P))

# ---------------- models.csv ----------------
MC = ["model", "repo", "hf_id", "hf_sha", "code_license", "weights_license", "default_resolution", "preprocess",
      "bf16", "vram_note", "streaming", "finetune_code", "install", "verified", "notes"]
M = []
def m(**k):
    assert set(k) == set(MC), set(MC) ^ set(k)
    M.append(k)

PI3_PRE = ("pi3/utils/basic.py load_images_as_tensor(PIXEL_LIMIT=255000): scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)); k, m = round(W_target / 14), round(H_target / 14); "
           "while (k*14)*(m*14) > PIXEL_LIMIT: shrink k or m; img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS); no crop; target size from first image. 1280x1024 -> 560x448 (isotropic 0.4375).")
m(model="pi3 (Pi3, original)", repo="https://github.com/yyfz/Pi3", hf_id="yyfz233/Pi3", hf_sha="ae722e7039287d0c8fde9f11f197f804f44b510c",
  code_license="BSD-3-Clause", weights_license="CC BY-NC 4.0 per GitHub README (HF card tag says bsd-2-clause - conflict, treat as non-commercial)",
  default_resolution="~255k pixels, multiples of 14 (README gives no fixed size)", preprocess=PI3_PRE,
  bf16="Y (README: bfloat16 if compute capability >= 8 else float16, torch.amp.autocast)", vram_note="UNVERIFIED (not stated in README)",
  streaming="N (single batched call over an unordered set)",
  finetune_code="Y training branch (3 stages: low-res, high-res, conf); losses in pi3/models/loss.py: PointLoss (scale-aligned local pointmap L1 with 1/z weighting, normal loss, global points), CameraLoss (rotation angle + translation, alpha=100), confidence stage; also MapAnything train.md has pi3_finetuning.sh",
  install="git clone https://github.com/yyfz/Pi3.git; cd Pi3; pip install -r requirements.txt (torch==2.5.1, torchvision==0.20.1, numpy==1.26.4); Python/CUDA not stated",
  verified="Y", notes="Images-only inference: Y. Output camera_poses are camera-to-world OpenCV 4x4; conf are logits. HF last modified 2025-07-19.")

m(model="Pi3X (pi3 engineering update)", repo="https://github.com/yyfz/Pi3", hf_id="yyfz233/Pi3X", hf_sha="bb1deea4d7423de5b30691739cb451a3f57dc1d5",
  code_license="BSD-3-Clause", weights_license="CC BY-NC 4.0 (README and HF tag agree)",
  default_resolution="same loader as pi3 (PIXEL_LIMIT=255000); MapAnything wrapper lists 518", preprocess=PI3_PRE + " load_multimodal_data() adds optional pose/intrinsics/depth conditions (depth resized INTER_NEAREST).",
  bf16="Y (same README example)", vram_note="UNVERIFIED", streaming="N",
  finetune_code="Partial: training branch contains pi3/models/pi3x_dirty_training/pi3x_loss.py; README documents training for pi3 stages only",
  install="same as pi3; run python example_mm.py", verified="Y",
  notes="Released 2025-12-28: conv head, optional injection of poses/intrinsics/depth, continuous confidence, approximate metric scale. README calls it recommended. HF last modified 2026-03-11.")

VGGT_PRE = ("vggt/utils/load_fn.py load_and_preprocess_images(mode='crop'): target_size = 518; new_width = target_size; new_height = round(height * (new_width / width) / 14) * 14; "
            "img.resize((new_width, new_height), Image.Resampling.BICUBIC); if mode == 'crop' and new_height > target_size: center-crop height to 518. mode='pad': longest side 518, pad to 518x518 with white (1.0). 1280x1024 -> 518x420, no crop.")
m(model="VGGT-1B", repo="https://github.com/facebookresearch/vggt", hf_id="facebook/VGGT-1B", hf_sha="860abec7937da0a4c03c41d3c269c366e82abdf9",
  code_license="VGGT License v1 (2025-07-29; commercial use allowed excluding military; GitHub SPDX NOASSERTION)", weights_license="CC-BY-NC-4.0 (non-commercial; README: original checkpoint remains non-commercial)",
  default_resolution="518 wide (crop mode)", preprocess=VGGT_PRE,
  bf16="Y (README: bfloat16 if compute capability >= 8 else float16; torch.cuda.amp.autocast)",
  vram_note="Old README (commit 44b3afbd18, Oct 2025) aggregator on H100 w/ FlashAttention3: 1 fr 1.88 GB, 10 fr 3.63 GB, 20 fr 5.58 GB, 50 fr 11.41 GB, 100 fr 21.15 GB, 200 fr 40.63 GB; 2026-05-15 fix lets 2-3x more frames at same memory",
  streaming="N",
  finetune_code="Y training/ (released 2025-07-06): MultitaskLoss = camera (L1 on T, R, FL; weight 5) + depth (conf-weighted regression + gradient loss; weight 1), optional point and track losses; default.yaml freezes '*aggregator*', amp bfloat16, lr 5e-5",
  install="git clone ...vggt; pip install -r requirements.txt (torch==2.3.1, torchvision==0.18.1, numpy==1.26.1); docs/package.md: pip install torch==2.3.1 torchvision==0.18.1 --index-url .../cu121 then pip install -e .; requires-python >=3.10",
  verified="Y", notes="Images-only: Y. Evaluation code (Co3D) on 'evaluation' branch.")

m(model="VGGT-1B-Commercial", repo="https://github.com/facebookresearch/vggt", hf_id="facebook/VGGT-1B-Commercial", hf_sha="ebb29a532abe92960eeb6903a5530f16990ef4ab",
  code_license="VGGT License v1", weights_license="Commercial use allowed (HF license_name 'vggt-aup-license', gated manual approval form)",
  default_resolution="518", preprocess=VGGT_PRE, bf16="Y", vram_note="same architecture as VGGT-1B",
  streaming="N", finetune_code="same as VGGT-1B",
  install="same as VGGT-1B; HF access request needed", verified="Y",
  notes="README: 'only the newly released checkpoint VGGT-1B-Commercial is licensed for commercial usage'; Co3D AUC@30 90.37 vs 89.98 for original. Files: model.safetensors, vggt_1B_commercial.pt.")

MA_PRE = ("mapanything/utils/image.py load_images(resize_mode='fixed_mapping', resolution_set=518, norm_type='dinov2', patch_size=14): average aspect ratio -> closest key in RESOLUTION_MAPPINGS[518] "
          "{1.000:(518,518), 1.321:(518,392), 1.542:(518,336), 1.762:(518,294), ...}; crop_resize_if_necessary(): Lanczos rescale by max(target/size) then centered crop. "
          "1280x1024 (ar 1.25) -> resize 518x414 -> centre-crop to 518x392 (11 resized rows ~27 original rows cut top and bottom).")
m(model="MapAnything (CC-BY-NC)", repo="https://github.com/facebookresearch/map-anything", hf_id="facebook/map-anything", hf_sha="a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa",
  code_license="Apache-2.0", weights_license="CC-BY-NC 4.0 (13 training datasets)",
  default_resolution="518 longest side (fixed aspect mappings)", preprocess=MA_PRE,
  bf16="Y (model.infer(..., use_amp=True, amp_dtype='bf16'); falls back to fp16 if bf16 unsupported)",
  vram_note="memory_efficient_inference=True: 'up to 2000 views on 140 GB'; minibatch_size=1 smallest footprint; profiling on H200-140GB in paper",
  streaming="N", finetune_code="Y train.md: 13-dataset training, ablations, fine-tuning scripts for MoGe-2, VGGT, pi3 (README also lists VGGT-Omega); losses in configs/loss/ (log-space losses on ray depth, pointmaps, scale; scale-invariant translation)",
  install="git clone ...map-anything; conda create -n mapanything python=3.12 -y; pip install -e . (or '.[all]'); torch/CUDA not pinned; requires-python >=3.10",
  verified="Y", notes="Images-only: Y (optional intrinsics/poses/depth). Unified wrapper also runs vggt, vggt_omega, pi3, pi3x, da3, dust3r, mast3r, must3r, pow3r with listed resolutions. V1 (Sep 2025) checkpoints: facebook/map-anything-v1 (sha 515f332b34d0751c262da4a625327b6110db8f66).")

m(model="MapAnything (Apache-2.0)", repo="https://github.com/facebookresearch/map-anything", hf_id="facebook/map-anything-apache", hf_sha="00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a",
  code_license="Apache-2.0", weights_license="Apache-2.0 (trained on 6 datasets per paper supplement)",
  default_resolution="518", preprocess=MA_PRE, bf16="Y", vram_note="as above", streaming="N",
  finetune_code="as above (Apache-variant training scripts)", install="as above; demo scripts take --apache", verified="Y",
  notes="The only commercially permissive MapAnything checkpoint. V1 Apache: facebook/map-anything-apache-v1 (sha 59bcb2c461d5439939f91e5608877db5c58363eb). Paper: Apache variant competitive with VGGT but below the CC-BY-NC model.")

DA3_PRE = ("src/depth_anything_3/api.py inference(process_res=504, process_res_method='upper_bound_resize'); input_processor.py: _resize_longest_side: scale = target_size / float(longest); new_w = int(round(w * scale)); "
           "cv2.resize(..., INTER_CUBIC if scale > 1.0 else INTER_AREA); then _make_divisible_by_resize: each side to nearest multiple of 14 by a small resize ('*crop' methods floor by centre crop instead). "
           "Batch with mixed sizes is centre-cropped to the smallest. 1280x1024 -> 504x403 -> 504x406.")
DA3_FT = "N (repo has no training code; has bench/ evaluation code and da3_streaming/)"
DA3_INST = "pip install xformers torch>=2 torchvision; pip install -e . (optional gsplat for GS head; '.[app]' needs Python>=3.10); requires-python >=3.9,<=3.13"
m(model="Depth Anything 3 Nested Giant-Large 1.1", repo="https://github.com/ByteDance-Seed/Depth-Anything-3", hf_id="depth-anything/DA3NESTED-GIANT-LARGE-1.1", hf_sha="b2359bdf726fb44ef62acca04d629dcf158053e7",
  code_license="Apache-2.0", weights_license="CC BY-NC 4.0", default_resolution="504 longest side", preprocess=DA3_PRE,
  bf16="Y (api.py: torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, torch.autocast)",
  vram_note="UNVERIFIED for offline model; DA3-Streaming: long video in <12 GB", streaming="Y DA3-Streaming (sliding window, released 2025-12-11)",
  finetune_code=DA3_FT, install=DA3_INST, verified="Y",
  notes="1.40B params; pose est., pose conditioning, GS, metric depth, sky seg. -1.1 suffix = retrained after a training-bug fix, README says prefer these. Deprecated DA3NESTED-GIANT-LARGE sha 8615eefb62f2db4f8d6ebaa59160086981672829.")
m(model="Depth Anything 3 Giant 1.1", repo="https://github.com/ByteDance-Seed/Depth-Anything-3", hf_id="depth-anything/DA3-GIANT-1.1", hf_sha="72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19",
  code_license="Apache-2.0", weights_license="CC BY-NC 4.0", default_resolution="504", preprocess=DA3_PRE, bf16="Y", vram_note="UNVERIFIED", streaming="Y via DA3-Streaming",
  finetune_code=DA3_FT, install=DA3_INST, verified="Y", notes="1.15B; any-view (rel. depth, pose est., pose cond., GS). Used as 'DA3' in VGGT-Omega comparisons (Giant). Deprecated DA3-GIANT sha 7cd62ae9315b9dff094d2d300e4ad012640607dd.")
m(model="Depth Anything 3 Large 1.1", repo="https://github.com/ByteDance-Seed/Depth-Anything-3", hf_id="depth-anything/DA3-LARGE-1.1", hf_sha="0e109ae307c5982f319a67cf6f9f99ccdc0ec97c",
  code_license="Apache-2.0", weights_license="CONFLICT: GitHub README table says CC BY-NC 4.0, HF card tag says apache-2.0 - treat as non-commercial",
  default_resolution="504", preprocess=DA3_PRE, bf16="Y", vram_note="UNVERIFIED", streaming="Y via DA3-Streaming",
  finetune_code=DA3_FT, install=DA3_INST, verified="Y", notes="0.35B. Deprecated DA3-LARGE sha c54c26b16ec04d218e8d584ecf4bce082a9fcc20 (HF tag cc-by-nc-4.0).")
m(model="Depth Anything 3 Base", repo="https://github.com/ByteDance-Seed/Depth-Anything-3", hf_id="depth-anything/DA3-BASE", hf_sha="f4a6c9b3c95e41c82048423d3493a81ec3fa810e",
  code_license="Apache-2.0", weights_license="Apache-2.0 (README and HF agree)", default_resolution="504", preprocess=DA3_PRE, bf16="Y", vram_note="UNVERIFIED", streaming="Y via DA3-Streaming",
  finetune_code=DA3_FT, install=DA3_INST, verified="Y", notes="0.12B; largest commercially permissive any-view DA3.")
m(model="Depth Anything 3 Small", repo="https://github.com/ByteDance-Seed/Depth-Anything-3", hf_id="depth-anything/DA3-SMALL", hf_sha="e08cab65ca0ec38e7826075418411ab90cab4da3",
  code_license="Apache-2.0", weights_license="Apache-2.0", default_resolution="504", preprocess=DA3_PRE, bf16="Y", vram_note="UNVERIFIED", streaming="Y via DA3-Streaming",
  finetune_code=DA3_FT, install=DA3_INST, verified="Y", notes="0.08B. Monocular-only checkpoints DA3METRIC-LARGE (sha 4010e39f3634a45bc60553321fb49fb760bd594e) and DA3MONO-LARGE (sha f465978e618db8cc79c83b8bbf24964857db1875) are Apache-2.0 but have no pose head.")

m(model="VGGT-Omega 1B (VGGT-Ω)", repo="https://github.com/facebookresearch/vggt-omega", hf_id="facebook/VGGT-Omega (files: vggt_omega_1b_512.pt, vggt_omega_1b_416_reproduce.pt, vggt_omega_1b_256_text.pt)",
  hf_sha="1041e80fc0e911235d3426b0a3d9a81075111a53",
  code_license="FAIR Noncommercial Research License (LICENSE v1, 2024-10-16)", weights_license="FAIR Noncommercial Research License (HF license_name fair-noncommercial-research-license; gated, manual)",
  default_resolution="512 'balanced' (token count ~ (512/16)^2, patch 16); 416 for the benchmark-reproduction checkpoint",
  preprocess=("vggt_omega/utils/load_fn.py load_and_preprocess_images(mode='balanced', image_resolution=512, patch_size=16): first centre-crop aspect ratio into [0.5, 2.0]; "
              "token_number = (image_resolution // patch_size) ** 2; w_patches = np.sqrt(token_number / aspect_ratio); h_patches = token_number / w_patches; rounded; image.resize(..., BICUBIC); "
              "mode='max_size' resizes longest side to 512. 1280x1024 -> 576x464 (balanced) or 512x416 (max_size)."),
  bf16="UNVERIFIED for inference (README example uses torch.inference_mode() without autocast); training supports AMP/FSDP bf16",
  vram_note="README, A100, 624x416, end-to-end incl. weights: 1 fr 6.02 GB, 10 fr 6.67, 50 fr 9.66, 100 fr 13.37, 200 fr 20.82, 500 fr 43.15 GB",
  streaming="N (not mentioned)",
  finetune_code="Y training/ (released 2026-09-08; default config 128 GPUs): losses camera, depth, point, matching, consistency, gradient (training/losses/); paper advises full warm-up+cosine schedule and dropping the confidence loss when fine-tuning",
  install="git clone ...vggt-omega; pip install -r requirements.txt (torch>=2.6, torchvision>=0.21, numpy<2); pip install -e .; requires-python >=3.10",
  verified="Y", notes="Public code + gated weights exist (question answered: yes). Images-only: Y. Eval code for ETH3D and Sintel (AUC@3, AUC@30, delta125, AbsRel). 10B variant reported in paper; not in HF file list.")

m(model="CUT3R", repo="https://github.com/CUT3R/CUT3R", hf_id="none official (Google Drive: cut3r_224_linear_4.pth, cut3r_512_dpt_4_64.pth); third-party HF mirrors exist", hf_sha="n/a",
  code_license="CC BY-NC-SA 4.0 (LICENSE file)", weights_license="not separately stated (repo license CC BY-NC-SA 4.0 assumed - UNVERIFIED)",
  default_resolution="512 (demo --size 512; DPT checkpoint trained at 512x384/336/288/256/160 and portrait)",
  preprocess=("src/dust3r/utils/image.py load_images(size=512): _resize_pil_image(img, size) long edge -> 512 with int(round(x*512/S)); halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8; img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh)) "
              "(centre crop to multiples of 16; square inputs forced to 4:3). 1280x1024 -> 512x410 -> 512x400 (5 resized rows cut top and bottom)."),
  bf16="UNVERIFIED (not in README/demo)", vram_note="README: encoder runs frames in parallel -> memory linear in number of frames; no numbers",
  streaming="Y (recurrent state; inference_recurrent); README webcam online demo still TODO",
  finetune_code="Y docs/train.md (losses not described on README); 32 training datasets with preprocessing scripts",
  install="conda create -n cut3r python=3.11 cmake=3.14.0; conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia; pip install -r requirements.txt; conda install 'llvm-openmp<16'; build curope (cd src/croco/models/curope; python setup.py build_ext --inplace)",
  verified="Y", notes="Working public code + weights (repo last push 2025-08-27). Eval scripts: monodepth, video_depth, relpose (TUM/ScanNet 90-frame clips), mv_recon.")

m(model="StreamVGGT", repo="https://github.com/wzzheng/StreamVGGT", hf_id="lch01/StreamVGGT", hf_sha="f9ba55b1955bc4f34337c51142158a9aa2862c7f",
  code_license="CC BY-NC-SA 4.0 (LICENSE.txt)", weights_license="HF tag cc-by-nc-4.0",
  default_resolution="518 (VGGT loader)",
  preprocess="src/streamvggt/utils/load_fn.py load_and_preprocess_images(mode='crop'): identical to VGGT (width 518, height rounded to /14, centre-crop if >518). 1280x1024 -> 518x420.",
  bf16="UNVERIFIED (README mentions FlashAttention-2 inference code; dtype not stated)", vram_note="UNVERIFIED (cached memory tokens grow with sequence length)",
  streaming="Y (temporal causal attention + cached memory tokens)",
  finetune_code="Y train.py (config/train.yaml: DistillLoss with VGGT-1B teacher) and finetune.py for VGGT",
  install="conda create -n StreamVGGT python=3.11 cmake=3.14.0; pip install -r requirements.txt (torch==2.3.1, torchvision==0.18.1); conda install 'llvm-openmp<16'",
  verified="Y", notes="Working public code + HF checkpoint (checkpoints.pth, model.safetensors). Needs VGGT-1B teacher only for training. Repo last push 2025-10-27.")

with open(os.path.join(OUT, "models.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=MC, quoting=csv.QUOTE_MINIMAL)
    w.writeheader()
    for r in M:
        w.writerow(r)
print("models:", len(M))
