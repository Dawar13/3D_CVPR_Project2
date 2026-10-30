"""Writes ../scoop.csv with Python's csv module. Every row was opened (WebFetch / curl / API) in this session;
url_opened is the page actually read. Run: python -I make_csv.py"""
import csv, os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scoop.csv')
COLS = ['topic','title','authors','venue','year','id','url_opened','verified','what_it_does','what_it_does_not_do','code','how_we_differ','threat']
R = []
def add(**k): R.append(k)

# ---------------- nearest / most threatening ----------------
add(topic='scoop:adjacent-intra-frame-motion', title='Image as an IMU: Estimating Camera Motion from a Single Motion-Blurred Image',
    authors='Jerred Chen, Ronald Clark', venue='ICCV 2025 (Oral) [from project page + GitHub README; arXiv comments field lists only the project page]', year='2025',
    id='arXiv:2503.17358', url_opened='https://arxiv.org/abs/2503.17358 ; https://jerredchen.github.io/image-as-imu/ ; https://arxiv.org/html/2503.17358',
    verified='Y',
    what_it_does='From ONE motion-blurred image predicts a dense motion-flow field + monocular depth, then solves a small-motion linear least-squares for instantaneous camera angular+translational velocity (IMU-like); trained on synthetic blur from ScanNet++v2; compared with MASt3R and COLMAP.',
    what_it_does_not_do='Paper text (Sec. 4.2): "we do not model for rolling shutter or non-uniform motion within exposure". Single image; not inside a multi-view pointmap FM; velocity comes from blur, not row timing; needs exposure time; 180-degree direction ambiguity.',
    code='Y https://github.com/jerredchen/image-as-an-imu (weights: Y? README model table has a Google Drive "Download" link - UNVERIFIED whether it is the full model; MIT)',
    how_we_differ='We get omega,v from RS row-time geometry across multiple views inside a 3D FM and use them to unproject the pointmap per row; they use blur in one image, explicitly ignore RS, and output velocity only (no multi-view reconstruction).',
    threat='med')
add(topic='scoop:adjacent-RS-velocity-learned', title='Rolling Shutter Image as an IMU: Estimating Camera Motion from a Single Rolling Shutter Image',
    authors='Prashanth Shyamala', venue='University of Toronto CSC2529 course project report (not peer reviewed; no arXiv found)', year='2025',
    id='none', url_opened='https://www.cs.toronto.edu/~lindell/teaching/2529/past_projects/2025/report/prashanth-shyamala.pdf',
    verified='PARTIAL (PDF downloaded via curl and opening pages read from raw text; WebFetch failed on size; no venue/ID exists)',
    what_it_does='Extends Image-as-an-IMU to a single RS image: predicts a per-row RS-to-GS flow field + monocular depth, then solves a linear least-squares for per-row instantaneous camera angular/translational velocity given the sensor readout time.',
    what_it_does_not_do='Single image only; two-stage (flow+depth then LS), not inside a multi-view 3D FM; needs readout time; no multi-view pose/pointmap; course report, not a publication.',
    code='N (no code link found in report text or by search)',
    how_we_differ='Same images-only RS-velocity intuition, but we embed it in a multi-view FM (omega,v per frame, per-row unprojection, LoRA) and evaluate multi-view pose; theirs is single-view and post-hoc.',
    threat='med')
add(topic='scoop:RS+depth-foundation-model', title='Anchor Points for Monocular Depth Estimation Under Severe Rolling Shutter Ego-motion',
    authors='Jacob Carter, Vaishnav Ramesh, Md Jahidul Islam, Sanjeev J. Koppal', venue='UNVERIFIED (lab page lists it under 2026; no venue or arXiv given)', year='2026',
    id='none found', url_opened='https://focus.ece.ufl.edu/research/2026/rolling-shutter-anchor-point/ ; https://focus.ece.ufl.edu/research/2026/rolling-shutter-anchor-point/paper.pdf',
    verified='PARTIAL (project page + PDF opened; no arXiv ID or venue stated)',
    what_it_does='Two-stage pipeline: a Marigold U-Net fine-tuned to map RS images to GS frames at a chosen "anchor" row time, then a pretrained monocular depth foundation model; new RS dataset with motion distortion; legged-robot demo (<400 ms).',
    what_it_does_not_do='Monocular depth only; no camera pose or velocity estimation; no multi-view FM; correction is image-space preprocessing; trains on RS-GS paired data.',
    code='Y https://github.com/JacobACarter/Stable-Derolling (weights: Y, Dropbox checkpoint linked in README; Apache-2.0 per LICENSE.txt)',
    how_we_differ='We do not undistort images; we put the RS motion model inside the multi-view 3D FM and output per-row geometry + poses. Their anchor-time choice is analogous to our reference row time (cite).',
    threat='med')
add(topic='scoop:adjacent-time-conditioned-pointmap', title='Interp3R: Continuous-time 3D Geometry Estimation with Frames and Events',
    authors='Shuang Guo, Filbert Febryanto, Lei Sun, Luc Van Gool, Guillermo Gallego', venue='arXiv', year='2026',
    id='arXiv:2603.14528', url_opened='https://arxiv.org/abs/2603.14528 ; https://arxiv.org/html/2603.14528',
    verified='Y',
    what_it_does='Extends frame-based pointmap models (DUSt3R/MonST3R/Align3R) to estimate depth and camera pose at arbitrary time instants between frames by interpolating pointmaps with a co-captured event stream; trained only on synthetic data; DSEC/EDS results.',
    what_it_does_not_do='Requires an event camera; time is inter-frame (between frame timestamps), not intra-frame row time; full-text HTML has zero mentions of rolling shutter; no camera-velocity head.',
    code='N (no repo on arXiv page; none found by search)',
    how_we_differ='Images only, no events; we model time inside each frame (row readout) with a per-frame omega,v, so one image yields many poses.',
    threat='med')
add(topic='scoop:FM-with-RS-in-pipeline', title='Foundation4D: Towards a unified perception and reconstruction model for autonomous vehicles',
    authors='Sergio Agostinho, Maximilian Apodaca, Emmanuel Attia, Guillem Braso, George Dong, Ilya Fedorov, et al. (NVIDIA)', venue='arXiv', year='2026',
    id='arXiv:2610.30340', url_opened='https://arxiv.org/abs/2610.30340 ; https://arxiv.org/html/2610.30340',
    verified='Y',
    what_it_does='Single feed-forward multi-camera driving model (geometry, 3D detection, occupancy, Gaussian reconstruction); predicts rig calibration and a time-varying ego trajectory T_ego(tau), each image posed at its own capture time; full text: GT depth labels are projected "accounting for camera motion during rolling-shutter exposure", and for Gaussian rendering target poses are "queried from the predicted rig trajectory at each sensor timestamp and at the rolling-shutter exposure boundaries".',
    what_it_does_not_do='RS appears only in label generation and downstream rendering; the network still predicts one pose per image and a global-shutter pointmap/depth (no per-row unprojection inside the model); driving rig with known intrinsics/extrinsics; abstract does not mention RS.',
    code='N (no repo link on arXiv page)',
    how_we_differ='Closest FM that touches RS at all: it has a continuous ego trajectory but does not use row time when predicting geometry; we put row time into the pointmap itself, per frame, from images only, for generic handheld cameras. Must cite and contrast.',
    threat='med')
add(topic='scoop:adjacent-time-conditioned-pointmap', title='Point4Cast: Streaming Dynamic Scene Reconstruction and Forecasting',
    authors='Xinhang Liu, Pedro Miraldo, Suhas Lohit, Huaizu Jiang, Naoko Sawada, Yu-Wing Tai, et al.', venue='CVPR 2026 (CVF open access, pp. 14600-14611)', year='2026',
    id='none found (no arXiv); MERL TR2026-077', url_opened='https://openaccess.thecvf.com/content/CVPR2026/html/Liu_Point4Cast_Streaming_Dynamic_Scene_Reconstruction_and_Forecasting_CVPR_2026_paper.html (via curl; WebFetch got 403) ; https://www.merl.com/research/highlights/point4cast',
    verified='Y',
    what_it_does='Streaming latent spacetime state updated per frame; a readout predicts time-conditioned pointmaps and camera parameters for any queried time (past, present, future); scene flow for free.',
    what_it_does_not_do='Queried times are frame-level instants for dynamic scenes; no per-row pose; RS not mentioned in the abstract (full PDF text not checked).',
    code='N (CVF abstract: "code will be released"; MERL page has no repo link)',
    how_we_differ='We condition geometry on row time inside a frame to absorb camera motion during readout; they condition on frame time to model scene dynamics.',
    threat='low')
add(topic='scoop:adjacent-velocity-field-4D', title='Marrying Optimal Transport and ODEs for Unified Continuous-Time 4D Reconstruction and Tracking',
    authors='Liying Yang, Hao Mo, Jialun Liu, Chen Liu, Xinxing Yu, Chenhao Guan, et al.', venue='NeurIPS 2026 main track (arXiv comments field)', year='2026',
    id='arXiv:2608.09613', url_opened='https://arxiv.org/abs/2608.09613 ; https://arxiv.org/html/2608.09613',
    verified='Y',
    what_it_does='Uni4R: learns continuous velocity fields and decodes velocities at arbitrary timestamps for 4D reconstruction and point tracking with pointmap outputs.',
    what_it_does_not_do='Velocities are of scene points across frames, not of the camera within a frame; no rolling shutter (0 mentions in full text).',
    code='N (no repo link on arXiv page)',
    how_we_differ='We predict a rigid camera twist per frame and apply it per row; they model non-rigid scene motion over time.',
    threat='low')
add(topic='scoop:RS-learned-classic', title='Learning Structure-And-Motion-Aware Rolling Shutter Correction',
    authors='Bingbing Zhuang, Quoc-Huy Tran, Pan Ji, Loong-Fah Cheong, Manmohan Chandraker', venue='CVPR 2019', year='2019',
    id='CVF CVPR 2019 pp. 4551-4560 (DOI not recorded)', url_opened='https://openaccess.thecvf.com/content_CVPR_2019/html/Zhuang_Learning_Structure-And-Motion-Aware_Rolling_Shutter_Correction_CVPR_2019_paper.html (via curl; WebFetch got 403)',
    verified='Y',
    what_it_does='CNN learns camera motion and scene structure (depth) from a single RS image and uses them to correct the RS image; proves RS two-view geometry is degenerate under pure translation.',
    what_it_does_not_do='Single image, 2D correction output; pre-foundation-model CNN; no multi-view pointmap reconstruction or pose output.',
    code='N (not checked; pre-2025 background)',
    how_we_differ='Earliest images-only "predict RS motion with a network" prior; we do it inside a multi-view 3D FM and output geometry directly. Their pure-translation degeneracy is relevant to our omega-v collinearity issue.',
    threat='low')
add(topic='scoop:RS-learned-classic', title='IMU-Assisted Learning of Single-View Rolling Shutter Correction',
    authors='Jiawei Mo, Md Jahidul Islam, Junaed Sattar', venue='arXiv (OpenReview/dblp record lists CoRL 2021; arXiv page shows no venue)', year='2020',
    id='arXiv:2011.03106', url_opened='https://arxiv.org/abs/2011.03106',
    verified='Y',
    what_it_does='Network predicts depth and row-wise pose from a single RS image for RS correction; refines poses with IMU data.',
    what_it_does_not_do='Single view; uses IMU for pose refinement; no multi-view 3D FM.',
    code='N (no link on arXiv page)',
    how_we_differ='We use multi-view constraints inside a 3D FM to make images-only RS motion estimation work without an IMU.',
    threat='low')

# ---------------- RS geometry 2025-2026 (non-learned) ----------------
add(topic='scoop:RS-geometry', title='Rolling Shutter Relative Pose Estimation Made Practical', authors='Daniel Barath',
    venue='ECCV 2026 (ECCV 2026 accepted-papers list; repo README citation)', year='2026', id='arXiv:2606.26863',
    url_opened='https://arxiv.org/abs/2606.26863 ; https://eccv.ecva.net/Conferences/2026/AcceptedPapers ; https://github.com/danini/rolling_shutter_made_practical',
    verified='Y', what_it_does='RS-corrected affine constraints; 7-correspondence linearized RS relative-pose solver (~1.2 ms) in RANSAC; evaluated on TUM RS and EuRoC.',
    what_it_does_not_do='Geometric two-view solver; no learning, no dense geometry, no foundation model.',
    code='Y https://github.com/danini/rolling_shutter_made_practical (weights: N/A solver; MIT)',
    how_we_differ='Classical RS-aware baseline on the same TUM RS data family; we are learned, dense and multi-view. Good baseline to report.', threat='low')
add(topic='scoop:RS-geometry', title='Rolling Shutter Camera Self-Calibration',
    authors='Yongcong Zhang, Navid Rabbani, Bangyan Liao, Chengbo Wang, Yizhen Lao, Adrien Bartoli', venue='ECCV 2026 (ECCV 2026 accepted-papers list; arXiv has no comments)', year='2026',
    id='arXiv:2608.01509', url_opened='https://arxiv.org/abs/2608.01509 ; https://eccv.ecva.net/Conferences/2026/AcceptedPapers', verified='Y',
    what_it_does='First target-free self-calibration of intrinsics + readout-time ratio from image sequences via self-calibrating BA (continuous-time trajectory + dual projection model).',
    what_it_does_not_do='Optimisation-based; no learned/feed-forward model.', code='N (no repo link on arXiv page)',
    how_we_differ='Relevant if we must estimate readout time instead of assuming it; we assume known (or learnable) readout inside a feed-forward FM.', threat='low')
add(topic='scoop:RS-geometry', title='Estimating Velocity and Spin of Spherical Objects from Rolling-Shutter Image(s)',
    authors='Wenjie Xue, Jun Yang, Jingmin Wang, Limin Shang', venue='ECCV 2026 (ECCV 2026 accepted-papers list; arXiv has no comments)', year='2026',
    id='arXiv:2606.31760', url_opened='https://arxiv.org/abs/2606.31760 ; https://eccv.ecva.net/Conferences/2026/AcceptedPapers', verified='Y',
    what_it_does='Uses RS distortion as a temporal signal to recover 3D translational and angular velocity of a patterned sphere from a single RS frame (geometric back-projection).',
    what_it_does_not_do='Object (not camera) motion; needs a designed pattern; no learning; no scene reconstruction.', code='N (no repo link on arXiv page)',
    how_we_differ='Same "RS as a velocity signal" idea, but for camera ego-motion inside a learned multi-view model.', threat='low')
add(topic='scoop:RS-geometry', title='Single-View Rolling-Shutter SfM',
    authors='Sofia Errazuriz Munoz, Kim Kiehn, Petr Hruby, Kathlen Kohn', venue='arXiv', year='2026', id='arXiv:2603.11888',
    url_opened='https://arxiv.org/abs/2603.11888', verified='Y',
    what_it_does='Characterises single-view RS geometry of points/lines; which motion and scene parameters are recoverable from one RS image; minimal problems + proof-of-concept solvers.',
    what_it_does_not_do='Theory/minimal solvers; no learning; no FM.', code='N (no repo link on arXiv page)',
    how_we_differ='Theory supporting that a single RS image carries motion information; we learn it.', threat='low')
add(topic='scoop:RS-geometry', title='Single-Scanline Relative Pose Estimation for Rolling Shutter Cameras',
    authors='Petr Hruby, Marc Pollefeys', venue='ICCV 2025 (arXiv comments; CVF ICCV 2025 listing)', year='2025', id='arXiv:2506.22069',
    url_opened='https://arxiv.org/abs/2506.22069 ; https://openaccess.thecvf.com/ICCV2025?day=all', verified='Y',
    what_it_does='Relative pose between RS cameras from intersections of line projections with a single scanline per image, without a motion model.',
    what_it_does_not_do='Geometric; no learning; sparse lines.', code='N (arXiv: "will be made publicly available"; no link)',
    how_we_differ='Per-scanline geometry without learning; we are dense, learned, multi-view.', threat='low')
add(topic='scoop:RS-geometry', title='Minimal Solvers for Full DoF Motion Estimation from Asynchronous Tracks',
    authors='Petr Hruby, Marc Pollefeys', venue='3DV 2026 (OpenReview venue field "3DV 2026 Poster"; arXiv comments give only page count)', year='2025',
    id='arXiv:2508.17537', url_opened='https://arxiv.org/abs/2508.17537 ; OpenReview API search record (forum rjSgM65aTh)', verified='Y',
    what_it_does='Minimal solvers for camera translational + angular velocity from asynchronous point tracks (RS and event cameras) via polynomial approximation.',
    what_it_does_not_do='Sparse tracks, geometric; no learning; no dense reconstruction.', code='N (arXiv: "will be made publicly available"; no link)',
    how_we_differ='Same unknowns (omega,v) but solved geometrically from tracks; we regress them inside a FM.', threat='low')
add(topic='scoop:RS-geometry', title='A Linear N-Point Solver for Structure and Motion from Asynchronous Tracks',
    authors='Hang Su, Yunlong Feng, Daniel Gehrig, Panfeng Jiang, Ling Gao, Xavier Lagorce, et al.', venue='arXiv', year='2025', id='arXiv:2507.22733',
    url_opened='https://arxiv.org/abs/2507.22733 ; https://github.com/suhang99/AsyncTrack-Motion-Solver', verified='Y',
    what_it_does='Linear solver recovering 3D points and constant linear velocity from 2D correspondences captured at arbitrary times (GS, RS or event cameras).',
    what_it_does_not_do='Linear velocity model; geometric; no FM.',
    code='Y https://github.com/suhang99/AsyncTrack-Motion-Solver (weights: N/A solver; Apache-2.0)',
    how_we_differ='Geometric baseline for time-stamped correspondences; we are learned and dense.', threat='low')
add(topic='scoop:RS-geometry', title='Order-One Rolling Shutter Cameras',
    authors='Marvin Anas Hahn, Kathlen Kohn, Orlando Marigliano, Tomas Pajdla', venue='CVPR 2025 (CVF CVPR 2025 listing)', year='2025', id='arXiv:2403.11295',
    url_opened='https://arxiv.org/abs/2403.11295 ; https://openaccess.thecvf.com/CVPR2025?day=all', verified='Y',
    what_it_does='Unified algebraic theory of order-one RS cameras (each 3D point maps to exactly one image point): modelling, parameterisation, relative pose.',
    what_it_does_not_do='Theory; no learning.', code='N (no link on arXiv page)', how_we_differ='Background theory only.', threat='low')

# ---------------- RS in reconstruction / rendering (per-scene) ----------------
add(topic='scoop:RS-per-scene-recon', title='Scanline-Aware Animatable Gaussian Avatars from Rolling-Shutter Videos', authors='Youxiang Wang',
    venue='arXiv', year='2026', id='arXiv:2608.17314', url_opened='https://arxiv.org/abs/2608.17314 ; https://arxiv.org/html/2608.17314', verified='Y',
    what_it_does='RS-Avatar: per-subject optimisation of an animatable 3DGS avatar that renders sub-frame body states and composites them scanline by scanline; RS-ZJU benchmark; readout from calibration.',
    what_it_does_not_do='Per-scene optimisation on a calibrated static multi-view rig; models human (not camera) motion; no feed-forward FM.', code='N (no repo link on arXiv page)',
    how_we_differ='Same "pose per scanline" image formation, but for camera ego-motion and inside a feed-forward FM.', threat='low')
add(topic='scoop:RS-per-scene-recon', title='Photoreal Scene Reconstruction from an Egocentric Device',
    authors='Zhaoyang Lv, Maurizio Monge, Ka Chen, Yufeng Zhu, Michael Goesele, Jakob Engel, et al.', venue='SIGGRAPH 2025 Conference Paper (arXiv comments)', year='2025',
    id='arXiv:2506.04444', url_opened='https://arxiv.org/abs/2506.04444', verified='Y',
    what_it_does='Uses visual-inertial BA to get precise RS timestamps and high-rate motion, then a physical RS image-formation model inside Gaussian Splatting (Project Aria).',
    what_it_does_not_do='Needs IMU/VIO; per-scene optimisation; no feed-forward FM.', code='UNVERIFIED (project page http://www.projectaria.com/photoreal-reconstruction/ not opened)',
    how_we_differ='Images only, feed-forward.', threat='low')
add(topic='scoop:RS-per-scene-recon', title='TRGS-SLAM: IMU-Aided Gaussian Splatting SLAM for Blurry, Rolling Shutter, and Noisy Thermal Images',
    authors='Spencer Carmichael, Katherine A. Skinner', venue='arXiv', year='2026', id='arXiv:2603.20443', url_opened='https://arxiv.org/abs/2603.20443', verified='Y',
    what_it_does='Thermal-inertial 3DGS SLAM with B-spline trajectory optimisation and model-aware rendering for blur/RS/noise.',
    what_it_does_not_do='IMU-aided, per-scene optimisation; thermal; no FM.', code='UNVERIFIED (project page https://umautobots.github.io/trgs_slam not opened)',
    how_we_differ='No IMU; feed-forward.', threat='low')
add(topic='scoop:RS-data-synthesis', title='Every Camera Effect, Every Time, All at Once: 4D Gaussian Ray Tracing for Physics-based Camera Effect Data Generation',
    authors='Yi-Ruei Liu, You-Zhe Xie, Yu-Hsiang Hsu, I-Sheng Fang, Yu-Lun Liu, Jun-Cheng Chen', venue='NeurIPS 2025 Workshop SpaVLE (arXiv comments)', year='2025',
    id='arXiv:2509.10759', url_opened='https://arxiv.org/abs/2509.10759', verified='Y',
    what_it_does='4D-GRT: 4DGS reconstruction + physically based ray tracing to render videos with controllable camera effects (RS named as a target effect) as training data.',
    what_it_does_not_do='Data generation only; abs page does not confirm which 4 effects the benchmark covers; no RS-aware 3D FM.', code='N (project page https://shigon255.github.io/4DGRT-project-page/ ; no repo on arXiv page)',
    how_we_differ='Possible RS training-data source; not a competing method.', threat='low')
add(topic='scoop:RS-data-synthesis', title='Unified Sensor Simulation for Autonomous Driving',
    authors='Nikolay Patakin, Arsenii Shirokov, Anton Konushin, Dmitry Senushkin', venue='arXiv', year='2026', id='arXiv:2602.05617',
    url_opened='http://export.arxiv.org/api/query (record 2602.05617)', verified='PARTIAL (arXiv API metadata only; abs page not opened with WebFetch)',
    what_it_does='XSIM: extends 3DGUT splatting with generalised rolling-shutter modelling for camera and LiDAR simulation in driving.',
    what_it_does_not_do='Rendering/simulation; no feed-forward pose/pointmap model.', code='UNVERIFIED (not checked)',
    how_we_differ='Possible RS data source.', threat='low')
add(topic='scoop:RS-2D-correction', title='StableMotion: One-Step Motion Estimation with Diffusion Prior',
    authors='Ziyi Wang, Haipeng Li, Lin Sui, Tianhao Zhou, Hai Jiang, Lang Nie, et al.', venue='arXiv', year='2025', id='arXiv:2505.06668',
    url_opened='https://arxiv.org/abs/2505.06668 ; https://github.com/ivowang/StableMotion', verified='Y',
    what_it_does='Repurposes Stable Diffusion as a one-step image-to-motion estimator for single-image rectification tasks incl. RS correction.',
    what_it_does_not_do='2D image correction with a 2D foundation model; no 3D/pose.', code='Y https://github.com/ivowang/StableMotion (weights: Y, Hugging Face; MIT)',
    how_we_differ='Image-space correction baseline (correct-then-reconstruct), the pipeline we argue against.', threat='low')
add(topic='scoop:RS-2D-correction', title='Thermal Sequence Reconstruction under Coupled Motion Blur and Rolling-Shutter Readout',
    authors='Anonymous (ICLR 2027 submission)', venue='ICLR 2027 Conference Submission (OpenReview, under review)', year='2026',
    id='OpenReview:BMYxcibovq', url_opened='https://api2.openreview.net/notes/search (record for forum BMYxcibovq) ; https://openreview.net/forum?id=BMYxcibovq returned a browser challenge',
    verified='PARTIAL (title/venue/abstract from OpenReview API search record; forum page not readable)',
    what_it_does='Unified spatiotemporal observation model coupling microbolometer thermal persistence and RS readout for thermal sequence reconstruction.',
    what_it_does_not_do='Image restoration; thermal; no 3D FM or pose.', code='N (anonymous submission)', how_we_differ='Unrelated task; shows ICLR 2027 RS submissions found are restoration only.', threat='low')

# ---------------- adjacent FM + time/blur/events ----------------
add(topic='scoop:adjacent-4D-FM', title='V-DPM: 4D Video Reconstruction with Dynamic Point Maps',
    authors='Edgar Sucar, Eldar Insafutdinov, Zihang Lai, Andrea Vedaldi', venue='CVPR 2026 (CVF CVPR 2026 listing; arXiv comments list only project page)', year='2026',
    id='arXiv:2601.09499', url_opened='https://arxiv.org/abs/2601.09499 ; https://openaccess.thecvf.com/CVPR2026?day=all ; https://arxiv.org/html/2601.09499', verified='Y',
    what_it_does='Adapts VGGT with modest synthetic data to predict time-varying and time-invariant dynamic point maps over video (per-point 3D motion).',
    what_it_does_not_do='Inter-frame scene motion; one camera pose per frame; no RS (0 mentions in full text).', code='N (project page https://www.robots.ox.ac.uk/~vgg/research/vdpm/ ; no repo on arXiv page)',
    how_we_differ='Precedent that VGGT can be cheaply adapted to time-dependent pointmaps; we add intra-frame camera time instead of scene time.', threat='low')
add(topic='scoop:adjacent-4D-FM', title='DynamicVGGT: Learning Dynamic Point Maps for 4D Scene Reconstruction in Autonomous Driving',
    authors='Zhuolin He, Jing Li, Guanghao Li, Xiaolei Chen, Jiacheng Tang, Siyang Zhang, et al.', venue='arXiv', year='2026', id='arXiv:2603.08254',
    url_opened='https://arxiv.org/abs/2603.08254 ; https://arxiv.org/html/2603.08254', verified='Y',
    what_it_does='Extends VGGT to predict current+future point maps, motion-aware temporal attention, and per-Gaussian velocities via motion tokens (scene-flow supervised).',
    what_it_does_not_do='Velocity is of scene Gaussians, not the camera within a frame; no RS.', code='N (no repo link on arXiv page)',
    how_we_differ='Velocity tokens in a VGGT-style model exist, but for scene flow; ours is a camera twist applied per row.', threat='low')
add(topic='scoop:adjacent-blur-FM', title='Unblur-SLAM: Dense Neural SLAM for Blurry Inputs',
    authors='Qi Zhang, Denis Rozumny, Francesco Girlanda, Sezer Karaoglu, Marc Pollefeys, Theo Gevers, et al.', venue='CVPR 2026 (arXiv comments; CVF CVPR 2026 listing)', year='2026',
    id='arXiv:2603.26810', url_opened='https://arxiv.org/abs/2603.26810', verified='Y',
    what_it_does='Dense neural SLAM for blurry input; feed-forward deblurring model; models sub-frame camera poses within the exposure time.',
    what_it_does_not_do='Blur not RS; not a multi-view pointmap FM; rolling shutter not mentioned.', code='N (no repo link on arXiv page)',
    how_we_differ='Intra-exposure poses for blur in SLAM; ours are intra-readout poses for RS inside a feed-forward FM.', threat='low')
add(topic='scoop:adjacent-blur-FM', title='RoMo-GS: Efficient Online 3D Gaussian Splatting for Robot Mapping via Exposure-Integrated Motion Blur',
    authors='Sangwoo Jung, Di Deng, Jiahui Fu, Laura Herlant, Ayoung Kim, Yafei Hu', venue='arXiv', year='2026', id='arXiv:2610.24605',
    url_opened='https://arxiv.org/abs/2610.24605', verified='Y',
    what_it_does='Online 3DGS mapping that refines pose, velocity and Gaussians while integrating over the exposure interval.',
    what_it_does_not_do='Blur, per-scene optimisation; no RS; no FM stated.', code='N (no repo link on arXiv page)', how_we_differ='Per-frame velocity but in optimisation, for blur.', threat='low')
add(topic='scoop:adjacent-event-FM', title='EAG3R: Event-Augmented 3D Geometry Estimation for Dynamic and Extreme-Lighting Scenes',
    authors='Xiaoshan Wu, Yifei Yu, Xiaoyang Lyu, Yihua Huang, Bo Wang, Baoheng Zhang, et al.', venue='NeurIPS 2025 (spotlight) (arXiv comments; project page)', year='2025',
    id='arXiv:2512.00771', url_opened='https://arxiv.org/abs/2512.00771 ; https://candy-crusher.github.io/EAG3R_Proj ; https://github.com/Candy-Crusher/EAG3R', verified='Y',
    what_it_does='MonST3R backbone + Retinex enhancement + event adapter with SNR-aware fusion + event photometric loss in global optimisation for low-light dynamic scenes.',
    what_it_does_not_do='Needs events; no RS or intra-frame time; has an optimisation stage.', code='Y https://github.com/Candy-Crusher/EAG3R (weights: Y? README says via download_ckpt.sh, no direct link visible; license: none stated)',
    how_we_differ='Images only, RS-specific.', threat='low')
add(topic='scoop:adjacent-event-FM', title='Event3R: Asynchronous-to-Global 3D Reconstruction from Event Camera via Spatial-Temporal Feature Aggregation',
    authors='Jian Huang, Haotian Shen, Xinhao Lou, Chengrui Dong, Wenpu Li, Peidong Liu', venue='IROS 2026 (arXiv comments)', year='2026', id='arXiv:2607.15727',
    url_opened='https://arxiv.org/abs/2607.15727 ; https://arxiv.org/html/2607.15727', verified='Y',
    what_it_does='Feed-forward DUSt3R-based reconstruction from event streams via spatio-temporal voxels and temporal attention; masked-bin pre-training.',
    what_it_does_not_do='Event input only; no RS (0 mentions in full text).', code='N (no repo link on arXiv page)', how_we_differ='Different sensor.', threat='low')
add(topic='scoop:adjacent-event-FM', title='EventVGGT: Exploring Cross-Modal Distillation for Consistent Event-based Depth Estimation',
    authors='Yinrui Ren, Jinjing Zhu, Kanghao Chen, Zhuoxiao Li, Jing Ou, Zidong Cao, et al.', venue='ECCV 2026 (arXiv comments; ECCV 2026 accepted list)', year='2026',
    id='arXiv:2603.09385', url_opened='https://arxiv.org/abs/2603.09385 ; https://github.com/yinruiRen/EventVGGT', verified='Y',
    what_it_does='Distils VGGT spatio-temporal/multi-view knowledge into an event-based monocular depth model in three stages.',
    what_it_does_not_do='Events; no RS.', code='Y https://github.com/yinruiRen/EventVGGT (weights: Y, Hugging Face; MIT; inference code only)',
    how_we_differ='Different sensor; shows adapting VGGT to new sensor physics is publishable at ECCV.', threat='low')
add(topic='scoop:adjacent-blur-FM', title='DeblurSplat: SfM-free 3D Gaussian Splatting with Event Camera for Robust Deblurring',
    authors='Pengteng Li, Yunfan Lu, Pinhao Song, Weiyu Guo, Huizai Yao, F. Richard Yu, et al.', venue='TMM 2026 (arXiv comments)', year='2025', id='arXiv:2509.18898',
    url_opened='https://arxiv.org/abs/2509.18898', verified='Y',
    what_it_does='Uses DUSt3R on blurred images for SfM-free point-cloud init, event stream to decode sharp latents, then 3DGS.',
    what_it_does_not_do='DUSt3R used as-is; no in-model motion modelling; needs events; no RS.', code='N (no repo link on arXiv page)', how_we_differ='We change the FM itself.', threat='low')
add(topic='scoop:adjacent-blur-FM', title='AsyncEvGS: Asynchronous Event-Assisted Gaussian Splatting for Handheld Motion-Blurred Scenes',
    authors='Jun Dai, Renbiao Jin, Bo Xu, Yutian Chen, Linning Xu, Mulin Yu, et al.', venue='arXiv', year='2026', id='arXiv:2605.07192',
    url_opened='https://arxiv.org/abs/2605.07192', verified='Y',
    what_it_does='Event-based sharp-image reconstruction, VGGT-based cross-domain pose initialisation, per-scene 3DGS for handheld blur.',
    what_it_does_not_do='VGGT used as an off-the-shelf initialiser; events required; no RS.', code='N (project page https://openimaginglab.github.io/AsyncEvGS/ only)', how_we_differ='We modify the FM for RS.', threat='low')
add(topic='scoop:adjacent-FM-probing', title='ProMo3D: Probing Motion Cues in Frozen 3D Foundation Models',
    authors='Xiaoang Zhang, Mert Kiray, Yordanka Velikova, Manoj Biswanath, Benjamin Busam', venue='NeurIPS 2026 Main Track (arXiv comments)', year='2026',
    id='arXiv:2610.33802', url_opened='https://arxiv.org/abs/2610.33802', verified='Y',
    what_it_does='Shows static/dynamic separation is decodable from frozen 3D FM tokens with light readouts; Probing-based Motion Distillation for motion masks.',
    what_it_does_not_do='Object motion masks; no camera velocity; no RS.', code='UNVERIFIED (not checked)',
    how_we_differ='Supports our premise that light heads on FM tokens can read motion; we read the camera twist and use it geometrically.', threat='low')
add(topic='scoop:method-ingredient', title='Fin3R: Fine-tuning Feed-forward 3D Reconstruction Models via Monocular Knowledge Distillation',
    authors='Weining Ren, Hongjun Wang, Xiao Tan, Kai Han', venue='NeurIPS 2025 (arXiv comments)', year='2025', id='arXiv:2511.22429',
    url_opened='https://arxiv.org/abs/2511.22429 ; https://arxiv.org/html/2511.22429', verified='Y',
    what_it_does='Freezes the decoder and LoRA-adapts the image encoder of DUSt3R/MASt3R/CUT3R/VGGT via monocular-teacher distillation.',
    what_it_does_not_do='No RS, no motion modelling (0 RS mentions in full text).', code='UNVERIFIED (repo not located)',
    how_we_differ='Precedent for light LoRA fine-tuning of 3D FMs; our LoRA serves a new RS head and per-row unprojection.', threat='low')
add(topic='scoop:measurement', title='The Fast and the Distorted - How Sensor Readout Shapes Image-based 3D Modelling', authors='Geert J. Verhoeven',
    venue='ISPRS Archives XLVIII-2/W12-2026 (3D-ARCH 2026), pp. 495-502', year='2026', id='doi:10.5194/isprs-archives-XLVIII-2-W12-2026-495-2026',
    url_opened='https://isprs-archives.copernicus.org/articles/XLVIII-2-W12-2026/495/2026/', verified='Y',
    what_it_does='Shutter taxonomy; measures row-timing offsets with a 1 kHz LED; shows RS artefacts degrade exterior orientation and surfaces in Agisoft Metashape and that compensation helps.',
    what_it_does_not_do='Only classical photogrammetry (Metashape); no learned/feed-forward 3D models.', code='N',
    how_we_differ='Same message for classical SfM; we measure and fix it for feed-forward FMs.', threat='low')
add(topic='scoop:measurement', title='Robust Visual SLAM for UAV Navigation in GPS-Denied and Degraded Environments: A Multi-Paradigm Evaluation and Deployment Study',
    authors='Prasoon Kumar, Akshay Deepak, Sandeep Kumar', venue='arXiv', year='2026', id='arXiv:2605.03678', url_opened='https://arxiv.org/abs/2605.03678', verified='Y',
    what_it_does='Benchmarks ORB-SLAM3, DPVO, DROID-SLAM, DUSt3R and MASt3R under five degradations incl. motion blur.',
    what_it_does_not_do='No RS degradation named in the abstract; no new method.', code='UNVERIFIED (not checked)', how_we_differ='FM robustness evaluations exist for blur but not for RS.', threat='low')

with open(OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=COLS, quoting=csv.QUOTE_MINIMAL)
    w.writeheader()
    for r in R:
        assert set(r) == set(COLS), (r['title'], set(COLS) ^ set(r))
        w.writerow(r)
print('wrote', len(R), 'rows to', os.path.abspath(OUT))
