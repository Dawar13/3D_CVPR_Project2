# unroll2d: learned 2D rolling-shutter correction baselines (EXP2)

Everything for these baselines lives in `~/rs3d/ext/unroll2d/` (code), `~/rs3d/third_party/<repo>` (pinned clones,
unmodified), `~/rs3d/weights/<name>` (checkpoints), `~/rs3d/envs/u2d` (env), `~/rs3d/logs/unroll2d_*.log`,
`~/rs3d/results/exp2/baselines/unroll2d/` (results). Nothing in the git repo `~/rs3d/src` was modified.

## 1. Environment

```
export UV_CACHE_DIR=~/rs3d/.uv-cache
~/rs3d/bin/uv venv -p 3.10 ~/rs3d/envs/u2d
~/rs3d/bin/uv pip install --python ~/rs3d/envs/u2d/bin/python torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124
~/rs3d/bin/uv pip install --python ~/rs3d/envs/u2d/bin/python "numpy<2" opencv-python-headless==4.10.0.84 \
    scikit-image imageio scipy easydict pyyaml tqdm pandas gdown lmdb tensorboardX pillow matplotlib
```
Result: Python 3.10, torch 2.5.1+cu124 (A100 sm_80, driver 550 / CUDA 12.4), numpy 1.26.4, opencv 4.10.
Full freeze: `~/rs3d/logs/unroll2d_pip_freeze.txt`; install log `~/rs3d/logs/unroll2d_env.log`.
One env serves all six methods: the authors' network code runs unmodified on torch 2.5; the compiled ops it needs
are replaced by pure-PyTorch shims (section 3). No nvcc on the VM, so nothing is compiled.
`prep_nb.py` and the 3D scoring use the main env (`~/rs3d/.venv`).

## 2. Clones (pinned) and weights

| repo | URL | commit | licence |
|---|---|---|---|
| DFRSC | https://github.com/ljzycmd/DFRSC | adde28a1e7e9b287d451dba7402fff2005d94315 | none in repo (research use) |
| JAMNet | https://github.com/GitCVfb/JAMNet | 7a4756c155ad559c32384c9ef213f1d0b7e9da62 | none; README: "research purpose only" |
| BSRSC (AdaRSC) | https://github.com/ljzycmd/BSRSC | d8cd948a98fc6603aedb8a717c8cf455145dd55e | MIT |
| LBCNet | https://github.com/GitCVfb/LBCNet | b4c9524218c9eeddd302455a798412fd2d754ef9 | none; README: "research purpose only" |
| DeepUnrollNet | https://github.com/ethliup/DeepUnrollNet | 8ffe0eb8832419b4fbadb4180dd9ac17f61774ef | MIT |
| RSCD (JCD) | https://github.com/zzh-tech/RSCD | 676a504bade9b4385d2e3fb19f52b769273464ff | MIT |
| SimDeblur (read only, for AdaRSC pre/post-processing) | https://github.com/ljzycmd/SimDeblur | 0a6da06beda9c62aebc0f049360ab80986acfc26 | MIT |

Weights (all public, no sign-in; sha256):

| file | source URL | sha256 |
|---|---|---|
| weights/dfrsc/dfrsc_2f_bsrsc.pth | https://github.com/ljzycmd/DFRSC/releases/download/ckpts/dfrsc_2f_bsrsc.pth | 236919653fa45544b9e91ba94bc1c9b6ed311f0b48bbfc3a1655d83955c50007 |
| weights/dfrsc/dfrsc_2f_carlars.pth | .../ckpts/dfrsc_2f_carlars.pth | 64134a9450ef2c06bcb11dc15eea846f30291edbe5c01192d4b8d8bf65513018 |
| weights/dfrsc/dfrsc_2f_fastecrs.pth | .../ckpts/dfrsc_2f_fastecrs.pth | eeff5babb7a96b311a83da1863bec7bbf2bea29b8d03d8fac670b64d2fc61fae |
| weights/dfrsc/dfrsc_3f_bsrsc.pth | .../ckpts/dfrsc_3f_bsrsc.pth | bfadc5aeab50362dac4da4200be8eaf31f2911364f30862616949a3eb0c720fc |
| weights/dfrsc/dfrsc_3f_carlars.pth | .../ckpts/dfrsc_3f_carlars.pth | ed53a2e37d321774cca640abbb4a90c8a8660e6aa75c818d858c29eb1b7f657e |
| weights/dfrsc/dfrsc_3f_fastecrs.pth | .../ckpts/dfrsc_3f_fastecrs.pth | 7120964d9cdca2b53a9d365d8c893bf3d3b59df463bb24dbba914c19cd63bcdf |
| weights/adarsc/adarsc_bsrsc.pth | https://github.com/ljzycmd/BSRSC/releases/download/v1.0.0/adarsc_bsrsc.pth | 2ea1b02858cd7dbd24629894f43414ee6b8921e9908ebf569914973e2a25ecfe |
| weights/adarsc/adarsc_fastecrs.pth | .../v1.0.0/adarsc_fastecrs.pth | c356c38dcbcfbf2441575d863177287f0d5fba9bf2f83185004b024697d4d5a8 |
| third_party/JAMNet/deep_unroll_weights/model_weights/bsrsc/pre_net_flow.pth | in repo | 897582d91958b9ec350cacc9051dc6db605470683670b5670c38594b4c0a1d24 |
| .../JAMNet/.../carla/pre_net_flow.pth | in repo | db4b283758892880c9af102a7d300ae4c6e983a9c26cd378fb34fa96954eed52 |
| .../JAMNet/.../fastec/pre_net_flow.pth | in repo | 1353d5fe23f56ee6e203cf5749f9c1f863eb4cb58a5280a361064ea314b366c2 |
| third_party/LBCNet/deep_unroll_weights/model_weights/bsrsc/pre_net_flow.pth | in repo | f26110670df867bde250635b5e31723a3b82f9d54e117a87a048665c4b46d8c2 |
| .../LBCNet/.../carla/pre_net_flow.pth | in repo | 59c759808278f03e670736d6339ab9095a308732e1b19299531b6899b4c884af |
| .../LBCNet/.../fastec/pre_net_flow.pth | in repo | 306bbedd686507d177a722e10097b2942d1f4c005d62003afbc6627d7e26f192 |
| weights/deepunrollnet/pretrained_carla_net_G.pth | https://drive.google.com/uc?id=10ThiIUKW3kEWXI_v4ZeS0DD7I_jdJa6m (download_pretrained_models.sh) | a1d6134cedecad075f558e3d8f398fb2f60f7814c572257885266b8fd312633f |
| weights/deepunrollnet/pretrained_fastec_net_G.pth | https://drive.google.com/uc?id=18tCt8q9kVGrImgmxaUiCt6JSFi5EwJLZ | 913b16b0239b423564360764628837b9f4de8252e0c3dcfd1f64e5210a4b856c |
| weights/jcd/checkpoints.zip | https://drive.google.com/file/d/1_B6Q9K9V1WycKkEQscS5Hhd6WxR3GRT9 | ad2e4f90840b19862ed96d96ab344043cb9cea772cce341c88805470f92cc22b |
| weights/jcd/checkpoints/JCD_BS-RSCD.tar | from the zip | efde2c4626b7c40b9834792d0844790ea2a96f10bb85ce6ffa4f4e1c3b86814b |
| weights/jcd/checkpoints/JCD_Fastec-RS.tar | from the zip (not named in the README, but present) | 44c5c7ec8a222c05533a60edea0a8f56b3057487a3ba0ad7f4278188700a8af9 |

Downloads: GitHub release assets with `curl -sL`, Google Drive with `~/rs3d/envs/u2d/bin/gdown "https://drive.google.com/uc?id=<id>"`.

Test data for the reproduction checks:
- Repo demo pairs: `third_party/{JAMNet,LBCNet,DeepUnrollNet}/demo/{Fastec,Carla,BSRSC}/seq_*` (rs_0, rs_1, GT GS).
- Fastec-RS test set: JCD's public LMDB mirror `fastec_rs_lmdb.zip` (https://drive.google.com/file/d/1vs9sxav9h9Yjjs11KCxKtFuzH2gaRZAx,
  3.13 GB) -> `~/rs3d/data/rsc2d/fastec_rs_lmdb/` (19 test sequences, 645 frames; GT = `*_global_middle.png`, stored BGR).
- BS-RSC (https://drive.google.com/file/d/1h7UP1kci8zbg3TQp37J-imrvzlh2X6zn): **not downloadable** on 2026-10-29
  (Google Drive "too many users / quota", tried twice). So BS-RSC-trained 3-frame models (DFRSC-3F-bsrsc,
  AdaRSC-bsrsc) and JCD-BS-RSCD have **no numeric reproduction check** (UNVERIFIED beyond strict weight loading + the
  same code passing on Fastec).

## 3. Shims (no compiled ops available) - `~/rs3d/ext/unroll2d/shims/`

| shim | replaces | used by | how verified |
|---|---|---|---|
| `correlation_package` | FlowNet/PWC correlation CUDA op (`corr_cuda_kernel.cu`) | LBCNet, AdaRSC, DeepUnrollNet, JCD | semantics copied from the kernel (zero pad, channel = dy_idx*(2r+1)+dx_idx, sum/(k*k*C)); demo/Fastec PSNR reproduce (sec. 5) |
| `forward_warp_package` | DeepUnrollNet Gaussian forward splatting CUDA op | DeepUnrollNet, JCD | kernel semantics (window int(p-r+0.5)..int(p+r+0.5), w = exp(-d2/(2 sigma2)), normalised); floor vs C truncation differs only for off-image targets (weights <= exp(-6.25)); PSNR reproduces |
| `detectron2.layers` | detectron2 ModulatedDeformConv / DeformConv | JAMNet, JCD | torchvision.ops.deform_conv2d, same parameter names and offset layout; JAMNet demo PSNR +10 to +17 dB over RS |
| `basicsr.utils.registry`, `simdeblur.model.build` | framework registries | DFRSC, AdaRSC | decorators only |

Inference-only patches applied at load time in `methods.py` (no change to the maths):
- AdaRSC `AdaWarp.feats_sampling` builds its integer pixel grid on the GPU once instead of on the CPU 81x per frame
  (7.9 s/frame -> 0.7 s/frame on the loaded shared CPU); output verified bit-identical (max abs diff 0.0).
- DFRSC `FlowAttention` grid cache disabled (it is keyed by H, W only and breaks when the batch size changes).
- bf16 autocast was tested (DFRSC-3F, LBCNet on Fastec): no speed gain on the shared GPU and DFRSC output changed
  (mean |diff| 1.1 grey levels, -0.09 dB), so everything runs in fp32.

## 4. Per-method facts

All six nets are trained on RGB; TUM is grey, so the grey frame is replicated to 3 channels and the output's 3
channels are averaged. All target the **middle scanline** of the corrected frame (scored against the centre-row RS
GT `{key}_RS`); none of the released models outputs the first row (LBCNet could, see below).

| method | inputs (20 fps neighbours) | output time | released weights | training readout ratio gamma | training resolution | net input on TUM |
|---|---|---|---|---|---|---|
| DFRSC-3F (CVPR'24) | t-1, t, t+1 + time map (row-(H-1)/2)/(H-1) | middle row of t (configs `pos: middle`) | Fastec-RS, Carla-RS, BS-RSC | Fastec/Carla: 1 (dataset definition); BS-RSC: not stated | Fastec 640x480 (crops 480x256), Carla 640x448, BS-RSC 1024x768 | 640x512 (Fastec/Carla), 960x768 (BS-RSC) |
| DFRSC-2F | t-1, t + time map | middle row of t (order determined empirically: (t-1, t) gives 28.20 dB on Fastec test vs 19.16 dB for (t, t+1); no 2F config in the repo) | same 3 | same | same | same |
| JAMNet (CVPR'23) | t-1, t | middle row of t (demo.sh; explicit `gamma = 1.0` in code) | Fastec, Carla, BS-RSC | 1 (hard-coded) | as above | as above |
| AdaRSC (CVPR'22) | t-1, t, t+1 + time map (time map multiplies the flows) | middle row of t | BS-RSC, Fastec-RS | Fastec 1; BS-RSC not stated | BS-RSC 1024x768 (crops 480x256) | 640x512 / 960x768 |
| LBCNet (TPAMI'24) | t-1, t + time t | any time; we use 1.0 = middle row of t (BS-RSC weights trained only at 1.0); first row of t = 1 - gamma/2 | Fastec, Carla, BS-RSC | 1 (hard-coded `gamma = 1.0`) | as above | as above |
| DeepUnrollNet (CVPR'20) | t-1, t | middle row of t: released models are run with `--net_type netMiddle`, which forces `pred_mid_gs=True` (the repo demo GT `gs_1.png` is byte-identical to JAMNet's `gs_1_m.png`). The lit note "first scanline by default" applies to the data loader only, not to the released models | Fastec, Carla | 1 | 640x480 / 640x448 | 640x512 |
| JCD (CVPR'21, RS + deblur) | t-1, t, t+1 + time map | middle row of t | BS-RSCD, Fastec-RS (both in checkpoints.zip) | Fastec 1; BS-RSCD not stated | BS-RSCD 640x480 (half res), Fastec 640x480 | 640x512 |

TUM: readout 30.18 ms / 50 ms frame interval -> gamma = 0.60, i.e. the zero-shot nets (gamma = 1 for Fastec/Carla)
are expected to over-correct by ~1/0.6 if they learned the readout.

## 5. Reproduction on the authors' sample data (`check_repro.py`, PSNR on uint8 RGB)

Fastec-RS test = first 60 test frames that have the needed neighbours (2 sequences); demo = mean over the repo
demo pairs of that domain (GT = middle-row GS of frame 1). `results/exp2/baselines/unroll2d/repro/*.json` + images.

| method / weights | Fastec test RS -> out | demo RS -> out (n pairs) |
|---|---|---|
| DFRSC-3F fastecrs | 21.56 -> 29.09 | - (3-frame net, demos have 2 frames) |
| DFRSC-2F fastecrs | 21.61 -> 28.20 | Fastec 21.87 -> 27.83 (10) |
| JAMNet fastec / carla / bsrsc | 21.61 -> 28.30 | Fastec 21.87 -> 27.88 (10); Carla 21.20 -> 31.23 (10); BS-RSC 18.44 -> 34.28 (6) |
| AdaRSC fastecrs | 21.56 -> 28.15 | - |
| LBCNet fastec / carla / bsrsc | 21.61 -> 28.23 | Fastec 21.87 -> 27.54 (10); Carla 21.20 -> 31.56 (10); BS-RSC 18.44 -> 33.65 (6) |
| DeepUnrollNet fastec / carla | 21.61 -> 27.02 | Fastec 21.87 -> 26.42 (10); Carla 21.20 -> 27.93 (10) |
| JCD fastec | 21.56 -> 25.24 | - |

All in the range the papers report for these benchmarks (roughly 26.5-29 dB on Fastec-RS for the 2020-2024 methods,
e.g. DeepUnrollNet's paper value 26.52 dB on the full Fastec test set). Two LBCNet Fastec demo pairs
(`LBCNet/demo/Fastec/seq_0{1,2}`) give ~17.6 dB for every method (RS 19.1-19.3 dB), i.e. a property of those demo
files, not of a model.

## 6. TUM processing (`prep_nb.py`, `run_tum.py`)

- Neighbours: `prep_nb.py` (main env) preps cam1 frames i-2..i+2 around every kept window frame with
  `prep.shared_new_K / _init_worker / _prep_one` imported from `src/prep.py` into `~/rs3d/data/prep_nb/seq{s}/cam1/`
  (9437 frames, 4.7 GB). K_new asserted equal to `prep_info.json`. Recipe check: all 9437 also exist in
  `data/prep` (prep.py had prepped every cam1 frame) and are pixel-identical (max abs diff 0). Frame intervals of the
  needed pairs 49.90-50.10 ms, no dropped frames. 3 of 9248 unique window frames have no t-1 or t+1 (sequence ends:
  seq3 first+last frame, seq4 first frame); the frame itself is used in place of the missing neighbour.
- Resolution: 1280x1024 -> exact downscale by s (antialiased bilinear, align_corners=False): s = 0.5 (640x512) for
  Fastec/Carla/BS-RSCD weights, s = 0.75 (960x768) for BS-RSC weights, i.e. the net runs at its training pixel
  scale; output -> mean of 3 channels -> exact x1/s bicubic upscale to 1280x1024 -> uint8. Pixel centres map as
  x_full = (x_net + 0.5)/s - 0.5 both ways, so K_new is unchanged and nothing is cropped (no padding was needed:
  both sizes are multiples of 64).
- Outputs: unique straightened frames in `~/rs3d/data/u2d/<COND>/seq{s}/{t}.png`; window folders
  `~/rs3d/data/windows/seq{s}/{wid:04d}_<COND>/{k:02d}_{t}.png` are symlinks to them (same convention as the RS / GS
  folders, which are symlinks into data/prep). Windows overlap, so 9248 unique frames serve 17984 window slots.
  `COND = U2D-<method>-<weights>[-g<gamma>]`.

## 7. GPU discipline

From 14:38 UTC all GPU jobs run sequentially from one queue (queue file `~/rs3d/ext/unroll2d/queue.txt`, log
`~/rs3d/logs/unroll2d_queue.log`, job helpers `jobs.sh`) through `~/rs3d/src/gpu_slot.sh` with `RS3D_GPU_SLOTS=1`:
first `queue_runner.sh` (tmux `u2d-queue`; private lock dir via a redirected HOME, because gpu_slot.sh then
hard-coded `~/rs3d/locks`), then, after the fine-tuning job, `queue_runner2.sh` (tmux `u2d-queue2`) with
`RS3D_LOCK_DIR=~/rs3d/locks_u2d` as asked. To add work: append a line to queue.txt (see jobs.sh for helpers: `tum`,
`synth`, `tum3d`, `synth3d_all`). Before 14:38 the smoke runs used up to 3 concurrent processes; one of them (DFRSC
smoke) was killed by an OOM on the shared GPU and re-run.

## 8. Synthetic sets (`run_synth.py`)

- Input per frame k: `rs_nb/{k}_m1.png`, `rs/{k}.png`, `rs_nb/{k}_p1.png` (RS frames exactly 1/fps apart), RGB
  640x512, in the net's order; output `<clip>/u2d-<method>-<weights>/{k:02d}.png` (640x512 RGB).
- Same resolution rule as TUM: net height 512 for Fastec/Carla/BS-RSCD weights (scale 1.0), 768 for BS-RSC weights
  (scale 1.5, 960x768; bilinear up, antialiased bicubic down); the fine-tuned model runs at 512 (its training
  scale). The synthetic camera (640x512, fx = 320) has nearly the same pixel scale as TUM at s = 0.5 (fx = 322).
- Scan direction canonicalised to top->bottom ('bt' flipped vertically, 'lr' transposed) and mapped back; core_test
  is all 'tb' (30 ms readout, 20 fps, gamma = 0.6).
- Per-frame PSNR of RS and of the output vs the GS twin: `results/exp2/baselines/unroll2d/synth/<set>_<split>_<name>.csv`,
  summary `summarize_synth2d.py` -> `results/exp2/baselines/unroll2d/synth2d_core_test_test.{csv,md}`.
- 3D: `jobs.sh: synth3d_all` builds `clips/synth_core_test_u2d.csv` with src/synth_clips.py (`u2d-...:d`, dense) for
  every complete u2d condition and runs src/infer2.py (pi3, vggt) into `results/exp2/synth/core_test/preds`
  (`<model>_u2d-....npz`); scoring with src/synth_eval.py -> `results/exp2/baselines/unroll2d/synth_core_test.csv`.

## 9. Fine-tuning (`finetune_dfrsc.py`) -> condition `U2D-dfrsc3f-ft` / `u2d-dfrsc3f-ft`

- Method: DFRSC-3F (best released model on our checks; the only one with a time-map input, which lets the readout
  be given to the net). Init chosen on **val** (never test), iteration 0 = released weights with the gamma-scaled
  time map, 240-480 val frames: bsrsc 31.9 dB, carlars 28.9 dB, fastecrs 27.9 dB -> init = dfrsc_3f_bsrsc.
- Data: core_trainval/train clips with `rs_nb` (clip ids 0-39 of 12 scenes = 480 clips, 7680 frames); readout
  {0,5,10,20,30,40} ms, direction tb/bt/lr (canonicalised to tb); val: core_trainval/val (60 clips), fixed random
  subset of 240 frames, full 640x512 (lr clips 512x640 after transposition).
- Readout conditioning: time map x gamma (= readout x fps), i.e. per-row time offset in frame intervals. TUM: gamma 0.6.
- Recipe: authors' losses (Charbonnier + 0.05 x masked distortion-flow loss), Adam(0.9, 0.99), EMA 0.999, crops
  512x256 + hflip, 30% grey samples (TUM is grey); lr 1e-4, batch 8, val every 500 iters (EMA weights), early stop
  after 4 evals without +0.01 dB, max 6000 iters. Checkpoints: `~/rs3d/weights/dfrsc_ft/dfrsc3f-ft/{best,last}.pth`,
  `log.csv`, `config.json`.
- TUM inference: `run_tum.py --method dfrsc3f --weights bsrsc --gamma 0.6 --scale 0.5 --ckpt .../best.pth --cond U2D-dfrsc3f-ft`
  (640x512, the training pixel scale).
- Note on the val PSNR: val contains readout-0 clips where RS == GS, so the RS input's mean val PSNR is high
  (47.4 dB over 480 frames, identical frames capped by the 1e-10 MSE floor); the released nets lose detail even when
  nothing needs correcting, which is a large part of what fine-tuning recovers.
