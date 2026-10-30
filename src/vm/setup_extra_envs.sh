#!/bin/bash
# EXP2 VM: every environment beyond the main .venv (src/setup_env.sh), pinned as on the first VM (the ENV.md files
# under src/ext/*), built in parallel. Each env logs to logs/env_<name>.log and appends ENV_<name>_DONE or
# ENV_<name>_FAILED to logs/setup_extra.log. GPU envs end with a bf16 matmul on the card (Blackwell needs cu128).
# Change from the first VM (DEVIATIONS D19): the 2D-unroller env moves from torch 2.5.1+cu124 to 2.7.1+cu128,
# because cu124 builds have no kernels for the RTX PRO 6000 (sm_120); its compiled ops are pure-PyTorch shims.
source ~/rs3d/src/vm/lib.sh
UV=$R/bin/uv
TP=$R/third_party
mkdir -p $TP $R/envs $R/weights $R/cache/torch
TORCH="torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128"
CV=opencv-python-headless==4.10.0.84

clone() {  # clone <url> <dir> <commit>
  [ -d $TP/$2/.git ] || git clone -q $1 $TP/$2
  git -C $TP/$2 fetch -q origin $3 2>/dev/null || true
  git -C $TP/$2 checkout -q $3
  echo "clone $2 $(git -C $TP/$2 rev-parse HEAD)"
}
venv() { [ -x $R/envs/$1/bin/python ] || $UV venv -q --python ${2:-3.11} $R/envs/$1; }
pipi() { local e=$1; shift; $UV pip install -q --python $R/envs/$e/bin/python "$@"; }
gpu_ok() {
  $R/envs/$1/bin/python -c "
import torch
x = torch.randn(256, 256, device='cuda', dtype=torch.bfloat16)
print('$1 torch', torch.__version__, torch.cuda.get_device_name(0), 'cc', torch.cuda.get_device_capability(0), float((x @ x).float().abs().mean()))"
}
hf() {  # hf <env> <repo> <revision> [file]: pinned snapshot / file into HF_HOME
  $R/envs/$1/bin/python -c "
import sys
from huggingface_hub import snapshot_download, hf_hub_download
r, v, f = '$2', '$3', '${4:-}'
print('hf', r, v, hf_hub_download(r, f, revision=v) if f else snapshot_download(r, revision=v))"
}
run() {  # run <name>: env_<name> function, logged, with a marker
  ( set -ex; env_$1 ) > $L/env_$1.log 2>&1 && echo "$(stamp) ENV_$1_DONE" >> $L/setup_extra.log \
    || echo "$(stamp) ENV_$1_FAILED" >> $L/setup_extra.log
}

env_da3() {
  clone https://github.com/ByteDance-Seed/Depth-Anything-3.git da3 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
  venv da3; pipi da3 $TORCH
  pipi da3 "numpy<2" pandas==2.3.3 $CV pillow einops huggingface_hub safetensors omegaconf addict trimesh imageio \
    "moviepy==1.0.3" plyfile pillow_heif evo e3nn pycolmap tqdm scipy matplotlib typer requests pyyaml
  pipi da3 xformers==0.0.31.post1 torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  gpu_ok da3
  # xformers must have kernels for this GPU; otherwise remove it (DA3's DINOv2 layers fall back to PyTorch SDPA)
  $R/envs/da3/bin/python -c "
import torch, xformers.ops as xo
q = torch.randn(1, 256, 8, 64, device='cuda', dtype=torch.bfloat16)
print('xformers ok', float(xo.memory_efficient_attention(q, q, q).float().abs().mean()))" \
    || { $UV pip uninstall -q --python $R/envs/da3/bin/python xformers; echo "XFORMERS_REMOVED (no kernels for this GPU)"; }
  hf da3 depth-anything/DA3-GIANT-1.1 72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19
}

env_streamvggt() {
  clone https://github.com/wzzheng/StreamVGGT.git streamvggt 05682bcb05683c4c71522e9cb618fc8ca349063c
  venv streamvggt; pipi streamvggt $TORCH
  pipi streamvggt numpy==1.26.4 pandas==2.3.3 $CV pillow einops huggingface_hub safetensors transformers matplotlib scipy
  gpu_ok streamvggt
  hf streamvggt lch01/StreamVGGT f9ba55b1955bc4f34337c51142158a9aa2862c7f
}

env_vggt_omega() {  # code only; the gated weights are fetched by run_models.sh once the user's token exists
  clone https://github.com/facebookresearch/vggt-omega.git vggt_omega 48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6
  venv vggt_omega; pipi vggt_omega $TORCH
  pipi vggt_omega numpy==1.26.4 pandas==2.3.3 $CV pillow einops huggingface_hub safetensors scipy pyyaml
  gpu_ok vggt_omega
}

env_omnivggt() {
  clone https://github.com/Livioni/OmniVGGT-official.git omnivggt c8cbfc6bcd57a07bf08a1f97804da9827847af8c
  venv omnivggt; pipi omnivggt $TORCH
  pipi omnivggt numpy==1.26.4 pandas==2.3.3 $CV pillow einops huggingface_hub safetensors scipy pyyaml matplotlib tqdm
  gpu_ok omnivggt
  hf omnivggt Livioni/OmniVGGT 29c676258c17b950afc3f19daa62d726bc331116 OmniVGGT.safetensors
  $R/envs/omnivggt/bin/python -c "
import sys; sys.path.append('$TP/omnivggt')
from omnivggt.models.omnivggt import OmniVGGT
from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri
print('omnivggt import ok')"
}

env_aria() {
  venv aria; pipi aria projectaria-tools==2.3.0 pandas==2.3.3 scipy pillow
  ls $R/envs/aria/bin/aria_dataset_downloader
}

env_waymo() {
  if [ ! -d $R/bin/google-cloud-sdk ]; then
    curl -sSL --max-time 600 -o $R/bin/gcloud.tar.gz \
      https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
    tar -xzf $R/bin/gcloud.tar.gz -C $R/bin && rm $R/bin/gcloud.tar.gz
    $R/bin/google-cloud-sdk/install.sh --quiet --usage-reporting false --path-update false --command-completion false
  fi
  cat > $R/bin/waymo_login.sh <<'EOF'
#!/bin/bash
# Waymo Open Dataset access for RS-3D: logs YOUR Google account into gcloud on this VM. The credential is kept in
# ~/rs3d/.gcloud (our project folder only). Run it yourself: it prints a link; sign in with the same Google account
# that accepted the Waymo licence, then paste the code back into this terminal (not into the chat).
export CLOUDSDK_CONFIG=$HOME/rs3d/.gcloud
$HOME/rs3d/bin/google-cloud-sdk/bin/gcloud auth login --no-launch-browser || exit 1
echo
if $HOME/rs3d/bin/google-cloud-sdk/bin/gcloud storage ls gs://waymo_open_dataset_v_1_4_3/ > /dev/null 2>&1; then
  echo "LOGIN DONE and Waymo data is readable. The download starts by itself."
else
  echo "LOGIN DONE, but the bucket is not readable yet: accept the licence at https://waymo.com/open first."
fi
EOF
  chmod +x $R/bin/waymo_login.sh
  venv waymo; pipi waymo "protobuf>=5,<6" "grpcio-tools>=1.66,<1.70" numpy pandas pillow scipy $CV
  if [ ! -d $TP/waymo-open-dataset ]; then
    git clone -q --depth 1 --filter=blob:none --sparse https://github.com/waymo-research/waymo-open-dataset.git \
      $TP/waymo-open-dataset
    git -C $TP/waymo-open-dataset sparse-checkout set src/waymo_open_dataset/protos
    git -C $TP/waymo-open-dataset sparse-checkout add "src/waymo_open_dataset/*.proto"
  fi
  echo "waymo-open-dataset $(git -C $TP/waymo-open-dataset rev-parse HEAD)"
  mkdir -p $R/ext/waymo_pb
  (cd $TP/waymo-open-dataset/src && $R/envs/waymo/bin/python -m grpc_tools.protoc -I. --python_out=$R/ext/waymo_pb \
    waymo_open_dataset/dataset.proto waymo_open_dataset/label.proto waymo_open_dataset/protos/map.proto \
    waymo_open_dataset/protos/vector.proto waymo_open_dataset/protos/keypoint.proto)
  touch $R/ext/waymo_pb/waymo_open_dataset/__init__.py $R/ext/waymo_pb/waymo_open_dataset/protos/__init__.py
  PYTHONPATH=$R/ext/waymo_pb $R/envs/waymo/bin/python -c "from waymo_open_dataset import dataset_pb2; print('proto ok')"
}

env_u2d() {  # src/ext/unroll2d/ENV.md, torch moved to 2.7.1+cu128 (D19)
  clone https://github.com/ljzycmd/DFRSC.git DFRSC adde28a1e7e9b287d451dba7402fff2005d94315
  clone https://github.com/GitCVfb/JAMNet.git JAMNet 7a4756c155ad559c32384c9ef213f1d0b7e9da62
  clone https://github.com/ljzycmd/BSRSC.git BSRSC d8cd948a98fc6603aedb8a717c8cf455145dd55e
  clone https://github.com/GitCVfb/LBCNet.git LBCNet b4c9524218c9eeddd302455a798412fd2d754ef9
  clone https://github.com/ethliup/DeepUnrollNet.git DeepUnrollNet 8ffe0eb8832419b4fbadb4180dd9ac17f61774ef
  clone https://github.com/zzh-tech/RSCD.git RSCD 676a504bade9b4385d2e3fb19f52b769273464ff
  clone https://github.com/ljzycmd/SimDeblur.git SimDeblur 0a6da06beda9c62aebc0f049360ab80986acfc26
  venv u2d 3.10; pipi u2d $TORCH
  pipi u2d "numpy<2" $CV scikit-image imageio scipy easydict pyyaml tqdm pandas gdown lmdb tensorboardX pillow matplotlib
  gpu_ok u2d
  local W=$R/weights
  mkdir -p $W/dfrsc $W/adarsc $W/deepunrollnet $W/jcd
  for f in dfrsc_2f_bsrsc dfrsc_2f_carlars dfrsc_2f_fastecrs dfrsc_3f_bsrsc dfrsc_3f_carlars dfrsc_3f_fastecrs; do
    [ -s $W/dfrsc/$f.pth ] || curl -sL -o $W/dfrsc/$f.pth https://github.com/ljzycmd/DFRSC/releases/download/ckpts/$f.pth
  done
  for f in adarsc_bsrsc adarsc_fastecrs; do
    [ -s $W/adarsc/$f.pth ] || curl -sL -o $W/adarsc/$f.pth https://github.com/ljzycmd/BSRSC/releases/download/v1.0.0/$f.pth
  done
  local GD=$R/envs/u2d/bin/gdown
  # Google Drive can refuse ("too many users"); a missing file only drops that method (logged)
  [ -s $W/deepunrollnet/pretrained_carla_net_G.pth ] || $GD -q "https://drive.google.com/uc?id=10ThiIUKW3kEWXI_v4ZeS0DD7I_jdJa6m" -O $W/deepunrollnet/pretrained_carla_net_G.pth || echo "GDRIVE_FAILED dun carla"
  [ -s $W/deepunrollnet/pretrained_fastec_net_G.pth ] || $GD -q "https://drive.google.com/uc?id=18tCt8q9kVGrImgmxaUiCt6JSFi5EwJLZ" -O $W/deepunrollnet/pretrained_fastec_net_G.pth || echo "GDRIVE_FAILED dun fastec"
  if [ ! -d $W/jcd/checkpoints ]; then
    $GD -q "https://drive.google.com/uc?id=1_B6Q9K9V1WycKkEQscS5Hhd6WxR3GRT9" -O $W/jcd/checkpoints.zip \
      && (cd $W/jcd && unzip -q -o checkpoints.zip) || echo "GDRIVE_FAILED jcd"
  fi
  (cd $W && sha256sum dfrsc/*.pth adarsc/*.pth deepunrollnet/*.pth jcd/checkpoints/*.tar 2>/dev/null)
}

env_classical() {  # src/ext/classical/ENV.md (COLMAP 4.2.1 with its built-in GLOMAP; CPU SIFT in run_sfm3.py)
  venv classical_cuda
  pipi classical_cuda pycolmap-cuda12==4.2.1 numpy==2.4.6 scipy pandas==2.3.3 pyyaml $CV tabulate
  $R/envs/classical_cuda/bin/python -c "import pycolmap; print('pycolmap', pycolmap.__version__)"
}

env_rsmp() {  # Barath RS relative pose (D17 scope); needs the apt packages from bootstrap.sh
  wait_for $L/apt.log "APT_"
  grep -q APT_DONE $L/apt.log
  clone https://github.com/danini/rolling_shutter_made_practical.git rolling_shutter_made_practical \
    d8340562be6a84a879e13ae61548f96cbdf807dd
  git -C $TP/rolling_shutter_made_practical submodule update --init --recursive
  git -C $TP/rolling_shutter_made_practical/external/superansac checkout -q a3015a3dbebe4695c188eed42609a744d0ee4768
  venv rsmp; pipi rsmp $TORCH
  pipi rsmp kornia kornia_moons h5py pandas matplotlib tqdm romatch pybind11 numpy scipy $CV pyyaml setuptools wheel cmake tabulate
  pipi rsmp numpy==2.2.6
  pipi rsmp fused-local-corr==0.3.27 numpy==2.2.6 torch==2.7.1 --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match --only-binary fused-local-corr || echo "FUSED_LOCAL_CORR_FAILED"
  gpu_ok rsmp
  bash $R/src/ext/classical/build_rsmp_py.sh
}

for e in da3 omnivggt streamvggt vggt_omega aria waymo u2d classical rsmp; do run $e & done
wait
echo "$(stamp) SETUP_EXTRA_DONE $(grep -c _DONE $L/setup_extra.log) done, $(grep -c _FAILED $L/setup_extra.log) failed" >> $L/setup_extra.log
