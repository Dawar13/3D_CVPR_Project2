#!/bin/bash
# EXP2 VM setup: folders, uv, the main venv (pi3 / VGGT / MapAnything at EXP0's commits) and model weights.
# Idempotent. Everything stays under ~/rs3d (shared-GPU rule). Log: ~/rs3d/logs/setup_env.log
set -euo pipefail
R=~/rs3d
mkdir -p $R/{src,configs,data,cache,results/exp2,logs,report,third_party,bin,.hf}
exec > >(tee -a $R/logs/setup_env.log) 2>&1
echo "== $(date -u +%FT%TZ) setup_env start"

# uv, because these images have no ensurepip
if [ ! -x $R/bin/uv ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$R/bin UV_NO_MODIFY_PATH=1 INSTALLER_NO_MODIFY_PATH=1 sh
fi
UV=$R/bin/uv
export UV_CACHE_DIR=$R/.uv-cache
[ -d $R/.venv ] || $UV venv --python 3.11 $R/.venv
source $R/.venv/bin/activate
$UV pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

cd $R/third_party
[ -d map-anything ] || { git clone -q https://github.com/facebookresearch/map-anything.git &&
                         git -C map-anything checkout -q 3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9; }
if [ ! -d vggt ]; then  # EXP0 did not record its VGGT commit: take the last one on or before the EXP0 run date
  git clone -q https://github.com/facebookresearch/vggt.git
  git -C vggt checkout -q "$(git -C vggt rev-list -1 --before=2026-10-27T23:59:59Z HEAD)"
fi
[ -d pi3 ] || { git clone -q https://github.com/yyfz/Pi3.git pi3 && git -C pi3 checkout -q 9fa3ddb; }
$UV pip install -e map-anything
$UV pip install -e vggt --no-deps
$UV pip install opencv-python-headless scipy pandas matplotlib pyyaml huggingface_hub safetensors einops tqdm \
  pillow open3d peft
# pi3/requirements.txt is NOT installed: it pins torch 2.5.1 and adds opencv-python 5, unlike EXP0

export HF_HOME=$R/.hf
python - <<'EOF'
from huggingface_hub import snapshot_download, HfApi
api = HfApi()
for rid in ["yyfz233/Pi3", "facebook/VGGT-1B", "facebook/map-anything-apache"]:
    p = snapshot_download(rid)
    print("weights", rid, api.model_info(rid).sha, p, flush=True)
EOF
for r in map-anything vggt pi3; do echo "repo $r $(git -C $R/third_party/$r rev-parse HEAD)"; done
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"
echo "== $(date -u +%FT%TZ) SETUP_ENV_DONE"
