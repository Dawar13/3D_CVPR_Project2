#!/bin/bash
# memory / speed smoke tests (16 frames per sample, 12 steps, 4 val clips); outputs go to *_smoke
source ~/rs3d/.venv/bin/activate
cd ~/rs3d/ext/lora
export HF_HOME=~/rs3d/.hf HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for m in pi3 vggt; do
  python train_lora.py --config ~/rs3d/results/exp2/training/configs/smoke_${m}_16f.json --smoke 12 > ~/rs3d/logs/lora_smoke_$m.log 2>&1
done
