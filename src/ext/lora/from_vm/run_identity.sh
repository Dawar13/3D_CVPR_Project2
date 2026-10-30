#!/bin/bash
# step-0 identity checks for both models, one after the other; each waits for >= 30 GB free GPU memory
# (training-type process: not run through gpu_slot.sh, which is for evaluation inference)
source ~/rs3d/.venv/bin/activate
cd ~/rs3d/ext/lora
export HF_HOME=~/rs3d/.hf HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for m in pi3 vggt; do
  until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -ge 30720 ]; do sleep 30; done
  python check_identity.py --model $m > ~/rs3d/logs/lora_identity_$m.log 2>&1
done
