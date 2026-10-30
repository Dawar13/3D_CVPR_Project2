#!/bin/bash
# Train the given runs one at a time (shared-GPU rule: one training process at a time, each waits for >= 35 GB free),
# then evaluate each run's best checkpoint in the background (eval_run.sh, through gpu_slot.sh).
#   usage: run_queue.sh <run> [<run> ...]      configs: ~/rs3d/results/exp2/training/configs/<run>.json
#   EVAL=0 skips the evaluation (pilots)
set -u
source ~/rs3d/.venv/bin/activate
export HF_HOME=~/rs3d/.hf HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/rs3d/ext/lora
T=~/rs3d/results/exp2/training
Q=~/rs3d/logs/lora_queue.log
for run in "$@"; do
  if [ -f $T/$run/run.json ]; then
    echo "$(date -u +%FT%TZ) $run already trained" >> $Q
  else
    echo "$(date -u +%FT%TZ) $run training" >> $Q
    python train_lora.py --config $T/configs/$run.json > ~/rs3d/logs/lora_$run.log 2>&1 \
      && echo "$(date -u +%FT%TZ) $run trained" >> $Q || echo "$(date -u +%FT%TZ) $run TRAIN FAILED" >> $Q
  fi
  if [ "${EVAL:-1}" = 1 ] && [ -f $T/$run/run.json ] && [ ! -f $T/eval/$run/tum_windows.csv ]; then
    m=$(python -c "import json; print(json.load(open('$T/configs/$run.json'))['model'])")
    nohup bash eval_run.sh $run $m ~/rs3d/weights/lora/$run/best.pt > /dev/null 2>&1 &
    echo "$(date -u +%FT%TZ) $run eval launched" >> $Q
  fi
done
wait
echo "$(date -u +%FT%TZ) QUEUE_DONE $*" >> $Q
