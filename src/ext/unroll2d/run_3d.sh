#!/bin/bash
# usage: run_3d.sh <log> <model> <clips-suffix ("" or _smoke)> COND [COND ...]
# Runs src/infer2.py (main env) for one 3D model on each U2D condition's clip list; waits for >= 14 GB free VRAM
# before each launch (shared GPU rule) and retries if infer2 refuses to start.
LOG=$1; M=$2; SUF=$3; shift 3
cd ~/rs3d && source .venv/bin/activate
export HF_HOME=~/rs3d/.hf HF_HUB_OFFLINE=1 RS3D_RES=~/rs3d/results/exp2/gap/tum
for C in "$@"; do
  CL=~/rs3d/results/exp2/baselines/unroll2d/clips/$C$SUF.csv
  for try in $(seq 1 200); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$FREE" -ge 14000 ]; then
      echo "=== $M $C ($CL) free=${FREE}MiB $(date)" >> "$LOG"
      python src/infer2.py --model $M --clips $CL --out ~/rs3d/results/exp2/gap/tum/preds --name $C >> "$LOG" 2>&1
      grep -q "INFER_DONE" <(tail -n 3 "$LOG") && break
    fi
    sleep 60
  done
done
echo "LIST3D_DONE $M $(date)" >> "$LOG"
