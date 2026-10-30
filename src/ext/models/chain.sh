#!/bin/bash
# Sequential GPU queue for the extra models (scope = DEVIATIONS D16): one GPU job at a time, each acquiring our own
# single slot (RS3D_LOCK_DIR=~/rs3d/locks_models, RS3D_GPU_SLOTS=1) through src/gpu_slot.sh (>= 14 GB free).
# infer2.py resumes from saved npz files, so the chain can be killed and restarted from job i.
# Per-job logs: logs/models_<model>_<set>.log; chain log: logs/models_chain.log
# usage: bash ext/models/chain.sh [first_job_index]
cd ~/rs3d
export HF_HOME=~/rs3d/.hf HF_HUB_OFFLINE=1 PYTHONPATH=~/rs3d/ext/models
G=~/rs3d/results/exp2/gap
SYN=~/rs3d/results/exp2/synth/core_test/preds_newmodels
C=~/rs3d/ext/models/clips

job() {  # job <model> <env> <clips> <out> <name> <logtag> [post-copy src dst]
  local M=$1 E=$2 CL=$3 O=$4 N=$5 T=$6
  echo "=== $(date) START $M $T ($CL)"
  env RS3D_LOCK_DIR=$HOME/rs3d/locks_models RS3D_GPU_SLOTS=1 bash ~/rs3d/src/gpu_slot.sh \
      envs/$E/bin/python src/infer2.py --model $M --clips $CL --out $O --name $N >> logs/models_${M}_${T}.log 2>&1
  local rc=$?
  echo "=== $(date) END $M $T rc=$rc"
  if [ $rc -eq 0 ] && [ $# -ge 8 ]; then cp "$7" "$8" && echo "copied $7 -> $8"; fi
  return $rc
}

JOBS=(
  "da3 da3 $G/advio/clips_RS_thin.csv $G/advio/preds RS advio $G/advio/preds/da3_RS.npz $G/advio/preds/da3.npz"
  "da3 da3 $G/eth3d/clips_GS_thin.csv $G/eth3d/preds GS eth3d $G/eth3d/preds/da3_GS.npz $G/eth3d/preds/da3.npz"
  "da3 da3 $C/synth_core_test.csv $SYN main synth"
  "streamvggt streamvggt $C/tum_gsrs_thin2.csv $G/tum/preds gsrs tum $G/tum/preds/streamvggt_gsrs.npz $G/tum/preds/streamvggt.npz"
  "streamvggt streamvggt $C/control.csv $G/tum/preds control tum_control"
  "streamvggt streamvggt $C/synth_core_test.csv $SYN main synth"
)
for ((i=${1:-0}; i<${#JOBS[@]}; i++)); do
  echo "--- job $i: ${JOBS[$i]}"
  job ${JOBS[$i]}
done
echo "=== $(date) CHAIN_DONE"
