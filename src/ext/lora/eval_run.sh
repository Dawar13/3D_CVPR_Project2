#!/bin/bash
# Evaluate one LoRA checkpoint (or the base model) on the synthetic core test set and on TUM RS-VIO.
#   usage: eval_run.sh <tag> <pi3|vggt> <ckpt.pt|base>
#   SYNTH_PREDS=<dir> SYNTH_MODEL=<name>: score existing synthetic predictions instead of running them (base models:
#   the main runs' core_test preds, verified bit-identical to this runner with LORA_CKPT=base, see eval/base_check)
# writes ~/rs3d/results/exp2/training/eval/<tag>/{synth/,tum/,synth_eval.csv,tum_windows.csv,eval_config.json}
# GPU work goes through gpu_slot.sh with the helper's own lock dir (1 slot, >= 14 GB free). Resumable.
# TUM: the fixed subset in eval/clips_tum.csv (every 4th kept window, both cameras; see score_lora.py tum_clips).
set -u
TAG=$1; M=$2; CK=$3
source ~/rs3d/.venv/bin/activate
export HF_HOME=~/rs3d/.hf HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=~/rs3d/ext/lora RS3D_EXTRA_RUNNERS=models_lora LORA_CKPT=$CK
SLOT="env RS3D_LOCK_DIR=$HOME/rs3d/locks_lora RS3D_GPU_SLOTS=1 bash $HOME/rs3d/src/gpu_slot.sh"
E=~/rs3d/results/exp2/training/eval
O=$E/$TAG
L=~/rs3d/logs
mkdir -p $O/synth $O/tum
log() { echo "$(date -u +%FT%TZ) [eval $TAG] $*" | tee -a $L/lora_eval.log; }
SP=${SYNTH_PREDS:-$O/synth}
SM=${SYNTH_MODEL:-${M}_lora}
echo "{\"tag\": \"$TAG\", \"model\": \"$M\", \"ckpt\": \"$CK\", \"synth_preds\": \"$SP\", \"synth_model\": \"$SM\", \"git_commit\": \"$(git -C ~/rs3d rev-parse HEAD)\"}" > $O/eval_config.json
log "start ($M, $CK)"
if [ -z "${SYNTH_PREDS:-}" ]; then
  $SLOT python ~/rs3d/src/infer2.py --model ${M}_lora --clips $E/clips_core_test.csv --out $O/synth --name core_test \
      > $L/lora_eval_${TAG}_synth.log 2>&1 || { log "synth infer FAILED"; exit 1; }
fi
$SLOT python ~/rs3d/src/infer2.py --model ${M}_lora --clips $E/clips_tum.csv --out $O/tum --name tum \
    > $L/lora_eval_${TAG}_tum.log 2>&1 || { log "tum infer FAILED"; exit 1; }
log "inference done"
python ~/rs3d/src/synth_eval.py --set ~/rs3d/data/synth/core_test --preds $SP --models $SM --conds rs,gs \
    --dense --workers 6 --out $O/synth_eval.csv > $L/lora_eval_${TAG}_score.log 2>&1 || { log "synth_eval FAILED"; exit 1; }
python ~/rs3d/ext/lora/score_lora.py tum_score --preds $O/tum/${M}_lora_tum.npz --out $O/tum_windows.csv \
    >> $L/lora_eval_${TAG}_score.log 2>&1 || { log "tum score FAILED"; exit 1; }
log "EVAL_DONE"
