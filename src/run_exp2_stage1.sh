#!/bin/bash
# EXP2 stage 1 (VM): rebuild TUM windows, reproduce EXP0 for pi3 + VGGT (EXP2 §2.5) and stop on a mismatch.
# Then the GS-GS control set for the fairness adjustment (pre-registration P3).
set -u
source ~/rs3d/src/vm/lib.sh  # waits for inputs (VM bootstrap); sets HF_HOME, thread caps
wait_for $L/setup_env.log SETUP_ENV_DONE $L/dl_tum.log DL_TUM_DONE
source ~/rs3d/.venv/bin/activate
cd ~/rs3d
export HF_HOME=~/rs3d/.hf HF_HUB_OFFLINE=1 RS3D_RES=~/rs3d/results/exp2/gap/tum
L=~/rs3d/logs
mkdir -p $RS3D_RES
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $L/exp2_stage1.log; }
log "start; GPU: $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader)"

python src/prep.py > $L/e2_prep.log 2>&1 || { log "prep FAILED"; exit 1; }
python src/prep.py control > $L/e2_prep_control.log 2>&1 || { log "prep control FAILED"; exit 1; }
log "prep done: $(tail -2 $L/e2_prep.log | head -1)"

for m in pi3 vggt; do
  python src/infer.py --model $m > $L/e2_infer_$m.log 2>&1 &
done
wait
log "inference done: $(grep -h 's/clip' $L/e2_infer_*.log | tail -2 | tr '\n' ' ')"

python src/evaluate2.py score --models pi3,vggt --conds GS,RS --out $RS3D_RES/per_window_repro.csv \
  > $L/e2_score_repro.log 2>&1
python src/evaluate2.py repro --new $RS3D_RES --old ~/rs3d/results/exp0 > $L/e2_repro.log 2>&1
if ! grep -q REPRO_PASS $L/e2_repro.log; then
  log "REPRO_FAIL: stopping here (EXP2 §2.5). See $L/e2_repro.log"
  exit 1
fi
log "REPRO_PASS"

for m in pi3 vggt; do
  python src/infer.py --model $m --set control > $L/e2_infer_${m}_control.log 2>&1 &
done
wait
log "STAGE1_DONE"
