#!/bin/bash
# EXP2 on the synthetic core test set (Replica test scenes, 30 ms top->bottom):
#   2D   rs vs gs twin;  2E  v3, v4 and V5 (scored from rs dense output) + readout x{0.5,0.75,1.25,1.5} / wrong direction
#   2F   self-estimate two-pass se2p, se2pt (ordered) and se2p-shuf (unordered: pass 1 and 2 shuffled); rsshuf = plain
#        RS as an unordered set
# Models run in parallel; every step is resumable.
set -u
source ~/rs3d/.venv/bin/activate
cd ~/rs3d/src
export HF_HOME=~/rs3d/.hf HF_HUB_OFFLINE=1
L=~/rs3d/logs
S=~/rs3d/data/synth/core_test
R=~/rs3d/results/exp2/synth/core_test
P=$R/preds
MODELS=${MODELS:-"pi3 vggt mapanything"}
mkdir -p $P
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $L/synth_core.log; }
until grep -q GENERATE_DONE $L/gen_core_test_test.log 2>/dev/null; do sleep 60; done
log "core_test generated: $(tail -n 1 $L/gen_core_test_test.log)"

SENS="v3_r0.5,v3_r0.75,v3_r1.25,v3_r1.5,v4_r0.5,v4_r0.75,v4_r1.25,v4_r1.5,v3_dir,v4_dir"
python synth_conditions.py --set $S --conds v3,v4,$SENS > $L/synth_core_cond.log 2>&1 || log "oracle conditions FAILED"
python synth_clips.py --set $S --conds rs:d,gs:d,v3:d,v4:d,rsshuf=rs:s,$SENS --out $R/clips_main.csv

chain() {
  m=$1
  python infer2.py --model $m --clips $R/clips_main.csv --out $P --name main > $L/synth_core_infer_$m.log 2>&1 \
    || log "infer $m main FAILED"
  python synth_conditions.py --set $S --conds se2p-$m,se2pt-$m,se2p-$m-shuf --preds $P > $L/synth_core_se2p_$m.log 2>&1 \
    || log "se2p conditions $m FAILED"
  python synth_clips.py --set $S --conds se2p-$m:d,se2pt-$m:d,se2p-$m-shuf:ds --out $R/clips_se2p_$m.csv
  python infer2.py --model $m --clips $R/clips_se2p_$m.csv --out $P --name se2p > $L/synth_core_infer_se2p_$m.log 2>&1 \
    || log "infer $m se2p FAILED"
  log "chain $m done"
}
for m in $MODELS; do chain $m & done
wait

C="rs,gs,v3,v4,rsshuf,$SENS"
for m in $MODELS; do C="$C,se2p-$m,se2pt-$m,se2p-$m-shuf"; done
python synth_eval.py --set $S --preds $P --models $(echo $MODELS | tr ' ' ',') --conds $C --dense --v5 \
  --out $R/per_clip.csv > $L/synth_core_eval.log 2>&1
log "SYNTH_CORE_DONE $(tail -n 1 $L/synth_core_eval.log)"
