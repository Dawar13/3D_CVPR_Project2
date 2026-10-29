#!/bin/bash
# EXP2 2D on Waymo (real RS, known readout; slope test, DEVIATIONS D15): extract, infer pi3/vggt/mapanything, score.
set -u
G="bash $HOME/rs3d/src/gpu_slot.sh"
L=~/rs3d/logs
R=~/rs3d/results/exp2/gap/waymo
MODELS=${MODELS:-"pi3 vggt mapanything"}
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $L/waymo.log; }
cd ~/rs3d/src
PYTHONPATH=~/rs3d/ext/waymo_pb ~/rs3d/envs/waymo/bin/python waymo_prep.py > $L/waymo_prep.log 2>&1 || { log "prep FAILED"; exit 1; }
log "prep: $(tail -n 1 $L/waymo_prep.log)"
source ~/rs3d/.venv/bin/activate
export HF_HOME=~/rs3d/.hf HF_HUB_OFFLINE=1
for m in $MODELS; do
  $G python infer2.py --model $m --clips $R/clips_RS.csv --out $R/preds --name RS > $L/waymo_$m.log 2>&1 \
    && cp $R/preds/${m}_RS.npz $R/preds/$m.npz || log "$m FAILED"
done
RS3D_RES=$R python evaluate2.py score --models $(echo $MODELS | tr ' ' ',') --conds RS --out $R/per_window.csv \
  > $L/waymo_score.log 2>&1
log "WAYMO_DONE $(wc -l < $R/per_window.csv) rows"
