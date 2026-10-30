#!/bin/bash
# EXP2 2D on Aria Digital Twin (real RS, 5 ms readout; slope test, DEVIATIONS D18): windows, then pi3 / VGGT /
# MapAnything in parallel (dense output kept for the secondary depth metrics), then scoring.
# The extra models run on the same clip list from src/vm/run_models.sh.
source ~/rs3d/src/vm/lib.sh
cd $R
export HF_HUB_OFFLINE=1
A=$R/results/exp2/gap/adt
mkdir -p $A
log() { echo "$(stamp) $*" | tee -a $L/adt.log; }
wait_for $L/adt_download.log ADT_MAIN_DONE $L/setup_env.log SETUP_ENV_DONE
envs/aria/bin/python src/adt_prep.py --workers $(( $(nproc) / 4 + 1 )) > $L/adt_prep.log 2>&1 || { log "prep FAILED"; exit 1; }
log "prep: $(tail -n 1 $L/adt_prep.log)"
source .venv/bin/activate
for m in pi3 vggt mapanything; do
  ( $G python src/infer2.py --model $m --clips $A/clips_RS.csv --out $A/preds --name RS --dense > $L/adt_$m.log 2>&1 \
      && cp $A/preds/${m}_RS.npz $A/preds/$m.npz || log "$m FAILED" ) &
done
wait
RS3D_RES=$A python src/evaluate2.py score --models pi3,vggt,mapanything --conds RS --out $A/per_window.csv > $L/adt_score.log 2>&1
log "ADT_DONE $(wc -l < $A/per_window.csv) rows"
