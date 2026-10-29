#!/bin/bash
# EXP2 2D on real data beyond TUM: ADVIO (RS, unknown readout; slope test) and ETH3D SLAM (GS; no-harm reference).
set -u
source ~/rs3d/.venv/bin/activate
cd ~/rs3d
export HF_HOME=~/rs3d/.hf HF_HUB_OFFLINE=1
L=~/rs3d/logs
MODELS=${MODELS:-"pi3 vggt mapanything"}
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $L/real_gap.log; }
for ds in advio eth3d; do
  python src/real_windows.py $ds > $L/real_windows_$ds.log 2>&1 || { log "$ds windows FAILED"; continue; }
  log "$ds: $(tail -n 1 $L/real_windows_$ds.log)"
done
for ds in advio eth3d; do
  R=~/rs3d/results/exp2/gap/$ds
  cam=$([ $ds = advio ] && echo RS || echo GS)
  for m in $MODELS; do
    python src/infer2.py --model $m --clips $R/clips_$cam.csv --out $R/preds --name $cam > $L/real_${ds}_$m.log 2>&1 \
      && cp $R/preds/${m}_$cam.npz $R/preds/$m.npz || log "$ds $m FAILED"
  done
  RS3D_RES=$R python src/evaluate2.py score --models $(echo $MODELS | tr ' ' ',') --conds $cam \
    --out $R/per_window.csv > $L/real_${ds}_score.log 2>&1
  log "$ds scored: $(wc -l < $R/per_window.csv) rows"
done
log "REAL_GAP_DONE"
