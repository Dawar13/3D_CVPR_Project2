#!/bin/bash
# EXP2 2D on real data beyond TUM: ADVIO (RS, unknown readout; slope test) and ETH3D SLAM (GS; no-harm reference).
set -u
source ~/rs3d/src/vm/lib.sh  # waits for inputs (VM bootstrap); sets HF_HOME, thread caps
wait_for $L/setup_env.log SETUP_ENV_DONE $L/dl_advio.log DL_advio_DONE $L/dl_eth3d.log DL_eth3d_DONE
G="bash $HOME/rs3d/src/gpu_slot.sh"  # every GPU step holds a slot (gpu_slot.sh)
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
  k=$([ $ds = advio ] && echo 2 || echo 3)  # D14: every 2nd (ADVIO) / 3rd (ETH3D) window per sequence
  python src/thin_clips.py $R/clips_$cam.csv $k $R/clips_${cam}_thin.csv
  for m in $MODELS; do
    $G python src/infer2.py --model $m --clips $R/clips_${cam}_thin.csv --out $R/preds --name $cam > $L/real_${ds}_$m.log 2>&1 \
      && cp $R/preds/${m}_$cam.npz $R/preds/$m.npz || log "$ds $m FAILED"
  done
  RS3D_RES=$R python src/evaluate2.py score --models $(echo $MODELS | tr ' ' ',') --conds $cam \
    --out $R/per_window.csv > $L/real_${ds}_score.log 2>&1
  log "$ds scored: $(wc -l < $R/per_window.csv) rows"
done
log "REAL_GAP_DONE"
