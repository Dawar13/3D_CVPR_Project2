#!/bin/bash
# EXP2 no-harm (2G-11) and real-phone extras, after the main runs:
#   TUM    SE2P_<m>_onGS: self-estimate on the GS camera's frames (assumes the RS readout)
#   ETH3D  SE2P_<m>_r30: self-estimate on GS frames assuming 30 ms top->bottom
#   synth  se2p(t)-<m>-ongs on the core_test GS twins
#   ADVIO  gyroscope straightening over assumed readout {10,20,30} ms x direction {lr,rl}; self-estimate at 30 ms
set -u
G="bash $HOME/rs3d/src/gpu_slot.sh"
source ~/rs3d/.venv/bin/activate
cd ~/rs3d
export HF_HOME=~/rs3d/.hf HF_HUB_OFFLINE=1
L=~/rs3d/logs
MODELS="pi3 vggt"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $L/noharm.log; }

# --- TUM (needs stage-1 GS predictions)
T=~/rs3d/results/exp2/gap/tum
for m in $MODELS; do
  c=SE2P_${m}_onGS
  RS3D_RES=$T $G python src/tum_conditions.py $c > $L/noharm_tum_$c.log 2>&1 || log "tum $c FAILED"
  RS3D_RES=$T $G python src/infer.py --model $m --cams $c --tag _$c >> $L/noharm_tum_$c.log 2>&1 || log "tum infer $c FAILED"
done
RS3D_RES=$T python src/evaluate2.py score --models pi3,vggt --conds GS,SE2P_pi3_onGS,SE2P_vggt_onGS \
  --out $T/per_window_noharm.csv > $L/noharm_tum_score.log 2>&1
log "tum no-harm scored"

# --- ETH3D and ADVIO (need run_real_gap.sh)
until grep -q REAL_GAP_DONE $L/real_gap.log 2>/dev/null; do sleep 120; done
for ds in eth3d advio; do
  R=~/rs3d/results/exp2/gap/$ds
  if [ $ds = eth3d ]; then conds="SE2P_pi3_r30 SE2P_vggt_r30"; gt="--gt GS"
  else conds="GYRO_r10_lr GYRO_r20_lr GYRO_r30_lr GYRO_r10_rl GYRO_r20_rl GYRO_r30_rl SE2P_pi3_r30_lr SE2P_pi3_r30_rl SE2P_vggt_r30_lr SE2P_vggt_r30_rl"; gt=""; fi
  all=""
  for c in $conds; do
    $G python src/real_conditions.py $ds $c > $L/noharm_${ds}_$c.log 2>&1 || { log "$ds $c FAILED"; continue; }
    for m in $MODELS; do
      case $c in SE2P_*) [[ $c == SE2P_${m}_* ]] || continue;; esac
      $G python src/infer2.py --model $m --clips $R/clips_$c.csv --out $R/preds --name $c >> $L/noharm_${ds}_$c.log 2>&1 \
        || log "$ds $m $c FAILED"
    done
    all="$all,$c"
  done
  cam=$([ $ds = advio ] && echo RS || echo GS)
  RS3D_RES=$R python src/evaluate2.py score --models pi3,vggt --conds $cam$all $gt --out $R/per_window_conds.csv \
    > $L/noharm_${ds}_score.log 2>&1
  log "$ds conditions scored"
done

# --- synthetic GS twins (needs run_synth_core.sh)
until grep -q SYNTH_CORE_DONE $L/synth_core.log 2>/dev/null; do sleep 120; done
cd ~/rs3d/src
S=~/rs3d/data/synth/core_test; R=~/rs3d/results/exp2/synth/core_test; P=$R/preds
for m in $MODELS; do
  $G python synth_conditions.py --set $S --conds se2p-$m-ongs,se2pt-$m-ongs --preds $P > $L/noharm_synth_$m.log 2>&1
  python synth_clips.py --set $S --conds se2p-$m-ongs:d,se2pt-$m-ongs:d --out $R/clips_ongs_$m.csv > /dev/null
  $G python infer2.py --model $m --clips $R/clips_ongs_$m.csv --out $P --name ongs >> $L/noharm_synth_$m.log 2>&1
done
python synth_eval.py --set $S --preds $P --models pi3,vggt --conds gs,se2p-pi3-ongs,se2pt-pi3-ongs,se2p-vggt-ongs,se2pt-vggt-ongs \
  --dense --out $R/per_clip_noharm.csv > $L/noharm_synth_eval.log 2>&1
log "NOHARM_DONE"
