#!/bin/bash
# EXP2 extra models (DEVIATIONS D16, extended by D19), each in its own env and its own sequential chain:
#   da3, omnivggt, vggt_omega (only once the user's HF token is in ~/rs3d/.hf/token):
#     TUM GS/RS + GS-GS control + oracle V3/V4, ADVIO and ETH3D (D14 thinned windows), Waymo, ADT, synthetic core
#     test rs/gs/v3/v4 (dense)
#   streamvggt (D16 scope, 10-15 s per clip): TUM every 2nd window + control, synthetic core test rs/gs
# The chains share two GPU slots (lock dir locks_models). Each job waits for its inputs; infer2 resumes from saved
# npz files, so a restart only redoes unfinished clips. Scoring at the end. Log: logs/models.log
source ~/rs3d/src/vm/lib.sh
cd $R
export HF_HUB_OFFLINE=1 PYTHONPATH=$R/src/ext/models
SLOT="env RS3D_LOCK_DIR=$R/locks_models RS3D_GPU_SLOTS=2 bash $R/src/gpu_slot.sh"
GP=$R/results/exp2/gap
C=$R/results/exp2/clips
SC=$R/results/exp2/synth/core_test
SYN=$SC/preds_newmodels
mkdir -p $C $SYN
log() { echo "$(stamp) $*" | tee -a $L/models.log; }

job() {  # job <ds> <model> <clips.csv> <outdir> <name> [copy-to]: infer2 once the clip list exists;
         # skipped if ~/rs3d/skip_<ds> exists (a dataset that will not arrive, e.g. Waymo without the user's login)
  local ds=$1 m=$2 cl=$3 o=$4 n=$5 cp_to=${6:-}
  until [ -e $cl ] || [ -e $R/skip_$ds ]; do sleep 60; done
  [ -e $cl ] || { log "$m $ds skipped (skip_$ds)"; return; }
  if $SLOT envs/$m/bin/python src/infer2.py --model $m --clips $cl --out $o --name $n >> $L/models_${m}_$n.log 2>&1; then
    [ -n "$cp_to" ] && cp $o/${m}_$n.npz $cp_to
    log "$m $n done ($(basename $cl))"
  else
    log "$m $n FAILED (see logs/models_${m}_$n.log)"
  fi
}

# --- clip lists (CPU), each as soon as its windows exist
(
  source .venv/bin/activate
  wait_for $L/exp2_stage1.log "prep done"
  python src/vm/make_clips.py --cams GS,RS --name gsrs --out $C/tum_gsrs.csv
  python src/vm/make_clips.py --cams GS,RS --name gsrs --thin 2 --out $C/tum_gsrs_thin2.csv
  python src/vm/make_clips.py --cams GS,RS --name control --set control --out $C/tum_control.csv
  wait_for $L/exp2_tum2.log "oracle conditions:"
  python src/vm/make_clips.py --cams V3,V4 --out $C/tum_v3v4.csv
  wait_for $L/synth_core_cond.log SYNTH_COND_DONE
  (cd src && python synth_clips.py --set $R/data/synth/core_test --conds rs:d,gs:d,v3:d,v4:d --out $C/synth_core.csv)
  (cd src && python synth_clips.py --set $R/data/synth/core_test --conds rs:d,gs:d --out $C/synth_core_rsgs.csv)
) >> $L/models_clips.log 2>&1 &

chain() {  # the full D19 scope for one model
  local m=$1
  wait_for $L/setup_extra.log "ENV_${m}_DONE"
  job tum $m $C/tum_gsrs.csv $GP/tum/preds gsrs $GP/tum/preds/$m.npz
  job tum $m $C/tum_control.csv $GP/tum/preds control
  job advio $m $GP/advio/clips_RS_thin.csv $GP/advio/preds RS $GP/advio/preds/$m.npz
  job eth3d $m $GP/eth3d/clips_GS_thin.csv $GP/eth3d/preds GS $GP/eth3d/preds/$m.npz
  job tum $m $C/tum_v3v4.csv $GP/tum/preds v3v4
  job synth $m $C/synth_core.csv $SYN main
  job adt $m $GP/adt/clips_RS.csv $GP/adt/preds RS $GP/adt/preds/$m.npz
  job waymo $m $GP/waymo/clips_RS.csv $GP/waymo/preds RS $GP/waymo/preds/$m.npz
  log "chain $m done"
}

chain da3 &
chain omnivggt &
(
  wait_for $L/setup_extra.log ENV_streamvggt_DONE
  job tum streamvggt $C/tum_gsrs_thin2.csv $GP/tum/preds gsrs $GP/tum/preds/streamvggt.npz
  job tum streamvggt $C/tum_control.csv $GP/tum/preds control
  job synth streamvggt $C/synth_core_rsgs.csv $SYN main
  log "chain streamvggt done"
) &
# VGGT-Omega is gated: it joins only when the user's token is in place (checked every 10 min)
(
  until [ -s $R/.hf/token ] || [ -e $R/skip_vggt_omega ]; do sleep 600; done
  [ -e $R/skip_vggt_omega ] && { log "vggt_omega skipped (no token)"; exit 0; }
  wait_for $L/setup_extra.log ENV_vggt_omega_DONE
  HF_HUB_OFFLINE=0 envs/vggt_omega/bin/python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('facebook/VGGT-Omega', 'vggt_omega_1b_512.pt', revision='1041e80fc0e911235d3426b0a3d9a81075111a53',
                token=open('$R/.hf/token').read().strip())" >> $L/models_vggt_omega_download.log 2>&1 \
    && chain vggt_omega || log "vggt_omega download FAILED (licence not accepted on HF yet?)"
) &
wait

# --- scoring (CPU): every model that has predictions
source .venv/bin/activate
have() { local out=""; for m in da3 omnivggt vggt_omega streamvggt; do [ -f $1/$m${2:-}.npz ] && out="$out,$m"; done; echo ${out#,}; }
M=$(have $GP/tum/preds)
[ -n "$M" ] && RS3D_RES=$GP/tum python src/evaluate2.py score --models $M --conds GS,RS --out $GP/tum/per_window_newmodels.csv >> $L/models_score.log 2>&1
M=$(have $GP/tum/preds _control)
[ -n "$M" ] && RS3D_RES=$GP/tum python src/evaluate2.py score --set control --models $M --conds GS,RS \
  --out $GP/tum/per_window_control_newmodels.csv >> $L/models_score.log 2>&1
M=$(have $GP/tum/preds _V3)
[ -n "$M" ] && RS3D_RES=$GP/tum python src/evaluate2.py score --models $M --conds V3,V4 --out $GP/tum/per_window_newmodels_oracle.csv >> $L/models_score.log 2>&1
for ds in advio waymo adt eth3d; do
  M=$(have $GP/$ds/preds)
  [ -z "$M" ] && continue
  cam=$([ $ds = eth3d ] && echo GS || echo RS); gt=$([ $ds = eth3d ] && echo "--gt GS" || echo "")
  RS3D_RES=$GP/$ds python src/evaluate2.py score --models $M --conds $cam $gt --out $GP/$ds/per_window_newmodels.csv >> $L/models_score.log 2>&1
done
mkdir -p $R/results/exp2/baselines/models
M=$(for m in da3 omnivggt vggt_omega streamvggt; do [ -f $SYN/${m}_main.npz ] && echo -n "$m,"; done); M=${M%,}
[ -n "$M" ] && (cd src && python synth_eval.py --set $R/data/synth/core_test --preds $SYN --models $M --conds rs,gs,v3,v4 \
  --dense --out $R/results/exp2/baselines/models/synth_core_test_newmodels.csv >> $L/models_score.log 2>&1)
log "MODELS_DONE"
