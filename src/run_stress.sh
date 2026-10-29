#!/bin/bash
# EXP2 2G stress tests on the synthetic stress sets (paired with core_test clips 0..14 per scene) and on post-capture
# variants of core_test. Per set: oracle v3/v4 (+V5 from rs dense), self-estimate se2p/se2pt, models pi3, vggt,
# mapanything; then synth_eval. Each set starts when its generation log says GENERATE_DONE. Resumable.
set -u
G="bash $HOME/rs3d/src/gpu_slot.sh"  # every GPU step holds a slot (gpu_slot.sh)
source ~/rs3d/.venv/bin/activate
cd ~/rs3d/src
export HF_HOME=~/rs3d/.hf HF_HUB_OFFLINE=1
L=~/rs3d/logs
MODELS=${MODELS:-"pi3 vggt mapanything"}
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $L/stress.log; }

run_set() {  # $1 = set name
  st=$1; S=~/rs3d/data/synth/$st; R=~/rs3d/results/exp2/stress/$st; P=$R/preds; mkdir -p $P
  until grep -q GENERATE_DONE $L/gen_${st}_test.log 2>/dev/null; do sleep 120; done
  $G python synth_conditions.py --set $S --conds v3,v4 > $L/stress_${st}_cond.log 2>&1 || log "$st conditions FAILED"
  conds="rs:d,gs:d,v3:d,v4:d"
  python synth_clips.py --set $S --conds $conds --out $R/clips_main.csv > /dev/null
  for m in $MODELS; do
    if [ $st = st_views64 ]; then
      for nv in 2 4 8 16 32 64; do
        $G python infer2.py --model $m --clips $R/clips_main.csv --out $R/preds_v$nv --name main --views $nv \
          > $L/stress_${st}_${m}_v$nv.log 2>&1 || log "$st $m v$nv FAILED"
      done
    else
      $G python infer2.py --model $m --clips $R/clips_main.csv --out $P --name main > $L/stress_${st}_$m.log 2>&1 \
        || log "$st $m FAILED"
      $G python synth_conditions.py --set $S --conds se2p-$m,se2pt-$m --preds $P >> $L/stress_${st}_cond.log 2>&1
      python synth_clips.py --set $S --conds se2p-$m:d,se2pt-$m:d --out $R/clips_se2p_$m.csv > /dev/null
      $G python infer2.py --model $m --clips $R/clips_se2p_$m.csv --out $P --name se2p >> $L/stress_${st}_$m.log 2>&1
    fi
  done
  M=$(echo $MODELS | tr ' ' ',')
  if [ $st = st_views64 ]; then
    for nv in 2 4 8 16 32 64; do
      python synth_eval.py --set $S --preds $R/preds_v$nv --models $M --conds rs,gs,v3,v4 --dense --v5 \
        --out $R/per_clip_v$nv.csv > $L/stress_${st}_eval_v$nv.log 2>&1
    done
  else
    C="rs,gs,v3,v4"; for m in $MODELS; do C="$C,se2p-$m,se2pt-$m"; done
    python synth_eval.py --set $S --preds $P --models $M --conds $C --dense --v5 --out $R/per_clip.csv \
      > $L/stress_${st}_eval.log 2>&1
  fi
  log "$st done"
}

# post-capture variants of core_test (needs core_test v3 from run_synth_core.sh)
post() {
  S=~/rs3d/data/synth/core_test; R=~/rs3d/results/exp2/stress/post; P=$R/preds; mkdir -p $P
  until [ -n "$(ls -d $S/test/room_0/00014/v3 2>/dev/null)" ] && grep -q "SYNTH_COND_DONE" $L/synth_core_cond.log 2>/dev/null; do sleep 120; done
  $G python -m rsgen.post --set $S --max_clip 14 > $L/stress_post.log 2>&1 || log "post FAILED"
  python synth_clips.py --set $S --max_clip 14 --out $R/clips.csv --conds \
rs:d,gs:d,rs_noise:d,gs_noise:d,rs_lowlight:d,gs_lowlight:d,rs_gamma:d,gs_gamma:d,rs_crop,gs_crop,rs_cropzoom,gs_cropzoom,v3,v3_crop,v3_cropnaive,rs_eis,gs_eis \
    > /dev/null
  for m in $MODELS; do
    $G python infer2.py --model $m --clips $R/clips.csv --out $P --name post > $L/stress_post_$m.log 2>&1 || log "post $m FAILED"
  done
  python synth_eval.py --set $S --preds $P --models $(echo $MODELS | tr ' ' ',') --dense --out $R/per_clip.csv --conds \
rs,gs,rs_noise,gs_noise,rs_lowlight,gs_lowlight,rs_gamma,gs_gamma,rs_crop,gs_crop,rs_cropzoom,gs_cropzoom,v3,v3_crop,v3_cropnaive,rs_eis,gs_eis \
    > $L/stress_post_eval.log 2>&1
  log "post done"
}
post
for st in st_readout0 st_readout5 st_readout10 st_readout20 st_readout40 st_dir_bt st_dir_lr st_objects_r0 st_objects_r30 \
          st_lowtex st_blur_r0_e16 st_blur_r30_e8 st_blur_r30_e16 st_stride1 st_stride6 st_stride12 st_views64; do
  run_set $st
done
log "STRESS_DONE"
