#!/bin/bash
# EXP2 2F learned 2D unrollers -> 3D model (scope as on the first VM; DEVIATIONS D19 records it):
#   TUM, every kept window: DFRSC-3F with its Fastec-RS and BS-RSC weights (the best released model on the authors'
#     own benchmark reproduction, src/ext/unroll2d/ENV.md section 5) and DFRSC-3F fine-tuned on our synthetic train
#     split; then pi3 and VGGT on each
#   synthetic core test: 12 released method/weight pairs + the fine-tuned model, then pi3 and VGGT (dense)
# One sequential queue on its own GPU slot (lock dir locks_u2d). Resumable. Log: logs/u2d.log (+ unroll2d_queue.log)
source ~/rs3d/src/vm/lib.sh
cd $R
U=$R/ext/unroll2d
B=$R/results/exp2/baselines/unroll2d
FT=$R/weights/dfrsc_ft/dfrsc3f-ft/best.pth
SLOT="env RS3D_LOCK_DIR=$R/locks_u2d RS3D_GPU_SLOTS=1 bash $R/src/gpu_slot.sh"
mkdir -p $B
log() { echo "$(stamp) $*" | tee -a $L/u2d.log; }
q() {  # one queue job (jobs.sh helpers), logged with its exit code
  echo "=== START $(stamp) :: $*" >> $L/unroll2d_queue.log
  $SLOT bash -c "source $U/jobs.sh; $*" >> $L/unroll2d_queue.log 2>&1
  local rc=$?
  echo "=== END rc=$rc $(stamp) :: $*" >> $L/unroll2d_queue.log
  [ $rc -eq 0 ] || log "FAILED: $*"
}

wait_for $L/setup_extra.log ENV_u2d_DONE $L/exp2_stage1.log "prep done"
(source .venv/bin/activate && python $U/prep_nb.py > $L/unroll2d_prep_nb.log 2>&1) || log "prep_nb FAILED"
log "neighbours prepped"
for w in fastecrs bsrsc; do
  q tum dfrsc3f $w
  q tum3d pi3 U2D-dfrsc3f-$w
  q tum3d vggt U2D-dfrsc3f-$w
done
wait_for $L/gen_core_test_test.log GENERATE_DONE
for mw in dfrsc3f:fastecrs dfrsc3f:bsrsc dfrsc3f:carlars dfrsc2f:fastecrs dfrsc2f:bsrsc jamnet:fastec jamnet:bsrsc \
          lbcnet:fastec lbcnet:bsrsc adarsc:fastecrs dun:fastec jcd:fastec; do
  q synth ${mw%%:*} ${mw##*:}
done
wait_for $L/gen_core_trainval_val.log GENERATE_DONE $L/gen_core_trainval_train.log GENERATE_DONE
q 'cd $U && $PY finetune_dfrsc.py --init bsrsc --frames 3 --name dfrsc3f-ft --iters 6000 --val_every 500 --val_max 240 --patience 4 --batch 8 --workers 8'
q synth dfrsc3f bsrsc --gamma_from_meta --ckpt $FT --name u2d-dfrsc3f-ft --net_h 512
q tum dfrsc3f bsrsc --gamma 0.6 --scale 0.5 --ckpt $FT --cond U2D-dfrsc3f-ft
q tum3d pi3 U2D-dfrsc3f-ft
q tum3d vggt U2D-dfrsc3f-ft
wait_for $L/synth_core.log SYNTH_CORE_DONE  # synth3d_all writes into the core test's preds folder
q synth3d_all pi3,vggt

# --- scoring (CPU)
source .venv/bin/activate
C=$(ls $R/results/exp2/gap/tum/preds/pi3_U2D-*.npz 2>/dev/null | sed 's#.*/pi3_##; s#\.npz##' | paste -sd,)
[ -n "$C" ] && RS3D_RES=$R/results/exp2/gap/tum python src/evaluate2.py score --models pi3,vggt --conds $C \
  --out $B/tum_per_window_u2d.csv > $L/u2d_score.log 2>&1
S=$(ls -d $R/data/synth/core_test/test/*/*/u2d-* 2>/dev/null | xargs -n1 basename | sort -u | paste -sd,)
[ -n "$S" ] && (cd src && python synth_eval.py --set $R/data/synth/core_test --preds $R/results/exp2/synth/core_test/preds \
  --models pi3,vggt --conds rs,gs,$S --dense --out $B/synth_core_test_u2d.csv >> $L/u2d_score.log 2>&1)
$R/envs/u2d/bin/python $U/summarize_synth2d.py >> $L/u2d_score.log 2>&1
log "U2D_DONE"
