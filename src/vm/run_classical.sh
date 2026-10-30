#!/bin/bash
# EXP2 2F classical baselines (src/ext/classical/ENV.md):
#   COLMAP 4.2.1 incremental + its built-in GLOMAP global SfM on every TUM window, GS and RS, CPU-only SIFT
#     (run_sfm3.py; low CPU priority so the synthetic generator and the GPU jobs' loaders come first)
#   Barath RS relative pose (7AC / 5PC, RoMa matches) on the first 81 windows of its fixed random order (D17),
#     consecutive pairs, on its own GPU slot (lock dir locks_classical)
# Log: logs/classical.log
source ~/rs3d/src/vm/lib.sh
cd $R
C=$R/ext/classical
B=$R/results/exp2/baselines/classical
mkdir -p $B
log() { echo "$(stamp) $*" | tee -a $L/classical.log; }
wait_for $L/exp2_stage1.log "prep done" $L/dl_tum.log DL_TUM_DONE

# the repo loader's dataset layout, as a symlink tree (no copy)
mkdir -p $C/tum_rs_link/euroc $C/tum_rs_link/calibration
for d in $R/data/tumrs/dataset-seq*; do n=$(basename $d); ln -sfn $d/$n $C/tum_rs_link/euroc/$n; done
ln -sf $R/data/tumrs/camchain-calibration-equidistant4_camimu_dataset-calib-imu1.yaml $C/tum_rs_link/calibration/

( wait_for $L/setup_extra.log ENV_rsmp_
  if grep -q ENV_rsmp_DONE $L/setup_extra.log; then
    export TORCH_HOME=$R/cache/torch
    env RS3D_LOCK_DIR=$R/locks_classical RS3D_GPU_SLOTS=1 bash $R/src/gpu_slot.sh bash -c "
      cd $C && $R/envs/rsmp/bin/python run_repro.py --sequences 1 --output $B/barath/repro/results_rs_tum_seq1.csv \
        > $L/classical_barath_repro.log 2>&1
      $R/envs/rsmp/bin/python barath_windows.py --conds RS --gaps 1 --workers 3 --limit_windows 81 \
        > $L/classical_barath_windows.log 2>&1" || log "Barath FAILED"
    (cd $C && $R/envs/rsmp/bin/python eval_barath.py --cond RS > $L/classical_barath_eval.log 2>&1) || log "Barath eval FAILED"
    log "Barath done"
  else
    log "Barath skipped: rsmp env failed (logs/env_rsmp.log)"
  fi ) &

wait_for $L/setup_extra.log ENV_classical_
grep -q ENV_classical_DONE $L/setup_extra.log || { log "COLMAP skipped: env failed"; wait; log "CLASSICAL_DONE"; exit 0; }
P=$(( $(nproc) / 3 )); [ $P -lt 2 ] && P=2
(cd $C && nice -n 15 $R/envs/classical_cuda/bin/python run_sfm3.py --procs $P --out $B/sfm_cpu > $L/classical_sfm.log 2>&1) \
  || log "SfM FAILED"
(cd $C && $R/envs/classical_cuda/bin/python collect_sfm.py --jobs $B/sfm_cpu/jobs_cpu.jsonl --out $B/sfm_cpu \
  > $L/classical_collect.log 2>&1) || log "collect FAILED"
wait
log "CLASSICAL_DONE"
