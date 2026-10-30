#!/bin/bash
# EXP2 2C: generate every synthetic set (configs/synth/*.yaml), core test first. Resumable (finished clips are skipped).
set -u
source ~/rs3d/src/vm/lib.sh  # waits for inputs (VM bootstrap); sets HF_HOME, thread caps
wait_for $L/setup_env.log SETUP_ENV_DONE $L/dl_replica.log DL_replica_DONE
source ~/rs3d/.venv/bin/activate
cd ~/rs3d/src
L=~/rs3d/logs
W=${WORKERS:-$(( $(nproc) * 2 / 3 ))}  # the generator is the critical path to LoRA training
run() { python -m rsgen.generate --config ~/rs3d/configs/synth/$1.yaml --split $2 --workers $W > $L/gen_$1_$2.log 2>&1;
        echo "$(date -u +%FT%TZ) $1/$2: $(tail -n 1 $L/gen_$1_$2.log)" | tee -a $L/gen_all.log; }
run core_test test
run core_trainval val
run core_trainval train
for c in st_readout0 st_readout5 st_readout10 st_readout20 st_readout40 st_dir_bt st_dir_lr st_objects_r0 st_objects_r30 \
         st_lowtex st_blur_r0_e16 st_blur_r30_e8 st_blur_r30_e16 st_stride1 st_stride6 st_stride12 st_views64; do
  run $c test
done
echo "$(date -u +%FT%TZ) GEN_ALL_DONE" | tee -a $L/gen_all.log
