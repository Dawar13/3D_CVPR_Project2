#!/bin/bash
# EXP1 pipeline: E1a (rotation-only unrolling) and E1b (rotation + translation), 3 models each.
set -u
source ~/rs3d/.venv/bin/activate
cd ~/rs3d
export HF_HOME=~/rs3d/.hf HF_HUB_OFFLINE=1
L=~/rs3d/logs
echo "$(date -u +%T) start; free VRAM $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)" > $L/e1_progress.log

python src/pi3_depth.py > $L/e1_pi3_depth.log 2>&1 &
DEP=$!
python src/unroll.py rot > $L/e1_unroll_rot.log 2>&1
echo "$(date -u +%T) E1a images done" >> $L/e1_progress.log
for m in mapanything vggt pi3; do
  python src/infer.py --model $m --cams RSU --tag _e1rot > $L/e1_infer_${m}_rot.log 2>&1 &
done
wait $DEP
echo "$(date -u +%T) pi3 depth done" >> $L/e1_progress.log
python src/unroll.py rotrans > $L/e1_unroll_rotrans.log 2>&1
echo "$(date -u +%T) E1b images done" >> $L/e1_progress.log
wait
echo "$(date -u +%T) E1a inference done" >> $L/e1_progress.log
for m in mapanything vggt pi3; do
  python src/infer.py --model $m --cams RSUT --tag _e1rotrans > $L/e1_infer_${m}_rotrans.log 2>&1 &
done
wait
echo "$(date -u +%T) E1_ALL_DONE" >> $L/e1_progress.log
