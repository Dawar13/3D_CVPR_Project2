#!/bin/bash
# Single sequential GPU queue for the unroll2d jobs (main-agent rule: at most ONE GPU process of ours at a time).
# Pops the first line of $Q (a shell command) and runs it through ~/rs3d/src/gpu_slot.sh with RS3D_GPU_SLOTS=1 and a
# private lock dir (gpu_slot.sh uses ~/rs3d/locks, so HOME is pointed at ~/rs3d/ext/unroll2d/slot for the wrapper
# only and restored for the job). Lines starting with # are skipped; a line STOP ends the runner; when the queue is
# empty it waits for new lines. Edit $Q at any time to reorder/append. Log: ~/rs3d/logs/unroll2d_queue.log
Q=~/rs3d/ext/unroll2d/queue.txt
LOG=~/rs3d/logs/unroll2d_queue.log
SLOTHOME=/home/gpunet/rs3d/ext/unroll2d/slot
mkdir -p $SLOTHOME/rs3d/locks
touch $Q
while true; do
  # wait for any of my GPU jobs started outside the queue (smoke runs) to finish
  while pgrep -f "^/home/gpunet/rs3d/envs/u2d/bin/python run_(tum|synth)" >/dev/null || \
        pgrep -f "^python src/infer2.py .*unroll2d" >/dev/null; do sleep 20; done
  line=$(head -n 1 $Q)
  if [ -z "$line" ]; then sleep 60; continue; fi
  sed -i 1d $Q
  [[ "$line" == \#* ]] && continue
  [ "$line" == "STOP" ] && { echo "QUEUE_STOP $(date)" >> $LOG; exit 0; }
  echo "=== START $(date) :: $line" >> $LOG
  env HOME=$SLOTHOME RS3D_GPU_SLOTS=1 bash /home/gpunet/rs3d/src/gpu_slot.sh \
      env HOME=/home/gpunet bash -c "$line" >> $LOG 2>&1
  echo "=== END rc=$? $(date) :: $line" >> $LOG
done
