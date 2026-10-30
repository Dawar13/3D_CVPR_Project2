#!/bin/bash
# Single sequential GPU queue for the unroll2d jobs (main-agent rule: at most ONE GPU process of ours at a time).
# Pops the first line of $Q (a shell command) and runs it through ~/rs3d/src/gpu_slot.sh with RS3D_GPU_SLOTS=1 and
# our own lock dir RS3D_LOCK_DIR=~/rs3d/locks_u2d (waits for >= 14 GB free VRAM). Lines starting with # are skipped;
# a line STOP ends the runner; when the queue is empty it waits for new lines. Edit $Q at any time to reorder/append.
# Log: ~/rs3d/logs/unroll2d_queue.log   (queue_runner.sh = earlier version that redirected HOME for the lock dir)
Q=~/rs3d/ext/unroll2d/queue.txt
LOG=~/rs3d/logs/unroll2d_queue.log
touch $Q
while true; do
  line=$(head -n 1 $Q)
  if [ -z "$line" ]; then sleep 60; continue; fi
  sed -i 1d $Q
  [[ "$line" == \#* ]] && continue
  [ "$line" == "STOP" ] && { echo "QUEUE_STOP $(date)" >> $LOG; exit 0; }
  echo "=== START $(date) :: $line" >> $LOG
  env RS3D_LOCK_DIR=$HOME/rs3d/locks_u2d RS3D_GPU_SLOTS=1 bash $HOME/rs3d/src/gpu_slot.sh bash -c "$line" >> $LOG 2>&1
  echo "=== END rc=$? $(date) :: $line" >> $LOG
done
