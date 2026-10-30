#!/bin/bash
# usage: run_synth_list.sh <log> <set dir> <extra args> -- method:weights[:gamma] ...
# run_synth.py for each pair; waits for >= 12.5 GB free VRAM before each launch, retries up to 50 times.
LOG=$1; SET=$2; shift 2
EXTRA=()
while [ "$1" != "--" ]; do EXTRA+=("$1"); shift; done; shift
cd ~/rs3d/ext/unroll2d
for mw in "$@"; do
  IFS=: read m w g <<< "$mw"
  GA=(); [ -n "$g" ] && GA=(--gamma "$g")
  for try in $(seq 1 50); do
    until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 12800 ]; do sleep 60; done
    echo "=== synth $m $w $g try $try $(date)" >> "$LOG"
    ~/rs3d/envs/u2d/bin/python run_synth.py --set "$SET" --method "$m" --weights "$w" "${GA[@]}" "${EXTRA[@]}" 2>&1 \
       | grep -v "UserWarning\|_VF.meshgrid\|warnings.warn\|kaiming_normal" >> "$LOG"
    tail -n 2 "$LOG" | grep -q "SYNTH_DONE" && break
    sleep 30
  done
done
echo "LIST_DONE $(date)" >> "$LOG"
