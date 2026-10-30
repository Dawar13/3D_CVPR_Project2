#!/bin/bash
# usage: run_list.sh <log> <extra args for run_tum.py> -- method:weights[:gamma] ...
# Runs run_tum.py for each pair in turn (frames already done are skipped). Before each launch waits until the
# shared GPU has >= 12.5 GB free (rule: launch only with >= 12 GB free); retries a pair up to 50 times if
# run_tum refuses to start. Then writes a check image.
LOG=$1; shift
EXTRA=()
while [ "$1" != "--" ]; do EXTRA+=("$1"); shift; done; shift
cd ~/rs3d/ext/unroll2d
for mw in "$@"; do
  IFS=: read m w g <<< "$mw"
  GA=(); [ -n "$g" ] && GA=(--gamma "$g")
  C="U2D-$m-$w"; [ -n "$g" ] && C="$C-g$(printf %.2f $g)"
  for try in $(seq 1 50); do
    until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 12800 ]; do sleep 60; done
    echo "=== $m $w $g try $try $(date)" >> "$LOG"
    ~/rs3d/envs/u2d/bin/python run_tum.py --method "$m" --weights "$w" "${GA[@]}" "${EXTRA[@]}" 2>&1 \
       | grep -v "UserWarning\|_VF.meshgrid\|warnings.warn\|kaiming_normal" >> "$LOG"
    tail -n 2 "$LOG" | grep -q "U2D_DONE" && break
    sleep 30
  done
  ~/rs3d/envs/u2d/bin/python check_image.py --conds "$C" --half >> "$LOG" 2>&1
done
echo "LIST_DONE $(date)" >> "$LOG"
