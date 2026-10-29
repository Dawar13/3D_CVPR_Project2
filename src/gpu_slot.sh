#!/bin/bash
# Run a command while holding one of N GPU slots (N = RS3D_GPU_SLOTS, default 3) and only once >= RS3D_MIN_FREE_GB
# (default 14) of GPU memory is free. Caps how many of our jobs share the GPU with the helpers and the other project.
# usage: gpu_slot.sh <command> [args...]
N=${RS3D_GPU_SLOTS:-3}
MIN_MB=$(( ${RS3D_MIN_FREE_GB:-14} * 1024 ))
D=~/rs3d/locks
mkdir -p $D
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while true; do
  for i in $(seq 0 $((N - 1))); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
    [ "$free" -lt "$MIN_MB" ] && break
    flock -n -E 200 $D/gpu$i "$@"
    rc=$?
    [ $rc -ne 200 ] && exit $rc
  done
  sleep 20
done
