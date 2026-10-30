#!/bin/bash
# Run a command while holding one of N GPU slots (N = RS3D_GPU_SLOTS, default 3) and only once >= RS3D_MIN_FREE_GB
# (default 14) of GPU memory is free. Caps how many of our jobs share the GPU at once.
# Each helper uses its own lock dir (RS3D_LOCK_DIR) with RS3D_GPU_SLOTS=1, so it never waits on the main runs' slots.
# LoRA reservation: while ~/rs3d/locks/lora_reserve exists and no train_lora.py process is on the GPU (a run is
# waiting for its 35 GB), new jobs also leave 37 GB free, so training restarts as soon as running jobs make room.
# A failed command is retried RS3D_RETRIES times (default 2) after 60 s (OOM races; infer2 resumes from its npz).
# usage: gpu_slot.sh <command> [args...]
N=${RS3D_GPU_SLOTS:-3}
MIN_MB=$(( ${RS3D_MIN_FREE_GB:-14} * 1024 ))
D=${RS3D_LOCK_DIR:-$HOME/rs3d/locks}
RES=$HOME/rs3d/locks/lora_reserve
mkdir -p $D
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

reserve_mb() {
  [ -f $RES ] || { echo 0; return; }
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    ps -o args= -p $p 2>/dev/null | grep -q train_lora && { echo 0; return; }
  done
  echo 37888
}

try=0
while true; do
  for i in $(seq 0 $((N - 1))); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
    [ "$free" -lt $(( MIN_MB + $(reserve_mb) )) ] && break
    flock -n -E 200 $D/gpu$i "$@"
    rc=$?
    [ $rc -eq 200 ] && continue
    [ $rc -eq 0 ] && exit 0
    try=$((try + 1))
    [ $try -gt ${RS3D_RETRIES:-2} ] && exit $rc
    echo "gpu_slot: rc=$rc, retry $try in 60 s: $*" >&2
    sleep 60
    continue 2
  done
  sleep 20
done
