# Shared helpers for the EXP2 VM drivers (sourced, never run).
R=$HOME/rs3d
L=$R/logs
mkdir -p $L
export HF_HOME=$R/.hf UV_CACHE_DIR=$R/.uv-cache
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}  # many GPU jobs share the CPUs
G="bash $R/src/gpu_slot.sh"
stamp() { date -u +%FT%TZ; }

# wait_for <file> <pattern> [<file> <pattern> ...]: block until every file contains its pattern
wait_for() {
  while [ $# -ge 2 ]; do
    until grep -q -- "$2" "$1" 2>/dev/null; do sleep 30; done
    shift 2
  done
}

# wait_ok <file> <ok-pattern> <fail-pattern>: 0 when ok appears, 1 when fail appears first
wait_ok() {
  while true; do
    grep -q -- "$2" "$1" 2>/dev/null && return 0
    grep -q -- "$3" "$1" 2>/dev/null && return 1
    sleep 30
  done
}

# wait_file <path> [...]: block until every path exists
wait_file() {
  for p in "$@"; do until [ -e "$p" ]; do sleep 30; done; done
}
