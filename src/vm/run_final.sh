#!/bin/bash
# EXP2 2H: refresh the pre-registered analysis (analyze_exp2.py) every hour while the drivers run, and once more
# when every driver has finished. A driver counts as finished when its DONE line is in its log, or when
# ~/rs3d/skip_<name> exists (a part the user chose to drop, e.g. skip_waymo). Log: logs/final.log
source ~/rs3d/src/vm/lib.sh
cd $R
source .venv/bin/activate
DONE=(
  "tum2 $L/exp2_tum2.log TUM2_DONE"
  "synth $L/synth_core.log SYNTH_CORE_DONE"
  "realgap $L/real_gap.log REAL_GAP_DONE"
  "waymo $L/waymo.log WAYMO_DONE"
  "adt $L/adt.log ADT_DONE"
  "noharm $L/noharm.log NOHARM_DONE"
  "stress $L/stress.log STRESS_DONE"
  "models $L/models.log MODELS_DONE"
  "u2d $L/u2d.log U2D_DONE"
  "classical $L/classical.log CLASSICAL_DONE"
  "lora $L/lora.log LORA_DONE"
)
wait_for $L/setup_env.log SETUP_ENV_DONE
while true; do
  n=0; left=""
  for d in "${DONE[@]}"; do
    set -- $d
    if grep -q "$3" $2 2>/dev/null || [ -e $R/skip_$1 ]; then n=$((n + 1)); else left="$left $1"; fi
  done
  python src/analyze_exp2.py > $L/analyze.log 2>&1
  echo "$(stamp) analysis refreshed; $n/${#DONE[@]} drivers done; waiting for:$left" >> $L/final.log
  [ $n -eq ${#DONE[@]} ] && break
  sleep 3600
done
echo "$(stamp) ALL_DONE" >> $L/final.log
