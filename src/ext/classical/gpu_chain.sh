#!/usr/bin/env bash
# All GPU work of the classical-baseline agent, strictly sequential (one GPU process at a time), run as
#   bash ~/rs3d/src/gpu_slot.sh bash ~/rs3d/ext/classical/gpu_chain.sh
# 1) Barath repo's own TUM-RS example (seq 1)  2) Barath on our RS windows, consecutive pairs (gap 1)
# (COLMAP/GLOMAP moved to CPU-only SIFT on 2026-10-29 16:50 after 2 h without a free GPU slot.)
set -x
export TORCH_HOME=~/rs3d/cache/torch OMP_NUM_THREADS=3
C=~/rs3d/ext/classical
R=~/rs3d/results/exp2/baselines/classical
cd $C
if ! grep -q "AUC summary" ~/rs3d/logs/classical_barath_repro.log 2>/dev/null; then
  ~/rs3d/envs/rsmp/bin/python run_repro.py --sequences 1 --output $R/barath/repro/results_rs_tum_seq1.csv \
    > ~/rs3d/logs/classical_barath_repro.log 2>&1
fi
~/rs3d/envs/rsmp/bin/python barath_windows.py --conds RS --gaps 1 --workers 3 >> ~/rs3d/logs/classical_barath_windows.log 2>&1
echo GPU_CHAIN_DONE
