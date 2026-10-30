#!/bin/bash
# Full EXP2 TUM runs for one extra model: GS+RS main set, then the GS-vs-GS control set.
# usage: bash ext/models/run_full.sh <model> <env-name>     (logs go to the caller-chosen file)
set -u
M=$1; ENV=$2
cd ~/rs3d
source envs/$ENV/bin/activate
export HF_HOME=~/rs3d/.hf HF_HUB_OFFLINE=1 RS3D_RES=~/rs3d/results/exp2/gap/tum PYTHONPATH=~/rs3d/ext/models
P=$RS3D_RES/preds
echo "=== $(date) $M gsrs"
python src/infer2.py --model $M --clips ext/models/clips/gsrs.csv --out $P --name gsrs && cp $P/${M}_gsrs.npz $P/${M}.npz && echo "copied ${M}_gsrs.npz -> ${M}.npz"
echo "=== $(date) $M control"
python src/infer2.py --model $M --clips ext/models/clips/control.csv --out $P --name control
echo "=== $(date) RUN_FULL_DONE $M"
