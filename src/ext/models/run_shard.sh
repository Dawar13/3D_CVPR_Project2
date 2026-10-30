#!/bin/bash
# One shard of a long run (StreamVGGT is ~10 s/clip on the shared GPU): main-set shard, then control-set shard.
# usage: bash ext/models/run_shard.sh <model> <env> <a|b>
#   shard a writes preds/<model>_gsrs.npz and <model>_control.npz, shard b writes <model>_gsrs_b.npz and
#   <model>_control_b.npz; merge_shards.py combines them into <model>.npz and <model>_control.npz.
set -u
M=$1; ENV=$2; S=$3
cd ~/rs3d
source envs/$ENV/bin/activate
export HF_HOME=~/rs3d/.hf HF_HUB_OFFLINE=1 RS3D_RES=~/rs3d/results/exp2/gap/tum PYTHONPATH=~/rs3d/ext/models
P=$RS3D_RES/preds
SUF=$([ "$S" = a ] && echo "" || echo "_b")
echo "=== $(date) $M gsrs shard $S"
python src/infer2.py --model $M --clips ext/models/clips/gsrs_$S.csv --out $P --name gsrs$SUF
echo "=== $(date) $M control shard $S"
python src/infer2.py --model $M --clips ext/models/clips/control_$S.csv --out $P --name control$SUF
echo "=== $(date) RUN_SHARD_DONE $M $S"
