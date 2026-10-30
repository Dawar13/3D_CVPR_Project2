#!/bin/bash
# DA3 on the full thinned ADVIO RS list (1641 clips). The first attempt crashed on a truncated frame
# (frames_advio/advio-02/03384.png), which the main agent has since re-decoded. infer2 skips keys already in
# preds/da3_RS.npz. Uses our own single GPU slot, so it never runs at the same time as a chain.sh job.
cd ~/rs3d
export HF_HOME=~/rs3d/.hf HF_HUB_OFFLINE=1 PYTHONPATH=~/rs3d/ext/models
P=~/rs3d/results/exp2/gap/advio/preds
echo "=== $(date) START da3 advio_full_thin"
env RS3D_LOCK_DIR=$HOME/rs3d/locks_models RS3D_GPU_SLOTS=1 bash ~/rs3d/src/gpu_slot.sh \
    envs/da3/bin/python src/infer2.py --model da3 --clips results/exp2/gap/advio/clips_RS_thin.csv --out $P --name RS \
    >> logs/models_da3_advio.log 2>&1 && cp $P/da3_RS.npz $P/da3.npz && echo "copied da3_RS.npz -> da3.npz"
echo "=== $(date) END da3 advio_full_thin rc=$?"
