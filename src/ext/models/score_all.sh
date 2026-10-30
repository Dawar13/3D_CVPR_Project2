#!/bin/bash
# Score the extra models on every set they were run on (CPU only). Log: logs/models_score_all.log
cd ~/rs3d
source .venv/bin/activate
G=~/rs3d/results/exp2/gap
S=ext/models/summarize.py
TUM_MODELS=${TUM_MODELS:-da3,streamvggt}
RS3D_RES=$G/tum python src/evaluate2.py score --models $TUM_MODELS --conds GS,RS --out $G/tum/per_window_newmodels.csv
python $S main --csv $G/tum/per_window_newmodels.csv
RS3D_RES=$G/tum python src/evaluate2.py score --set control --models $TUM_MODELS --conds GS,RS --out $G/tum/per_window_control_newmodels.csv
python $S main --csv $G/tum/per_window_control_newmodels.csv
RS3D_RES=$G/advio python src/evaluate2.py score --models da3 --conds RS --out $G/advio/per_window_newmodels.csv
python $S main --csv $G/advio/per_window_newmodels.csv
RS3D_RES=$G/eth3d python src/evaluate2.py score --models da3 --conds GS --gt GS --out $G/eth3d/per_window_newmodels.csv
python $S main --csv $G/eth3d/per_window_newmodels.csv
SYN_MODELS=${SYN_MODELS:-da3,streamvggt}
mkdir -p ~/rs3d/results/exp2/baselines/models
cd src && python synth_eval.py --set ~/rs3d/data/synth/core_test --preds ~/rs3d/results/exp2/synth/core_test/preds_newmodels \
    --models $SYN_MODELS --conds rs,gs --dense --out ~/rs3d/results/exp2/baselines/models/synth_core_test_newmodels.csv
