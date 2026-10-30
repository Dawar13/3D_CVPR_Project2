#!/bin/bash
# EXP2 2F training controls (brief section 8.6; src/ext/lora): LoRA on pi3 and VGGT, 6a = synthetic RS train split,
# 6b = its GS twins, 3 seeds each, the models' own losses, no motion modelling.
#   1. step-0 identity check (LoRA at init == base model)
#   2. confidence-loss pilots (6a seed 0, 400 steps, with / without) and the D19 rule, per model, on val only:
#      use the confidence loss iff its pilot had no skipped non-finite steps and a best val ATE <= the pilot without
#   3. the 12 runs, one at a time (each needs ~35 GB; gpu_slot.sh holds new jobs back while a run waits for memory),
#      each evaluated in the background on the synthetic core test and the TUM eval subset (eval_run.sh)
#   4. base-model evaluation on the same subsets, export for analyze_exp2.py
# Log: logs/lora.log (+ lora_queue.log, lora_eval.log)
source ~/rs3d/src/vm/lib.sh
cd $R
source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=$R/results/exp2/training
E=$T/eval
X=$R/ext/lora
mkdir -p $E $T/configs
log() { echo "$(stamp) $*" | tee -a $L/lora.log; }

wait_for $L/setup_env.log SETUP_ENV_DONE $L/exp2_stage1.log "prep done" $L/gen_core_test_test.log GENERATE_DONE \
  $L/gen_core_trainval_val.log GENERATE_DONE $L/gen_core_trainval_train.log GENERATE_DONE
python $X/score_lora.py tum_clips --out $E/clips_tum.csv > $L/lora_clips.log 2>&1
(cd src && python synth_clips.py --set $R/data/synth/core_test --conds rs:d,gs:d --out $E/clips_core_test.csv) >> $L/lora_clips.log 2>&1
log "eval clip lists: $(wc -l < $E/clips_tum.csv) TUM, $(wc -l < $E/clips_core_test.csv) synthetic"

touch $R/locks/lora_reserve  # gpu_slot.sh: keep 37 GB free whenever a run is waiting for memory
for m in pi3 vggt; do
  until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -ge 30720 ]; do sleep 30; done
  (cd $X && python check_identity.py --model $m > $L/lora_identity_$m.log 2>&1) || log "identity $m FAILED"
done
log "identity: $(grep -h -i -E 'pass|fail' $L/lora_identity_*.log | tail -2 | tr '\n' ' ')"

(cd $X && python make_configs.py pilot) > /dev/null
PILOTS="pilot_pi3_6a_s0_conf pilot_pi3_6a_s0_noconf pilot_vggt_6a_s0_conf pilot_vggt_6a_s0_noconf"
(cd $X && EVAL=0 bash run_queue.sh $PILOTS)
python - > $T/conf_decision.json <<EOF
import json
T = "$T"
out = {}
for m in ("pi3", "vggt"):
    r = {k: json.load(open(f"{T}/pilot_{m}_6a_s0_{k}/run.json")) for k in ("conf", "noconf")}
    use = r["conf"]["n_skip_nonfinite"] == 0 and r["conf"]["best_score"] <= r["noconf"]["best_score"]
    out[m] = dict(conf_loss=bool(use), rule="D19: conf iff no non-finite steps and best val ATE <= without",
                  conf=dict(best=r["conf"]["best_score"], nonfinite=r["conf"]["n_skip_nonfinite"]),
                  noconf=dict(best=r["noconf"]["best_score"], nonfinite=r["noconf"]["n_skip_nonfinite"]))
print(json.dumps(out, indent=1))
EOF
log "conf decision: $(python -c "import json; d=json.load(open('$T/conf_decision.json')); print({m: v['conf_loss'] for m, v in d.items()})")"

(cd $X && python make_configs.py main --conf 0) > /dev/null
python - <<EOF
import json, glob
dec = json.load(open("$T/conf_decision.json"))
for p in glob.glob("$T/configs/*_6[ab]_s[0-9].json"):
    c = json.load(open(p))
    c["conf_loss"] = dec[c["model"]]["conf_loss"]
    json.dump(c, open(p, "w"), indent=1)
EOF
RUNS=""
for s in 0 1 2; do for m in pi3 vggt; do for a in 6a 6b; do RUNS="$RUNS ${m}_${a}_s$s"; done; done; done
(cd $X && bash run_queue.sh $RUNS)
rm -f $R/locks/lora_reserve
log "training queue done: $(ls $T/*/run.json 2>/dev/null | wc -l) run.json files"

wait_for $L/synth_core.log SYNTH_CORE_DONE  # the base models' core-test predictions come from the main runs
for m in pi3 vggt; do
  (cd $X && SYNTH_PREDS=$R/results/exp2/synth/core_test/preds SYNTH_MODEL=$m bash eval_run.sh base_$m $m base) \
    || log "base eval $m FAILED"
done
while pgrep -f "[e]val_run.sh" > /dev/null; do sleep 60; done  # the runs' background evaluations
python $X/score_lora.py summary --out $T/summary > $L/lora_summary.log 2>&1
python $X/export_lora.py >> $L/lora_summary.log 2>&1
log "LORA_DONE"
