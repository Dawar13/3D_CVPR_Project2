#!/bin/bash
# EXP2 stage 2 on TUM RS-VIO (after stage 1's REPRO_PASS):
#   2D   MapAnything GS/RS + GS-GS control (pi3 and VGGT come from stage 1)
#   2E   oracle ladder V3 (gyro w) and V4 (gyro w + GT v + pi3 depth in metres, EXP1-b), readout x{0.5,0.75,1.25,1.5}
#        and wrong scan direction (bt)
#   2F   self-estimate two-pass: SE2P (w), SE2PT (w, v, own depth), SE2P_shuf (unordered: shuffled pass 1 and 2)
# Each model's inference chain runs in parallel; every step is resumable.
set -u
G="bash $HOME/rs3d/src/gpu_slot.sh"  # every GPU step holds a slot (gpu_slot.sh)
source ~/rs3d/.venv/bin/activate
cd ~/rs3d
export HF_HOME=~/rs3d/.hf HF_HUB_OFFLINE=1 RS3D_RES=~/rs3d/results/exp2/gap/tum
L=~/rs3d/logs
P=$RS3D_RES/preds
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $L/exp2_tum2.log; }
source ~/rs3d/src/vm/lib.sh
wait_ok $L/exp2_stage1.log REPRO_PASS REPRO_FAIL || { log "no REPRO_PASS: not running"; exit 1; }

ORACLE="V3 V4 V3_r0.5 V3_r0.75 V3_r1.25 V3_r1.5 V4_r0.5 V4_r0.75 V4_r1.25 V4_r1.5 V3_bt V4_bt"
MODELS="pi3 vggt mapanything"

python - <<'EOF'
import os, pandas as pd
R = os.environ["RS3D_RES"]
w = pd.read_csv(f"{R}/windows.csv"); w = w[w.keep]
W = os.path.expanduser("~/rs3d/data/windows")
for tag in ("RS", "RSshuf"):  # RSshuf: the plain RS windows, run as an unordered set
    pd.DataFrame(dict(key=[f"{k}_{tag}" for k in w.key], dir=[f"{W}/seq{s}/{i:04d}_RS" for s, i in zip(w.seq, w.window_id)])
                 ).to_csv(f"{R}/clips_{tag}.csv", index=False)
EOF

# --- MapAnything gap (EXP0 recipe) and pi3 depth for V4, then the oracle images (GPU unroll, fast)
$G python src/infer.py --model mapanything > $L/e2_infer_mapanything.log 2>&1 &
$G python src/pi3_depth.py > $L/e2_pi3_depth.log 2>&1
log "pi3 depth: $(tail -n 1 $L/e2_pi3_depth.log)"
for c in $ORACLE; do
  $G python src/tum_conditions.py $c >> $L/e2_cond_oracle.log 2>&1 || log "condition $c FAILED"
done
log "oracle conditions: $(grep -c COND_DONE $L/e2_cond_oracle.log) done"
wait
$G python src/infer.py --model mapanything --set control > $L/e2_infer_mapanything_control.log 2>&1
log "mapanything GS/RS + control done"

chain() {  # one model: pass 1 (dense, shuffled), its self-estimate conditions, then every condition's inference
  m=$1
  $G python src/infer2.py --model $m --clips $RS3D_RES/clips_RS.csv --out $P --name RSdense --dense > $L/e2_p1dense_$m.log 2>&1
  $G python src/infer2.py --model $m --clips $RS3D_RES/clips_RSshuf.csv --out $P --name RSshuf --order shuffled > $L/e2_p1shuf_$m.log 2>&1
  for c in SE2P_$m SE2PT_$m SE2P_${m}_shuf; do
    $G python src/tum_conditions.py $c > $L/e2_cond_$c.log 2>&1 || log "condition $c FAILED"
  done
  conds="$ORACLE SE2P_$m SE2PT_$m"
  [ $m = mapanything ] && conds="V3 V4 SE2P_$m SE2PT_$m"  # secondary model: oracle ends + self-estimate only
  for c in $conds; do
    $G python src/infer.py --model $m --cams $c --tag _$c > $L/e2_infer_${m}_$c.log 2>&1 || log "infer $m $c FAILED"
  done
  # unordered set: pass 2 also sees the shuffled order
  python - <<EOF
import os, pandas as pd
R = os.environ["RS3D_RES"]; c = "SE2P_${m}_shuf"
w = pd.read_csv(f"{R}/windows.csv"); w = w[w.keep]
W = os.path.expanduser("~/rs3d/data/windows")
pd.DataFrame(dict(key=[f"{k}_{c}" for k in w.key], dir=[f"{W}/seq{s}/{i:04d}_{c}" for s, i in zip(w.seq, w.window_id)])
             ).to_csv(f"{R}/clips_{c}.csv", index=False)
EOF
  $G python src/infer2.py --model $m --clips $RS3D_RES/clips_SE2P_${m}_shuf.csv --out $P --name SE2P_${m}_shuf --order shuffled \
    > $L/e2_infer_${m}_SE2P_${m}_shuf.log 2>&1
  log "chain $m done"
}
for m in $MODELS; do chain $m & done
wait

# --- scoring (RSshuf = plain RS seen as an unordered set)
python src/evaluate2.py score --models pi3,vggt,mapanything \
  --conds GS,RS,RSshuf,$(echo $ORACLE | tr ' ' ','),SE2P_pi3,SE2PT_pi3,SE2P_pi3_shuf,SE2P_vggt,SE2PT_vggt,SE2P_vggt_shuf,SE2P_mapanything,SE2PT_mapanything,SE2P_mapanything_shuf \
  --out $RS3D_RES/per_window_all.csv > $L/e2_score_all.log 2>&1
log "TUM2_DONE"
