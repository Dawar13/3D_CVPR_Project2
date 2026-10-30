# Job helpers for queue.txt lines (each line: `source ~/rs3d/ext/unroll2d/jobs.sh; <helper> args`).
U=~/rs3d/ext/unroll2d
PY=~/rs3d/envs/u2d/bin/python
FILT='UserWarning|_VF.meshgrid|warnings.warn|kaiming_normal'
# tum <method> <weights> [run_tum args...]   -> 2D on all kept TUM windows + check image
tum() { m=$1; w=$2; shift 2; cd $U && $PY run_tum.py --method $m --weights $w --batch 8 "$@" 2>&1 | grep --line-buffered -Ev "$FILT";
        c=$(grep -o '"cond": "[^"]*"' $(ls -t ~/rs3d/results/exp2/baselines/unroll2d/runs/*.json | head -1) | cut -d'"' -f4);
        $PY check_image.py --conds "$c" --half; }
# synth <method> <weights> [run_synth args...] -> 2D on the synthetic core test set
synth() { m=$1; w=$2; shift 2; cd $U && $PY run_synth.py --set ~/rs3d/data/synth/core_test --method $m --weights $w "$@" 2>&1 | grep --line-buffered -Ev "$FILT"; }
# tum3d <model> <COND>  -> src/infer2.py on that condition's clip list
tum3d() { cd ~/rs3d && source .venv/bin/activate && export HF_HOME=~/rs3d/.hf HF_HUB_OFFLINE=1 RS3D_RES=~/rs3d/results/exp2/gap/tum &&
          python src/infer2.py --model $1 --clips ~/rs3d/results/exp2/baselines/unroll2d/clips/$2.csv --out ~/rs3d/results/exp2/gap/tum/preds --name $2; }
# synth3d <model> <csv>  -> src/infer2.py on a synth_clips.py CSV (name column routes conditions)
synth3d() { cd ~/rs3d && source .venv/bin/activate && export HF_HOME=~/rs3d/.hf HF_HUB_OFFLINE=1 &&
            python src/infer2.py --model $1 --clips $2 --out ~/rs3d/results/exp2/synth/core_test/preds --name u2d; }
# synth3d_all [models]  -> clip CSV for every u2d-* condition complete on all core_test clips (dense), then
#                          infer2 per model (sequential; already-done keys are skipped by infer2)
synth3d_all() {
  ms=${1:-pi3,vggt}
  local S=~/rs3d/data/synth/core_test CSV=~/rs3d/results/exp2/baselines/unroll2d/clips/synth_core_test_u2d.csv
  local n=$(ls -d $S/test/*/* | wc -l)
  local conds=$(for c in $(ls $S/test/*/*/ | grep '^u2d-' | sort | uniq -c | awk -v n=$n '$1==n {print $2}'); do
                  [ $(ls $S/test/*/*/$c/*.png 2>/dev/null | wc -l) -eq $((n*16)) ] && echo -n "$c:d,"; done)
  conds=${conds%,}
  echo "synth3d_all conds: $conds"
  [ -z "$conds" ] && return 0
  (cd ~/rs3d/src && source ~/rs3d/.venv/bin/activate && python synth_clips.py --set $S --conds $conds --out $CSV) || return 1
  for m in ${ms//,/ }; do synth3d $m $CSV; done
}
