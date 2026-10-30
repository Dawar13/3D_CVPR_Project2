#!/bin/bash
# Sequential GPU queue for the extra models (one GPU process of ours at a time, via src/gpu_slot.sh).
# 1) wait for the running DA3 chain (pid given) to finish; 2) StreamVGGT main+control (resumes from saved npz);
# 3) DA3-nested main+control.
cd ~/rs3d
while kill -0 $1 2>/dev/null; do sleep 30; done
echo "=== $(date) DA3 chain finished; starting StreamVGGT"
bash src/gpu_slot.sh bash ext/models/run_full.sh streamvggt streamvggt >> logs/models_streamvggt_full.log 2>&1
echo "=== $(date) StreamVGGT finished rc=$?; starting DA3-nested"
rm -f ext/models/skip_da3nested
bash src/gpu_slot.sh bash ext/models/run_full.sh da3nested da3 >> logs/models_da3nested_full.log 2>&1
echo "=== $(date) QUEUE_DONE"
