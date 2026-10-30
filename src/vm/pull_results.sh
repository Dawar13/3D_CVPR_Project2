#!/bin/bash
# Run on the LAPTOP: copy every small EXP2 result file (csv, json, md, txt, png, jpg, yaml) and all logs from the VM
# into results/exp2_vm/ (results) and results/exp2_vm/_logs/ (logs). Model outputs (npz) stay on the VM.
# A terminated VM then loses only raw predictions, never the numbers. usage: bash src/vm/pull_results.sh <ip> [user]
set -uo pipefail
IP=$1; U=${2:-gpunet}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
D=$REPO/results/exp2_vm
mkdir -p $D/_logs
ssh $U@$IP 'cd ~/rs3d/results/exp2 && find . -type f \( -name "*.csv" -o -name "*.json" -o -name "*.md" -o -name "*.txt" \
    -o -name "*.png" -o -name "*.jpg" -o -name "*.yaml" \) -size -50M -not -path "*_dense/*" -print0 | tar --null -czf - -T -' \
  | tar -xzf - -C $D
ssh $U@$IP 'cd ~/rs3d/logs && tar -czf - --exclude="*.gz" .' | tar -xzf - -C $D/_logs
echo "$(date -u +%FT%TZ) pulled: $(find $D -type f | wc -l) files, $(du -sh $D | cut -f1)"
