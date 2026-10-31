#!/bin/bash
# usage: get.sh name url [start] [len]
D="C:/Users/djtde/AppData/Local/Temp/claude/c--Users-djtde-Downloads-3D-CVPR-Exp-2-Rolling-Shutter/270d52c8-a58d-416d-8845-c86ba3700ab9/scratchpad"
code=$(curl -sL -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64)" -w "%{http_code}" "$2" -o "$D/pages/$1.html")
echo "[$1] HTTP $code  $2"
python -I "$D/scripts/strip.py" "$D/pages/$1.html" ${3:-0} ${4:-2500}
echo
