#!/bin/bash
# Re-download TUM RS-VIO (EuRoC export, the calibrated sequences 1-10 plus the GS-GS calib-imu1 control) and the
# Kalibr camchain, into the layout common.py expects: data/tumrs/dataset-seqN/dataset-seqN/mav0. Log: logs/dl_tum.log
set -uo pipefail
D=~/rs3d/data/tumrs
mkdir -p $D ~/rs3d/logs
exec > >(tee -a ~/rs3d/logs/dl_tum.log) 2>&1
cd $D
B=https://cdn3.vision.in.tum.de/rolling
wget -q -nc $B/calibration/camchain-calibration-equidistant4_camimu_dataset-calib-imu1.yaml

fetch() {  # $1 = sequence name, e.g. dataset-seq3
  local n=$1
  [ -d $n ] && { echo "$n present"; return 0; }
  wget -q -c $B/exported/euroc/$n.tar -O $n.tar || { echo "$n DOWNLOAD_FAILED"; return 1; }
  mkdir -p $n.tmp && tar -xf $n.tar -C $n.tmp && mv $n.tmp $n && rm -f $n.tar && echo "$n ok $(date -u +%T)"
}
export -f fetch; export B
printf '%s\n' dataset-seq{1..10} dataset-calib-imu1 | xargs -P 4 -I{} bash -c 'fetch {}'
du -sh $D/* | sort -k2
echo "DL_TUM_DONE"
