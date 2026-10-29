#!/bin/bash
# EXP2 downloads beyond TUM, all approved by the user on 2026-10-29 (see results/exp2/DATA_AUDIT.md):
#   Replica (Replica Dataset Research Terms; same parts and URLs as the official download.sh)
#   ADVIO (CC BY-NC 4.0, Zenodo record 1476931, as in the official README)
#   ETH3D SLAM training sequences, monocular + IMU (CC BY-NC-SA 4.0; URLs as in download_eth3d_slam_datasets.py)
# usage: bash download_more.sh {replica|advio|eth3d}     log: logs/dl_<what>.log
set -uo pipefail
D=~/rs3d/data
what=$1
mkdir -p ~/rs3d/logs
exec > >(tee -a ~/rs3d/logs/dl_$what.log) 2>&1
echo "== $(date -u +%FT%TZ) start $what"
case $what in
  replica)
    mkdir -p $D/replica_dl && cd $D/replica_dl
    printf '%s\n' {a..q} | xargs -P 4 -I{} wget -q --continue \
      https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0/replica_v1_0.tar.gz.parta{}
    ls -la
    cat replica_v1_0.tar.gz.part?? | unpigz -p 8 | tar -xC $D && mv $D/Replica-Dataset $D/replica 2>/dev/null
    [ -d $D/replica ] || mv $D/replica_v1_0 $D/replica 2>/dev/null || true
    ls $D/replica* | head -40
    ;;
  advio)
    mkdir -p $D/advio && cd $D/advio
    printf '%02d\n' $(seq 1 23) | xargs -P 4 -I{} sh -c \
      '[ -d advio-{} ] || { wget -q -c -O advio-{}.zip https://zenodo.org/record/1476931/files/advio-{}.zip && unzip -q advio-{}.zip && rm advio-{}.zip && echo advio-{} ok; }'
    ;;
  eth3d)
    mkdir -p $D/eth3d_slam/training && cd $D/eth3d_slam/training
    curl -s https://www.eth3d.net/data/slam/dataset_list_training.txt | grep -v '^$' > ../training_list.txt
    cat ../training_list.txt | xargs -P 4 -I{} sh -c 'for k in mono imu; do
        [ -f {}/.$k ] && continue
        wget -q -c https://www.eth3d.net/data/slam/datasets/{}_$k.zip && unzip -q -o {}_$k.zip && rm {}_$k.zip && touch {}/.$k
      done; echo {} ok'
    ;;
esac
du -sh $D/* 2>/dev/null
echo "DL_${what}_DONE"
