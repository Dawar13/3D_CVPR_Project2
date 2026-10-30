#!/bin/bash
# ADT download (DEVIATIONS D18): the 40 sequences in configs/adt_selected.txt (every 6th of the 235 sorted names,
# fixed before any download). Needs the user's signed-URL file at ~/rs3d/data/adt/ADT_download_urls.json (copied
# from the laptop; never committed). Stage 1: main_vrs (0) + main_groundtruth (6); stage 2: depth (8).
# The downloader stages files in $TMPDIR, so that points at the big disk. Log: logs/adt_download.log
source ~/rs3d/src/vm/lib.sh
exec >> $L/adt_download.log 2>&1
A=$R/data/adt
mkdir -p $A/raw $R/cache/tmp_adt
export TMPDIR=$R/cache/tmp_adt
wait_file $A/ADT_download_urls.json
wait_for $L/setup_extra.log ENV_aria_DONE
DL=$R/envs/aria/bin/aria_dataset_downloader
cd $A/raw
fetch() {  # fetch <data-group ids> <tag>
  for s in $(cat $R/configs/adt_selected.txt); do
    for try in 1 2 3; do
      $DL -c ../ADT_download_urls.json -o . -d $1 -l $s > /dev/null 2>>../dl_err.log && { echo "$(stamp) OK $2 $s"; break; }
      echo "$(stamp) retry $try $2 $s"; sleep 30
    done
  done
}
fetch "0 6" main
echo "$(stamp) ADT_MAIN_DONE"
fetch 8 depth
echo "$(stamp) ADT_DEPTH_DONE"
