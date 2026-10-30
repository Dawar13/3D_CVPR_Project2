#!/bin/bash
# Waymo Open Dataset v1.4.3 download (DEVIATIONS D15): validation split, every 6th segment of the sorted bucket
# listing. Waits until the user has run ~/rs3d/bin/waymo_login.sh (their own Google account; the credential stays
# in ~/rs3d/.gcloud). `touch ~/rs3d/skip_waymo` gives up. Log: logs/dl_waymo.log
source ~/rs3d/src/vm/lib.sh
exec >> $L/dl_waymo.log 2>&1
export CLOUDSDK_CONFIG=$R/.gcloud
GC=$R/bin/google-cloud-sdk/bin/gcloud
W=$R/data/waymo
mkdir -p $W/raw
wait_for $L/setup_extra.log ENV_waymo_DONE
until [ -n "$($GC auth list --format='value(account)' 2>/dev/null)" ] || [ -e $R/skip_waymo ]; do sleep 60; done
[ -e $R/skip_waymo ] && { echo "$(stamp) skipped (skip_waymo)"; exit 0; }
echo "$(stamp) login found: listing the validation split"
$GC storage ls gs://waymo_open_dataset_v_1_4_3/individual_files/validation/ | grep tfrecord | sort > $W/all_validation.txt
awk 'NR % 6 == 1' $W/all_validation.txt > $W/selected.txt
echo "$(stamp) $(wc -l < $W/all_validation.txt) segments, $(wc -l < $W/selected.txt) selected"
for try in 1 2 3; do
  cat $W/selected.txt | $GC storage cp -n -I $W/raw/ && break
  sleep 60
done
echo "$(stamp) $(ls $W/raw/*.tfrecord | wc -l) files"
echo "DL_WAYMO_DONE"
