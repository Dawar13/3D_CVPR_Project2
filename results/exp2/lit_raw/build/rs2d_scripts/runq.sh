#!/bin/bash
# name|query
while IFS='|' read -r n s; do
  [ -z "$n" ] && continue
  for k in 1 2 3 4 5 6; do
    sleep 20
    curl -sSL -m 90 "https://export.arxiv.org/api/query?search_query=$s&sortBy=submittedDate&sortOrder=descending&max_results=200" -o $n.xml
    sz=$(wc -c < $n.xml); [ "$sz" -gt 300 ] && break
  done
  echo "$n $sz"
done << 'Q'
q3|abs:%22rolling%20shutter%22+AND+(abs:blur+OR+abs:deblurring+OR+abs:exposure)
q4|abs:%22rolling%20shutter%22+AND+abs:event
q6|abs:%22rolling%20shutter%22+AND+(abs:unrolling+OR+abs:undistortion+OR+abs:rectification)
q7|ti:%22rolling%20shutter%22
q8|all:%22rolling%20shutter%22+AND+(all:network+OR+all:learning+OR+all:diffusion)
q9|abs:%22rolling%20shutter%22+AND+(abs:SLAM+OR+abs:reconstruction)
Q
