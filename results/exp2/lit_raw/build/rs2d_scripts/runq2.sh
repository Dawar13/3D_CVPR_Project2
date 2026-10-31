#!/bin/bash
while IFS='|' read -r n s; do
  [ -z "$n" ] && continue
  for k in 1 2 3 4 5 6 7 8; do
    sleep 25
    curl -sSL -m 90 "https://export.arxiv.org/api/query?search_query=$s&sortBy=submittedDate&sortOrder=descending&max_results=200" -o $n.xml
    sz=$(wc -c < $n.xml); [ "$sz" -gt 300 ] && break
  done
  echo "$n $sz"
done << 'Q'
q3|abs:%22rolling%20shutter%22+AND+(abs:blur+OR+abs:deblurring+OR+abs:exposure)
q10|all:%22rolling%20shutter%22+AND+(all:gaussian+OR+all:nerf)
q11|ti:rolling+AND+ti:shutter
q12|abs:%22rolling%20shutter%22+AND+(abs:depth+OR+abs:pointmap+OR+abs:%22feed-forward%22)
Q
