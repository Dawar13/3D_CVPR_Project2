#!/bin/bash
# label|url
while IFS='|' read -r lab u; do
  [ -z "$lab" ] && continue
  out=$(curl -s -o body.tmp -w "%{http_code} %{url_effective} %{size_download}" -L -m 40 -A "Mozilla/5.0" "$u")
  extra=""
  if grep -qi "does not exist\|not found\|Sorry, the file you have requested\|404" body.tmp 2>/dev/null; then extra="[body has not-found text]"; fi
  if grep -qi "accounts.google.com/ServiceLogin\|Sign in" body.tmp 2>/dev/null; then extra="$extra[body mentions sign-in]"; fi
  t=$(grep -o "<title>[^<]*</title>" body.tmp 2>/dev/null | head -1 | cut -c1-120)
  echo "$lab | $out | $t $extra"
done
