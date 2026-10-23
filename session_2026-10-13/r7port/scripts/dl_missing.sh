#!/bin/bash
# sequential, rate-limit-aware download of missing code files
SP="$1"; cd "$SP/r7port"
list="$2"
while read -r f; do
  d="ofl/$(dirname "$f")"; b="$(basename "$f")"
  [ -s "$d/$b" ] && continue
  mkdir -p "$d"
  for t in 1 2 3 4 5 6 7 8; do
    out=$(kaggle datasets download heliosli/rsna-orthofoundation-fold0-epoch7-20261004 -f "$f" -p "$d" 2>&1)
    if [ -s "$d/$b" ]; then break; fi
    if [ -s "$d/$b.zip" ]; then (cd "$d" && unzip -o -q "$b.zip" && rm -f "$b.zip"); break; fi
    echo "retry $t $f: $(echo "$out" | tail -1 | cut -c1-80)"
    sleep $((t*15))
  done
  sleep 2
done < "$list"
echo DONE_DL
