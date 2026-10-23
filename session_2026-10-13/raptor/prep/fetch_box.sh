#!/bin/bash
# fetch_box.sh (box): curl the prep kernel outputs listed in ~/raptor_in/urls.tsv (6 in parallel), then check every tar's
# sha256 against the 3 manifests. Pattern = ~/fleet_in/fetch_all.sh.
cd ~/raptor_in
while IFS=$'\t' read -r name url size; do
  [ -z "$name" ] && continue
  [ -s "$name" ] && { echo "HAVE $name"; continue; }
  ( curl -sS -L --http1.1 --retry 8 --retry-all-errors --retry-delay 5 -o "$name.part" "$url" && mv "$name.part" "$name" && echo "OK $name $(stat -c %s "$name")" || echo "FAIL $name" ) &
  while [ $(jobs -r | wc -l) -ge 6 ]; do sleep 2; done
done < urls.tsv
wait
echo FETCH_DONE
~/miniconda3/envs/rsna/bin/python - <<'PY'
import json, hashlib, glob, os
bad = 0; n = 0
for m in sorted(glob.glob('raptor_d96_manifest_*.json')):
    j = json.load(open(m)); print(m, j['status'], j['n_studies'], j['n_png'], j['versions']['numpy'], j['versions']['cv2'])
    for f, sh in j['parts'].items():
        n += 1
        h = hashlib.sha256(open(f, 'rb').read()).hexdigest() if os.path.exists(f) else 'missing'
        if h != sh: bad += 1; print('  BAD', f, h[:12])
print(f'PARTS_VERIFIED {n - bad}/{n}')
PY
