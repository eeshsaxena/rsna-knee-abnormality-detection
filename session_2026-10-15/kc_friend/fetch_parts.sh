#!/bin/bash
# fetch_parts.sh: copy the remaining KC store zip parts + MANIFEST from the box, verify sha256 vs the box MANIFEST.
cd /c/Users/eeshs/Downloads/rsna-knee/session_2026-10-15/kc_friend/upload/rsna-knee-kc-store-v1 || exit 1
for k in 1 2 3; do scp -q training@192.168.10.30:kc_pack/kc_store_part$k.zip . || exit 2; echo "part$k copied $(date +%H:%M:%S)"; done
scp -q training@192.168.10.30:kc_pack/MANIFEST.json ../MANIFEST_kc_pack.json
python - <<'PY'
import json, hashlib, os
m = json.load(open('../MANIFEST_kc_pack.json'))['parts']
ok = True
for n, r in m.items():
    h = hashlib.sha256()
    with open(n, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    good = h.hexdigest() == r['sha256'] and os.path.getsize(n) == r['bytes']
    ok &= good; print(n, 'OK' if good else 'MISMATCH')
print('KC_FETCH', 'PASS' if ok else 'FAIL')
PY
