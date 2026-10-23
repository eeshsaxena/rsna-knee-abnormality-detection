#!/bin/bash
# setup.sh (instance, detached): RSNA-ONLY environment for the Raptor seeds. Writes $R/status/setup.ok or setup.failed.
#  1 pip: timm 1.0.29 (= box), pandas, scikit-learn, opencv-python-headless (PNG decode only; lossless, checked by sha)
#  2 RSNA d96 corpus from the Kaggle prep-kernel outputs (signed URLs in kit/urls.tsv, written by the laptop minutes before):
#    33 tar parts, sha256 = manifests; unpack_d96.py -> $R/rsna/<sid>.npy + $R/rsna_index.csv, every volume sha256 = receipt
#  3 OAI ASSERTION (fail loud): no file name, index id, path or csv cell under $R mentions "oai"
#  4 pretrained coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k into $R/hf (sha256 printed; box blob sha in kit.env)
set -u
R=/workspace/rsna; K=$R/kit; S=$R/status; mkdir -p $S $R/in $R/rsna $R/runs $R/hf; L=$R/setup.log
say(){ echo "[setup $(date -u '+%m-%d %H:%M:%S')Z] $*" | tee -a $L; }
fail(){ say "FAILED: $*"; echo "$*" > $S/setup.failed; exit 1; }
rm -f $S/setup.ok $S/setup.failed
PY=$(command -v python); NP=$(nproc)
say "start | python $PY | nproc $NP | gpus $(nvidia-smi -L | wc -l) | disk free $(df -h $R | tail -n 1 | awk '{print $4}')"
command -v curl > /dev/null || { apt-get update -qq >> $L 2>&1 && apt-get install -y -qq curl >> $L 2>&1; } || fail "curl"
$PY -m pip install -q --no-cache-dir timm==1.0.29 pandas scikit-learn opencv-python-headless >> $L 2>&1 || fail "pip"
$PY -c "import torch, timm, cv2, pandas, sklearn; print('versions torch', torch.__version__, 'timm', timm.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'bf16', torch.cuda.is_bf16_supported())" >> $L 2>&1 || fail "imports"
cd $R/in
while IFS=$'\t' read -r name url size; do
  [ -z "$name" ] && continue
  [ -s "$name" ] && continue
  ( curl -sS -L --http1.1 --retry 8 --retry-all-errors --retry-delay 5 -o "$name.part" "$url" && mv "$name.part" "$name" || echo "FAIL $name" >> $L ) &
  while [ $(jobs -r | wc -l) -ge 8 ]; do sleep 2; done
done < $K/urls.tsv
wait
say "download done: $(ls *.tar 2>/dev/null | wc -l) tars, $(du -sh . | cut -f1)"
$PY - >> $L 2>&1 <<'PYV' || fail "part sha256"
import json, hashlib, glob, os, sys
bad = n = 0
for m in sorted(glob.glob('raptor_d96_manifest_*.json')):
    j = json.load(open(m)); assert j['status'] == 'OK', m
    for f, sh in j['parts'].items():
        n += 1
        h = hashlib.sha256(open(f, 'rb').read()).hexdigest() if os.path.exists(f) else 'missing'
        if h != sh: bad += 1; print('BAD', f)
print(f'PARTS_VERIFIED {n - bad}/{n}'); sys.exit(1 if bad or n != 33 else 0)
PYV
cd $R
$PY $K/unpack_d96.py --parts $R/in --receipts $R/in/raptor_d96_receipt_0.csv $R/in/raptor_d96_receipt_1.csv $R/in/raptor_d96_receipt_2.csv \
    --index $R/rsna_index.csv --out $R/rsna --nproc $(( NP > 16 ? 16 : NP )) >> $L 2>&1 || fail "unpack"
rm -f $R/in/*.tar
$PY - >> $L 2>&1 <<'PYO' || fail "OAI ASSERTION"
import os, pandas as pd
R = '/workspace/rsna'
names = [os.path.join(d, f) for d, _, fs in os.walk(R) for f in fs if 'site-packages' not in d]
bad = [p for p in names if 'oai' in os.path.basename(p).lower()]
i = pd.read_csv(f'{R}/rsna_index.csv', dtype=str)
bad += [x for x in list(i.id) + list(i.path) if 'oai' in x.lower()]
for c in (f'{R}/kit/labels_raptor_rsna_blendjev_sol.csv', f'{R}/kit/gold74_eval_set.csv'):
    t = open(c).read().lower(); bad += [c] if 'oai_' in t else []
assert len(i) == 4407 and not bad, f'OAI ASSERTION FAILED: {bad[:5]} rows {len(i)}'
print('OAI_ASSERT_OK: 4407 RSNA studies, no oai name / id / path / label anywhere under', R)
PYO
HF_HOME=$R/hf $PY -c "
import timm, glob, hashlib
timm.create_model('coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', pretrained=True, num_classes=0, img_size=384, global_pool='avg')
f = glob.glob('$R/hf/hub/models--timm--coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k/blobs/*')
print('pretrained blobs', [(x.split('/')[-1][:16], hashlib.sha256(open(x,'rb').read()).hexdigest()[:16]) for x in f])
" >> $L 2>&1 || fail "pretrained download"
say "SETUP OK | rsna npy $(ls $R/rsna | wc -l) | disk free $(df -h $R | tail -n 1 | awk '{print $4}')"
date +%s > $S/setup.ok
