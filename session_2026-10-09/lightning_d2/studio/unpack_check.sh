#!/bin/bash
# unpack_check.sh: STUDIO side (free CPU Studio, zero credits). After all tars are uploaded to ~/rsna_d2/upload:
#  1. md5 of every tar == the box (meta/tar_md5.txt)
#  2. unpack into ~/rsna (same layout as the box home: fast_build/code2, novel/frac75, rsna_train, hf, fleet_store)
#  3. md5 of all 4407 store files == the box manifest (meta/store_md5.txt) and of the 18 code/csv/HF files
#     (meta/small_md5.txt); file lists must match exactly
#  4. parity.py (CPU, TTR=224) == the box json for contract, select_sha, jpeg_fp, model_sha, n, fold_counts,
#     decode_probe, backbone_sha, state_keys, pretrained_cfg
#  5. delete the uploaded tars (smaller Studio snapshot for the job), re-check the venv imports, write ~/rsna_d2/PARITY_OK
set -eu
U=~/rsna_d2/upload; M=~/rsna_d2/meta; R=~/rsna; P=~/rsna_d2/venv/bin/python
say(){ echo "[check $(date '+%F %T')] $*"; }
rm -f ~/rsna_d2/PARITY_OK
cd $U && md5sum -c $M/tar_md5.txt || { say "FAIL: tar md5"; exit 1; }
say "tar md5 OK (7 tars)"
rm -rf $R; mkdir -p $R/fast_build $R/fleet_store
tar -xf $U/small.tar -C $R
mv $R/code2 $R/fast_build/code2
for k in 0 1 2 3 4 5; do tar -xf $U/store_$k.tar -C $R/fleet_store; done
n=$(find $R/fleet_store -maxdepth 1 -type f -name '*.npz' | wc -l); [ "$n" = 4407 ] || { say "FAIL: $n npz"; exit 1; }
diff <(sed 's/^[0-9a-f]* *//; s#^\*##; s#^\./##' $M/store_md5.txt | LC_ALL=C sort) <(cd $R/fleet_store && ls | LC_ALL=C sort) > /dev/null || { say "FAIL: store file list differs from the box manifest"; exit 1; }
t0=$(date +%s)
(cd $R/fleet_store && md5sum -c --quiet $M/store_md5.txt) || { say "FAIL: store md5"; exit 1; }
say "store 4407/4407 md5 IDENTICAL to the box ($(( $(date +%s) - t0 ))s, $(du -sb $R/fleet_store | cut -f1) bytes)"
(cd $R && sed 's#  code2/#  fast_build/code2/#' $M/small_md5.txt | md5sum -c --quiet -) || { say "FAIL: small md5"; exit 1; }
say "code/csv/HF $(wc -l < $M/small_md5.txt) files md5 IDENTICAL to the box"
cd $R/fast_build/code2
TTR=224 CUDA_VISIBLE_DEVICES= HF_HOME=$R/hf HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 $P $M/parity.py --code $R/fast_build/code2 \
  --store $R/fleet_store --labels $R/novel/frac75/labels_frac75_logit.csv --folds_csv $R/rsna_train/folds5.csv \
  --exclude $R/rsna_train/gold74_eval_set.csv --folds 0 1 2 3 4 > ~/rsna_d2/studio_parity_ttr224.json 2> ~/rsna_d2/logs/parity.err \
  || { tail -n 5 ~/rsna_d2/logs/parity.err; say "FAIL: parity.py"; exit 1; }
$P - $M/box_parity_ttr224.json ~/rsna_d2/studio_parity_ttr224.json <<'EOF'
import json, sys
b, i = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])); bad = 0
for k in ('contract', 'select_sha', 'jpeg_fp', 'model_sha', 'n', 'fold_counts', 'decode_probe', 'backbone_sha', 'state_keys', 'pretrained_cfg'):
    same = b.get(k) == i.get(k); bad += not same
    print('  %-14s %s %s' % (k, 'IDENTICAL' if same else 'DIFFERS', json.dumps(b.get(k))[:90] if same else 'box %s | studio %s' % (b.get(k), i.get(k))))
print('  versions box    %s' % b['versions']); print('  versions studio %s' % i['versions'])
sys.exit(1 if bad else 0)
EOF
say "parity.py: every hash and count IDENTICAL to the box"
find $R -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
rm -f $U/*.tar
$P -c 'import torch, torchvision, cv2, timm, sklearn, pandas; print("VENV_IMPORT_OK torch", torch.__version__)'
du -sh ~/rsna ~/rsna_d2 2>/dev/null
date '+%F %T' > ~/rsna_d2/PARITY_OK
say "PARITY_OK"
