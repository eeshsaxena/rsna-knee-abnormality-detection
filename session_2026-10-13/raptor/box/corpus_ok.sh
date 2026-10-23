#!/bin/bash
# corpus_ok.sh (box CPU): after fetch_box.sh, unpack the 33 Kaggle tar parts to ~/raptor_store/rsna (receipt sha256 per
# study), byte-compare the 458 studies that also exist in the box-mirror build, and write ~/raptor_store/CORPUS_OK only if:
# 3 manifests OK, 33/33 parts verified, 4,407 studies unpacked with sha OK, and the mirror compare is identical or within
# max |diff| <= 1 on <= 0.1 % of pixels (numpy / cv2 rounding; documented). Otherwise CORPUS_HOLD with the reason.
set -u
L=$HOME/raptor_store/corpus_ok.log; IN=$HOME/raptor_in; PY=$HOME/miniconda3/envs/rsna/bin/python
until grep -q PARTS_VERIFIED $IN/fetch.log 2>/dev/null; do sleep 20; done
grep PARTS_VERIFIED $IN/fetch.log >> $L
grep -q "PARTS_VERIFIED 33/33" $IN/fetch.log || { echo "CORPUS_HOLD parts not 33/33" >> $L; exit 1; }
[ $(grep -c "json OK " $IN/fetch.log) = 3 ] || { echo "CORPUS_HOLD manifest status not OK x3" >> $L; exit 1; }
cd $HOME/raptor2 && nice -n 19 taskset -c 8-15 $PY unpack_d96.py --parts $IN --receipts $IN/raptor_d96_receipt_0.csv $IN/raptor_d96_receipt_1.csv $IN/raptor_d96_receipt_2.csv \
   --index $HOME/raptor_store/rsna_index.csv --out $HOME/raptor_store/rsna --compare_dir $HOME/raptor_store/mirror_npy >> $L 2>&1 || { echo "CORPUS_HOLD unpack failed" >> $L; exit 1; }
$PY - >> $L 2>&1 <<'PY'
import pandas as pd, os, sys
H = os.path.expanduser('~')
i = pd.read_csv(f'{H}/raptor_store/rsna_index.csv'); c = pd.read_csv(f'{H}/raptor_store/rsna_index.csv.compare.csv')
ok_n = len(i) == 4407 and i.id.is_unique
ident = int((c.same_as_compare == 1).sum()); maxd = int(c.max_abs_diff.max()); frac = float(c.frac_px_diff.max())
ok_cmp = len(c) == 458 and (ident == 458 or (maxd <= 1 and frac <= 1e-3))
msg = f'rows {len(i)} | mirror compare {ident}/{len(c)} identical, max |diff| {maxd}, worst pixel fraction {frac:.2e}'
if ok_n and ok_cmp:
    open(f'{H}/raptor_store/CORPUS_OK', 'w').write(msg + '\n'); print('CORPUS_OK', msg)
else:
    print('CORPUS_HOLD', msg); sys.exit(1)
PY
