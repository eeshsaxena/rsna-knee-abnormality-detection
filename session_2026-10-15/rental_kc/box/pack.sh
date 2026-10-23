#!/bin/bash
# pack.sh [OUT] (rental_kc, BOX, CPU only: nice 19, ionice idle, cores ${CORES:-30-39}; GPU, ~/queue and ~/fleet_runs untouched).
# Builds the KC rental bundle = EXACTLY what the rented members need, laid out as the instance root /workspace/rsna so that
# every queue-line path ~/X becomes /workspace/rsna/X (argv token-identical except paths):
#   fast_build/code2/*.py                         the INSTALLED box trainer code (shimmed fleet_train.py md5 8d12515f +
#                                                 fleet_train_kc.py + kc_common.py + kc_g1.py + the fleet modules)
#   kc_store/store/<uid>.npz                      HARD LINKS to ~/kc_store/store (4,407 RSNA studies, ~13 GB; no copy)
#   r7t_labels/real/labels_r7t_b.csv             T_b (md5 d206bbcf, the 10-14 verified RSNA-only table)
#   kc/labels/labels_dread_fleet.csv              dread soft labels (RSNA reports; make_dread_labels.py receipt)
#   rsna_train/folds5.csv, gold74_eval_set.csv
#   r7port/pretrained/convnext_small_in22k_ft_in1k.safetensors   (sha256 2457488b)
#   hf/hub/models--timm--tf_efficientnetv2_s.in21k_ft_in1k     (KC2; blob sha256 6f1933fb)
#   kit/                                          instance scripts + kit.env + members.tsv + pins.txt + parity_box.json
#   MANIFEST.sha256 (every regular file), BUNDLE.json (receipt incl. code md5s), BUNDLE_OK (written last)
# STOP (exit 3) unless: shim installed, KC code present, tables OK (pack_check tables), OAI assertion OK, parity probe OK.
# OUT default ~/rkc/bundle (built in OUT.tmp, then swapped in). Log ~/rkc/logs/pack.log.
set -u
OUT=${1:-$HOME/rkc/bundle}; SRC=$HOME/rkc/src
PY=$HOME/miniconda3/envs/rsna/bin/python; CORES=${CORES:-30-39}; NICE="nice -n 19 ionice -c3 taskset -c $CORES"
mkdir -p $HOME/rkc/logs $HOME/rkc/cache; LOG=$HOME/rkc/logs/pack.log
say(){ echo "[pack $(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }
die(){ say "STOP: $*"; rm -rf "$OUT.tmp"; exit 3; }
. $SRC/remote/kit.env
CODE=${RKC_CODE:-$HOME/fast_build/code2}; STORE=$HOME/kc_store/store; TB=$HOME/r7t_labels/real/labels_r7t_b.csv
DR=${RKC_DR:-$HOME/kc/labels/labels_dread_fleet.csv}; PRE=$HOME/r7port/pretrained/convnext_small_in22k_ft_in1k.safetensors
HFM=$HOME/.cache/huggingface/hub/models--timm--tf_efficientnetv2_s.in21k_ft_in1k
DRC=${RKC_DRC:-}; [ -n "$DRC" ] || for c in $DR.receipt.json $HOME/kc/labels/labels_dread_fleet.receipt.json $HOME/kc/labels/receipt.json; do [ -f $c ] && { DRC=$c; break; }; done
say "== pack -> $OUT"
# 0 sources: never an OAI / union path, no symlinked store, shim + KC code installed
for p in $CODE $STORE $TB $DR $PRE $HFM $HOME/rsna_train/folds5.csv $HOME/rsna_train/gold74_eval_set.csv; do
  [ -e "$p" ] || die "missing source $p"
  rp=$(readlink -f "$p"); case $rp in *oai*|*OAI*|*union*) die "source $p resolves to $rp (OAI data never leaves the box)";; esac
done
[ -n "$DRC" ] || die "no receipt for $DR (make_dread_labels.py writes one)"
[ -z "$(find $STORE -maxdepth 1 -type l)" ] || die "symlinks in $STORE"
[ "$(md5sum < $CODE/fleet_train.py | cut -c1-32)" = "$SHIM_MD5" ] || die "code2 fleet_train.py md5 != the shimmed $SHIM_MD5 (M4 not installed yet)"
for f in fleet_train_kc.py kc_common.py kc_g1.py fleet_model.py fleet_model_ttr.py fleet_model_f2.py fleet_aug.py; do [ -f $CODE/$f ] || die "code2/$f missing (C8 not installed?)"; done
$PY $SRC/box/pack_check.py tables $TB $DR --ready $HOME/r7t_labels/final/LABELS_READY --dread_receipt $DRC | tee -a $LOG
[ "${PIPESTATUS[0]}" = 0 ] || die "label table check (see above)"
# 1 member registry from the live queue (argv templates, ~ -> @R@)
bash $SRC/box/ops.sh members $HOME/rkc/members.tsv.new | tee -a $LOG; [ "${PIPESTATUS[0]}" = 0 ] || die "member registry"
# 2 stage the tree
rm -rf "$OUT.tmp"; mkdir -p "$OUT.tmp"/{fast_build/code2,kc_store/store,r7t_labels/real,kc/labels,kc/gate,rsna_train,r7port/pretrained,hf/hub,kit}
cp -p $CODE/*.py "$OUT.tmp/fast_build/code2/"
if [ -n "${RKC_SUBSET:-}" ]; then while read -r s; do [ -n "$s" ] && ln "$STORE/$s.npz" "$OUT.tmp/kc_store/store/$s.npz"; done < "$RKC_SUBSET"   # tests only
else ( cd $STORE && find . -maxdepth 1 -name '*.npz' -printf '%f\0' | xargs -0 -n 500 ln -t "$OUT.tmp/kc_store/store/" ) || die "hard-linking the store"; fi
say "store linked: $(ls "$OUT.tmp/kc_store/store" | wc -l) npz"
cp -p $TB "$OUT.tmp/r7t_labels/real/"; cp -p $DR "$OUT.tmp/kc/labels/labels_dread_fleet.csv"
cp -p $HOME/rsna_train/folds5.csv $HOME/rsna_train/gold74_eval_set.csv "$OUT.tmp/rsna_train/"
cp -p $PRE "$OUT.tmp/r7port/pretrained/"
cp -a $HFM "$OUT.tmp/hf/hub/"; rm -rf "$OUT.tmp/hf/hub/$(basename $HFM)/.no_exist"
cp -p $SRC/remote/{guard.sh,kc_lane.sh,start_lanes.sh,stop_member.sh,status.sh,setup.sh,parity.py,kit.env} "$OUT.tmp/kit/"
cp -p $HOME/rkc/members.tsv.new "$OUT.tmp/kit/members.tsv"
"${PY%python}pip" list --format=freeze 2>/dev/null | grep -i -E '^(opencv-python-headless|opencv-python|numpy|pandas|scikit-learn|scipy|timm|huggingface[-_]hub|safetensors|pyyaml|pillow)==' | sort -f > "$OUT.tmp/kit/pins.txt"
[ "$(wc -l < "$OUT.tmp/kit/pins.txt")" -ge 10 ] || die "could not read the box package pins"
HB=$(sha256sum "$OUT.tmp/hf/hub/$(basename $HFM)/blobs/$HF_EV2S_BLOB" | cut -c1-64); [ "$HB" = "$HF_EV2S_BLOB" ] || die "HF blob sha256 $HB"
PS=$(sha256sum "$OUT.tmp/r7port/pretrained/$(basename $PRE)" | cut -c1-16); [ "$PS" = "$PRETRAINED_SHA16" ] || die "pretrained sha256 $PS"
say "staged: code $(ls "$OUT.tmp/fast_build/code2" | wc -l) py, store $(ls "$OUT.tmp/kc_store/store" | wc -l) npz (hard links), 2 tables, 2 csvs, ConvNeXt-S safetensors, EffNetV2-S HF cache, kit $(ls "$OUT.tmp/kit" | wc -l) files"
# 3 OAI assertion on the staged tree
$PY $SRC/box/pack_check.py oai "$OUT.tmp" ${RKC_SUBSET:+--subset} | tee -a $LOG; [ "${PIPESTATUS[0]}" = 0 ] || die "OAI assertion"
# 4 parity probe on the bundle itself (box CPU; HF offline from the bundled cache) -> kit/parity_box.json
( cd "$OUT.tmp/fast_build/code2" && CUDA_VISIBLE_DEVICES= HF_HOME="$OUT.tmp/hf" HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 \
  $NICE $PY "$OUT.tmp/kit/parity.py" --code "$OUT.tmp/fast_build/code2" --store "$OUT.tmp/kc_store/store" \
  --tables "$OUT.tmp/r7t_labels/real/labels_r7t_b.csv" "$OUT.tmp/kc/labels/labels_dread_fleet.csv" \
  --pretrained "$OUT.tmp/r7port/pretrained/$(basename $PRE)" ) > "$OUT.tmp/kit/parity_box.json" 2> $HOME/rkc/logs/parity_box.err \
  || { tail -n 5 $HOME/rkc/logs/parity_box.err | tee -a $LOG; die "parity.py on the bundle"; }
say "parity_box: $($PY -c "import json; d=json.load(open('$OUT.tmp/kit/parity_box.json')); print('contract', d['contract_hash'], '| model_sha ttr', d['model_sha_ttr'], 'f2', d['model_sha_f2'], '| store', d['n_store'], '| cnxs', d.get('backbone_sha_cnxs'), '| ev2s', d.get('backbone_sha_ev2s'))")"
grep -q "\"contract_hash\": \"$CONTRACT_HASH\"" "$OUT.tmp/kit/parity_box.json" || die "kc_common CONTRACT_HASH != the A6 contract $CONTRACT_HASH"
# 5 manifest: sha256 of every regular file (store hashes cached by name + size + mtime_ns in ~/rkc/cache; misses hashed by 6
#   nice-19 processes pinned to $CORES); symlinks (hf snapshot -> blob) are not listed, setup.sh proves them by loading the model
$PY - "$OUT.tmp" $HOME/rkc/cache/store_sha256.tsv "$CORES" <<'PYM' || die "manifest"
import os, sys, hashlib
from multiprocessing import Pool
B, C, CORES = sys.argv[1:4]
def cores(spec):
    out = set()
    for part in spec.split(','):
        a, _, b = part.partition('-'); out |= set(range(int(a), int(b or a) + 1))
    return out
def init():
    os.nice(19)
    try: os.sched_setaffinity(0, cores(CORES))
    except Exception: pass
def h(p):
    m = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 22), b''): m.update(blk)
    return m.hexdigest()
cache = {}
if os.path.exists(C):
    for l in open(C):
        n, sz, mt, hx = l.rstrip(chr(10)).split(chr(9)); cache[(n, sz, mt)] = hx
files = []
for d, _, fs in os.walk(B):
    for f in fs:
        p = os.path.join(d, f); rel = os.path.relpath(p, B)
        if os.path.islink(p) or rel in ('MANIFEST.sha256', 'BUNDLE.json', 'BUNDLE_OK'): continue
        files.append((rel, p))
keys, todo = {}, []
for rel, p in files:
    if rel.startswith('kc_store/store/'):
        st = os.stat(p); k = (os.path.basename(p), str(st.st_size), str(st.st_mtime_ns)); keys[rel] = k
        if k not in cache: todo.append((rel, p))
    else: todo.append((rel, p))
with Pool(6, initializer=init) as pool:
    got = dict(zip([r for r, _ in todo], pool.map(h, [p for _, p in todo], chunksize=8)))
nmiss = 0
for rel, k in keys.items():
    if rel in got: cache[k] = got[rel]; nmiss += 1
with open(C + '.tmp', 'w') as fh:
    for (n, sz, mt), hx in sorted(cache.items()): fh.write(chr(9).join((n, sz, mt, hx)) + chr(10))
os.replace(C + '.tmp', C)
out = sorted(f'{got.get(rel) or cache[keys[rel]]}  {rel}' for rel, _ in files)
out.sort(key=lambda l: l[66:])
open(f'{B}/MANIFEST.sha256', 'w').write(chr(10).join(out) + chr(10))
print(f'manifest: {len(out)} files ({nmiss} store files hashed now, {len(keys) - nmiss} from the cache)')
PYM
NMAN=$(grep -c . "$OUT.tmp/MANIFEST.sha256"); BYTES=$(du -sbL "$OUT.tmp" | cut -f1)
( cd "$OUT.tmp" && grep '  kc_store/store/' MANIFEST.sha256 | shuf -n 64 --random-source=<(yes) | $NICE sha256sum -c --quiet ) || die "manifest spot check failed (stale cache?)"
MSHA=$(sha256sum "$OUT.tmp/MANIFEST.sha256" | cut -c1-64)
$PY - "$OUT.tmp/BUNDLE.json" "$OUT.tmp" <<PYB
import json, sys, time, hashlib, glob, os
B = sys.argv[2]; sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
json.dump(dict(mode="REAL", kit="rental_kc", subset="${RKC_SUBSET:+1}", created=time.strftime('%Y-%m-%d %H:%M:%S %Z'),
               tables={'labels_r7t_b.csv': sha(f'{B}/r7t_labels/real/labels_r7t_b.csv'), 'labels_dread_fleet.csv': sha(f'{B}/kc/labels/labels_dread_fleet.csv')},
               dread_receipt="$DRC", code_md5={os.path.basename(p): hashlib.md5(open(p, 'rb').read()).hexdigest() for p in sorted(glob.glob(f'{B}/fast_build/code2/*.py'))},
               members=open(f'{B}/kit/members.tsv').read().splitlines(), manifest_files=$NMAN, manifest_sha256="$MSHA", bytes=$BYTES,
               oai_assert="PASS", table_check="PASS", parity_box=json.load(open(f'{B}/kit/parity_box.json'))), open(sys.argv[1], 'w'), indent=1)
PYB
date '+%F %T' > "$OUT.tmp/BUNDLE_OK"
rm -rf "$OUT.prev"; [ -e "$OUT" ] && mv "$OUT" "$OUT.prev"; mv "$OUT.tmp" "$OUT"; rm -rf "$OUT.prev"
say "BUNDLE READY: $OUT | $NMAN files, $((BYTES / 1000000)) MB | manifest sha256 ${MSHA:0:16} | code2 fleet_train_kc.py md5 $(md5sum < $OUT/fast_build/code2/fleet_train_kc.py | cut -c1-8)"
