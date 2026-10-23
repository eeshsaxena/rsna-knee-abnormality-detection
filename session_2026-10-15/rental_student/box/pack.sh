#!/bin/bash
# pack.sh MODE [OUT] (BOX, CPU only: nice 19, ionice idle, cores ${CORES:-30-39}; GPU, ~/queue and ~/fleet_runs untouched).
# Builds the T_b rental bundle = EXACTLY what the rented student needs, laid out as the instance root /workspace/rsna:
#   fast_build/code2/*.py                      the trainer code (no __pycache__, no other files)
#   fleet_store/<uid>.npz                      HARD LINKS to ~/fleet_store (4,407 RSNA studies, 11.66 GB; no copy)
#   r7t_labels/real/labels_r7t_b.csv           the T_b table (box sheet path, so the argv is token-identical to the sheet)
#   rsna_train/folds5.csv, gold74_eval_set.csv (FX4: --eval_labels = the T_b table, so v4 is NOT shipped)
#   hf/hub/models--timm--convnext_tiny.fb_in22k_ft_in1k   pretrained weights (blob sha256 77b619e1..)
#   kit/                                       instance scripts + kit.env + argv_template + pins.txt + parity_box.json
#   MANIFEST.sha256 (every regular file), BUNDLE.json (receipt), BUNDLE_OK (written last)
# MODE REAL: table = ~/r7t_labels/final/labels_r7t_b.csv (or FINAL_TABLE) whose sha256 is in ~/r7t_labels/final/LABELS_READY;
#            provenance (build receipt + fleet receipt) must be v4 + OAI-free fleet OOF + R7 OOF, else STOP (exit 3).
# MODE DRY:  table = the A1 dry-run SIM table (~/r7t_labels/sim_a1/final/b1/SIM_labels_r7t_b.csv); BUNDLE.json mode DRY
#            (launch.sh refuses a DRY bundle). SUBSET=FILE (tests): store = only the listed study ids.
# OUT default ~/rstud/bundle (built in OUT.tmp, then swapped in). Log ~/rstud/logs/pack.log.
set -u
MODE=${1:?usage: pack.sh REAL|DRY [OUT]}; OUT=${2:-$HOME/rstud/bundle}; SRC=$HOME/rstud/src
PY=$HOME/miniconda3/envs/rsna/bin/python; CORES=${CORES:-30-39}; NICE="nice -n 19 ionice -c3 taskset -c $CORES"
mkdir -p $HOME/rstud/logs $HOME/rstud/cache; LOG=$HOME/rstud/logs/pack.log
say(){ echo "[pack $(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }
die(){ say "STOP: $*"; rm -rf "$OUT.tmp"; exit 3; }
case $MODE in REAL|DRY) ;; *) die "MODE must be REAL or DRY";; esac
. $SRC/remote/kit.env
FIN=$HOME/r7t_labels/final
if [ $MODE = REAL ]; then
  [ -e $FIN/LABELS_READY ] || die "~/r7t_labels/final/LABELS_READY does not exist yet (labels not final)"
  TAB=${FINAL_TABLE:-$FIN/labels_r7t_b.csv}; [ -f "$TAB" ] || die "final T_b table $TAB missing"
  RCPT=""; for c in $FIN/teacher_labels_receipt.json $HOME/r7t_labels/real/teacher_labels_receipt.json; do [ -f $c ] && { RCPT=$c; break; }; done
  FRC=$HOME/r7t_labels/fleet/fleet_oof_receipt_fleetOF.json
  CHK=(--mode REAL --ready $FIN/LABELS_READY --receipt "$RCPT" --fleet_receipt $FRC --real_copy $HOME/r7t_labels/real/$TABLE
       --verdict $HOME/r7t_labels/real/labels_gate_labels_r7t_b.verdict)
  [ -n "$RCPT" ] || die "no label build receipt (final/ or real/teacher_labels_receipt.json)"
else
  TAB=${DRY_TABLE:-$HOME/r7t_labels/sim_a1/final/b1/SIM_labels_r7t_b.csv}
  CHK=(--mode DRY --receipt "$(dirname $TAB)/SIM_teacher_labels_receipt.json" --fleet_receipt $HOME/r7t_labels/fleet/fleet_oof_receipt_fleetOF.json)
  [ -n "${DRY_READY:-}" ] && CHK+=(--ready $DRY_READY)
fi
say "== pack $MODE -> $OUT | table $TAB | subset ${SUBSET:-none}"
# 0 sources: never an OAI / union path, never a symlinked store, trainer md5 = PREREG frozen
for p in $HOME/fast_build/code2 $HOME/fleet_store "$TAB" $HOME/rsna_train/folds5.csv $HOME/rsna_train/gold74_eval_set.csv \
         $HOME/.cache/huggingface/hub/models--timm--convnext_tiny.fb_in22k_ft_in1k; do
  rp=$(readlink -f "$p"); case $rp in *oai*|*OAI*|*union*) die "source $p resolves to $rp (OAI data never leaves the box)";; esac
done
[ -z "$(find $HOME/fleet_store -maxdepth 1 -type l)" ] || die "symlinks in ~/fleet_store"
[ "$(md5sum < $HOME/fast_build/code2/fleet_train.py | cut -c1-32)" = "$TRAINER_MD5" ] || die "fleet_train.py md5 != PREREG frozen $TRAINER_MD5"
$PY $SRC/box/pack_check.py table "$TAB" "${CHK[@]}" | tee -a $LOG; [ "${PIPESTATUS[0]}" = 0 ] || die "T_b table check (see above)"
# 1 stage the tree
rm -rf "$OUT.tmp"; mkdir -p "$OUT.tmp"/{fast_build/code2,fleet_store,r7t_labels/real,rsna_train,hf/hub,kit}
cp -p $HOME/fast_build/code2/*.py "$OUT.tmp/fast_build/code2/"
if [ -n "${SUBSET:-}" ]; then while read -r s; do [ -n "$s" ] && ln "$HOME/fleet_store/$s.npz" "$OUT.tmp/fleet_store/$s.npz"; done < "$SUBSET"
else ( cd $HOME/fleet_store && find . -maxdepth 1 -name '*.npz' -printf '%f\0' | xargs -0 -n 500 ln -t "$OUT.tmp/fleet_store/" ) || die "hard-linking the store"; fi
cp -p "$TAB" "$OUT.tmp/r7t_labels/real/$TABLE"
cp -p $HOME/rsna_train/folds5.csv $HOME/rsna_train/gold74_eval_set.csv "$OUT.tmp/rsna_train/"
cp -a $HOME/.cache/huggingface/hub/models--timm--convnext_tiny.fb_in22k_ft_in1k "$OUT.tmp/hf/hub/"
rm -rf "$OUT.tmp/hf/hub/models--timm--convnext_tiny.fb_in22k_ft_in1k/.no_exist"
cp -p $SRC/remote/{guard.sh,rstud_lane.sh,start_lanes.sh,stop_fold.sh,status.sh,setup.sh,parity.py,kit.env,argv_template.txt} "$OUT.tmp/kit/"
"${PY%python}pip" list --format=freeze 2>/dev/null | grep -i -E '^(opencv-python-headless|opencv-python|numpy|pandas|scikit-learn|scipy|timm|huggingface[-_]hub|safetensors|pyyaml|pillow)==' | sort -f > "$OUT.tmp/kit/pins.txt"
[ "$(wc -l < "$OUT.tmp/kit/pins.txt")" -ge 10 ] || die "could not read the box package pins"
HB=$(sha256sum "$OUT.tmp/hf/hub/models--timm--convnext_tiny.fb_in22k_ft_in1k/blobs/77b619e187de71bce58721689da0c009e84cc20677293d7d43105307e34d868a" | cut -c1-64)
[ "$HB" = 77b619e187de71bce58721689da0c009e84cc20677293d7d43105307e34d868a ] || die "HF blob sha256 $HB"
say "staged: code $(ls "$OUT.tmp/fast_build/code2" | wc -l) py, store $(ls "$OUT.tmp/fleet_store" | wc -l) npz (hard links), table, 2 csvs, HF cache, kit $(ls "$OUT.tmp/kit" | wc -l) files"
# 2 OAI assertion on the staged tree
$PY $SRC/box/pack_check.py oai "$OUT.tmp" ${SUBSET:+--subset} | tee -a $LOG; [ "${PIPESTATUS[0]}" = 0 ] || die "OAI assertion"
# 3 parity probe on the bundle itself (box CPU; HF offline from the bundled cache) -> kit/parity_box.json
( cd "$OUT.tmp/fast_build/code2" && TTR=$TTR CUDA_VISIBLE_DEVICES= HF_HOME="$OUT.tmp/hf" HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 \
  $NICE $PY "$OUT.tmp/kit/parity.py" --code "$OUT.tmp/fast_build/code2" --store "$OUT.tmp/fleet_store" --labels "$OUT.tmp/r7t_labels/real/$TABLE" \
  --folds_csv "$OUT.tmp/rsna_train/folds5.csv" --exclude "$OUT.tmp/rsna_train/gold74_eval_set.csv" --folds $FOLDS ) > "$OUT.tmp/kit/parity_box.json" 2> $HOME/rstud/logs/parity_box.err \
  || { tail -n 5 $HOME/rstud/logs/parity_box.err | tee -a $LOG; die "parity.py on the bundle"; }
say "parity_box: $($PY -c "import json; d=json.load(open('$OUT.tmp/kit/parity_box.json')); print('model_sha', d['model_sha'], '| n', d['n'], '| folds', d['fold_counts'], '| backbone', d.get('backbone_sha'))")"
# 4 manifest: sha256 of every regular file (store hashes cached by name + size + mtime_ns in ~/rstud/cache; misses hashed by 6
#   nice-19 processes pinned to $CORES); symlinks (hf snapshot -> blob) are not listed, setup.sh proves them by loading the model
$PY - "$OUT.tmp" $HOME/fleet_store $HOME/rstud/cache/store_sha256.tsv "$CORES" <<'PYM' || die "manifest"
import os, sys, hashlib
from multiprocessing import Pool
B, ST, C, CORES = sys.argv[1:5]
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
    if rel.startswith('fleet_store/'):
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
# spot re-hash 64 random store files against the manifest (cache sanity)
( cd "$OUT.tmp" && grep '  fleet_store/' MANIFEST.sha256 | shuf -n 64 --random-source=<(yes) | $NICE sha256sum -c --quiet ) || die "manifest spot check failed (stale cache?)"
TSHA=$(sha256sum "$OUT.tmp/r7t_labels/real/$TABLE" | cut -c1-64); MSHA=$(sha256sum "$OUT.tmp/MANIFEST.sha256" | cut -c1-64)
$PY - "$OUT.tmp/BUNDLE.json" <<PYB
import json, sys, time
json.dump(dict(mode="$MODE", created=time.strftime('%Y-%m-%d %H:%M:%S %Z'), table_source="$TAB", table="r7t_labels/real/$TABLE",
               table_sha256="$TSHA", labels_ready_sha256="$( [ $MODE = REAL ] && sha256sum $FIN/LABELS_READY | cut -c1-64 )",
               manifest_files=$NMAN, manifest_sha256="$MSHA", bytes=$BYTES, subset="${SUBSET:-}", runtag="$RUNTAG", seed=$SEED,
               trainer_md5="$TRAINER_MD5", argv_template=open("$OUT.tmp/kit/argv_template.txt").read().strip(),
               oai_assert="PASS", table_check="PASS"), open(sys.argv[1], 'w'), indent=1)
PYB
date '+%F %T' > "$OUT.tmp/BUNDLE_OK"
rm -rf "$OUT.prev"; [ -e "$OUT" ] && mv "$OUT" "$OUT.prev"; mv "$OUT.tmp" "$OUT"; rm -rf "$OUT.prev"
say "BUNDLE READY ($MODE): $OUT | $NMAN files, $((BYTES / 1000000)) MB | table sha256 ${TSHA:0:16} | manifest sha256 ${MSHA:0:16}"
