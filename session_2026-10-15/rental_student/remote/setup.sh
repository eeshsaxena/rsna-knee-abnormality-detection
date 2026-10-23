#!/bin/bash
# setup.sh (instance, detached by the box driver after the bundle push; RSNA_ROOT + SETUP_MOCK=1 for the box CPU mock).
# Writes status/setup.ok or status/setup.failed; log $R/setup.log.
#  1 MANIFEST: every file of the pushed bundle sha256-checked against MANIFEST.sha256 (8 parallel checkers); no file
#    under the bundle dirs that the manifest does not list (no extras)
#  2 OAI ASSERTION (fail loud, same rules as the box packer): no file name / path / csv cell mentions oai or union, the
#    store holds exactly the RSNA studies of the manifest (RSNA UID pattern), table ids == store ids
#  3 packages pinned to the box rsna env (kit/pins.txt; torch / torchvision of the image kept; opencv --no-deps as on the
#    box), then pip freeze == pins (the 10-10 vastS1044 recipe)
#  4 GPU facts: >= NG GPUs, torch CUDA, a matmul on every GPU, /dev/shm >= 8 GB, >= 25 GB free disk
#  5 parity.py (CPU) -> status/parity_inst.json (the box compares it with kit/parity_box.json made by the packer)
set -u
R=${RSNA_ROOT:-/workspace/rsna}; K=$R/kit; S=$R/status; L=$R/setup.log; MOCK=${SETUP_MOCK:-0}
mkdir -p $S $R/logs
say(){ echo "[setup $(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
fail(){ say "FAILED: $*"; echo "$*" > $S/setup.failed; exit 1; }
rm -f $S/setup.ok $S/setup.failed
. $K/kit.env; . $K/plan.env
PYI=${PYI:-$( [ -x /opt/conda/bin/python ] && echo /opt/conda/bin/python || command -v python3 )}
say "start | python $PYI | nproc $(nproc) | mock $MOCK | disk free $(df -h $R | tail -n 1 | awk '{print $4}')"

# ---- 1 manifest ----
cd $R || fail "no $R"
[ -s MANIFEST.sha256 ] || fail "MANIFEST.sha256 missing"
NM=$(grep -c . MANIFEST.sha256)
rm -f /tmp/rstud_man_*; split -n l/8 -d MANIFEST.sha256 /tmp/rstud_man_
for p in /tmp/rstud_man_0*; do ( sha256sum -c --quiet "$p" > "$p.out" 2>&1; echo $? > "$p.rc" ) & done; wait
bad=$(cat /tmp/rstud_man_0*.out | grep -v -E ': OK$' | grep -c .)
[ "$bad" = 0 ] && ! grep -qv '^0$' /tmp/rstud_man_0*.rc || { cat /tmp/rstud_man_0*.out | head -n 5 >> $L; fail "manifest: $bad bad line(s) of $NM"; }
extra=$(comm -13 <(awk '{print $2}' MANIFEST.sha256 | sed 's#^\*##' | LC_ALL=C sort) \
                 <(find fast_build fleet_store r7t_labels rsna_train hf kit -type f | LC_ALL=C sort) | grep -v -E '^kit/(pins_check\.txt|plan\.env)$' )
[ -z "$extra" ] || fail "files not in the manifest: $(echo $extra | head -c 300)"
say "MANIFEST OK: $NM files sha256-verified, no extra files"

# ---- 2 OAI assertion ----
$PYI - "$R" >> $L 2>&1 <<'PYO' || fail "OAI ASSERTION"
import os, re, sys, csv
R = sys.argv[1]; bad = []
UID = re.compile(r'^1\.2\.826\.0\.1\.3680043\.8\.498\.\d+$')
for d, ds, fs in os.walk(R):
    if '/site-packages' in d or d.startswith('/proc'): continue
    for f in ds + fs:
        p = os.path.join(d, f)
        if re.search(r'oai|union', f, re.I): bad.append('name ' + p)
store = sorted(f for f in os.listdir(f'{R}/fleet_store'))
sids = [f[:-4] for f in store if f.endswith('.npz')]
if len(sids) != len(store): bad.append('non-npz file in fleet_store')
bad += ['store id ' + s for s in sids if not UID.match(s)][:5]
def ids(p):
    with open(p, newline='') as fh: return [r['StudyInstanceUID'] for r in csv.DictReader(fh)]
for c in ('r7t_labels/real/labels_r7t_b.csv', 'rsna_train/folds5.csv', 'rsna_train/gold74_eval_set.csv'):
    t = open(f'{R}/{c}', errors='replace').read()
    if re.search(r'oai|union', t, re.I): bad.append('content ' + c)
    bad += [f'{c} id {s}' for s in ids(f'{R}/{c}') if not UID.match(s)][:5]
tab = ids(f'{R}/r7t_labels/real/labels_r7t_b.csv')
import json
subset = json.load(open(f'{R}/BUNDLE.json')).get('subset')          # tests only: a store subset (the real bundle has none)
if not subset and len(sids) != 4407: bad.append(f'store holds {len(sids)} studies, want 4,407')
if set(tab) - set(sids) and not subset: bad.append(f'{len(set(tab) - set(sids))} table ids missing from the store')
if set(sids) - set(tab): bad.append(f'{len(set(sids) - set(tab))} store ids not in the table')
for p in [os.path.join(d, f) for d, _, fs in os.walk(f'{R}/fast_build') for f in fs]:
    if re.search(r'oai|union_store', open(p, errors='replace').read(), re.I): bad.append('code mentions oai: ' + p)
assert not bad, f'OAI ASSERTION FAILED: {bad[:8]}'
print(f'OAI_ASSERT_OK: {len(sids)} RSNA studies (UID pattern), table ids {"cover the test subset" if subset else "== store ids"} ({len(tab)}), no oai/union name, path, csv cell or code reference under {R}')
PYO

# ---- 3 packages ----
if [ "$MOCK" = 1 ]; then say "mock: pip step skipped (box env)"; else
  command -v pgrep > /dev/null && command -v pkill > /dev/null && command -v setsid > /dev/null \
    || { apt-get -qq update >> $L 2>&1; DEBIAN_FRONTEND=noninteractive apt-get -qq install -y procps util-linux >> $L 2>&1; }
  command -v pgrep > /dev/null && command -v pkill > /dev/null || fail "procps (pgrep / pkill) missing on the instance"
  PK=$(grep -v -i -E '^(torch|torchvision)==' $K/pins.txt | tr '\n' ' ')
  PND=$(printf '%s\n' $PK | grep -i '^opencv' | tr '\n' ' '); PD=$(printf '%s\n' $PK | grep -v -i '^opencv' | tr '\n' ' ')
  $PYI -c 'import torch, torchvision; print("torch==" + torch.__version__.split("+")[0]); print("torchvision==" + torchvision.__version__.split("+")[0])' > /tmp/constraints.txt || fail "torch import"
  $PYI -m pip install -q --no-cache-dir --retries 10 --timeout 60 -c /tmp/constraints.txt $PD >> $L 2>&1 || fail "pip (deps)"
  $PYI -m pip install -q --no-cache-dir --retries 10 --timeout 60 --no-deps $PND >> $L 2>&1 || fail "pip (opencv no-deps)"
  $PYI -m pip list --format=freeze 2>/dev/null | grep -i -E '^(opencv-python-headless|opencv-python|numpy|pandas|scikit-learn|scipy|timm|huggingface[-_]hub|safetensors|pyyaml|pillow)==' | sort -f > $K/pins_check.txt
  if ! diff <(printf '%s\n' $PK | sort -f) $K/pins_check.txt >> $L 2>&1; then fail "package pins differ from the box (see setup.log)"; fi
  say "packages identical to the box: $(wc -l < $K/pins_check.txt) pins | torch $($PYI -c 'import torch; print(torch.__version__)')"
fi

# ---- 4 GPU facts ----
if [ "$MOCK" = 1 ]; then say "mock: GPU checks skipped"; else
  NGX=$(nvidia-smi -L | wc -l); [ "$NGX" -ge "$NG" ] || fail "nvidia-smi lists $NGX GPUs, plan needs $NG"
  $PYI - "$NG" >> $L 2>&1 <<'PYG' || fail "GPU matmul check"
import sys, torch
n = int(sys.argv[1]); assert torch.cuda.is_available() and torch.cuda.device_count() >= n, 'cuda devices'
for g in range(n):
    x = torch.randn(2048, 2048, device=f'cuda:{g}', dtype=torch.float16); v = float((x @ x).float().abs().mean())
    assert v == v and v > 0, f'gpu {g} matmul {v}'
    print('gpu', g, torch.cuda.get_device_name(g), 'fp16 matmul ok', round(v, 2), 'mem GB', round(torch.cuda.get_device_properties(g).total_memory / 2**30, 1))
PYG
  SHM=$(df -BG --output=avail /dev/shm | tail -n 1 | tr -dc 0-9); DF=$(df -BG --output=avail $R | tail -n 1 | tr -dc 0-9)
  [ "${SHM:-0}" -ge 8 ] || fail "/dev/shm ${SHM:-?} GB < 8 GB (5 trainers x 10 loader workers)"
  [ "${DF:-0}" -ge 25 ] || fail "only ${DF:-?} GB free disk (need ~25 GB: 5 x 3.6 GB epoch ckpts + logs)"
  say "GPU OK: $(grep -c 'fp16 matmul ok' $L) GPU(s) | shm ${SHM} GB | disk free ${DF} GB | $(nproc) vCPU | RAM $(free -g | awk '/Mem/{print $2}') GB"
fi

# ---- 5 parity probe ----
( cd $R/fast_build/code2 && TTR=$TTR CUDA_VISIBLE_DEVICES= HF_HOME=$R/hf HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  ${PARITY_PRE:-} $PYI $K/parity.py --code $R/fast_build/code2 --store $R/fleet_store --labels $R/r7t_labels/real/$TABLE \
  --folds_csv $R/rsna_train/folds5.csv --exclude $R/rsna_train/gold74_eval_set.csv --folds $FOLDS ) > $S/parity_inst.json 2> $R/logs/parity.err \
  || { tail -n 5 $R/logs/parity.err >> $L; fail "parity.py"; }
say "parity probe written ($(wc -c < $S/parity_inst.json) bytes)"
say "SETUP OK"; date +%s > $S/setup.ok
