#!/bin/bash
# vast_setup.sh: prepare the rented instance (after vast_launch.sh). Laptop side; every step prints what it does,
# is skipped when its marker exists, and can be re-run after any failure.
#   S1 instance facts + gates (GPUs, torch CUDA, /dev/shm, disk, CPU)
#   S2 packages pinned to the BOX versions (pip freeze of the rsna env; torch/torchvision of the image kept), rsync
#   S3 kit files -> /workspace/rsna/kit (lane.sh, status.sh, parity.py, argv_template.txt, plan.env)
#   S4 BOX -> instance copy, run detached ON the box (box_push.sh): code, 4 csvs, HF ConvNeXt cache, 4407-file store,
#      then md5 parity of every store file + code + csvs + the HF blob (sha256)
#   S5 parity.py on box and instance (CPU): contract/jpeg/model/decode-probe/backbone hashes + fold counts identical
#   bash vast_setup.sh [--dry-run]      (--dry-run: box-side checks + plan only, touches no instance)
SCRIPT_TAG=setup; source "$(dirname "$0")/config.sh"
DRY=0; [ "${1:-}" = --dry-run ] && DRY=1
IID=$(st_get INSTANCE_ID); FOLDS=$(st_get FOLDS); FOLDS=${FOLDS:-0}
box_up || die "box $BOX unreachable (lab network?)"
sync_box_ops || die "could not copy the kit helpers to the box"
BOXV=$(bops versions); [ -n "$BOXV" ] || die "could not read the box package versions"
PKGS=$(printf '%s\n' "$BOXV" | grep -v -i -E '^(torch|torchvision)==' | tr '\n' ' ')
log "== vast_setup | instance ${IID:-none} | folds [$FOLDS] | box pins: $PKGS"

if [ $DRY = 1 ] || [ -z "$IID" ] || [ "$(st_get DESTROYED)" = 1 ]; then
  log "DRY: box-side checks only (no live instance)"
  bops paths $FOLDS | sed 's/^/  box /'
  log "DRY: box parity.py run (CPU only, nice 19) for comparison with the instance later:"
  PB=$(bops parity $FOLDS) || die "parity.py failed on the box"
  printf '%s' "$PB" | jq_py '
print("  box parity: contract pixel_sha %s tensor_sha %s | select_sha %s | jpeg_fp %s | model_sha %s" % (d["contract"]["pixel_sha"], d["contract"]["tensor_sha"], d["select_sha"], d["jpeg_fp"], d["model_sha"]))
print("  box parity: n %s | fold counts (train, val) %s | decode probe %s | backbone_sha %s | state keys %s" % (d["n"], d["fold_counts"], d["decode_probe"], d.get("backbone_sha"), d.get("state_keys")))
print("  box versions: %s" % d["versions"])'
  log "DRY: S1-S5 would run against the instance after vast_launch.sh; nothing touched"; exit 0
fi
[ "$(st_get DESTROYED)" != 1 ] || die "instance $IID is destroyed"
ix true || die "laptop cannot ssh to instance $IID"
done_(){ ix "test -e $R/status/setup_$1.ok"; }
mark(){ ix "mkdir -p $R/status && date '+%F %T' > $R/status/setup_$1.ok"; }

# ---- S1 facts + gates ----
NGPU_NEED=$(echo $FOLDS | wc -w); [ "$NGPU_NEED" -gt "$(st_get NGPU)" ] && NGPU_NEED=$(st_get NGPU)
log "S1 instance facts"
FACTS=$(ix "nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader; echo NPROC \$(nproc); free -g | awk '/Mem/{print \"RAM_GB \" \$2}'; df -BG --output=avail /dev/shm | tail -n 1 | sed 's/^ */SHM /'; df -BG --output=avail /workspace | tail -n 1 | sed 's/^ */DISK /'; PYX=\$( [ -x /opt/conda/bin/python ] && echo /opt/conda/bin/python || command -v python || command -v python3 ); \$PYX -c 'import sys, torch, torchvision; print(\"PY\", sys.executable, sys.version.split()[0], \"TORCH\", torch.__version__, torchvision.__version__, \"CUDA\", torch.cuda.is_available(), torch.cuda.device_count(), torch.version.cuda)'") || die "S1 ssh failed"
printf '%s\n' "$FACTS" | sed 's/^/  inst /'
NG=$(printf '%s\n' "$FACTS" | grep -c -E '^[0-9]+, '); SHM=$(printf '%s\n' "$FACTS" | sed -n 's/^SHM \([0-9]*\)G/\1/p')
DISK=$(printf '%s\n' "$FACTS" | sed -n 's/^DISK \([0-9]*\)G/\1/p'); PYI=$(printf '%s\n' "$FACTS" | sed -n 's/^PY \([^ ]*\) .*/\1/p')
[ "$NG" -ge "$NGPU_NEED" ] || die "instance shows $NG GPUs, plan needs $NGPU_NEED"
printf '%s\n' "$FACTS" | grep -q 'CUDA True' || die "torch cannot see CUDA on the instance"
[ "${SHM:-0}" -ge $((2 * NGPU_NEED)) ] || die "/dev/shm ${SHM:-?} GB < 2 GB per trainer (DataLoader workers would crash): destroy and pick another offer"
[ "${DISK:-0}" -ge 40 ] || die "only ${DISK:-?} GB free on /workspace"
[ -n "$PYI" ] || die "no python found on the instance"
st_set PYI "$PYI"; mark S1

# ---- S2 packages ----
if done_ S2; then log "S2 packages: already done"; else
  # opencv-python-headless 4.14.0.94 declares numpy>=2 but the box runs it on numpy 1.26.4: install it like the box
  # (--no-deps, pinned). S5 then proves the cv2 build decodes identically (jpeg_fp + 4-study decode probe).
  PKGS_NODEPS=$(printf '%s\n' $PKGS | grep -i '^opencv' | tr '\n' ' '); PKGS_DEPS=$(printf '%s\n' $PKGS | grep -v -i '^opencv' | tr '\n' ' ')
  log "S2 apt rsync + pip install pinned to the box (torch/torchvision of the image kept): $PKGS"
  ix "command -v rsync > /dev/null || (apt-get -qq update > /dev/null && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y rsync > /dev/null); command -v rsync" || die "S2 could not install rsync"
  ix "cd /tmp && $PYI -c 'import torch, torchvision; print(\"torch==\" + torch.__version__.split(\"+\")[0]); print(\"torchvision==\" + torchvision.__version__.split(\"+\")[0])' > /tmp/constraints.txt && $PYI -m pip install -q --no-cache-dir -c /tmp/constraints.txt $PKGS_DEPS 2>&1 | tail -n 3 && $PYI -m pip install -q --no-cache-dir --no-deps $PKGS_NODEPS 2>&1 | tail -n 3" || die "S2 pip install failed"
  CHK=$(ix "$PYI -m pip list --format=freeze 2>/dev/null" | grep -i -E '^(opencv-python-headless|opencv-python|numpy|pandas|scikit-learn|scipy|timm|huggingface[-_]hub|safetensors|pyyaml|pillow)==' | sort -f)
  if [ "$CHK" != "$(printf '%s\n' "$BOXV" | grep -v -i -E '^(torch|torchvision)==' | sort -f)" ]; then
    log "S2 version mismatch box vs instance:"; diff <(printf '%s\n' "$BOXV" | grep -v -i -E '^(torch|torchvision)==' | sort -f) <(printf '%s\n' "$CHK") | sed 's/^/  /'
    die "S2 package versions differ"
  fi
  mark S2; log "S2 packages identical to the box ($(printf '%s\n' "$CHK" | wc -l) pins)"
fi

# ---- S3 kit files + plan ----
log "S3 kit files -> $R/kit"
ix "mkdir -p $R/kit $R/status $R/logs $R/fleet_runs" || die "S3 mkdir"
for f in lane.sh status.sh parity.py argv_template.txt kit.env; do ix "cat > $R/kit/$f" < "$KIT/remote/$f" || die "S3 copy $f"; done
LANE_DL=$(( $(st_get DEADLINE_EPOCH) - 1200 ))     # 20 min before the watcher cap: room to fetch
printf 'FOLDS="%s"\nDEADLINE_EPOCH=%s\nMODEL_FOLD_S=%s\nPYI=%s\nEPOCHPT=%s\n' "$FOLDS" "$LANE_DL" "$(st_get MODEL_FOLD_S)" "$PYI" "${FETCH_EPOCH_PT:-1}" \
  | ix "cat > $R/kit/plan.env" || die "S3 plan.env"
ix "bash -n $R/kit/lane.sh && bash -n $R/kit/status.sh && $PYI -m py_compile $R/kit/parity.py && echo kit-parse-ok" | grep -q kit-parse-ok || die "S3 kit does not parse on the instance"
log "S3 plan: folds [$FOLDS], lanes may start a fold only if it ends before $(date -d @$LANE_DL '+%m-%d %H:%M')"

# ---- S4 box -> instance copy + md5 parity (detached on the box) ----
BH=$(st_get B_HOST); BP=$(st_get B_PORT); [ -n "$BH" ] || die "no box route in state.env (vast_launch.sh step 7)"
if done_ S4; then log "S4 data: already done"; else
  bops push-start "$BH" "$BP" "$R" | sed 's/^/  box /'
  t0=$(date +%s); last=""
  while :; do
    PS=$(bops push-status 2>/dev/null); L1=$(printf '%s\n' "$PS" | head -n 1)
    n=$(ix "ls $R/fleet_store 2>/dev/null | wc -l" 2>/dev/null)
    [ "$L1 $n" != "$last" ] && log "  $L1 | instance store files: ${n:-?}/4407"; last="$L1 $n"
    case $L1 in *"status: DONE"*) break;; *"status: FAILED"*) printf '%s\n' "$PS" | sed 's/^/  /'; die "S4 box push failed";; esac
    case $L1 in *"pid alive: no"*) case $L1 in *RUNNING*) die "S4 box push died (see ~/vastL5/push.log on the box)";; esac;; esac
    [ $(( $(date +%s) - t0 )) -lt 7200 ] || die "S4 still not done after 120 min"
    sleep 30
  done
  printf '%s\n' "$PS" | sed 's/^/  box /'; mark S4
  log "S4 store + code + csvs + HF weights on the instance, md5/sha256 parity OK"
fi

# ---- S5 parity.py box vs instance ----
if done_ S5; then log "S5 parity: already done"; else
  log "S5 parity.py on the box (CPU, nice 19) and on the instance (CPU)"
  PB=$(bops parity $FOLDS) || die "S5 parity.py failed on the box"
  PI=$(ix "cd $R/fast_build/code2 && TTR=$(sed -n "s/^TTR=//p" "$KIT/remote/kit.env") CUDA_VISIBLE_DEVICES= HF_HOME=$R/hf HF_HUB_OFFLINE=1 $PYI $R/kit/parity.py --code $R/fast_build/code2 --store $R/fleet_store --labels $R/novel/frac75/labels_frac75_logit.csv --folds_csv $R/rsna_train/folds5.csv --exclude $R/rsna_train/gold74_eval_set.csv --folds $FOLDS 2> $R/logs/parity.err || { tail -n 5 $R/logs/parity.err; exit 1; }") || die "S5 parity.py failed on the instance: $PI"
  "$PYL" - "$PB" "$PI" <<'EOF' || die "S5 PARITY MISMATCH: do not train (destroy, or fix and re-run)"
import json, sys
b, i = json.loads(sys.argv[1]), json.loads(sys.argv[2]); bad = 0
for k in ('contract', 'select_sha', 'jpeg_fp', 'model_sha', 'n', 'fold_counts', 'decode_probe', 'backbone_sha', 'state_keys', 'pretrained_cfg'):
    same = b.get(k) == i.get(k); bad += not same
    v = b.get(k)
    if k == 'contract': v = {x: v[x] for x in ('pixel_sha', 'tensor_sha')}
    print('  %-14s %s  %s' % (k, 'IDENTICAL' if same else 'DIFFERS', json.dumps(v)[:110] if same else 'box %s | inst %s' % (json.dumps(b.get(k))[:80], json.dumps(i.get(k))[:80])))
print('  versions box  %s' % b['versions']); print('  versions inst %s' % i['versions'])
sys.exit(1 if bad else 0)
EOF
  mark S5; log "S5 parity: every hash and count identical"
fi
ix "date '+%F %T' > $R/status/SETUP_OK"; st_set SETUP_OK 1
log "SETUP DONE. Next: bash vast_train.sh   (folds [$FOLDS])"
