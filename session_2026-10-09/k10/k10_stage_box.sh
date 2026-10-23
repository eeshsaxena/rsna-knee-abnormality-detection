#!/bin/bash
# k10_stage_box.sh (laptop, Git Bash; 2026-10-09, Claude; from k9_stage_box.sh + k9n/kfn_stage_box.sh): stage EVERY K10
# kernel for the box CPU A0, under ~/$A only (A = k10_test/a0 for the real build, k10_test/a0sim for the simulation).
#   bash "$SP/k10/k10_stage_box.sh" <ROOT> <A> <K10T>
#     ROOT  laptop dir holding build/ (K10 main + solo), build_w/ (k10w upload), build_n/ (K10N-a), build_fn/ (FN10),
#           build_fn3/ (FN10-3): $SP/k10 (real) or $SP/k10/_sim (simulation)
#     A     box A0 area (k10_test/a0 | k10_test/a0sim)      K10T  box collect area with ckpt/ + members.json (k10_test |
#           k10_test/simtest)
# - datasets ~/$A/ds/rsna-knee-cnxt-fleet-k10 (10) and ~/$A/ds/rsna-knee-cnxt-fleet-k10w (15) = byte copies of the laptop
#   upload folders: small files scp'd, members cp -p box-side from ~/$K10T/ckpt under their SHIPPED names (K5_BUILD picks);
#   each folder's sha256 list must equal the laptop's ("DS IDENTICAL")
# - kernels -> ~/$A/kernel/, build records -> ~/$A/{K5_BUILD.json, K5_BUILD_w.json, K10N_BUILD.json, FN_BUILD_fn10.json,
#   FN_BUILD_fn10_3.json}, ~/$K10T/members.json -> ~/$A/members.json (OOF map for the per-member parity)
# - tools -> ~/$A/k10/ ; N weights ~/$A/wts/<file> (copy of ~/kfn_test/wts, sha256 pinned); ids copied from ~/kfn_test/ids
# Never writes outside ~/k10_test; never touches the GPU, ~/queue, ~/fleet_runs or a running process.
set -uo pipefail
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
PY=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe
K=$SP/k10
ROOT=${1:?ROOT}; A=${2:?A}; K10T=${3:?K10T}
B=training@192.168.10.30
SSH="ssh -o BatchMode=yes $B"; SCP="scp -q -o BatchMode=yes"
WSHA=7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef
WF=raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt
die(){ echo "K10STAGE FAIL: $*"; exit 1; }
case "$A" in k10_test/*) ;; *) die "A=$A must be under k10_test/";; esac
case "$K10T" in k10_test|k10_test/*) ;; *) die "K10T=$K10T must be k10_test or under it";; esac
for f in build/K5_BUILD.json build_w/K5_BUILD.json build_n/K10N_BUILD.json build_fn/FN_BUILD.json build_fn3/FN_BUILD.json; do
  [ -f "$ROOT/$f" ] || die "missing $ROOT/$f (build everything first)"; done
$SSH "mkdir -p ~/$A/k10/box ~/$A/k10/n949 ~/$A/kernel ~/$A/ids ~/$A/runs ~/$A/ds ~/$A/wts && { [ -f ~/$A/wts/$WF ] || cp -p ~/kfn_test/wts/$WF ~/$A/wts/$WF; } && echo '$WSHA  /home/training/$A/wts/$WF' | sha256sum -c --quiet && cp -p ~/kfn_test/ids/a0.csv ~/kfn_test/ids/n4.csv ~/kfn_test/ids/gold_a.csv ~/kfn_test/ids/gold_labels.csv ~/$A/ids/ && head -n 3 ~/$A/ids/a0.csv > ~/$A/ids/neg2.csv && cp -p ~/$K10T/members.json ~/$A/members.json" < /dev/null || die "box dirs / weights / ids / members.json"
$SCP "$K/k2_check.py" "$K/k9n_check.py" "$K/k10n_check.py" "$K/fn_check.py" "$K/fn10_check.py" $B:$A/k10/ || die "scp checks"
$SCP "$K/n949/log949.txt" $B:$A/k10/n949/ || die "scp log949"
$SCP "$K/box/k10_box_stage.py" "$K/box/k10n_box_stage.py" "$K/box/k10fn_box_stage.py" "$K/box/k5_ids.py" "$K/box/kfn_gold_sanity.py" "$K/box/k10_box_launch.sh" $B:$A/k10/box/ || die "scp box tools"
stage_ds(){   # build dir, slug
  local BD=$1 S=$2 U="$1/upload/$2" MAP kv st src
  [ -f "$U/fleet_manifest.json" ] || die "no upload at $U"
  MAP=$($PY -c "import json,sys; b=json.load(open(sys.argv[1])); print(' '.join(p['staged']+'='+p['file'] for p in b['picks']))" "$BD/K5_BUILD.json" | tr -d '\r') || die "picks $BD"
  $SSH "rm -rf ~/$A/ds/$S && mkdir -p ~/$A/ds/$S" < /dev/null || die ssh
  (cd "$U" && $SCP fleet_*.py fleet_manifest.json dataset-metadata.json parity_*.npz $B:$A/ds/$S/) || die "scp small files $S"
  for kv in $MAP; do st=${kv%%=*}; src=${kv#*=}
    $SSH "cp -p ~/$K10T/ckpt/$src ~/$A/ds/$S/$st" < /dev/null || die "cp $src -> $st (collect first)"; done
  $SSH "cd ~/$A/ds/$S && sha256sum * | LC_ALL=C sort -k2" < /dev/null > "$BD/_box_ds_sha.txt" || die "box sha $S"
  (cd "$U" && sha256sum * | sed 's/ \*/  /' | LC_ALL=C sort -k2) > "$BD/_lap_ds_sha.txt"
  diff "$BD/_box_ds_sha.txt" "$BD/_lap_ds_sha.txt" > /dev/null || { diff "$BD/_box_ds_sha.txt" "$BD/_lap_ds_sha.txt"; die "box $S != laptop upload folder"; }
  echo "DS IDENTICAL $S ($(wc -l < "$BD/_lap_ds_sha.txt") files)"
}
stage_ds "$ROOT/build" rsna-knee-cnxt-fleet-k10
stage_ds "$ROOT/build_w" rsna-knee-cnxt-fleet-k10w
$SCP "$ROOT"/build/kernel/*/*.ipynb "$ROOT"/build_n/kernel/*/*.ipynb "$ROOT"/build_fn/kernel/*/*.ipynb "$ROOT"/build_fn3/kernel/*/*.ipynb $B:$A/kernel/ || die "scp notebooks"
$SCP "$ROOT/build/K5_BUILD.json" $B:$A/K5_BUILD.json && $SCP "$ROOT/build_w/K5_BUILD.json" $B:$A/K5_BUILD_w.json && \
  $SCP "$ROOT/build_n/K10N_BUILD.json" $B:$A/K10N_BUILD.json && $SCP "$ROOT/build_fn/FN_BUILD.json" $B:$A/FN_BUILD_fn10.json && \
  $SCP "$ROOT/build_fn3/FN_BUILD.json" $B:$A/FN_BUILD_fn10_3.json || die "scp build records"
$SSH "cd ~/$A && sha256sum kernel/*.ipynb && wc -l ids/*.csv" < /dev/null
(cd "$ROOT" && sha256sum build*/kernel/*/*.ipynb)
echo "K10STAGE OK -> ssh -o BatchMode=yes $B 'A=$A bash ~/$A/k10/box/k10_box_launch.sh q1 q2' < /dev/null"
