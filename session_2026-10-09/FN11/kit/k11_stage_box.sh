#!/bin/bash
# k11_stage_box.sh (laptop, Git Bash; 2026-10-09, Claude; from k10_stage_box.sh + kfn70_stage_box.sh): stage FN11 and
# FN11-N70 for the box CPU A0 under ~/k11_test/<A> ONLY.
#   bash "$SP/k11/k11_stage_box.sh" <W> <FN> <FN70> [A]
#     W     laptop build_k11.py dir (k11/build_w11 or a TEST build_w11_i19)    FN / FN70  build_k11fn.py dirs
#     A     box area under ~/k11_test (default a0)
# - dataset ~/k11_test/<A>/ds/rsna-knee-cnxt-fleet-k11w = byte copy of the laptop upload folder: small files scp'd, the
#   members hard-linked box-side (cp -l) from ~/k11_test/ckpt (E2) or ~/k10_test/ckpt (C / D / E) under their SHIPPED
#   names; its sha256 list must equal the laptop's ("DS IDENTICAL")
# - stale-version negative: ~/k11_test/<A>/ds/stale_k10w = cp -al of ~/k10_test/a0/ds/rsna-knee-cnxt-fleet-k10w (sha256
#   list == the laptop k10w upload folder)
# - N weights ~/k11_test/<A>/wts/<file> hard link of ~/kfn_test/wts/<file> (sha256 pinned); ids n70_3.csv (the 3 gold
#   studies of the N70 A0, where .30/.70 and .40/.60 differ) + neg2.csv from ~/kfn70_test/ids / ~/kfn_test/ids
# - tools -> ~/k11_test/<A>/k11/ (fn_check, k2_check, k9n_check, fnw_check [K11 slugs], fn11_check, n949/log949.txt,
#   box/k10fnw_box_stage.py [byte copy, K10_DS_SLUG env], box/k11_box_launch.sh); records FN_BUILD_fn11.json,
#   FN_BUILD_fn11n70.json, K5_BUILD_w11.json; notebooks -> kernel/
# Never writes outside ~/k11_test; never touches the GPU, ~/queue, ~/fleet_runs, ~/oai_* or a running process.
set -uo pipefail
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
PY=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe
K=$SP/k11
W=${1:?W}; FN=${2:?FN}; F70=${3:?FN70}; A=k11_test/${4:-a0}
B=training@192.168.10.30
SSH="ssh -o BatchMode=yes $B"; SCP="scp -q -o BatchMode=yes"
WSHA=7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef
WF=raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt
S=rsna-knee-cnxt-fleet-k11w
UW10=$SP/k10/build_w/upload/rsna-knee-cnxt-fleet-k10w
die(){ echo "K11STAGE FAIL: $*"; exit 1; }
for f in "$W/K5_BUILD.json" "$FN/FN_BUILD.json" "$F70/FN_BUILD.json" "$W/upload/$S/fleet_manifest.json"; do [ -f "$f" ] || die "missing $f"; done
$PY -B "$K/make_k11_tools.py" --check | tail -n 1 | grep -q "MAKE_K11_TOOLS OK" || die "make_k11_tools --check"
$SSH "mkdir -p ~/$A/k11/box ~/$A/k11/n949 ~/$A/kernel ~/$A/ids ~/$A/runs ~/$A/ds ~/$A/wts ~/$A/tmp && { [ -f ~/$A/wts/$WF ] || ln ~/kfn_test/wts/$WF ~/$A/wts/$WF || cp -p ~/kfn_test/wts/$WF ~/$A/wts/$WF; } && echo '$WSHA  /home/training/$A/wts/$WF' | sha256sum -c --quiet && cp -p ~/kfn70_test/ids/n70_3.csv ~/kfn_test/ids/neg2.csv ~/kfn_test/ids/gold_labels.csv ~/$A/ids/" < /dev/null || die "box dirs / weights / ids"
$SCP "$K/fn_check.py" "$K/k2_check.py" "$K/k9n_check.py" "$K/fnw_check.py" "$K/fn11_check.py" $B:$A/k11/ || die "scp checks"
$SCP "$K/n949/log949.txt" $B:$A/k11/n949/ || die "scp log949"
$SCP "$K/box/k10fnw_box_stage.py" "$K/box/k11_box_launch.sh" $B:$A/k11/box/ || die "scp box tools"
# ---- the k11w dataset (members hard-linked box-side, never re-uploaded)
MAP=$($PY -c "import json,sys; b=json.load(open(sys.argv[1])); print(' '.join(p['staged']+'='+p['file'] for p in b['picks']))" "$W/K5_BUILD.json" | tr -d '\r') || die "picks"
$SSH "rm -rf ~/$A/ds/$S && mkdir -p ~/$A/ds/$S" < /dev/null || die ssh
(cd "$W/upload/$S" && $SCP fleet_*.py fleet_manifest.json dataset-metadata.json parity_*.npz $B:$A/ds/$S/) || die "scp small files"
CPL=""
for kv in $MAP; do st=${kv%%=*}; src=${kv#*=}
  CPL="$CPL { [ -f ~/k11_test/ckpt/$src ] && cp -l ~/k11_test/ckpt/$src ~/$A/ds/$S/$st || cp -l ~/k10_test/ckpt/$src ~/$A/ds/$S/$st; } &&"; done
$SSH "$CPL true" < /dev/null || die "link members (collect first)"
$SSH "cd ~/$A/ds/$S && sha256sum * | LC_ALL=C sort -k2" < /dev/null > "$W/_box_ds_sha.txt" || die "box sha"
(cd "$W/upload/$S" && sha256sum * | sed 's/ \*/  /' | LC_ALL=C sort -k2) > "$W/_lap_ds_sha.txt"
diff "$W/_box_ds_sha.txt" "$W/_lap_ds_sha.txt" > /dev/null || { diff "$W/_box_ds_sha.txt" "$W/_lap_ds_sha.txt"; die "box $S != laptop upload folder"; }
echo "DS IDENTICAL $S ($(wc -l < "$W/_lap_ds_sha.txt") files) == $W/upload/$S"
# ---- stale-version negative dataset (the FN10-3 k10w bytes)
$SSH "rm -rf ~/$A/ds/stale_k10w && cp -al ~/k10_test/a0/ds/rsna-knee-cnxt-fleet-k10w ~/$A/ds/stale_k10w && cd ~/$A/ds/stale_k10w && sha256sum * | LC_ALL=C sort -k2" < /dev/null > "$W/_box_stale_sha.txt" || die "stale ds"
(cd "$UW10" && sha256sum * | sed 's/ \*/  /' | LC_ALL=C sort -k2) > "$W/_lap_k10w_sha.txt"
diff "$W/_box_stale_sha.txt" "$W/_lap_k10w_sha.txt" > /dev/null || die "box stale_k10w != the laptop k10w upload"
echo "DS IDENTICAL stale_k10w ($(wc -l < "$W/_lap_k10w_sha.txt") files) == the FN10-3 k10w upload"
# ---- records + notebooks
$SCP "$FN/FN_BUILD.json" $B:$A/FN_BUILD_fn11.json && $SCP "$F70/FN_BUILD.json" $B:$A/FN_BUILD_fn11n70.json && \
  $SCP "$W/K5_BUILD.json" $B:$A/K5_BUILD_w11.json || die "scp records"
$SCP "$FN"/kernel/*/*.ipynb "$F70"/kernel/*/*.ipynb $B:$A/kernel/ || die "scp notebooks"
$SSH "cd ~/$A && sha256sum kernel/*.ipynb && wc -l ids/*.csv" < /dev/null
(sha256sum "$FN"/kernel/*/*.ipynb "$F70"/kernel/*/*.ipynb)
echo "K11STAGE OK -> ssh -o BatchMode=yes $B 'A=$A bash ~/$A/k11/box/k11_box_launch.sh q1 q2' < /dev/null"
