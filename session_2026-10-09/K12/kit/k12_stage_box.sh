#!/bin/bash
# k12_stage_box.sh (laptop, Git Bash; 2026-10-09, Claude; from k11_stage_box.sh): stage a K12 kernel pair (F .40/N .60 +
# N70) for the box CPU A0 under ~/k12_test/<A> ONLY.
#   bash "$SP/k12/k12_stage_box.sh" <W12> <FN> <FN70> [A]       W12 = build_k12.py dataset dir; FN / FN70 kernel dirs
# - ds/rsna-knee-cnxt-fleet-k11w = cp -al of the A0-tested FN11 copy ~/k11_test/a0/ds/rsna-knee-cnxt-fleet-k11w (sha256 list ==
#   the laptop k11w upload: "DS IDENTICAL")
# - ds/rsna-knee-cnxt-fleet-k12w = the 2 small files scp'd + the D-OAI members hard-linked box-side from their picked src
#   (K12W_BUILD.json picks; never a file outside ~/fleet_runs or ~/k12_test); sha256 list == the laptop upload folder
# - negatives: ds/k12w_tampered (D-OAI f1 replaced by set D f1 bytes under the same name), ds/stale_k10w (the FN10-3 dataset)
# - N weights hard link, ids n70_3.csv + neg2.csv, tools (fn_check, k2_check, k9n_check, fnw_check [K12], fn12_check,
#   log949, box/k12fnw_box_stage.py, box/k12_box_launch.sh), records FN_BUILD_k12.json / _k12n70.json, K12_BUILD.json,
#   merged_manifest.json, notebooks. Never writes outside ~/k12_test; never touches the GPU, ~/queue, ~/fleet_runs, ~/oai_*.
set -uo pipefail
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
PY=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe
K=$SP/k12
W=${1:?W12}; FN=${2:?FN}; F70=${3:?FN70}; A=k12_test/${4:-a0}
B=training@192.168.10.30
SSH="ssh -o BatchMode=yes $B"; SCP="scp -q -o BatchMode=yes"
WSHA=7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef
WF=raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt
S11=rsna-knee-cnxt-fleet-k11w; S12=rsna-knee-cnxt-fleet-k12w
U11=$SP/k11/build_w11/upload/$S11
die(){ echo "K12STAGE FAIL: $*"; exit 1; }
for f in "$W/K12W_BUILD.json" "$W/merged_manifest.json" "$FN/FN_BUILD.json" "$F70/FN_BUILD.json" "$FN/K12_BUILD.json"; do [ -f "$f" ] || die "missing $f"; done
$PY -B "$K/make_k12_tools.py" --check | tail -n 1 | grep -q "MAKE_K12_TOOLS OK" || die "make_k12_tools --check"
$SSH "mkdir -p ~/$A/k12/box ~/$A/k12/n949 ~/$A/kernel ~/$A/ids ~/$A/runs ~/$A/ds ~/$A/wts ~/$A/tmp && { [ -f ~/$A/wts/$WF ] || ln ~/kfn_test/wts/$WF ~/$A/wts/$WF; } && echo '$WSHA  /home/training/$A/wts/$WF' | sha256sum -c --quiet && cp -p ~/kfn70_test/ids/n70_3.csv ~/kfn_test/ids/neg2.csv ~/$A/ids/" < /dev/null || die "box dirs / weights / ids"
$SCP "$K/fn_check.py" "$K/k2_check.py" "$K/k9n_check.py" "$K/fnw_check.py" "$K/fn12_check.py" $B:$A/k12/ || die "scp checks"
$SCP "$K/n949/log949.txt" $B:$A/k12/n949/ || die "scp log949"
$SCP "$K/box/k12fnw_box_stage.py" "$K/box/k12_box_launch.sh" $B:$A/k12/box/ || die "scp box tools"
# ---- k11w (the live FN11 dataset, A0-tested copy)
$SSH "rm -rf ~/$A/ds/$S11 && cp -al ~/k11_test/a0/ds/$S11 ~/$A/ds/$S11 && cd ~/$A/ds/$S11 && sha256sum * | LC_ALL=C sort -k2" < /dev/null > "$W/_box_k11w_sha.txt" || die "k11w"
(cd "$U11" && sha256sum * | sed 's/ \*/  /' | LC_ALL=C sort -k2) > "$W/_lap_k11w_sha.txt"
diff "$W/_box_k11w_sha.txt" "$W/_lap_k11w_sha.txt" > /dev/null || die "box k11w != the laptop FN11 upload"
echo "DS IDENTICAL $S11 ($(wc -l < "$W/_lap_k11w_sha.txt") files) == the FN11 upload"
# ---- k12w
MAP=$($PY -c "import json,sys; b=json.load(open(sys.argv[1])); print(' '.join(p['staged']+'='+p['src'] for p in b['picks']))" "$W/K12W_BUILD.json" | tr -d '\r') || die "picks"
$SSH "rm -rf ~/$A/ds/$S12 && mkdir -p ~/$A/ds/$S12" < /dev/null || die ssh
(cd "$W/upload/$S12" && $SCP k12w_manifest.json dataset-metadata.json $B:$A/ds/$S12/) || die "scp k12w small files"
CPL=""
for kv in $MAP; do st=${kv%%=*}; src=${kv#*=}
  case "$src" in /home/training/fleet_runs/*|/home/training/k12_test/*) ;; *) die "src outside ~/fleet_runs / ~/k12_test: $src";; esac
  CPL="$CPL cp -l $src ~/$A/ds/$S12/$st &&"; done
$SSH "$CPL true" < /dev/null || die "link D-OAI members"
$SSH "cd ~/$A/ds/$S12 && sha256sum * | LC_ALL=C sort -k2" < /dev/null > "$W/_box_ds_sha.txt" || die "box sha"
(cd "$W/upload/$S12" && sha256sum * | sed 's/ \*/  /' | LC_ALL=C sort -k2) > "$W/_lap_ds_sha.txt"
diff "$W/_box_ds_sha.txt" "$W/_lap_ds_sha.txt" > /dev/null || { diff "$W/_box_ds_sha.txt" "$W/_lap_ds_sha.txt"; die "box $S12 != laptop"; }
echo "DS IDENTICAL $S12 ($(wc -l < "$W/_lap_ds_sha.txt") files) == $W/upload/$S12"
# ---- negatives
F1=$(echo "$MAP" | tr ' ' '\n' | grep '^f1_' | cut -d= -f1)
D1=$($SSH "ls ~/k10_test/ckpt/f1_s1043_raw_ttr224-*_best.pt | head -1" < /dev/null | tr -d '\r')
$SSH "rm -rf ~/$A/ds/k12w_tampered ~/$A/ds/stale_k10w && cp -al ~/$A/ds/$S12 ~/$A/ds/k12w_tampered && rm ~/$A/ds/k12w_tampered/$F1 && cp -l $D1 ~/$A/ds/k12w_tampered/$F1 && cp -al ~/k10_test/a0/ds/rsna-knee-cnxt-fleet-k10w ~/$A/ds/stale_k10w" < /dev/null || die "negative datasets"
echo "negatives: k12w_tampered ($F1 <- $D1), stale_k10w"
# ---- records + notebooks
$SCP "$FN/FN_BUILD.json" $B:$A/FN_BUILD_k12.json && $SCP "$F70/FN_BUILD.json" $B:$A/FN_BUILD_k12n70.json && \
  $SCP "$FN/K12_BUILD.json" $B:$A/K12_BUILD.json && $SCP "$W/merged_manifest.json" $B:$A/merged_manifest.json || die "scp records"
$SCP "$FN"/kernel/*/*.ipynb "$F70"/kernel/*/*.ipynb $B:$A/kernel/ || die "scp notebooks"
$SSH "cd ~/$A && sha256sum kernel/*.ipynb" < /dev/null
(sha256sum "$FN"/kernel/*/*.ipynb "$F70"/kernel/*/*.ipynb)
echo "K12STAGE OK -> ssh -o BatchMode=yes $B 'A=$A bash ~/$A/k12/box/k12_box_launch.sh q1 q2' < /dev/null"
