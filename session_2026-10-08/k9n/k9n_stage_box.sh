#!/bin/bash
# k9n_stage_box.sh (laptop, Git Bash; K9N 2026-10-08): stage the K9N kit for the box CPU A0 under ~/k9n_test ONLY.
#   bash "$SP/k9n/k9n_stage_box.sh" [K9N build dir, default k9n/build] [K9 build dir, default k9n/k9base]
# - tools: k9n_check.py, k2_check.py, box/k9n_box_stage.py, box/k5_ids.py, n949/log949.txt -> ~/k9n_test/k9n/
# - kernels -> ~/k9n_test/kernel/, K9N_BUILD.json + the K9 K5_BUILD.json -> ~/k9n_test/
# - fleet dataset ~/k9n_test/ds/rsna-knee-cnxt-fleet = byte copy of the K9 upload folder: small files scp'd, the 10
#   members copied box-side from a SNAPSHOT ~/k9n_test/ckpt (taken from ~/k9_test/ckpt, read only, sha256 == members.json)
#   under their shipped names; the whole folder's sha256 list must equal the laptop's ("DS IDENTICAL")
# - weights: ~/k9n_test/wts/<file> must already be there with the pinned sha256 (kaggle datasets download, read only)
# - ids: ~/k9n_test/ids/a0.csv (k5_ids.py, the K9 A0 pair, fold 0) and ~/k9n_test/ids/n4.csv (4 fold-0 val studies)
# Never writes ~/k9_test, ~/fleet_runs, ~/queue; never touches the GPU.
set -uo pipefail
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
PY=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe
NB=${1:-$SP/k9n/build}
KB=${2:-$SP/k9n/k9base}
U="$KB/upload/rsna-knee-cnxt-fleet"
B=training@192.168.10.30
T=k9n_test
SSH="ssh -o BatchMode=yes $B"; SCP="scp -q -o BatchMode=yes"
WSHA=7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef
WF=raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt
die(){ echo "K9NSTAGE FAIL: $*"; exit 1; }
[ -f "$NB/K9N_BUILD.json" ] || die "no $NB/K9N_BUILD.json"
[ -f "$KB/K5_BUILD.json" ] && [ -f "$U/fleet_manifest.json" ] || die "no K9 build at $KB"
$SSH "mkdir -p ~/$T/k9n/box ~/$T/k9n/n949 ~/$T/kernel ~/$T/ids ~/$T/runs ~/$T/ckpt && rm -rf ~/$T/ds/rsna-knee-cnxt-fleet && mkdir -p ~/$T/ds/rsna-knee-cnxt-fleet && echo \"$WSHA  /home/training/$T/wts/$WF\" | sha256sum -c --quiet" < /dev/null || die "box dirs / weights sha256"
$SCP "$SP/k9n/k9n_check.py" "$SP/k9n/k2_check.py" $B:$T/k9n/ || die scp
$SCP "$SP/k9n/box/k9n_box_stage.py" "$SP/k9n/box/k5_ids.py" "$SP/k9n/box/k9n_box_launch.sh" $B:$T/k9n/box/ || die scp
$SCP "$SP/k9n/n949/log949.txt" $B:$T/k9n/n949/ || die scp
$SCP "$NB/K9N_BUILD.json" $B:$T/K9N_BUILD.json && $SCP "$KB/K5_BUILD.json" $B:$T/K5_BUILD.json || die "scp build json"
$SCP "$NB"/kernel/*/*.ipynb $B:$T/kernel/ || die "scp notebooks"
MAP=$($PY -c "import json,sys; b=json.load(open(sys.argv[1])); print(' '.join(p['staged']+'='+p['file']+'='+p['sha256'] for p in b['picks']))" "$KB/K5_BUILD.json" | tr -d '\r') || die picks
for kv in $MAP; do
  st=${kv%%=*}; rest=${kv#*=}; src=${rest%%=*}; h=${rest#*=}
  $SSH "cd ~/$T/ckpt && { [ -f $src ] && echo \"$h  $src\" | sha256sum -c --quiet 2>/dev/null || { cp -p ~/k9_test/ckpt/$src . && echo \"$h  $src\" | sha256sum -c --quiet; }; } && cp -p $src ~/$T/ds/rsna-knee-cnxt-fleet/$st" < /dev/null || die "member $src"
done
(cd "$U" && $SCP fleet_*.py fleet_manifest.json dataset-metadata.json parity_*.npz $B:$T/ds/rsna-knee-cnxt-fleet/) || die "scp small files"
$SSH "cd ~/$T/ds/rsna-knee-cnxt-fleet && sha256sum * | LC_ALL=C sort -k2" < /dev/null > "$NB/_box_ds_sha.txt" || die "box sha"
(cd "$U" && sha256sum * | sed 's/ \*/  /' | LC_ALL=C sort -k2) > "$NB/_lap_ds_sha.txt"
diff "$NB/_box_ds_sha.txt" "$NB/_lap_ds_sha.txt" > /dev/null || { diff "$NB/_box_ds_sha.txt" "$NB/_lap_ds_sha.txt"; die "box dataset != laptop upload folder"; }
echo "DS IDENTICAL ($(wc -l < "$NB/_lap_ds_sha.txt") files)"
$SSH "cp -p ~/k9_test/members.json ~/$T/members.json && cd ~/$T && ~/miniconda3/envs/kmimic/bin/python k9n/box/k5_ids.py --members ~/$T/members.json --out ~/$T/ids/a0.csv && ~/miniconda3/envs/kmimic/bin/python k9n/box/k5_ids.py --members ~/$T/members.json --per_fold 4 --out ~/$T/ids/n4.csv && ls ~/$T/kernel && cat ~/$T/ids/a0.csv ~/$T/ids/n4.csv" < /dev/null || die "k5_ids"
echo "K9NSTAGE OK -> ssh -o BatchMode=yes $B 'bash ~/$T/k9n/box/k9n_box_launch.sh all' < /dev/null"
