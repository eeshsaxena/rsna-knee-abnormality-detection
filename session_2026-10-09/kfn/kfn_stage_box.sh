#!/bin/bash
# kfn_stage_box.sh (laptop, Git Bash; kfn 2026-10-09): stage the FN / N-only kit for the box CPU A0 under ~/kfn_test ONLY.
#   bash "$SP/kfn/kfn_stage_box.sh" [kfn build dir, default kfn/build] [K9 build dir, default k9/build]
# - tools: fn_check.py, k2_check.py, k9n_check.py, box/kfn_box_stage.py, box/kfn_box_launch.sh, box/kfn_gold_sanity.py,
#   n949/log949.txt -> ~/kfn_test/kfn/ ; kernels -> ~/kfn_test/kernel/ ; FN_BUILD.json + the K9 K5_BUILD.json -> ~/kfn_test/
# - fleet dataset ~/kfn_test/ds/rsna-knee-cnxt-fleet = box-side copy of ~/k9n_test/ds/rsna-knee-cnxt-fleet (read only);
#   its sha256 list must equal the laptop K9 upload folder's (the live Kaggle version): "DS IDENTICAL"
# - weights ~/kfn_test/wts/<file> = copy of ~/k9n_test/wts/<file>, sha256 pinned
# - ids: a0.csv, n4.csv, gold_a.csv, gold_labels.csv copied from ~/k9n_test/ids (read only)
# Never writes outside ~/kfn_test; never touches the GPU, ~/queue, ~/fleet_runs or a running process.
set -uo pipefail
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
NB=${1:-$SP/kfn/build}
KB=${2:-$SP/k9/build}
U="$KB/upload/rsna-knee-cnxt-fleet"
B=training@192.168.10.30
T=kfn_test
SSH="ssh -o BatchMode=yes $B"; SCP="scp -q -o BatchMode=yes"
WSHA=7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef
WF=raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt
die(){ echo "KFNSTAGE FAIL: $*"; exit 1; }
[ -f "$NB/FN_BUILD.json" ] || die "no $NB/FN_BUILD.json"
[ -f "$KB/K5_BUILD.json" ] && [ -f "$U/fleet_manifest.json" ] || die "no K9 build at $KB"
$SSH "mkdir -p ~/$T/kfn/box ~/$T/kfn/n949 ~/$T/kernel ~/$T/ids ~/$T/runs ~/$T/ds ~/$T/wts && { [ -f ~/$T/wts/$WF ] || cp -p ~/k9n_test/wts/$WF ~/$T/wts/$WF; } && echo '$WSHA  /home/training/$T/wts/$WF' | sha256sum -c --quiet && { [ -d ~/$T/ds/rsna-knee-cnxt-fleet ] || cp -a ~/k9n_test/ds/rsna-knee-cnxt-fleet ~/$T/ds/; } && cp -p ~/k9n_test/ids/a0.csv ~/k9n_test/ids/n4.csv ~/k9n_test/ids/gold_a.csv ~/k9n_test/ids/gold_labels.csv ~/$T/ids/" < /dev/null || die "box dirs / weights / ds / ids"
$SCP "$SP/kfn/fn_check.py" "$SP/kfn/k2_check.py" "$SP/kfn/k9n_check.py" $B:$T/kfn/ || die scp
$SCP "$SP/kfn/box/kfn_box_stage.py" "$SP/kfn/box/kfn_box_launch.sh" "$SP/kfn/box/kfn_gold_sanity.py" $B:$T/kfn/box/ || die scp
$SCP "$SP/kfn/n949/log949.txt" $B:$T/kfn/n949/ || die scp
$SCP "$NB/FN_BUILD.json" $B:$T/FN_BUILD.json && $SCP "$KB/K5_BUILD.json" $B:$T/K5_BUILD.json || die "scp build json"
$SCP "$NB"/kernel/*/*.ipynb $B:$T/kernel/ || die "scp notebooks"
$SSH "cd ~/$T/ds/rsna-knee-cnxt-fleet && sha256sum * | LC_ALL=C sort -k2" < /dev/null > "$NB/_box_ds_sha.txt" || die "box sha"
(cd "$U" && sha256sum * | sed 's/ \*/  /' | LC_ALL=C sort -k2) > "$NB/_lap_ds_sha.txt"
diff "$NB/_box_ds_sha.txt" "$NB/_lap_ds_sha.txt" > /dev/null || { diff "$NB/_box_ds_sha.txt" "$NB/_lap_ds_sha.txt"; die "box dataset != laptop K9 upload folder"; }
echo "DS IDENTICAL ($(wc -l < "$NB/_lap_ds_sha.txt") files) == $U"
$SSH "cd ~/$T && sha256sum kernel/*.ipynb && wc -l ids/*.csv" < /dev/null
(cd "$NB/kernel" && sha256sum */*.ipynb)
echo "KFNSTAGE OK -> ssh -o BatchMode=yes $B 'bash ~/$T/kfn/box/kfn_box_launch.sh all' < /dev/null"
