#!/bin/bash
# k7_stage_box.sh (laptop, Git Bash): stage a BUILT K7 upload folder + notebooks on the box for the A0 CPU test.
#   bash "$SP/k7/k7_stage_box.sh" [build dir, default k7/build]          (env K7T = box test area, default k7_test)
# ~/$K7T/ds/rsna-knee-cnxt-fleet = byte copy of the laptop upload folder (small files scp'd, member checkpoints cp -p from
# ~/$K7T/ckpt under their SHIPPED name: K7-KEY re-keyed members (f0_s1044t288_...) map to the collected file through
# K5_BUILD.json picks), then the WHOLE folder's sha256 list on the box must equal the laptop's ("DS IDENTICAL"),
# notebooks -> ~/$K7T/kernel/, K5_BUILD.json -> ~/$K7T/K5_BUILD.json (its 'tta' map is the A0 --expect_tta and its picks
# the staged -> OOF map), A0 study list -> ~/$K7T/ids/a0.csv (one pick per fold, shared by that fold's members).
# Then: ssh $B 'K7T=k7_test bash ~/k7_test/k5/box/k5_box_launch.sh all'  (CPU, detached)
set -uo pipefail
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
PY=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe
K7T=${K7T:-k7_test}
BUILD=${1:-$SP/k7/build}
U="$BUILD/upload/rsna-knee-cnxt-fleet"
B=training@192.168.10.30
SSH="ssh -o BatchMode=yes $B"; SCP="scp -q -o BatchMode=yes"
die(){ echo "K7STAGE FAIL: $*"; exit 1; }
[ -f "$U/fleet_manifest.json" ] || die "no build at $U"
[ -f "$BUILD/K5_BUILD.json" ] || die "no $BUILD/K5_BUILD.json (not a build_k7.py build)"
MAP=$($PY -c "import json,sys; b=json.load(open(sys.argv[1])); print(' '.join(p['staged']+'='+p['file'] for p in b['picks']))" "$BUILD/K5_BUILD.json" | tr -d '\r') || die "K5_BUILD.json picks"
$SSH "rm -rf ~/$K7T/ds/rsna-knee-cnxt-fleet ~/$K7T/kernel ~/$K7T/K5_BUILD.json && mkdir -p ~/$K7T/ds/rsna-knee-cnxt-fleet ~/$K7T/kernel ~/$K7T/ids" < /dev/null || die ssh
(cd "$U" && $SCP fleet_*.py fleet_manifest.json dataset-metadata.json parity_*.npz $B:$K7T/ds/rsna-knee-cnxt-fleet/) || die "scp small files"
for p in "$U"/*.pt; do
  b=$(basename "$p"); src=""
  for kv in $MAP; do [ "${kv%%=*}" = "$b" ] && src="${kv#*=}"; done
  [ -n "$src" ] || die "$b is not in K5_BUILD.json picks"
  $SSH "cp -p ~/$K7T/ckpt/$src ~/$K7T/ds/rsna-knee-cnxt-fleet/$b" < /dev/null || die "cp $src -> $b (run k7_collect.sh first)"
  [ "$src" = "$b" ] || echo "  K7-KEY staged $src as $b"
done
$SSH "cd ~/$K7T/ds/rsna-knee-cnxt-fleet && sha256sum * | LC_ALL=C sort -k2" < /dev/null > "$BUILD/_box_ds_sha.txt" || die "box sha"
(cd "$U" && sha256sum * | sed 's/ \*/  /' | LC_ALL=C sort -k2) > "$BUILD/_lap_ds_sha.txt"
diff "$BUILD/_box_ds_sha.txt" "$BUILD/_lap_ds_sha.txt" > /dev/null || { diff "$BUILD/_box_ds_sha.txt" "$BUILD/_lap_ds_sha.txt"; die "box dataset != laptop upload folder"; }
echo "DS IDENTICAL ($(wc -l < "$BUILD/_lap_ds_sha.txt") files)"
$SCP "$BUILD"/kernel/*/*.ipynb $B:$K7T/kernel/ || die "scp notebooks"
$SCP "$BUILD/K5_BUILD.json" $B:$K7T/K5_BUILD.json || die "scp K5_BUILD.json"
$SSH "~/miniconda3/envs/kmimic/bin/python -c \"import json; b = json.load(open('/home/training/$K7T/K5_BUILD.json')); print('tta', b['tta'])\"; ls ~/$K7T/kernel; ~/miniconda3/envs/kmimic/bin/python ~/$K7T/k5/box/k5_ids.py --members ~/$K7T/members.json --out ~/$K7T/ids/a0.csv" < /dev/null || die "k5_ids"
echo "K7STAGE OK -> ssh -o BatchMode=yes $B 'K7T=$K7T bash ~/$K7T/k5/box/k5_box_launch.sh all' < /dev/null"
