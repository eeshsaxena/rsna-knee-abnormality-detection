#!/bin/bash
# k5_stage_box.sh (laptop, Git Bash): stage the BUILT K5 upload folder + notebooks on the box for the A0 CPU test.
#   bash "$SP/k6/k5_stage_box.sh" [build dir, default k5/build]
# ~/k6_test/ds/rsna-knee-cnxt-fleet = byte copy of the laptop upload folder (small files scp'd, TTR checkpoints cp -p from
# ~/k6_test/ckpt, M1 from ~/fleet_runs/f0_s42 when the build has it), then the WHOLE folder's sha256 list on the box
# must equal the laptop's ("DS IDENTICAL"), notebooks -> ~/k6_test/kernel/, K5_BUILD.json -> ~/k6_test/K5_BUILD.json
# (its 'tta' map is the A0 --expect_tta), A0 study list -> ~/k6_test/ids/a0.csv.
# Then: ssh $B 'bash ~/k6_test/k5/box/k5_box_launch.sh all'  (CPU, detached, ~5-20 min)
set -uo pipefail
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
BUILD=${1:-$SP/k6/build}
U="$BUILD/upload/rsna-knee-cnxt-fleet"
B=training@192.168.10.30
SSH="ssh -o BatchMode=yes $B"; SCP="scp -q -o BatchMode=yes"
die(){ echo "K5STAGE FAIL: $*"; exit 1; }
[ -f "$U/fleet_manifest.json" ] || die "no build at $U"
[ -f "$BUILD/K5_BUILD.json" ] || die "no $BUILD/K5_BUILD.json (not a build_k5.py build)"
$SSH 'rm -rf ~/k6_test/ds/rsna-knee-cnxt-fleet ~/k6_test/kernel ~/k6_test/K5_BUILD.json && mkdir -p ~/k6_test/ds/rsna-knee-cnxt-fleet ~/k6_test/kernel ~/k6_test/ids' < /dev/null || die ssh
(cd "$U" && $SCP fleet_*.py fleet_manifest.json dataset-metadata.json parity_*.npz $B:k6_test/ds/rsna-knee-cnxt-fleet/) || die "scp small files"
for p in "$U"/*.pt; do
  b=$(basename "$p")
  case $b in
    f0_s42_raw_cnxt320x24_cv8810_best.pt) $SSH "cp -p ~/fleet_runs/f0_s42/$b ~/k6_test/ds/rsna-knee-cnxt-fleet/" < /dev/null || die "cp $b" ;;
    *) $SSH "cp -p ~/k6_test/ckpt/$b ~/k6_test/ds/rsna-knee-cnxt-fleet/" < /dev/null || die "cp $b (run k5_collect.sh first)" ;;
  esac
done
$SSH 'cd ~/k6_test/ds/rsna-knee-cnxt-fleet && sha256sum * | LC_ALL=C sort -k2' < /dev/null > "$BUILD/_box_ds_sha.txt" || die "box sha"
(cd "$U" && sha256sum * | sed 's/ \*/  /' | LC_ALL=C sort -k2) > "$BUILD/_lap_ds_sha.txt"
diff "$BUILD/_box_ds_sha.txt" "$BUILD/_lap_ds_sha.txt" > /dev/null || { diff "$BUILD/_box_ds_sha.txt" "$BUILD/_lap_ds_sha.txt"; die "box dataset != laptop upload folder"; }
echo "DS IDENTICAL ($(wc -l < "$BUILD/_lap_ds_sha.txt") files)"
$SCP "$BUILD"/kernel/*/*.ipynb $B:k6_test/kernel/ || die "scp notebooks"
$SCP "$BUILD/K5_BUILD.json" $B:k6_test/K5_BUILD.json || die "scp K5_BUILD.json"
$SSH '~/miniconda3/envs/kmimic/bin/python -c "import json; b = json.load(open(\"/home/training/k6_test/K5_BUILD.json\")); print(\"tta\", b[\"tta\"])"; ls ~/k6_test/kernel; ~/miniconda3/envs/kmimic/bin/python ~/k6_test/k5/box/k5_ids.py' < /dev/null || die "k5_ids"
echo "K5STAGE OK -> ssh -o BatchMode=yes $B 'bash ~/k6_test/k5/box/k5_box_launch.sh all' < /dev/null"
