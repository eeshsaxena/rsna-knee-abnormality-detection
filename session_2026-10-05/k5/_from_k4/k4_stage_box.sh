#!/bin/bash
# k4_stage_box.sh (laptop, Git Bash): stage the BUILT K4 upload folder + notebooks on the box for the A0 CPU test.
#   bash "$SP/k4/k4_stage_box.sh" [build dir, default k4/build]
# ~/k4_test/ds/rsna-knee-cnxt-fleet = byte copy of the laptop upload folder (small files scp'd, checkpoints cp -p from
# ~/k4_test/ckpt, M1 from ~/fleet_runs/f0_s42 when the build has it), then the WHOLE folder's sha256 list on the box
# must equal the laptop's ("DS IDENTICAL"), notebooks -> ~/k4_test/kernel/, A0 study list -> ~/k4_test/ids/a0.csv.
# Then: ssh $B 'bash ~/k4_test/k4/box/k4_box_launch.sh all'  (CPU, detached, ~5-15 min)
set -uo pipefail
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
BUILD=${1:-$SP/k4/build}
U="$BUILD/upload/rsna-knee-cnxt-fleet"
B=training@192.168.10.30
SSH="ssh -o BatchMode=yes $B"; SCP="scp -q -o BatchMode=yes"
die(){ echo "K4STAGE FAIL: $*"; exit 1; }
[ -f "$U/fleet_manifest.json" ] || die "no build at $U"
$SSH 'rm -rf ~/k4_test/ds/rsna-knee-cnxt-fleet ~/k4_test/kernel && mkdir -p ~/k4_test/ds/rsna-knee-cnxt-fleet ~/k4_test/kernel ~/k4_test/ids' || die ssh
(cd "$U" && $SCP fleet_*.py fleet_manifest.json dataset-metadata.json parity_*.npz $B:k4_test/ds/rsna-knee-cnxt-fleet/) || die "scp small files"
for p in "$U"/*.pt; do
  b=$(basename "$p")
  case $b in
    f0_s42_raw_cnxt320x24_cv8810_best.pt) $SSH "cp -p ~/fleet_runs/f0_s42/$b ~/k4_test/ds/rsna-knee-cnxt-fleet/" || die "cp $b" ;;
    *) $SSH "cp -p ~/k4_test/ckpt/$b ~/k4_test/ds/rsna-knee-cnxt-fleet/" || die "cp $b (run k4_collect.sh first)" ;;
  esac
done
$SSH 'cd ~/k4_test/ds/rsna-knee-cnxt-fleet && sha256sum * | LC_ALL=C sort -k2' > "$BUILD/_box_ds_sha.txt" || die "box sha"
(cd "$U" && sha256sum * | sed 's/ \*/  /' | LC_ALL=C sort -k2) > "$BUILD/_lap_ds_sha.txt"
diff "$BUILD/_box_ds_sha.txt" "$BUILD/_lap_ds_sha.txt" > /dev/null || { diff "$BUILD/_box_ds_sha.txt" "$BUILD/_lap_ds_sha.txt"; die "box dataset != laptop upload folder"; }
echo "DS IDENTICAL ($(wc -l < "$BUILD/_lap_ds_sha.txt") files)"
$SCP "$BUILD"/kernel/*/*.ipynb $B:k4_test/kernel/ || die "scp notebooks"
$SSH 'ls ~/k4_test/kernel; ~/miniconda3/envs/kmimic/bin/python ~/k4_test/k4/box/k4_ids.py' || die "k4_ids"
echo "K4STAGE OK -> ssh -o BatchMode=yes $B 'bash ~/k4_test/k4/box/k4_box_launch.sh all'"
