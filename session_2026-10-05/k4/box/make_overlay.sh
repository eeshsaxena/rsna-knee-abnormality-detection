#!/usr/bin/env bash
# make_overlay.sh (box): a KMIMIC_HOME overlay under ~/k2_test/mimic so ~/kmimic/run_parent.py can run a BUILT K2
# notebook (parent + appended K2 cells) with the two extra datasets attached, WITHOUT writing anything under ~/kmimic.
# Everything here is a symlink into the read-only mirror, plus links to the K2 datasets staged in ~/k2_test/ds.
#   bash ~/k2_test/k2/box/make_overlay.sh
set -euo pipefail
O=~/k2_test/mimic
M=~/kmimic
mkdir -p "$O/kaggle/input"
for e in "$M"/kaggle/input/*; do                       # every mirrored source + the competition dir
  n=$(basename "$e"); [ -e "$O/kaggle/input/$n" ] || ln -s "$e" "$O/kaggle/input/$n"
done
for d in rsna-knee-cnxt-fleet rsna-knee-cropaug-members; do   # Kaggle mounts them at /kaggle/input/<slug> too
  [ -d ~/k2_test/ds/$d ] && { [ -e "$O/kaggle/input/$d" ] || ln -s ~/k2_test/ds/$d "$O/kaggle/input/$d"; }
done
for d in runner comp_real_csv parent ids; do [ -e "$O/$d" ] || ln -s "$M/$d" "$O/$d"; done
ls -la "$O" "$O/kaggle/input" | sed -n 1,40p
echo "overlay ready: KMIMIC_HOME=$O ~/miniconda3/envs/kmimic/bin/python ~/kmimic/run_parent.py --nb <built.ipynb> ..."
