#!/bin/bash
# stage.sh: BOX side. Stages ONLY the competition inputs of the D2 recipe for upload to the Lightning Studio:
#   ~/fast_build/code2 (*.py, no __pycache__), ~/novel/frac75/labels_frac75_logit.csv,
#   ~/rsna_train/{folds5,gold74_eval_set,llm_labels_v4_blend}.csv, the ConvNeXt-T HF snapshot (rev bc48a87f, sha256
#   77b619e1..), and ~/fleet_store (4407 competition npz, 11.66 GB) as 6 tar shards; plus meta/ (argv template, md5
#   manifests, box parity json for TTR224, parity.py). Refuses any source that resolves under ~/oai_* / union_store /
#   anything with "oai" in the path, any symlink, and any store entry that is not one of the 4407 regular *.npz files.
#   Output: ~/lightning_d2/stage/{small.tar,store_0..5.tar,meta/} + ~/lightning_d2/stage/SHA (md5 of every tar)
set -eu
E=~/lightning_d2; S=$E/stage; mkdir -p $E/logs
say(){ echo "[stage $(date '+%m-%d %H:%M:%S')] $*" | tee -a $E/logs/stage.log; }
HFD=~/.cache/huggingface/hub/models--timm--convnext_tiny.fb_in22k_ft_in1k; REV=bc48a87f119bae1b06418f5340bfa97ef75d4609
SRCS=(~/fast_build/code2 ~/novel/frac75/labels_frac75_logit.csv ~/rsna_train/folds5.csv ~/rsna_train/gold74_eval_set.csv ~/rsna_train/llm_labels_v4_blend.csv $HFD ~/fleet_store)
for p in "${SRCS[@]}"; do
  rp=$(readlink -f "$p"); case $rp in *oai*|*OAI*|*union_store*) say "REFUSE: $p resolves to $rp"; exit 1;; esac
  case $rp in /home/training/fast_build/code2|/home/training/novel/frac75/*|/home/training/rsna_train/*|$HFD|/home/training/fleet_store) ;; *) say "REFUSE: $p resolves outside the allowed inputs ($rp)"; exit 1;; esac
done
[ -z "$(find ~/fast_build/code2 ~/fleet_store -maxdepth 1 -type l)" ] || { say "REFUSE: symlinks in code2 or fleet_store"; exit 1; }
NF=$(find ~/fleet_store -mindepth 1 -maxdepth 1 | wc -l); NN=$(find ~/fleet_store -mindepth 1 -maxdepth 1 -type f -name '*.npz' | wc -l)
[ "$NF" = 4407 ] && [ "$NN" = 4407 ] || { say "REFUSE: store has $NF entries, $NN npz (want 4407 npz only)"; exit 1; }
find ~/fleet_store -maxdepth 1 -name '*.npz' -printf '%f\n' | grep -i -q oai && { say "REFUSE: an oai name in the store"; exit 1; }
say "source check OK: no OAI/union path, no symlink, store = 4407 npz only"

# store md5 manifest: reuse ~/vast320/store_md5.txt only if its signature still matches the store
sig="4407 $(du -sb ~/fleet_store | cut -f1) $(find ~/fleet_store -maxdepth 1 -name '*.npz' -printf '%T@\n' | sort -n | tail -n 1)"
rm -rf $S; mkdir -p $S/meta $S/small/code2 $S/small/novel/frac75 $S/small/rsna_train $S/small/hf/hub/models--timm--convnext_tiny.fb_in22k_ft_in1k/refs $S/small/hf/hub/models--timm--convnext_tiny.fb_in22k_ft_in1k/snapshots/$REV
if [ "$(cat ~/vast320/store_md5.sig 2>/dev/null)" = "$sig" ] && [ "$(wc -l < ~/vast320/store_md5.txt)" = 4407 ]; then
  cp ~/vast320/store_md5.txt $S/meta/store_md5.txt; say "store md5 manifest reused (signature matches: $sig)"
else
  say "computing a fresh store md5 manifest (nice 19)"
  (cd ~/fleet_store && find . -maxdepth 1 -name '*.npz' -printf '%f\n' | sort | xargs -n 64 -P 4 nice -n 19 md5sum | LC_ALL=C sort -k 2) > $S/meta/store_md5.txt
fi
echo "$sig" > $S/meta/store_md5.sig
cp -p $(find ~/fast_build/code2 -maxdepth 1 -type f -name '*.py') $S/small/code2/
cp -p ~/novel/frac75/labels_frac75_logit.csv $S/small/novel/frac75/
cp -p ~/rsna_train/folds5.csv ~/rsna_train/gold74_eval_set.csv ~/rsna_train/llm_labels_v4_blend.csv $S/small/rsna_train/
cp -p $HFD/refs/main $S/small/hf/hub/models--timm--convnext_tiny.fb_in22k_ft_in1k/refs/
cp -pL $HFD/snapshots/$REV/model.safetensors $S/small/hf/hub/models--timm--convnext_tiny.fb_in22k_ft_in1k/snapshots/$REV/
hb=$(sha256sum $S/small/hf/hub/models--timm--convnext_tiny.fb_in22k_ft_in1k/snapshots/$REV/model.safetensors | cut -c1-64)
[ "$hb" = 77b619e187de71bce58721689da0c009e84cc20677293d7d43105307e34d868a ] || { say "FAIL: staged HF weights sha256 $hb"; exit 1; }
(cd $S/small && find . -type f -printf '%P\n' | LC_ALL=C sort | xargs md5sum) > $S/meta/small_md5.txt
cp -p ~/vastS1044/argv_template.txt $S/meta/argv_template.txt
cp -p ~/vastS1044/parity.py $S/meta/parity.py
cp -p $E/meta/box_parity_ttr224.json $S/meta/box_parity_ttr224.json
grep -q -- '--seed 1044 ' $S/meta/argv_template.txt && grep -q -- '--epochs 21 ' $S/meta/argv_template.txt || { say "FAIL: argv template is not the s1044 21-epoch D2 recipe"; exit 1; }
tar -cf $S/small.tar -C $S/small .
say "small: $(find $S/small -type f | wc -l) files, $(du -sb $S/small | cut -f1) bytes; HF sha256 77b619e1.. OK"
(cd ~/fleet_store && find . -maxdepth 1 -name '*.npz' -printf '%f\n' | LC_ALL=C sort) > $S/store_list.txt
split -n l/6 -d -a 1 $S/store_list.txt $S/store_list_
for k in 0 1 2 3 4 5; do nice -n 19 tar -cf $S/store_$k.tar -C ~/fleet_store -T $S/store_list_$k; done
(cd $S && md5sum small.tar store_?.tar) > $S/meta/tar_md5.txt
cat $S/meta/tar_md5.txt | tee -a $E/logs/stage.log
say "STAGE_DONE: small.tar + 6 store shards ($(du -scb $S/store_?.tar | tail -n 1 | cut -f1) bytes)"
