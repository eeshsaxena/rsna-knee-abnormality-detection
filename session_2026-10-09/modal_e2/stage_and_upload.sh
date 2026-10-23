#!/bin/bash
# stage_and_upload.sh: stage the small inputs, refuse anything OAI/union/symlinked, then upload to the Modal volume
# rsna-e2. Uploads ONLY: code2 (no __pycache__), the 4 csvs, the ConvNeXt-T HF snapshot, the 4407-npz competition
# store, and meta (argv template + md5 manifests + box parity json). Log: ~/modal_e2/logs/upload.log
set -eu
M=~/modalenv/bin/modal; E=~/modal_e2; S=$E/stage; VOL=rsna-e2
say(){ echo "[upload $(date '+%m-%d %H:%M:%S')] $*" | tee -a $E/logs/upload.log; }
HFD=~/.cache/huggingface/hub/models--timm--convnext_tiny.fb_in22k_ft_in1k; REV=bc48a87f119bae1b06418f5340bfa97ef75d4609
SRCS=(~/fast_build/code2 ~/novel/frac75/labels_frac75_logit.csv ~/rsna_train/folds5.csv ~/rsna_train/gold74_eval_set.csv ~/rsna_train/llm_labels_v4_blend.csv $HFD ~/fleet_store)
for p in "${SRCS[@]}"; do
  rp=$(readlink -f "$p"); case $rp in *oai*|*OAI*|*union_store*) say "REFUSE: $p resolves to $rp"; exit 1;; esac
  case $rp in /home/training/fast_build/code2|/home/training/novel/frac75/*|/home/training/rsna_train/*|$HFD|/home/training/fleet_store) ;; *) say "REFUSE: $p resolves outside the allowed inputs ($rp)"; exit 1;; esac
done
[ -z "$(find ~/fast_build/code2 ~/fleet_store -maxdepth 1 -type l)" ] || { say "REFUSE: symlinks in code2 or fleet_store"; exit 1; }
NF=$(find ~/fleet_store -mindepth 1 -maxdepth 1 | wc -l); NN=$(find ~/fleet_store -mindepth 1 -maxdepth 1 -type f -name '*.npz' | wc -l)
[ "$NF" = 4407 ] && [ "$NN" = 4407 ] || { say "REFUSE: store has $NF entries, $NN npz (want 4407 npz only)"; exit 1; }
say "source check OK: no OAI/union path, no symlink, store = 4407 npz only"

# store md5 manifest: reuse ~/vast320/store_md5.txt only if its signature still matches the store
sig="4407 $(du -sb ~/fleet_store | cut -f1) $(find ~/fleet_store -maxdepth 1 -name '*.npz' -printf '%T@\n' | sort -n | tail -n 1)"
if [ "$(cat ~/vast320/store_md5.sig 2>/dev/null)" = "$sig" ] && [ "$(wc -l < ~/vast320/store_md5.txt)" = 4407 ]; then
  cp ~/vast320/store_md5.txt $E/meta/store_md5.txt; say "store md5 manifest reused (signature matches: $sig)"
else
  say "computing a fresh store md5 manifest (nice 19)"
  (cd ~/fleet_store && find . -maxdepth 1 -name '*.npz' -printf '%f\n' | sort | xargs -n 64 -P 4 nice -n 19 md5sum | LC_ALL=C sort -k 2) > $E/meta/store_md5.txt
fi
echo "$sig" > $E/meta/store_md5.sig

rm -rf $S; mkdir -p $S/code2 $S/novel/frac75 $S/rsna_train $S/hf/hub/models--timm--convnext_tiny.fb_in22k_ft_in1k/refs $S/hf/hub/models--timm--convnext_tiny.fb_in22k_ft_in1k/snapshots/$REV
cp -p $(find ~/fast_build/code2 -maxdepth 1 -type f) $S/code2/
cp -p ~/novel/frac75/labels_frac75_logit.csv $S/novel/frac75/
cp -p ~/rsna_train/folds5.csv ~/rsna_train/gold74_eval_set.csv ~/rsna_train/llm_labels_v4_blend.csv $S/rsna_train/
cp -p $HFD/refs/main $S/hf/hub/models--timm--convnext_tiny.fb_in22k_ft_in1k/refs/
cp -pL $HFD/snapshots/$REV/model.safetensors $S/hf/hub/models--timm--convnext_tiny.fb_in22k_ft_in1k/snapshots/$REV/
(cd $S && find . -type f ! -path './meta/*' -printf '%P\n' | LC_ALL=C sort | xargs md5sum) > $E/meta/small_md5.txt
(cd ~ && md5sum fast_build/code2/fleet_*.py novel/frac75/labels_frac75_logit.csv rsna_train/folds5.csv rsna_train/gold74_eval_set.csv rsna_train/llm_labels_v4_blend.csv) > $E/meta/small_md5_box_paths.txt
hb=$(sha256sum $S/hf/hub/models--timm--convnext_tiny.fb_in22k_ft_in1k/snapshots/$REV/model.safetensors | cut -c1-64)
[ "$hb" = 77b619e187de71bce58721689da0c009e84cc20677293d7d43105307e34d868a ] || { say "FAIL: staged HF weights sha256 $hb"; exit 1; }
say "staged: $(find $S -type f | wc -l) small files, $(du -sb $S | cut -f1) bytes; HF sha256 77b619e1.. OK"
mkdir -p $S/meta; cp -p $E/meta/* $S/meta/

$M volume list 2>&1 | grep -q " $VOL " || $M volume create $VOL 2>&1 | tail -n 2 | tee -a $E/logs/upload.log
t0=$(date +%s)
for d in code2 novel rsna_train hf meta; do
  $M volume put -f $VOL $S/$d /$d 2>&1 | tail -n 2 | tee -a $E/logs/upload.log
done
say "small files uploaded in $(( $(date +%s) - t0 ))s; uploading the store (11.66 GB)"
t0=$(date +%s)
$M volume put -f $VOL /home/training/fleet_store /fleet_store 2>&1 | tail -n 3 | tee -a $E/logs/upload.log
dt=$(( $(date +%s) - t0 )); [ $dt -lt 1 ] && dt=1
say "store uploaded in ${dt}s ($(( 11656152104 / dt / 1000000 )) MB/s)"
echo DONE > $E/logs/upload.status
