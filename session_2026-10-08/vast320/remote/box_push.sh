#!/bin/bash
# box_push.sh HOST PORT R: BOX -> Vast instance copy of everything fleet_train.py needs, then md5 parity of every file.
# Runs ON the box, detached (box_ops.sh push-start), so a laptop network drop cannot kill it. Re-runnable: rsync only
# sends what is missing or different. Status in ~/vast320/push.status (line 1 RUNNING|DONE|FAILED <msg>, line 2
# HOST:PORT:R), log in ~/vast320/push.log. Uses the dedicated key ~/.ssh/vast_rsna_ed25519 (public half attached to
# the instance by vast_launch.sh); never copies any key.
set -u
HOST=$1 PORT=$2 R=$3
D=~/vast320; ST=$D/push.status; L=$D/push.log; K=~/.ssh/vast_rsna_ed25519
SSH="ssh -i $K -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/.ssh/known_hosts_vast -o LogLevel=ERROR -p $PORT"
say(){ echo "[box_push $(date '+%m-%d %H:%M:%S')] $*" >> "$L"; }
status(){ printf '%s\n%s\n' "$1" "$HOST:$PORT:$R" > "$ST.tmp" && mv -f "$ST.tmp" "$ST"; }
fail(){ say "FAILED: $*"; status "FAILED $*"; exit 1; }
status "RUNNING started $(date '+%F %T')"; say "start -> root@$HOST:$PORT $R"
# OAI / union data never leaves the box: every source must resolve outside ~/oai_* and ~/union_store, no symlinks
for p in ~/fast_build/code2 ~/novel/frac75/labels_frac75_logit.csv ~/rsna_train/folds5.csv ~/rsna_train/gold74_eval_set.csv          ~/rsna_train/llm_labels_v4_blend.csv ~/.cache/huggingface/hub/models--timm--convnext_tiny.fb_in22k_ft_in1k ~/fleet_store; do
  rp=$(readlink -f "$p"); case $rp in *oai*|*union_store*|*OAI*) fail "refusing: $p resolves to $rp (OAI data never leaves the box)";; esac
done
[ -z "$(find ~/fast_build/code2 ~/fleet_store -maxdepth 1 -type l)" ] || fail "symlinks in code2 or fleet_store: refusing"
[ "$(find ~/fleet_store -maxdepth 1 -type f | wc -l)" = "$(find ~/fleet_store -maxdepth 1 -type f -name '*.npz' | wc -l)" ] || fail "non-npz files in ~/fleet_store: refusing"
say "source check OK: no OAI/union path, no symlink, store = npz only"
rx(){ $SSH root@"$HOST" "$@"; }
rx "mkdir -p $R/fast_build/code2 $R/fleet_store $R/novel/frac75 $R/rsna_train $R/fleet_runs $R/hf/hub $R/status" || fail "ssh to the instance"
rx 'command -v rsync' > /dev/null 2>&1 || fail "rsync missing on the instance (vast_setup.sh step S2 installs it)"

# 1 small files: code (no __pycache__), the 4 label/fold csvs, the ConvNeXt HF cache (symlinked snapshot kept)
rsync -a -e "$SSH" --exclude=__pycache__ ~/fast_build/code2/ root@"$HOST":"$R/fast_build/code2/" >> "$L" 2>&1 || fail "code rsync"
rsync -a -e "$SSH" ~/novel/frac75/labels_frac75_logit.csv root@"$HOST":"$R/novel/frac75/" >> "$L" 2>&1 || fail "labels rsync"
rsync -a -e "$SSH" ~/rsna_train/folds5.csv ~/rsna_train/gold74_eval_set.csv ~/rsna_train/llm_labels_v4_blend.csv root@"$HOST":"$R/rsna_train/" >> "$L" 2>&1 || fail "csv rsync"
rsync -a -e "$SSH" ~/.cache/huggingface/hub/models--timm--convnext_tiny.fb_in22k_ft_in1k root@"$HOST":"$R/hf/hub/" >> "$L" 2>&1 || fail "hf cache rsync"
say "small files done"

# 2 the store: 8 parallel rsync streams over sorted shards of the file list
t0=$(date +%s)
find ~/fleet_store -maxdepth 1 -name '*.npz' -printf '%f\n' | sort > $D/store_list.txt
N=$(wc -l < $D/store_list.txt); BYTES=$(du -sb ~/fleet_store | cut -f1)
rm -f $D/store_shard_0*; split -n l/8 -d $D/store_list.txt $D/store_shard_
say "store: $N files, $BYTES bytes, 8 streams"
pids=()
for s in $D/store_shard_0*; do
  rsync -a --partial --files-from="$s" -e "$SSH" ~/fleet_store/ root@"$HOST":"$R/fleet_store/" >> "$L.$(basename $s)" 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
dt=$(( $(date +%s) - t0 )); [ $dt -lt 1 ] && dt=1
[ $rc -eq 0 ] || fail "store rsync (see $L.store_shard_*)"
say "store copied in ${dt}s ($(( BYTES / dt / 1000000 )) MB/s)"

# 3 md5 parity: every store file + code + csvs + HF blob, box vs instance
sig="$N $BYTES $(find ~/fleet_store -maxdepth 1 -name '*.npz' -printf '%T@\n' | sort -n | tail -n 1)"
if [ ! -s $D/store_md5.txt ] || [ "$(cat $D/store_md5.sig 2>/dev/null)" != "$sig" ]; then
  say "computing the box store md5 manifest"
  (cd ~/fleet_store && xargs -a $D/store_list.txt -n 64 -P 4 nice -n 19 md5sum | LC_ALL=C sort -k 2) > $D/store_md5.txt
  echo "$sig" > $D/store_md5.sig
fi
rx "cd $R/fleet_store && find . -maxdepth 1 -name '*.npz' -printf '%f\n' | xargs -n 64 -P 8 md5sum | LC_ALL=C sort -k 2" > $D/inst_md5.txt || fail "instance md5"
if ! cmp -s $D/store_md5.txt $D/inst_md5.txt; then
  bad=$(diff $D/store_md5.txt $D/inst_md5.txt | grep -c '^[<>]')
  say "store md5 differs on $bad line(s): re-sending those files with rsync -c"
  diff $D/store_md5.txt $D/inst_md5.txt | grep '^<' | awk '{print $3}' > $D/resend.txt
  rsync -a -c --files-from=$D/resend.txt -e "$SSH" ~/fleet_store/ root@"$HOST":"$R/fleet_store/" >> "$L" 2>&1
  rx "cd $R/fleet_store && find . -maxdepth 1 -name '*.npz' -printf '%f\n' | xargs -n 64 -P 8 md5sum | LC_ALL=C sort -k 2" > $D/inst_md5.txt
  cmp -s $D/store_md5.txt $D/inst_md5.txt || fail "store md5 parity still fails after a resend ($(diff $D/store_md5.txt $D/inst_md5.txt | grep -c '^[<>]') lines)"
fi
say "store md5 parity: $(wc -l < $D/inst_md5.txt) / $N files IDENTICAL"
small_box=$(cd ~ && md5sum fast_build/code2/fleet_*.py novel/frac75/labels_frac75_logit.csv rsna_train/folds5.csv rsna_train/gold74_eval_set.csv rsna_train/llm_labels_v4_blend.csv | LC_ALL=C sort -k 2)
small_ins=$(rx "cd $R && md5sum fast_build/code2/fleet_*.py novel/frac75/labels_frac75_logit.csv rsna_train/folds5.csv rsna_train/gold74_eval_set.csv rsna_train/llm_labels_v4_blend.csv | LC_ALL=C sort -k 2")
[ "$small_box" = "$small_ins" ] || fail "code/csv md5 differ"
hb=$(rx "sha256sum $R/hf/hub/models--timm--convnext_tiny.fb_in22k_ft_in1k/snapshots/bc48a87f119bae1b06418f5340bfa97ef75d4609/model.safetensors" | cut -c1-64)
[ "$hb" = 77b619e187de71bce58721689da0c009e84cc20677293d7d43105307e34d868a ] || fail "HF weights sha256 differs on the instance ($hb)"
say "code + csv md5 identical ($(echo "$small_box" | wc -l) files); HF weights sha256 77b619e1.. identical"
status "DONE store $N files ${BYTES} bytes in ${dt}s, md5 parity OK $(date '+%F %T')"
say "DONE"
