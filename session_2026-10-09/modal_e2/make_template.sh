#!/bin/bash
# make_template.sh: build ~/modal_e2/meta/argv_template.txt from the box queue line "Q4 TTR288 s1045 f0 frac75",
# changing ONLY --seed (1045 -> 1046), --fold (0 -> @F@) and --out (-> @OUT@); every path "~/" -> "@R@/".
# Fails loudly if the line is missing or if anything other than those 3 tokens would change.
set -eu
Q=~/queue/queue.txt; M=~/modal_e2/meta
L=$(grep -E '^#RUN .* Q4 TTR288 s1045 f0 frac75 \| ~/fleet_runs/f0_s1045_ttr288frac75 \|' "$Q" | tail -n 1)
[ -n "$L" ] || { echo "FAIL: queue line Q4 TTR288 s1045 f0 frac75 not found"; exit 1; }
SRC=$(printf '%s' "${L##*|}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
printf '%s\n' "$SRC" > $M/argv_source_queue_line.txt
NEW=$(printf '%s\n' "$SRC" | tr ' ' '\n' | awk '
  prev=="--seed" {if ($0!="1045") {print "BADSEED"; exit 1}; $0="1046"}
  prev=="--fold" {if ($0!="0") {print "BADFOLD"; exit 1}; $0="@F@"}
  prev=="--out"  {if ($0!="~/fleet_runs/f0_s1045_ttr288frac75") {print "BADOUT"; exit 1}; $0="@OUT@"}
  {print; prev=$0}' | sed 's#^~/#@R@/#' | tr '\n' ' ' | sed 's/ $//')
printf '%s\n' "$NEW" > $M/argv_template.txt
NS=$(printf '%s\n' "$SRC" | wc -w); NN=$(wc -w < $M/argv_template.txt)
ND=$(diff <(printf '%s\n' "$SRC" | tr ' ' '\n' | sed 's#^~/#@R@/#') <(tr ' ' '\n' < $M/argv_template.txt) | grep -c '^>' || true)
echo "source tokens $NS | template tokens $NN | tokens changed $ND (want 3: seed, fold, out)"
diff <(printf '%s\n' "$SRC" | tr ' ' '\n' | sed 's#^~/#@R@/#') <(tr ' ' '\n' < $M/argv_template.txt) || true
[ "$NS" = "$NN" ] && [ "$ND" = 3 ] || { echo "FAIL: template differs from the queue line in more than seed/fold/out"; exit 1; }
grep -q -- '--epochs 21' $M/argv_template.txt && grep -q -- '--grad_ckpt 0' $M/argv_template.txt && grep -q -- '--ttr_size 288' $M/argv_template.txt \
  && grep -q -- '--labels @R@/novel/frac75/labels_frac75_logit.csv' $M/argv_template.txt && grep -q -- '--workers 10' $M/argv_template.txt || { echo "FAIL: key tokens missing"; exit 1; }
echo "OK template: $(cat $M/argv_template.txt)"
