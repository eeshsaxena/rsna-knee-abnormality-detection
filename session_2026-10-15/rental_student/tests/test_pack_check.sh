#!/bin/bash
# test_pack_check.sh (BOX, CPU, seconds; writes only under ~/rstud_tests/pack): negative + positive controls of the packer's
# OAI assertion and T_b provenance checks. Each line: EXPECT vs GOT.
set -u
T=$HOME/rstud_tests/pack; rm -rf $T; mkdir -p $T; PY=$HOME/miniconda3/envs/rsna/bin/python; PC=$HOME/rstud/src/box/pack_check.py
SIM=$HOME/r7t_labels/sim_a1/final/b1; FRC=$HOME/r7t_labels/fleet/fleet_oof_receipt_fleetOF.json
n=0; fails=0
t(){ local want=$1 name=$2; shift 2; local out rc; out=$("$@" 2>&1); rc=$?; n=$((n + 1))
  local got=PASS; [ $rc -eq 3 ] && got=STOP; [ $rc -ne 0 ] && [ $rc -ne 3 ] && got="ERR$rc"
  [ "$got" = "$want" ] && r=ok || { r=WRONG; fails=$((fails + 1)); }
  echo "$r | $name | expect $want got $got | $(echo "$out" | tail -n 1 | cut -c1-170)"; }
TAB=$SIM/SIM_labels_r7t_b.csv; RC=$SIM/SIM_teacher_labels_receipt.json
t PASS "DRY table + SIM receipt + fleet receipt" $PY $PC table $TAB --mode DRY --receipt $RC --fleet_receipt $FRC
sed '2s/^1\.2\.826\.0\.1\.3680043\.8\.498\.[0-9]*/oai_9999/' $TAB > $T/oaiid.csv
t STOP "table with an oai_ id row" $PY $PC table $T/oaiid.csv --mode DRY
t STOP "REAL mode without LABELS_READY / receipt" $PY $PC table $TAB --mode REAL
echo "$(printf '0%.0s' $(seq 1 64))  labels_r7t_b.csv" > $T/READY_bad; echo "$(sha256sum $TAB | cut -c1-64)  labels_r7t_b.csv" > $T/READY_ok
t STOP "LABELS_READY sha256 != table" $PY $PC table $TAB --mode DRY --ready $T/READY_bad --table_name labels_r7t_b.csv
t PASS "LABELS_READY sha256 == table" $PY $PC table $TAB --mode DRY --ready $T/READY_ok --table_name labels_r7t_b.csv
$PY -c "import json; r=json.load(open('$FRC')); r['sets']['DT']={'tag':'s1043_ttr_frac75oaiDT'}; json.dump(r, open('$T/fleet_dt.json','w'))"
t STOP "fleet receipt with an OAI DT set" $PY $PC table $TAB --mode DRY --receipt $RC --fleet_receipt $T/fleet_dt.json
$PY -c "import json; r=json.load(open('$RC')); r['inputs']['fleet_oof']['md5']='0'*32; json.dump(r, open('$T/rc_fleetmd5.json','w'))"
t STOP "receipt fleet OOF md5 != the OAI-free mean" $PY $PC table $TAB --mode DRY --receipt $T/rc_fleetmd5.json
$PY -c "import json; r=json.load(open('$RC')); r['tables']['b']['oai_l13']=0.1; json.dump(r, open('$T/rc_extra.json','w'))"
t STOP "receipt T_b with an extra component" $PY $PC table $TAB --mode DRY --receipt $T/rc_extra.json
$PY -c "import json; r=json.load(open('$RC')); r['real']=True; r['teacher_tag']='R7OOF'; r['label_gate']['train_table']='labels_r7t_b.csv'; json.dump(r, open('$T/rc_real.json','w'))"
t STOP "REAL mode with a SIM teacher input path" $PY $PC table $TAB --mode REAL --ready $T/READY_ok --receipt $T/rc_real.json --fleet_receipt $FRC
echo "FAIL labels_r7t_b.csv label gate" > $T/verdict_fail
t STOP "labels_gate verdict FAIL" $PY $PC table $TAB --mode DRY --verdict $T/verdict_fail
# OAI assertion on a mini bundle (subset mode)
B=$T/mini; mkdir -p $B/{fast_build/code2,fleet_store,r7t_labels/real,rsna_train,hf/hub,kit}
cp $HOME/fast_build/code2/*.py $B/fast_build/code2/; cp $TAB $B/r7t_labels/real/labels_r7t_b.csv; cp $HOME/rsna_train/folds5.csv $HOME/rsna_train/gold74_eval_set.csv $B/rsna_train/
for s in $(ls $HOME/novel/frac75/dry/store | head -n 3); do ln $(readlink -f $HOME/novel/frac75/dry/store/$s) $B/fleet_store/$s; done
t PASS "mini bundle (3 studies, subset)" $PY $PC oai $B --subset
t STOP "mini bundle without --subset (store != 4,407)" $PY $PC oai $B
touch $B/fleet_store/oai_9001234.npz; t STOP "store file named oai_*" $PY $PC oai $B --subset; rm $B/fleet_store/oai_9001234.npz
ln -s $HOME/fleet_store $B/rsna_train/link; t STOP "symlink outside hf/" $PY $PC oai $B --subset; rm $B/rsna_train/link
echo "# load /data/OAI/x" >> $B/fast_build/code2/fleet_aug.py; t STOP "code mentions OAI" $PY $PC oai $B --subset; cp $HOME/fast_build/code2/fleet_aug.py $B/fast_build/code2/
cp $T/oaiid.csv $B/r7t_labels/real/labels_r7t_b.csv; t STOP "bundle table with an oai_ id" $PY $PC oai $B --subset; cp $TAB $B/r7t_labels/real/labels_r7t_b.csv
t STOP "pack.sh REAL before LABELS_READY exists" bash $HOME/rstud/src/box/pack.sh REAL $T/realbundle
echo "== $n checks, $fails WRONG"
