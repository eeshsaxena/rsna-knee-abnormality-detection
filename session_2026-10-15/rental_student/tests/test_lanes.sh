#!/bin/bash
# test_lanes.sh (BOX, seconds, no trainer: a fake "python" that prints trainer-like lines; writes only ~/rstud_tests/lanes):
# rstud_lane.sh + start_lanes.sh + stop_fold.sh + status.sh: a paired fold is stopped by its recorded trainer pid (works
# without procps), moved aside, re-run by an "any" lane on the first free GPU; finished folds get a receipt + sha256 list.
set -u
T=$HOME/rstud_tests/lanes; rm -rf $T; R=$T/workspace/rsna; mkdir -p $R/kit $R/status $R/fleet_runs $R/r7t_labels/real $R/fast_build/code2
cp $HOME/rstud/src/remote/{rstud_lane.sh,start_lanes.sh,stop_fold.sh,status.sh,kit.env,argv_template.txt} $R/kit/
echo "StudyInstanceUID" > $R/r7t_labels/real/labels_r7t_b.csv; echo "# fake" > $R/fast_build/code2/fleet_train.py; touch $R/status/setup.ok
cat > $T/fakepy.sh <<'FP'
#!/bin/bash
# fake python: $1 = fleet_train.py, then the argv; sleeps FAKE_S seconds per "epoch"
F=$(echo "$*" | sed -E 's/.*--fold ([0-9]+).*/\1/'); O=$(echo "$*" | sed -E 's/.*--out ([^ ]+).*/\1/'); mkdir -p $O/epochs
echo "ep 0 it 0/10 loss 0.7 1.00s/it"; sleep ${FAKE_S:-6}; echo "ep 0 TRAIN-ONLY 0.1 min (G0 timing)"
echo x > $O/f${F}_s1047_raw_ttr224-320x24_cv5000_best.pt; echo x > $O/f${F}_s1047_ema_ttr224-320x24_cv5000_best.pt
echo "a,b" > $O/oof_f${F}_s1047_raw.csv; echo "a,b" > $O/oof_f${F}_s1047_ema.csv; echo e > $O/epochs/f${F}_ep6_best.pt
echo '{"raw": 0.5, "ema": 0.5}'
FP
chmod +x $T/fakepy.sh
printf 'FOLDS="0 1 2"\nDEADLINE_EPOCH=%s\nFOLD_EST_S=30\nPYI=%s\nNG=2\n' $(( $(date +%s) + 3600 )) "$T/fakepy.sh" > $R/kit/plan.env
export PYR=$HOME/miniconda3/envs/rsna/bin/python RSNA_ROOT=$R MOCK_CPU=1 MOCK_CPUSETS="0-39 0-39" LANE_ANY_DELAY_S=2 LANE_ANY_POLL_S=2 FAKE_S=8
sed -i 's#PRE="taskset -c $CS nice -n 19 ionice -c3"#PRE="nice -n 19"#' $R/kit/rstud_lane.sh   # test copy only: no taskset needed
bash $R/kit/start_lanes.sh "0:0 1:2 0:1"; sleep 3
echo "--- running: $(ls $R/status/*.running | xargs -n1 basename | tr '\n' ' ') | tpids: $(cat $R/status/f*.tpid | tr '\n' ' ')"
p1=$(cat $R/status/f1.tpid); bash $R/kit/stop_fold.sh 1; sleep 1
kill -0 $p1 2>/dev/null && echo "WRONG: fold-1 trainer pid $p1 still alive" || echo "ok: fold-1 trainer pid $p1 gone"
bash $R/kit/start_lanes.sh "any:1"
for i in $(seq 1 40); do [ -e $R/status/f0.done ] && [ -e $R/status/f1.done ] && [ -e $R/status/f2.done ] && break; sleep 1; done
bash $R/kit/status.sh | grep -E 'FOLD|ALL_'
for F in 0 1 2; do N=f${F}_s1047_ttr_r7tb; ( cd $R/fleet_runs && sha256sum -c --quiet $N.sha256 && echo "ok: $N sha256 list verifies ($(wc -l < $N.sha256) files), receipt gpu $(sed -n 's/.*"gpu_index": "\([0-9]*\)".*/\1/p' $N.receipt.json)" ); done
ls $R/fleet_runs | grep smokepaired && echo "ok: the stopped run was moved aside"
grep -h "any-lane took" $R/logs/lane_*.log
