#!/bin/bash
# adv_fixtures.sh (2026-10-06, Claude, adversarial verify). CPU only; writes only under ~/gold_prep/verify_gs.
# Fake run roots made of symlinks / small copies; ~/fleet_runs is only read.
V=/home/training/gold_prep/verify_gs; FR0=/home/training/fleet_runs; G=/home/training/gold_prep/gates
PY=/home/training/miniconda3/envs/rsna/bin/python
sc(){ nice -n 19 ionice -c3 env CUDA_VISIBLE_DEVICES= $PY $G/score_run.py "$@"; }
J1=f0_s1043_ttr_mix50
link_j1(){ ln -s $FR0/$J1 $1/$J1; ln -s $FR0/$J1.log $1/$J1.log; }

echo "=== F1: R50-format run (the CPU dry run dryA, 6 val studies) scored with --gate r50"
R=$V/frF1; rm -rf $R; mkdir -p $R; link_j1 $R
ln -s /home/training/gold_prep/r50/dryA $R/f0_s243_r50mix50
grep -v '^\s' /home/training/gold_prep/r50/dryA.log | sed -n '1,/^{"raw"/p' > $R/f0_s243_r50mix50.log
sc --runs_root $R --gate r50 --brief; echo "rc=$?"

echo "=== F2: J1 without its checkpoint (best epoch must come from the epochs/ OOF match)"
R=$V/frF2; rm -rf $R; mkdir -p $R/$J1
ln -s $FR0/$J1/oof_f0_s1043_raw.csv $R/$J1/; ln -s $FR0/$J1/oof_f0_s1043_ema.csv $R/$J1/
ln -s $FR0/$J1/epochs $R/$J1/epochs; ln -s $FR0/$J1.log $R/$J1.log
sc --runs_root $R --gate none --ref none $J1 | grep -E 'best epoch|gold58 /|repro|note'; echo "rc=${PIPESTATUS[0]}"

echo "=== F3: J1 without checkpoint and without epochs/ (log walk)"
R=$V/frF3; rm -rf $R; mkdir -p $R/$J1
ln -s $FR0/$J1/oof_f0_s1043_raw.csv $R/$J1/; ln -s $FR0/$J1/oof_f0_s1043_ema.csv $R/$J1/; ln -s $FR0/$J1.log $R/$J1.log
sc --runs_root $R --gate none --ref none $J1 | grep -E 'best epoch|gold58 /|note'; echo "rc=${PIPESTATUS[0]}"

echo "=== F4: closing line followed by a trailing line (e.g. a shutdown warning): what do scorer and watcher see?"
R=$V/frF4; rm -rf $R; mkdir -p $R; link_j1 $R
ln -s $FR0/f0_s42 $R/f0_s1044_ttr288mix50
{ cat $FR0/f0_s42.log; echo "/usr/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects"; } > $R/f0_s1044_ttr288mix50.log
sc --runs_root $R --gate t288 --brief | tail -n 3; echo "rc=${PIPESTATUS[0]}"

echo "=== F5: wrong-fold run vs J1 ref, and the unscorable / not-finished exit codes"
sc --runs_root $FR0 --gate none --brief f1_s1043_ttr_mix50 | grep -E 'note|f1_s1043'; echo "rc=${PIPESTATUS[0]}"
sc --runs_root $V/frF1 --gate t288 --brief | tail -n 2; echo "rc(t288 missing)=${PIPESTATUS[0]}"
sc --runs_root $FR0 --gate none --brief f0_s143_f2mix50ttr | tail -n 2; echo "rc(diverged, no closing)=${PIPESTATUS[0]}"
echo "=== done"
