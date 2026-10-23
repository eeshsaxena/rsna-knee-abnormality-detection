#!/bin/bash
# K10 box CPU A0 (CUDA hidden, nice 19, ionice idle; never touches the GPU; writes only ~/$A). usage:
#   A=k10_test/a0 bash ~/k10_test/a0/k10/box/k10_box_launch.sh <q1|q2|JOB> [...]
# Each argument starts ONE detached queue (setsid nohup bash <queue file>; ssh returns at once) whose jobs run one after
# another, pinned with taskset to 3 cores (q1: CPUs 34-36, q2: 37-39, a single JOB: 34-36): two queues = at most 6 CPUs.
# Torch threads 3 in the fleet worker, 2 in the N child. `touch ~/$A/STOP` ends every queue before its next job.
#   q1: neg_k9slug main n_neg_fleet_fail n_neg_child_sha n_neg_missing_weights n_chain solo
#   q2: fn_neg_fleet_missing fn3_neg_fleet_missing fn_neg_fleet_fail fn_neg_child_crash fn_chain fn3_chain n_stub fn_stub
# Jobs:
#   neg_k9slug  K10-main notebook with the fleet dataset mounted under the K9 slug: the K10-DS cell must find no
#               dataset ("not attached"), strict -> no submission.csv (k10_box_stage BOXSTAGE PASS = consistent fail-loud)
#   main / solo K10-main / K10-solo: real 10-member E + D fleet on the A0 pair, per-member OOF parity (fold-0 members)
#   n_chain     K10N-a cells 31 + 32 (real fleet + N); n_stub: N stage on 4 studies, stubbed strict fleet
#   n_neg_*     K10N-a negatives (stub fleet, 2 studies): N NOT clean, submission.csv ABSENT, why names the cause
#   fn_chain / fn3_chain  FN10 / FN10-3: FN-IDS + real fleet (10 / 15 members) + N; fn_stub: FN10 with a stub fleet
#   fn_neg_* / fn3_neg_fleet_missing  FN negatives (fleet dataset not attached; stub fleet_fail; child_crash)
# Logs ~/$A/runs/<NAME>_<HHMM>.log, queue logs ~/$A/runs/<q>_<HHMM>.log (one line per job: START / END rc).
set -u
A=${A:?set A=k10_test/a0 or k10_test/a0sim}
cd ~/$A || exit 1
PYK=~/miniconda3/envs/kmimic/bin/python
[ -f K5_BUILD.json ] && [ -f K10N_BUILD.json ] && [ -f FN_BUILD_fn10.json ] && [ -d ds/rsna-knee-cnxt-fleet-k10 ] || { echo "not staged (k10_stage_box.sh)"; exit 1; }
H=$(date +%H%M)
OOFARGS=$($PYK - ids/a0.csv <<'EOF'
import csv, json, sys
man = json.load(open('ds/rsna-knee-cnxt-fleet-k10/fleet_manifest.json'))['checkpoints']
picks = json.load(open('K5_BUILD.json'))['picks']
oof_by_file = {r['file']: r['oof'] for r in json.load(open('members.json'))['members']}
mem = {p['staged']: oof_by_file[p['file']] for p in picks}
assert not [n for n in man if n not in mem], 'no OOF for a manifest member'
ids = {r['StudyInstanceUID'] for r in csv.DictReader(open(sys.argv[1]))}
have = [n for n in man if ids & {r['StudyInstanceUID'] for r in csv.DictReader(open(mem[n]))}]
assert have, 'no manifest member has an A0 study in its OOF'
print(' '.join(f'--oof_member {n}={mem[n]}' for n in have))
EOF
) || { echo "OOF args failed"; exit 1; }
A0_AGE_H=${A0_AGE_H:-7.0}
K2S="$PYK -W ignore k10/box/k10_box_stage.py --cpu --fleet_ds ds/rsna-knee-cnxt-fleet-k10 --threads 3 --expect_tta K5_BUILD.json"
NS="$PYK -W ignore k10/box/k10n_box_stage.py --k9n_build K10N_BUILD.json --weights_dir wts --nb kernel/rsna_sub_k10n_a.ipynb --tag a --threads 3 --n_threads 2"
FS="$PYK -W ignore k10/box/k10fn_box_stage.py --weights_dir wts --kernel fn --threads 3 --n_threads 2"
FN10="--fn_build FN_BUILD_fn10.json --nb kernel/rsna_sub_k10fn.ipynb"
FN3="--fn_build FN_BUILD_fn10_3.json --nb kernel/rsna_sub_k10fn3.ipynb"
R10="--fleet real --fleet_ds ds/rsna-knee-cnxt-fleet-k10 --expect_tta K5_BUILD.json --expect_manifest ds/rsna-knee-cnxt-fleet-k10/fleet_manifest.json"
R3="--fleet real --fleet_ds ds/rsna-knee-cnxt-fleet-k10w --expect_tta K5_BUILD_w.json --expect_manifest ds/rsna-knee-cnxt-fleet-k10w/fleet_manifest.json"
job(){   # NAME -> the command line (env prefix + command), out dir runs/NAME_H
  local o="--out runs/$1_$H"
  case $1 in
    neg_k9slug) echo "K10_DS_SLUG=rsna-knee-cnxt-fleet $K2S --nb kernel/rsna_sub_k10_fleet.ipynb --ids ids/neg2.csv --age_h $A0_AGE_H $o";;
    main) echo "$K2S --nb kernel/rsna_sub_k10_fleet.ipynb --ids ids/a0.csv --age_h $A0_AGE_H $OOFARGS $o";;
    solo) echo "$K2S --nb kernel/rsna_sub_k10_solo.ipynb --ids ids/a0.csv --age_h $A0_AGE_H $OOFARGS $o";;
    n_chain) echo "$NS $R10 --ids ids/a0.csv $o";;
    n_stub) echo "$NS --fleet stub --ids ids/n4.csv $o";;
    n_neg_*) echo "$NS --fleet stub --ids ids/neg2.csv --neg ${1#n_neg_} $o";;
    fn_chain) echo "$FS $FN10 $R10 --ids ids/a0.csv $o";;
    fn_stub) echo "$FS $FN10 --fleet stub --ids ids/n4.csv $o";;
    fn_neg_fleet_missing) echo "$FS $FN10 $R10 --ids ids/neg2.csv --neg fleet_missing $o";;
    fn_neg_*) echo "$FS $FN10 --fleet stub --ids ids/neg2.csv --neg ${1#fn_neg_} $o";;
    fn3_chain) echo "K10_DS_SLUG=rsna-knee-cnxt-fleet-k10w $FS $FN3 $R3 --ids ids/a0.csv $o";;
    fn3_neg_fleet_missing) echo "K10_DS_SLUG=rsna-knee-cnxt-fleet-k10w $FS $FN3 $R3 --ids ids/neg2.csv --neg fleet_missing $o";;
    *) echo "";;
  esac
}
start_queue(){   # queue name, cores, jobs...
  local q=$1 cores=$2; shift 2
  local f=runs/${q}_$H.sh
  { echo "cd ~/$A || exit 1"
    for j in "$@"; do
      c=$(job "$j"); [ -n "$c" ] || { echo "unknown job $j" >&2; return 1; }
      echo "[ -e STOP ] && { echo \"\$(date +%H:%M:%S) STOP file: queue ends before $j\"; exit 0; }"
      echo "echo \"\$(date +%H:%M:%S) START $j\""
      echo "CUDA_VISIBLE_DEVICES= taskset -c $cores nice -n 19 ionice -c3 env $c > runs/${j}_$H.log 2>&1 < /dev/null; echo \"\$(date +%H:%M:%S) END $j rc \$?\""
    done
    echo "echo \"\$(date +%H:%M:%S) QUEUE $q DONE\""; } > "$f" || return 1
  setsid nohup bash "$f" > runs/${q}_$H.log 2>&1 < /dev/null &
  echo "started $q (pid $!) on CPUs $cores: $* | log ~/$A/runs/${q}_$H.log"
}
rm -f STOP
for arg in "$@"; do
  case $arg in
    q1) start_queue q1 34-36 neg_k9slug main n_neg_fleet_fail n_neg_child_sha n_neg_missing_weights n_chain solo ;;
    q2) start_queue q2 37-39 fn_neg_fleet_missing fn3_neg_fleet_missing fn_neg_fleet_fail fn_neg_child_crash fn_chain fn3_chain n_stub fn_stub ;;
    *) start_queue "$arg" 34-36 "$arg" ;;
  esac
done
