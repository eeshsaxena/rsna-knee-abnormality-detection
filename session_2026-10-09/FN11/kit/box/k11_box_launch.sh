#!/bin/bash
# k11 box CPU A0 (2026-10-09, Claude; from kfn70_box_launch.sh): FN11 (rsna-sub-k11fn, F .40 / N .60) and FN11-N70
# (rsna-sub-k11fn-n70, F .30 / N .70) on the REAL 19/20-member C + D + E + E2 fleet (dataset -k11w) + N. CUDA hidden,
# nice 19, ionice idle, taskset to 3 CPUs per queue (q1: 34-36, q2: 37-39 => at most 6 CPUs); writes only ~/$A; never
# touches the GPU, ~/queue, ~/fleet_runs, ~/oai_* or a running process. usage:
#   A=k11_test/a0 bash ~/k11_test/a0/k11/box/k11_box_launch.sh <q1|q2|JOB> [...]
# Each argument starts ONE detached queue (setsid nohup bash <queue file>); its jobs run one after another.
# `touch ~/$A/STOP` ends every queue before its next job. Every job runs with TMPDIR / XDG_CACHE_HOME / MPLCONFIGDIR under
# ~/$A (the fleet cell's member dir and the N child's temp dir land there, not in /tmp).
#   q1: fn11_neg_fleet_missing fn11_negx_stale_ds fn11_chain        q2: fn11n70_neg_missing_weights fn11n70_chain
# Harness = k10fnw_box_stage.py (byte copy of the tested FN10-3-N70 harness; fnw_check with the K11 slugs; the fleet
# dataset symlink name from K10_DS_SLUG). FNW_EXPECT_W3 pins the weights; FNW_DIFFERS_FROM = W4 (the 3 studies must tell
# the two weightings apart).
#   fn11_chain / fn11n70_chain   FN-IDS + the real fleet on CPU + N, 3 gold studies (ids/n70_3.csv); gate: KFNBOX PASS
#                                (fnw_check: fn_check + W1-W4: submission.csv == an independent pandas rank blend)
#   fn11_neg_fleet_missing       the fleet dataset not attached: fleet NOT clean, submission.csv ABSENT (KFNBOX PASS)
#   fn11n70_neg_missing_weights  the N weights absent (stub fleet): N NOT clean, submission.csv ABSENT (KFNBOX PASS)
#   fn11_negx_stale_ds           the FN10-3 dataset (k10w bytes, 15 members) mounted under the k11w slug = a stale dataset
#                                version: the fleet cell must refuse (MEMBERS not in the manifest), no submission.csv. The
#                                harness expects a clean run, so its KFNBOX line says FAIL; graded by the K11NEGX line.
set -u
A=${A:?set A=k11_test/a0}
cd ~/$A || exit 1
PYK=~/miniconda3/envs/kmimic/bin/python
S=rsna-knee-cnxt-fleet-k11w
[ -f FN_BUILD_fn11.json ] && [ -f FN_BUILD_fn11n70.json ] && [ -f K5_BUILD_w11.json ] && [ -f ids/n70_3.csv ] && \
  [ -d ds/$S ] && [ -d ds/stale_k10w ] || { echo "not staged (k11_stage_box.sh)"; exit 1; }
H=$(date +%H%M)
mkdir -p ~/$A/tmp ~/$A/cache/mpl || exit 1
E0="TMPDIR=$HOME/$A/tmp XDG_CACHE_HOME=$HOME/$A/cache MPLCONFIGDIR=$HOME/$A/cache/mpl K10_DS_SLUG=$S"
E40="$E0 FNW_EXPECT_W3=0.40,0.60 FNW_DIFFERS_FROM=0.30,0.70"
E70="$E0 FNW_EXPECT_W3=0.30,0.70 FNW_DIFFERS_FROM=0.40,0.60"
HS="$PYK -W ignore k11/box/k10fnw_box_stage.py --weights_dir wts --kernel fn --threads 3 --n_threads 2"
F11="--fn_build FN_BUILD_fn11.json --nb kernel/rsna_sub_k11fn.ipynb"
F70="--fn_build FN_BUILD_fn11n70.json --nb kernel/rsna_sub_k11fn_n70.ipynb"
R="--fleet real --fleet_ds ds/$S --expect_tta K5_BUILD_w11.json --expect_manifest ds/$S/fleet_manifest.json"
RS="--fleet real --fleet_ds ds/stale_k10w --expect_tta K5_BUILD_w11.json --expect_manifest ds/$S/fleet_manifest.json"
job(){   # NAME -> the env + command line, out dir runs/NAME_H
  local o="--out runs/$1_$H"
  case $1 in
    fn11_chain) echo "$E40 $HS $F11 $R --ids ids/n70_3.csv $o";;
    fn11n70_chain) echo "$E70 $HS $F70 $R --ids ids/n70_3.csv $o";;
    fn11_neg_fleet_missing) echo "$E40 $HS $F11 $R --ids ids/neg2.csv --neg fleet_missing $o";;
    fn11n70_neg_missing_weights) echo "$E70 $HS $F70 --fleet stub --ids ids/neg2.csv --neg missing_weights $o";;
    fn11_negx_stale_ds) echo "$E40 $HS $F11 $RS --ids ids/neg2.csv $o";;
    *) echo "";;
  esac
}
grade_x(){   # the stale-dataset negative: fail-loud = fleet NOT clean naming the manifest, no submission.csv, no N result
  local d=runs/fn11_negx_stale_ds_$H l=runs/fn11_negx_stale_ds_$H.log
  local m=$(grep -c "not in the fleet manifest" $l) s=0; [ -f $d/working/submission.csv ] && s=1
  local n=$(grep -c "RESULT OK N stage clean" $l)
  if [ "$m" -ge 1 ] && [ $s -eq 0 ] && [ "$n" -eq 0 ]; then echo "K11NEGX PASS stale k10w dataset under the k11w slug: fleet refused (MEMBERS not in the manifest x$m), submission.csv absent, no N result"
  else echo "K11NEGX FAIL (manifest msg $m, submission.csv present $s, N clean lines $n)"; fi
}
start_queue(){   # queue name, cores, jobs...
  local q=$1 cores=$2; shift 2
  local f=runs/${q}_$H.sh
  { echo "cd ~/$A || exit 1"
    declare -f grade_x | sed "s/\$H/$H/g"
    for j in "$@"; do
      c=$(job "$j"); [ -n "$c" ] || { echo "unknown job $j" >&2; return 1; }
      echo "[ -e STOP ] && { echo \"\$(date +%H:%M:%S) STOP file: queue ends before $j\"; exit 0; }"
      echo "echo \"\$(date +%H:%M:%S) START $j\""
      echo "CUDA_VISIBLE_DEVICES= taskset -c $cores nice -n 19 ionice -c3 env $c > runs/${j}_$H.log 2>&1 < /dev/null; echo \"\$(date +%H:%M:%S) END $j rc \$? | left in tmp: \$(ls tmp | wc -l)\""
      [ "$j" = fn11_negx_stale_ds ] && echo "grade_x | tee -a runs/${j}_$H.log"
    done
    echo "echo \"\$(date +%H:%M:%S) QUEUE $q DONE\""; } > "$f" || return 1
  setsid nohup bash "$f" > runs/${q}_$H.log 2>&1 < /dev/null &
  echo "started $q (pid $!) on CPUs $cores: $* | log ~/$A/runs/${q}_$H.log"
}
rm -f STOP
for arg in "$@"; do
  case $arg in
    q1) start_queue q1 34-36 fn11_neg_fleet_missing fn11_negx_stale_ds fn11_chain ;;
    q2) start_queue q2 37-39 fn11n70_neg_missing_weights fn11n70_chain ;;
    *) start_queue "$arg" 34-36 "$arg" ;;
  esac
done
