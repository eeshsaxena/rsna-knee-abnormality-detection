#!/bin/bash
# k12 box CPU A0 (2026-10-09, Claude; from k11_box_launch.sh): a K12 kernel pair (rsna-sub-k12fn F .40/N .60 and
# rsna-sub-k12fn-n70) on the REAL merged fleet (k11w + k12w via the K12-MERGE cell) + N. CUDA hidden, nice 19, ionice idle,
# 3 CPUs per queue (q1: 34-36, q2: 37-39); writes only ~/$A. usage: A=k12_test/a0 bash ~/$A/k12/box/k12_box_launch.sh q1 q2
#   q1: k12_neg_fleet_missing k12_negx_k12w_missing k12_negx_tampered k12_negx_base_stale k12_chain
#   q2: k12n70_neg_missing_weights k12n70_chain
# Harness k12fnw_box_stage.py (generated from the tested FN11 harness: + the 2nd dataset symlink K12_DS2, the 4-code-cell
# layout, /tmp/k12_fleet -> the run dir). Chains: KFNBOX PASS (fnw_check: fn_check + W1-W4). negx jobs (the harness
# expects a clean run, so their KFNBOX says FAIL by design) are graded by K12NEGX lines: the K12-MERGE cell must print
# RESULT FAIL naming the cause, no submission.csv, no N result:
#   k12_negx_k12w_missing  the D-OAI weights dataset not attached        -> 'k12w ... not attached'
#   k12_negx_tampered      D-OAI f1 replaced by set D f1 bytes, same name  -> 'sha256'
#   k12_negx_base_stale    the FN10-3 dataset (k10w bytes) as the base     -> 'fleet_manifest.json != the build'
set -u
A=${A:?set A=k12_test/a0}
cd ~/$A || exit 1
PYK=~/miniconda3/envs/kmimic/bin/python
S11=rsna-knee-cnxt-fleet-k11w; S12=rsna-knee-cnxt-fleet-k12w
[ -f FN_BUILD_k12.json ] && [ -f FN_BUILD_k12n70.json ] && [ -f K12_BUILD.json ] && [ -f merged_manifest.json ] && \
  [ -d ds/$S11 ] && [ -d ds/$S12 ] && [ -d ds/k12w_tampered ] && [ -d ds/stale_k10w ] || { echo "not staged (k12_stage_box.sh)"; exit 1; }
H=$(date +%H%M)
mkdir -p ~/$A/tmp ~/$A/cache/mpl || exit 1
E0="TMPDIR=$HOME/$A/tmp XDG_CACHE_HOME=$HOME/$A/cache MPLCONFIGDIR=$HOME/$A/cache/mpl K10_DS_SLUG=$S11"
E40="$E0 K12_DS2=$S12=ds/$S12 FNW_EXPECT_W3=0.40,0.60 FNW_DIFFERS_FROM=0.30,0.70"
E70="$E0 K12_DS2=$S12=ds/$S12 FNW_EXPECT_W3=0.30,0.70 FNW_DIFFERS_FROM=0.40,0.60"
HS="$PYK -W ignore k12/box/k12fnw_box_stage.py --weights_dir wts --kernel fn --threads 3 --n_threads 2"
F12="--fn_build FN_BUILD_k12.json --nb kernel/rsna_sub_k12fn.ipynb"
F70="--fn_build FN_BUILD_k12n70.json --nb kernel/rsna_sub_k12fn_n70.ipynb"
R="--expect_tta K12_BUILD.json --expect_manifest merged_manifest.json"
job(){
  local o="--out runs/$1_$H"
  case $1 in
    k12_chain) echo "$E40 $HS $F12 --fleet real --fleet_ds ds/$S11 $R --ids ids/n70_3.csv $o";;
    k12n70_chain) echo "$E70 $HS $F70 --fleet real --fleet_ds ds/$S11 $R --ids ids/n70_3.csv $o";;
    k12_neg_fleet_missing) echo "$E40 $HS $F12 --fleet real --fleet_ds ds/$S11 $R --ids ids/neg2.csv --neg fleet_missing $o";;
    k12n70_neg_missing_weights) echo "$E70 $HS $F70 --fleet stub --ids ids/neg2.csv --neg missing_weights $o";;
    k12_negx_k12w_missing) echo "$E0 K12_DS2= FNW_EXPECT_W3=0.40,0.60 $HS $F12 --fleet real --fleet_ds ds/$S11 $R --ids ids/neg2.csv $o";;
    k12_negx_tampered) echo "$E0 K12_DS2=$S12=ds/k12w_tampered FNW_EXPECT_W3=0.40,0.60 $HS $F12 --fleet real --fleet_ds ds/$S11 $R --ids ids/neg2.csv $o";;
    k12_negx_base_stale) echo "$E40 $HS $F12 --fleet real --fleet_ds ds/stale_k10w $R --ids ids/neg2.csv $o";;
    *) echo "";;
  esac
}
start_queue(){
  local q=$1 cores=$2; shift 2
  local f=runs/${q}_$H.sh
  { echo "cd ~/$A || exit 1"
    echo "gx(){ local l=runs/\$1_$H.log d=runs/\$1_$H; local m=\$(grep -c \"k12-merge.*RESULT FAIL.*\$2\" \$l); local s=0; [ -f \$d/working/submission.csv ] && s=1; local n=\$(grep -c 'RESULT OK N stage clean' \$l); if [ \"\$m\" -ge 1 ] && [ \$s -eq 0 ] && [ \"\$n\" -eq 0 ]; then echo \"K12NEGX PASS \$1: merge refused (\$2), submission.csv absent, no N result\"; else echo \"K12NEGX FAIL \$1 (merge msg \$m, submission.csv \$s, N clean \$n)\"; fi; }"
    for j in "$@"; do
      c=$(job "$j"); [ -n "$c" ] || { echo "unknown job $j" >&2; return 1; }
      echo "[ -e STOP ] && { echo \"\$(date +%H:%M:%S) STOP file: queue ends before $j\"; exit 0; }"
      echo "echo \"\$(date +%H:%M:%S) START $j\""
      echo "CUDA_VISIBLE_DEVICES= taskset -c $cores nice -n 19 ionice -c3 env $c > runs/${j}_$H.log 2>&1 < /dev/null; echo \"\$(date +%H:%M:%S) END $j rc \$? | left in tmp: \$(ls tmp | wc -l)\""
      case $j in
        k12_negx_k12w_missing) echo "gx $j 'not attached' | tee -a runs/${j}_$H.log";;
        k12_negx_tampered) echo "gx $j 'sha256' | tee -a runs/${j}_$H.log";;
        k12_negx_base_stale) echo "gx $j 'fleet_manifest.json != the build' | tee -a runs/${j}_$H.log";;
      esac
    done
    echo "echo \"\$(date +%H:%M:%S) QUEUE $q DONE\""; } > "$f" || return 1
  setsid nohup bash "$f" > runs/${q}_$H.log 2>&1 < /dev/null &
  echo "started $q (pid $!) on CPUs $cores: $* | log ~/$A/runs/${q}_$H.log"
}
rm -f STOP
for arg in "$@"; do
  case $arg in
    q1) start_queue q1 34-36 k12_neg_fleet_missing k12_negx_k12w_missing k12_negx_tampered k12_negx_base_stale k12_chain ;;
    q2) start_queue q2 37-39 k12n70_neg_missing_weights k12n70_chain ;;
    *) start_queue "$arg" 34-36 "$arg" ;;
  esac
done
