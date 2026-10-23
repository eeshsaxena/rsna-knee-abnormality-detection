#!/bin/sh
cd /c/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/day/plan1022_work
for a in "tag=base" "tag=lskep lomm=skeptic" "tag=kap72 kapm=0.72" "tag=mstack mpri=stack" "tag=PAREA prox=PAREA" "tag=lskepmzero lomm=skeptic mpri=zero"; do
  t=$(echo $a | sed 's/tag=\([^ ]*\).*/\1/')
  py -3.12 -I plan1022_mc.py $a > run_$t.out 2>&1 || echo "FAIL $t"
  echo "pol $t $(date +%H:%M)"
done
echo POLDONE
