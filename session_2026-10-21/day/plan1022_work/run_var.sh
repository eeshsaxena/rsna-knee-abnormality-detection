#!/bin/sh
cd /c/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/day/plan1022_work
for a in "tag=lskep lomm=skeptic" "tag=mzero mpri=zero" "tag=mstack mpri=stack" "tag=kap0 kapm=0 kaps=0.0001" "tag=kap72 kapm=0.72" "tag=PAREA prox=PAREA" "tag=flatU psh=flat ps=logU" "tag=kpess k_mu=0.6 k_sd=0.3" "tag=lskepmzero lomm=skeptic mpri=zero"; do
  t=$(echo $a | sed 's/tag=\([^ ]*\).*/\1/')
  py -3.12 -I plan1022_mc.py $a > run_$t.out 2>&1
  py -3.12 -I summ1022.py $t > /dev/null 2>&1
  echo "done $t $(date +%H:%M)"
done
echo ALLDONE
