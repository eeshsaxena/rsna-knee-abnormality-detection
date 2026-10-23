#!/bin/sh
cd /c/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/private/dose_work
for a in "tag=PAREA prox=PAREA" "tag=flatU psh=flat ps=logU" "tag=fieldBad field=fieldBad" "tag=fieldGood field=fieldGood" "tag=drift1 fshift=1.0" "tag=kpess k_mu=0.6 k_sd=0.3" "tag=seed29 seed=29" "tag=siteWide site_d=1.0 site_s=0.5 site_rc=0.05" "tag=PAREAflatU prox=PAREA psh=flat ps=logU"; do
  t=$(echo $a | sed 's/tag=\([^ ]*\).*/\1/')
  py -3.12 -I dose_mc.py $a > dose_$t.log 2>&1
  echo "done $t"
done
