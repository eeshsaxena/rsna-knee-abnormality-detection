#!/bin/sh
cd /c/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/lastday
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
until grep -q ALLDONE run_apick.log; do sleep 5; done
for a in "tag=base" "tag=flatU psh=flat ps=logU" "tag=PAREA prox=PAREA" "tag=fieldBad field=fieldBad" "tag=kpess k_mu=0.6 k_sd=0.3"; do
  echo "== $a" ; python -B apick_sweep.py $a > /dev/null 2>&1; echo "rc $?"
done
echo ALLDONE
