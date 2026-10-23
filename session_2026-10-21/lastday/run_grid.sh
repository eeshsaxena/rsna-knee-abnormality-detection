#!/bin/sh
# remaining cells of the 3 x 5 stack-prior grid (lomm05 stack/skeptic/oai x MCLmm stack/skep2/neg/zero/eb), base dose world. MODEL only.
cd /c/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/lastday
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for a in "tag=wSKstack lomm=skeptic mcl=stack" "tag=wSK lomm=skeptic malt=neg" "tag=wSKzero lomm=skeptic mcl=zero" "tag=wSK lomm=skeptic malt=eb" \
         "tag=wOAIstack lomm=oai mcl=stack" "tag=wOAIskep2 lomm=oai" "tag=wOAI lomm=oai malt=neg"; do
  echo "== $a" ; python -B apick_mc.py $a > /dev/null 2>&1; echo "rc $?"
done
echo ALLDONE
