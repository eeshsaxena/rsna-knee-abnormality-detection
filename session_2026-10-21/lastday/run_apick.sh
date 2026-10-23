#!/bin/sh
# apick_mc variants, sequential, light CPU (2 threads). MODEL only.
cd /c/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/lastday
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for a in "tag=base rank=0" "tag=lskep lomm=skeptic" "tag=mzero mcl=zero" "tag=worst lomm=oai mcl=zero" "tag=mstack mcl=stack" \
         "tag=rhoS0 rhoS=0" "tag=rhoA0 rhoS=0 rhoA=0" "tag=nf15 nf=1.5" "tag=pf5 pf=.5" "tag=skepall lomm=skeptic mcl=zero rhoS=0 nf=1.5" \
         "tag=pure lomm=skeptic mcl=zero pf=.01" "tag=flatU psh=flat ps=logU" "tag=PAREA prox=PAREA" "tag=kpess k_mu=0.6 k_sd=0.3" \
         "tag=fieldBad field=fieldBad" "tag=fieldGood field=fieldGood" "tag=c4over c4shift=2" "tag=c4real c4shift=-2" \
         "tag=base malt=neg" "tag=base malt=eb" "tag=worst lomm=oai malt=eb" "tag=worstA0 lomm=oai malt=eb rhoS=0 rhoA=0" \
         "tag=flatU psh=flat ps=logU malt=eb" "tag=PAREA prox=PAREA lomm=oai malt=eb" "tag=skepallnf2 lomm=skeptic mcl=zero rhoS=0 nf=2"; do
  echo "== $a" ; python -B apick_mc.py $a > /dev/null 2>&1; echo "rc $?"
done
echo ALLDONE
