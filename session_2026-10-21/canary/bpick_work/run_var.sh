#!/bin/sh
# bpick_mc variants, sequential (light CPU). MODEL only.
cd /c/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/canary/bpick_work
export PYTHONDONTWRITEBYTECODE=1
for a in "tag=base rank=0" "tag=lskep lomm=skeptic" "tag=mzero mcl=zero" "tag=worst lomm=oai mcl=zero" "tag=mstack mcl=stack" \
         "tag=rhoS0 rhoS=0" "tag=rhoA0 rhoS=0 rhoA=0" "tag=nf15 nf=1.5" "tag=pf5 pf=.5" "tag=skepall lomm=skeptic mcl=zero rhoS=0 nf=1.5" \
         "tag=pure lomm=skeptic mcl=zero pf=.01" "tag=flatU psh=flat ps=logU" "tag=PAREA prox=PAREA" "tag=kpess k_mu=0.6 k_sd=0.3" \
         "tag=fieldBad field=fieldBad" "tag=fieldGood field=fieldGood"; do
  echo "== $a" ; python -B bpick_mc.py $a > /dev/null 2>&1; echo "rc $?"
done
echo ALLDONE
