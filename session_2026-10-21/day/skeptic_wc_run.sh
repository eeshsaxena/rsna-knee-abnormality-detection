#!/bin/sh
# skeptic_wc variants, sequential (light CPU). MODEL only.
cd /c/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/day
for a in "tag=rhoS0 rhoS=0" "tag=rhoA5 rhoS=0 rhoA=.5" "tag=nf15 nf=1.5" "tag=pf5 pf=.5" "tag=noise rhoS=0 nf=1.5 pf=.5" \
         "tag=skepall lomm=skeptic mcl=zero rhoS=0 nf=1.5" "tag=worst lomm=oai mcl=zero" "tag=worstsplit lomm=oai mcl=zero rhoS=0 nf=1.5" \
         "tag=mstack mcl=stack" "tag=lskep lomm=skeptic"; do
  echo "== $a" ; python -B skeptic_wc.py $a > /dev/null 2>&1; echo "rc $?"
done
echo ALLDONE
