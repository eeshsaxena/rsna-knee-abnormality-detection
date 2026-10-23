#!/bin/sh
# second batch, after the first runner ends (sequential; light CPU). MODEL only.
cd /c/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/day
until grep -q ALLDONE skeptic_wc_run.log; do sleep 5; done
for a in "tag=base" "tag=pure lomm=skeptic mcl=zero pf=.01" "tag=pureneg lomm=oai mcl=zero pf=.01" "tag=rhoA0 rhoS=0 rhoA=0"; do
  echo "== $a" ; python -B skeptic_wc.py $a > /dev/null 2>&1; echo "rc $?"
done
echo ALLDONE2
