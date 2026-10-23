#!/bin/sh
# run_sens.sh: sensitivity rows for bpick_mc.py (rep 4 each; CPU numpy, ~20 s per row)
cd /c/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push
: > sens.log
for a in "tag=p_more prior=more" "tag=p_ladder prior=ladder" "tag=p_orig prior=orig" "tag=p_null prior=null" "tag=p_neg prior=neg" \
         "tag=noise1.5 noise=1.5" "tag=k_skep k_mu=0.6 k_sd=0.3" "tag=k_pess k_mu=0.4 k_sd=0.4" "tag=k_opt k_mu=1.0 k_sd=0.1" \
         "tag=rho0.6 rho=0.6" "tag=rho0.9 rho=0.9" \
         "tag=o_skepS6 our=skepS6" "tag=o_strongS6 our=strongS6" "tag=o_tabReal our=tabReal" "tag=o_tabOver our=tabOver" \
         "tag=o_r6better our=r6better" "tag=o_noise1.5 our=noise1.5" "tag=o_bigKnobs our=bigKnobs" "tag=o_smallKnobs our=smallKnobs" \
         "tag=f_fieldBad field=fieldBad" "tag=f_fieldGood field=fieldGood" "tag=f_Wbig field=Wbig" \
         "tag=g_low g_mu=0.5e-4 g_sd=0.5e-4" "tag=g_high g_mu=2.5e-4" "tag=u_flat u10=0.454 u20=0.946" \
         "tag=d30_opt d30shift=0" "tag=d30_pess d30shift=-3e-4" "tag=k_pess_noise1.5 k_mu=0.4 k_sd=0.4 noise=1.5"; do
  python bpick_mc.py rep=4 $a >> sens.log 2>&1
done
echo SENS_DONE >> sens.log
