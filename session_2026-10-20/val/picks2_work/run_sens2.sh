#!/bin/sh
# run_sens2.sh: sensitivity rows for picks2_mc.py (rep 2 each; CPU numpy ~20 s per row). Read-only on every input.
cd /c/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/val/picks2_work
: > sens2.log
python picks2_mc.py tag=base info_a=1 > picks2_base.log 2>&1
for a in "tag=a_PFT_N01_logU abc=PFT_N01_logU" "tag=a_PFT_flat_logN abc=PFT_flat_logN" "tag=a_PFT_flat_logU abc=PFT_flat_logU" \
         "tag=a_PAREA_N01_logN abc=PAREA_N01_logN" "tag=a_PAREA_flat_logU abc=PAREA_flat_logU" "tag=a_tnoise1.25 abc=PFT_N01_logN_tnoise1.25" \
         "tag=a_rho.78 abc=PFT_N01_logN_rho_mu0.78" "tag=fam_mix family=mix" "tag=shape1 shape_sd=1e-4" "tag=shape2 shape_sd=2e-4" \
         "tag=k_skep k_mu=0.6 k_sd=0.3" "tag=k_pess k_mu=0.4 k_sd=0.4" "tag=k_opt k_mu=1.0 k_sd=0.1" "tag=rho0.6 rho=0.6" "tag=rho0.9 rho=0.9" \
         "tag=noise1.5 noise=1.5" "tag=noise0.7 noise=0.7" \
         "tag=o_skepS6 our=skepS6" "tag=o_strongS6 our=strongS6" "tag=o_tabReal our=tabReal" "tag=o_tabOver our=tabOver" \
         "tag=o_r6better our=r6better" "tag=o_noise1.5 our=noise1.5" "tag=o_bigKnobs our=bigKnobs" "tag=o_smallKnobs our=smallKnobs" \
         "tag=f_fieldBad field=fieldBad" "tag=f_fieldGood field=fieldGood" "tag=f_Wbig field=Wbig" \
         "tag=k_pess_noise1.5 k_mu=0.4 k_sd=0.4 noise=1.5" "tag=mix_k_pess family=mix k_mu=0.4 k_sd=0.4" "tag=mix_shape_o_r6better family=mix our=r6better" "tag=a_ALLruler abc=PFT_N01_logN_ALLruler"; do
  python picks2_mc.py $a >> sens2.log 2>&1
done
echo SENS2_DONE >> sens2.log
