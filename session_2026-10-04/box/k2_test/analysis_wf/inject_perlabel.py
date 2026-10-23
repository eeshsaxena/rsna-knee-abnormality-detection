"""Why does a raptor_public gain not survive into raptor_coat_side? Per-label view + bootstrap CI of the pub delta."""
import numpy as np, pandas as pd
from inject_common import *
D = load_all(); Y = D['Y']; F = D['fleet']; boot = Boot(Y, B=2000, seed=0)
arms = list(D['raptor_raw']); cps = [D['coat_prob'][m] for m in COAT_MEMBERS]
pub0 = raptor_public(arms, RAPTOR_W); fam0 = coat_family(cps); side0 = hybrid(pub0, fam0)
pb0 = np.tensordot(RAPTOR_W, np.stack([np.clip(a, 0, 1) for a in arms]), axes=(0, 0))
pubR = rank01(.8 * rank_pct(pb0) + .2 * rank_pct(F)).astype(np.float32).astype(np.float64)   # 5th arm rank f=.20
sideR = hybrid(pubR, fam0)
lo, hi, p = boot.delta(pubR, pub0); print(f'5th arm rank f=.20: dPub {mauc(Y, pubR) - mauc(Y, pub0):+.4f} CI90 [{lo:+.4f},{hi:+.4f}] P>0 {p:.2f}')
lo, hi, p = boot.delta(sideR, side0); print(f'                    dSide {mauc(Y, sideR) - mauc(Y, side0):+.4f} CI90 [{lo:+.4f},{hi:+.4f}] P>0 {p:.2f}')
pubP = raptor_public(arms, RAPTOR_W, F, .25 / .75)
lo, hi, p = boot.delta(pubP, pub0); print(f'5th arm prob f=.25: dPub {mauc(Y, pubP) - mauc(Y, pub0):+.4f} CI90 [{lo:+.4f},{hi:+.4f}] P>0 {p:.2f}')
from scipy.stats import spearmanr
tab = pd.DataFrame({'fleet': auc_cols(Y, F), 'pub': auc_cols(Y, pub0), 'fam': auc_cols(Y, fam0), 'side': auc_cols(Y, side0),
                    'dPub_rk20': auc_cols(Y, pubR) - auc_cols(Y, pub0), 'dSide_rk20': auc_cols(Y, sideR) - auc_cols(Y, side0),
                    'rho_F_pub': [spearmanr(F[:, j], pub0[:, j]).correlation for j in range(12)],
                    'rho_F_fam': [spearmanr(F[:, j], fam0[:, j]).correlation for j in range(12)],
                    'rho_pub_fam': [spearmanr(pub0[:, j], fam0[:, j]).correlation for j in range(12)],
                    'outer_w': [COATNET_W[t] for t in T]}, index=T)
print(tab.round(4).to_string()); print('mean'); print(tab.mean().round(4).to_string())
# blend of fam with (pub + fleet) vs fam with pub: does the fleet add anything beyond pub+fam? fleet residual vs side
print('rho(fleet, side) mean', round(rho_mean(F, side0), 3), ' rho(pub, fam) mean', round(rho_mean(pub0, fam0), 3))
