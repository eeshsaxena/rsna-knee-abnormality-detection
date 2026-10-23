# skp_syn_mc.py (10-20 night, Claude, SKEPTIC). MODEL only, no data. E2s probe value under R1 with skeptic-corrected inputs
# (macro e-4). Same structure as syn_slate_mc.py: adopt E2s iff its public paired read > 0 (alone) or > max(0, pw25 read) (slate).
# pw25 - pw30 true ~ N(+0.3, 0.8), read sd 1.4 (CLAIMED implied2 via syn_slate_mc.py). E2s read sd 2.9 (MEASURED syn_more.log).
import numpy as np
rng = np.random.default_rng(17); N = 2000000
SC = {  # name: p_real, (m_real, s_real), (m_null, s_null)
    'proposer (repro)': (.50, (2.8, .6 * 2.8 + .4), (-2.0, .6 * 2.8 + .4)),
    'skeptic base': (.30, (1.0, 0.7), (-0.8, 0.8)),
    'skeptic optimistic': (.45, (1.5, 1.0), (-0.5, 0.8)),
    'skeptic pessimistic': (.20, (0.8, 0.6), (-1.0, 1.0)),
}
for nm, (p, (mr, sr), (mn, sn)) in SC.items():
    real = rng.random(N) < p
    t = np.where(real, rng.normal(mr, sr, N), rng.normal(mn, sn, N)); pub = t + rng.normal(0, 2.9, N)
    t25 = rng.normal(.3, .8, N); p25 = t25 + rng.normal(0, 1.4, N)
    adopt = pub > 0; adopt_s = (pub > 0) & (pub > p25)
    slate = np.where(adopt_s, t, np.where(p25 > 0, t25, 0)); only25 = np.where(p25 > 0, t25, 0)
    print('%-20s E[true] %+.2f P(true>0) %.2f | alone: P(move) %.2f E[gain probe+R1] %+.2f P(adopted & harmful) %.2f E[true | adopted] %+.2f | '
          'slate w/ pw25: P(E2s becomes A) %.2f, slate %+.2f vs pw25 alone %+.2f (increment %+.2f)' % (
          nm, t.mean(), (t > 0).mean(), adopt.mean(), np.where(adopt, t, 0).mean(), (adopt & (t < 0)).mean(), t[adopt].mean(),
          adopt_s.mean(), slate.mean(), only25.mean(), slate.mean() - only25.mean()))
