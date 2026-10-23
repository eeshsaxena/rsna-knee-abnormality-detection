# skep_mcl.py (10-21 ~00:00 IST, Claude, STACK mini-skeptic on MCLmm). RESEARCH ONLY, local CPU, NO OAI. Aggregates only.
# Tries to break MCLmm (MCL = pct(pct(MCL) - .1 pct(MM)) on pw30) on the raked-OOF ruler:
#  (a) influence: leave one gold74 study out of the (MCL, MM) target table (74 refits), min / max / share > 0;
#  (b) gold-like mix: rake to the gold74 joint INCLUDING its marginals (MCL, MM, ACL, Contusion), i.e. an acute-enriched population;
#  (c) single members instead of the c4-shaped nest (R7, s1044, s1045 alone);
#  (d) placebo dose: MCL += .1 MM (the borrow, wrong direction) must lose;
#  (e) half-gold: target table from gold58 only vs from the 16 extra studies only.
import sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
import load as LD
src = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/stack_local.py', encoding='utf-8').read()
ns_ = {}; exec(src.split('summary = {}')[0].replace("log = open(OUT + '.log', 'w')", "log = open(OUT + '_skep_tmp.log', 'w')"), ns_)
pipeline, wauc, rake, BASE = ns_['pipeline'], ns_['wauc'], ns_['rake'], ns_['BASE']
log = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/skep_mcl.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
L = LD.L; J = {c: i for i, c in enumerate(L)}; G = LD.G; pct = lambda v: rankdata(v) / len(v)
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}); allg = set(g.StudyInstanceUID)
r7 = LD.r7(); s44, _ = LD.s6seed(1044); s45, _ = LD.s6seed(1045); lv2 = LD.labels('v2')
tr = [i for i in r7.index if i not in allg and i in lv2.index]; tr3 = [i for i in tr if i in s44.index and i in s45.index]
X = {'R7': r7.loc[tr3, ['p|' + c for c in L]].values.astype(float), 's1044': s44.loc[tr3, L].values.astype(float),
     's1045': s45.loc[tr3, L].values.astype(float)}
Y = lv2.loc[tr3, L].values.astype(float); YF = np.nan_to_num(Y, nan=0.); jm = J['MCL']
def table(G_, cols):
    Yg = G_[cols].values.astype(float); ok = ~np.isnan(Yg).any(1); T = np.zeros((2,) * len(cols))
    for r in Yg[ok].astype(int): T[tuple(r)] += 1
    return T
def wts(cols, m, Tp, to_train=True):
    jc = [J[c] for c in cols]; Ys = Y[m][:, jc].astype(int); obs = np.zeros((2,) * len(cols))
    for r in Ys: obs[tuple(r)] += 1
    obs /= obs.sum(); tgt = rake(Tp, Ys.mean(0), it=200) if to_train else Tp / Tp.sum(); w = (tgt / obs)[tuple(Ys.T)]
    return w / w.mean()
def dmcl(B0, B1, cols, T, to_train=True):
    m = ~np.isnan(Y[:, [J[x] for x in cols]]).any(1); w = wts(cols, m, T + .5, to_train)
    return (wauc(B1[m, jm], YF[m, jm], w) - wauc(B0[m, jm], YF[m, jm], w)) / 12 * 1e4
NESTS = {'N_A': ('R7', 's1044', 's1045'), 'N_B': ('s1044', 'R7', 's1045'), 'N_C': ('s1045', 's1044', 'R7')}
c1 = dict(BASE); c1['mclmm'] = .1; cneg = dict(BASE); cneg['mclmm'] = -.1
C2 = ['MCL', 'Medial Meniscus']
for nn, (ra, r6, s6) in NESTS.items():
    M = dict(rap=[(1., X[ra])], r6=X[r6], s6=X[s6]); B0 = pipeline(M, dict(BASE)); B1 = pipeline(M, c1); Bn = pipeline(M, cneg)
    full = dmcl(B0, B1, C2, table(g, C2))
    loo = np.array([dmcl(B0, B1, C2, table(g.drop(index=i), C2)) for i in g.index])
    P('%s | (MCL, MM) raking: full %+.2f | (a) LOO over 74 gold studies: min %+.2f max %+.2f, >0 in %d / 74' % (nn, full, loo.min(), loo.max(), (loo > 0).sum()))
    gl = dmcl(B0, B1, C2 + ['ACL', 'Contusion'], table(g, C2 + ['ACL', 'Contusion']), to_train=False)
    gl2 = dmcl(B0, B1, C2, table(g, C2), to_train=False)
    P('   (b) gold-like mix (gold74 joint incl. its marginals): (MCL, MM) %+.2f | (MCL, MM, ACL, Cont) %+.2f' % (gl2, gl))
    P('   (d) wrong-direction borrow MCL += .1 MM: %+.2f (must be < 0)' % dmcl(B0, Bn, C2, table(g, C2)))
    P('   (e) target from gold58 only %+.2f | from the 16 extra only %+.2f' % (dmcl(B0, B1, C2, table(g[g.source == 'original_gold58'], C2)),
                                                                           dmcl(B0, B1, C2, table(g[g.source != 'original_gold58'], C2))))
for k in X:
    M = dict(rap=[(1., X[k])], r6=X[k], s6=X[k]); B0 = pipeline(M, dict(BASE)); B1 = pipeline(M, c1)
    P('(c) single member %-6s (MCL, MM) raked %+.2f | raw report %+.2f' % (k, dmcl(B0, B1, C2, table(g, C2)),
      (lambda m: (wauc(B1[m, jm], YF[m, jm], np.ones(m.sum())) - wauc(B0[m, jm], YF[m, jm], np.ones(m.sum()))) / 12 * 1e4)(~np.isnan(Y[:, [jm, J['Medial Meniscus']]]).any(1))))
P('SKEP_MCL DONE')
