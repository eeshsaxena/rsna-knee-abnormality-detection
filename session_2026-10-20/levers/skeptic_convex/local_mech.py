# skeptic_convex/local_mech.py (10-20 skeptic of convex E-OA). Competition data only.
# Re-derive the "report coupling is convex" mechanism numbers and put error bars on them.
# C = [P(Y_t|k=2) - P(Y_t|k=1)] - [P(Y_t|k=1) - P(Y_t|k=0)], k = other compartments positive under the SAME labeller.
import numpy as np, pandas as pd
D = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58/'
LBL = 'C:/Users/eeshs/Downloads/rsna-knee/_labels/rsna-knee-llm-labels/report_labels_v2.csv'
OA = ['Medial OA', 'Lateral OA', 'PF OA']
rng = np.random.default_rng(11)
lg = lambda p: np.log(p / (1 - p))


def cond(Y):
    k = np.stack([Y[:, [o for o in range(3) if o != t]].sum(1) for t in range(3)], 1)
    return np.array([Y[k == kk].mean() if (k == kk).any() else np.nan for kk in range(3)])


def C(Y):
    p = cond(Y); return (p[2] - p[1]) - (p[1] - p[0])


g = pd.read_csv(D + 'gold74_eval_set.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
RL = pd.read_csv(LBL, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
idx = g.index.intersection(RL.index)
E = g.loc[idx, OA].astype(float); R = RL.loc[idx, OA].astype(float); src = g.loc[idx, 'source']
ok = E.notna().all(axis=1) & R.notna().all(axis=1)
E = E[ok].values.astype(int); R = (R[ok].values >= .5).astype(int); src = src[ok].values
print('gold74 matched n %d (gold58 part %d, extras %d)' % (len(E), (src == 'original_gold58').sum(), (src != 'original_gold58').sum()))
for nm, Yx in (('expert', E), ('report', R)):
    p = cond(Yx)
    print('  %-6s P(Y|k=0,1,2) %s | logit steps %s | C %+.3f | 111 knees %d | positives %d' % (
        nm, np.round(p, 3).tolist(), np.round(np.diff(lg(np.clip(p, .01, .99))), 2).tolist(), C(Yx), int((Yx.sum(1) == 3).sum()), int(Yx.sum())))
# gold58-only (pure expert image reads)
m58 = src == 'original_gold58'
for nm, Yx in (('expert58', E[m58]), ('report58', R[m58])):
    p = cond(Yx); print('  %-8s P(Y|k) %s | C %+.3f | 111 %d' % (nm, np.round(p, 3).tolist(), C(Yx), int((Yx.sum(1) == 3).sum())))
# knee bootstrap of C(report) - C(expert) and of the k=2 / k=1 gaps
dC, d2, d1 = [], [], []
for b in range(5000):
    ix = rng.integers(0, len(E), len(E))
    pe, pr = cond(E[ix]), cond(R[ix])
    if np.isnan(pe).any() or np.isnan(pr).any():
        continue
    dC.append(((pr[2] - pr[1]) - (pr[1] - pr[0])) - ((pe[2] - pe[1]) - (pe[1] - pe[0]))); d2.append(pr[2] - pe[2]); d1.append(pr[1] - pe[1])
dC, d2, d1 = map(np.array, (dC, d2, d1))
print('  knee boot (B %d): C(report) - C(expert) %+.3f (sd %.3f, P>0 %.3f) | k2 report - expert %+.3f (sd %.3f) | k1 report - expert %+.3f (sd %.3f, P<0 %.2f)' % (
    len(dC), dC.mean(), dC.std(), (dC > 0).mean(), d2.mean(), d2.std(), d1.mean(), d1.std(), (d1 < 0).mean()))
# expert-only bootstrap of P(k=2) and C
ce, pe2 = [], []
for b in range(5000):
    ix = rng.integers(0, len(E), len(E)); p = cond(E[ix])
    if np.isnan(p).any(): continue
    ce.append((p[2] - p[1]) - (p[1] - p[0])); pe2.append(p[2])
print('  expert C %+.3f (sd %.3f) | expert P(k=2) %.3f (sd %.3f)' % (np.mean(ce), np.std(ce), np.mean(pe2), np.std(pe2)))

# train report labels (all with labels, minus gold74)
RT = RL.drop(index=RL.index.intersection(g.index))[OA].astype(float).dropna()
YT = (RT.values >= .5).astype(int)
p = cond(YT)
print('train report n %d | P(Y|k) %s | logit steps %s | C %+.3f | 111 share %.3f' % (
    len(YT), np.round(p, 3).tolist(), np.round(np.diff(lg(p)), 2).tolist(), C(YT), (YT.sum(1) == 3).mean()))
# how typical is the gold74 report C (= .53) for a 70-knee draw from the train report population?
cs = []
for b in range(5000):
    ix = rng.integers(0, len(YT), len(E)); cs.append(C(YT[ix]))
cs = np.array(cs)
print('  70-knee draws of train report labels: C mean %+.3f sd %.3f | P(C >= gold74 report C %.3f) = %.3f' % (cs.mean(), cs.std(), C(R), (cs >= C(R)).mean()))
# prevalence-matched comparison: expert gold74 vs train report, conditional excess by k
pe = cond(E)
print('  excess train-report minus gold74-expert by k: %s (expert sd at k2 ~%.3f)' % (np.round(p - pe, 3).tolist(), np.std(pe2)))
