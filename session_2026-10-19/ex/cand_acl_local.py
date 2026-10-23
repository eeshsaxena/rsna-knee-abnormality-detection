# cand_acl_local.py (10-20 ex, Claude). RESEARCH ONLY, local CPU, aggregates printed only, writes nothing but its own log
# (redirected by the caller). Candidate "acl": ACL decorrelated from the menisci / Effusion,
#   out_ACL = pct(pct(f_ACL) - a * mean(pct(f_o) for o in group)), a in {-.1, .1, .2}.
# Groups: LM (the only ACL pair with consistent excess coupling in coupling_map.md), MM (mixed), MM+LM, Eff (deficit, rule
# says do NOT decorrelate; reference only), MM+LM+Eff (the literal task group).
# Reads here (consistency only, gold58 is NOT a selection ruler):
#   (i)   gold58 c4 final (m14_gate rebuild, parity macro .94102) + its members R6 / S6fall / Raptor
#   (ii)  R7 OOF (report-trained) on gold74 and gold58 expert labels
#   (iii) R7 OOF on train (minus gold74) vs report_labels_v2 (mechanism: decorrelation should LOSE there)
#   (iv)  gold74 label coupling for the ACL pairs (2x2 counts, tetrachoric) + the 4-label joint cells for the box R1 scheme
import json, itertools
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr, norm

R = 'C:/Users/eeshs/Downloads/rsna-knee'
G = R + '/session_2026-10-18/next/g58'
E = R + '/session_2026-10-19/ex/'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]
GROUPS = {'LM': ['Lateral Meniscus'], 'MM': ['Medial Meniscus'], 'MM+LM': ['Medial Meniscus', 'Lateral Meniscus'],
          'Eff': ['Effusion'], 'MM+LM+Eff': ['Medial Meniscus', 'Lateral Meniscus', 'Effusion']}
AS = (-.1, .1, .2)
pct = lambda v: rankdata(v) / len(v)
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def tr(F, grp, a):
    """F: dict label -> score array. Returns the transformed ACL score."""
    return pct(pct(F['ACL']) - a * np.mean([pct(F[o]) for o in grp], 0))


def evaluate(F, y, tag, B=2000, seed=0):
    n = len(y); base = auc(F['ACL'], y); rng = np.random.default_rng(seed)
    outs = {(g, a): tr(F, grp, a) for g, grp in GROUPS.items() for a in AS}
    pt = {k: auc(v, y) - base for k, v in outs.items()}
    bd = {k: [] for k in outs}
    for _ in range(B):
        ii = rng.integers(0, n, n); yy = y[ii]
        if yy.sum() < 2 or yy.sum() > n - 2: continue
        b0 = auc(F['ACL'][ii], yy)
        for k, v in outs.items(): bd[k].append(auc(v[ii], yy) - b0)
    zt = ns(F['ACL']); ct = np.corrcoef(zt, y)[0, 1]
    print('[%s] n %d ACL prev %.3f base ACL AUC %.4f' % (tag, n, y.mean(), base))
    res = {}
    for g, grp in GROUPS.items():
        zo = ns(np.mean([pct(F[o]) for o in grp], 0)); rho = np.corrcoef(zt, zo)[0, 1]; q = np.corrcoef(zo, y)[0, 1] / ct
        s = ' | '.join('a %+.1f d %+.4f (macro %+.5f, P>0 %.2f)' % (a, pt[(g, a)], pt[(g, a)] / 12, np.mean(np.array(bd[(g, a)]) > 0))
                       for a in AS)
        print('   ACL <- %-10s rho_ns %+.3f q %+.3f (theory gain iff rho > q: %s) | %s' % (g, rho, q, 'yes' if rho > q else 'no', s))
        res[g] = {str(a): [float(pt[(g, a)]), float(np.mean(np.array(bd[(g, a)]) > 0))] for a in AS}
        res[g]['rho'] = float(rho); res[g]['q'] = float(q)
    return res


OUT = {}
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
g58 = g[g.source == 'original_gold58'].reset_index(drop=True); ids = g58.StudyInstanceUID.tolist(); n58 = len(ids)

# ---------------- (iv) gold74 label coupling ----------------
def tetra_counts(c):
    from scipy.stats import multivariate_normal
    from scipy.optimize import brentq
    c00, c01, c10, c11 = [x + .5 for x in c]; N = c00 + c01 + c10 + c11
    h = norm.ppf(1 - (c11 + c10) / N); k = norm.ppf(1 - (c11 + c01) / N); p11 = c11 / N
    f = lambda r: multivariate_normal.cdf([-h, -k], mean=[0, 0], cov=[[1, r], [r, 1]]) - p11
    if f(-.995) > 0: return -.995
    if f(.995) < 0: return .995
    return brentq(f, -.995, .995, xtol=1e-4)


L7 = ['ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Effusion', 'Medial OA', 'Lateral OA', 'PF OA']
G74P = {}
print('gold74 pairwise label coupling (counts c00, c01, c10, c11; a first)')
for a, b in itertools.combinations(L7, 2):
    d = g[[a, b]].dropna().astype(int)
    c = [int(((d[a] == i) & (d[b] == j)).sum()) for i in (0, 1) for j in (0, 1)]
    G74P[a + ' | ' + b] = c
    if a == 'ACL':
        cs = np.array(c, float) + .5; lor = np.log(cs[3] * cs[0] / (cs[1] * cs[2])); lsd = np.sqrt((1 / cs).sum())
        print('   %-30s n %d counts %s tet %+.2f log OR %+.2f (sd %.2f)' % (a + ' | ' + b, sum(c), c, tetra_counts(c), lor, lsd))
d4 = g[['ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Effusion']].dropna().astype(int)
k4 = (d4.values * (2 ** np.arange(4))).sum(1)
G74J4 = np.bincount(k4, minlength=16).tolist()
print('gold74 4-label joint (ACL, MM, LM, Eff; bit0 = ACL) n %d cells %s' % (len(d4), G74J4))
OUT['G74P'] = G74P; OUT['G74J4'] = G74J4
print('gold74 ACL prevalence %.3f (n %d); gold58 ACL prevalence %.3f' % (g.ACL.mean(), g.ACL.notna().sum(), g58.ACL.mean()))

# ---------------- (i) gold58 c4 + members ----------------
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4W = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
C4 = np.zeros((n58, 12)); RAP = np.zeros((n58, 12))
for j, c in enumerate(L):
    w = C4W.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); RAP[:, j] = rr
    b = w * rr + (1 - w) * pct(R6[:, j]); C4[:, j] = .8 * pct(b) + .2 * pct(S6[:, j])
Y58 = g58[L].values.astype(int)
print('c4 rebuild gold58 macro %.5f (m14_gate.log: .94102)' % np.mean([auc(C4[:, j], Y58[:, j]) for j in range(12)]))
y58 = g58['ACL'].values.astype(int)
OUT['g58'] = {}
for nm, M in (('c4', C4), ('R6', R6), ('S6fall', S6), ('Raptor', RAP)):
    F = {c: M[:, L.index(c)] for c in ['ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Effusion']}
    OUT['g58'][nm] = evaluate(F, y58, 'gold58 %s' % nm, B=2000 if nm == 'c4' else 1000)
# c4 with the menisci binarised at expert labels: does the gain (if any) come from the meniscus LABEL part or the score noise?
# (diagnostic) q from labels: corr(y_LM, y_ACL) on gold58
print('gold58 label phi ACL-MM %+.3f ACL-LM %+.3f ACL-Eff %+.3f' % tuple(
    np.corrcoef(y58, g58[c].values.astype(float))[0, 1] for c in ['Medial Meniscus', 'Lateral Meniscus', 'Effusion']))

# ---------------- (ii) R7 OOF on gold74 / gold58 expert labels ----------------
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
F7all = r7[['p|' + c for c in L]].set_axis(L, axis=1)
for tag, gg in (('gold74', g), ('gold58', g58)):
    d = gg.set_index('StudyInstanceUID')[['ACL']].dropna().join(F7all[['ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Effusion']],
                                                                 rsuffix='_f').dropna()
    F = {'ACL': d['ACL_f'].values, 'Medial Meniscus': d['Medial Meniscus'].values, 'Lateral Meniscus': d['Lateral Meniscus'].values,
         'Effusion': d['Effusion'].values}
    OUT['r7_' + tag] = evaluate(F, d['ACL'].values.astype(int), 'R7 OOF on %s expert' % tag, B=1000)

# ---------------- (iii) R7 OOF on train vs report labels ----------------
r7t = F7all[~F7all.index.isin(g.StudyInstanceUID)]
rep = pd.read_csv(R + '/_labels/rsna-knee-llm-labels/report_labels_v2.csv', dtype={'StudyInstanceUID': str}).drop_duplicates(
    'StudyInstanceUID').set_index('StudyInstanceUID')
cm = r7t.index.intersection(rep.index)
v = rep.loc[cm, 'ACL__verdict']
for tag, yv in (('report YES/NO (UNK dropped)', np.where(v == 'YES', 1., np.where(v == 'NO', 0., np.nan))),
                ('report UNK=0', np.where(v == 'YES', 1., 0.))):
    ok = ~np.isnan(yv); d = r7t.loc[cm][ok]
    F = {c: d[c].values for c in ['ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Effusion']}
    OUT['r7_rep_' + tag.split()[1]] = evaluate(F, yv[ok].astype(int), 'R7 OOF train vs %s' % tag, B=300)
# report-label coupling for the ACL pairs (for the inheritance read)
for c in ['Medial Meniscus', 'Lateral Meniscus', 'Effusion']:
    vv = rep.loc[cm, [c + '__verdict']].join(rep.loc[cm, ['ACL__verdict']])
    m = vv.isin(['YES', 'NO']).all(1); a = (vv.loc[m, 'ACL__verdict'] == 'YES').astype(int).values; b = (vv.loc[m, c + '__verdict'] == 'YES').astype(int).values
    cc = [int(((a == i) & (b == j)).sum()) for i in (0, 1) for j in (0, 1)]
    print('report v2 ACL | %-18s n %d tet %+.2f | R7 train rho %+.2f' % (c, m.sum(), tetra_counts(cc), spearmanr(r7t.loc[cm, 'ACL'], r7t.loc[cm, c]).correlation))
json.dump(OUT, open(E + 'cand_acl_local.json', 'w'), indent=1)
print('wrote cand_acl_local.json (aggregates only)')
