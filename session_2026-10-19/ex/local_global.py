# local_global.py (10-20 ex, Claude). RESEARCH ONLY, local CPU. Global-factor removal candidate, local side:
#   (i)  gold58 c4 final (m14_gate rebuild, parity macro .94102): consistency only, gold58 is NOT a selection ruler;
#   (ii) R7 OOF (report-trained heliosli family) on gold58 and on the 16 extra gold74 studies, expert labels;
#   (iii) R7 OOF on train (minus gold74) vs report_labels_v2 YES/NO: the mechanism says decorrelation should LOSE there.
#   (iv) coupling summary from local_coupling.json: mean model rho vs mean expert tetrachoric over the 36 covered pairs.
# Transforms: same as box_global_template.py, plus the deployed 12-label forms (G12all, E1+N12all) that also move MCL,
# Contusion and Fracture (not scorable on OAI).
import json, itertools
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
R = 'C:/Users/eeshs/Downloads/rsna-knee'
G = R + '/session_2026-10-18/next/g58'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]
L9 = ["ACL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's"]
OA = ['Medial OA', 'Lateral OA', 'PF OA']
J9 = [L.index(c) for c in L9]; JOA = [L.index(c) for c in OA]
CORE = [L.index(c) for c in L9 if c not in ('ACL', 'Synovitis')]
pct = lambda v: rankdata(v) / len(v)


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def transforms(F):
    U = np.stack([pct(F[:, k]) for k in range(12)], 1)
    S9 = U[:, J9].sum(1); S12 = U.sum(1); out = {}

    def eoa(a):
        X = U.copy(); so = U[:, JOA].sum(1)
        for j in JOA: X[:, j] = U[:, j] - a * (so - U[:, j]) / 2
        return X
    for a in (.1, .2):
        out['EOA %.1f' % a] = eoa(a)
    for a in (-.1, .1, .2):
        X = U.copy()
        for j in J9: X[:, j] = U[:, j] - a * (S9 - U[:, j]) / 8
        out['G9 %+.1f' % a] = X
        X = U.copy()
        for j in J9: X[:, j] = U[:, j] - a * (S12 - U[:, j]) / 11
        out['G12 %+.1f' % a] = X
        X = U.copy()
        for j in range(12): X[:, j] = U[:, j] - a * (S12 - U[:, j]) / 11
        out['G12all %+.1f' % a] = X
    E1 = eoa(.1)
    for a in (-.1, .1, .2):
        X = E1.copy()
        for j in J9:
            if j not in JOA: X[:, j] = U[:, j] - a * (S9 - U[:, j]) / 8
        out['E1+N9 %+.1f' % a] = X
    for a in (-.1, .1, .2):
        X = E1.copy()
        for j in J9:
            if j not in JOA: X[:, j] = U[:, j] - a * (S12 - U[:, j]) / 11
        out['E1+N12 %+.1f' % a] = X
        X = E1.copy()
        for j in range(12):
            if j not in JOA: X[:, j] = U[:, j] - a * (S12 - U[:, j]) / 11
        out['E1+N12all %+.1f' % a] = X
    return U, out


def evaluate(F, Y, tag, B=2000, mincls=2):
    """F: n x 12 scores, Y: n x 12 labels with NaN where absent. Per-label AUC on present rows. macro = sum of deltas / 12."""
    U, TR = transforms(F); n = len(F)
    lab = [j for j in range(12) if np.isfinite(Y[:, j]).sum() > 0 and np.nansum(Y[:, j]) >= mincls and
           (np.isfinite(Y[:, j]).sum() - np.nansum(Y[:, j])) >= mincls]

    def deltas(ii):
        d0 = {}
        for j in lab:
            s = ii[np.isfinite(Y[ii, j])]; y = Y[s, j].astype(int)
            if y.sum() < 1 or y.sum() == len(y): return None
            d0[j] = (s, y, auc(U[s, j], y))
        out = {}
        for t, X in TR.items():
            d = np.zeros(12)
            for j, (s, y, b) in d0.items():
                if not np.allclose(X[:, j], U[:, j]): d[j] = auc(X[s, j], y) - b
            out[t] = d
        return out
    pt = deltas(np.arange(n))
    rng = np.random.default_rng(0); BS = {t: [] for t in TR}
    for _ in range(B):
        dd = deltas(rng.integers(0, n, n))
        if dd is None: continue
        for t in TR: BS[t].append(dd[t])
    base = np.mean([auc(U[np.isfinite(Y[:, j]), j], Y[np.isfinite(Y[:, j]), j].astype(int)) for j in lab])
    print('[%s] n %d, scored labels %d, base macro over scored %.5f, boots %d' % (tag, n, len(lab), base, len(BS['EOA 0.1'])))
    res = {}
    for t in TR:
        d = pt[t]; bs = np.array(BS[t]); be = np.array(BS['EOA 0.1'])
        m12 = d.sum() / 12; m9 = d[J9].sum() / 12; mc = d[CORE].sum() / 12
        inc = d.sum() / 12 - pt['EOA 0.1'].sum() / 12
        P = (bs.sum(1) > 0).mean(); Pinc = ((bs.sum(1) - be.sum(1)) > 0).mean()
        res[t] = dict(macro=m12, cov9=m9, core=mc, inc=inc, P=P, Pinc=Pinc, per=np.round(d, 4).tolist())
        print('   %-14s macro %+.5f (P %.2f) | covered-9 %+.5f core-7 %+.5f | vs EOA.1 %+.5f (P %.2f) | per %s' % (
            t, m12, P, m9, mc, inc, Pinc, np.round(d, 4).tolist()))
    return res


OUT = {}
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
g58 = g[g.source == 'original_gold58'].reset_index(drop=True); ids = g58.StudyInstanceUID.tolist(); n58 = len(ids)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4W = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
C4 = np.zeros((n58, 12))
for j, c in enumerate(L):
    w = C4W.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j])
    b = w * rr + (1 - w) * pct(R6[:, j]); C4[:, j] = .8 * pct(b) + .2 * pct(S6[:, j])
Y58 = g58[L].values.astype(float)
print('c4 rebuild gold58 macro %.5f (m14_gate.log .94102)' % np.mean([auc(C4[:, j], Y58[:, j].astype(int)) for j in range(12)]))
OUT['gold58_c4'] = evaluate(C4, Y58, 'gold58 c4 final (consistency only)')

r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
F7 = r7[['p|' + c for c in L]].set_axis(L, axis=1)
OUT['gold58_R7'] = evaluate(F7.reindex(ids).values, Y58, 'gold58 R7 OOF')
g74 = g.set_index('StudyInstanceUID'); i74 = g74.index.intersection(F7.dropna().index)
print('R7 OOF covers %d of gold74' % len(i74))
OUT['gold74_R7'] = evaluate(F7.loc[i74].values, g74.loc[i74, L].values.astype(float), 'gold74 R7 OOF (all with R7)')

F7t = F7[~F7.index.isin(g.StudyInstanceUID)].dropna()
rep = pd.read_csv(R + '/_labels/rsna-knee-llm-labels/report_labels_v2.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
cm = F7t.index.intersection(rep.index)
V = pd.DataFrame({c: np.where(rep.loc[cm, c + '__verdict'] == 'YES', 1.0, np.where(rep.loc[cm, c + '__verdict'] == 'NO', 0.0, np.nan)) for c in L}, index=cm)
OUT['train_R7_report'] = evaluate(F7t.loc[cm].values, V.values, 'train R7 OOF vs report v2 YES/NO (per label, UNK dropped)', B=300)

# (iv) coupling summary over the 36 covered pairs (from local_coupling.json, MEASURED earlier tonight)
lc = json.load(open(R + '/session_2026-10-19/ex/local_coupling.json'))['pairs']
oa = [k for k in lc if all(x.strip() in OA for x in k.split(' | '))]; oth = [k for k in lc if k not in oa]
for src, fn in (('gold74 label tet', lambda e: e['g74']['tet']), ('gold74 label phi', lambda e: e['g74']['phi']),
                ('gold58 label phi', lambda e: e['g58']['phi']), ('c4 gold58 rho', lambda e: e['g58_model_rho']['c4']),
                ('R7 OOF train rho', lambda e: e['r7_rho']), ('report v2 tet', lambda e: e['rep_dropUNK']['tet'])):
    v = {k: fn(e) for k, e in lc.items()}
    print('coupling %-18s mean 36 %+.3f | OA-trio 3 %+.3f | other 33 %+.3f' % (src, np.nanmean(list(v.values())),
          np.nanmean([v[k] for k in oa]), np.nanmean([v[k] for k in oth])))
gap = {k: e['g58_model_rho']['c4'] - e['g74']['tet'] for k, e in lc.items()}
print('c4 gap (rho - gold74 tet) other 33: mean %+.3f, #>0 %d, #<0 %d' % (np.mean([gap[k] for k in oth]),
      sum(gap[k] > 0 for k in oth), sum(gap[k] < 0 for k in oth)))
json.dump(OUT, open(R + '/session_2026-10-19/ex/local_global.json', 'w'), indent=1, default=float)
print('wrote local_global.json')
