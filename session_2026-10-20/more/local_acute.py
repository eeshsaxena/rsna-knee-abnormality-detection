# local_acute.py (10-20 more, Claude). RESEARCH ONLY, local CPU, existing local files, prints aggregates only.
# Hypothesis written from the rubric text (forum 733343) after seeing the part C coupling table of local_sev.log (so it is
# POST HOC on gold58; this script only sizes it and runs the controls that do not reuse gold58 for selection):
#   "ACUTE" labels: the rubric keeps only acute findings ("high-grade partial or complete ACUTE tear ... low-grade sprains and
#   chronic or remote stress changes are graded negative" for MCL; "an ACUTE cortical break or fracture line" for Fracture).
#   Reports (and report-trained heads) also call chronic / degenerative variants, so the heads inherit a positive coupling with
#   degenerative findings that experts do not have (expert tet MCL-MM -.35/-.31, Fracture-MOA -.39/-.40, gold58/gold74).
#   Fix: out_t = pct(f_t) - a * mean(pct(f_h) for h in DEG) for t in ACUTE, DEG = rubric degenerative set
#   {Medial Meniscus, Medial OA, Lateral OA, PF OA, Baker's}. a in {.1, .2}. Other 10 columns untouched.
# Rulers:
#   gold58 c4 (contaminated: the pair table was read first) -> size, bootstrap, split-half, random-helper-set null;
#   report OOF (R7, s1044, s1045 vs v2 / v4) -> NEGATIVE CONTROL, the mechanism predicts a loss;
#   theory: q (cross-predictiveness of the factor for the expert label) vs rho (model coupling) on gold58.
import sys, itertools
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, pct, gold, G, r7, s6seed, labels
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
ACUTE = ['MCL', 'Fracture']; DEG = ['Medial Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', "Baker's"]
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int); n = len(ids)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
F = np.zeros((n, 12))
for j, c in enumerate(L):
    w = C4.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); b = w * rr + (1 - w) * pct(R6[:, j])
    F[:, j] = .8 * pct(b) + .2 * pct(S6[:, j])
print('c4 gold58 macro %.5f (parity target .94102)' % np.mean([auc(F[:, j], Y[:, j]) for j in range(12)]))


def transform(F, a, targets=ACUTE, helpers=DEG):
    O = F.copy()
    for t in targets:
        jt = L.index(t); hs = [L.index(h) for h in helpers if h != t]
        D = np.mean([pct(F[:, h]) for h in hs], 0)
        O[:, jt] = pct(pct(F[:, jt]) - a * D)
    return O


def macro_delta(F, O, Y, rows=None):
    if rows is None: rows = np.arange(len(Y))
    return np.mean([auc(O[rows, j], Y[rows, j]) - auc(F[rows, j], Y[rows, j]) for j in range(12)])


rng = np.random.default_rng(1020)
for a in (.1, .2):
    O = transform(F, a); d = macro_delta(F, O, Y)
    per = {t: auc(O[:, L.index(t)], Y[:, L.index(t)]) - auc(F[:, L.index(t)], Y[:, L.index(t)]) for t in ACUTE}
    bs = []
    for _ in range(2000):
        ii = rng.integers(0, n, n)
        if any(Y[ii, L.index(t)].min() == Y[ii, L.index(t)].max() for t in ACUTE): continue
        bs.append(np.mean([auc(O[ii, L.index(t)], Y[ii, L.index(t)]) - auc(F[ii, L.index(t)], Y[ii, L.index(t)]) for t in ACUTE]) * 2 / 12)
    bs = np.array(bs)
    sh = []
    for _ in range(1000):
        p = rng.permutation(n); A, B = p[:n // 2], p[n // 2:]
        if any(Y[A, L.index(t)].min() == Y[A, L.index(t)].max() or Y[B, L.index(t)].min() == Y[B, L.index(t)].max() for t in ACUTE): continue
        sh.append((macro_delta(F, O, Y, A) > 0) == (macro_delta(F, O, Y, B) > 0))
    print('gold58 c4 ACUTE<-DEG a %.1f: macro %+.5f | per label %s | boot sd %.5f P>0 %.3f | split-half same sign %.2f' % (
        a, d, {k: round(v, 4) for k, v in per.items()}, bs.std(), (bs > 0).mean(), np.mean(sh)))
    # each acute label alone
    for t in ACUTE:
        O1 = transform(F, a, [t]); print('    %s alone: macro %+.5f' % (t, macro_delta(F, O1, Y)))

# random helper-set null: 5-label helper sets drawn from the 10 non-acute labels (DEG is one of C(10,5)=252)
others = [c for c in L if c not in ACUTE]
null = []
for hs in itertools.combinations(others, 5):
    null.append((set(hs) == set(DEG), macro_delta(F, transform(F, .1, ACUTE, list(hs)), Y)))
vals = np.array([v for _, v in null]); dv = [v for k, v in null if k][0]
print('random-helper null (252 five-label sets, a .1): DEG %+.5f rank %d of %d (null mean %+.5f sd %.5f, 90th pct %+.5f)' % (
    dv, int((vals > dv).sum()) + 1, len(vals), vals.mean(), vals.std(), np.quantile(vals, .9)))
# per helper alone
for h in others:
    print('    single helper %-17s a .1: MCL %+.4f Fracture %+.4f' % (
        h, *[auc(transform(F, .1, [t], [h])[:, L.index(t)], Y[:, L.index(t)]) - auc(F[:, L.index(t)], Y[:, L.index(t)]) for t in ACUTE]))

# theory on gold58: rho = corr(ns f_t, ns D), q = corr(ns D, Y_t) / corr(ns f_t, Y_t)
for t in ACUTE:
    jt = L.index(t); D = np.mean([pct(F[:, L.index(h)]) for h in DEG], 0)
    zt, zd = ns(F[:, jt]), ns(D); y = Y[:, jt]
    rho = np.corrcoef(zt, zd)[0, 1]; q = np.corrcoef(zd, y)[0, 1] / np.corrcoef(zt, y)[0, 1]
    qb = []
    for _ in range(2000):
        ii = rng.integers(0, n, n)
        if y[ii].min() == y[ii].max(): continue
        qb.append(np.corrcoef(zd[ii], y[ii])[0, 1] / np.corrcoef(zt[ii], y[ii])[0, 1])
    qb = np.array(qb)
    print('theory %s: rho(f_t, D) %.3f, q %.3f (boot sd %.3f), P(q < rho) %.3f, positives %d' % (t, rho, q, qb.std(), (qb < rho).mean(), y.sum()))

# NEGATIVE CONTROL: report OOF
d7 = r7(); s44, _ = s6seed(1044); s45, _ = s6seed(1045); goldids = set(g.StudyInstanceUID)
for src in ('v2', 'v4'):
    lab = labels(src)
    tr = [i for i in s44.index if i in d7.index and i in s45.index and i in lab.index and i not in goldids]
    Yr = lab.loc[tr, L].values.astype(float)
    for nm, M in (('R7', d7.loc[tr, ['p|' + c for c in L]].values), ('s1044', s44.loc[tr, L].values), ('s1045', s45.loc[tr, L].values),
                  ('mean3', np.column_stack([np.mean([pct(X[:, j]) for X in (d7.loc[tr, ['p|' + c for c in L]].values, s44.loc[tr, L].values, s45.loc[tr, L].values)], 0) for j in range(12)]))):
        out = []
        for a in (.1, .2):
            O = transform(M, a); dd = []
            for t in ACUTE:
                jt = L.index(t); ok = ~np.isnan(Yr[:, jt]); dd.append(auc(O[ok, jt], Yr[ok, jt].astype(int)) - auc(M[ok, jt], Yr[ok, jt].astype(int)))
            out.append('a %.1f MCL %+.4f Frac %+.4f (macro %+.5f)' % (a, dd[0], dd[1], sum(dd) / 12))
        print('report OOF %s %-6s n %d | %s' % (src, nm, len(tr), ' | '.join(out)))
