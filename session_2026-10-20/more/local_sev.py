# local_sev.py (10-20 more, Claude). RESEARCH ONLY, local CPU, reads existing local files, writes only its own log via stdout.
# Part A (negative control, report regime): the same AND / OR member combiners as box_sev.py on train OOF members
#   (R7 OOF p, our S6 seeds s1044, s1045; equal weights) vs report labels (v2 YES/NO, v4 >= .5), per label, paired bootstrap.
# Part B (expert regime, c4 members, gold58; consistency / code check only, NOT a selection ruler): c4's three families
#   (Raptor mix, R6, S6fall) with c4's effective per-label weights, flat form, AND / OR combiners vs the flat rank base.
# Part C (coupling for the 3 labels the 10-19 map did not cover: MCL, Contusion, Fracture): expert tetrachoric (gold58,
#   gold74), c4 Spearman (gold58), R7 Spearman (train), report v2 tetrachoric (train), and the directional decorrelation
#   out_t = pct(f_t) - a pct(f_o) at a .1 on gold58 c4 and on R7 vs report v2 (the report-inheritance mechanism predicts a
#   loss on report labels and a gain on expert labels only where c4 couples more than experts).
import sys, itertools
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr, norm, multivariate_normal
from scipy.optimize import brentq
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, pct, gold, G, r7, s6seed, labels

OPS = ['pow2', 'invgeo', 'msd+', 'geo', 'p05', 'msd-']


def comb(op, U, w):
    w = np.asarray(w, float)[:, None]; w = w / w.sum()
    m = (w * U).sum(0)
    if op == 'base': return m
    if op == 'pow2': return (w * U ** 2).sum(0)
    if op == 'invgeo': return -(w * np.log(1 - U + .5 / U.shape[1])).sum(0)
    sd = np.sqrt((w * (U - m) ** 2).sum(0))
    if op == 'msd+': return m + .5 * sd
    if op == 'msd-': return m - .5 * sd
    if op == 'geo': return (w * np.log(U)).sum(0)
    if op == 'p05': return (w * np.sqrt(U)).sum(0)
    raise ValueError(op)


STRICT = ['ACL', 'MCL', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', "Baker's", 'Contusion']   # m1: expert stricter
BROAD = ['Synovitis']


def run(name, Ms, Y, w, B=300, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for j, c in enumerate(L):
        y = Y[:, j]; ok = ~np.isnan(y); yy = y[ok].astype(int); n = len(yy)
        U = np.stack([pct(M[ok, j]) for M in Ms]); ww = w[c] if isinstance(w, dict) else w
        S0 = comb('base', U, ww); a0 = auc(S0, yy); outs = {o: comb(o, U, ww) for o in OPS}
        d = {o: auc(outs[o], yy) - a0 for o in OPS}
        bd = {o: [] for o in OPS}
        for _ in range(B):
            ii = rng.integers(0, n, n); yb = yy[ii]
            if yb.sum() < 2 or yb.sum() > n - 2: continue
            b0 = auc(S0[ii], yb)
            for o in OPS: bd[o].append(auc(outs[o][ii], yb) - b0)
        rows.append(dict(label=c, n=n, prev=yy.mean(), base=a0, mdis=float(np.sqrt(((U - U.mean(0)) ** 2).mean(0)).mean()),
                         **{'d_' + o: d[o] for o in OPS}, **{'P_' + o: float(np.mean(np.array(bd[o]) > 0)) for o in OPS}))
    R = pd.DataFrame(rows).set_index('label')
    R['dAND'] = R[['d_geo', 'd_p05', 'd_msd-']].mean(1); R['dOR'] = R[['d_pow2', 'd_invgeo', 'd_msd+']].mean(1)
    print('\n[%s]' % name)
    print(R[['n', 'prev', 'base', 'mdis', 'dAND', 'dOR', 'd_geo', 'P_geo', 'd_msd-', 'P_msd-', 'd_pow2', 'P_pow2']].round(4).to_string())
    print('  macro (sum/12): dAND %+.5f dOR %+.5f | STRICT-label mean dAND %+.4f dOR %+.4f | Synovitis dAND %+.4f dOR %+.4f' % (
        R.dAND.sum() / 12, R.dOR.sum() / 12, R.loc[STRICT, 'dAND'].mean(), R.loc[STRICT, 'dOR'].mean(), R.loc['Synovitis', 'dAND'],
        R.loc['Synovitis', 'dOR']))
    return R


# ---------------- Part A: report-regime OOF ----------------
d7 = r7(); s44, _ = s6seed(1044); s45, _ = s6seed(1045)
g, g58 = gold(); goldids = set(g.StudyInstanceUID)
for src in ('v2', 'v4'):
    lab = labels(src)
    ids = [i for i in s44.index if i in d7.index and i in s45.index and i in lab.index and i not in goldids]
    Ms = [d7.loc[ids, ['p|' + c for c in L]].values, s44.loc[ids, L].values, s45.loc[ids, L].values]
    run('A report OOF %s: R7 + s1044 + s1045, equal weights, n %d' % (src, len(ids)), Ms, lab.loc[ids, L].values.astype(float), [1, 1, 1])

# ---------------- Part B: gold58 c4 families ----------------
ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(float)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
RR = np.column_stack([.75 * pct(R384[:, j]) + .25 * pct(R224[:, j]) for j in range(12)])
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
W = {c: [.8 * C4.get(c, .5), .8 * (1 - C4.get(c, .5)), .2] for c in L}
RB = run('B gold58 c4 families (Raptor mix, R6, S6fall), c4 effective weights, flat', [RR, R6, S6], Y, W, B=1000)

# ---------------- Part C: coupling for MCL / Contusion / Fracture ----------------
def tetra(a, b):
    a = np.asarray(a, int); b = np.asarray(b, int)
    c11 = ((a == 1) & (b == 1)).sum() + .5; c10 = ((a == 1) & (b == 0)).sum() + .5
    c01 = ((a == 0) & (b == 1)).sum() + .5; c00 = ((a == 0) & (b == 0)).sum() + .5; N = c11 + c10 + c01 + c00
    h = norm.ppf(1 - (c11 + c10) / N); k = norm.ppf(1 - (c11 + c01) / N); p11 = c11 / N
    f = lambda r: multivariate_normal.cdf([-h, -k], mean=[0, 0], cov=[[1, r], [r, 1]]) - p11
    if f(-.995) > 0: return -.995
    if f(.995) < 0: return .995
    return brentq(f, -.995, .995, xtol=1e-4)


F = np.zeros((len(ids), 12))
for j, c in enumerate(L):
    w = C4.get(c, .5); b = w * RR[:, j] + (1 - w) * pct(R6[:, j]); F[:, j] = .8 * pct(b) + .2 * pct(S6[:, j])
g74 = g[L].astype(float)
lab2 = labels('v2'); tr = [i for i in d7.index if i in lab2.index and i not in goldids]
P7 = d7.loc[tr, ['p|' + c for c in L]].values; Y2 = lab2.loc[tr, L].values
rows = []
for t in ('MCL', 'Contusion', 'Fracture'):
    for o in L:
        if o == t: continue
        jt, jo = L.index(t), L.index(o)
        e58 = tetra(Y[:, jt], Y[:, jo]); m = g74[[t, o]].dropna(); e74 = tetra(m[t].values, m[o].values)
        rc4 = spearmanr(F[:, jt], F[:, jo]).correlation; r7r = spearmanr(P7[:, jt], P7[:, jo]).correlation
        ok = ~np.isnan(Y2[:, jt]) & ~np.isnan(Y2[:, jo]); erep = tetra(Y2[ok, jt], Y2[ok, jo])
        dg = auc(pct(F[:, jt]) - .1 * pct(F[:, jo]), Y[:, jt].astype(int)) - auc(F[:, jt], Y[:, jt].astype(int))
        okt = ~np.isnan(Y2[:, jt])
        drep = auc(pct(P7[okt, jt]) - .1 * pct(P7[okt, jo]), Y2[okt, jt].astype(int)) - auc(P7[okt, jt], Y2[okt, jt].astype(int))
        rows.append(dict(pair='%s <- %s' % (t, o), g58_tet=e58, g74_tet=e74, c4_rho=rc4, gap58=rc4 - e58, gap74=rc4 - e74,
                         r7_train_rho=r7r, rep_tet=erep, d_g58_decor=dg, d_rep_decor=drep))
C = pd.DataFrame(rows)
print('\n[C] coupling for MCL / Contusion / Fracture (gap = c4 Spearman - expert tetrachoric; decor a .1 single-label AUC delta)')
print(C.round(3).to_string(index=False))
