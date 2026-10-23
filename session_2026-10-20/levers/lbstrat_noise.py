# lbstrat_noise.py (10-20 ~16:10 IST, levers/LB-strategy lens, Claude). CPU only, local competition data only (gold58 serves),
# no OAI data, nothing written outside session_2026-10-20/levers. Output: lbstrat_noise.log + lbstrat_noise.npz.
# Purpose: the JOINT public / private read-noise of every one-parameter probe we have read on the LB since 10-16 (plus the E-OA
# shapes and R6e20), on the kernel code path (R6 leg, Raptor r384 / r224 kernel serves, S6fall n42gpu), so the LB-history
# empirical-Bayes fit (lbstrat_eb.py) can put every read on a common z-scale.
#   - paired study bootstrap over gold58 (predictions fixed, AUC recomputed per resample), per-label delta covariance,
#   - scaled to the public 390 / private 910 sizes per label with a Hanley 1/P + 1/Q factor, under two prevalence regimes
#     (gold58 = expert-like, report = train-report rates, metric.md sec 2).
# gold58 POINT deltas are printed for information only (gold58 is NOT a selection ruler; A19 showed it is miscalibrated).
import numpy as np, pandas as pd
from scipy.stats import rankdata
D = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58/'
OUT = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/levers/'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
REPORT = dict(zip(L, [.204, .147, .394, .151, .359, .257, .450, .593, .126, .247, .168, .068]))   # metric.md sec 2 (MEASURED there)
g = pd.read_csv(D + "gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); Y = g[L].values.astype(int); n = len(ids)
R6 = pd.read_csv(D + "R6_gold58_serve.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
R384 = np.load(D + "gold58_r384_kernelserve_cpu.npy"); R224 = np.load(D + "gold58_r224_kernelserve.npy")
S6 = pd.read_csv(D + "n42gpu_gold58.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
pct = lambda v: rankdata(v) / len(v)


def auc(s, y):
    p = y.sum(); q = len(y) - p
    r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


UNI = {}
HWT = {"Lateral OA": 0.2, "PF OA": 0.6, "Synovitis": 0.8, "Baker's": 0.35, "Medial OA": 0.5}
C4T = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}
NONTAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Effusion", "Contusion", "Fracture"]
PW = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}
OA = ['Medial OA', 'Lateral OA', 'PF OA']


def final(table, A=0.25, ws6=0.2, wdef=0.5, eoa=0.0, mode='sym'):
    out = np.zeros((n, 12))
    for j, c in enumerate(L):
        rr = (1 - A) * pct(R384[:, j]) + A * pct(R224[:, j]); w = table.get(c, wdef)
        b = w * rr + (1 - w) * pct(R6[:, j]); out[:, j] = (1 - ws6) * pct(b) + ws6 * pct(S6[:, j])
    if eoa:
        U = {c: pct(out[:, L.index(c)]) for c in OA}; new = {}
        for c in OA:
            o = [k for k in OA if k != c]
            if mode == 'sym':
                new[c] = pct(U[c] - eoa * (U[o[0]] + U[o[1]]) / 2)
            else:
                w_ = [PW[tuple(sorted((c, k), key=OA.index))] for k in o]
                new[c] = pct(U[c] - eoa / 2 * (w_[0] * U[o[0]] + w_[1] * U[o[1]]))
        for c in OA: out[:, L.index(c)] = new[c]
    return out


def tab(base, **kw): t = dict(base); t.update(kw); return t


# entries (name -> blend). Bases: R6 (uniform, A .35), HW (heliosli table, A .35), a25 (HW, A .25), c4 (C4T, A .25)
E = {
    'R6': final(UNI, A=.35), 'HW': final(HWT, A=.35), 'a25': final(HWT, A=.25), 'c4': final(C4T, A=.25),
    'hw3': final(HWT, A=.35, ws6=.3), 's615': final(HWT, A=.35, ws6=.15),
    'wr45': final({**{c: .45 for c in NONTAB}, **HWT}, A=.35),
    'fr65': final(tab(HWT, Fracture=.65), A=.25), 'pf50': final(tab(HWT, **{'PF OA': .5}), A=.25),
    'bk50': final(tab(HWT, **{"Baker's": .5}), A=.25), 'lm40': final(tab(HWT, **{'Lateral Meniscus': .4}), A=.25),
    'a15': final(HWT, A=.15), 'c4a15': final(C4T, A=.15),
    'fr100': final(tab(C4T, Fracture=1.0), A=.25), 'fr90': final(tab(C4T, Fracture=.9), A=.25),
    'pf10': final(tab(C4T, **{'PF OA': .1}), A=.25), 'lat10': final(tab(C4T, **{'Lateral OA': .1}), A=.25),
    'c2max': final(tab(C4T, Fracture=1.0, **{'PF OA': .1}), A=.25), 'c2': final(tab(C4T, Fracture=.9, **{'PF OA': .1}), A=.25),
    'c3max': final(tab(C4T, Fracture=1.0, **{'PF OA': .1, 'Lateral OA': .1}), A=.25),
    'eoa10': final(C4T, A=.25, eoa=.1), 'eoa20': final(C4T, A=.25, eoa=.2), 'eoa30': final(C4T, A=.25, eoa=.3),
    'pw20': final(C4T, A=.25, eoa=.2, mode='pw'), 'pw25': final(C4T, A=.25, eoa=.25, mode='pw'), 'pw30': final(C4T, A=.25, eoa=.3, mode='pw'),
    'R6e20': final(UNI, A=.35, eoa=.2),
}
# comparisons (probe, base) exactly as read on the LB (R1, vs the then-best)
CMP = [('HW', 'R6'), ('hw3', 'HW'), ('s615', 'HW'), ('wr45', 'HW'), ('a25', 'HW'),
       ('c4', 'a25'), ('fr65', 'a25'), ('pf50', 'a25'), ('bk50', 'a25'), ('lm40', 'a25'), ('a15', 'a25'), ('c4a15', 'c4'),
       ('fr100', 'c4'), ('fr90', 'c4'), ('pf10', 'c4'), ('lat10', 'c4'), ('c2max', 'c4'), ('c2', 'c4'), ('c3max', 'c4'),
       ('eoa10', 'c4'), ('eoa20', 'c4'), ('eoa30', 'c4'), ('pw20', 'c4'), ('pw25', 'c4'), ('pw30', 'c4'),
       ('pw30', 'eoa20'), ('pw20', 'eoa20'), ('eoa30', 'eoa20'), ('pw25', 'pw30'), ('R6e20', 'R6')]
rng = np.random.default_rng(2010); B = 2000
idxs = []
while len(idxs) < B:
    ix = rng.integers(0, n, n)
    if all(0 < Y[ix, j].sum() < n for j in range(12)): idxs.append(ix)
names = sorted(E)
A_ = {k: np.array([[auc(E[k][ix, j], Y[ix, j]) for j in range(12)] for ix in idxs]) for k in names}   # B x 12 per entry
full = {k: np.array([auc(E[k][:, j], Y[:, j]) for j in range(12)]) for k in names}
P58 = Y.sum(0); Q58 = n - P58


def scale(nt, prev):
    P = np.maximum(nt * prev, 1.0); Q = nt - P
    return (1 / P + 1 / Q) / (1 / P58 + 1 / Q58)


prev_g = P58 / n; prev_r = np.array([REPORT[c] for c in L])
f = {('pub', 'g'): scale(390, prev_g), ('pub', 'r'): scale(390, prev_r), ('priv', 'g'): scale(910, prev_g), ('priv', 'r'): scale(910, prev_r)}
DL = np.stack([A_[p] - A_[b] for p, b in CMP], 0)            # K x B x 12 per-label deltas
K = len(CMP)
res = {}
lines = []
pr = lambda s: (print(s), lines.append(s))
pr('lbstrat_noise: gold58 paired bootstrap B %d, kernel path (R6 leg, r384/r224 kernel serves, S6fall n42gpu); sd in e-4 macro' % B)
pr('%-14s %9s %8s | %8s %8s | %8s %8s | %s' % ('probe-base', 'g58 pt', 'sd@58', 'pub g', 'pub r', 'priv g', 'priv r', 'top labels of the public (g) variance'))
covs = {}
for key, fac in f.items():
    s = np.sqrt(fac)
    Z = DL * s[None, None, :]                               # scaled per-label deltas
    M = Z.mean(2)                                          # K x B macro
    covs[key] = np.cov(M)
for i, (p, b) in enumerate(CMP):
    pt = (full[p] - full[b]).mean() * 1e4
    sd58 = DL[i].mean(1).std() * 1e4
    sds = {k: np.sqrt(covs[k][i, i]) * 1e4 for k in covs}
    Zg = DL[i] * np.sqrt(f[('pub', 'g')])[None, :]
    lv = np.array([np.cov(Zg[:, j], Zg.mean(1))[0, 1] / 12 for j in range(12)])   # label contribution to var(macro) (x12 since macro = mean)
    share = lv / max(lv.sum(), 1e-18); top = np.argsort(-share)[:3]
    pr('%-14s %+9.1f %8.1f | %8.2f %8.2f | %8.2f %8.2f | %s' % (p + '-' + b, pt, sd58, sds[('pub', 'g')], sds[('pub', 'r')], sds[('priv', 'g')], sds[('priv', 'r')],
       ', '.join('%s %.0f%%' % (L[t], 100 * share[t]) for t in top if share[t] > .05)))
# correlations among the A21 / E-OA reads (public, gold regime)
cg = covs[('pub', 'g')]; sd = np.sqrt(np.diag(cg)); cor = cg / np.outer(sd, sd)
ix = {f'{p}-{b}': i for i, (p, b) in enumerate(CMP)}
pr('\npublic-noise correlations (gold regime):')
for a_, b_ in (('eoa10-c4', 'eoa20-c4'), ('eoa20-c4', 'eoa30-c4'), ('eoa20-c4', 'pw20-c4'), ('eoa20-c4', 'pw30-c4'), ('pw20-c4', 'pw30-c4'),
               ('pw25-c4', 'pw30-c4'), ('fr100-c4', 'pf10-c4'), ('c2max-c4', 'fr100-c4'), ('a25-HW', 's615-HW'), ('eoa20-c4', 'R6e20-R6')):
    pr('  corr(%s, %s) = %.3f' % (a_, b_, cor[ix[a_], ix[b_]]))
pr('\nratio private/public sd (gold regime), median over probes: %.3f' % np.median(np.sqrt(np.diag(covs[('priv', 'g')]) / np.diag(covs[('pub', 'g')]))))
np.savez(OUT + 'lbstrat_noise.npz', names=np.array(['%s-%s' % c for c in CMP]),
         cov_pub_g=covs[('pub', 'g')], cov_pub_r=covs[('pub', 'r')], cov_priv_g=covs[('priv', 'g')], cov_priv_r=covs[('priv', 'r')],
         pt=np.array([(full[p] - full[b]).mean() for p, b in CMP]))
open(OUT + 'lbstrat_noise.log', 'w').write('\n'.join(lines) + '\n')
