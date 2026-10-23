# local_curve.py (10-20 ex, Claude): the box_curve.py variant family on local RSNA data (no OAI content).
#  (i) gold58 c4 final (m14_gate rebuild, as local_compose.py; consistency only, gold58 is NOT a selection ruler)
#  (ii) R7 OOF on gold58 and on gold74 (expert labels)
#  (iii) R7 OOF on train vs report_labels_v2 OA trio (mechanism check: decorrelation should LOSE there)
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
R = 'C:/Users/eeshs/Downloads/rsna-knee'
G = R + '/session_2026-10-18/next/g58'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]
OA = ['Medial OA', 'Lateral OA', 'PF OA']
GAP = {(0, 1): .26, (0, 2): .21, (1, 2): .42}; mg = np.mean(list(GAP.values())); PW = {k: v / mg for k, v in GAP.items()}
GRID = [0, .025, .05, .075, .1, .125, .15]
pct = lambda v: rankdata(v) / len(v)
oth = lambda t: [m for m in range(3) if m != t]
COLS = {}
for i, b1 in enumerate(GRID):
    for j, b2 in enumerate(GRID):
        COLS['g%d%d' % (i, j)] = {t: (b1, b2) for t in range(3)}
COLS['ctrl'] = {t: (-.05, -.05) for t in range(3)}
for a in (.1, .2):
    COLS['PW%02d' % round(a * 100)] = {t: tuple(a / 2 * PW[tuple(sorted((t, o)))] for o in oth(t)) for t in range(3)}
CN = list(COLS)
sym = lambda a: 'g%d%d' % ((int(round(a / .05)),) * 2)
VAR = {}
for a in (.05, .1, .15, .2, .25, .3):
    VAR['S%.2f' % a] = [sym(a)] * 3
VAR['S-0.10'] = ['ctrl'] * 3
for nm, v in {'LP10': (0, .1, .1), 'LP15': (0, .15, .15), 'LP20': (0, .2, .2), 'RAMP': (.05, .1, .15), 'RAMP2': (.1, .15, .2),
              'PFH': (.1, .1, .2), 'PFH2': (.05, .1, .2), 'LH': (.1, .2, .1), 'MH_ctrl': (.2, .1, .1)}.items():
    VAR[nm] = [sym(x) for x in v]
VAR['PW10'] = ['PW10'] * 3; VAR['PW20'] = ['PW20'] * 3


def auc_cols(S, y):
    Rk = rankdata(S, axis=0); p = y.sum(); q = len(y) - p
    return (Rk[y == 1].sum(0) - p * (p + 1) / 2) / (p * q)


def perlabel(U, Y, ii):
    out = []
    for t in range(3):
        o1, o2 = oth(t); B1 = np.array([COLS[c][t][0] for c in CN]); B2 = np.array([COLS[c][t][1] for c in CN])
        O = U[ii, t][:, None] - U[ii, o1][:, None] * B1[None] - U[ii, o2][:, None] * B2[None]
        a = auc_cols(O, Y[ii, t]); out.append(a - a[CN.index('g00')])
    return np.stack(out)


def vs(D, cols):
    return sum(D[..., t, CN.index(cols[t])] for t in range(3))


def describe(U, Y):
    Z = norm.ppf((U * len(U) - .5) / len(U)); rz = np.corrcoef(Z.T); q = []
    for k in range(3):
        ck = np.corrcoef(Z[:, k], Y[:, k])[0, 1]; q.append(np.mean([np.corrcoef(Z[:, j], Y[:, k])[0, 1] for j in oth(k)]) / ck)
    return [float(round(rz[0, 1], 3)), float(round(rz[0, 2], 3)), float(round(rz[1, 2], 3))], np.round(q, 3).tolist()


def evaluate(F, Y, tag, B=2000, minpos=2):
    n = len(Y); U = np.stack([pct(F[:, k]) for k in range(3)], 1)
    D0 = perlabel(U, Y, np.arange(n)); rng = np.random.default_rng(0); DB = []
    for _ in range(B):
        ii = rng.integers(0, n, n); y = Y[ii]
        if (y.sum(0) < minpos).any() or (y.sum(0) > n - minpos).any(): continue
        DB.append(perlabel(U, Y, ii))
    DB = np.array(DB); base = [auc_cols(U[:, [t]], Y[:, t])[0] for t in range(3)]
    rho, q = describe(U, Y)
    print('[%s] n %d prev %s base OA AUC %s | model rho(ns) ML/MPF/LPF %s | q %s | boot %d' % (
        tag, n, np.round(Y.mean(0), 3).tolist(), np.round(base, 4).tolist(), rho, q, len(DB)))
    S1b = vs(DB, VAR['S0.10']); S2b = vs(DB, VAR['S0.20']); S1 = vs(D0, VAR['S0.10']); S2 = vs(D0, VAR['S0.20'])
    res = {}
    for nm, cols in VAR.items():
        x0 = vs(D0, cols); xb = vs(DB, cols)
        i1 = (xb - S1b) / 12; i2 = (xb - S2b) / 12
        res[nm] = (x0 / 12, (x0 - S1) / 12, (x0 - S2) / 12)
        print('   %-8s macro %+.5f (P>0 %.2f) | per-label %s | vs S.1 %+.5f (sd %.5f P>0 %.2f) | vs S.2 %+.5f (sd %.5f P>0 %.2f)' % (
            nm, x0 / 12, np.mean(xb > 0), np.round([D0[t, CN.index(cols[t])] for t in range(3)], 4).tolist(),
            (x0 - S1) / 12, i1.std(), np.mean(i1 > 0), (x0 - S2) / 12, i2.std(), np.mean(i2 > 0)))
    return res


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
# parity with m14_gate (macro .94102) as a code check
mac = np.mean([auc_cols(C4[:, [j]], g58[c].values.astype(int))[0] for j, c in enumerate(L) if 0 < g58[c].sum() < n58])
print('gold58 c4 rebuild macro %.5f (m14_gate parity target .94102)' % mac)
Y58 = g58[OA].values.astype(int)
evaluate(C4[:, [L.index(c) for c in OA]], Y58, 'gold58 c4 final')
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
F7 = r7[['p|' + c for c in OA]].set_axis(OA, axis=1)
evaluate(F7.loc[ids].values, Y58, 'gold58 R7 OOF')
g74 = g.dropna(subset=OA); evaluate(F7.loc[g74.StudyInstanceUID].values, g74[OA].values.astype(int), 'gold74 R7 OOF (OA complete)')
F7t = F7[~F7.index.isin(g.StudyInstanceUID)]
rep = pd.read_csv(R + '/_labels/rsna-knee-llm-labels/report_labels_v2.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
V = pd.DataFrame({c: np.where(rep[c + '__verdict'] == 'YES', 1.0, np.where(rep[c + '__verdict'] == 'NO', 0.0, np.nan)) for c in OA}, index=rep.index)
cm = F7t.index.intersection(V.dropna().index)
evaluate(F7t.loc[cm].values, V.loc[cm].values.astype(int), 'train R7 OOF vs report v2 (OA trio YES/NO)', B=300, minpos=5)
