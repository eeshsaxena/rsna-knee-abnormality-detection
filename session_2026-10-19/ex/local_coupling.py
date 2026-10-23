# local_coupling.py (10-19 ex, Claude). RESEARCH ONLY, local CPU. Coupling map, local side:
#   gold74 / gold58 expert label coupling (phi, tetrachoric with bootstrap sd),
#   gold58 model coupling (Spearman) for c4 final (m14_gate rebuild), R6, S6fall, Raptor (.75 r384 + .25 r224),
#   R7 OOF (report-trained, train studies minus gold74) model coupling, and report_labels_v2 coupling on the same studies,
#   directional pair decorrelation out_t = pct(pct(f_t) - a * pct(f_o)) on gold58 c4 (consistency only) and on R7 OOF vs
#   report labels (the mechanism predicts it LOSES there).
# Writes ex/local_coupling.json (aggregates: per-pair numbers + gold74 2x2 counts for the box reweighting) and prints a log.
import json, itertools, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr, norm, multivariate_normal
from scipy.optimize import brentq

R = 'C:/Users/eeshs/Downloads/rsna-knee'
G = R + '/session_2026-10-18/next/g58'
OUT = R + '/session_2026-10-19/ex/local_coupling.json'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]
L9 = ["ACL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's"]
pct = lambda v: rankdata(v) / len(v)
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def bvn_up(h, k, r):
    return multivariate_normal.cdf([-h, -k], mean=[0, 0], cov=[[1, r], [r, 1]])


def tetra(a, b):
    a = np.asarray(a, int); b = np.asarray(b, int)
    c11 = ((a == 1) & (b == 1)).sum() + .5; c10 = ((a == 1) & (b == 0)).sum() + .5
    c01 = ((a == 0) & (b == 1)).sum() + .5; c00 = ((a == 0) & (b == 0)).sum() + .5; N = c11 + c10 + c01 + c00
    h = norm.ppf(1 - (c11 + c10) / N); k = norm.ppf(1 - (c11 + c01) / N); p11 = c11 / N
    f = lambda r: bvn_up(h, k, r) - p11
    lo, hi = f(-.995), f(.995)
    if lo > 0: return -.995
    if hi < 0: return .995
    return brentq(f, -.995, .995, xtol=1e-4)


def phi(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0: return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def counts(a, b):
    a = np.asarray(a, int); b = np.asarray(b, int)
    return [int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())]


def dir_delta(F, Y, t, o, a):
    """AUC(label t) change when out_t = pct(f_t) - a * pct(f_o); F, Y are dicts of arrays on the same rows."""
    base = auc(F[t], Y[t]); out = pct(F[t]) - a * pct(F[o])
    return auc(out, Y[t]) - base


# ---------------- gold74 / gold58 labels ----------------
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
g58 = g[g.source == 'original_gold58'].reset_index(drop=True); ids = g58.StudyInstanceUID.tolist(); n58 = len(ids)
rng = np.random.default_rng(0)
pairs = list(itertools.combinations(L9, 2))
res = {'pairs': {}, 'pairs12_extra': {}}
print('gold74 n %d, gold58 n %d' % (len(g), n58))


def label_coupling(df, a, b, B=200, seed=0):
    d = df[[a, b]].dropna(); x = d[a].values.astype(int); y = d[b].values.astype(int)
    out = {'n': len(d), 'prev': [float(x.mean()), float(y.mean())], 'phi': phi(x, y), 'tet': tetra(x, y), 'counts': counts(x, y)}
    if B:
        r = np.random.default_rng(seed); ph = []; te = []
        for _ in range(B):
            ii = r.integers(0, len(d), len(d)); xx = x[ii]; yy = y[ii]
            if xx.std() == 0 or yy.std() == 0: continue
            ph.append(phi(xx, yy)); te.append(tetra(xx, yy))
        out['phi_sd'] = float(np.std(ph)); out['tet_sd'] = float(np.std(te))
    return out


# ---------------- gold58 serves ----------------
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4W = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
C4 = np.zeros((n58, 12)); RAP = np.zeros((n58, 12))
for j, c in enumerate(L):
    w = C4W.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); RAP[:, j] = rr
    b = w * rr + (1 - w) * pct(R6[:, j]); C4[:, j] = .8 * pct(b) + .2 * pct(S6[:, j])
SERVES = {'c4': C4, 'R6': R6, 'S6fall': S6, 'Raptor': RAP}
# parity check vs m14_gate.log (a 0 gold58 macro .94102)
Y58 = g58[L].values.astype(int)
print('c4 rebuild gold58 macro %.5f (m14_gate.log: .94102)' % np.mean([auc(C4[:, j], Y58[:, j]) for j in range(12)]))

# ---------------- R7 OOF on train + report labels v2 ----------------
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
R7G = r7.reindex(ids)[['p|' + c for c in L]].set_axis(L, axis=1)   # R7 OOF on gold58 (same population as the c4 serves)
r7 = r7[~r7.index.isin(g.StudyInstanceUID)]
F7 = r7[['p|' + c for c in L]].set_axis(L, axis=1)
rep = pd.read_csv(R + '/_labels/rsna-knee-llm-labels/report_labels_v2.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
common = F7.index.intersection(rep.index); F7 = F7.loc[common]; rep = rep.loc[common]
V = pd.DataFrame({c: np.where(rep[c + '__verdict'] == 'YES', 1.0, np.where(rep[c + '__verdict'] == 'NO', 0.0, np.nan)) for c in L}, index=common)
V0 = V.fillna(0.0)   # UNK read as absent ("silent = negative")
SOFT = rep[L].astype(float)
print('R7 OOF train (minus gold74) n %d; report v2 YES/NO coverage %s' % (len(common), {c: int(V[c].notna().sum()) for c in L9}))


def model_rho(M, a, b):
    return float(spearmanr(M[:, L.index(a)], M[:, L.index(b)]).correlation)


def do_pair(a, b, full=True):
    e = {}
    e['g74'] = label_coupling(g, a, b, B=200 if full else 0)
    e['g58'] = label_coupling(g58, a, b, B=0)
    e['g58_model_rho'] = {k: model_rho(M, a, b) for k, M in SERVES.items()}
    # c4 directional decorrelation on gold58 (consistency only; gold58 is NOT a selection ruler)
    F = {c: C4[:, L.index(c)] for c in (a, b)}; Yd = {c: Y58[:, L.index(c)] for c in (a, b)}
    e['g58_c4_dir'] = {}
    for t, o in ((a, b), (b, a)):
        e['g58_c4_dir'][t + ' <- ' + o] = {str(x): float(dir_delta(F, Yd, t, o, x)) for x in (-.1, .1, .2)}
    # R7 OOF train coupling + report labels
    e['r7_rho'] = float(spearmanr(F7[a], F7[b]).correlation)
    okg = R7G[[a, b]].dropna(); e['r7_g58_rho'] = float(spearmanr(okg[a], okg[b]).correlation); e['r7_g58_n'] = len(okg)
    d = V[[a, b]].dropna()
    e['rep_dropUNK'] = {'n': len(d), 'prev': [float(d[a].mean()), float(d[b].mean())], 'phi': phi(d[a], d[b]), 'tet': tetra(d[a], d[b])}
    e['rep_UNK0'] = {'n': len(V0), 'prev': [float(V0[a].mean()), float(V0[b].mean())], 'phi': phi(V0[a], V0[b]), 'tet': tetra(V0[a], V0[b])}
    e['rep_soft_rho'] = float(spearmanr(SOFT[a], SOFT[b]).correlation)
    # R7 OOF directional decorrelation vs report labels (mechanism: should NOT gain; report labels carry the coupling)
    e['r7_rep_dir'] = {}
    for t, o in ((a, b), (b, a)):
        dd = V[[t]].join(F7[[t, o]], rsuffix='_f').dropna()
        Fd = {t: dd[t + '_f'].values, o: dd[o].values}; Yd = {t: dd[t].values.astype(int)}
        e['r7_rep_dir'][t + ' <- ' + o] = {str(x): float(dir_delta(Fd, Yd, t, o, x)) for x in (-.1, .1, .2)}
    return e


for a, b in pairs:
    res['pairs'][a + ' | ' + b] = do_pair(a, b)
    e = res['pairs'][a + ' | ' + b]
    print('%-34s g74 n %2d phi %+.2f tet %+.2f (sd %.2f) | g58 phi %+.2f | c4 rho %+.2f R6 %+.2f S6 %+.2f Rap %+.2f | R7 rho %+.2f | rep phi %+.2f tet %+.2f (UNK0 tet %+.2f, soft rho %+.2f)' % (
        a + ' | ' + b, e['g74']['n'], e['g74']['phi'], e['g74']['tet'], e['g74']['tet_sd'], e['g58']['phi'], e['g58_model_rho']['c4'],
        e['g58_model_rho']['R6'], e['g58_model_rho']['S6fall'], e['g58_model_rho']['Raptor'], e['r7_rho'], e['rep_dropUNK']['phi'],
        e['rep_dropUNK']['tet'], e['rep_UNK0']['tet'], e['rep_soft_rho']))
    sys.stdout.flush()
# appendix: pairs involving MCL / Contusion / Fracture (no OAI coverage, so they cannot pass criterion 1; context only)
for a, b in itertools.combinations(L, 2):
    if a in L9 and b in L9: continue
    res['pairs12_extra'][a + ' | ' + b] = do_pair(a, b, full=False)
json.dump(res, open(OUT, 'w'), indent=1)
print('wrote', OUT)
