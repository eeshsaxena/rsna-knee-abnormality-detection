# box_oaihead.py (10-21 lastday, Claude). BOX ONLY, read-only, CPU (nice 19, 2 threads), prints AGGREGATES ONLY (AUC, deltas, sd,
# P, rho, n, prevalence). Never prints an id, a row or a per-study value. Writes nothing. Uses NO OAI rows: box_count.py showed that
# every OAI knee with images is a TRAINING row (fold 9) of every OAI-trained model, so OAI cannot rule an OAI-trained head.
# Embedded payload = RSNA gold58 serves of the c4 members (R6, S6fall, Raptor r384 / r224) from the laptop (local_proxy_check.py,
# proxy == real cell-18 replay of c4 / pw25 / swide, ranks identical).
# Question: does the OA trio of our OAI-expert-trained TTR224 set (DT = K14 T123v00, 5 folds; DO = K12 T12v00) add to swide when
# blended into Medial / Lateral / PF OA at rank weight w (.05-.3), before E-OA (pre) or after the whole cell-18 chain (post)?
# Placebos: the same-recipe OAI-free set D (frac75 s1043, the paired control of DT / DO) and OAI-free E / E2 (288) / C (mix50).
# Rulers: G = gold58 expert labels (INFO only, n 58); O = RSNA 5-fold OOF on the v4 report labels (n 4,333; R0 unweighted, R1 rows
# reweighted so the OA-trio label joint matches gold74's 8 cells = the coupling-matched / expert-regime ruler).
import base64, io, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
H = os.path.expanduser('~')
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]
OA = ['Medial OA', 'Lateral OA', 'PF OA']; JOA = [L.index(c) for c in OA]
B = int(os.environ.get('BOOT', '2000')); BO = int(os.environ.get('BOOTO', '300'))
pct = lambda v: pd.Series(np.asarray(v, np.float64)).rank(method='average', pct=True).values
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
Wp = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}
G74 = {'000': 33, '001': 12, '010': 3, '011': 2, '100': 4, '101': 8, '110': 2, '111': 7}
CELLS = [format(i, '03b') for i in range(8)]
WS = (.05, .1, .15, .2, .3)


def auc(s, y, w=None):
    s = np.asarray(s, float); y = np.asarray(y, int)
    if w is None:
        r = rankdata(s); p = y.sum(); q = len(y) - p
        return np.nan if p == 0 or q == 0 else (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
    o = np.argsort(s, kind='mergesort'); s, y, w = s[o], y[o], w[o]
    _, inv = np.unique(s, return_inverse=True)
    wn = np.bincount(inv, w * (1 - y)); wp = np.bincount(inv, w * y); cn = np.cumsum(wn) - wn
    return float((wp * (cn + .5 * wn)).sum() / (wp.sum() * wn.sum()))


def chain(V, S, X=None, w=0.0, mode='none', a=.25, loamm=.05):
    """swide cell 18 (OA part + lomm05) on stage-3 V and S6fall S; X = extra OA head (n x 3 raw), blended at rank weight w."""
    out = {c: .8 * pct(V[:, L.index(c)]) + .2 * pct(S[:, L.index(c)]) for c in OA + ['Medial Meniscus']}
    if mode == 'pre':
        for k, c in enumerate(OA):
            out[c] = (1 - w) * pct(out[c]) + w * pct(X[:, k])
    U = {c: pct(out[c]) for c in OA}
    for c in OA:
        o = [k for k in OA if k != c]; ww = [Wp[tuple(sorted((c, k), key=OA.index))] for k in o]
        out[c] = pct(U[c] - a / 2.0 * (ww[0] * U[o[0]] + ww[1] * U[o[1]]))
    out['Lateral OA'] = pct(pct(out['Lateral OA']) - loamm * pct(out['Medial Meniscus']))
    if mode == 'post':
        for k, c in enumerate(OA):
            out[c] = pct((1 - w) * pct(out[c]) + w * pct(X[:, k]))
    return np.stack([out[c] for c in OA], 1)


def boot_idx(Y, Bn, seed, p=None):
    rng = np.random.default_rng(seed); n = len(Y); out = []
    while len(out) < Bn:
        ii = rng.choice(n, n, p=p) if p is not None else rng.integers(0, n, n)
        yy = Y[ii]
        if ((yy.sum(0) < 2) | (yy.sum(0) > n - 2)).any():
            continue
        out.append(ii)
    return out


def oa_mean(F, Y, w=None):
    return np.mean([auc(F[:, k], Y[:, k], w) for k in range(3)])


def fleet(paths, ids):
    M = []
    for p in paths:
        d = pd.read_csv(p, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
        M.append(d.loc[ids, L].values.astype(np.float64))
    return np.mean(M, 0), len(M)


def coup(F):
    Z = np.stack([ns(F[:, k]) for k in range(3)], 1); return np.corrcoef(Z.T)[np.triu_indices(3, 1)]


# ------------------------------------------------------------------ G. gold58 (expert, INFO only)
pay = np.load(io.BytesIO(base64.b64decode(PAYLOAD)))
ids = [str(x) for x in pay['ids']]; n = len(ids)
R6s, S6, R384, R224 = pay['R6s'], pay['S6'], pay['R384'], pay['R224']
g = pd.read_csv(H + '/rsna_train/gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
g58 = g[g.source == 'original_gold58'].set_index('StudyInstanceUID')
assert sorted(g58.index) == sorted(ids), 'gold58 id set differs'
Y = g58.loc[ids, OA].values.astype(int)
C4T = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
V = np.zeros((n, 12)); RAP = np.zeros((n, 12))
for j, c in enumerate(L):
    w = C4T.get(c, .5); RAP[:, j] = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); V[:, j] = w * RAP[:, j] + (1 - w) * pct(R6s[:, j])
V = V.astype(np.float32).astype(np.float64)
SETS = {
    'DT (OAI T123, 5f)': [H + '/k14_test/gold/preds/f%d_s1043_ttr_frac75oaiT123v00__raw__A.csv' % f for f in range(5)],
    'DO (OAI T12, 5f)': [H + '/k12_test/gold/preds/f%d_s1043_ttr_frac75oaiT12v00__raw__A.csv' % f for f in range(5)],
    'D (OAI-free twin, 5f)': [H + '/novel/k9gold/preds/f%d_s1043_ttr_frac75__raw__A.csv' % f for f in range(5)],
    'E (OAI-free 288 s1045, 5f)': [H + '/k10_test/gold/preds/f%d_s1045_ttr288frac75__raw__A.csv' % f for f in range(5)],
    'E2 (OAI-free 288 s1046, 5f)': [H + '/k11_test/gold/preds/f%d_s1046_ttr288frac75__raw__A.csv' % f for f in range(5)],
    'C (OAI-free mix50, 5f)': [H + '/novel/gold74_preds/f%d_s1043_ttr_mix50__raw__A.csv' % f for f in range(5)],
    'PL0 (OAI T123+PL, f0)': [H + '/oai_build/gold74_preds/f0_s1043_ttr_frac75oaiT123PL__raw__A.csv'],
    'SW0 (OAI T123 swinT, f0)': [H + '/oai_build/gold74_preds/f0_s1043_ttr_frac75oaiT123swinT__raw__A.csv'],
    'AV0 (OAI T123 allv, f0)': [H + '/oai_build/gold74_preds/f0_s1043_ttr_frac75oaiT123allv__raw__A.csv'],
    'DT0 (OAI T123, f0)': [H + '/oai_build/gold74_preds/f0_s1043_ttr_frac75oaiT123v00__raw__A.csv'],
    'D0 (OAI-free twin, f0)': [H + '/novel/k9gold/preds/f0_s1043_ttr_frac75__raw__A.csv'],
}
FS = {}
for k, ps in SETS.items():
    ok = [p for p in ps if os.path.exists(p)]
    if len(ok) == len(ps):
        FS[k] = fleet(ok, ids)[0]
    else:
        print('MISSING', k, len(ok), 'of', len(ps))
FS['DT+DO (10)'] = (FS['DT (OAI T123, 5f)'] + FS['DO (OAI T12, 5f)']) / 2
print('== G1 gold58 n %d | OA positives M/L/PF %s | label phi ML/MPF/LPF %s' % (n, Y.sum(0).tolist(),
      np.round(np.corrcoef(Y.T)[np.triu_indices(3, 1)], 3).tolist()))
base = chain(V, S6)
rows = [('R6 (exact serve)', R6s[:, JOA]), ('S6fall', S6[:, JOA]), ('Raptor .75 r384 + .25 r224', RAP[:, JOA]), ('c4 stage-3 V', V[:, JOA]),
        ('c4 (.8 V + .2 S6)', np.stack([.8 * pct(V[:, j]) + .2 * pct(S6[:, j]) for j in JOA], 1)),
        ('pw25', chain(V, S6, loamm=0)), ('swide (pw25 + lomm05)', base)] + [(k, F[:, JOA]) for k, F in FS.items()]
for nm, F in rows:
    a3 = [auc(F[:, k], Y[:, k]) for k in range(3)]
    print('  %-28s OA AUC M/L/PF %s mean %.4f | model OA rho(ns) %s mean %.3f' % (nm, np.round(a3, 4).tolist(), np.mean(a3),
          np.round(coup(F), 3).tolist(), coup(F).mean()))
# gold74 member AUCs (fleets only; n 74)
g74 = g.set_index('StudyInstanceUID'); i74 = g74.index.tolist(); Y74 = g74[OA].values.astype(float); M74 = np.isfinite(Y74)
print('== G1b gold74 (fleets only) n %d OA labelled %s positives %s' % (len(i74), M74.sum(0).tolist(), np.nansum(Y74, 0).astype(int).tolist()))
for k, ps in SETS.items():
    if k not in FS:
        continue
    F, m = fleet(ps, i74); a3 = [auc(F[M74[:, j], L.index(c)], Y74[M74[:, j], j].astype(int)) for j, c in enumerate(OA)]
    print('  %-28s gold74 OA AUC %s mean %.4f (members %d)' % (k, np.round(a3, 4).tolist(), np.mean(a3), m))

print('== G2 add X OA head to swide on gold58 (B %d paired study bootstrap; d = OA-mean AUC delta vs swide; macro = d*3/12)' % B)
IDX = boot_idx(Y, B, 7)
bb = np.array([oa_mean(base[ii], Y[ii]) for ii in IDX]); b0 = oa_mean(base, Y)
RES = {}
for k in FS:
    X = FS[k][:, JOA]
    for mode in ('pre', 'post'):
        cells = []
        for w in WS:
            F = chain(V, S6, X, w, mode); d0 = oa_mean(F, Y) - b0
            bd = np.array([oa_mean(F[ii], Y[ii]) for ii in IDX]) - bb
            RES[(k, mode, w)] = (F, bd)
            cells.append('w%.2f %+.4f(sd %.4f P %.2f)' % (w, d0, bd.std(), (bd > 0).mean()))
        print('  %-28s %-4s %s' % (k, mode, ' '.join(cells)))
print('== G3 OAI-specific part: X - placebo at the same w (paired, same bootstrap draws; d OA-mean)')
for x, pl in (('DT (OAI T123, 5f)', 'D (OAI-free twin, 5f)'), ('DO (OAI T12, 5f)', 'D (OAI-free twin, 5f)'),
              ('DT (OAI T123, 5f)', 'E (OAI-free 288 s1045, 5f)'), ('DT+DO (10)', 'D (OAI-free twin, 5f)'),
              ('DT0 (OAI T123, f0)', 'D0 (OAI-free twin, f0)')):
    for mode in ('pre', 'post'):
        cells = []
        for w in WS:
            Fx, bx = RES[(x, mode, w)]; Fp, bp = RES[(pl, mode, w)]
            d0 = oa_mean(Fx, Y) - oa_mean(Fp, Y); bd = bx - bp
            cells.append('w%.2f %+.4f(sd %.4f P %.2f)' % (w, d0, bd.std(), (bd > 0).mean()))
        print('  %-18s - %-24s %-4s %s' % (x[:18], pl[:24], mode, ' '.join(cells)))
print('== G4 per-label deltas vs swide (gold58), DT / DO / D at w .1 and .2')
for k in ('DT (OAI T123, 5f)', 'DO (OAI T12, 5f)', 'D (OAI-free twin, 5f)'):
    for mode in ('pre', 'post'):
        for w in (.1, .2):
            F = RES[(k, mode, w)][0]
            print('  %-24s %-4s w%.2f per-label d M/L/PF %s' % (k[:24], mode, w,
                  np.round([auc(F[:, j], Y[:, j]) - auc(base[:, j], Y[:, j]) for j in range(3)], 4).tolist()))

print('== G5 the OOF stand-in base (rank mean C + E | .2 E2 -> pw25 -> lomm05) on gold58, and DT / D added to it')
VBg = np.stack([(pct(FS['C (OAI-free mix50, 5f)'][:, j]) + pct(FS['E (OAI-free 288 s1045, 5f)'][:, j])) / 2 for j in range(12)], 1)
SBg = FS['E2 (OAI-free 288 s1046, 5f)']; bsg = chain(VBg, SBg); b0g = oa_mean(bsg, Y); bbg = np.array([oa_mean(bsg[ii], Y[ii]) for ii in IDX])
print('  stand-in base gold58 OA-mean %.4f (swide %.4f)' % (b0g, b0))
for k in ('DT (OAI T123, 5f)', 'D (OAI-free twin, 5f)'):
    for mode in ('pre', 'post'):
        cells = []
        for w in WS:
            F = chain(VBg, SBg, FS[k][:, JOA], w, mode); bd = np.array([oa_mean(F[ii], Y[ii]) for ii in IDX]) - bbg
            cells.append('w%.2f %+.4f(sd %.4f P %.2f)' % (w, oa_mean(F, Y) - b0g, bd.std(), (bd > 0).mean()))
        print('  %-24s %-4s %s' % (k[:24], mode, ' '.join(cells)))

# ------------------------------------------------------------------ O. RSNA 5-fold OOF, v4 report labels (n 4,333)
lab = pd.read_csv(H + '/rsna_train/llm_labels_v4_blend.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
OOF = {'DT': H + '/fleet_runs/f{F}_s1043_ttr_frac75oaiT123v00/oof_f{F}_s1043_raw.csv',
       'DO': H + '/fleet_runs/f{F}_s1043_ttr_frac75oaiT12v00/oof_f{F}_s1043_raw.csv',
       'D': H + '/fleet_runs/f{F}_s1043_ttr_frac75/oof_f{F}_s1043_raw.csv',
       'E': H + '/fleet_runs/f{F}_s1045_ttr288frac75/oof_f{F}_s1045_raw.csv',
       'E2': H + '/fleet_runs/f{F}_s1046_ttr288frac75/oof_f{F}_s1046_raw.csv',
       'C': H + '/fleet_runs/f{F}_s1043_ttr_mix50/oof_f{F}_s1043_raw.csv'}
OO = {}
for k, pat in OOF.items():
    parts = []
    for f in range(5):
        p = pat.format(F=f)
        if not os.path.exists(p):
            parts = None; break
        parts.append(pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[L])
    if parts is None:
        print('OOF MISSING', k); continue
    d = pd.concat(parts); assert not d.index.duplicated().any(); assert not d.index.str.startswith('oai_').any()
    OO[k] = d
common = None
for d in OO.values():
    common = d.index if common is None else common.intersection(d.index)
common = common.intersection(lab.index); common = common[~common.isin(g.StudyInstanceUID)]
YO = (lab.loc[common, OA].values >= .5).astype(int); nO = len(common)
key = np.array([''.join(map(str, r)) for r in YO]); po = pd.Series(key).value_counts(normalize=True).reindex(CELLS).fillna(0).values
pg = np.array([G74[c] + .5 for c in CELLS]); pg = pg / pg.sum(); w1 = (pg / np.maximum(po, 1e-9))[[CELLS.index(k) for k in key]]
p1 = w1 / w1.sum(); ess = 1 / (p1 ** 2).sum()
print('== O1 RSNA OOF n %d (gold74 excluded) | v4 OA prev %s | v4 label phi %s | R1 ESS %.0f' % (nO, np.round(YO.mean(0), 3).tolist(),
      np.round(np.corrcoef(YO.T)[np.triu_indices(3, 1)], 3).tolist(), ess))
for k, d in OO.items():
    F = d.loc[common, OA].values
    print('  %-3s OA AUC R0 %s mean %.4f | R1 mean %.4f | model OA rho(ns) %s mean %.3f' % (k, np.round([auc(F[:, j], YO[:, j]) for j in range(3)], 4).tolist(),
          oa_mean(F, YO), oa_mean(F, YO, w1), np.round(coup(F), 3).tolist(), coup(F).mean()))
# stand-in for c4 on OOF: OAI-free report-trained base = rank mean of C, E, E2 (none of D / DT / DO); S slot = E2 (stand-in, .2)
VB = np.stack([(pct(OO['C'].loc[common, c]) + pct(OO['E'].loc[common, c])) / 2 for c in L], 1)
SB = OO['E2'].loc[common, L].values
baseO = chain(VB, SB)
IR0 = boot_idx(YO, BO, 11); IR1 = boot_idx(YO, BO, 13, p1)
b0R0 = oa_mean(baseO, YO); b0R1 = oa_mean(baseO, YO, w1)
bbR0 = np.array([oa_mean(baseO[ii], YO[ii]) for ii in IR0]); bbR1 = np.array([oa_mean(baseO[ii], YO[ii]) for ii in IR1])
print('== O2 add X OA head to the OOF stand-in (C+E | .2 E2 -> pw25 -> lomm05); R0 = report labels, R1 = gold74-coupling reweight '
      '(bootstrap B %d; R1 draws rows with p ~ w1)' % BO)
print('  base OA-mean R0 %.4f R1 %.4f' % (b0R0, b0R1))
RO = {}
for k in ('DT', 'DO', 'D'):
    X = OO[k].loc[common, OA].values
    for mode in ('pre', 'post'):
        c0, c1 = [], []
        for w in WS:
            F = chain(VB, SB, X, w, mode)
            d0 = oa_mean(F, YO) - b0R0; bd0 = np.array([oa_mean(F[ii], YO[ii]) for ii in IR0]) - bbR0
            d1 = oa_mean(F, YO, w1) - b0R1; bd1 = np.array([oa_mean(F[ii], YO[ii]) for ii in IR1]) - bbR1
            RO[(k, mode, w)] = (bd0, bd1, d0, d1)
            c0.append('w%.2f %+.4f(P %.2f)' % (w, d0, (bd0 > 0).mean())); c1.append('w%.2f %+.4f(P %.2f)' % (w, d1, (bd1 > 0).mean()))
        print('  %-3s %-4s R0 %s' % (k, mode, ' '.join(c0)))
        print('  %-3s %-4s R1 %s' % (k, mode, ' '.join(c1)))
print('== O3 OAI-specific part on OOF: X - D at the same w (paired draws)')
for k in ('DT', 'DO'):
    for mode in ('pre', 'post'):
        c0, c1 = [], []
        for w in WS:
            a0, a1, d0, d1 = RO[(k, mode, w)]; q0, q1, e0, e1 = RO[('D', mode, w)]
            c0.append('w%.2f %+.4f(P %.2f)' % (w, d0 - e0, ((a0 - q0) > 0).mean())); c1.append('w%.2f %+.4f(P %.2f)' % (w, d1 - e1, ((a1 - q1) > 0).mean()))
        print('  %s-D %-4s R0 %s' % (k, mode, ' '.join(c0)))
        print('  %s-D %-4s R1 %s' % (k, mode, ' '.join(c1)))
print('DONE')
