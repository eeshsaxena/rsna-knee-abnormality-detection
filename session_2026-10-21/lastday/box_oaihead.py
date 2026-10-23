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
PAYLOAD = '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'
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
