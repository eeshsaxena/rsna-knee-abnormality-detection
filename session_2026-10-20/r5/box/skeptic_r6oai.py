# skeptic_r6oai.py (10-20 r5 SKEPTIC, Claude). Box only, read-only, prints AGGREGATES ONLY (n, prevalence, AUC, deltas, P,
# rho, q). Never prints an id or a row; writes nothing. Independent re-implementation (does NOT import r5_analyze):
#  1 labels: MOAKS cartilage codes parsed as STRINGS "area.ft" (not float arithmetic), compared with the float parse
#  2 prefix: are the R6-scored OAI knees exactly a prefix of the stratified order, and does the prefix keep the OAI 8-cell joint
#  3 AUCs with sklearn.roc_auc_score
#  4 R1 as an EXACT importance-weighted estimate (weighted mid-rank pct + weighted Mann-Whitney AUC) instead of multinomial
#    resampling; weighted rho / q
#  5 target uncertainty: gold74 8-cell joint ~ Dirichlet(counts + .5) x knee Poisson(1) bootstrap, joint re-matched per draw
#  6 the REAL c4 OA structure (kernel_v224r6a25c4j cell 14 + S6F cell 18): per label
#      c4 = .8 pct( wr pct(Raptor) + (1 - wr) pct(R6) ) + .2 pct(S6fall),  wr = .5 Medial / .2 Lateral / .5 PF
#    with D5 (OAI-free fleet) standing in for Raptor ("c4D5")
#  7 RSNA-side coupling of D5 (fleet OOF on RSNA train, competition data only) and gold58 / gold74 rho, q, gap, E-OA
import os, json, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
from sklearn.metrics import roc_auc_score

H = os.path.expanduser('~')
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
       "Baker's", 'Contusion', 'Fracture']
OA = ['Medial OA', 'Lateral OA', 'PF OA']
COMP = {'Medial OA': ['cmf', 'pmf', 'amt', 'cmt', 'pmt'], 'Lateral OA': ['clf', 'plf', 'alt', 'clt', 'plt'],
        'PF OA': ['mp', 'lp', 'amf', 'alf']}
G74 = np.array([33, 12, 3, 2, 4, 8, 2, 7], float)       # cells 000..111 (M, L, PF bits), gold74 complete n 71
WR = np.array([.5, .2, .5])                             # c4 stage-3 Raptor weight per OA label
AL = (.05, .1, .15, .2)
B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000


def rdp(p):
    d = pd.read_csv(p, dtype={'StudyInstanceUID': str})
    return d.drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')


def pct(v):
    return rankdata(v) / len(v)


def wpct(v, w):
    """weighted mid-rank CDF: (sum w[f < v_i] + .5 sum w[f == v_i]) / sum w."""
    o = np.argsort(v, kind='mergesort'); vs = v[o]; ws = w[o]
    u, start = np.unique(vs, return_index=True)
    gw = np.add.reduceat(ws, start); below = np.concatenate([[0.], np.cumsum(gw)[:-1]])
    g = np.searchsorted(u, v)
    return (below[g] + .5 * gw[g]) / w.sum()


def wauc(s, y, w):
    pos = y == 1; neg = ~pos
    sn, wn = s[neg], w[neg]
    o = np.argsort(sn, kind='mergesort'); sn, wn = sn[o], wn[o]
    u, start = np.unique(sn, return_index=True)
    gw = np.add.reduceat(wn, start); cum = np.concatenate([[0.], np.cumsum(gw)])
    lo = np.searchsorted(u, s[pos], 'left'); hi = np.searchsorted(u, s[pos], 'right')
    less = cum[lo]; eq = cum[hi] - cum[lo]
    return float((w[pos] * (less + .5 * eq)).sum() / (w[pos].sum() * wn.sum()))


def decor_w(F, a, w):
    U = np.stack([wpct(F[:, k], w) for k in range(3)], 1)
    return np.stack([U[:, k] - a * U[:, [m for m in range(3) if m != k]].mean(1) for k in range(3)], 1)


def deltas_w(F, Y, w):
    base = np.mean([wauc(F[:, k], Y[:, k], w) for k in range(3)])
    return np.array([np.mean([wauc(decor_w(F, a, w)[:, k], Y[:, k], w) for k in range(3)]) - base for a in AL])


def wcorr(x, y, w):
    p = w / w.sum(); mx = (p * x).sum(); my = (p * y).sum()
    return (p * (x - mx) * (y - my)).sum() / np.sqrt((p * (x - mx) ** 2).sum() * (p * (y - my) ** 2).sum())


def rho_q_w(F, Y, w):
    Z = np.stack([norm.ppf(np.clip(wpct(F[:, k], w), 1e-6, 1 - 1e-6)) for k in range(3)], 1)
    rho = np.mean([wcorr(Z[:, i], Z[:, j], w) for i, j in ((0, 1), (0, 2), (1, 2))])
    q = np.mean([np.mean([wcorr(Z[:, j], Y[:, k], w) for j in range(3) if j != k]) / wcorr(Z[:, k], Y[:, k], w) for k in range(3)])
    phi = np.mean([wcorr(Y[:, i].astype(float), Y[:, j].astype(float), w) for i, j in ((0, 1), (0, 2), (1, 2))])
    return rho, q, phi


def cells_of(Y):
    return Y[:, 0] * 4 + Y[:, 1] * 2 + Y[:, 2]


def r1_weights(cell, pg, base_w=None):
    bw = np.ones(len(cell)) if base_w is None else base_w
    po = np.bincount(cell, weights=bw, minlength=8) / bw.sum()
    return bw * (pg / np.maximum(po, 1e-12))[cell]


# ---------------- 1. labels (string parse) ----------------
S = pd.read_csv(H + '/oai_build/oai_label_sources.csv', dtype=str).set_index('kid')


def parse_codes(col):
    v = col.fillna('').str.strip()
    ok = v.str.match(r'^\d(\.\d)?$')
    area = pd.to_numeric(v.str[0].where(ok), errors='coerce')
    ft = pd.to_numeric(v.str[2].where(ok & (v.str.len() == 3)), errors='coerce').where(ok, np.nan)
    ft = ft.where(~(ok & (v.str.len() == 1)), 0.)
    return area.values, ft.values


YS = {}
for name in ('P_FT', 'P_AREA'):
    Y = pd.DataFrame(index=S.index, columns=OA, dtype=float)
    for lab, regs in COMP.items():
        A = np.stack([parse_codes(S['m_cm_' + r])[0] for r in regs], 1); FT = np.stack([parse_codes(S['m_cm_' + r])[1] for r in regs], 1)
        fin = np.isfinite(A)
        pos = ((FT >= 1) & fin) if name == 'P_FT' else ((A >= 2) & fin)
        y = pos.any(1).astype(float); y[~fin.any(1)] = np.nan; Y[lab] = y
    YS[name] = Y.dropna()
# float parse (the r5 / skeptic code path) for comparison
Sf = S.copy()
mism = {}
for name in ('P_FT', 'P_AREA'):
    Yf = pd.DataFrame(index=S.index, columns=OA, dtype=float)
    for lab, regs in COMP.items():
        c = Sf[['m_cm_' + r for r in regs]].astype(float).values
        area = np.floor(c + 1e-6); ft = np.round((c - area) * 10); fin = np.isfinite(c)
        pos = ((ft >= 1) & fin) if name == 'P_FT' else ((area >= 2) & fin)
        y = pos.any(1).astype(float); y[~fin.any(1)] = np.nan; Yf[lab] = y
    Yf = Yf.dropna()
    same_idx = YS[name].index.equals(Yf.index)
    mism[name] = (bool(same_idx), int((YS[name].loc[Yf.index].values != Yf.values).sum()) if same_idx else -1, len(Yf))
print('LABELS string-vs-float parse: {name: (same knee set, cell mismatches, n)}', mism)

# ---------------- 2. prefix ----------------
order = json.load(open(H + '/r5_oai/ids_oai_all.json'))
r6 = rdp(H + '/r5_oai/run_r6/worker0/predictions.csv')[LAB].astype(float)
s6 = rdp(H + '/r5_oai/s6_r5/out/predictions.csv')[LAB].astype(float)
d5 = rdp(H + '/oai_build/teacher/teacher_D5_mean_v00.csv')[LAB].astype(float)
r6o = r6[r6.index.str.startswith('oai_')].dropna()
k = len(r6o)
print('PREFIX R6 OAI scored', k, '| equals first-k of stratified order:', set(order[:k]) == set(r6o.index),
      '| R6 OAI rows with any NaN:', int(r6[r6.index.str.startswith('oai_')].isna().any(axis=1).sum()),
      '| exact duplicate OA-trio rows among scored:', int(r6o[OA].duplicated().sum()),
      '| S6 OAI scored', int(s6.index.str.startswith('oai_').sum()))
Yall = YS['P_FT']
ca = np.bincount(cells_of(Yall.loc[Yall.index.intersection(order)].values.astype(int)), minlength=8)
cp = np.bincount(cells_of(Yall.loc[Yall.index.intersection(r6o.index)].values.astype(int)), minlength=8)
print('PREFIX 8-cell P_FT frac all-eligible', np.round(ca / ca.sum(), 4).tolist(), '| R6 prefix', np.round(cp / cp.sum(), 4).tolist())

# ---------------- models on the common knees ----------------
common = r6o.index.intersection(s6.dropna().index).intersection(d5.dropna().index).intersection(Yall.index)
M = {'R6': r6o.loc[common, OA].values, 'S6fall': s6.loc[common, OA].values, 'D5': d5.loc[common, OA].values}


def pctc(X):
    return np.stack([pct(X[:, j]) for j in range(X.shape[1])], 1)


M['R6.8S6.2'] = .8 * pctc(M['R6']) + .2 * pctc(M['S6fall'])
v224 = WR * pctc(M['D5']) + (1 - WR) * pctc(M['R6'])
M['c4D5'] = .8 * pctc(v224) + .2 * pctc(M['S6fall'])
M['R6+D5'] = .5 * pctc(M['R6']) + .5 * pctc(M['D5'])
print('COMMON knees', len(common))

# ---------------- 3. AUCs (sklearn) ----------------
Tl = pd.read_csv(H + '/oai_build/oai_labels_T123.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
rows = []
for c in ['ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Effusion']:
    y = (Tl[c] >= .5).astype(float).where(Tl[c].notna()).reindex(common).dropna()
    rows.append((c, len(y), round(float(y.mean()), 3), round(roc_auc_score(y.values, r6o.loc[y.index, c].values), 4),
                 round(roc_auc_score(y.values, s6.loc[y.index, c].values), 4)))
for name in ('P_FT', 'P_AREA'):
    Y = YS[name].reindex(common)
    for c in OA:
        rows.append((c + ' ' + name, len(Y), round(float(Y[c].mean()), 3), round(roc_auc_score(Y[c].values, r6o.loc[common, c].values), 4),
                     round(roc_auc_score(Y[c].values, s6.loc[common, c].values), 4)))
print('AUC sklearn (label, n, prev, R6, S6fall):')
for r in rows:
    print('  ', r)

# ---------------- 4. R0 / R1 exact weighted ----------------
pg0 = (G74 + .5) / (G74 + .5).sum()
for name in ('P_FT', 'P_AREA'):
    Yv = YS[name].reindex(common).values.astype(int); cell = cells_of(Yv)
    w1 = r1_weights(cell, pg0); one = np.ones(len(Yv))
    print(f'== {name} n {len(Yv)} R0 / R1 (exact weighted; OA-mean dAUC at a {AL})')
    for m, F in M.items():
        r0 = deltas_w(F, Yv, one); r1 = deltas_w(F, Yv, w1)
        rq0 = rho_q_w(F, Yv, one); rq1 = rho_q_w(F, Yv, w1)
        print(f'  {m:9s} R0 {np.round(r0, 4).tolist()} rho {rq0[0]:.3f} q {rq0[1]:.3f} phi {rq0[2]:.2f} | R1 {np.round(r1, 4).tolist()} '
              f'rho {rq1[0]:.3f} q {rq1[1]:.3f} gap {rq1[0] - rq1[1]:+.3f} phi {rq1[2]:.2f}')

# ---------------- 5. R1 with target (Dirichlet) and knee (Poisson) uncertainty, P_FT ----------------
Yv = YS['P_FT'].reindex(common).values.astype(int); cell = cells_of(Yv)
rng = np.random.default_rng(1020)
res = {m: {'tgt': [], 'knee': [], 'both': []} for m in M}
for b in range(B):
    pgb = rng.dirichlet(G74 + .5); pois = rng.poisson(1., len(Yv)).astype(float)
    wt = r1_weights(cell, pgb); wk = r1_weights(cell, pg0, pois); wb = r1_weights(cell, pgb, pois)
    for m, F in M.items():
        res[m]['tgt'].append(deltas_w(F, Yv, wt)); res[m]['knee'].append(deltas_w(F, Yv, wk)); res[m]['both'].append(deltas_w(F, Yv, wb))
print(f'== P_FT R1 uncertainty, B {B}: per a {AL}: mean / sd / P(>0); target = Dirichlet(gold74 + .5), knee = Poisson(1) bootstrap')
for m in M:
    for src in ('knee', 'tgt', 'both'):
        X = np.array(res[m][src])
        print(f'  {m:9s} {src:4s} ' + ' | '.join(f'a{a:.2f} {X[:, i].mean():+.4f} sd {X[:, i].std():.4f} P {np.mean(X[:, i] > 0):.2f} '
                                              f'[{np.percentile(X[:, i], 5):+.4f},{np.percentile(X[:, i], 95):+.4f}]' for i, a in enumerate(AL)))
    X = np.array(res[m]['both']); print(f'  {m:9s} P(a.2 > a.1) {np.mean(X[:, 3] > X[:, 1]):.2f}')
# phi of the target draws (what coupling the Dirichlet spans)
ph = []
Yc = np.array([[int(c) for c in format(i, '03b')] for i in range(8)], float)
for b in range(400):
    pgb = rng.dirichlet(G74 + .5)
    ph.append(np.mean([wcorr(Yc[:, i], Yc[:, j], pgb) for i, j in ((0, 1), (0, 2), (1, 2))]))
print('TARGET phi under Dirichlet(gold74 + .5): mean %.3f sd %.3f 5-95%% [%.3f, %.3f]' % (np.mean(ph), np.std(ph), *np.percentile(ph, [5, 95])))

# ---------------- 7. RSNA side: D5 fleet OOF coupling and gold58 / gold74 ----------------
oof = []
for f in range(5):
    p = f'{H}/fleet_runs/f{f}_s1043_ttr_frac75/oof_f{f}_s1043_raw.csv'
    d = rdp(p)[LAB].astype(float); d['fold'] = f; oof.append(d)
oof = pd.concat(oof); oof = oof[~oof.index.duplicated()]
rr = []
for f, g in oof.groupby('fold'):
    Z = np.stack([norm.ppf((rankdata(g[c]) - .5) / len(g)) for c in OA], 1); rr.append(np.corrcoef(Z.T)[np.triu_indices(3, 1)].mean())
print('RSNA D5 OOF (train, report-label model, no labels needed) OA model rho per fold', np.round(rr, 3).tolist(), 'mean %.3f n %d' % (np.mean(rr), len(oof)))
G = pd.read_csv(H + '/rsna_train/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')


def gstats(F, Y):
    one = np.ones(len(Y)); rho, q, phi = rho_q_w(F, Y, one)
    return rho, q, rho - q, phi, deltas_w(F, Y, one)


exact = {'R6 exact': rdp(H + '/r6g58/run_b/worker0/predictions.csv')[LAB].astype(float),
         'S6 exact': rdp(H + '/s6seed/gold58_serve_gpu/nartaa_s42/out/predictions.csv')[LAB].astype(float),
         'R6 approx': r6[~r6.index.str.startswith('oai_')], 'S6 approx': s6[~s6.index.str.startswith('oai_')], 'D5 OOF': oof[LAB]}
for setname, gg in (('gold58', G[G.source == 'original_gold58']), ('gold74', G)):
    Yg = gg[OA].astype(float).dropna()
    for m, P in exact.items():
        ids = Yg.index.intersection(P.dropna().index)
        if len(ids) < 30:
            continue
        rho, q, gap, phi, d = gstats(P.loc[ids, OA].values, Yg.loc[ids].values.astype(int))
        print(f'{setname} {m:9s} n {len(ids)} phi {phi:.2f} rho {rho:.3f} q {q:.3f} gap {gap:+.3f} | E-OA a {AL} {np.round(d, 4).tolist()}')
