# census.py (10-20 push, Claude). RESEARCH ONLY, local CPU, competition files only (no OAI rows; the OAI numbers are the
# aggregate JSON block already in session_2026-10-19/ex/box_coupling.log). Prints aggregates and writes census_*.csv / .json.
#
# TRACK 1 coupling census, all 66 label pairs:
#   expert coupling  : gold74 phi / tetrachoric (+ bootstrap sd), gold58 tetrachoric
#   model coupling   : Spearman of the FINAL c4 output on gold58 (cell-18 exec, parity .94113), eoa20 for OA pairs,
#                      R7 OOF on gold58 / gold74 / train (minus gold74), s1044 OOF on train
#   report coupling  : report_labels_v2 (YES/NO, UNK dropped pairwise) tetrachoric on train minus gold74; v4 as a second reader
#   excess           : c4 rho - expert tet (gold58 same studies, paired bootstrap; gold74 larger label set)
# Directional transforms (132 directions): out_t = pct(pct(f_t) - a * pct(f_o)) on the CURRENT base (eoa20 final = c4 on the
#   9 non-OA columns), a in {-.2,-.1,-.05,.05,.1,.2} (a > 0 decorrelate, a < 0 borrow). Theory: d = rho_ns - q with
#   q = corr(z_o, y_t) / corr(z_t, y_t) (d > 0 -> decorrelation helps at small a). Paired bootstrap (2000), split-half sign agreement
#   (1000 random 29/29 splits), R7 OOF on gold74 (n <= 74) as a second model, report-OOF mechanism control (R7 and the mean of
#   R7 / s1044 / s1045 vs v2 and v4; the report-inheritance mechanism predicts decorrelation LOSES there), OAI (CLAIMED, 10-19 box).
# Honesty: selection-CV on gold58 (pick on one half, score on the other) and empirical-Bayes shrinkage across directions.
import sys, json, itertools, functools
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
P = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push/'
sys.path.insert(0, P); sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from cbase import load, L, AB, pct
import load as LD
rng = np.random.default_rng(20261020)
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
OA = ['Medial OA', 'Lateral OA', 'PF OA']; J = {c: j for j, c in enumerate(L)}
AV = [-.2, -.1, -.05, .05, .1, .2]


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def spear(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


TG = np.linspace(-.995, .995, 3981); I0 = int(np.argmin(np.abs(TG))); DT = np.diff(TG)


@functools.lru_cache(maxsize=None)
def tet_counts(c00, c01, c10, c11):
    c00, c01, c10, c11 = c00 + .5, c01 + .5, c10 + .5, c11 + .5; N = c00 + c01 + c10 + c11
    pa = (c10 + c11) / N; pb = (c01 + c11) / N; p11 = c11 / N
    h = norm.ppf(1 - pa); k = norm.ppf(1 - pb); s = 1 - TG ** 2
    f = np.exp(-(h * h - 2 * TG * h * k + k * k) / (2 * s)) / (2 * np.pi * np.sqrt(s))
    C = np.concatenate([[0.], np.cumsum((f[1:] + f[:-1]) / 2 * DT)]); C -= C[I0]
    P11 = pa * pb + C
    return float(np.interp(p11, P11, TG))


def cnt(a, b):
    a = np.asarray(a, int); b = np.asarray(b, int)
    return (int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum()))


def tet(a, b): return tet_counts(*cnt(a, b))


def phi(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return np.nan if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------- data
d = load(); g74 = d['g74']; Y58 = d['Y58']; n58 = len(Y58)
C4 = d['c4']; CUR = d['eoa20']          # CUR = current scored best-id base (eoa20); non-OA columns == c4
assert np.allclose(C4[:, [J[c] for c in L if c not in OA]], CUR[:, [J[c] for c in L if c not in OA]])
r7 = LD.r7(); allg = set(g74.StudyInstanceUID)
R7cols = ['p|' + c for c in L]
R7_74 = r7.reindex(d['ids74'])[R7cols].values.astype(float)
R7_58 = r7.reindex(d['ids'])[R7cols].values.astype(float)
s44, _ = LD.s6seed(1044); s45, _ = LD.s6seed(1045)
lv2 = LD.labels('v2'); lv4 = LD.labels('v4')
tr = [i for i in r7.index if i not in allg and i in lv2.index]
F7 = r7.loc[tr, R7cols].values.astype(float); Yv2 = lv2.loc[tr, L].values; Yv4 = lv4.reindex(tr)[L].values
tr3 = [i for i in tr if i in s44.index and i in s45.index]
M3 = np.column_stack([np.mean([pct(X[:, j]) for X in (r7.loc[tr3, R7cols].values.astype(float), s44.loc[tr3, L].values, s45.loc[tr3, L].values)], 0) for j in range(12)])
Y3v2 = lv2.loc[tr3, L].values; Y3v4 = lv4.reindex(tr3)[L].values
S44 = s44.reindex(tr)[L].values
print('n gold58 %d, gold74 %d, R7 train (minus gold74, v2-covered) %d, R7+s1044+s1045 %d' % (n58, len(g74), len(tr), len(tr3)))
print('c4 gold58 macro %.5f, eoa20 %.5f (parity .94113 / .94360)' % (np.mean([auc(C4[:, j], Y58[:, j]) for j in range(12)]),
                                                                    np.mean([auc(CUR[:, j], Y58[:, j]) for j in range(12)])))
sys.stdout.flush()

# ---------------------------------------------------------------- part A: 66-pair coupling census
BOOT = rng.integers(0, n58, (1000, n58))
rows = []
for a, b in itertools.combinations(L, 2):
    ja, jb = J[a], J[b]; r = dict(pair=AB[a] + '-' + AB[b], a=a, b=b)
    m = g74[[a, b]].dropna(); x = m[a].values.astype(int); y = m[b].values.astype(int)
    r['g74_n'] = len(m); r['g74_pos'] = (int(x.sum()), int(y.sum())); r['g74_phi'] = phi(x, y); r['g74_tet'] = tet(x, y)
    bt = []
    for _ in range(300):
        ii = rng.integers(0, len(m), len(m)); bt.append(tet(x[ii], y[ii]))
    r['g74_tet_sd'] = float(np.std(bt))
    r['g58_tet'] = tet(Y58[:, ja], Y58[:, jb]); r['g58_phi'] = phi(Y58[:, ja], Y58[:, jb])
    r['c4_rho'] = spear(C4[:, ja], C4[:, jb]); r['cur_rho'] = spear(CUR[:, ja], CUR[:, jb])
    ok = ~np.isnan(R7_58[:, ja]); r['r7_g58_rho'] = spear(R7_58[ok, ja], R7_58[ok, jb])
    ok = ~np.isnan(R7_74[:, ja]); r['r7_g74_rho'] = spear(R7_74[ok, ja], R7_74[ok, jb])
    r['r7_tr_rho'] = spear(F7[:, ja], F7[:, jb])
    ok = ~np.isnan(S44[:, ja]); r['s44_tr_rho'] = spear(S44[ok, ja], S44[ok, jb])
    for nm, Yr in (('v2', Yv2), ('v4', Yv4)):
        ok = ~np.isnan(Yr[:, ja]) & ~np.isnan(Yr[:, jb])
        r['rep_%s_tet' % nm] = tet(Yr[ok, ja].astype(int), Yr[ok, jb].astype(int)); r['rep_%s_n' % nm] = int(ok.sum())
    # gaps
    r['gap58'] = r['cur_rho'] - r['g58_tet']; r['gap74'] = r['cur_rho'] - r['g74_tet']
    gb = []
    for ii in BOOT:
        ya, yb = Y58[ii, ja], Y58[ii, jb]
        gb.append(spear(CUR[ii, ja], CUR[ii, jb]) - tet(ya, yb))
    gb = np.array(gb); r['gap58_sd'] = float(gb.std()); r['gap58_P'] = float((gb > 0).mean())
    r['rep_minus_exp'] = r['rep_v2_tet'] - r['g74_tet']          # report-inheritance: reports couple more than experts
    r['r7tr_minus_rep'] = r['r7_tr_rho'] - r['rep_v2_tet']       # does the model reproduce the report coupling?
    gs = [r['gap58'], r['gap74']]
    r['cons'] = min(gs) if all(v > 0 for v in gs) else (max(gs) if all(v < 0 for v in gs) else 0.0)
    rows.append(r)
cen = pd.DataFrame(rows).sort_values('cons', key=lambda s: -s.abs()).reset_index(drop=True)
cen.to_csv(P + 'census_pairs.csv', index=False)
print('\nPART A: 66 pairs (cur = eoa20 final; == c4 except OA pairs). consistent = min(gap58, gap74) if same sign')
cols = ['pair', 'g74_n', 'g74_pos', 'g74_tet', 'g74_tet_sd', 'g58_tet', 'c4_rho', 'cur_rho', 'r7_g74_rho', 'r7_tr_rho', 's44_tr_rho', 'rep_v2_tet',
        'rep_v4_tet', 'gap58', 'gap58_sd', 'gap58_P', 'gap74', 'rep_minus_exp', 'r7tr_minus_rep', 'cons']
with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.max_rows', 100):
    print(cen[cols].round(2).to_string())
sys.stdout.flush()


# ---------------------------------------------------------------- part B: 132 directions
def K(s):
    s = np.asarray(s); return (s[:, None] > s[None, :]).astype(float) + .5 * (s[:, None] == s[None, :])


def tr_score(F, t, o, a):
    return pct(pct(F[:, t]) - a * pct(F[:, o]))


def wdelta(Dm, y, W):
    """AUC delta for kernel-difference matrix Dm under row weights W (S x n)."""
    Wp = W * y; Wn = W * (1 - y); den = Wp.sum(1) * Wn.sum(1)
    num = np.einsum('si,ij,sj->s', Wp, Dm, Wn)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)


Wboot = np.zeros((2000, n58))
for s in range(2000): Wboot[s] = np.bincount(rng.integers(0, n58, n58), minlength=n58)
SPL = np.zeros((1000, 2, n58))
for s in range(1000):
    p_ = rng.permutation(n58); SPL[s, 0, p_[:29]] = 1; SPL[s, 1, p_[29:]] = 1

box = json.loads(open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/ex/box_coupling.log', encoding='utf-8', errors='replace').read()
                 .split('JSON_BEGIN')[1].split('JSON_END')[0].strip())
OM = ['D5mean_s1043', 's1044_mix50', 's1045_288']


def oai_dir(t, o):
    for p, b in box.items():
        x, y = p.split(' | ')
        if {x, y} == {t, o}:
            dd = b['dir'].get(t + ' <- ' + o)
            if dd is None: return None
            return dict(d_m1=[dd[k]['d']['-0.1'] for k in OM], d_p1=[dd[k]['d']['0.1'] for k in OM], P_p1=[dd[k]['P']['0.1'] for k in OM],
                        P_m1=[dd[k]['P']['-0.1'] for k in OM], R2_p1=[dd[k]['rw']['R2']['d.1'] for k in OM], R1_p1=[dd[k]['rw']['R1']['d.1'] for k in OM],
                        R2hi_p1=[dd[k]['rw']['R2hi']['d.1'] for k in OM], gap=[dd[k]['rho_ns'] - dd[k]['q'] for k in OM])
    return None


def ctrl(M, Yr, t, o, a):
    ok = ~np.isnan(Yr[:, t]) & ~np.isnan(M[:, t]) & ~np.isnan(M[:, o]); y = Yr[ok, t].astype(int)
    if y.sum() < 5 or (1 - y).sum() < 5: return np.nan
    return auc(pct(pct(M[ok, t]) - a * pct(M[ok, o])), y) - auc(M[ok, t], y)


drows = []; DM = {}
for t in L:
    jt = J[t]; y = Y58[:, jt]; zt = ns(CUR[:, jt]); ct = np.corrcoef(zt, y)[0, 1]; Kb = K(CUR[:, jt])
    for o in L:
        if o == t: continue
        jo = J[o]; zo = ns(CUR[:, jo]); co = np.corrcoef(zo, y)[0, 1]; rho = np.corrcoef(zt, zo)[0, 1]
        r = dict(dir=AB[t] + ' <- ' + AB[o], t=t, o=o, oa_internal=(t in OA and o in OA), npos=int(y.sum()), base_auc=auc(CUR[:, jt], y),
                 rho_ns=rho, q=co / ct, theory=rho - co / ct, a_star=(ct * rho - co) / (ct - co * rho))
        # bootstrap of the theory gap
        tb = []
        for ii in BOOT[:1000]:
            yy = y[ii]
            if yy.min() == yy.max(): continue
            a1 = np.corrcoef(zt[ii], yy)[0, 1]; a2 = np.corrcoef(zo[ii], yy)[0, 1]; rr = np.corrcoef(zt[ii], zo[ii])[0, 1]
            tb.append(rr - a2 / a1)
        r['theory_P'] = float((np.array(tb) > 0).mean())
        for a in AV:
            Dm = K(tr_score(CUR, jt, jo, a)) - Kb; DM[(t, o, a)] = Dm
            r['d%+.2f' % a] = float(wdelta(Dm, y, np.ones((1, n58)))[0])
        for a in (-.1, .1):
            bb = wdelta(DM[(t, o, a)], y, Wboot); bb = bb[~np.isnan(bb)]
            r['P%+.1f' % a] = float((bb > 0).mean()); r['sd%+.1f' % a] = float(bb.std())
            hA = wdelta(DM[(t, o, a)], y, SPL[:, 0]); hB = wdelta(DM[(t, o, a)], y, SPL[:, 1]); okk = ~np.isnan(hA) & ~np.isnan(hB)
            r['sh_same%+.1f' % a] = float((np.sign(hA[okk]) == np.sign(hB[okk])).mean())
            r['sh_both%+.1f' % a] = float(((hA[okk] > 0) & (hB[okk] > 0)).mean())
            # R7 OOF on gold74 (second model, 16 extra studies) and on gold58 only
            for nm, M, Yg in (('r7g74', R7_74, g74[L].values.astype(float)), ('r7g58', R7_58, Y58.astype(float))):
                r['%s%+.1f' % (nm, a)] = ctrl(M, Yg, jt, jo, a)
            # report-OOF control (mechanism: should lose for a report-inherited coupling)
            r['repR7v2%+.1f' % a] = ctrl(F7, Yv2, jt, jo, a); r['repR7v4%+.1f' % a] = ctrl(F7, Yv4, jt, jo, a)
            r['repM3v2%+.1f' % a] = ctrl(M3, Y3v2, jt, jo, a); r['repM3v4%+.1f' % a] = ctrl(M3, Y3v4, jt, jo, a)
        # R7 gold74 bootstrap P at the theory-signed a
        ox = oai_dir(t, o)
        if ox:
            for k, v in ox.items(): r['oai_' + k] = v
        drows.append(r)
dd = pd.DataFrame(drows)
print('\nPART B: 132 directions computed'); sys.stdout.flush()

# ---------------------------------------------------------------- part C: honesty: empirical Bayes + selection CV
EB = {}
for a in (-.1, .1):
    x = dd.loc[~dd.oa_internal, 'd%+.2f' % a].values; se = dd.loc[~dd.oa_internal, 'sd%+.1f' % a].values
    mu = x.mean(); tau2 = max(0.0, x.var() - np.mean(se ** 2))
    EB[a] = dict(mu=float(mu), tau=float(np.sqrt(tau2)), obs_sd=float(x.std()), mean_se=float(np.sqrt(np.mean(se ** 2))))
    print('EB a %+.1f over 120 non-OA-internal directions: mean %+.4f, observed sd %.4f, mean se %.4f -> true-effect sd tau %.4f' % (
        a, mu, x.std(), np.sqrt(np.mean(se ** 2)), np.sqrt(tau2)))
    k = 'eb%+.1f' % a
    dd[k] = mu + tau2 / (tau2 + dd['sd%+.1f' % a] ** 2) * (dd['d%+.2f' % a] - mu)
    dd[k + '_sd'] = np.sqrt(1 / (1 / max(tau2, 1e-12) + 1 / dd['sd%+.1f' % a] ** 2))

# selection CV: pick on one half of gold58 (S1), evaluate on the other (S2); 1000 random 29/29 splits and 1000 46/12 splits
pairs_ix = {}
for c in itertools.combinations(L, 2): pairs_ix[frozenset(c)] = c
cands = [(t, o, a) for t in L for o in L if o != t and not (t in OA and o in OA) for a in (-.1, .1)]
def cv(frac, S=1000):
    out = {k: [] for k in ('top1_any', 'top1_mech', 'top3_mech', 'all_mech', 'n_mech', 'top1_mech_label')}
    for s in range(S):
        p_ = rng.permutation(n58); m = int(round(frac * n58)); A_, B_ = np.zeros(n58), np.zeros(n58); A_[p_[:m]] = 1; B_[p_[m:]] = 1
        ia = p_[:m]
        # half-A excess per pair (model rho - expert tet, same rows)
        exA = {}
        for (a_, b_) in itertools.combinations(L, 2):
            exA[frozenset((a_, b_))] = spear(CUR[ia, J[a_]], CUR[ia, J[b_]]) - tet(Y58[ia, J[a_]], Y58[ia, J[b_]])
        sc = []
        for (t, o, a) in cands:
            y = Y58[:, J[t]]
            if y[ia].min() == y[ia].max() or y[p_[m:]].min() == y[p_[m:]].max(): continue
            dA = wdelta(DM[(t, o, a)], y, A_[None])[0]; dB = wdelta(DM[(t, o, a)], y, B_[None])[0]
            e = exA[frozenset((t, o))]; mech = (a > 0 and e >= .15) or (a < 0 and e <= -.15)
            sc.append((dA, dB, mech, t, o, a))
        sc.sort(key=lambda z: -z[0])
        out['top1_any'].append(sc[0][1])
        mm = [z for z in sc if z[2] and z[0] > 0]
        out['n_mech'].append(len(mm))
        if mm:
            out['top1_mech'].append(mm[0][1]); out['top3_mech'].append(np.mean([z[1] for z in mm[:3]])); out['all_mech'].append(np.mean([z[1] for z in mm]))
            out['top1_mech_label'].append('%s <- %s %+.1f' % (AB[mm[0][3]], AB[mm[0][4]], mm[0][5]))
    return out
CV = {}
for frac in (.5, .8):
    o = cv(frac); CV[frac] = o
    print('\nselection CV (select on %d studies, score on %d), single-label AUC on the held-out part:' % (round(frac * n58), n58 - round(frac * n58)))
    for k in ('top1_any', 'top1_mech', 'top3_mech', 'all_mech'):
        v = np.array(o[k]); print('  %-10s mean %+.4f  median %+.4f  P(>0) %.2f  (n splits %d)' % (k, v.mean(), np.median(v), (v > 0).mean(), len(v)))
    print('  mean number of mechanism-consistent positive directions on the selection part: %.1f' % np.mean(o['n_mech']))
    vc = pd.Series(o['top1_mech_label']).value_counts().head(8); print('  most frequent top1_mech picks:', vc.to_dict())
sys.stdout.flush()

dd.to_csv(P + 'census_dirs.csv', index=False)
json.dump(dict(EB=EB, CV={str(k): {kk: (list(map(float, vv)) if kk != 'top1_mech_label' else vv) for kk, vv in v.items()} for k, v in CV.items()}),
          open(P + 'census_cv.json', 'w'))
print('\nwrote census_pairs.csv, census_dirs.csv, census_cv.json')
