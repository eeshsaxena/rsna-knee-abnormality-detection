# census_d.py (10-20 push, Claude). RESEARCH ONLY, local CPU, competition files only. Reads census_pairs.csv / census_dirs.csv
# (census.py) and adds: (1) the candidate table (pairs outside the OA trio with |consistent excess| >= .15, signed a), (2) a
# held-out test of the MECHANISM itself (does the half-A coupling excess predict the half-B decorrelation gain across directions?),
# (3) the per-label "general factor" family (t minus the mean of the other 11) with a 12-label null, (4) public / private read
# noise per candidate (R7 OOF + sol56 labels, subsamples of 390 / 910, the noise_corr.py method), (5) posterior, P(>0) and the
# expected value of ONE public probe under rule R1 (follow the API best-id).
import sys, itertools, json
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm, spearmanr
P = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push/'
sys.path.insert(0, P); sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
import census_lib as CL
from cbase import L, AB, pct
import load as LD
rng = np.random.default_rng(1021)
OA = ['Medial OA', 'Lateral OA', 'PF OA']; J = {c: j for j, c in enumerate(L)}
pairs = pd.read_csv(P + 'census_pairs.csv'); dirs = pd.read_csv(P + 'census_dirs.csv')
d = CL.data(); CUR = d['CUR']; Y58 = d['Y58']; n58 = len(Y58)
auc, K, wdelta, tet, spear = CL.auc, CL.K, CL.wdelta, CL.tet, CL.spear

# ------------------------------------------------ (1) candidates
cons = {frozenset((r.a, r.b)): r.cons for r in pairs.itertuples()}
dirs['cons'] = [cons[frozenset((t, o))] for t, o in zip(dirs.t, dirs.o)]
dirs['sgn'] = np.sign(dirs.cons)
cand = dirs[(~dirs.oa_internal) & (dirs.cons.abs() >= .15)].copy()
def pick(r, k):
    s = '+0.1' if r.cons > 0 else '-0.1'; return r[k + s]
for k in ('P', 'sd', 'sh_same', 'sh_both', 'r7g74', 'r7g58', 'repR7v2', 'repR7v4', 'repM3v2', 'repM3v4'):
    cand[k] = [pick(r, k) for _, r in cand.iterrows()]
cand['a'] = np.where(cand.cons > 0, .1, -.1)
cand['d'] = np.where(cand.cons > 0, cand['d+0.10'], cand['d-0.10'])
cand['d05'] = np.where(cand.cons > 0, cand['d+0.05'], cand['d-0.05']); cand['d20'] = np.where(cand.cons > 0, cand['d+0.20'], cand['d-0.20'])
cand['theory_signed'] = cand.theory * np.sign(cand.cons)      # > 0 means the theory agrees with the sign of a
cand['theory_P_signed'] = np.where(cand.cons > 0, cand.theory_P, 1 - cand.theory_P)


def oai_str(r):
    if not isinstance(r.oai_d_p1, str): return 'no OAI label'
    dp = json.loads(r.oai_d_p1); dm = json.loads(r.oai_d_m1); R2 = json.loads(r.oai_R2_p1); R1 = json.loads(r.oai_R1_p1)
    v = dp if r.cons > 0 else dm
    s = ' / '.join('%+.4f' % x for x in v)
    if r.cons > 0: s += ' (R2 %s; R1 %s)' % ('/'.join('%+.4f' % x for x in R2), '/'.join('%+.4f' % x for x in R1))
    if 'Synovitis' in (r.t, r.o): s += ' [OAI Syn = Hoffa; OAI-free Syn = Eff: INVALID]'
    return s


cand['oai'] = [oai_str(r) for r in cand.itertuples()]
cand = cand.sort_values('d', ascending=False)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 40); pd.set_option('display.max_colwidth', 120)
print('(1) CANDIDATES: directions outside the OA trio, pair |consistent excess| >= .15; a = +.1 if excess > 0 (decorrelate) else -.1 (borrow)')
print('    d = gold58 single-label AUC delta on the current base; P = paired bootstrap P(>0); sh = split-half same sign / both > 0')
print(cand[['dir', 'a', 'cons', 'npos', 'base_auc', 'theory_signed', 'theory_P_signed', 'd05', 'd', 'd20', 'P', 'sd', 'sh_same', 'sh_both', 'r7g74', 'r7g58',
            'repR7v2', 'repR7v4', 'repM3v2', 'repM3v4']].round(4).to_string(index=False))
print('\nOAI (CLAIMED, 10-19 box_coupling.log; 3 OAI-free models, d at the signed a .1; R2 = OR-matched to gold74, R1 = cell-matched):')
for r in cand.itertuples(): print('  %-14s %s' % (r.dir, r.oai))
sys.stdout.flush()

# ------------------------------------------------ (2) does the coupling excess predict the decorrelation gain on held-out rows?
print('\n(2) MECHANISM TEST: across the 120 non-OA-internal directions, Spearman(half-A excess or theory gap, half-B delta at a .1)')
dirlist = [(r.t, r.o) for r in dirs.itertuples() if not r.oa_internal]
DM = {(t, o): K(CL.tr_score(CUR, J[t], J[o], .1)) - K(CUR[:, J[t]]) for t, o in dirlist}
cx, ct, cfull = [], [], []
for s in range(400):
    p_ = rng.permutation(n58); A_ = p_[:29]; B_ = np.zeros(n58); B_[p_[29:]] = 1
    exA, thA, dB = [], [], []
    for t, o in dirlist:
        y = Y58[:, J[t]]
        if y[p_[29:]].min() == y[p_[29:]].max() or y[A_].min() == y[A_].max(): continue
        exA.append(spear(CUR[A_, J[t]], CUR[A_, J[o]]) - tet(Y58[A_, J[t]], Y58[A_, J[o]]))
        zt = CL.ns(CUR[A_, J[t]]); zo = CL.ns(CUR[A_, J[o]]); yy = y[A_]
        thA.append(np.corrcoef(zt, zo)[0, 1] - np.corrcoef(zo, yy)[0, 1] / np.corrcoef(zt, yy)[0, 1])
        dB.append(wdelta(DM[(t, o)], y, B_[None])[0])
    cx.append(spearmanr(exA, dB).correlation); ct.append(spearmanr(thA, dB).correlation)
cx = np.array(cx); ct = np.array(ct)
print('    excess (rho - tet) on half A vs delta on half B: mean Spearman %+.3f (sd over splits %.3f, P(>0) %.2f)' % (cx.mean(), cx.std(), (cx > 0).mean()))
print('    theory gap (rho - q) on half A vs delta on half B: mean Spearman %+.3f (sd %.3f, P(>0) %.2f)' % (ct.mean(), ct.std(), (ct > 0).mean()))
nd = dirs[~dirs.oa_internal]
print('    full-sample (in-sample, for reference): Spearman(cons, d+0.10) %+.3f; Spearman(theory, d+0.10) %+.3f' % (
    spearmanr(nd.cons, nd['d+0.10']).correlation, spearmanr(nd.theory, nd['d+0.10']).correlation))
# same question on the report-OOF ruler (large n): does the REPORT-regime excess predict the report-OOF delta? (should: theory)
print('    report OOF (R7 vs v2, n 4,333): Spearman(theory-free excess r7tr_minus_rep of the pair, repR7v2 at +.1) %+.3f' % (
    spearmanr([pairs.set_index('pair').loc[AB[t] + '-' + AB[o] if (AB[t] + '-' + AB[o]) in set(pairs.pair) else AB[o] + '-' + AB[t], 'r7tr_minus_rep'] for t, o in zip(nd.t, nd.o)],
              nd['repR7v2+0.1'], nan_policy='omit').correlation))
sys.stdout.flush()

# ------------------------------------------------ (3) general-factor family: t minus the mean of the other 11 labels
print('\n(3) GENERAL-FACTOR family: out_t = pct(pct(f_t) - a * mean_o pct(f_o)), o = the other 11 labels (12 tests; MCL / Fracture were')
print('    flagged by the census because their excess is positive against almost every other label)')
Wb = np.zeros((2000, n58))
for s in range(2000): Wb[s] = np.bincount(rng.integers(0, n58, n58), minlength=n58)
SPL = np.zeros((1000, 2, n58))
for s in range(1000):
    p_ = rng.permutation(n58); SPL[s, 0, p_[:29]] = 1; SPL[s, 1, p_[29:]] = 1
gf = []
for t in L:
    jt = J[t]; y = Y58[:, jt]; Mo = np.mean([pct(CUR[:, J[o]]) for o in L if o != t], 0)
    row = dict(t=t, npos=int(y.sum()), mean_excess=float(np.mean([cons[frozenset((t, o))] for o in L if o != t])))
    for a in (.05, .1, .2):
        Dm = K(pct(pct(CUR[:, jt]) - a * Mo)) - K(CUR[:, jt]); row['d%.2f' % a] = float(wdelta(Dm, y, np.ones((1, n58)))[0])
        if a == .1:
            bb = wdelta(Dm, y, Wb); bb = bb[~np.isnan(bb)]; row['P.1'] = float((bb > 0).mean()); row['sd.1'] = float(bb.std())
            hA = wdelta(Dm, y, SPL[:, 0]); hB = wdelta(Dm, y, SPL[:, 1]); ok = ~np.isnan(hA) & ~np.isnan(hB)
            row['sh_same'] = float((np.sign(hA[ok]) == np.sign(hB[ok])).mean())
    # report-OOF control
    for nm, M, Yr in (('R7v2', d['F7'], d['Yv2']), ('M3v2', d['M3'], d['Y3v2'])):
        ok = ~np.isnan(Yr[:, jt]); Mo2 = np.mean([pct(M[:, J[o]]) for o in L if o != t], 0)
        row[nm] = auc(pct(pct(M[ok, jt]) - .1 * Mo2[ok]), Yr[ok, jt].astype(int)) - auc(M[ok, jt], Yr[ok, jt].astype(int))
    ok = ~np.isnan(d['R7_74'][:, jt]) & ~np.isnan(d['Y74'][:, jt]); Mo3 = np.mean([pct(np.nan_to_num(d['R7_74'][:, J[o]])) for o in L if o != t], 0)
    yy = d['Y74'][ok, jt].astype(int); row['r7g74'] = auc(pct(pct(d['R7_74'][ok, jt]) - .1 * Mo3[ok]), yy) - auc(d['R7_74'][ok, jt], yy)
    gf.append(row)
gf = pd.DataFrame(gf).sort_values('d0.10', ascending=False)
print(gf.round(4).to_string(index=False))
print('    Spearman over the 12 labels: mean excess vs gold58 d0.10 %+.2f' % spearmanr(gf.mean_excess, gf['d0.10']).correlation)
sys.stdout.flush()

# ------------------------------------------------ (4) public / private read noise per candidate (macro units)
r7 = LD.r7(); g, _ = LD.gold()
lab = pd.read_csv(LD.R + '/_labels/rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[L]
idx = [u for u in r7.index if u not in set(g.StudyInstanceUID) and u in lab.index]
X = r7.loc[idx, ['p|' + c for c in L]].values.astype(float); Yn = (lab.loc[idx].fillna(0).values >= .5).astype(int)


def noise(t, o, a, mean_other=False, reps=400):
    jt = J[t]; out = {}
    for size in (390, 910):
        v = []
        for _ in range(reps):
            ix = rng.choice(len(idx), size, replace=False)
            h = np.mean([pct(X[ix, J[q]]) for q in L if q != t], 0) if mean_other else pct(X[ix, J[o]])
            v.append((auc(pct(X[ix, jt]) - a * h, Yn[ix, jt]) - auc(X[ix, jt], Yn[ix, jt])) / 12)
        out[size] = float(np.std(v))
    return out


# ------------------------------------------------ (5) posterior, P(>0), expected public / private macro, value of one probe
# Prior for a single-label effect at |a| .1 on hidden expert labels (label-AUC units). Three priors (MODEL / GUESS):
#   'mech'  N(+.001, .002): mechanism-consistent pair, no independent ruler (GUESS; E-OA's hidden estimate is about +.0004 to
#           +.001 per OA label with an OAI ruler, so a pair without one should not get more)
#   'cv'    N(-.002, .002): the selection-CV held-out mean of census-style picks on gold58 (MEASURED census.log) as the prior mean
#   'flat'  N(0, .004): weakly informative
# Likelihood: gold58 observed d with bootstrap sd. gold74-R7 and report OOF share labels / are controls, so they are not added.
PRI = {'mech': (.001, .002), 'cv': (-.002, .002), 'flat': (0., .004)}


def post(dobs, se, m0, s0):
    w = s0 ** 2 / (s0 ** 2 + se ** 2); return m0 + w * (dobs - m0), np.sqrt(1 / (1 / s0 ** 2 + 1 / se ** 2))


def probe_value(m, s, sp):
    """E[private macro gain] of probing then following R1 = E[delta * 1(delta + e_pub > 0)], delta ~ N(m, s), e_pub ~ N(0, sp)."""
    z = rng.normal(m, s, 200000); e = rng.normal(0, sp, 200000); return float(np.mean(z * (z + e > 0))), float(np.mean(z + e > 0))


rows = []
tops = list(cand.head(10).itertuples())
for r in tops:
    nz = noise(r.t, r.o, r.a); out = dict(dir=r.dir, a=r.a, d_g58=r.d, se=r.sd, n_pub=nz[390], n_priv=nz[910])
    for k, (m0, s0) in PRI.items():
        m, s = post(r.d, r.sd, m0, s0); m /= 12; s /= 12
        out['E_' + k] = m; out['P_' + k] = float(norm.cdf(m / s)); out['Ppriv_' + k] = float(norm.cdf(m / np.sqrt(s ** 2 + nz[910] ** 2)))
        out['probe_' + k], out['Pmove_' + k] = probe_value(m, s, nz[390])
    rows.append(out)
for t in ('MCL', 'Fracture'):
    rr = gf.set_index('t').loc[t]; nz = noise(t, None, .1, mean_other=True); out = dict(dir=AB[t] + ' <- mean11', a=.1, d_g58=rr['d0.10'], se=rr['sd.1'], n_pub=nz[390], n_priv=nz[910])
    for k, (m0, s0) in PRI.items():
        m, s = post(rr['d0.10'], rr['sd.1'], m0, s0); m /= 12; s /= 12
        out['E_' + k] = m; out['P_' + k] = float(norm.cdf(m / s)); out['Ppriv_' + k] = float(norm.cdf(m / np.sqrt(s ** 2 + nz[910] ** 2)))
        out['probe_' + k], out['Pmove_' + k] = probe_value(m, s, nz[390])
    rows.append(out)
pv = pd.DataFrame(rows)
print('\n(4)+(5) POSTERIOR per candidate (macro units; E = expected hidden macro = expected public = expected private; P = P(true > 0);')
print('    Ppriv = P(realised private delta > 0); n_pub / n_priv = read-noise sd on 390 / 910 studies; probe = E[private gain] of one public')
print('    probe followed by R1, Pmove = P(the probe becomes the API best-id)')
fmt = pv.copy()
for c in fmt.columns:
    if c.startswith(('E_', 'probe_', 'n_')): fmt[c] = fmt[c].map(lambda x: '%+.1e' % x)
print(fmt.round(3).to_string(index=False))
pv.to_csv(P + 'census_posterior.csv', index=False); gf.to_csv(P + 'census_genfactor.csv', index=False); cand.to_csv(P + 'census_candidates.csv', index=False)
json.dump(dict(cx_mean=float(cx.mean()), cx_P=float((cx > 0).mean()), ct_mean=float(ct.mean()), ct_P=float((ct > 0).mean())), open(P + 'census_mech.json', 'w'))
print('\nwrote census_candidates.csv, census_genfactor.csv, census_posterior.csv, census_mech.json')
