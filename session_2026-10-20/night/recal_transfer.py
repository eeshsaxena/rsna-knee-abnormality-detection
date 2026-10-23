# recal_transfer.py (10-20 night, Claude, TRACK 1). RESEARCH ONLY, local CPU. No OAI data (OAI enters only as aggregates already on the
# laptop, read in recal_rulers.json). Writes recal_transfer.log / recal_transfer.json next to itself.
#
# What it does
#  1. A calibrated coupling-theory map (CTM) anchored on the E-OA public reads:
#       per-label dAUC_t(a) = s * A(AUC_t, prev_t) * [ log(1 - a q_m) - 0.5 log(1 - 2 a r_tm + a^2 v_m) - m a^2 ]
#     q_m, r_tm, v_m = c4's own gold58 normal-score coupling with the partner combination; A = dAUC / dlog(latent corr) of a
#     binormal threshold model at the label's base AUC and prevalence. s (amplitude) and m (extra curvature) are solved per LB
#     posterior draw from G10 / G20 (eoa10 / eoa20 minus c4), then checked on pw30 (LB-only frozen ruler) and eoa30 / pw20 (CLAIMED).
#     Two anchors: T1 = trio q as measured on gold58 (the gold58 -> hidden gap shift is absorbed in s), T2 = trio q lowered by the
#     implied shift (c4 gap .128 instead of gold58 .102), so a label-side shift can be applied to other pairs explicitly.
#  2. Empirical-Bayes shrinkage of every non-OA-internal directional theory gap (rho - q) on gold58 (bootstrap se), because the
#     census showed raw gold58 gaps do not predict held-out deltas.
#  3. Each killed candidate (component directions) -> calibrated public macro mean / sd, P(best-id move vs pw30) with the
#     candidate's own paired public read noise (R7 OOF + sol56 labels, 400 subsamples of 390 / 910, the census_d.py method),
#     private expected and the R1 value E[delta 1(read > 0)].
import sys, json
import numpy as np, pandas as pd
from scipy.stats import norm, rankdata
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
import census_lib as CL
from census_lib import auc, pct, L, AB
import load as LD

OUTD = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/night/'
log = open(OUTD + 'recal_transfer.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()
rng = np.random.default_rng(1020)
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
J = {c: i for i, c in enumerate(L)}; IA = {v: k for k, v in AB.items()}
OA = ['Medial OA', 'Lateral OA', 'PF OA']
W_PW = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}

# ---------------------------------------------------------------- binormal threshold model: A(AUC, prev) = dAUC / dlog c
Zs = rng.standard_normal(400000); Es = rng.standard_normal(400000)
def auc_c(c, prev):
    tau = norm.ppf(1 - prev); s = c * Zs + np.sqrt(1 - c * c) * Es; y = Zs > tau
    return auc(s[:120000], y[:120000].astype(int))
CG = np.linspace(.2, .995, 60)
_cache = {}
def A_of(base, prev):
    key = (round(base, 4), round(prev, 3))
    if key in _cache: return _cache[key]
    pv = min(max(prev, .03), .97)
    au = np.array([auc_c(c, pv) for c in CG])
    c0 = float(np.interp(base, au, CG))
    h = .01; a1 = np.interp(c0 * np.exp(h), CG, au); a0 = np.interp(c0 * np.exp(-h), CG, au)
    _cache[key] = float((a1 - a0) / (2 * h)); return _cache[key]

# ---------------------------------------------------------------- data (c4 final on gold58, CUR = scored eoa20 final; same as census)
D = CL.data(); C4 = D['C4']; CUR = D['CUR']; Y = D['Y58']; n = len(Y)
Z4 = np.column_stack([ns(C4[:, j]) for j in range(12)]); ZC = np.column_stack([ns(CUR[:, j]) for j in range(12)])


def coup(Z, t, parts, idx=None):
    """r_tm, v_m, q_m of target t vs partner combination m = sum_k w_k z_k, normalised so the kernel's a multiplies m."""
    ii = np.arange(n) if idx is None else idx
    jt = J[t]; y = Y[ii, jt].astype(float); zt = Z[ii, jt]
    m = sum(w * Z[ii, J[o]] for o, w in parts)
    ct = np.corrcoef(zt, y)[0, 1]
    r = np.cov(zt, m)[0, 1] / np.var(zt); v = np.var(m) / np.var(zt); q = np.cov(m, y)[0, 1] / np.cov(zt, y)[0, 1]
    return r, v, q


def logR(a, r, v, q, mcurv):
    return np.log(np.maximum(1 - a * q, 1e-6)) - .5 * np.log(1 - 2 * a * r + a * a * v) - mcurv * a * a


def trio_parts(t, mode):
    o = [k for k in OA if k != t]
    if mode == 'sym': return [(o[0], .5), (o[1], .5)]
    return [(k, W_PW[tuple(sorted((t, k), key=OA.index))] / 2.) for k in o]


base = {c: auc(C4[:, J[c]], Y[:, J[c]]) for c in L}; prev = {c: Y[:, J[c]].mean() for c in L}
A = {c: A_of(base[c], prev[c]) for c in L}
P('A(AUC, prev) per label (c4 gold58 base): ' + ' '.join('%s %.3f(AUC %.3f p %.2f)' % (AB[c], A[c], base[c], prev[c]) for c in L))
TR = {(t, mode): coup(Z4, t, trio_parts(t, mode)) for t in OA for mode in ('sym', 'pw')}
for k, v in TR.items(): P('trio %s %s: r_tm %.3f v_m %.3f q_m %.3f gap %.3f' % (AB[k[0]], k[1], v[0], v[1], v[2], v[0] - v[2]))
gap_g58 = np.mean([TR[(t, 'sym')][0] - TR[(t, 'sym')][2] for t in OA])
P('trio mean sym gap on gold58 (this parametrisation) %.3f' % gap_g58)


def trio_macro(a, mode, s, m, dq=0.):
    return s * sum(A[t] * logR(a, TR[(t, mode)][0], TR[(t, mode)][1], TR[(t, mode)][2] - dq, m) for t in OA) / 12.


# ---------------------------------------------------------------- LB posterior draws (MEASURED counts -> MODEL conversion, val/work)
LB = np.load('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/val/work/lb_post2.npz')
w = LB['w'] / LB['w'].sum(); idx = rng.choice(len(w), 4000, p=w)
G10 = LB['G10'][idx] * 1e-3; G20 = LB['G20'][idx] * 1e-3; Dpw = LB['d'][idx] * 1e-3
P('LB draws: G10 %.2f [%.2f, %.2f]e-4  G20 %.2f [%.2f, %.2f]e-4  pw30-eoa20 %.2f [%.2f, %.2f]e-4' % tuple(
    np.r_[[np.percentile(x, [50, 10, 90]) * 1e4 for x in (G10, G20, Dpw)]].ravel()))

# public sampling noise of the deltas themselves (n 390): the LB draw is the PUBLIC realisation, not the population effect. Box D5
# aggregates (CLAIMED push/implied.md sec 2.2): sd S10 .00025, S20 .00052, corr .94, in OAI units x s (~.82). Draw a population pair.
sd10, sd20, rr = .00025 * .82, .00052 * .82, .94
Cn = np.array([[sd10 ** 2, rr * sd10 * sd20], [rr * sd10 * sd20, sd20 ** 2]])


def solve_sm(g10, g20, dq):
    """solve s, m from two macro targets at a .1 / .2 (sym) by 1-d root finding on m."""
    ms = np.linspace(-2., 6., 801)
    f10 = np.array([trio_macro(.1, 'sym', 1., mm, dq) for mm in ms]); f20 = np.array([trio_macro(.2, 'sym', 1., mm, dq) for mm in ms])
    ratio = f20 / f10; tgt = g20 / g10
    ok = np.isfinite(ratio) & (f10 > 0)
    if not ok.any(): return np.nan, np.nan
    k = np.argmin(np.abs(ratio[ok] - tgt)); mm = ms[ok][k]; s = g10 / f10[ok][k]
    return s, mm


FIT = {}
for anchor, dq in (('T1_g58q', 0.), ('T2_pubgap', .128 - gap_g58)):
    for popn in (False, True):
        S_, M_ = [], []
        for i in range(1500):
            g10, g20 = G10[i], G20[i]
            if popn:  # population effect = public realisation minus a public-noise draw
                e = rng.multivariate_normal([0, 0], Cn); g10, g20 = g10 - e[0], g20 - e[1]
            if g10 <= 0 or g20 <= 0: continue
            s, mm = solve_sm(g10, g20, dq)
            if np.isfinite(s): S_.append(s); M_.append(mm)
        S_, M_ = np.array(S_), np.array(M_)
        key = anchor + ('|population' if popn else '|public')
        FIT[key] = dict(s=S_, m=M_, dq=dq)
        pr = {}
        for nm, a, mode in (('S30', .3, 'sym'), ('PW20', .2, 'pw'), ('PW30', .3, 'pw'), ('S20', .2, 'sym'), ('PW25', .25, 'pw')):
            pr[nm] = np.array([trio_macro(a, mode, s, mm, dq) for s, mm in zip(S_[:600], M_[:600])])
        P('%-22s s %.2f [%.2f, %.2f]  m %.2f [%.2f, %.2f] | predicted vs c4 (e-4): S30 %.1f [%.1f, %.1f]  PW20 %.1f  PW25 %.1f  PW30 %.1f [%.1f, %.1f] | PW30-S20 %+.2f [%+.2f, %+.2f]' % (
            key, *np.percentile(S_, [50, 10, 90]), *np.percentile(M_, [50, 10, 90]),
            *(np.percentile(pr['S30'], [50, 10, 90]) * 1e4), np.median(pr['PW20']) * 1e4, np.median(pr['PW25']) * 1e4,
            *(np.percentile(pr['PW30'], [50, 10, 90]) * 1e4), *(np.percentile(pr['PW30'] - pr['S20'], [50, 10, 90]) * 1e4)))
P('PUBLIC check (CLAIMED val/implied2.md joint refit, e-4 vs c4): eoa30 9.6 [8.1, 11.1]  pw20 10.4  pw30 11.3 [10.4, 11.9]; LB-only pw30 - eoa20 +0.5 [0.2, 1.2]')

# ---------------------------------------------------------------- empirical Bayes on the non-OA-internal directional gaps (gold58, CUR)
BI = [rng.integers(0, n, n) for _ in range(1000)]
rows = []
for t in L:
    for o in L:
        if o == t or (t in OA and o in OA): continue
        r, v, q = coup(ZC, t, [(o, 1.)])
        gb = []
        for ii in BI:
            if Y[ii, J[t]].min() == Y[ii, J[t]].max(): continue
            rb, vb, qb = coup(ZC, t, [(o, 1.)], ii); gb.append(rb - qb)
        rows.append(dict(t=t, o=o, r=r, v=v, q=q, gap=r - q, se=float(np.std(gb))))
EB = pd.DataFrame(rows)
mu = EB.gap.mean(); tau2 = max(EB.gap.var() - (EB.se ** 2).mean(), 0.)
# ML for tau with heteroscedastic se
from scipy.optimize import minimize_scalar
nll = lambda lt: .5 * np.sum(np.log(np.exp(2 * lt) + EB.se ** 2) + (EB.gap - mu) ** 2 / (np.exp(2 * lt) + EB.se ** 2))
lt = minimize_scalar(nll, bounds=(-8, 0), method='bounded').x; tau_ml = float(np.exp(lt))
P('EB over %d non-OA-internal directions (gold58, eoa20 base): mean gap %+.3f, sd %.3f, mean se %.3f -> tau(MoM) %.3f, tau(ML) %.3f' % (
    len(EB), mu, EB.gap.std(), np.sqrt((EB.se ** 2).mean()), np.sqrt(tau2), tau_ml))
EB['w'] = tau_ml ** 2 / (tau_ml ** 2 + EB.se ** 2); EB['gap_post'] = mu + EB.w * (EB.gap - mu); EB['sd_post'] = np.sqrt(1 / (1 / max(tau_ml, 1e-4) ** 2 + 1 / EB.se ** 2))
EBi = EB.set_index(['t', 'o'])

# ---------------------------------------------------------------- read noise (R7 OOF + sol56, the census_d.py method), macro units
r7 = LD.r7(); g, _ = LD.gold()
lab = pd.read_csv(LD.R + '/_labels/rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[L]
ids = [u for u in r7.index if u not in set(g.StudyInstanceUID) and u in lab.index]
X = r7.loc[ids, ['p|' + c for c in L]].values.astype(float); Yn = (lab.loc[ids].fillna(0).values >= .5).astype(int)


def apply(F, comps):
    G = F.copy(); U = {c: pct(F[:, J[c]]) for c in L}
    for t, parts, a in comps:
        G[:, J[t]] = pct(U[t] - a * sum(wt * U[o] for o, wt in parts))
    return G


def noise(comps, reps=300):
    out = {}
    for size in (390, 910):
        v = []
        for _ in range(reps):
            ix = rng.choice(len(ids), size, replace=False); F = X[ix]; G = apply(F, comps)
            v.append(sum(auc(G[:, J[t]], Yn[ix, J[t]]) - auc(F[:, J[t]], Yn[ix, J[t]]) for t in {c[0] for c in comps}) / 12)
        out[size] = float(np.std(v))
    return out


# ---------------------------------------------------------------- candidate evaluation
def evaluate(name, comps, dq_label=0., use_eb=True, anchor='T1_g58q|population', ndraw=4000, Z=None):
    """comps: list of (target, [(partner, weight)], a). Partner weights sum to 1 for a mean. Monte Carlo over (s, m) and gap posteriors."""
    F = FIT[anchor]; k = rng.integers(0, len(F['s']), ndraw); S_, M_ = F['s'][k], F['m'][k]
    tot = np.zeros(ndraw)
    for t, parts, a in comps:
        r, v, q = coup(ZC if Z is None else Z, t, parts)
        if use_eb and not (t in OA and all(o in OA for o, _ in parts)):
            # shrink q through the component gaps (weighted by partner weight), keep r, v as measured (label-free)
            gp = sum(wt * EBi.loc[(t, o)].gap_post for o, wt in parts) / sum(wt for _, wt in parts)
            gs = np.sqrt(sum((wt * EBi.loc[(t, o)].sd_post) ** 2 for o, wt in parts)) / sum(wt for _, wt in parts)
            g_draw = gp + gs * rng.standard_normal(ndraw)
            qd = r - g_draw
        else:
            qd = np.full(ndraw, q)
        qd = qd - dq_label
        tot += S_ * A[t] * (np.log(np.maximum(1 - a * qd, 1e-6)) - .5 * np.log(1 - 2 * a * r + a * a * v) - M_ * a * a)
    tot /= 12.
    return tot


CANDS = [
    # name, components (target, partners, a), note
    ('C6 menisci MM<->LM a.1', [('Lateral Meniscus', [('Medial Meniscus', 1.)], .1), ('Medial Meniscus', [('Lateral Meniscus', 1.)], .1)]),
    ('C7 fluid FL3 a.1', [('Effusion', [('Synovitis', .5), ("Baker's", .5)], .1), ('Synovitis', [('Effusion', .5), ("Baker's", .5)], .1),
                          ("Baker's", [('Effusion', .5), ('Synovitis', .5)], .1)]),
    ('C8 fluid ES2 Eff<->Syn a.1', [('Effusion', [('Synovitis', 1.)], .1), ('Synovitis', [('Effusion', 1.)], .1)]),
    ('C9 OA-meniscus 4 dirs a.1 (on E-OA)', [('Medial OA', [('Medial Meniscus', 1.)], .1), ('Lateral OA', [('Lateral Meniscus', 1.)], .1),
                                             ('Medial Meniscus', [('Medial OA', 1.)], .1), ('Lateral Meniscus', [('Lateral OA', 1.)], .1)]),
    ('C10 ACL<-LM a.1', [('ACL', [('Lateral Meniscus', 1.)], .1)]),
    ('C10 ACL<-MM a.1', [('ACL', [('Medial Meniscus', 1.)], .1)]),
    ('C10 ACL<-Eff a.1', [('ACL', [('Effusion', 1.)], .1)]),
    ('C11 global G12 a.1 (each label minus mean of the other 11; OA trio excluded as targets)', [
        (t, [(o, 1. / 11) for o in L if o != t], .1) for t in L if t not in OA]),
    ('C12 LOA<-MM b.1 (on E-OA)', [('Lateral OA', [('Medial Meniscus', 1.)], .1)]),
    ('C13 BOR MOA+=MM, LOA+=LM (a -.1)', [('Medial OA', [('Medial Meniscus', 1.)], -.1), ('Lateral OA', [('Lateral Meniscus', 1.)], -.1)]),
    ('C14 LOA+=LM (a -.1)', [('Lateral OA', [('Lateral Meniscus', 1.)], -.1)]),
    ('MCL<-MM a.1', [('MCL', [('Medial Meniscus', 1.)], .1)]),
    ('Frac<-MOA a.1', [('Fracture', [('Medial OA', 1.)], .1)]),
    ('MCL<-mean11 a.1', [('MCL', [(o, 1. / 11) for o in L if o != 'MCL'], .1)]),
    ('LOA<-Eff borrow a-.1', [('Lateral OA', [('Effusion', 1.)], -.1)]),
    ('MCL<-Syn a.1', [('MCL', [('Synovitis', 1.)], .1)]),
    ('MCL<-Cont a.1', [('MCL', [('Contusion', 1.)], .1)]),
    ('MCL<-Frac a.1', [('MCL', [('Fracture', 1.)], .1)]),
    ('MCL<-MOA a.1', [('MCL', [('Medial OA', 1.)], .1)]),
    ('LM<-MM a.1', [('Lateral Meniscus', [('Medial Meniscus', 1.)], .1)]),
    ('MM<-LOA a.1', [('Medial Meniscus', [('Lateral OA', 1.)], .1)]),
    ('MCL+Frac minus mean(DEG) a.1', [(t, [(o, .2) for o in ('Medial Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', "Baker's")], .1) for t in ('MCL', 'Fracture')]),
    ('Cont<->Frac a.1', [('Contusion', [('Fracture', 1.)], .1), ('Fracture', [('Contusion', 1.)], .1)]),
    ('E2 Syn+=Eff a-.1', [('Synovitis', [('Effusion', 1.)], -.1)]),
    ('E2 Syn+=Eff a-.2', [('Synovitis', [('Effusion', 1.)], -.2)]),
]
RES = []
P('\nCANDIDATES (macro e-4; public = population + read noise; P(move) = P(public delta vs pw30 > 0); R1 = E[delta 1(read>0)])')
P('%-60s | %-26s | %-26s | %-26s | %s' % ('candidate', 'EB T1 mean [p10,p90] P>0', 'EB T2 + label shift -.02', 'NO shrink T1 (optimistic)', 'noise pub/priv | Pmove | priv | R1'))
for name, comps in CANDS:
    a1 = evaluate(name, comps, 0., True, 'T1_g58q|population')
    a2 = evaluate(name, comps, .02, True, 'T2_pubgap|population')
    a3 = evaluate(name, comps, 0., False, 'T1_g58q|population')
    nz = noise(comps)
    # decision summary on the primary (EB T1), with the read: delta_pub = delta + e, e ~ N(0, nz390)
    e = rng.standard_normal(len(a1)) * nz[390]; mv = (a1 + e) > 0
    r1 = float(np.mean(a1 * mv))
    row = dict(name=name, eb_t1=[float(np.mean(a1)), *np.percentile(a1, [10, 90]).tolist(), float(np.mean(a1 > 0))],
               eb_t2=[float(np.mean(a2)), *np.percentile(a2, [10, 90]).tolist(), float(np.mean(a2 > 0))],
               raw_t1=[float(np.mean(a3)), *np.percentile(a3, [10, 90]).tolist(), float(np.mean(a3 > 0))],
               n_pub=nz[390], n_priv=nz[910], Pmove=float(mv.mean()), priv=float(np.mean(a1)), R1=r1)
    e3 = rng.standard_normal(len(a3)) * nz[390]; row['Pmove_raw'] = float(np.mean(a3 + e3 > 0)); row['R1_raw'] = float(np.mean(a3 * (a3 + e3 > 0)))
    RES.append(row)
    f = lambda v: '%+5.2f [%+5.2f,%+5.2f] %.2f' % (v[0] * 1e4, v[1] * 1e4, v[2] * 1e4, v[3])
    P('%-60s | %-26s | %-26s | %-26s | %.2f/%.2f | %.2f (raw %.2f) | %+.2f | %+.2f (raw %+.2f)' % (
        name[:60], f(row['eb_t1']), f(row['eb_t2']), f(row['raw_t1']), nz[390] * 1e4, nz[910] * 1e4, row['Pmove'], row['Pmove_raw'],
        row['priv'] * 1e4, r1 * 1e4, row['R1_raw'] * 1e4))

# E-OA self-check through the same evaluate() path (trio components, no shrink): must reproduce the anchor
for nm, mode, a in (('S10', 'sym', .1), ('S20', 'sym', .2), ('PW30', 'pw', .3)):
    comps = [(t, trio_parts(t, mode), a) for t in OA]
    v = evaluate('selfcheck', comps, 0., False, 'T1_g58q|public', Z=Z4)
    P('selfcheck %s through evaluate() on c4 coupling (public anchor): %.2f [%.2f, %.2f]e-4' % (nm, *np.percentile(v, [50, 10, 90]) * 1e4))
json.dump(dict(cands=RES, eb=dict(mu=float(mu), tau_ml=tau_ml), A=A,
               fit={k: dict(s=np.percentile(v['s'], [10, 50, 90]).tolist(), m=np.percentile(v['m'], [10, 50, 90]).tolist(), dq=v['dq']) for k, v in FIT.items()}),
          open(OUTD + 'recal_transfer.json', 'w'), indent=1)
EB.round(4).to_csv(OUTD + 'recal_eb_gaps.csv', index=False)
P('RECAL_TRANSFER DONE')
