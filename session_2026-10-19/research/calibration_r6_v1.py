# calibration_r6.py (10-19, Claude): calibrate the gold58 R6 screen against every LB read we have, then a simple Bayesian
# model for P(public move) of today's five entries (c2max, fr100, c2, pf10, c3max) vs c4 57343435.
# Blend = the kernel's: rr = (1-A) pct(r384) + A pct(r224); b = w_c rr + (1-w_c) pct(ours); final = (1-ws6) pct(b) + ws6 pct(S6fall).
# w_c = table.get(c, W_RAPTOR). pct = average-tie rank / n over the 58 scored studies. Light CPU numpy only.
# Bootstrap: predictions fixed on all 58, studies resampled (multiplicity-weighted exact Mann-Whitney AUC), same resamples for
# every config so paired deltas and the cross-entry covariance are exact on the resample set.
import json, os, numpy as np, pandas as pd
from scipy.stats import rankdata, norm

D = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58'
OUT = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/research'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]
HW = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, "Baker's": 0.35}
C4 = {**HW, 'PF OA': 0.5, "Baker's": 0.5, 'Lateral Meniscus': 0.4, 'Fracture': 0.65}
assert C4 == {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}

g = pd.read_csv(D + "/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); Y = g[L].values.astype(int); n = len(ids)
assert pd.read_csv(D + "/gold58_r224_ids.csv", dtype=str).StudyInstanceUID.tolist() == ids
R6 = pd.read_csv(D + "/R6_gold58_serve.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
R7 = pd.read_csv(D + "/R7OOF.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, ["p|" + l for l in L]].values
R384 = np.load(D + "/gold58_r384_kernelserve_cpu.npy"); R224 = np.load(D + "/gold58_r224_kernelserve.npy")
S6 = pd.read_csv(D + "/n42gpu_gold58.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
assert R6.shape == R7.shape == R384.shape == R224.shape == S6.shape == (n, 12)
OURS = {'R6': R6, 'R7': R7, 'R67': 0.5 * (R6 + R7)}   # R67 = the "R7 late leg": per-study probability mean (cell_r7late_tpl.py l.93)
pct = lambda v: rankdata(v) / len(v)


def final(leg, A=0.35, table=None, ws6=0.2, wr=0.5, ours='X'):
    table = table or {}; O = OURS[leg if ours == 'X' else ours]
    out = np.zeros((n, 12))
    for j, c in enumerate(L):
        rr = (1 - A) * pct(R384[:, j]) + A * pct(R224[:, j]); w = table.get(c, wr)
        b = w * rr + (1 - w) * pct(O[:, j]); out[:, j] = (1 - ws6) * pct(b) + ws6 * pct(S6[:, j])
    return out


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


mac = lambda P: np.array([auc(P[:, j], Y[:, j]) for j in range(12)])
cfg = lambda **kw: kw
CFG = {
    'v0': cfg(A=0.0, ws6=0.0), 'v224': cfg(A=0.35, ws6=0.0), 's6f': cfg(), 's6f_oursR7': cfg(ours='R7'),
    'hw': cfg(table=HW), 'hw3': cfg(table=HW, ws6=0.3), 's615': cfg(table=HW, ws6=0.15), 'wr45': cfg(table=HW, wr=0.45),
    'hwr7': cfg(table=HW, ours='R67'), 'a25': cfg(A=0.25, table=HW), 'c4': cfg(A=0.25, table=C4), 'c4a15': cfg(A=0.15, table=C4),
    'fr65': cfg(A=0.25, table={**HW, 'Fracture': 0.65}), 'pf50': cfg(A=0.25, table={**HW, 'PF OA': 0.5}), 'a15': cfg(A=0.15, table=HW),
    # today (all on c4)
    'c2max': cfg(A=0.25, table={**C4, 'Fracture': 1.0, 'PF OA': 0.1}), 'fr100': cfg(A=0.25, table={**C4, 'Fracture': 1.0}),
    'c2': cfg(A=0.25, table={**C4, 'Fracture': 0.9, 'PF OA': 0.1}), 'pf10': cfg(A=0.25, table={**C4, 'PF OA': 0.1}),
    'c3max': cfg(A=0.25, table={**C4, 'Fracture': 1.0, 'PF OA': 0.1, 'Lateral OA': 0.1}),
    'fr90': cfg(A=0.25, table={**C4, 'Fracture': 0.9}), 'lat10': cfg(A=0.25, table={**C4, 'Lateral OA': 0.1}),
}
# (name, base, entry, LB sign, group, note). LB sign: +1 = entry above base on public, -1 = entry below base.
CAL = [
    ("S6fall .2 entry (uniform, A .35)", 'v224', 's6f', +1, 'core', "LB entry used the pre-R6 ours leg (heliosli R7 bundle)"),
    ("heliosli table (uniform -> HW)", 's6f', 'hw', +1, 'core', "HW 57262115 best-id move vs R6 57248084"),
    ("S6fall .2 -> .3 (hw3)", 'hw', 'hw3', -1, 'core', "57291203 non-move"),
    ("S6fall .2 -> .15 (s615)", 'hw', 's615', -1, 'core', "57313454 non-move"),
    ("W_RAPTOR .5 -> .45 (wr45)", 'hw', 'wr45', -1, 'core', "57310758 non-move"),
    ("A224 .35 -> .25 (a25)", 'hw', 'a25', +1, 'core', "57316395 best-id move"),
    ("c4 table vs a25", 'a25', 'c4', +1, 'core', "57343435 best-id move"),
    ("c4 -> c4a15 (A224 .25 -> .15 on c4)", 'c4', 'c4a15', -1, 'core', "c4a15 below c4 (parallel)"),
    ("c4 -> fr65 (fr65 below c4)", 'c4', 'fr65', -1, 'core', "parallel, censored vs a25; ordering vs c4 known"),
    ("c4 -> pf50 (pf50 below c4)", 'c4', 'pf50', -1, 'core', "parallel, censored vs a25; ordering vs c4 known"),
    ("c4 -> a15 (a15 below c4)", 'c4', 'a15', -1, 'core', "parallel, censored vs a25; ordering vs c4 known"),
    ("V224 lift (r224 .35 into uniform no-S6)", 'v0', 'v224', +1, 'supp', "pre-R6 and pre-S6F base; ours leg was heliosli R7"),
    ("R6 swap (ours R7OOF -> R6, S6F base)", 's6f_oursR7', 's6f', +1, 'supp', "BIASED: R6 = 4-fold held-out ensemble vs R7 single-fold OOF"),
    ("HW + R7 late leg (ours -> mean R6,R7)", 'hw', 'hwr7', -1, 'supp', "UNREAD per A14, practically negative"),
]
TODAY = ['c2max', 'fr100', 'c2', 'pf10', 'c3max']
REF = ['fr90', 'lat10']

# ---------------- bootstrap (fixed predictions) ----------------
B = 6000
rng = np.random.default_rng(20261019)
Wm = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).astype(np.float64)
ok = np.ones(B, bool)
for j in range(12):
    ok &= (Wm[:, Y[:, j] == 1].sum(1) > 0) & (Wm[:, Y[:, j] == 0].sum(1) > 0)
Wm = Wm[ok]


def boot_auc(P):
    A = np.zeros((len(Wm), 12))
    for j in range(12):
        pos, neg = Y[:, j] == 1, Y[:, j] == 0
        Cm = (P[pos, j][:, None] > P[neg, j][None, :]) + 0.5 * (P[pos, j][:, None] == P[neg, j][None, :])
        wp, wn = Wm[:, pos], Wm[:, neg]
        A[:, j] = np.einsum('bi,ij,bj->b', wp, Cm, wn) / (wp.sum(1) * wn.sum(1))
    return A


res = dict(n=n, pos=dict(zip(L, Y.sum(0).tolist())), n_boot=int(len(Wm)), legs={})
cache = {}
for leg in ('R6', 'R7'):
    for k, kw in CFG.items():
        if leg == 'R7' and kw.get('ours') == 'R67': continue
        P = final(leg, **kw); cache[(leg, k)] = (P, mac(P), boot_auc(P))


def delta(leg, b, e):
    Pb, mb, bb = cache[(leg, b)]; Pe, me, be = cache[(leg, e)]
    pl = me - mb; d = (be - bb).mean(1)
    return dict(point=float(pl.mean()), sd=float(d.std()), p_gt0=float((d > 0).mean()), p_eq0=float((d == 0).mean()),
                z=float(pl.mean() / d.std()) if d.std() > 0 else 0.0, per_label=dict(zip(L, pl.round(5).tolist())),
                drivers=[(L[j], round(float(pl[j]), 4)) for j in np.argsort(-np.abs(pl))[:3] if abs(pl[j]) > 0]), d


rows = []
for nm, b, e, s, grp, note in CAL:
    r = dict(step=nm, base=b, entry=e, lb=s, group=grp, note=note)
    for leg in ('R6', 'R7'):
        if (leg, e) not in cache or (leg, b) not in cache or (leg == 'R7' and e == 's6f' and b == 's6f_oursR7'):
            r[leg] = None; continue
        st, _ = delta(leg, b, e); st['match'] = 1.0 if np.sign(st['point']) == s else (0.5 if st['point'] == 0 else 0.0)
        r[leg] = st
    rows.append(r)
res['calibration'] = rows
today = {}
for leg in ('R6', 'R7'):
    today[leg] = {}
    D_ = []
    for k in TODAY + REF:
        st, d = delta(leg, 'c4', k); today[leg][k] = st
        if k in TODAY: D_.append(d)
    D_ = np.array(D_)
    today[leg]['_cov'] = np.cov(D_).tolist(); today[leg]['_corr'] = np.corrcoef(D_).tolist()
res['today'] = today

# ---------------- Bayesian calibration models ----------------
rho = np.sqrt(58 / 390)                      # public ~390 studies vs gold58 58 (MODEL: same per-study noise)
gmax = 1 / np.sqrt(1 + rho ** 2)             # probit slope ceiling: no shift, infinite effect spread
G = np.linspace(0, gmax, 1501)


def m1_post(z, s):
    ll = np.array([np.sum(norm.logcdf(s * gg * z)) for gg in G]); w = np.exp(ll - ll.max()); w /= w.sum()
    marg = np.log(np.mean(np.exp(ll)))       # uniform prior on [0, gmax]
    return w, marg, ll


def m2_post(z, s):                            # intercept a ~ N(0, 1), slope ~ U(0, gmax)
    Aa = np.linspace(-2.5, 2.5, 201); Gg = np.linspace(0, gmax, 301)
    ll = np.array([[np.sum(norm.logcdf(s * (a + gg * z))) for gg in Gg] for a in Aa]) + norm.logpdf(Aa)[:, None]
    w = np.exp(ll - ll.max()); w /= w.sum(); return Aa, Gg, w


def pred_m1(w, zq): return float(np.sum(w * norm.cdf(G * zq)))
def pred_m2(Aa, Gg, w, zq): return float(np.sum(w * norm.cdf(Aa[:, None] + Gg[None, :] * zq)))


bayes = {}
for leg in ('R6', 'R7'):
    for setname, groups in (('core', ('core',)), ('core+supp', ('core', 'supp'))):
        use = [r for r in rows if r['group'] in groups and r[leg] is not None]
        z = np.array([r[leg]['z'] for r in use]); s = np.array([r['lb'] for r in use])
        k = sum(r[leg]['match'] for r in use); m = len(use)
        w, marg, ll = m1_post(z, s)
        gmean = float(np.sum(w * G)); gq = [float(G[np.searchsorted(np.cumsum(w), q)]) for q in (0.05, 0.5, 0.95)]
        # LOO predictive log score of the LB sign (M1) vs coin
        loo = []
        for i in range(m):
            wi, _, _ = m1_post(np.delete(z, i), np.delete(s, i)); p = pred_m1(wi, z[i]); loo.append(p if s[i] > 0 else 1 - p)
        Aa, Gg, w2 = m2_post(z, s)
        ent = dict(m=m, matches=k, beta_post_mean=(k + 1) / (m + 2),
                   beta_q05_q95=[float(x) for x in __import__('scipy.stats', fromlist=['beta']).beta.ppf([0.05, 0.95], k + 1, m - k + 1)],
                   gamma_mean=gmean, gamma_q05_med_q95=gq, gamma_mle=float(G[np.argmax(ll)]), log_bf_vs_null=float(marg - m * np.log(0.5)),
                   loo_mean_logscore=float(np.mean(np.log(loo))), loo_coin_logscore=float(np.log(0.5)), loo_p_obs=[round(x, 3) for x in loo],
                   m2_a_mean=float(np.sum(w2.sum(1) * Aa)), m2_g_mean=float(np.sum(w2.sum(0) * Gg)))
        ent['pred'] = {}
        for kq in TODAY + REF:
            zq = today[leg][kq]['z']
            ent['pred'][kq] = dict(z=zq, M0_sign=(k + 1) / (m + 2) if zq > 0 else 1 - (k + 1) / (m + 2), M1=pred_m1(w, zq),
                                   M2=pred_m2(Aa, Gg, w2, zq))
        # joint for the five entries under M1: p' ~ MVN(gamma * d, Sigma_boot), gamma ~ posterior
        dvec = np.array([today[leg][kq]['point'] for kq in TODAY]); Sig = np.array(today[leg]['_cov'])
        rs = np.random.default_rng(5); NS = 200000
        gs = rs.choice(G, size=NS, p=w)
        Lc = np.linalg.cholesky(Sig + 1e-14 * np.eye(len(TODAY)))
        draws = gs[:, None] * dvec[None, :] + rs.standard_normal((NS, len(TODAY))) @ Lc.T
        anyp = (draws > 0).any(1); am = np.argmax(draws, 1)
        ent['joint'] = dict(p_any_move=float(anyp.mean()), p_all_move=float((draws > 0).all(1).mean()),
                            p_best=dict(zip(TODAY, [float(((am == i) & anyp).mean()) for i in range(len(TODAY))])),
                            p_none=float(1 - anyp.mean()),
                            n_movers_dist=[float(((draws > 0).sum(1) == c).mean()) for c in range(len(TODAY) + 1)])
        bayes['%s|%s' % (leg, setname)] = ent
res['bayes'] = bayes
res['model'] = dict(rho=float(rho), gmax=float(gmax),
                    M0='sign-only Beta(1,1)-Binomial on screen/LB sign agreement',
                    M1='P(LB sign = +) = Phi(gamma z), z = gold58 paired delta / bootstrap sd, gamma ~ U(0, 1/sqrt(1+rho^2))',
                    M2='P = Phi(a + gamma z), a ~ N(0,1), gamma ~ U(0, gmax)')

# ---------------- structural magnitude model (t, beta grid), R6 core only: expected public delta ----------------
use = [r for r in rows if r['group'] == 'core']
z = np.array([r['R6']['z'] for r in use]); s = np.array([r['lb'] for r in use])
T = np.exp(np.linspace(np.log(0.3), np.log(10), 60)); BE = np.linspace(0, 3, 61)
post = np.zeros((len(T), len(BE))); k1m = np.zeros_like(post); gam = np.zeros_like(post)
for a, t in enumerate(T):
    for bb_, be in enumerate(BE):
        k1 = t * t / (t * t + 1 + be * be); k2 = t * t * (1 + be * be) / (t * t + 1 + be * be) + rho ** 2; gg = k1 / np.sqrt(k2)
        post[a, bb_] = np.exp(np.sum(norm.logcdf(s * gg * z))); k1m[a, bb_] = k1; gam[a, bb_] = gg
post /= post.sum()
res['structural'] = dict(E_k1=float((post * k1m).sum()), E_gamma=float((post * gam).sum()),
                         expected_public_delta={kq: float((post * k1m).sum() * today['R6'][kq]['point']) for kq in TODAY + REF})
json.dump(res, open(os.path.join(OUT, 'calibration_r6.json'), 'w'), indent=1)

# ---------------- print ----------------
print('pos per label', res['pos'], '| boot', res['n_boot'])
for leg in ('R6', 'R7'):
    print('c4 macro (%s ours): %.5f' % (leg, cache[(leg, 'c4')][1].mean()))
for r in rows:
    def f(x): return '%+.5f sd %.5f z %+.2f P>0 %.3f %s' % (x['point'], x['sd'], x['z'], x['p_gt0'], {1.0: 'MATCH', 0.0: 'miss', 0.5: 'tie'}[x['match']]) if x else 'n/a'
    print('%-4s %-42s LB %+d | R6 %s | R7 %s | R6 drivers %s' % (r['group'], r['step'], r['lb'], f(r['R6']), f(r['R7']), r['R6']['drivers'] if r['R6'] else ''))
for leg in ('R6', 'R7'):
    for k in TODAY + REF:
        t_ = today[leg][k]; print('TODAY %s %-6s %+.5f sd %.5f z %+.2f P>0 %.3f %s' % (leg, k, t_['point'], t_['sd'], t_['z'], t_['p_gt0'], t_['drivers']))
for key, e in bayes.items():
    print('BAYES %-14s m %d match %.1f | Beta mean %.3f [%.2f,%.2f] | gamma mean %.3f q %s mle %.3f | logBF %.2f | LOO log %.3f (coin %.3f) | M2 a %.2f g %.2f' % (
        key, e['m'], e['matches'], e['beta_post_mean'], *e['beta_q05_q95'], e['gamma_mean'], [round(x, 3) for x in e['gamma_q05_med_q95']],
        e['gamma_mle'], e['log_bf_vs_null'], e['loo_mean_logscore'], e['loo_coin_logscore'], e['m2_a_mean'], e['m2_g_mean']))
    for kq, p in e['pred'].items():
        print('   %-6s z %+.2f  M0 %.3f  M1 %.3f  M2 %.3f' % (kq, p['z'], p['M0_sign'], p['M1'], p['M2']))
    print('   joint', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in e['joint'].items() if k != 'n_movers_dist'},
          'n_movers', [round(x, 3) for x in e['joint']['n_movers_dist']])
print('structural', res['structural'])
print('corr R6 today', np.round(np.array(today['R6']['_corr']), 2).tolist())
