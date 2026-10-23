# bpick_mc.py (10-20 push, Track 3 B pick; CPU only, research only; no OAI data, no LB writes).
# Pick B for the final pair now that E-OA is LB-confirmed (eoa10 57413380 .959 r167, then eoa20 57413368 .959 r135 = API best-id).
# Built on r5/decision_mc.py (same our-side posterior, same E-OA prior family, same read noise, same field curves), with:
#   1. MAGNITUDE-conditioned read (MODEL): the in-band positions of eoa10 / eoa20 inside .959 (MEASURED api_lb_1020_eoaread1/2.json:
#      eoa10 r167 = 29 of 65 others below it in r131-196; eoa20 r135 = 61 below) plus c4's gap under .959 (picks / density ruler)
#      give a Gaussian measurement of the public paired deltas (pub10, pub20) vs c4. Importance weights on prior x noise draws.
#   2. New B candidates: R6e20 = R6 57248084 + E-OA a .2 (NOT BUILT; transfer k x d20, k from gold58 kernel-path replay + structure),
#      plus a display-guard policy (B = R6e20 unless its 3-dp public display is .957, then R6).
#   3. A21 module: eoa30 / pw20 / pw30 (in flight) simulated jointly, A = R1 winner over {eoa20, eoa30, pw20, pw30}.
#   4. Rank thresholds r130 / r167 (.960+ public band today / projected at the deadline) from field_curves960.py.
# Tags: MEASURED (file named), CLAIMED, MODEL, GUESS. Units: margins = private macro AUC over the heliosli copy block (x1e-4 in prints).
import sys, json, numpy as np
R5 = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/r5'
PW = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/picks_work'
OUTD = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push'
args = dict(a.split('=') for a in sys.argv[1:])
f = lambda k, d: float(args.get(k, d))
OUR = args.get('our', 'base'); FS = args.get('field', 'base'); PRIOR = args.get('prior', 'mix'); NSC = f('noise', 1.0)
K_MU, K_SD = f('k_mu', 0.9), f('k_sd', 0.15); RHO = f('rho', 0.8); EPS = f('eps', 1e-4)
G_MU, G_SD = f('g_mu', 1.5e-4), f('g_sd', 0.9e-4); U10, U10S = f('u10', 0.40), f('u10s', 0.075); U20, U20S = f('u20', 0.93), f('u20s', 0.03)
D30_SHIFT, D30_SD = f('d30shift', -1.5e-4), f('d30sd', 3e-4); PW_MU, PW_SD = f('pw_mu', 0.3e-4), f('pw_sd', 1.5e-4)
REP = int(args.get('rep', 6)); TAG = args.get('tag', 'base'); rng = np.random.default_rng(int(args.get('seed', 21)))

# ---- our side (MODEL, think/picks posterior conditioned on every public read up to 10-19) ----
Z = np.load(f'{PW}/post_{OUR}_report.npz'); ENT = list(Z['ent']); e = {k: i for i, k in enumerate(ENT)}
PV = Z['priv'].astype(float); M0 = PV.shape[0]
N = M0 * REP; ib = np.tile(np.arange(M0), REP)
X = {k: PV[ib, e[k]] for k in ('c4', 'R6u', 'HW', 'a25', 'S6F')}
S6FD = f('s6fdefl', 1.0)
if S6FD != 1.0:   # deflate S6F's idiosyncratic private noise vs R6u (R7 OOF single-fold stand-in overstates it; r5/decision_mc.py s6fdefl)
    TV = Z['true'].astype(float); nS = PV[ib, e['S6F']] - TV[ib, e['S6F']]; nR = PV[ib, e['R6u']] - TV[ib, e['R6u']]
    X['S6F'] = TV[ib, e['S6F']] + nR + S6FD * (nS - nR)

# ---- E-OA prior on (d10, d20) (verbatim families from r5/decision_mc.py) ----
MIXN = ['more', 'ladder', 'orig', 'null', 'neg']; MIXW = np.array([.40, .25, .15, .10, .10])
def prior_draw(n, P):
    if P == 'mix':
        comp = rng.choice(len(MIXN), n, p=MIXW); out = np.zeros((n, 2))
        for i, nm in enumerate(MIXN):
            kk = comp == i
            if kk.any(): out[kk] = prior_draw(int(kk.sum()), nm)[0]
        return out, comp
    if P == 'ladder':
        C = json.load(open(R5 + '/ladder_cells.json')); C = [c for c in C if c['cpl'] != 'R0']
        W = np.array([{'R1g74': .375, 'tilt1.0': .375, 'tilt1.5': .25}[c['cpl']] for c in C]); W /= W.sum()
        j = rng.choice(len(C), n, p=W)
        d1 = np.array([c['d1'] for c in C])[j]; d2 = np.array([c['d2'] for c in C])[j]
        s1 = np.array([c['s1'] for c in C])[j]; s2 = np.array([c['s2'] for c in C])[j]
        dg = rng.normal(.03, .04, n); z = rng.standard_normal(n); u = rng.standard_normal(n)
        return np.stack([d1 + s1 * u + .0039 * dg + 1.5e-4 * z, d2 + s2 * u + .0079 * dg + 3e-4 * z], 1), None
    mu, sd = {'more': ([3e-4, 1e-4], [4.5e-4, 9e-4]), 'orig': ([5e-4, 7e-4], [5e-4, 1.1e-3]), 'null': ([0., 0.], [2e-4, 4e-4]),
              'neg': ([-1.5e-4, -8.4e-4], [2.5e-4, 5e-4])}[P]
    mu, sd, r = np.array(mu), np.array(sd), .95
    L = np.linalg.cholesky(np.array([[sd[0] ** 2, r * sd[0] * sd[1]], [r * sd[0] * sd[1], sd[1] ** 2]]))
    return mu + rng.standard_normal((n, 2)) @ L.T, None
D2, COMP = prior_draw(N, PRIOR)
if COMP is None and PRIOR == 'mix': raise SystemExit('mix lost comp')
d10, d20 = D2[:, 0], D2[:, 1]
# A21 doses (GUESS): quadratic-through-0 extrapolation d30 = 3 (d20 - d10), shifted down (gold58 kernel-path dose curve peaks near .2:
# c4 +.00142 / +.00247 / +.00201 at .1 / .2 / .3, MEASURED r6transfer_g58.log) with structural spread; pair weights = small shape term.
d30 = 3 * (d20 - d10) + rng.normal(D30_SHIFT, D30_SD, N)
dl = rng.normal(PW_MU, PW_SD, N); dpw20 = d20 + dl; dpw30 = d30 + 1.5 * dl + rng.normal(0, .5e-4, N)

# ---- read noise (MEASURED sd / corr at a .1 / .2 on report OOF, r5/noise_corr.log; a .3 by the measured power law; pw ~ same dose) ----
CN = np.array([[1, .96, .93, .95, .92], [.96, 1, .985, .985, .97], [.93, .985, 1, .97, .985], [.95, .985, .97, 1, .985], [.92, .97, .985, .985, 1]])
LC = np.linalg.cholesky(CN)
spub = np.array([2.1e-4, 4.6e-4, 7.3e-4, 4.6e-4, 7.3e-4]) * NSC; spriv = np.array([1.46e-4, 3.15e-4, 4.9e-4, 3.15e-4, 4.9e-4]) * NSC
NP_ = (rng.standard_normal((N, 5)) @ LC.T) * spub; NV = (rng.standard_normal((N, 5)) @ LC.T) * spriv
DD = np.stack([d10, d20, d30, dpw20, dpw30], 1); PUB = DD + NP_
for j, nm in enumerate(['eoa10', 'eoa20', 'eoa30', 'pw20', 'pw30']): X[nm] = X['c4'] + DD[:, j] + NV[:, j]

# ---- R6e20 (HYPOTHETICAL, NOT BUILT): R6 57248084 + the same 9 E-OA lines at a .2 ----
# transfer k ~ N(.9, .15): gold58 kernel path ratio .88 (boot corr .75; MEASURED r6transfer_g58.log), OA columns of the two entries
# agree at Spearman .989-.9996 (same members; only Lateral OA W .5 vs .2 and A224 .35 vs .25 differ); private E-OA noise corr RHO.
kk = rng.normal(K_MU, K_SD, N); eps = rng.normal(0, EPS, N)
nv_r6e = RHO * NV[:, 1] + np.sqrt(1 - RHO ** 2) * spriv[1] * rng.standard_normal(N)
X['R6e20'] = X['R6u'] + kk * d20 + eps + nv_r6e
# 3-dp public display of R6e20 (MODEL): R6 sits at u .10 (p10-p90 .03-.23) of .958 (density.md, truncation convention)
uR6 = np.clip(rng.normal(.10, .07, N), .01, .4)
dpub_r6e = kk * d20 + eps + .8 * NP_[:, 1] + .6 * spub[1] * rng.standard_normal(N)
disp = np.where(dpub_r6e > (1 - uR6) * 1e-3, 959, np.where(dpub_r6e < -uR6 * 1e-3, 957, 958))
X['R6e20g'] = np.where(disp >= 958, X['R6e20'], X['R6u'])     # policy: B = R6e20 unless it displays .957, then R6

# ---- measurement of (pub10, pub20) from the in-band positions (MODEL) ----
m = np.array([G_MU + U10 * 1e-3, G_MU + U20 * 1e-3])
S = np.array([[G_SD ** 2 + (U10S * 1e-3) ** 2, G_SD ** 2], [G_SD ** 2, G_SD ** 2 + (U20S * 1e-3) ** 2]])
Si = np.linalg.inv(S); r = PUB[:, :2] - m
w = np.exp(-0.5 * np.einsum('ni,ij,nj->n', r, Si, r)); w /= w.sum()
ESS = 1.0 / (w ** 2).sum()

# ---- A21 R1 winner over {eoa20 (observed), eoa30, pw20, pw30} ----
P4 = np.stack([PUB[:, 1], PUB[:, 2], PUB[:, 3], PUB[:, 4]], 1); NM4 = ['eoa20', 'eoa30', 'pw20', 'pw30']; win = P4.argmax(1)
X['A21'] = np.choose(win, [X['eoa20'], X['eoa30'], X['pw20'], X['pw30']])
X['eoa20_ifnotA'] = np.where(win == 0, X['eoa10'], X['eoa20'])   # B = eoa20 when A moved; eoa10 when A stayed eoa20

CV = np.load(f'{OUTD}/curves_{FS}.npz'); grid = CV['grid']
def Sk(k, x): return np.interp(x, grid, CV[k], left=0.0, right=float(CV[k][-1]))
wm = lambda x: float((w * x).sum())
wsd = lambda x: float(np.sqrt(max(wm(x * x) - wm(x) ** 2, 0)))
COPY_PUB = .959 - 9.8e-4   # copy block on the public scale (field_mc f959, truncation convention; MODEL)

def stats(xm):
    return dict(E=wm(xm), sd=wsd(xm), pub_scale=COPY_PUB + wm(xm), silver=wm(Sk('silver', xm)), bronze=wm(Sk('bronze', xm)),
                r200=wm(Sk('r200', xm)), r167=wm(Sk('r167', xm)), r130=wm(Sk('r130', xm)), r100=wm(Sk('r100', xm)), gold=wm(Sk('gold', xm)), abs960=wm((xm >= .960 - COPY_PUB).astype(float)))
res = dict(tag=TAG, args=args, N=int(N), ESS=float(ESS), post={}, A_eoa20={}, A_A21={}, A21_outcomes={})
post = res['post']
for nm, v in (('d10', d10), ('d20', d20), ('d30', d30), ('dpw20', dpw20), ('dpw30', dpw30), ('pub10', PUB[:, 0]), ('pub20', PUB[:, 1]), ('k', kk)):
    post[nm] = [wm(v), wsd(v), wm((v > 0).astype(float))]
post['P_d20_gt_d10'] = wm((d20 > d10).astype(float)); post['P_d30_gt_d20'] = wm((d30 > d20).astype(float))
post['R6e20_disp'] = {str(c): wm((disp == c).astype(float)) for c in (957, 958, 959)}
post['R6e20_true_gain_over_R6'] = [wm(kk * d20 + eps), wsd(kk * d20 + eps)]
if COMP is not None: post['prior_weights'] = {nm: wm((COMP == i).astype(float)) for i, nm in enumerate(MIXN)}
post['A21_win'] = {nm: wm((win == i).astype(float)) for i, nm in enumerate(NM4)}

BS = ['R6u', 'R6e20', 'R6e20g', 'c4', 'eoa10', 'HW', 'S6F', 'a25']
for Aname, store in (('eoa20', res['A_eoa20']), ('A21', res['A_A21'])):
    A = X[Aname]; store['A alone'] = stats(A)
    for b in BS + (['eoa20_ifnotA'] if Aname == 'A21' else []):
        xm = np.maximum(A, X[b]); s = stats(xm)
        s['P_B_gt_A'] = wm((X[b] > A).astype(float)); s['E_gain'] = s['E'] - store['A alone']['E']
        store[b] = s
store = res['A21_outcomes']
for i, nm in enumerate(NM4):
    sel = win == i; pw_ = wm(sel.astype(float))
    if pw_ < 1e-3: continue
    ws = w * sel; ws = ws / ws.sum(); A = X[nm]; row = {'P': pw_}
    for b in ['R6u', 'R6e20', 'c4', 'eoa10'] + (['eoa20'] if nm != 'eoa20' else []):
        xm = np.maximum(A, X[b]); row[b] = dict(silver=float((ws * Sk('silver', xm)).sum()), r167=float((ws * Sk('r167', xm)).sum()),
                                                  E=float((ws * xm).sum()))
    row['d_A'] = float((ws * (A - X['c4'])).sum()); store[nm] = row

json.dump(res, open(f'{OUTD}/bpick_{TAG}.json', 'w'), indent=1)
# ---- print ----
print(f'=== {TAG} | our {OUR} field {FS} prior {PRIOR} noise x{NSC} k N({K_MU},{K_SD}) rho {RHO} | N {N} ESS {ESS:.0f}')
print('posterior: ' + ' | '.join(f'{k} {v[0]*1e4:+.2f}e-4 sd {v[1]*1e4:.2f} P>0 {v[2]:.2f}' for k, v in post.items() if k in ('d10', 'd20', 'd30', 'dpw20', 'pub10', 'pub20')))
print(f'  k {post["k"][0]:.2f} | P(d20>d10) {post["P_d20_gt_d10"]:.3f} P(d30>d20) {post["P_d30_gt_d20"]:.2f} | R6e20 gain over R6 '
      f'{post["R6e20_true_gain_over_R6"][0]*1e4:+.2f} sd {post["R6e20_true_gain_over_R6"][1]*1e4:.2f} | R6e20 display {post["R6e20_disp"]}')
if 'prior_weights' in post: print('  prior weights', {k: round(v, 3) for k, v in post['prior_weights'].items()})
print('  A21 R1 winner', {k: round(v, 3) for k, v in post['A21_win'].items()})
for Aname, store in (('eoa20', res['A_eoa20']), ('A21', res['A_A21'])):
    print(f'-- A = {Aname}')
    for b, s in store.items():
        extra = f' P(B>A) {s["P_B_gt_A"]:.3f} dE {s["E_gain"]*1e4:+.2f}' if 'P_B_gt_A' in s else ''
        print(f'   {b:13s} E {s["E"]*1e4:+6.2f}e-4 (pub-scale {s["pub_scale"]:.5f}) silver {s["silver"]:.4f} bronze+ {s["bronze"]:.4f} '
              f'r200 {s["r200"]:.3f} r167 {s["r167"]:.3f} r130 {s["r130"]:.3f} r100 {s["r100"]:.3f} abs960 {s["abs960"]:.3f}' + extra)
print('-- by A21 outcome (A = winner):')
for nm, row in res['A21_outcomes'].items():
    print(f'   {nm} P {row["P"]:.3f} dA {row["d_A"]*1e4:+.2f}e-4 | ' + ' '.join(f'{b} {v["silver"]:.4f}/{v["r167"]:.3f}' for b, v in row.items() if isinstance(v, dict)))
