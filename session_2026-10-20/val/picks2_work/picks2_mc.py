# picks2_mc.py (10-20 val Track D; RESEARCH ONLY, local CPU; no OAI rows, no LB writes, nothing submitted).
# Pick model update with A = pw30 57425970 (API best-id since the A21 read, MEASURED val/picks2_work/a21_frozen.log), or A = the R1
# winner if 10-21 adds pw25 / pw35 / pw40. B candidates: R6 57248084, R6pw30 (R6 + the pw .3 block, UNBUILT), R6e20 (built, push/),
# eoa20 57413368, c4 57343435 (+ reference rows eoa10, pw20, eoa30, HW).
# Built on push/bpick_mc.py (same our-side posterior, same field model) with:
#   E-OA family (default 'abc'): posterior draws of the push/work ABC updated with the A21 read (abc_a21.py -> abc_a21_draws_<tag>.npz):
#     population deltas vs c4 (= private expectation) + public deltas for S10..S35, PW20..PW40, and the n-390 public noise covariance C.
#     Private paired E-OA noise = MVN(0, C * 390 / 910) (MODEL: same reader process, random split).
#   Alternative family 'mix' (bpick_mc priors + bpick read model + the A21 likelihood), A = pw30 only.
#   R6 transfers: R6X = R6 + k_X Pop_X + eps_X + private noise (corr RHO with c4's), k_X = k_c + N(0, .05), k_c ~ N(K_MU, K_SD)
#     (gold58 kernel-path ratios sym20 .88 / pw30 .86 / pw20 .81, MEASURED r6transfer_pw.log). R6pw20 = UNBUILT reference row.
# Tags: MEASURED (file named), CLAIMED, MODEL, GUESS. Units: margins = private macro AUC over the heliosli copy block (x1e-4 in prints).
import sys, json, numpy as np
PWK = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/picks_work'
HERE = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/val/picks2_work'
R5 = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/r5'
args = dict(a.split('=') for a in sys.argv[1:])
f = lambda k, d: float(args.get(k, d))
OUR = args.get('our', 'base'); FS = args.get('field', 'base'); FAM = args.get('family', 'abc'); ABC = args.get('abc', 'PFT_N01_logN')
K_MU, K_SD = f('k_mu', 0.86), f('k_sd', 0.15); RHO = f('rho', 0.8); EPS = f('eps', 1e-4); NSC = f('noise', 1.0)
SHAPE = f('shape_sd', 0.0); REP = int(args.get('rep', 2)); TAG = args.get('tag', 'base'); rng = np.random.default_rng(int(args.get('seed', 1020)))
SDP = np.sqrt(390 / 910)

# ---- our side (MODEL; think/picks posterior conditioned on every public read up to 10-19) ----
Z = np.load(f'{PWK}/post_{OUR}_report.npz'); ENT = list(Z['ent']); e = {k: i for i, k in enumerate(ENT)}
PV = Z['priv'].astype(float); M0 = PV.shape[0]; N = M0 * REP; ib = np.tile(np.arange(M0), REP)
X = {k: PV[ib, e[k]] for k in ('c4', 'R6u', 'HW')}

# ---- E-OA family ----
VARS = ['S10', 'S20', 'S25', 'S30', 'S35', 'PW20', 'PW25', 'PW30', 'PW35', 'PW40']; ix = {v: i for i, v in enumerate(VARS)}
if FAM == 'abc':
    D = np.load(f'{HERE}/abc_a21_draws_{ABC}.npz'); assert list(D['VARS']) == VARS
    j = rng.integers(0, D['Pop'].shape[0], N)      # E-OA draws paired at random with our-side draws (independence, MODEL)
    POP = D['Pop'][j].copy(); GP = D['G'][j]; C = D['C']
    if SHAPE > 0:   # GUESS structural spread of the curve shape around eoa20 (the ABC ties every variant to one D5 curve)
        dist = {v: abs(int(v.lstrip('SPW')) / 100 - .2) / .1 + (.5 if v.startswith('PW') else 0) for v in VARS}
        for v in VARS:
            if v != 'S20':
                z = rng.standard_normal(N) * SHAPE * dist[v]; POP[:, ix[v]] += z; GP[:, ix[v]] = GP[:, ix[v]] + z
    LC = np.linalg.cholesky(C + 1e-14 * np.eye(len(VARS)))
    NV = (rng.standard_normal((N, len(VARS))) @ LC.T) * SDP * NSC
    NV2 = (rng.standard_normal((N, len(VARS))) @ LC.T) * SDP * NSC       # independent draw for the R6 versions' unshared part
    PUBN = GP - POP                                                       # the public sample noise of each sim (for R6X display)
    sdpub = np.sqrt(np.diag(C)) * NSC
elif FAM == 'mix':
    # bpick_mc.py E-OA family (prior mix, read noise, d30 / pw GUESS extrapolation), conditioned on the eoa10 / eoa20 in-band reads AND
    # the A21 read; resampled to equal weights. Only S10, S20, S30, PW20, PW30 exist in this family.
    NB = N * 4
    MIXN = ['more', 'ladder', 'orig', 'null', 'neg']; MIXW = np.array([.40, .25, .15, .10, .10])
    def prior_draw(n, P):
        if P == 'mix':
            comp = rng.choice(len(MIXN), n, p=MIXW); out = np.zeros((n, 2))
            for i, nm in enumerate(MIXN):
                kk = comp == i
                if kk.any(): out[kk] = prior_draw(int(kk.sum()), nm)
            return out
        if P == 'ladder':
            Cc = json.load(open(R5 + '/ladder_cells.json')); Cc = [c for c in Cc if c['cpl'] != 'R0']
            W = np.array([{'R1g74': .375, 'tilt1.0': .375, 'tilt1.5': .25}[c['cpl']] for c in Cc]); W /= W.sum()
            jj = rng.choice(len(Cc), n, p=W)
            d1 = np.array([c['d1'] for c in Cc])[jj]; d2 = np.array([c['d2'] for c in Cc])[jj]
            s1 = np.array([c['s1'] for c in Cc])[jj]; s2 = np.array([c['s2'] for c in Cc])[jj]
            dg = rng.normal(.03, .04, n); z = rng.standard_normal(n); u = rng.standard_normal(n)
            return np.stack([d1 + s1 * u + .0039 * dg + 1.5e-4 * z, d2 + s2 * u + .0079 * dg + 3e-4 * z], 1)
        mu, sd = {'more': ([3e-4, 1e-4], [4.5e-4, 9e-4]), 'orig': ([5e-4, 7e-4], [5e-4, 1.1e-3]), 'null': ([0., 0.], [2e-4, 4e-4]),
                  'neg': ([-1.5e-4, -8.4e-4], [2.5e-4, 5e-4])}[P]
        mu, sd, r = np.array(mu), np.array(sd), .95
        L = np.linalg.cholesky(np.array([[sd[0] ** 2, r * sd[0] * sd[1]], [r * sd[0] * sd[1], sd[1] ** 2]]))
        return mu + rng.standard_normal((n, 2)) @ L.T
    D2 = prior_draw(NB, 'mix'); d10, d20 = D2[:, 0], D2[:, 1]
    d30 = 3 * (d20 - d10) + rng.normal(-1.5e-4, 3e-4, NB)
    dl = rng.normal(.3e-4, 1.5e-4, NB); dpw20 = d20 + dl; dpw30 = d30 + 1.5 * dl + rng.normal(0, .5e-4, NB)
    CN = np.array([[1, .96, .93, .95, .92], [.96, 1, .985, .985, .97], [.93, .985, 1, .97, .985], [.95, .985, .97, 1, .985], [.92, .97, .985, .985, 1]])
    LCm = np.linalg.cholesky(CN)
    spub = np.array([2.1e-4, 4.6e-4, 7.3e-4, 4.6e-4, 7.3e-4]) * NSC; spriv = np.array([1.46e-4, 3.15e-4, 4.9e-4, 3.15e-4, 4.9e-4]) * NSC
    NP_ = (rng.standard_normal((NB, 5)) @ LCm.T) * spub
    DD = np.stack([d10, d20, d30, dpw20, dpw30], 1); PUB = DD + NP_
    g = np.abs(rng.normal(1.5e-4, .9e-4, NB)); x10 = PUB[:, 0] - g; x20 = PUB[:, 1] - g
    w = np.exp(-.5 * ((x10 - .40e-3) / .075e-3) ** 2 - .5 * ((x20 - .93e-3) / .03e-3) ** 2) * (x20 < 1e-3) * (x20 > 0)
    gap = 1e-3 - x20; t = (PUB[:, 4] - PUB[:, 1]) / np.maximum(gap, 1e-9)
    w *= np.where((t > 0) & (t < 1), t ** 2 * (1 - t) ** 2, 0) * (np.argmax(PUB[:, 1:], 1) == 3) * (PUB[:, 2] - PUB[:, 1] > -x20) * (PUB[:, 3] - PUB[:, 1] > -x20)
    print('mix family: ESS %.0f of %d' % (w.sum() ** 2 / (w ** 2).sum(), NB))
    jj = rng.choice(NB, N, p=w / w.sum())
    POP = np.full((N, len(VARS)), np.nan); GP = np.full((N, len(VARS)), np.nan); PUBN = np.zeros((N, len(VARS)))
    for k_, v in enumerate(['S10', 'S20', 'S30', 'PW20', 'PW30']):
        POP[:, ix[v]] = DD[jj, k_]; GP[:, ix[v]] = PUB[jj, k_]; PUBN[:, ix[v]] = NP_[jj, k_]
    NV = np.zeros((N, len(VARS))); NV2 = np.zeros((N, len(VARS))); sdpub = np.zeros(len(VARS))
    for k_, v in enumerate(['S10', 'S20', 'S30', 'PW20', 'PW30']): sdpub[ix[v]] = spub[k_]
    A1 = (rng.standard_normal((N, 5)) @ LCm.T) * spriv; A2 = (rng.standard_normal((N, 5)) @ LCm.T) * spriv
    for k_, v in enumerate(['S10', 'S20', 'S30', 'PW20', 'PW30']): NV[:, ix[v]] = A1[:, k_]; NV2[:, ix[v]] = A2[:, k_]
else:
    raise SystemExit('family?')

NAME = {'S10': 'eoa10', 'S20': 'eoa20', 'S25': 'eoa25', 'S30': 'eoa30', 'S35': 'eoa35', 'PW20': 'pw20', 'PW25': 'pw25', 'PW30': 'pw30', 'PW35': 'pw35', 'PW40': 'pw40'}
for v in VARS: X[NAME[v]] = X['c4'] + POP[:, ix[v]] + NV[:, ix[v]]

# ---- R6 versions ----
kc = rng.normal(K_MU, K_SD, N); KX = {v: kc + rng.normal(0, .05, N) for v in ('S20', 'PW30', 'PW20')}   # gold58 ratios .88 / .86 / .81 (MEASURED r6transfer_pw.log): one common k
e_c = rng.standard_normal(N); EPSX = {v: EPS * (.8 * e_c + .6 * rng.standard_normal(N)) for v in KX}
uR6 = np.clip(rng.normal(.10, .07, N), .01, .4)   # R6's public position inside .958 (density.md, CLAIMED in bpick.md)
DISP = {}
for v, nm in (('S20', 'R6e20'), ('PW30', 'R6pw30'), ('PW20', 'R6pw20')):
    gain = KX[v] * POP[:, ix[v]] + EPSX[v]
    X[nm] = X['R6u'] + gain + RHO * NV[:, ix[v]] + np.sqrt(1 - RHO ** 2) * NV2[:, ix[v]]
    pubg = gain + .8 * PUBN[:, ix[v]] + .6 * sdpub[ix[v]] * rng.standard_normal(N)
    d = np.where(pubg > (1 - uR6) * 1e-3, 959, np.where(pubg < -uR6 * 1e-3, 957, 958)); DISP[nm] = d
    X[nm + 'g'] = np.where(d >= 958, X[nm], X['R6u'])      # display guard: fall back to R6 if it shows .957
    X[nm + '_gain'] = gain

# ---- display-conditioned B policies (fixed before any R6X score): B = R6X if its public display is .959, else pw20 (scored, free) ----
for nm in ('R6e20', 'R6pw20', 'R6pw30'):
    X[nm + '|pw20'] = np.where(DISP[nm] == 959, X[nm], X['pw20'])
    X[nm + '>pw20'] = np.where(DISP[nm] >= 958, X[nm], X['pw20'])   # guard variant: fall back to pw20 (not R6) only on .957

# ---- A definitions (R1: A = max public among our entries; pw30 is already above eoa20 / eoa30 / pw20 in every draw) ----
def r1(cands):
    if any(np.isnan(GP[:, ix[c]]).any() for c in cands): return None, None
    win = np.argmax(np.stack([GP[:, ix[c]] for c in cands], 1), 1)
    return np.choose(win, [X[NAME[c]] for c in cands]), win
AOPT = {'pw30': (X['pw30'], None)}
INFO_A = args.get('info_a', '0') == '1'
if FAM == 'abc':
    for nm, cands in (('R1{pw30,pw25}', ['PW30', 'PW25']), ('R1{pw30,pw35}', ['PW30', 'PW35']), ('R1{pw30,pw35,pw40}', ['PW30', 'PW35', 'PW40']),
                      ('R1{pw30,pw25,pw35}', ['PW30', 'PW25', 'PW35'])):
        AOPT[nm] = r1(cands) + (cands,)
if INFO_A:   # NOT R1-compliant; information only (what the pair would be if A were a lower-dose entry)
    for nm in ('pw20', 'eoa20'): AOPT['INFO ' + nm] = (X[nm], None)

CV = np.load(f'{HERE}/curves_{FS}.npz'); grid = CV['grid']
def Sk(k, x): return np.interp(x, grid, CV[k], left=0.0, right=float(CV[k][-1]))
COPY_PUB = .959 - 9.8e-4   # copy block on the public scale (field_mc f959, truncation convention; MODEL, as bpick.md)
KEYS = ('silver', 'bronze', 'r200', 'r167', 'r150', 'r140', 'r130', 'r100', 'gold')
def stats(xm):
    s = dict(E=float(xm.mean()), sd=float(xm.std()), pub_scale=COPY_PUB + float(xm.mean()))
    for k in KEYS: s[k] = float(Sk(k, xm).mean())
    return s
BS = ['R6pw30', 'R6e20', 'R6u', 'eoa20', 'c4', 'pw20', 'eoa10', 'eoa30', 'HW', 'R6pw20', 'R6pw30g', 'R6e20g', 'R6e20|pw20', 'R6pw20|pw20', 'R6pw30|pw20', 'R6e20>pw20']
res = dict(tag=TAG, args=args, N=int(N), singles={}, A={})
for nm in ['pw30', 'eoa20', 'pw20', 'eoa30', 'eoa10', 'c4', 'R6u', 'R6e20', 'R6pw30', 'R6pw20', 'HW'] + (['pw25', 'pw35', 'pw40'] if FAM == 'abc' else []):
    res['singles'][nm] = dict(E=float(X[nm].mean()), sd=float(X[nm].std()), pub_scale=COPY_PUB + float(X[nm].mean()), silver=float(Sk('silver', X[nm]).mean()),
                              r150=float(Sk('r150', X[nm]).mean()))
res['R6X'] = {nm: dict(gain=[float(X[nm + '_gain'].mean()), float(X[nm + '_gain'].std())], P_gt_R6=float((X[nm] > X['R6u']).mean()),
                       disp={str(c): float((DISP[nm] == c).mean()) for c in (957, 958, 959)}) for nm in ('R6e20', 'R6pw30', 'R6pw20')}
for an, val in AOPT.items():
    A = val[0]; st = {'A alone': stats(A)}
    if len(val) > 2 and val[1] is not None:
        st['A alone']['win'] = {NAME[c]: float((val[1] == i).mean()) for i, c in enumerate(val[2])}
    bl = [b for b in BS + (['pw30'] if an != 'pw30' else []) if not an.endswith(' ' + b)]
    for b in bl:
        xm = np.maximum(A, X[b]); s = stats(xm); s['P_B_gt_A'] = float((X[b] > A).mean()); s['dE'] = s['E'] - st['A alone']['E']; st[b] = s
    res['A'][an] = st
json.dump(res, open(f'{HERE}/picks2_{TAG}.json', 'w'), indent=1)

print(f'=== {TAG} | family {FAM} abc {ABC} | our {OUR} field {FS} | k_c N({K_MU},{K_SD}) rho {RHO} noise x{NSC} shape_sd {SHAPE} | N {N}')
print('singles (E margin e-4 | pub-scale | P(silver) | P(r<=150) if entered alone): ' + ' | '.join(
    '%s %+.2f %.5f %.3f %.3f' % (k, v['E'] * 1e4, v['pub_scale'], v['silver'], v['r150']) for k, v in res['singles'].items()))
for nm, v in res['R6X'].items():
    print('  %s gain over R6 %+.2fe-4 sd %.2f | P(> R6) %.3f | public display %s' % (nm, v['gain'][0] * 1e4, v['gain'][1] * 1e4, v['P_gt_R6'], {k: round(x, 3) for k, x in v['disp'].items()}))
for an, st in res['A'].items():
    print('-- A = %s %s' % (an, st['A alone'].get('win', '')))
    for b, s in st.items():
        ex = ' P(B>A) %.3f dE %+.2f' % (s['P_B_gt_A'], s['dE'] * 1e4) if 'P_B_gt_A' in s else ''
        print('   %-9s E %+6.2f (pub %.5f) silver %.4f bronze %.4f r200 %.3f r167 %.3f r150 %.3f r140 %.3f r130 %.3f r100 %.3f gold %.3f%s' % (
            b, s['E'] * 1e4, s['pub_scale'], s['silver'], s['bronze'], s['r200'], s['r167'], s['r150'], s['r140'], s['r130'], s['r100'], s['gold'], ex))
