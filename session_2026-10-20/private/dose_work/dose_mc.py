# dose_mc.py (10-20 ~22:15 IST, PRIVATE track, Claude). RESEARCH ONLY, local CPU, no OAI rows, no box, no LB writes, nothing submitted.
# Question: private-optimal A dose (eoa20 / pw20 / pw25 / pw30) and the best (A, B) pair, under private label-coupling scenarios.
# Inputs (all already on disk, read only):
#   - val/work/box/perlabel_{PFT,PAREA}.log : box AGGREGATES (per-label D5 mean curves per regime at n 390 + pooled noise cov) (implied2 Track B)
#   - val/work/lb_post2.npz                 : public-LB posterior draws (implied2)
#   - 10-19 think/picks_work/post_base_report.npz : our-side private posterior (c4, R6u, HW) (picks.md)
#   - val/picks2_work/curves_<field>.npz    : field model curves incl. erank (picks2)
# Model = val/work/abc3.py (implied2 primary: O1 kernel on eoa10/eoa20, O2 frozen A21 ruler, O3 R1 argmax, O4 .959 floors), re-run here with
# the per-draw regime (sh), amplitude (s) and c4 co-firing (rc4) kept, so the PRIVATE population can be re-evaluated at a shifted label regime:
#   private regime = sh + d (d in sd of gold74 log OR; + = private labels MORE coupled), c4 gap = rc4 - q(sh + d), curve at the D5 regime with
#   that gap (same mapping as abc3 / abc_local2).  SITE scenario (GUESS): d ~ N(0, .75), rc4 + N(0, .03), s x logN(0, .30), E-OA private
#   noise x1.25, all independent of the public fit (public / private site-split with unseen sites).
# Pick layer = val/picks2_work/picks2_mc.py conventions (R6 transfer k ~ N(.86, .15), rho .8, R6pw20 display guard, field curves).
# Tags in the report: MODEL (this file), CLAIMED (sources), GUESS (scenario parameters).
import sys, os, json, numpy as np
VW = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/val/work'
sys.path.insert(0, VW)
import abc3 as B3
PWK = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/picks_work'
P2 = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/val/picks2_work'
OUT = os.path.dirname(os.path.abspath(__file__))
args = dict(a.split('=', 1) for a in sys.argv[1:])
g = lambda k, d: args.get(k, d)
PROX = g('prox', 'PFT'); PSH = g('psh', 'N01'); PS = g('ps', 'logN'); M_ = int(g('M', 600000)); NREP = int(g('rep', 2))
FS = g('field', 'base'); FSHIFT = float(g('fshift', 0.0)) * 1e-4    # field drift: others gain FSHIFT (= our margin - FSHIFT)
K_MU, K_SD = float(g('k_mu', .86)), float(g('k_sd', .15)); RHO = float(g('rho', .8)); EPS = 1e-4
TAG = g('tag', 'base'); SEED = int(g('seed', 11))
SITE_D, SITE_RC, SITE_S, SITE_N = float(g('site_d', .75)), float(g('site_rc', .03)), float(g('site_s', .30)), float(g('site_n', 1.25))
SH = ['S10', 'S20', 'S30', 'PW20', 'PW25', 'PW30', 'PW35']; LAB = B3.LAB
NAME = {'S10': 'eoa10', 'S20': 'eoa20', 'S30': 'eoa30', 'PW20': 'pw20', 'PW25': 'pw25', 'PW30': 'pw30', 'PW35': 'pw35'}


def fit(seed=SEED, h=2e-5, nlb=400, rho_mu=.816):
    rng = np.random.default_rng(seed)
    E = B3.load(PROX); VN = E['VN']; m = 'D5'
    regs = sorted([k.split('|', 1)[1] for k in E['mean'] if k.startswith(m + '|')], key=lambda r: B3.SHX[r])
    xs = np.array([B3.SHX[r] for r in regs]); Mu = np.array([E['mean'][m + '|' + r] for r in regs]); C = np.array(E['cov'][m])
    iv = {v: i for i, v in enumerate(VN)}
    LBP = np.load(VW + '/lb_post2.npz'); J = rng.choice(len(LBP['G10']), nlb, replace=False)
    LG10, LG20, Lx20, Lk = [LBP[k][J] for k in ('G10', 'G20', 'x20', 'k')]; LG10 = LG10 * 1e-3; LG20 = LG20 * 1e-3
    sh = np.clip(rng.normal(0, 1, M_), -2.5, 2.0) if PSH == 'N01' else rng.uniform(-2.5, 2.0, M_)
    s = np.exp(rng.normal(0, .35, M_)) if PS == 'logN' else np.exp(rng.uniform(np.log(.4), np.log(2.5), M_))
    rc4 = rng.normal(rho_mu, .035, M_)
    gapD = Mu[:, iv['rho']] - Mu[:, iv['q']]; o = np.argsort(gapD)

    def sheq(shx, rcx):
        q_h = np.interp(shx, xs, Mu[:, iv['q']]); return np.interp(rcx - q_h, gapD[o], xs[o], left=xs[0] - 1.0, right=xs[-1] + .5)

    def curve(col, x):
        y = np.interp(x, xs, Mu[:, col]); lo = x < xs[0]; hi = x > xs[-1]
        y[lo] = Mu[0, col] + (x[lo] - xs[0]) * (Mu[1, col] - Mu[0, col]) / (xs[1] - xs[0])
        y[hi] = Mu[-1, col] + (x[hi] - xs[-1]) * (Mu[-1, col] - Mu[-2, col]) / (xs[-1] - xs[-2])
        return y
    cols = [iv['%s|%s' % (l, t)] for t in SH for l in LAB]
    Cs = C[np.ix_(cols, cols)]; Ls = np.linalg.cholesky(Cs + 1e-14 * np.eye(len(cols)))
    def popf(shx, rcx, sx):
        se = sheq(shx, rcx); return {t: sx * sum(curve(iv['%s|%s' % (l, t)], se) for l in LAB) for t in SH}
    NZ = rng.standard_normal((M_, len(cols))) @ Ls.T
    P0 = popf(sh, rc4, s)
    G = {t: P0[t] + NZ[:, [3 * k_ + j for j in range(3)]].sum(1) for k_, t in enumerate(SH)}
    del NZ
    ok3 = (G['PW30'] > G['S20']) & (G['PW30'] > G['PW20']) & (G['PW30'] > G['S30']) & (G['PW30'] > G['S10'])
    e0 = np.exp(-Lk * Lx20)[None]; e1 = np.exp(-Lk)[None]
    w = np.zeros(M_); jj = np.zeros(M_, int)
    for a in range(0, M_, 20000):
        b = min(M_, a + 20000); sl = slice(a, b)
        K = np.exp(-.5 * (((G['S10'][sl, None] - LG10[None]) / h) ** 2 + ((G['S20'][sl, None] - LG20[None]) / h) ** 2))
        xp = Lx20[None] + (G['PW30'][sl, None] - G['S20'][sl, None]) * 1e3
        Gq = (e0 - np.exp(-Lk[None] * xp)) / (e0 - e1); inside = (xp > Lx20[None]) & (xp < 1)
        K = K * np.where(inside, 6 * Gq ** 2 * (1 - Gq) ** 2, 0.)
        for v in ('PW20', 'S30'):
            K = K * ((Lx20[None] + (G[v][sl, None] - G['S20'][sl, None]) * 1e3) >= 0)
        K = K * ok3[sl, None]; w[sl] = K.sum(1)
        cum = np.cumsum(K, 1); u = rng.random(b - a) * np.maximum(cum[:, -1], 1e-300)
        jj[sl] = np.minimum((cum < u[:, None]).sum(1), nlb - 1)
    ess = w.sum() ** 2 / (w ** 2).sum()
    return dict(sh=sh, s=s, rc4=rc4, w=w, x20=Lx20[jj], G=G, P0=P0, popf=popf, Ls=Ls, ess=ess, sdpub={t: float(np.sqrt(Cs[3 * k_:3 * k_ + 3, 3 * k_:3 * k_ + 3].sum())) for k_, t in enumerate(SH)})


def main():
    F = fit(); rng = np.random.default_rng(SEED + 7)
    Z = np.load(PWK + '/post_base_report.npz'); ENT = list(Z['ent']); e = {k: i for i, k in enumerate(ENT)}
    PV = Z['priv'].astype(float); M0 = PV.shape[0]; N = M0 * NREP; ib = np.tile(np.arange(M0), NREP)
    OUR = {k: PV[ib, e[k]] for k in ('c4', 'R6u', 'HW')}
    j = rng.choice(len(F['w']), N, p=F['w'] / F['w'].sum())
    sh, s, rc4 = F['sh'][j], F['s'][j], F['rc4'][j]; G = {t: F['G'][t][j] for t in SH}; P0 = {t: F['P0'][t][j] for t in SH}
    x20 = F['x20'][j]
    SDP = np.sqrt(390 / 910)
    nc = F['Ls'].shape[0]
    # shared private-sample noise (one realisation of the 910 private studies per sim), reused across coupling scenarios (paired)
    Z1 = (rng.standard_normal((N, nc)) @ F['Ls'].T) * SDP; Z2 = (rng.standard_normal((N, nc)) @ F['Ls'].T) * SDP
    NP = {t: Z1[:, 3 * k_:3 * k_ + 3].sum(1) for k_, t in enumerate(SH)}; NV2 = {t: Z2[:, 3 * k_:3 * k_ + 3].sum(1) for k_, t in enumerate(SH)}
    del Z1, Z2
    # R6 transfer + display (as picks2_mc.py)
    kc = rng.normal(K_MU, K_SD, N) + rng.normal(0, .05, N); epsx = EPS * rng.standard_normal(N)
    uR6 = np.clip(rng.normal(.10, .07, N), .01, .4)
    pubg = kc * P0['PW20'] + epsx + .8 * (G['PW20'] - P0['PW20']) + .6 * F['sdpub']['PW20'] * rng.standard_normal(N)
    disp = np.where(pubg > (1 - uR6) * 1e-3, 959, np.where(pubg < -uR6 * 1e-3, 957, 958))
    r1win = G['PW25'] > G['PW30']             # R1 between pw30 and pw25 on the realised public noise
    # site-scenario perturbations (independent of the public fit)
    dS = rng.normal(0, SITE_D, N); rcS = rng.normal(0, SITE_RC, N); sS = np.exp(rng.normal(0, SITE_S, N))
    SCEN = {'S0 same': dict(d=0.0), 'C+ .5 more': dict(d=.5), 'C- .5 less': dict(d=-.5), 'C++ 1.0 more': dict(d=1.0),
            'SITE held-out': dict(d=dS, rc=rcS, sf=sS, nsc=SITE_N)}
    CV = np.load(P2 + '/curves_%s.npz' % FS); grid = CV['grid']
    Sk = lambda k, x: np.interp(x - FSHIFT, grid, CV[k], left=float(CV[k][0]), right=float(CV[k][-1]))
    COPY_PUB = .959 - 9.8e-4
    res = dict(tag=TAG, args=args, ess=float(F['ess']), N=int(N), scen={})
    for sn, sc in SCEN.items():
        pop = F['popf'](sh + sc['d'], rc4 + sc.get('rc', 0.0), s * sc.get('sf', 1.0)); nsc = sc.get('nsc', 1.0)
        X = {'c4': OUR['c4'], 'R6': OUR['R6u'], 'HW': OUR['HW']}
        for t in SH:
            X[NAME[t]] = OUR['c4'] + pop[t] + nsc * NP[t]
        r6g = kc * pop['PW20'] + epsx
        X['R6pw20raw'] = OUR['R6u'] + r6g + nsc * (RHO * NP['PW20'] + np.sqrt(1 - RHO ** 2) * NV2['PW20'])
        X['R6pw20'] = np.where(disp >= 958, X['R6pw20raw'], X['pw20'])      # VAL_1020 rule: fall back to pw20 on a .957 display
        X['R1{pw30,pw25}'] = np.where(r1win, X['pw25'], X['pw30'])
        ENTS = ['c4', 'R6', 'HW', 'eoa10', 'eoa20', 'eoa30', 'pw20', 'pw25', 'pw30', 'pw35', 'R6pw20', 'R1{pw30,pw25}']
        def met(x):
            q = np.percentile(x, [10, 50, 90])
            return dict(E=float(x.mean()), sd=float(x.std()), p10=float(q[0]), p50=float(q[1]), p90=float(q[2]), pub=COPY_PUB + float(x.mean()),
                        silver=float(Sk('silver', x).mean()), bronze=float(Sk('bronze', x).mean()), r200=float(Sk('r200', x).mean()),
                        r150=float(Sk('r150', x).mean()), r130=float(Sk('r130', x).mean()), erank=float(Sk('erank', x).mean()),
                        mrank=float(np.median(Sk('erank', x))))
        R = dict(singles={}, vs_pw30={}, pairs={}, pop_vs_pw30={})
        for en in ENTS:
            R['singles'][en] = met(X[en])
            dd = X[en] - X['pw30']; R['vs_pw30'][en] = dict(E=float(dd.mean()), p10=float(np.percentile(dd, 10)), p90=float(np.percentile(dd, 90)), P=float((dd > 0).mean()))
        for t in SH:
            dd = pop[t] - pop['PW30']; R['pop_vs_pw30'][NAME[t]] = dict(E=float(dd.mean()), P=float((dd > 0).mean()), sd=float(dd.std()))
        R['pop_vs_c4'] = {NAME[t]: dict(E=float(pop[t].mean()), p10=float(np.percentile(pop[t], 10)), p90=float(np.percentile(pop[t], 90))) for t in SH}
        R['argmax_dose'] = {NAME[t]: float(v) for t, v in zip(['S20', 'PW20', 'PW25', 'PW30', 'PW35'],
                            np.bincount(np.argmax(np.stack([pop[t] for t in ['S20', 'PW20', 'PW25', 'PW30', 'PW35']], 1), 1), minlength=5) / N)}
        for i, a in enumerate(ENTS):
            for b in ENTS[i + 1:]:
                R['pairs'][a + ' + ' + b] = met(np.maximum(X[a], X[b]))
        res['scen'][sn] = R
    json.dump(res, open(OUT + '/dose_%s.json' % TAG, 'w'), indent=1)
    print('=== %s | prox %s sh %s s %s | field %s shift %+.1fe-4 | k N(%.2f, %.2f) | ESS %.0f | N %d' % (TAG, PROX, PSH, PS, FS, FSHIFT * 1e4, K_MU, K_SD, F['ess'], N))
    print('    posterior sh %.2f [%.2f %.2f] | s %.2f | public pw30 - eoa20 %+.2fe-4 | pub pw25 - pw30 %+.2fe-4 P(R1 picks pw25) %.2f | R6pw20 display 957/958/959 %.3f/%.3f/%.3f' % (
        sh.mean(), *np.percentile(sh, [10, 90]), s.mean(), (G['PW30'] - G['S20']).mean() * 1e4, (G['PW25'] - G['PW30']).mean() * 1e4, r1win.mean(),
        (disp == 957).mean(), (disp == 958).mean(), (disp == 959).mean()))
    for sn, R in res['scen'].items():
        print('--- scenario %s | population argmax share %s' % (sn, {k: round(v, 2) for k, v in R['argmax_dose'].items()}))
        print('    pop vs pw30 (e-4, P>): ' + ' | '.join('%s %+.2f (%.2f)' % (k, v['E'] * 1e4, v['P']) for k, v in R['pop_vs_pw30'].items() if k != 'pw30'))
        for en in ('eoa20', 'pw20', 'pw25', 'pw30', 'R1{pw30,pw25}', 'R6pw20', 'R6', 'c4'):
            m_, v = R['singles'][en], R['vs_pw30'][en]
            print('    A=%-14s E %+6.2f sd %5.2f [%+6.2f %+6.2f] priv %.5f | vs pw30 %+5.2f [%+5.2f %+5.2f] P %.2f | silver %.3f r150 %.3f E[rank] %5.1f med %5.1f' % (
                en, m_['E'] * 1e4, m_['sd'] * 1e4, m_['p10'] * 1e4, m_['p90'] * 1e4, m_['pub'], v['E'] * 1e4, v['p10'] * 1e4, v['p90'] * 1e4, v['P'],
                m_['silver'], m_['r150'], m_['erank'], m_['mrank']))
        PR = R['pairs']
        for key in ('silver', 'r150', 'erank'):
            o = sorted(PR.items(), key=lambda kv: (kv[1][key] if key != 'erank' else -kv[1][key]), reverse=True)[:6]
            print('    top by %-6s: ' % key + ' | '.join('%s %.3f/%.3f/%.1f' % (k, v['silver'], v['r150'], v['erank']) for k, v in o))
        for p in ('pw30 + R6pw20', 'pw20 + R6pw20', 'eoa20 + R6pw20', 'pw25 + R6pw20', 'R6pw20 + R1{pw30,pw25}', 'R6 + pw30', 'pw20 + pw30',
                  'R6 + pw20', 'eoa20 + pw30', 'HW + pw30', 'HW + pw20'):
            k = p if p in PR else ' + '.join(p.split(' + ')[::-1])
            v = PR[k]
            print('    PAIR %-24s E %+6.2f silver %.3f r150 %.3f r130 %.3f E[rank] %5.1f' % (p, v['E'] * 1e4, v['silver'], v['r150'], v['r130'], v['erank']))


if __name__ == '__main__':
    main()
