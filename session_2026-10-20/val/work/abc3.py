# abc3.py (val 10-20, Claude): refit of the public-implied model (push/work/abc_local2.py) with ALL five E-OA reads + the A21 board.
# MODEL. Uses ONLY box aggregates (box/perlabel_{PFT,PAREA}.log: per-label mean curves per (model, regime) at n 390 and the pooled
# within-regime noise covariance) and public LB counts (lb_post2.npz). No OAI row is used or needed.
# Observations (all on the public LB):
#   O1 eoa10 - c4, eoa20 - c4: Gaussian kernel (h 2e-5) on the pooled LB draws (lb_post2.npz; = push lb_post.py model)
#   O2 pw30 - eoa20: frozen cross-board ruler, P(order | x_pw30) = 6 G^2 (1 - G)^2 with the draw's own x20, k (lb_post2.py)
#   O3 R1 argmax: pw30 > eoa20, pw20, eoa30 (and eoa10)          O4 pw20, eoa30 display .959: x20 + inc >= 0 (band floor)
# Population curve per label = s_lab * D5 per-label curve at the regime whose D5 gap equals c4's gap (as abc_local2.py);
# public sample = population + joint per-label noise (box covariance); private = population + noise x sqrt(390/910).
import json, sys, numpy as np
W = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/val/work/'
SHX = {'raw': -2.5}
SHX.update({'R3p%+.1f' % s: s for s in (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)})
LAB = ['M', 'L', 'P']


def load(prox):
    s = open(W + 'box/perlabel_%s.log' % prox).read()
    return json.loads(s.split('JSON_BEGIN')[1].split('JSON_END')[0])


def shape(name):
    # transform name -> per-label column (None = untouched)
    if name.startswith('LPW'):
        a = name[3:]; return {'M': None, 'L': 'PW' + a, 'P': 'PW' + a}
    if name.startswith('LP'):
        a = name[2:]; return {'M': None, 'L': 'S' + a, 'P': 'S' + a}
    if name.startswith('M') and '_' in name:          # e.g. M20_LP30: Medial sym .2, Lateral + PF sym .3
        m_, r_ = name.split('_'); return {'M': 'S' + m_[1:], 'L': 'S' + r_[2:], 'P': 'S' + r_[2:]}
    return {l: name for l in LAB}


OBS = ['S10', 'S20', 'S30', 'PW20', 'PW30']
CAND = ['S25', 'S35', 'S40', 'S45', 'PW25', 'PW35', 'PW40', 'PW45', 'PW50', 'LP20', 'LP25', 'LP30', 'LP35', 'LP40',
        'LPW30', 'LPW40', 'M20_LP30']
ALLT = OBS + CAND


def wquant(x, w, tot, ps=(.1, .5, .9)):
    o_ = np.argsort(x); cw = np.cumsum(w[o_]) / tot
    return [float(x[o_][min(len(x) - 1, np.searchsorted(cw, p))]) for p in ps]


def run(prox='PFT', m='D5', prior_sh='N01', prior_s='logN', M_=240000, tnoise=1.0, rho_mu=.816, h=2e-5, medfree=0.0,
        use=('O1', 'O2', 'O3', 'O4'), seed=11, nlb=400, quiet=False, tag=''):
    rng = np.random.default_rng(seed)
    E = load(prox); VN = E['VN']
    regs = sorted([k.split('|', 1)[1] for k in E['mean'] if k.startswith(m + '|')], key=lambda r: SHX[r])
    xs = np.array([SHX[r] for r in regs]); Mu = np.array([E['mean'][m + '|' + r] for r in regs]); C = np.array(E['cov'][m])
    iv = {v: i for i, v in enumerate(VN)}; nd = len(VN) - 3
    LBP = np.load(W + 'lb_post2.npz'); J = rng.choice(len(LBP['G10']), nlb, replace=False)
    LG10, LG20, Lx20, Lk, Lx10, Lw = [LBP[k][J] for k in ('G10', 'G20', 'x20', 'k', 'x10', 'w')]
    LG10 = LG10 * 1e-3; LG20 = LG20 * 1e-3
    lo_sh, hi_sh = -2.5, 2.0
    sh = np.clip(rng.normal(0, 1, M_), lo_sh, hi_sh) if prior_sh == 'N01' else rng.uniform(lo_sh, hi_sh, M_)
    s = np.exp(rng.normal(0, .35, M_)) if prior_s == 'logN' else np.exp(rng.uniform(np.log(.4), np.log(2.5), M_))
    sM = s * np.exp(rng.normal(0, medfree, M_)) if medfree > 0 else s
    rc4 = rng.normal(rho_mu, .035, M_)
    q_h = np.interp(sh, xs, Mu[:, iv['q']]); gap_c4 = rc4 - q_h
    gapD = Mu[:, iv['rho']] - Mu[:, iv['q']]; o = np.argsort(gapD)
    sh_eq = np.interp(gap_c4, gapD[o], xs[o], left=xs[0] - 1.0, right=xs[-1] + .5)

    def curve(col, x):
        y = np.interp(x, xs, Mu[:, col]); lo = x < xs[0]; hi = x > xs[-1]
        y[lo] = Mu[0, col] + (x[lo] - xs[0]) * (Mu[1, col] - Mu[0, col]) / (xs[1] - xs[0])
        y[hi] = Mu[-1, col] + (x[hi] - xs[-1]) * (Mu[-1, col] - Mu[-2, col]) / (xs[-1] - xs[-2])
        return y
    L = np.linalg.cholesky(C[:nd, :nd] + 1e-14 * np.eye(nd))
    Pop, G, Prv, G2 = {}, {}, {}, {}
    NZ = (rng.standard_normal((M_, nd)) @ L.T) * tnoise                        # realised public sample noise
    NP = (rng.standard_normal((M_, nd)) @ L.T) * tnoise * np.sqrt(390 / 910)    # private sample noise
    N2 = (rng.standard_normal((M_, nd)) @ L.T) * tnoise                        # fresh public replicate (winner's curse)
    for t in ALLT:
        sp = shape(t); pop = np.zeros(M_); nz = np.zeros(M_); npv = np.zeros(M_); n2 = np.zeros(M_)
        for l in LAB:
            if sp[l] is None:
                continue
            c = iv['%s|%s' % (l, sp[l])]; amp = sM if l == 'M' else s
            pop += amp * curve(c, sh_eq); nz += NZ[:, c]; npv += NP[:, c]; n2 += N2[:, c]
        Pop[t] = pop; G[t] = pop + nz; Prv[t] = pop + npv; G2[t] = pop + n2
    del NZ, NP, N2
    # weights: sum over LB draws of kernel x frozen-ruler likelihood x display floor (x argmax)
    w = np.zeros(M_); jj = np.zeros(M_, int)
    ok3 = np.ones(M_, bool)
    if 'O3' in use:
        ok3 = (G['PW30'] > G['S20']) & (G['PW30'] > G['PW20']) & (G['PW30'] > G['S30']) & (G['PW30'] > G['S10'])
    e0 = np.exp(-Lk * Lx20)[None]; e1 = np.exp(-Lk)[None]
    for a in range(0, M_, 20000):
        b = min(M_, a + 20000); sl = slice(a, b)
        K = np.ones((b - a, nlb))
        if 'O1' in use:
            K = np.exp(-.5 * (((G['S10'][sl, None] - LG10[None]) / h) ** 2 + ((G['S20'][sl, None] - LG20[None]) / h) ** 2))
        if 'O2' in use:
            xp = Lx20[None] + (G['PW30'][sl, None] - G['S20'][sl, None]) * 1e3
            Gq = (e0 - np.exp(-Lk[None] * xp)) / (e0 - e1); inside = (xp > Lx20[None]) & (xp < 1)
            K = K * np.where(inside, 6 * Gq ** 2 * (1 - Gq) ** 2, 0.)
        if 'O4' in use:
            for v in ('PW20', 'S30'):
                K = K * ((Lx20[None] + (G[v][sl, None] - G['S20'][sl, None]) * 1e3) >= 0)
        if 'O5' in use:   # sensitivity: 16 newcomers into .959 since 05:53, 1 above pw30, 15 below, iid on the band density
            xp = np.clip(Lx20[None] + (G['PW30'][sl, None] - G['S20'][sl, None]) * 1e3, 1e-6, 1 - 1e-6)
            GA = (1 - np.exp(-Lk[None] * xp)) / (1 - np.exp(-Lk[None]))
            K = K * 16 * (1 - GA) * GA ** 15
        K = K * ok3[sl, None]
        w[sl] = K.sum(1)
        cum = np.cumsum(K, 1); u = rng.random(b - a) * np.maximum(cum[:, -1], 1e-300)
        jj[sl] = np.minimum((cum < u[:, None]).sum(1), nlb - 1)
    tot = w.sum(); ess = tot ** 2 / (w ** 2).sum()
    x20 = Lx20[jj]; x10 = Lx10[jj]; uc4 = 1 - Lw[jj]
    mean = lambda x: float((w * x).sum() / tot)
    wq = lambda x: wquant(x, w, tot)
    R = dict(tag=tag, ess=float(ess), prox=prox, m=m, prior_sh=prior_sh, prior_s=prior_s, tnoise=tnoise, rho_mu=rho_mu,
             medfree=medfree, use=list(use))
    R['sh'] = (mean(sh), wq(sh)); R['s'] = (mean(s), wq(s)); R['gap_c4'] = (mean(gap_c4), wq(gap_c4)); R['q'] = (mean(q_h), wq(q_h))
    ph = np.interp(sh, xs, Mu[:, iv['phi']]); R['phi'] = (mean(ph), wq(ph))
    if medfree > 0:
        R['sM_over_s'] = (mean(sM / s), wq(sM / s))
    # public unrounded positions (band units of .959 for E-OA entries; c4 = its .958-band position minus 1)
    pub = {'c4': uc4 - 1, 'eoa10': x10, 'eoa20': x20}
    for e, v in (('eoa30', 'S30'), ('pw20', 'PW20'), ('pw30', 'PW30')):
        pub[e] = x20 + (G[v] - G['S20']) * 1e3
    R['pub_x'] = {e: (mean(v), wq(v)) for e, v in pub.items()}
    R['pub_inc_vs_pw30'] = {e: (mean(v - pub['pw30']), wq(v - pub['pw30'])) for e, v in pub.items()}
    R['pub_inc_vs_eoa20'] = {e: (mean(v - pub['eoa20']), wq(v - pub['eoa20'])) for e, v in pub.items()}
    R['pub_abs_trunc'] = {e: (mean(.959 + v / 1e3), wq(.959 + v / 1e3)) for e, v in pub.items()}
    # pre-read calibration: probability of the observed outcome class (pw30 argmax of the A21 four, every A21 entry below .960)
    allx = np.stack([x20 + (G[v] - G['S20']) * 1e3 for v in ('S30', 'PW20', 'PW30')], 1)
    am = (G['PW30'] > G['S20']) & (G['PW30'] > G['PW20']) & (G['PW30'] > G['S30']) & (G['PW30'] > G['S10'])
    R['P_outcome_class'] = mean(am & (allx.max(1) < 1) & (allx.min(1) >= 0))
    R['pw30_minus_eoa20_pct_of_read'] = (mean((G['PW30'] - G['S20']) * 1e3 <= .05), mean((G['PW30'] - G['S20']) * 1e3 <= .10))
    # 10-21 slates under R1 (pick = public argmax of pw30 + the slate, conditional on the realised public noise)
    SL = {}
    for nm, S_ in (('PW25', ['PW25']), ('M20_LP30', ['M20_LP30']), ('PW25 + M20_LP30', ['PW25', 'M20_LP30']),
                   ('PW25 + PW35', ['PW25', 'PW35']), ('PW25 + LP30', ['PW25', 'LP30']),
                   ('asked 5: PW35 PW40 PW45 S40 LP30', ['PW35', 'PW40', 'PW45', 'S40', 'LP30']),
                   ('PW25 M20_LP30 LP30 LPW30 PW35', ['PW25', 'M20_LP30', 'LP30', 'LPW30', 'PW35']),
                   ('PW35 + PW40', ['PW35', 'PW40'])):
        T_ = ['PW30'] + S_
        Gm = np.stack([G[v] for v in T_], 1); Vm = np.stack([Prv[v] for v in T_], 1); Pm = np.stack([Pop[v] for v in T_], 1)
        bi = np.argmax(Gm, 1); r_ = np.arange(M_); xb = x20 + (Gm[r_, bi] - G['S20']) * 1e3
        SL[nm] = dict(P_move=mean(bi > 0), d_pub=(mean(Gm[r_, bi] - G['PW30']), wq(Gm[r_, bi] - G['PW30'])),
                      d_pop=(mean(Pm[r_, bi] - Pop['PW30']), wq(Pm[r_, bi] - Pop['PW30'])),
                      d_prv=(mean(Vm[r_, bi] - Prv['PW30']), wq(Vm[r_, bi] - Prv['PW30'])), P_prv_up=mean(Vm[r_, bi] > Prv['PW30']),
                      P_prv_down=mean(Vm[r_, bi] < Prv['PW30']), P960=mean(xb >= 1), d_pop_oracle=mean(Pm.max(1) - Pop['PW30']))
    R['slates'] = SL
    R['_arr'] = dict(w=w, x20=x20, G=G, Pop=Pop)
    R['pw30_to_960'] = (mean(1 - pub['pw30']), wq(1 - pub['pw30']))
    R['P_order'] = {'eoa30>eoa20': mean(G['S30'] > G['S20']), 'pw20>eoa20': mean(G['PW20'] > G['S20']),
                    'pw20>eoa30': mean(G['PW20'] > G['S30'])}
    # candidates: public increment over pw30 (conditional on the realised public noise), display, population / private
    C_ = {}
    for v in ['S20', 'S30', 'PW20'] + CAND:
        dpub = G[v] - G['PW30']; xe = x20 + (G[v] - G['S20']) * 1e3
        dpop = Pop[v] - Pop['PW30']; dprv = Prv[v] - Prv['PW30']
        C_[v] = dict(pub=(mean(dpub), wq(dpub)), P_gt=mean(dpub > 0), P960=mean(xe >= 1), P961=mean(xe >= 2), P958=mean(xe < 0),
                     pop=(mean(dpop), wq(dpop)), prv=(mean(dprv), wq(dprv)), P_prv_gt=mean(dprv > 0),
                     pop_vs_c4=mean(Pop[v]), pop_vs_eoa20=mean(Pop[v] - Pop['S20']))
    C_['PW30'] = dict(pop_vs_c4=(mean(Pop['PW30']), wq(Pop['PW30'])),
                      pop_vs_eoa20=(mean(Pop['PW30'] - Pop['S20']), wq(Pop['PW30'] - Pop['S20'])),
                      P_prv_gt_c4=mean(Prv['PW30'] > 0), P_prv_gt_eoa20=mean(Prv['PW30'] > Prv['S20']),
                      realised_curse_vs_c4=mean(G['PW30'] - Pop['PW30']),
                      realised_curse_vs_eoa20=mean((G['PW30'] - G['S20']) - (Pop['PW30'] - Pop['S20'])))
    R['cand'] = C_
    # winner's curse (posterior predictive: fresh public replicate N2 + private draw); pick the public argmax of a set
    WC = {}
    for nm, S_ in (('A21 set 5', ['S10', 'S20', 'S30', 'PW20', 'PW30']),
                   ('set 7 (+PW25, PW35)', ['S10', 'S20', 'S30', 'PW20', 'PW30', 'PW25', 'PW35']),
                   ('set 7 (+PW40, LP30)', ['S10', 'S20', 'S30', 'PW20', 'PW30', 'PW40', 'LP30']),
                   ('set 3 (S20, PW20, PW30)', ['S20', 'PW20', 'PW30'])):
        Gm = np.stack([G2[v] for v in S_], 1); Pm = np.stack([Pop[v] for v in S_], 1); Vm = np.stack([Prv[v] for v in S_], 1)
        bi = np.argmax(Gm, 1); r_ = np.arange(M_)
        curse = Gm[r_, bi] - Pm[r_, bi]
        regret = Pm.max(1) - Pm[r_, bi]
        pbest = np.argmax(Pm, 1)
        WC[nm] = dict(curse=(mean(curse), wq(curse)), fixed_entry=mean(np.mean(Gm - Pm, 1)), regret=(mean(regret), wq(regret)),
                      P_pick_is_popbest=mean(bi == pbest), win_share={v: mean(bi == i) for i, v in enumerate(S_)},
                      E_priv_win_vs_c4=mean(Vm[r_, bi]), E_pub_win_vs_c4=mean(Gm[r_, bi]),
                      E_pop_best_vs_c4=mean(Pm.max(1)), E_pop_fixed_PW30=mean(Pop['PW30']), E_pop_fixed_PW20=mean(Pop['PW20']))
    R['wc'] = WC
    if not quiet:
        report(R)
    return R


def report(R):
    f = lambda t: '%+.2f [%+.2f, %+.2f]' % (t[0] * 1e4, t[1][0] * 1e4, t[1][2] * 1e4)
    print('== %s | prox %s | model %s | sh %s | s %s | noise x%.2f | rho_c4 %.3f | medfree %.2f | use %s | ESS %.0f' % (
        R['tag'], R['prox'], R['m'], R['prior_sh'], R['prior_s'], R['tnoise'], R['rho_mu'], R['medfree'], '+'.join(R['use']), R['ess']))
    for k in ('sh', 's', 'gap_c4', 'q', 'phi') + (('sM_over_s',) if 'sM_over_s' in R else ()):
        print('   %-9s mean %.3f p10/50/90 %s' % (k, R[k][0], np.round(R[k][1], 3).tolist()))
    print('   public position (band units; E-OA in .959, c4 = .958-band pos - 1):',
          ' | '.join('%s %.3f [%.3f, %.3f]' % (e, v[0], v[1][0], v[1][2]) for e, v in R['pub_x'].items()))
    print('   public vs pw30 (e-4):', ' | '.join('%s %+.2f [%+.2f, %+.2f]' % (e, v[0] * 10, v[1][0] * 10, v[1][2] * 10)
                                              for e, v in R['pub_inc_vs_pw30'].items() if e != 'pw30'))
    print('   public vs eoa20 (e-4):', ' | '.join('%s %+.2f [%+.2f, %+.2f]' % (e, v[0] * 10, v[1][0] * 10, v[1][2] * 10)
                                               for e, v in R['pub_inc_vs_eoa20'].items() if e != 'eoa20'))
    print('   unrounded public if display truncates (rounding: -.0005):', ' | '.join('%s %.5f [%.5f, %.5f]' % (e, v[0], v[1][0], v[1][2]) for e, v in R['pub_abs_trunc'].items()))
    print('   P(observed outcome class: pw30 argmax, all A21 in .959) %.3f | P(pw30 - eoa20 <= +0.5e-4) %.2f, <= +1.0e-4 %.2f' % ((R['P_outcome_class'],) + tuple(R['pw30_minus_eoa20_pct_of_read'])))
    print('   pw30 to .960 (e-4): %.2f [%.2f, %.2f] | order P: %s' % (
        R['pw30_to_960'][0] * 10, R['pw30_to_960'][1][0] * 10, R['pw30_to_960'][1][2] * 10, {k: round(v, 2) for k, v in R['P_order'].items()}))
    for v, d in R['cand'].items():
        if v == 'PW30':
            continue
        print('   %-8s PUB vs pw30 %s e-4 | P(>pw30) %.2f | P(.960) %.2f P(.961) %.2f P(.958) %.2f || POP vs pw30 %s | PRIV vs pw30 %s P(>) %.2f | pop vs c4 %+.2f vs eoa20 %+.2f' % (
            v, f(d['pub']), d['P_gt'], d['P960'], d['P961'], d['P958'], f(d['pop']), f(d['prv']), d['P_prv_gt'],
            d['pop_vs_c4'] * 1e4, d['pop_vs_eoa20'] * 1e4))
    p = R['cand']['PW30']
    print('   PW30 pop vs c4 %s | vs eoa20 %s | P(priv > c4) %.2f | P(priv > eoa20) %.2f | realised public overstatement vs c4 %+.2f, vs eoa20 %+.2f e-4' % (
        f(p['pop_vs_c4']), f(p['pop_vs_eoa20']), p['P_prv_gt_c4'], p['P_prv_gt_eoa20'], p['realised_curse_vs_c4'] * 1e4,
        p['realised_curse_vs_eoa20'] * 1e4))
    for nm, d in R['wc'].items():
        print('   WC %-24s curse %s (fixed entry %+.2f) | regret %s | P(pick=pop-best) %.2f | E pub(win) %+.2f priv(win) %+.2f pop-best %+.2f fixed PW30 %+.2f PW20 %+.2f | wins %s' % (
            nm, f(d['curse']), d['fixed_entry'] * 1e4, f(d['regret']), d['P_pick_is_popbest'], d['E_pub_win_vs_c4'] * 1e4,
            d['E_priv_win_vs_c4'] * 1e4, d['E_pop_best_vs_c4'] * 1e4, d['E_pop_fixed_PW30'] * 1e4, d['E_pop_fixed_PW20'] * 1e4,
            {k: round(v, 2) for k, v in d['win_share'].items()}))
    for nm, d in R['slates'].items():
        print('   SLATE %-34s P(best-id moves) %.2f | public gain of best-id %s | POP of pick vs pw30 %s (oracle %+.2f) | PRIVATE of pick vs pw30 %s P(up) %.2f P(down) %.2f | P(best shows .960) %.2f' % (
            nm, d['P_move'], f(d['d_pub']), f(d['d_pop']), d['d_pop_oracle'] * 1e4, f(d['d_prv']), d['P_prv_up'], d['P_prv_down'], d['P960']))
    sys.stdout.flush()


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'primary'
    RES = {}
    if mode == 'primary':
        RES['pre-read check (O1 only)'] = run(use=('O1',), tag='pre-read check (O1 only)', M_=300000)
        RES['primary'] = run(tag='primary', M_=600000)
    elif mode == 'variants':
        V = [dict(tag='flat sh', prior_sh='flat'), dict(tag='logU s', prior_s='logU'),
             dict(tag='flat+logU', prior_sh='flat', prior_s='logU'),
             dict(tag='P_AREA', prox='PAREA'), dict(tag='P_AREA flat+logU', prox='PAREA', prior_sh='flat', prior_s='logU'),
             dict(tag='noise x1.25', tnoise=1.25), dict(tag='noise x0.8', tnoise=.8), dict(tag='rho_c4 .78', rho_mu=.78),
             dict(tag='model s1044', m='s1044'), dict(tag='Medial amp free (sd .5)', medfree=.5),
             dict(tag='no O2 (argmax + floor only)', use=('O1', 'O3', 'O4')),
             dict(tag='+ newcomers (O5)', use=('O1', 'O2', 'O3', 'O4', 'O5')),
             dict(tag='seed 2 (MC check)', seed=29)]
        for kw in V:
            RES[kw['tag']] = run(M_=360000, **kw)
    for k in RES: RES[k].pop('_arr', None)
    json.dump(RES, open(W + 'abc3_%s.json' % mode, 'w'), indent=1, default=float)
