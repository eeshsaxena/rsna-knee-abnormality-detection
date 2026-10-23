# skeptic_val.py (10-20 ~20:00 IST, val SKEPTIC, Claude). LOCAL ONLY. Inputs are the box AGGREGATES already on disk in
# val/box_oaigrid_{PFT,PAREA}.log (per-regime population curves, OAI-size bootstrap covariance, public-size (n 390) resample
# mean + covariance, likelihood sums). No OAI id, row or per-knee value is read or produced here; no ssh.
# What it does:
#  0. Gaussian re-implementation of the Track A ABC on the 5 observed coordinates (S10, S20, S30, PW20, PW30), validated
#     against the logged P(order) and L_joint per regime.
#  1. Likelihood decomposition and prior / amplitude-prior / A21-ruler sensitivity of the regime posterior and of a*.
#  2. Population (private-relevant) differences vs pw30 for the proposed entries, with OAI-size knee noise.
#  3. Coupling-shift stress: private regime = posterior regime + d (d = -.5, +.5, +1.0 sd of gold74 log OR).
#  4. Posterior-predictive checks on contrasts the fit did not use (pw30 - eoa20 magnitude; pw/sym double difference).
# Usage: py -3.12 -I skeptic_val.py > skeptic_val.log
import json, os, re, ast
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); VAL = os.path.dirname(HERE)
H = 4e-5
SG0 = np.exp(np.linspace(np.log(.5), np.log(2.), 15))
SGW = np.exp(np.linspace(np.log(.2), np.log(4.), 27))


def load(fn):
    t = open(os.path.join(VAL, fn), encoding='utf-8').read().splitlines()
    return json.loads(t[t.index('JSON_BEGIN') + 1])


def lbd():
    src = open(os.path.join(VAL, 'box_oaigrid.py'), encoding='utf-8').read()
    m = re.search(r'^LBD = np\.array\((\[\[.*?\]\])\)', src, re.M)
    return np.array(ast.literal_eval(m.group(1)))


LB = lbd(); G10, G20, GAP = LB[:, 0], LB[:, 1], LB[:, 2]


def shift(rn):
    return float(rn[3:]) if rn.startswith('R3p') else None


def like(Z, SG, ruler=False, use_order=True, use_nod=True, use_mag=True):
    """Z: (J, 5) public draws of (S10, S20, S30, PW20, PW30), macro units. Returns (L, s posterior weights)."""
    S10, S20, S30, PW20, PW30 = Z.T
    ordj = np.ones(len(Z))
    if use_order:
        ordj = ((S10 > 0) & (S20 > S10) & (PW30 > S20) & (PW30 > PW20) & (PW30 > S30)).astype(float)
    Ls = np.zeros(len(SG))
    for j, s in enumerate(SG):
        if use_mag:
            a = (s * S10[:, None] - G10[None]) / H; b = (s * S20[:, None] - G20[None]) / H
            K = np.exp(-.5 * (a * a + b * b))
        else:
            K = np.ones((len(Z), len(G10)))
        d = s * (PW30 - S20)[:, None] / GAP[None]
        if ruler:
            g = np.where((d > 0) & (d < 1), 6 * d * d * (1 - d) ** 2 * 1.0, 0.)   # Beta(3,3) density: 2 of 4 frozen teams above
            # Beta(3,3) pdf = 30 t^2 (1-t)^2; constant irrelevant for posteriors
        elif use_nod:
            g = (d < 1).astype(float)
        else:
            g = np.ones_like(d)
        Ls[j] = (K * g * ordj[:, None]).mean()
    return Ls.mean(), Ls / max(Ls.sum(), 1e-300)


def main():
    rng = np.random.default_rng(20261020)
    for P, fn in (('P_FT', 'box_oaigrid_PFT.log'), ('P_AREA', 'box_oaigrid_PAREA.log')):
        E = load(fn); FN = E['FN']; AG = np.array(E['AG']); NA = len(AG); R = E['reg']
        ia = lambda a: int(np.argmin(np.abs(AG - a)))
        ix = lambda f, a: FN.index(f) * NA + ia(a)
        OBS = [ix('sym', .1), ix('sym', .2), ix('sym', .3), ix('pw', .2), ix('pw', .3)]
        rs = sorted([r for r in R if r.startswith('R3p')], key=shift); sh = np.array([shift(r) for r in rs])
        mids = np.concatenate([[sh[0] - (sh[1] - sh[0]) / 2], (sh[1:] + sh[:-1]) / 2, [sh[-1] + (sh[-1] - sh[-2]) / 2]]); wid = np.diff(mids)
        print('=' * 110); print('PROXY', P); print('=' * 110)
        # ---------- 0. Gaussian ABC reproduction ----------
        Zs = {}; LIK = {}
        print('\n-- 0. Gaussian re-implementation vs logged (J 20000 per regime)')
        print('   regime     P(order) log / gauss | L_joint log / gauss | L ruler(s[.5,2]) | L joint s[.2,4] | L ruler s[.2,4] | L mag only | L order only | E[s] joint / ruler')
        for rn in rs + ['raw', 'MIX0.25', 'MIX0.50', 'R1g74']:
            pb = R[rn]['pub']; mu = np.array(pb['mean']).reshape(-1)[OBS]; C = np.array(pb['cov'])[np.ix_(OBS, OBS)] * 1e-8
            Z = rng.multivariate_normal(mu, C, 20000); Zs[rn] = Z
            S10, S20, S30, PW20, PW30 = Z.T
            po = ((S10 > 0) & (S20 > S10) & (PW30 > S20) & (PW30 > PW20) & (PW30 > S30)).mean()
            Lj, wj = like(Z, SG0); Lr, wr = like(Z, SG0, ruler=True); Ljw, _ = like(Z, SGW); Lrw, wrw = like(Z, SGW, ruler=True)
            Lm, _ = like(Z, SG0, use_order=False, use_nod=False); Lo, _ = like(Z, SG0, use_mag=False, use_nod=False)
            LIK[rn] = dict(Lj=Lj, Lr=Lr, Ljw=Ljw, Lrw=Lrw, Lm=Lm, Lo=Lo, sj=float(wj @ SG0), sr=float(wr @ SG0), srw=float(wrw @ SGW))
            print('   %-9s  %.3f / %.3f   | %.2e / %.2e | %.2e | %.2e | %.2e | %.2e | %.3f | %.2f / %.2f (wide ruler %.2f)' % (
                rn, pb['p_order_all'], po, pb['L_joint'], Lj, Lr, Ljw, Lrw, Lm, Lo, LIK[rn]['sj'], LIK[rn]['sr'], LIK[rn]['srw']))
        # ---------- 1. posteriors ----------
        pops = {rn: np.array(R[rn]['pop']) for rn in R}            # (4, 28) macro units
        pw = FN.index('pw')

        def argmax_of(rn, f='pw'):
            p = pops[rn][FN.index(f)]
            return float(AG[np.argmax(p)]) if p.max() > 0 else 0.

        def post(prior, key):
            L = np.array([LIK[r][key] for r in rs]); p = prior * wid * L; return p / p.sum()

        priors = {'N01': np.exp(-.5 * sh ** 2), 'flat': np.ones_like(sh), 'N(-1.2,1)': np.exp(-.5 * (sh + 1.2) ** 2)}
        print('\n-- 1. regime posterior and pw optimum a* (population argmax per regime, mixed over the posterior)')
        print('   prior      likelihood         k mean (sd) | P(a*pw): <=.20  .225  .25  .275  .30  >=.325 >=.35 | mixed-curve argmax pw / sym | E[s]')
        VARS = []
        for pn, pr in priors.items():
            for ln, key in (('joint (Track A)', 'Lj'), ('joint + A21 ruler', 'Lr'), ('joint, s[.2,4]', 'Ljw'), ('ruler, s[.2,4]', 'Lrw'),
                            ('mag only', 'Lm'), ('order only', 'Lo')):
                p = post(pr, key); km = float(p @ sh); ks = float(np.sqrt(p @ (sh - km) ** 2))
                am = np.array([argmax_of(r) for r in rs])
                mix = sum(p[i] * pops[r] for i, r in enumerate(rs))
                amx_pw = float(AG[np.argmax(mix[pw])]); amx_sym = float(AG[np.argmax(mix[FN.index('sym')])])
                sk = 'sr' if key in ('Lr',) else ('srw' if key == 'Lrw' else 'sj')
                sbar = float(sum(p[i] * LIK[r][sk] for i, r in enumerate(rs)))
                bins = [(am <= .2001), np.isclose(am, .225), np.isclose(am, .25), np.isclose(am, .275), np.isclose(am, .3), am >= .3249, am >= .3499]
                print('   %-10s %-18s %+.2f (%.2f) | %s | %.3f / %.3f | %.2f' % (pn, ln, km, ks, '  '.join('%.2f' % float(p @ b) for b in bins),
                                                                            amx_pw, amx_sym, sbar))
                VARS.append((pn, ln, p, sbar))
        # ---------- 2. population differences vs pw30 ----------
        CANDS = [('eoa20', 'sym', .2), ('pw20', 'pw', .2), ('pw25', 'pw', .25), ('pw35', 'pw', .35), ('pw40', 'pw', .4), ('pw45', 'pw', .45),
                 ('eoa40', 'sym', .4), ('LP2 .20', 'LP2', .2), ('LP2 .30', 'LP2', .3), ('LPpw .30', 'LPpw', .3), ('LPpw .40', 'LPpw', .4)]
        print('\n-- 2. population (private-relevant) X - pw30, public units e-4 (x E[s]); P(X > pw30) over regime posterior x OAI-size knee noise')
        sel = [v for v in VARS if v[1] in ('joint (Track A)', 'joint + A21 ruler', 'ruler, s[.2,4]')]
        print('   ' + ' | '.join('%-30s' % (v[0] + ' ' + v[1]) for v in sel))
        for nm, f, a in CANDS:
            cells = []
            for pn, ln, p, sbar in sel:
                d_all = []; wts = []
                for i, r in enumerate(rs):
                    if p[i] < 1e-4:
                        continue
                    bk = R[r]['boot']['knee']; Cb = np.array(bk['cov']) * 1e-8; mu = pops[r].reshape(-1)
                    jj = [ix(f, a), ix('pw', .3)]
                    v = np.array([[1., -1.]]) @ Cb[np.ix_(jj, jj)] @ np.array([1., -1.]); m = mu[jj[0]] - mu[jj[1]]
                    dd = rng.normal(m, np.sqrt(max(v[0], 0)), 400); d_all.append(dd); wts.append(np.full(400, p[i] / 400))
                d_all = np.concatenate(d_all); wts = np.concatenate(wts); wts /= wts.sum()
                cells.append('%+6.2f (P %.2f)' % (float(wts @ d_all) * 1e4 * sbar, float(wts @ (d_all > 0))))
            print('   %-9s ' % nm + ' | '.join('%-30s' % c for c in cells))
        # ---------- 3. coupling-shift stress ----------
        print('\n-- 3. private regime shifted by d (sd of gold74 log OR; + = private labels MORE coupled than the fitted public regime)')
        print('   population X - pw30, public units e-4, prior N01 x joint (Track A posterior) and flat x ruler; interpolated between grid regimes')
        popR = np.array([pops[r] for r in rs])                       # (K, 4, 28)

        def pop_at(k):
            k = np.clip(k, sh[0], sh[-1]); j = np.searchsorted(sh, k)
            if j == 0:
                return popR[0]
            if j >= len(sh):
                return popR[-1]
            t = (k - sh[j - 1]) / (sh[j] - sh[j - 1]); return (1 - t) * popR[j - 1] + t * popR[j]
        for (pn, ln) in (('N01', 'joint (Track A)'), ('flat', 'ruler, s[.2,4]')):
            p, sbar = [(v[2], v[3]) for v in VARS if v[0] == pn and v[1] == ln][0]
            print('   [%s %s]' % (pn, ln))
            print('      d     ' + ' '.join('%9s' % c[0] for c in CANDS) + ' | best pw a')
            for d in (-.5, -.25, 0., .25, .5, 1.0):
                mix = sum(p[i] * pop_at(sh[i] + d) for i in range(len(sh)))
                ref = mix[pw, ia(.3)]
                row = [(mix[FN.index(f), ia(a)] - ref) * 1e4 * sbar for _, f, a in CANDS]
                print('      %+.2f ' % d + ' '.join('%+9.2f' % x for x in row) + ' | %.3f' % AG[np.argmax(mix[pw])])
        # ---------- 4. posterior-predictive checks ----------
        print('\n-- 4. posterior predictive (public, Track A posterior N01 x joint): quantities the Track A fit did not use')
        p = [v[2] for v in VARS if v[0] == 'N01' and v[1] == 'joint (Track A)'][0]
        pp1, pp2, ww = [], [], []
        for i, r in enumerate(rs):
            if p[i] < 1e-4:
                continue
            Z = Zs[r]; S10, S20, S30, PW20, PW30 = Z.T
            ordj = (S10 > 0) & (S20 > S10) & (PW30 > S20) & (PW30 > PW20) & (PW30 > S30)
            # weight each draw by its ABC kernel at the regime's mean s (approximation)
            s = LIK[r]['sj']
            a = (s * S10[:, None] - G10[None]) / H; b = (s * S20[:, None] - G20[None]) / H
            K = np.exp(-.5 * (a * a + b * b)) * (s * (PW30 - S20)[:, None] < GAP[None])
            wk = K.sum(1) * ordj
            if wk.sum() <= 0:
                continue
            pp1.append(s * (PW30 - S20) * 1e4); pp2.append(s * ((PW30 - S30) - (PW20 - S20)) * 1e4); ww.append(wk / wk.sum() * p[i])
        pp1 = np.concatenate(pp1); pp2 = np.concatenate(pp2); ww = np.concatenate(ww); ww /= ww.sum()

        def wq(x, q):
            o = np.argsort(x); c = np.cumsum(ww[o]); return [float(x[o][np.searchsorted(c, qq)]) for qq in q]
        print('   pw30 - eoa20 (public e-4): mean %+.2f  p10/50/90 %s   (A21 frozen ruler LB-only: +0.5 [0.2, 1.2], CLAIMED implied2.md)' % (
            float(ww @ pp1), np.round(wq(pp1, (.1, .5, .9)), 2).tolist()))
        print('   (pw30 - eoa30) - (pw20 - eoa20) public e-4: mean %+.2f  p10/50/90 %s   (implied2 joint estimates give +1.7 - +0.1 = +1.6)' % (
            float(ww @ pp2), np.round(wq(pp2, (.1, .5, .9)), 2).tolist()))
        # public paired sds at the MAP for the step contrasts
        r = 'R3p-1.25' if P == 'P_FT' else 'R3p-1.00'
        C = np.array(R[r]['pub']['cov']) * 1e-8
        def psd(cvec):
            idx = [i for i, _ in cvec]; w = np.array([c for _, c in cvec]); return float(np.sqrt(w @ C[np.ix_(idx, idx)] @ w)) * 1e4
        sm = LIK[r]['sj']
        print('   public paired sd at %s (public units, x s %.2f): pw30-eoa20 %.2f | pw20-eoa20 %.2f | eoa30-eoa20 %.2f | double diff %.2f | pw35-pw30 %.2f | pw25-pw30 %.2f | eoa20-eoa10 %.2f' % (
            r, sm, sm * psd([(ix('pw', .3), 1), (ix('sym', .2), -1)]), sm * psd([(ix('pw', .2), 1), (ix('sym', .2), -1)]),
            sm * psd([(ix('sym', .3), 1), (ix('sym', .2), -1)]),
            sm * psd([(ix('pw', .3), 1), (ix('sym', .3), -1), (ix('pw', .2), -1), (ix('sym', .2), 1)]),
            sm * psd([(ix('pw', .35), 1), (ix('pw', .3), -1)]), sm * psd([(ix('pw', .25), 1), (ix('pw', .3), -1)]),
            sm * psd([(ix('sym', .2), 1), (ix('sym', .1), -1)])))
        # how far apart are the regimes in what the public can see? population pw30-eoa20 and eoa30-eoa20 per regime (public units)
        print('   per-regime population (x s): k | pw30-eoa20 | eoa30-eoa20 | pw20-eoa20 | G20/G10 | pw a*')
        for rr in rs:
            pp = pops[rr] * 1e4 * LIK[rr]['sj']
            print('      %-9s %+6.2f | %+6.2f | %+6.2f | %5.2f | %.3f' % (rr, pp[pw, ia(.3)] - pp[0, ia(.2)], pp[0, ia(.3)] - pp[0, ia(.2)],
                                                                      pp[pw, ia(.2)] - pp[0, ia(.2)], pp[0, ia(.2)] / max(pp[0, ia(.1)], 1e-9), argmax_of(rr)))
        print('   LB (G20/G10) from the 100 LB posterior draws: median %.2f p10/p90 %s' % (
            float(np.median(G20 / G10)), np.round(np.percentile(G20 / G10, [10, 90]), 2).tolist()))


main()
