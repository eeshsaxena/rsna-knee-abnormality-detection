# abc_local.py (10-20 push): local ABC on the public LB reads using ONLY box aggregates (box_abc_{PFT,PAREA}.log JSON:
# per (model, regime) mean curves at n 390 and the pooled within-regime noise covariance). No OAI row is used or needed.
# c4 population curve = s * D5 curve at the regime whose D5 gap equals c4's gap (rho_c4 - q(regime)); public sample = curve + noise.
# Priors: hidden OA label regime sh (gold74 log-OR sd units), amplitude s, rho_c4 ~ N(.816, .035).
import json, sys, numpy as np
W = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push/work/'
rng = np.random.default_rng(11)
LB = np.load(W + 'lb_post_pooled.npz'); J = rng.choice(len(LB['G10']), 400, replace=False)
LBG = np.stack([LB['G10'][J], LB['G20'][J], LB['gap960'][J]], 1) * 1e-3


def load(f):
    s = open(W + f).read(); return json.loads(s.split('JSON_BEGIN')[1].split('JSON_END')[0])


SHX = {'raw': -2.5, 'R3p-2.0': -2.0, 'R3p-1.5': -1.5, 'R3p-1.0': -1.0, 'R3p-0.5': -0.5, 'R3p+0.0': 0.0, 'R3p+0.5': .5, 'R3p+1.0': 1.0}


def setup(E, m):
    VN = E['VN']; regs = sorted([k.split('|')[1] for k in E['mean'] if k.startswith(m + '|')], key=lambda r: SHX[r])
    xs = np.array([SHX[r] for r in regs]); M = np.array([E['mean'][m + '|' + r] for r in regs]); C = np.array(E['cov'][m])
    return VN, xs, M, C


def run(prox='PFT', m='D5', prior_sh='N01', prior_s='logN', M_=120000, tnoise=1.0, rho_mu=.816, h=2e-5, quiet=False):
    E = load('box_abc_%s.log' % prox); VN, xs, M, C = setup(E, m)
    iv = {v: VN.index(v) for v in VN}; nv = 12
    # hidden-regime prior
    sh = rng.normal(0, 1, M_) if prior_sh == 'N01' else rng.uniform(-2.5, 1.0, M_)
    sh = np.clip(sh, -2.5, 1.0)
    s = np.exp(rng.normal(0, .35, M_)) if prior_s == 'logN' else np.exp(rng.uniform(np.log(.4), np.log(2.5), M_))
    rc4 = rng.normal(rho_mu, .035, M_)
    interp = lambda col, x: np.interp(x, xs, M[:, col])
    q_h = interp(iv['q'], sh)                       # D5's q at the hidden regime (c4 q ~ D5 q: gold58 c4 .711 vs D5 R3p0 .712)
    gap_c4 = rc4 - q_h
    gapD5 = M[:, iv['rho']] - M[:, iv['q']]          # decreasing in sh
    o = np.argsort(gapD5); sh_eq = np.interp(gap_c4, gapD5[o], xs[o], left=-3.5, right=1.5)   # D5 regime with the same gap
    # linear extrapolation beyond the grid ends (rare)
    def curve(col, x):
        y = np.interp(x, xs, M[:, col]); lo = x < xs[0]; hi = x > xs[-1]
        y[lo] = M[0, col] + (x[lo] - xs[0]) * (M[1, col] - M[0, col]) / (xs[1] - xs[0])
        y[hi] = M[-1, col] + (x[hi] - xs[-1]) * (M[-1, col] - M[-2, col]) / (xs[-1] - xs[-2])
        return y
    Pop = np.stack([s * curve(c, sh_eq) for c in range(nv)], 1)
    L = np.linalg.cholesky(C[:nv, :nv] + 1e-14 * np.eye(nv)); Nz = (rng.standard_normal((M_, nv)) @ L.T) * tnoise
    G = Pop + Nz
    # kernel likelihood against the LB joint draws
    K = np.exp(-.5 * (((G[:, [iv['S10']]] - LBG[None, :, 0]) / h) ** 2 + ((G[:, [iv['S20']]] - LBG[None, :, 1]) / h) ** 2))
    w = K.sum(1); tot = w.sum(); ess = tot ** 2 / (w ** 2).sum()
    # pick one LB draw per sim (for the joint gap960), proportional to K
    cum = np.cumsum(K, 1); u = rng.random(M_) * cum[:, -1]; jj = np.minimum((cum < u[:, None]).sum(1), K.shape[1] - 1); gap = LBG[jj, 2]
    wq = lambda x: (lambda o_: (lambda cw: [float(x[o_][min(len(x) - 1, np.searchsorted(cw, p))]) for p in (.1, .5, .9)])(np.cumsum(w[o_]) / tot))(np.argsort(x))
    mean = lambda x: float((w * x).sum() / tot)
    out = {'ess': float(ess), 'sh': (mean(sh), wq(sh)), 's': (mean(s), wq(s)), 'gap_c4_pop': (mean(gap_c4), wq(gap_c4)),
           'q_pop': (mean(q_h), wq(q_h)), 'phi_pop': (mean(interp(iv['phi'], sh)), wq(interp(iv['phi'], sh)))}
    pred = {}
    for v in ('S15', 'S25', 'S30', 'S40', 'PW20', 'PW30', 'PW40'):
        inc = G[:, iv[v]] - G[:, iv['S20']]
        pred[v] = dict(mean=mean(inc), q=wq(inc), Ppos=mean(inc > 0), P960=mean(inc > gap))
    # public-optimal symmetric a
    grid = ['S05', 'S10', 'S15', 'S20', 'S25', 'S30', 'S40', 'S50']; bi = np.argmax(G[:, [iv[g] for g in grid]], 1)
    popt = {g: float(w[bi == k].sum() / tot) for k, g in enumerate(grid)}
    # population (no public-sample noise) deltas vs c4 -> private expectation; private-sample noise sd = public sd * sqrt(390/910)
    popd = {v: (mean(Pop[:, iv[v]]), wq(Pop[:, iv[v]])) for v in ('S10', 'S20', 'S25', 'S30', 'S40', 'PW20', 'PW30')}
    pinc = {}
    sdp = np.sqrt(390 / 910)
    for v in ('S10', 'S30', 'PW20', 'PW30', 'S25'):
        dpop = Pop[:, iv[v]] - Pop[:, iv['S20']]
        cv = C[iv[v], iv[v]] + C[iv['S20'], iv['S20']] - 2 * C[iv[v], iv['S20']]
        npv = rng.standard_normal(M_) * np.sqrt(cv) * sdp * tnoise
        pinc[v] = dict(pop=mean(dpop), pop_q=wq(dpop), Ppriv=mean(dpop + npv > 0))
    # private vs c4 for eoa10 / eoa20 / eoa30
    pvs = {}
    for v in ('S10', 'S20', 'S30', 'PW20', 'PW30'):
        npv = rng.standard_normal(M_) * np.sqrt(C[iv[v], iv[v]]) * sdp * tnoise
        pvs[v] = dict(pop=mean(Pop[:, iv[v]]), Ppriv=mean(Pop[:, iv[v]] + npv > 0))
    out.update(pred=pred, popt=popt, popd=popd, pinc=pinc, pvs=pvs)
    out['_arr'] = dict(G=G, Pop=Pop, w=w, gap=gap, iv=iv, sh=sh, s=s)
    if not quiet:
        print('== prox %s | model %s | prior sh %s | prior s %s | noise x%.2f | rho_c4 %.3f | ESS %.0f' % (prox, m, prior_sh, prior_s, tnoise, rho_mu, ess))
        for k in ('sh', 's', 'gap_c4_pop', 'q_pop', 'phi_pop'):
            print('   %-10s mean %.3f p10/50/90 %s' % (k, out[k][0], np.round(out[k][1], 3).tolist()))
        for v, d in pred.items():
            print('   PUBLIC %-4s - eoa20: mean %+.2fe-4 p10/50/90 %s e-4 | P(>0) %.2f | P(shows .960) %.2f' % (
                v, d['mean'] * 1e4, np.round(np.array(d['q']) * 1e4, 2).tolist(), d['Ppos'], d['P960']))
        print('   public-optimal sym a:', ' '.join('%s %.2f' % (k, v) for k, v in popt.items()))
        for v, d in pinc.items():
            print('   POP/PRIVATE %-4s - eoa20: pop %+.2fe-4 p10/50/90 %s e-4 | P(private > eoa20) %.2f' % (
                v, d['pop'] * 1e4, np.round(np.array(d['pop_q']) * 1e4, 2).tolist(), d['Ppriv']))
        print('   POP/PRIVATE vs c4:', ' | '.join('%s %+.2fe-4 (P priv>c4 %.2f)' % (v, d['pop'] * 1e4, d['Ppriv']) for v, d in pvs.items()))
    return out


if __name__ == '__main__':
    RES = {}
    for prox in ('PFT', 'PAREA'):
        for psh in ('N01', 'flat'):
            for ps in ('logN', 'logU'):
                RES['%s|%s|%s' % (prox, psh, ps)] = run(prox, 'D5', psh, ps)
    # sensitivities on the primary (PFT, N01, logN)
    RES['PFT|N01|logN|noise1.25'] = run('PFT', 'D5', 'N01', 'logN', tnoise=1.25)
    RES['PFT|N01|logN|noise0.8'] = run('PFT', 'D5', 'N01', 'logN', tnoise=0.8)
    RES['PFT|N01|logN|rho.78'] = run('PFT', 'D5', 'N01', 'logN', rho_mu=.78)
    RES['PFT|N01|logN|s1044'] = run('PFT', 's1044', 'N01', 'logN')
    for k in RES: RES[k].pop('_arr', None)
    json.dump(RES, open(W + 'abc_local.json', 'w'), indent=1, default=float)
