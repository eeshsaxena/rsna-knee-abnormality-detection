# abc_a21.py (10-20 val Track D; local CPU; uses ONLY box aggregates already on disk (push/work/box_abc2_*.log) + public LB counts).
# Updates the push/work ABC (abc_local2.run: OAI-free D5 curve family reweighted to RSNA coupling, public sample n 390, kernel on the
# eoa10 / eoa20 in-band reads) with the A21 read (MEASURED a21_frozen.log):
#   (1) R1: the API best-id moved to pw30 => pub(pw30) > pub(eoa20), pub(eoa30), pub(pw20);
#   (2) frozen ruler: pw30 sits between the 2nd and 3rd of the 4 frozen teams that were above eoa20 inside .959 => with the 4 teams
#       ~uniform on (eoa20, .960) (MODEL; in-band density varies <5% over that gap), t = (pub pw30 - pub eoa20) / gap960 has likelihood
#       6 t^2 (1 - t)^2 on (0, 1) (2 of 4 below), and pw30 still displays .959 (t < 1);
#   (3) eoa30 and pw20 also display .959 (CLAIMED from the task brief; only the floor binds).
# Writes resampled posterior draws (population = private-relevant deltas vs c4, public deltas, gap, noise cov) for picks2_mc.py.
import sys, json, numpy as np
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push/work')
import abc_local2 as A
OUT = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/val/picks2_work/'
VARS = ['S10', 'S20', 'S25', 'S30', 'S35', 'PW20', 'PW25', 'PW30', 'PW35', 'PW40']
def collect(prox, psh, ps, reps=3, **kw):
    acc = []
    for _ in range(reps):
        o = A.run(prox, 'D5', psh, ps, quiet=True, **kw); a = o['_arr']
        iv = a['iv']; acc.append(dict(G=a['G'][:, [iv[v] for v in VARS]], Pop=a['Pop'][:, [iv[v] for v in VARS]], w=a['w'], gap=a['gap'], sh=a['sh'], s=a['s']))
    E = A.load('box_abc2_%s.log' % prox); C = np.array(E['cov']['D5']); ivC = [E['VN'].index(v) for v in VARS]
    D = {k: np.concatenate([x[k] for x in acc]) for k in acc[0]}; D['C'] = C[np.ix_(ivC, ivC)] * kw.get('tnoise', 1.0) ** 2
    return D
ix = {v: i for i, v in enumerate(VARS)}
def a21_like(D, tshape=(2, 2)):
    G, gap = D['G'], D['gap']; inc = G[:, ix['PW30']] - G[:, ix['S20']]; t = inc / gap
    a_, b_ = tshape
    Lt = np.where((t > 0) & (t < 1), np.clip(t, 1e-9, 1) ** a_ * np.clip(1 - t, 1e-9, 1) ** b_, 0.)
    am = np.argmax(np.stack([G[:, ix[v]] for v in ('S20', 'S30', 'PW20', 'PW30')], 1), 1) == 3
    x20 = 1e-3 - gap   # eoa20 height above the .959 floor (AUC units)
    disp = ((G[:, ix['S30']] - G[:, ix['S20']]) > -x20) & ((G[:, ix['PW20']] - G[:, ix['S20']]) > -x20)
    return Lt * am * disp
def summ(D, w, tag, rng):
    t = w.sum(); ess = t ** 2 / (w ** 2).sum(); P = lambda x: float((w * x).sum() / t)
    def q(x):
        o = np.argsort(x); cw = np.cumsum(w[o]) / t; return [float(x[o][min(len(x) - 1, np.searchsorted(cw, p))]) for p in (.1, .5, .9)]
    G, Pop, gap = D['G'], D['Pop'], D['gap']
    r = dict(tag=tag, ess=float(ess), sh=[P(D['sh'])] + q(D['sh']), s=[P(D['s'])] + q(D['s']))
    inc = G[:, ix['PW30']] - G[:, ix['S20']]; r['pub_pw30_m_eoa20'] = [P(inc)] + q(inc); r['gap'] = [P(gap)] + q(gap)
    sdp = np.sqrt(390 / 910); C = D['C']
    r['pop_vs_c4'] = {v: [P(Pop[:, ix[v]])] + q(Pop[:, ix[v]]) for v in VARS}
    r['pop_vs_pw30'] = {}
    for v in VARS:
        d = Pop[:, ix[v]] - Pop[:, ix['PW30']]; cv = C[ix[v], ix[v]] + C[ix['PW30'], ix['PW30']] - 2 * C[ix[v], ix['PW30']]
        npv = rng.standard_normal(len(d)) * np.sqrt(max(cv, 0)) * sdp
        r['pop_vs_pw30'][v] = [P(d)] + q(d) + [P(d + npv > 0)]
    # 10-21 candidates vs the current best pw30 (public): P(beats pw30) = P(new best-id under R1); P(shows .960 / .961)
    x30 = (1e-3 - gap) + inc   # pw30 height above the .959 floor
    r['next'] = {}
    for v in ('S25', 'S35', 'PW25', 'PW35', 'PW40'):
        d = G[:, ix[v]] - G[:, ix['PW30']]; h = x30 + d
        r['next'][v] = dict(pub=[P(d)] + q(d), P_gt=P(d > 0), P960=P(h >= 1e-3), P961=P(h >= 2e-3),
                            pop_vs_pw30=P(Pop[:, ix[v]] - Pop[:, ix['PW30']]), pop_gain_if_R1=P((d > 0) * (Pop[:, ix[v]] - Pop[:, ix['PW30']])))
    return r
def show(r):
    print('== %s | ESS %.0f | sh %.2f [%.2f %.2f %.2f] | s %.2f | gap960 %.2fe-4 | pub pw30 - eoa20 %+.2fe-4 [%+.2f %+.2f %+.2f]' % (
        r['tag'], r['ess'], *r['sh'], r['s'][0], r['gap'][0] * 1e4, *[x * 1e4 for x in r['pub_pw30_m_eoa20']]))
    print('   POP vs c4 (e-4): ' + ' '.join('%s %+.2f' % (v, x[0] * 1e4) for v, x in r['pop_vs_c4'].items()))
    print('   POP vs pw30 (e-4, [p10 p50 p90], P(private > pw30)): ' + ' | '.join('%s %+.2f [%+.1f %+.1f %+.1f] %.2f' % (v, x[0] * 1e4, x[1] * 1e4, x[2] * 1e4, x[3] * 1e4, x[4]) for v, x in r['pop_vs_pw30'].items() if v != 'PW30'))
    for v, d in r['next'].items():
        print('   NEXT %-4s public vs pw30 %+.2fe-4 [%+.1f %+.1f %+.1f] | P(new best-id) %.2f | P(.960) %.2f P(.961) %.2f | pop vs pw30 %+.2fe-4 | E[pop gain if R1 picks it] %+.2fe-4' % (
            v, d['pub'][0] * 1e4, d['pub'][1] * 1e4, d['pub'][2] * 1e4, d['pub'][3] * 1e4, d['P_gt'], d['P960'], d['P961'], d['pop_vs_pw30'] * 1e4, d['pop_gain_if_R1'] * 1e4))
if __name__ == '__main__':
    rng = np.random.default_rng(1021); RES = {}
    rows = [('PFT', 'N01', 'logN', {}), ('PFT', 'N01', 'logU', {}), ('PFT', 'flat', 'logN', {}), ('PFT', 'flat', 'logU', {}),
            ('PAREA', 'N01', 'logN', {}), ('PAREA', 'flat', 'logU', {}), ('PFT', 'N01', 'logN', dict(tnoise=1.25)), ('PFT', 'N01', 'logN', dict(rho_mu=.78))]
    for prox, psh, ps, kw in rows:
        tag = '%s_%s_%s%s' % (prox, psh, ps, ''.join('_%s%s' % kv for kv in kw.items()))
        D = collect(prox, psh, ps, **kw)
        r0 = summ(D, D['w'], tag + ' PRE-A21', rng); show(r0)
        La = a21_like(D); w1 = D['w'] * La
        r1 = summ(D, w1, tag + ' POST-A21', rng); show(r1)
        RES[tag] = dict(pre=r0, post=r1)
        if tag == 'PFT_N01_logN':   # primary: also the uniform-t sensitivity (t flat on (0,1): only the bracket by the band edges)
            r2 = summ(D, D['w'] * a21_like(D, (0, 0)) , tag + ' POST-A21 t-flat', rng); show(r2); RES[tag]['post_tflat'] = r2
        # resample draws for the pick model
        p = w1 / w1.sum(); j = rng.choice(len(p), 200000, p=p)
        np.savez(OUT + 'abc_a21_draws_%s.npz' % tag, Pop=D['Pop'][j], G=D['G'][j], gap=D['gap'][j], C=D['C'], VARS=np.array(VARS),
                 pre_Pop=D['Pop'][rng.choice(len(D['w']), 200000, p=D['w'] / D['w'].sum())])
        del D
    json.dump(RES, open(OUT + 'abc_a21.json', 'w'), indent=1)
