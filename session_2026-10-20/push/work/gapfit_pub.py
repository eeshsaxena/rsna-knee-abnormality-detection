# gapfit_pub.py: translate the public LB gains (lb_post_pooled.npz) into an implied c4 gap (rho - q) via the OAI curve family
# (session_2026-10-19/ex/box_curve_*.log JSON aggregates; population-level, reweighted schemes incl. raw and stress).
# Model: delta(a) = s * (alpha_a + beta_a * gap), fitted per a across OAI model x scheme points (P_FT); s = c4 amplitude factor.
import json, numpy as np
E = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/ex/'
pts = []
for j in ('D5', 's1044', 's1045', 'ENS3', 'FOLDS', 'AREA'):
    s = open(E + 'box_curve_%s.log' % j, encoding='utf-8').read(); d = json.loads(s.split('JSON_BEGIN')[1].split('JSON_END')[0])
    for m, ss in d['res'].items():
        for sc, r in ss.items():
            gap = np.mean(r['desc']['rho']) - np.mean(r['desc']['q'])
            row = dict(src=j, m=m, s=sc, prox=d['proxy'], gap=gap, phi=np.mean(r['desc']['phi']))
            for a in (.05, .1, .15, .2, .25, .3):
                row['S%02d' % round(a * 100)] = r['S%.2f' % a]['macro']['m']
            row['PW10'] = r['PW10']['macro']['m']; row['PW20'] = r['PW20']['macro']['m']
            pts.append(row)
import pandas as pd
P = pd.DataFrame(pts)
print('points', len(P), P.groupby('prox').size().to_dict())
cols = ['S05', 'S10', 'S15', 'S20', 'S25', 'S30', 'PW10', 'PW20']
FIT = {}
for prox in ('P_FT', 'P_AREA'):
    Q = P[P.prox == prox]
    FIT[prox] = {}
    for c in cols:
        b = np.polyfit(Q.gap, Q[c], 1); r = np.corrcoef(Q.gap, Q[c])[0, 1]
        FIT[prox][c] = b
        print('%-6s %-4s = %+.5f + %+.5f gap (r %.2f, n %d)  ratio-to-S10 at gap .10/.15/.20/.25: %s' % (
            prox, c, b[1], b[0], r, len(Q), [round(np.polyval(b, g) / np.polyval(FIT[prox].get('S10', b), g), 2) if 'S10' in FIT[prox] else None for g in (.1, .15, .2, .25)]))
# ratio S20/S10 as a function of gap (population), and S30-S20
print('\nP_FT population curve vs gap (macro x1e3):')
for g in (.04, .06, .08, .10, .12, .14, .16, .18, .20, .25):
    v = {c: np.polyval(FIT['P_FT'][c], g) * 1e3 for c in cols}
    print('  gap %.2f: S10 %+.2f S20 %+.2f S30 %+.2f | S20/S10 %.2f | S30-S20 %+.2f | PW20-S20 %+.3f | best sym a %s' % (
        g, v['S10'], v['S20'], v['S30'], v['S20'] / v['S10'] if v['S10'] > 0 else float('nan'), v['S30'] - v['S20'], v['PW20'] - v['S20'],
        max(['S05', 'S10', 'S15', 'S20', 'S25', 'S30'], key=lambda c: v[c])))
# invert the public reads: for each LB draw, solve (s, gap) from G10, G20 (two equations, two unknowns)
d = np.load('lb_post_pooled.npz'); G10 = d['G10'] * 1e-3; G20 = d['G20'] * 1e-3; gap960 = d['gap960'] * 1e-3
for prox in ('P_FT', 'P_AREA'):
    F = FIT[prox]; a1, b1 = F['S10'][1], F['S10'][0]; a2, b2 = F['S20'][1], F['S20'][0]
    # G20/G10 = (a2 + b2 g) / (a1 + b1 g) -> g = (a2 - R a1) / (R b1 - b2)
    R = G20 / G10; g = (a2 - R * a1) / (R * b1 - b2); s = G10 / (a1 + b1 * g)
    ok = np.isfinite(g) & (s > 0) & (g > -0.1) & (g < 1)
    q = lambda v: '%.3f [%.3f, %.3f]' % (np.median(v[ok]), np.quantile(v[ok], .1), np.quantile(v[ok], .9))
    print('\n[%s] LB-implied (population fit, no public-sample noise): ratio G20/G10 %s | gap %s | amplitude s %s | valid %.2f' % (
        prox, q(R), q(g), q(s), ok.mean()))
    for c in ('S25', 'S30', 'PW20'):
        inc = s * (np.polyval(F[c], g) - np.polyval(F['S20'], g))
        print('   %s - eoa20 = %s e-3 | P(>0) %.2f | P(> gap to .960) %.2f' % (c, q(inc * 1e3), np.mean(inc[ok] > 0), np.mean(inc[ok] > gap960[ok])))
    # with s fixed (skeptic 1.1) and gap fitted by least squares to both reads
    for sfix in (0.7, 1.0, 1.1, 1.5):
        gg = np.linspace(-.05, .4, 901); pred10 = sfix * (a1 + b1 * gg); pred20 = sfix * (a2 + b2 * gg)
        i = np.argmin((pred10[None] - G10[:, None]) ** 2 + (pred20[None] - G20[:, None]) ** 2, 1); gfit = gg[i]
        res = np.sqrt(((pred10[i] - G10) ** 2 + (pred20[i] - G20) ** 2) / 2)
        inc30 = sfix * (np.polyval(F['S30'], gfit) - np.polyval(F['S20'], gfit))
        print('   s fixed %.1f: gap %.3f [%.3f, %.3f] | rms misfit %.2fe-4 | S30 - eoa20 %+.2fe-4' % (
            sfix, np.median(gfit), np.quantile(gfit, .1), np.quantile(gfit, .9), np.median(res) * 1e4, np.median(inc30) * 1e4))
json.dump({p: {c: list(map(float, v)) for c, v in F.items()} for p, F in FIT.items()}, open('gapfit_pub.json', 'w'), indent=1)
