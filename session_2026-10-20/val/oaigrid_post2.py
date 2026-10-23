# oaigrid_post2.py (10-20 val TRACK A, Claude): population (private-relevant) pairwise comparisons of built / submitted entries
# under the fitted regime posterior. Inputs: box AGGREGATES only (population means + knee-bootstrap covariances per regime).
# P(X > Y) = sum_r post_r * Phi(mu_r / sd_r), mu_r = population difference, sd_r = OAI-size paired knee-bootstrap sd.
import json, os
import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = [(('pw', .2), ('pw', .3)), (('sym', .2), ('pw', .3)), (('pw', .25), ('pw', .3)), (('pw', .25), ('pw', .2)),
         (('pw', .25), ('sym', .2)), (('sym', .3), ('pw', .3)), (('sym', .25), ('pw', .3)), (('pw', .35), ('pw', .3)),
         (('pw', .2), ('sym', .2)), (('sym', .1), ('sym', .2)), (('LP2', .2), ('pw', .25)), (('LPpw', .3), ('pw', .25))]
NM = {'sym': 'eoa', 'pw': 'pw', 'LP2': 'LP2_', 'LPpw': 'LPpw'}


def load(fn):
    t = open(os.path.join(HERE, fn), encoding='utf-8').read().splitlines()
    return json.loads(t[t.index('JSON_BEGIN') + 1])


for P, fn in (('P_FT', 'box_oaigrid_PFT.log'), ('P_AREA', 'box_oaigrid_PAREA.log')):
    E = load(fn); FN = E['FN']; AG = np.array(E['AG']); NA = len(AG); R = E['reg']
    ia = lambda a: int(np.argmin(np.abs(AG - a)))
    k = lambda f, a: FN.index(f) * NA + ia(a)
    rs = sorted([x for x in R if x.startswith('R3p')], key=lambda x: float(x[3:])); sh = np.array([float(x[3:]) for x in rs])
    mids = np.concatenate([[sh[0] - (sh[1] - sh[0]) / 2], (sh[1:] + sh[:-1]) / 2, [sh[-1] + (sh[-1] - sh[-2]) / 2]]); wid = np.diff(mids)
    LJ = np.array([R[x]['pub']['L_joint'] for x in rs])
    SG = np.exp(np.linspace(np.log(.5), np.log(2.), 15))
    print('=' * 90); print(P, '| population (private-relevant) differences, macro e-4; OAI units and x s (public units)')
    for pn, prior in (('N01', np.exp(-.5 * sh ** 2) * wid), ('flat', wid)):
        p = prior * LJ; p /= p.sum()
        sbar = sum(p[i] * float(np.dot(R[x]['pub']['s_post'], SG)) for i, x in enumerate(rs))
        print('  [%s prior x joint lik] s %.2f' % (pn, sbar))
        for X, Y in PAIRS:
            mu = []; sd = []
            for x in rs:
                pop = np.array(R[x]['pop']); C = np.array(R[x]['boot']['knee']['cov']) * 1e-8
                i, j = k(*X), k(*Y)
                mu.append(pop[FN.index(X[0]), ia(X[1])] - pop[FN.index(Y[0]), ia(Y[1])])
                sd.append(np.sqrt(max(C[i, i] + C[j, j] - 2 * C[i, j], 1e-16)))
            mu = np.array(mu); sd = np.array(sd)
            m = (p * mu).sum(); pr = (p * norm.cdf(mu / sd)).sum()
            sdt = np.sqrt((p * (sd ** 2 + mu ** 2)).sum() - m ** 2)
            print('    %s%02d - %s%02d: %+.2f e-4 OAI (sd regime+knee %.2f) = %+.2f e-4 public units | P(>) %.2f' % (
                NM[X[0]], round(X[1] * 100), NM[Y[0]], round(Y[1] * 100), m * 1e4, sdt * 1e4, m * sbar * 1e4, pr))
