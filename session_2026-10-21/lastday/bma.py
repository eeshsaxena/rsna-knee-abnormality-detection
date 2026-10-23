# bma.py: average the 15 stack-prior worlds (lomm05 x MCLmm, base dose world, rho 1) by prior weight x read evidence (evid.json).
# MODEL on apick_<world>.json; prior weights are GUESS (stated). Writes bma.out. Run with python -B.
import json, numpy as np
HERE = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/lastday'
FILE = {('stack', 'stack'): 'mstack', ('stack', 'skep2'): 'base', ('stack', 'neg'): 'base_mneg', ('stack', 'zero'): 'mzero', ('stack', 'eb'): 'base_meb',
        ('skeptic', 'stack'): 'wSKstack', ('skeptic', 'skep2'): 'lskep', ('skeptic', 'neg'): 'wSK_mneg', ('skeptic', 'zero'): 'wSKzero', ('skeptic', 'eb'): 'wSK_meb',
        ('oai', 'stack'): 'wOAIstack', ('oai', 'skep2'): 'wOAIskep2', ('oai', 'neg'): 'wOAI_mneg', ('oai', 'zero'): 'worst', ('oai', 'eb'): 'worst_meb'}
EV = json.load(open(HERE + '/evid.json'))
R = {k: json.load(open(HERE + '/apick_%s.json' % v)) for k, v in FILE.items()}
LW = {  # GUESS prior weights: lomm05 (stack, skeptic, oai) and MCLmm (stack, skep2, neg, zero, eb)
    'flat':      ({'stack': 1 / 3, 'skeptic': 1 / 3, 'oai': 1 / 3}, {'stack': .2, 'skep2': .2, 'neg': .2, 'zero': .2, 'eb': .2}),
    'A22 priors': ({'stack': 1., 'skeptic': 0., 'oai': 0.}, {'stack': 0., 'skep2': 1., 'neg': 0., 'zero': 0., 'eb': 0.}),
    'optimist':  ({'stack': .6, 'skeptic': .3, 'oai': .1}, {'stack': .3, 'skep2': .4, 'neg': .1, 'zero': .1, 'eb': .1}),
    'levers':    ({'stack': .4, 'skeptic': .4, 'oai': .2}, {'stack': .1, 'skep2': .35, 'neg': .2, 'zero': .15, 'eb': .2}),
    'skeptic':   ({'stack': .2, 'skeptic': .4, 'oai': .4}, {'stack': .05, 'skep2': .2, 'neg': .25, 'zero': .2, 'eb': .3}),
    'EB-heavy':  ({'stack': .1, 'skeptic': .4, 'oai': .5}, {'stack': 0., 'skep2': .1, 'neg': .3, 'zero': .2, 'eb': .4}),
}
AL = ['swide', 'pw25', 'pw30', 'pw20', 'eoa20']
out = open(HERE + '/bma.out', 'w')


def P(s=''):
    print(s); out.write(s + '\n')


P('bma.py (MODEL; prior weights GUESS). Worlds = 3 lomm05 x 5 MCLmm priors, base dose world, rho 1. Posterior world weight = prior x evidence.')
P('Per world: private stack E (posterior) | A=swide pair silver step/smooth, r150 step/smooth with B=R6 | best A on silver step / smooth / r150 step / E')
for k, r in R.items():
    pv = r['decomp']['priv']; p = r['pairs']
    best = []
    for i in (1, 3, 2, 0):
        v = {a: p['%s|R6' % a][i] for a in AL}; best.append(max(v, key=v.get))
    sw = p['swide|R6']
    P('  %-16s ev %.2f | stack %+.2f | %.3f/%.3f %.3f/%.3f | %s' % ('%s|%s' % k, EV['%s|%s' % k]['ev'] / EV['stack|skep2']['ev'], pv[5], sw[1], sw[3], sw[2], sw[5], ' / '.join(best)))
for wn, (wl, wm) in LW.items():
    pw = {k: wl[k[0]] * wm[k[1]] for k in R}; post = {k: pw[k] * EV['%s|%s' % k]['ev'] for k in R}
    s0, s1 = sum(pw.values()), sum(post.values())
    pw = {k: v / s0 for k, v in pw.items()}; post = {k: v / s1 for k, v in post.items()}
    stk_pr = sum(pw[k] * R[k]['decomp']['priv'][5] for k in R); stk_po = sum(post[k] * R[k]['decomp']['priv'][5] for k in R)
    P('\n== weights "%s": BMA private stack E %+.2f (prior-weighted, no evidence reweight %+.2f); P(stack world with E<0) %.2f ==' % (
        wn, stk_po, stk_pr, sum(post[k] for k in R if R[k]['decomp']['priv'][5] < 0)))
    for B in ('R6', 'r6pw20 rule', 'mix0.55'):
        sw = sum(post[k] * np.array(R[k]['pairs']['swide|%s' % B]) for k in R)
        line = '  B=%-11s swide %.3f/%.3f %.3f/%.3f %+.2f' % (B, sw[1], sw[3], sw[2], sw[5], sw[0])
        for a in AL[1:]:
            d = sum(post[k] * np.array(R[k]['pairs']['%s|%s' % (a, B)]) for k in R) - sw
            line += ' | %s %+4.1f/%+4.1f %+4.1f/%+4.1f b%+4.1f E%+.2f' % (a, 100 * d[1], 100 * d[3], 100 * d[2], 100 * d[5], 100 * d[4], d[0])
        P(line)
out.close()
