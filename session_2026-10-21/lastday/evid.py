# evid.py (10-21 ~20:15 IST, Claude, LASTDAY track). MODEL only, local CPU, no OAI rows, no box, no LB call. Run with python -B.
# Marginal likelihood of the A22 read under each stack-component prior ("world"), so the worlds can be averaged by their posterior
# weight instead of being counted equally. The dose world (G, x20, ...) is skeptic_wc.py sections 1-2b exec'd ONCE with base args;
# the stack components are then re-drawn per prior with the same construction as skeptic_wc section 2 (fixed seeds per prior).
# Evidence = E_prior[ 1(c1 c2 c3 c4) x t^4 (1-t)^46 ] (bpick conditioning incl. the band ruler). Output: evid.log / evid.json.
import sys, io, json, numpy as np
HERE = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/lastday'
SKW = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/day/skeptic_wc.py'
sys.argv = ['skeptic_wc.py', 'tag=evid']
sys.dont_write_bytecode = True
src = open(SKW, encoding='utf-8').read()
src = src[:src.index('# ---------------- 3. curves')]
src = src.replace("LOG = open(HERE + '/skeptic_wc_%s.log' % TAG, 'w')", "LOG = None")
g = {'__name__': 'skw_world'}
exec(compile(src, SKW, 'exec'), g)
N = g['N']; G = g['G']; pos = g['pos']; gv = g['gv']; z25 = g['z25']
PUB25, PUB30 = G['PW25'] * 1e4, G['PW30'] * 1e4
others = [pos(G[t] * 1e4) for t in ('S10', 'S20', 'S30', 'PW20')]
p25, p30 = pos(PUB25), pos(PUB30)
base_ok = (p25 >= 0) & (p25 < 1.0) & (p30 < 1.0) & np.all([o < 1.0 for o in others], axis=0)
LOG = open(HERE + '/evid.log', 'w')


def P(s=''):
    print(s, flush=True); LOG.write(s + '\n')


def draw(lomm, mcl, nf=1.0, seed=2110):
    rS = np.random.default_rng(seed)
    sc = rS.choice(3, N, p=[.40, .35, .25])
    if lomm == 'stack':
        muL = .85 * (np.array([1.10, -.48, -.26])[sc] + np.array([1.2, 1.2, 1.6])[sc] * rS.standard_normal(N)); I = rS.normal(.16, .12, N)
    elif lomm == 'skeptic':
        muL = rS.normal(0., .9, N); I = rS.normal(0., .12, N)
    else:
        muL = rS.normal(-.9, .9, N); I = rS.normal(0., .12, N)
    popg = np.where(sc == 0, False, np.where(sc == 2, True, rS.random(N) < .5))
    gv['N'] = N
    if mcl == 'skep2':
        muM = gv['mcl_skep2'](rS, popg)[0]
    elif mcl == 'zero':
        muM = np.zeros(N)
    elif mcl == 'stack':
        muM = .75 * (np.array([2.30, 0., 1.30])[sc] + np.array([1.0, .8, 1.2])[sc] * rS.standard_normal(N))
    else:
        r2 = np.random.default_rng(7331); muM = r2.normal({'neg': -.5, 'eb': -1.5}[mcl], 1.0, N)
    zL = rS.standard_normal(N); eLp = nf * 1.00 * (-.2 * z25 + np.sqrt(1 - .04) * zL)
    rS.standard_normal(N)                       # eLp_o stream (kept aligned with skeptic_wc)
    eMp = nf * 2.40 * rS.standard_normal(N)
    return muL, I, muM, eLp, eMp


res = {}
P('MODEL evid.py | dose world = skeptic_wc base | N %d | evidence of the A22 read (hard conditions x band ruler) per stack prior' % N)
P('  %-22s | evidence (x1e3) | rel to base | P(hard) | post E muL+I, muM, noise | post P(muL+I+muM > 0)' % 'world (lomm05, MCLmm)')
for lomm in ('stack', 'skeptic', 'oai'):
    for mcl in ('stack', 'skep2', 'neg', 'zero', 'eb'):
        muL, I, muM, eLp, eMp = draw(lomm, mcl)
        sw = PUB25 + muL + I + eLp + muM + eMp
        ps = pos(sw); hard = base_ok & (sw > PUB25) & (sw > PUB30) & (ps >= 1.0)
        t = np.clip(ps - 1.0, 0, 1); w = hard * np.where(t < 1, t ** 4 * (1 - t) ** 46, 0.)
        ev = w.mean(); wn = w / w.sum()
        res['%s|%s' % (lomm, mcl)] = dict(ev=float(ev), phard=float(hard.mean()), popL=float((wn * (muL + I)).sum()), popM=float((wn * muM).sum()),
                                         noise=float((wn * (eLp + eMp)).sum()), ppos=float((wn * ((muL + I + muM) > 0)).sum()),
                                         prior_pop=float((muL + I + muM).mean()))
ref = res['stack|skep2']['ev']
for k, r in res.items():
    P('  %-22s | %7.4f | %.2f | %.3f | %+.2f %+.2f %+.2f | %.2f (prior pop mean %+.2f)' % (k, 1e3 * r['ev'], r['ev'] / ref, r['phard'], r['popL'], r['popM'], r['noise'], r['ppos'], r['prior_pop']))
json.dump(res, open(HERE + '/evid.json', 'w'), indent=1)
P('EVID DONE')
LOG.close()
