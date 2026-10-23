# apick_sweep.py (copy of apick_mc.py world + read conditioning; NEW: private stack shift sweep for break-even) (10-21 ~19:30 IST, Claude, LASTDAY track "A-pick robustness"). MODEL only. RESEARCH ONLY: local CPU arithmetic, no OAI rows,
# no box, no GPU, no LB / Kaggle call, nothing written outside session_2026-10-21/lastday. Run with `python -B`.
#
# Question: given the A22 read (swide 57464326 .960 r146, pw25 57463742 .959, spl / salt / r6pw20 canary), which A maximises the private
# medal odds of the final pair: swide (R1 / H2 answer), pw25, pw30, pw20 (PF option of A22) or eoa20, with B = R6 57248084 or a clean
# r6pw20 resubmit?
#
# World = session_2026-10-21/day/skeptic_wc.py sections 1, 2, 2b VERBATIM (exec'd from source, its LOG line neutralised), i.e. dose_mc fit
# + our-side posterior + stack components (lomm05, I, MCLmm) + skeptic knobs. Read conditioning = canary/bpick_work/bpick_mc.py section 1
# VERBATIM (c1 swide > pw25, pw30 public; c2 swide .960; c3 pw25 .959; c4 the older entries < .960; uniform band ruler t^4 (1-t)^46).
# NEW here:
#   - A list x B list pair metrics (the pair private score = max(A, B)), A alone, pairwise P(A_i > A_j) private;
#   - decomposition of the public swide - pw25 gap into population components (lomm05 muL, interaction I, MCLmm muM) and read noise;
#   - extra MCLmm prior knob malt=neg (N(-.5, 1.0), skep_value_mc 'neg') / malt=eb (N(-1.5, 1.0): the levers/regime.md EB reading, about
#     -17..-48 label = -1.4..-4.0e-4 macro on rulers, GUESS haircut), applied AFTER the world exec so the rest of the world is unchanged;
#   - c4shift knob as bpick (c4-lineage private shift vs R6 / HW).
# Args: skeptic_wc / dose_mc args (lomm=, mcl=, rhoS=, rhoA=, nf=, pf=, prox=, psh=, ps=, k_mu=, k_sd=, field=, tag=) plus rank=1|0,
#   malt=neg|eb, c4shift=<e-4>, tag3=<suffix>.
import sys, os, io, json, numpy as np
HERE = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/lastday'
SKW = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/day/skeptic_wc.py'
MY = ('rank', 'tag3', 'c4shift', 'malt')
own3 = dict(a.split('=', 1) for a in sys.argv[1:] if a.split('=', 1)[0] in MY)
sys.argv = ['skeptic_wc.py'] + [a for a in sys.argv[1:] if a.split('=', 1)[0] not in MY]
RANK = int(own3.get('rank', 1)); C4SH = float(own3.get('c4shift', 0.0)); MALT = own3.get('malt', '')
sys.dont_write_bytecode = True
src = open(SKW, encoding='utf-8').read()
src = src[:src.index('# ---------------- 3. curves')]
src = src.replace("LOG = open(HERE + '/skeptic_wc_%s.log' % TAG, 'w')", "LOG = None")
assert 'LOG = None' in src and "open(HERE" not in src
g = {'__name__': 'skw_world'}
exec(compile(src, SKW, 'exec'), g)
D = g['D']; F = g['F']; N = g['N']; OUR = g['OUR']; PUB = g['PUB']; pos = g['pos']; G = g['G']

# ---- optional MCLmm prior override (after the world exec; separate RNG so everything else is unchanged) ----
if MALT:
    rM = np.random.default_rng(7331)
    mu_, sd_ = {'neg': (-.5, 1.0), 'eb': (-1.5, 1.0)}[MALT]
    g['muM'] = rM.normal(mu_, sd_, N); g['mM'] = g['muM'].mean(); g['sM'] = g['muM'].std()
    PUB['swide'] = PUB['spl'] + g['muM'] + g['eMp']
    PUB['salt'] = PUB['pw25'] + g['muM'] + g['eMp']

TAG = g['TAG'] + ('_' + own3['tag3'] if 'tag3' in own3 else '') + ('' if RANK else '_norank') + ('_m' + MALT if MALT else '')
if C4SH:
    OUR = dict(OUR); OUR['c4'] = OUR['c4'] - C4SH * 1e-4      # private only: the public fit (G) is untouched
LOG = open(HERE + '/apick_sweep_%s.log' % TAG, 'w')


def P(s=''):
    print(s, flush=True); LOG.write(s + '\n')


# ---------------- 1. condition on the A22 read (bpick_mc.py section 1, verbatim) ----------------
ps_, p25, p30 = pos(PUB['swide']), pos(PUB['pw25']), pos(PUB['pw30'])
others = [pos(G[t] * 1e4) for t in ('S10', 'S20', 'S30', 'PW20')]
c1 = (PUB['swide'] > PUB['pw25']) & (PUB['swide'] > PUB['pw30'])
c2 = ps_ >= 1.0
c3 = (p25 >= 0) & (p25 < 1.0)
c4m = (p30 < 1.0) & np.all([o < 1.0 for o in others], axis=0)
hard = c1 & c2 & c3 & c4m
wt = hard.astype(float)
if RANK:
    t = np.clip(ps_ - 1.0, 0, 1)
    wt = wt * np.where(t < 1, t ** 4 * (1 - t) ** 46, 0.)
wt = wt / wt.sum()
ESS = 1.0 / (wt ** 2).sum()
W5 = g['W']
pri = np.full(N, 1.0 / N)     # prior weights (for the before / after comparison)

# ---------------- 2. curves (bpick_mc.py section 2) ----------------
CV = np.load(D.P2 + '/curves_%s.npz' % D.FS); g19 = CV['grid'] * 1e4
c19 = lambda k, x: np.interp(x, g19, CV[k], left=float(CV[k][0]), right=float(CV[k][-1]))
CB = json.load(open(g['SHK'])); ms = np.array(sorted(float(k) for k in CB)); key = {float(k): k for k in CB}


def cshk(name, x):
    y = np.array([CB[key[m]][name] for m in ms]); slope = (y[-1] - y[-2]) / (ms[-1] - ms[-2])
    out = np.interp(x, ms, y, left=0.0, right=float(y[-1])); hi = x > ms[-1]; out[hi] = np.minimum(1.0, y[-1] + slope * (x[hi] - ms[-1]))
    out[x < ms[0]] = 0.0
    return out


def met(x):   # E | silver step | r150 step | silver smooth | bronze smooth | r150 smooth
    return np.column_stack([x, c19('silver', x), c19('r150', x), cshk('P(r<=316)', x), cshk('P(r<=633)', x), cshk('P(r<=150)', x)])


MN = ['E', 'silv_step', 'r150_step', 'silv_smooth', 'bronze_smooth', 'r150_smooth']
AL = ['swide', 'pw25', 'pw30', 'pw20', 'eoa20', 'eoa10']
BL = ['R6', 'r6pw20 rule', 'none (A alone)']
acc = {}


def add(k, v, w, ww=None):
    ww = wt if ww is None else ww
    acc[k] = acc.get(k, 0.) + w * (ww[:, None] * (v if v.ndim == 2 else v[:, None])).sum(0)



SHIFTS = [0., .5, 1., 1.5, 2., 2.5, 3., 4.]
AL2 = ['pw25', 'pw20', 'pw30', 'eoa20']
sh, s, rc4 = g['sh'], g['s'], g['rc4']; NP, NV2 = g['NP'], g['NV2']; kc, epsx, disp = g['kc'], g['epsx'], g['disp']
eLq, eMq = g['eLq'], g['eMq']
for sn, scd in g['SCEN'].items():
    pop = F['popf'](sh + scd['d'], rc4 + scd.get('rc', 0.0), s * scd.get('sf', 1.0)); nsc = scd.get('nsc', 1.0)
    qL, qM, qI = g['privcomp'](g['RHOS'] if sn == 'SITE' else g['RHOA'])
    X = {}
    for tt, nm in (('S20', 'eoa20'), ('PW20', 'pw20'), ('PW25', 'pw25'), ('PW30', 'pw30')):
        X[nm] = (OUR['c4'] + pop[tt] + nsc * NP[tt]) * 1e4
    X['R6'] = OUR['R6u'] * 1e4
    X['r6pw20 always'] = (OUR['R6u'] + kc * pop['PW20'] + epsx + nsc * (D.RHO * NP['PW20'] + np.sqrt(1 - D.RHO ** 2) * NV2['PW20'])) * 1e4
    X['r6pw20 rule'] = np.where(disp >= 958, X['r6pw20 always'], X['R6'])
    stk = qL + qI + nsc * eLq + qM + nsc * eMq
    wg = g['MIX'][sn]
    if sn == 'S0':
        pass
    add('stk', stk, wg)
    for b in ('R6', 'r6pw20 rule'):
        for a in AL2:
            add('P|%s|%s' % (a, b), met(np.maximum(X[a], X[b])), wg)
        for x in SHIFTS:
            add('S|%.1f|%s' % (x, b), met(np.maximum(X['pw25'] + stk - x, X[b])), wg)
P('MODEL apick_sweep.py | tag %s | ESS %.0f | posterior private stack total E %+.2f (e-4); swide_x = swide - x privately (x = how much of the posterior stack effect is not real)' % (TAG, ESS, acc['stk'][0]))
P('cols: silver step / silver smooth / r150 step / r150 smooth / bronze / E ; d = (swide_x pair) - (alt pair), pts')
out = {}
for b in ('R6', 'r6pw20 rule'):
    P('\n== B = %s ==' % b)
    for a in AL2:
        pa = acc['P|%s|%s' % (a, b)]
        P('  alt A = %s: %.3f / %.3f / %.3f / %.3f / %.3f / %+.2f' % (a, pa[1], pa[3], pa[2], pa[5], pa[4], pa[0]))
        be = {}
        for x in SHIFTS:
            ps = acc['S|%.1f|%s' % (x, b)]; d = ps - pa
            out['%s|%s|%.1f' % (b, a, x)] = [float(v) for v in d]
            P('    x %.1f: d silver %+5.2f / %+5.2f, r150 %+5.2f / %+5.2f, bronze %+5.2f, E %+5.2f' % (x, 100 * d[1], 100 * d[3], 100 * d[2], 100 * d[5], 100 * d[4], d[0]))
        for k, nm in ((1, 'silver step'), (3, 'silver smooth'), (2, 'r150 step'), (5, 'r150 smooth'), (4, 'bronze'), (0, 'E')):
            ds = np.array([(acc['S|%.1f|%s' % (x, b)] - pa)[k] for x in SHIFTS])
            if ds[0] <= 0:
                be[nm] = 0.0
            elif (ds > 0).all():
                be[nm] = float('inf')
            else:
                i = int(np.argmax(ds <= 0)); x0, x1 = SHIFTS[i - 1], SHIFTS[i]; be[nm] = x0 + (x1 - x0) * ds[i - 1] / (ds[i - 1] - ds[i])
        out['be|%s|%s' % (b, a)] = be
        P('    break-even x (swide still >= alt below it): ' + ', '.join('%s %.2f' % (k, v) for k, v in be.items()))
json.dump(dict(tag=TAG, stk=float(acc['stk'][0]), d=out), open(HERE + '/apick_sweep_%s.json' % TAG, 'w'), indent=1)
P('APICK_SWEEP DONE')
LOG.close()
