# bpick_mc.py (10-21 ~18:00 IST, Claude, CANARY track "B pick with A = swide"). MODEL only. RESEARCH ONLY: local CPU arithmetic,
# no OAI rows, no box, no GPU, no LB call, nothing written outside session_2026-10-21/canary. Run with `python -B`.
#
# World = session_2026-10-21/day/skeptic_wc.py sections 1, 2, 2b VERBATIM (exec'd from its source; its LOG line is neutralised so
# nothing is written into day/), i.e. dose_mc fit + our-side posterior + stack components (lomm05, I, MCLmm) + skeptic knobs.
# NEW here: condition on the A22 read (10-21 17:33 IST, MEASURED API list):
#   c1  swide public > pw25 public and > pw30 public           (swide = API best-id among the CLEAN runs; spl / salt / r6pw20 were
#                                                               canary runs, so their public values are unobserved and NOT used)
#   c2  swide shows .960: pos(swide) >= 1                       (pos = skeptic_wc's band coordinate, 1 = the .960 boundary)
#   c3  pw25 shows .959: 0 <= pos(pw25) < 1
#   c4  pw20 / eoa10 / eoa20 / eoa30 / pw30 still below .960 (they display .959)
#   rank (optional, rank=1 default): swide is r146 inside the .960 band r100-150 (51 teams): 4 teams below, 46 above. Uniform band
#        ruler (as picks2 sec 1.1): L(t) = t^4 (1 - t)^46 with t = pos(swide) - 1.
# Then evaluates A = swide with B in {R6, HW, c4, eoa10, eoa20, pw20, pw25, pw30, r6pw20 rule (.958 guard), r6pw20 always}; the
# resubmit value is linear in P(clean): metric = p * metric(r6pw20 rule) + (1 - p) * metric(R6).
# Args: any skeptic_wc.py / dose_mc arg (lomm=, mcl=, rhoS=, rhoA=, nf=, pf=, prox=, psh=, ps=, k_mu=, k_sd=, field=, tag=), plus
#   rank=1|0, tag3=<suffix>.
import sys, os, io, json, numpy as np
HERE = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/final'
SKW = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/day/skeptic_wc.py'
MY = ('rank', 'tag3', 'c4shift')
own3 = dict(a.split('=', 1) for a in sys.argv[1:] if a.split('=', 1)[0] in MY)
sys.argv = ['skeptic_wc.py'] + [a for a in sys.argv[1:] if a.split('=', 1)[0] not in MY]
RANK = int(own3.get('rank', 1)); C4SH = float(own3.get('c4shift', 0.0))   # GUESS knob: c4-lineage private shift vs R6 / HW (e-4; + = c4 table overfit)
sys.dont_write_bytecode = True
src = open(SKW, encoding='utf-8').read()
src = src[:src.index('# ---------------- 3. curves')]
src = src.replace("LOG = open(HERE + '/skeptic_wc_%s.log' % TAG, 'w')", "LOG = None")
assert 'LOG = None' in src and "open(HERE" not in src
g = {'__name__': 'skw_world'}
exec(compile(src, SKW, 'exec'), g)
D = g['D']; F = g['F']; N = g['N']; OUR = g['OUR']; PUB = g['PUB']; pos = g['pos']; G = g['G']
TAG = g['TAG'] + ('_' + own3['tag3'] if 'tag3' in own3 else '') + ('' if RANK else '_norank')
if C4SH:
    OUR = dict(OUR); OUR['c4'] = OUR['c4'] - C4SH * 1e-4      # private only: the public fit (G) is untouched
LOG = open(HERE + '/rt_bpick_%s.log' % TAG, 'w')


def P(s=''):
    print(s, flush=True); LOG.write(s + '\n')


# ---------------- 1. condition on the A22 read ----------------
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
W5 = g['W']                                     # readtable's 5-entry best-id (for reference only)

# ---------------- 2. curves (skeptic_wc section 3 verbatim, field curve selectable) ----------------
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
BL = ['R6', 'HW', 'c4', 'eoa10', 'eoa20', 'pw20', 'pw25', 'pw30', 'r6pw20 rule', 'r6pw20 always']
acc = {}; mom = {}


def add(k, v, w):
    acc[k] = acc.get(k, 0.) + w * (wt[:, None] * (v if v.ndim == 2 else v[:, None])).sum(0)


sh, s, rc4 = g['sh'], g['s'], g['rc4']; NP, NV2 = g['NP'], g['NV2']; kc, epsx, disp = g['kc'], g['epsx'], g['disp']
eLq, eMq = g['eLq'], g['eMq']
for sn, scd in g['SCEN'].items():
    pop = F['popf'](sh + scd['d'], rc4 + scd.get('rc', 0.0), s * scd.get('sf', 1.0)); nsc = scd.get('nsc', 1.0)
    qL, qM, qI = g['privcomp'](g['RHOS'] if sn == 'SITE' else g['RHOA'])
    X = {}
    for tt, nm in (('S10', 'eoa10'), ('S20', 'eoa20'), ('PW20', 'pw20'), ('PW25', 'pw25'), ('PW30', 'pw30')):
        X[nm] = (OUR['c4'] + pop[tt] + nsc * NP[tt]) * 1e4
    X['c4'] = OUR['c4'] * 1e4; X['R6'] = OUR['R6u'] * 1e4; X['HW'] = OUR['HW'] * 1e4
    X['r6pw20 always'] = (OUR['R6u'] + kc * pop['PW20'] + epsx + nsc * (D.RHO * NP['PW20'] + np.sqrt(1 - D.RHO ** 2) * NV2['PW20'])) * 1e4
    X['r6pw20 rule'] = np.where(disp >= 958, X['r6pw20 always'], X['R6'])
    stk = qL + qI + nsc * eLq + qM + nsc * eMq
    A = X['pw25'] + stk
    wg = g['MIX'][sn]
    add('A', met(A), wg)
    add('mom|A', np.column_stack([A, A ** 2]), wg)
    add('comp', np.column_stack([qL, qM, qI, stk, (stk < 0).astype(float), (qM < 0).astype(float), pop['PW25'] * 1e4,
                                 X['pw25'] - X['c4'], (X['pw25'] < X['pw20']).astype(float), (A < X['pw20']).astype(float)]), wg)
    lowE = (X['pw25'] - X['c4']) < 0          # E-OA did not help privately at a .25
    negM = qM < 0                              # MCLmm hurts privately
    for b in BL:
        Bv = X[b]
        add('B|' + b, met(np.maximum(A, Bv)), wg)
        add('Balone|' + b, met(Bv), wg)
        add('mom|' + b, np.column_stack([Bv, Bv ** 2, A * Bv, (Bv > A).astype(float), np.maximum(A, Bv) - A,
                                         (np.maximum(A, Bv) - A) * lowE, (np.maximum(A, Bv) - A) * negM, lowE.astype(float), negM.astype(float)]), wg)

# ---------------- 3. report ----------------
P('MODEL bpick_mc.py | tag %s | world skeptic_wc.py (dose_mc prox %s sh %s s %s field %s; lomm05 %s, MCLmm %s; rhoS %.2f rhoA %.2f nf %.2f pf %.2f; k N(%.2f, %.2f)) | rank ruler %s | N %d' % (
    TAG, D.PROX, D.PSH, D.PS, D.FS, g['LOMM'], g['MCLP'], g['RHOS'], g['RHOA'], g['NF'], g['PF'], D.K_MU, D.K_SD, 'on' if RANK else 'off', N))
P('Conditioning (MEASURED read 10-21 17:33 IST): P(c1 swide > pw25, pw30) %.3f | P(c2 swide .960 | c1) %.3f | P(c3 pw25 .959 | c1 c2) %.3f | P(all hard) %.3f | ESS %.0f' % (
    c1.mean(), (c1 & c2).mean() / max(c1.mean(), 1e-9), (c1 & c2 & c3).mean() / max((c1 & c2).mean(), 1e-9), hard.mean(), ESS))
P('  prior (pre-read) P(readtable 5-way best-id = swide) %.3f; posterior E[pos swide] %.2f, E[pos pw25] %.2f, E[pos pw30] %.2f' % (
    (W5 == 3).mean(), (wt * ps_).sum(), (wt * p25).sum(), (wt * p30).sum()))
pubgap = PUB['swide'] - PUB['pw25']
P('  public swide - pw25 (e-4): prior %+.2f, posterior %+.2f | public muM (MCLmm pop) prior %+.2f -> post %+.2f | muL prior %+.2f -> post %+.2f' % (
    pubgap.mean(), (wt * pubgap).sum(), g['muM'].mean(), (wt * g['muM']).sum(), g['muL'].mean(), (wt * g['muL']).sum()))
cm = acc['comp']
P('  PRIVATE (posterior, scenario mix): lomm05 %+.2f | MCLmm %+.2f | I %+.2f | stack total (incl. noise) %+.2f, P(<0) %.2f, P(MCLmm<0) %.2f | E-OA pop pw25 %+.2f, pw25 - c4 %+.2f | P(pw25 < pw20) %.2f | P(swide < pw20) %.2f' % tuple(cm))
mA = acc['mom|A']; sdA = np.sqrt(mA[1] - mA[0] ** 2)
a = acc['A']
P('\nA = swide alone: E %+.2f (sd %.2f) pub-scale %.5f | silver step %.3f smooth %.3f | r150 step %.3f smooth %.3f | bronze %.3f' % (
    a[0], sdA, .95802 + a[0] * 1e-4, a[1], a[3], a[2], a[5], a[4]))
P('\n== B for A = swide (pair = max). E e-4 | silver step / smooth | r150 step / smooth | bronze || B alone E | corr(A,B) | P(B>A) | E[max-A] | hedge given E-OA<0 / MCLmm<0 ==')
res = dict(tag=TAG, ess=float(ESS), A=[float(v) for v in a], B={})
base = acc['B|R6']
for b in BL:
    p = acc['B|' + b]; m = acc['mom|' + b]; ba = acc['Balone|' + b]
    sdB = np.sqrt(max(m[1] - m[0] ** 2, 1e-12)); cov = m[2] - mA[0] * m[0]; cor = cov / (sdA * sdB)
    hl = m[5] / max(m[7], 1e-9); hm = m[6] / max(m[8], 1e-9)
    res['B'][b] = dict(pair=[float(v) for v in p], alone=[float(v) for v in ba], corr=float(cor), pBgtA=float(m[3]), hedge=float(m[4]), hedge_lowE=float(hl), hedge_negM=float(hm))
    P('  B=%-14s %+6.2f | %.3f / %.3f | %.3f / %.3f | %.3f || %+6.2f | %.2f | %.2f | %+.2f | %+.2f / %+.2f || vs R6: silver %+.1f / %+.1f, r150 %+.1f / %+.1f, E %+.2f' % (
        b, p[0], p[1], p[3], p[2], p[5], p[4], ba[0], cor, m[3], m[4], hl, hm,
        100 * (p[1] - base[1]), 100 * (p[3] - base[3]), 100 * (p[2] - base[2]), 100 * (p[5] - base[5]), p[0] - base[0]))
mR = acc['mom|R6']; P('  P(E-OA at .25 below c4 privately) %.3f | P(MCLmm private < 0) %.3f' % (mR[7], mR[8]))
P('\n== r6pw20 RESUBMIT (B = r6pw20 if clean and display >= .958, else R6): gain vs B = R6 as a function of P(clean) ==')
d = acc['B|r6pw20 rule'] - acc['B|R6']
for pc in (.3, .4, .5, .64, .8, .9, 1.0):
    P('  P(clean) %.2f: E %+.3f | silver step %+.2f smooth %+.2f | r150 step %+.2f smooth %+.2f | bronze %+.2f pts' % (
        pc, pc * d[0], 100 * pc * d[1], 100 * pc * d[3], 100 * pc * d[2], 100 * pc * d[5], 100 * pc * d[4]))
P('  P(r6pw20 display .957 / .958 / .959) %.3f / %.3f / %.3f (posterior; the read does not inform it)' % tuple((wt * (disp == k)).sum() for k in (957, 958, 959)))
json.dump(res, open(HERE + '/rt_bpick_%s.json' % TAG, 'w'), indent=1)
P('BPICK_MC DONE')
LOG.close()

# [final red-team addition] idiosyncratic sds and the effective A-B correlation once a common family shock (sd SH e-4,
# shakeup.md ~19, CLAIMED) is added to both private margins.
LOG = open(HERE + '/rt_bpick_%s.log' % TAG, 'a')
P('\n== [red-team] sd and effective corr with a common family shock ==')
for b in ('R6', 'eoa10', 'c4', 'pw20', 'pw25', 'r6pw20 always'):
    m = acc['mom|' + b]; sdB = np.sqrt(max(m[1] - m[0] ** 2, 1e-12)); cov = m[2] - mA[0] * m[0]
    eff = ['%.2f' % ((cov + SH ** 2) / np.sqrt((sdA ** 2 + SH ** 2) * (sdB ** 2 + SH ** 2))) for SH in (10., 19., 25.)]
    P('  B=%-14s sdB %.2f | corr %.2f | eff corr at shock sd 10 / 19 / 25: %s | var share common at 19: %.2f' % (
        b, sdB, cov / (sdA * sdB), ' / '.join(eff), 19. ** 2 / (19. ** 2 + (sdA ** 2 + sdB ** 2) / 2)))
LOG.close()
