# apick_mc.py (10-21 ~19:30 IST, Claude, LASTDAY track "A-pick robustness"). MODEL only. RESEARCH ONLY: local CPU arithmetic, no OAI rows,
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
LOG = open(HERE + '/apick_%s.log' % TAG, 'w')


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


sh, s, rc4 = g['sh'], g['s'], g['rc4']; NP, NV2 = g['NP'], g['NV2']; kc, epsx, disp = g['kc'], g['epsx'], g['disp']
eLq, eMq = g['eLq'], g['eMq']
for sn, scd in g['SCEN'].items():
    pop = F['popf'](sh + scd['d'], rc4 + scd.get('rc', 0.0), s * scd.get('sf', 1.0)); nsc = scd.get('nsc', 1.0)
    qL, qM, qI = g['privcomp'](g['RHOS'] if sn == 'SITE' else g['RHOA'])
    X = {}
    for tt, nm in (('S10', 'eoa10'), ('S20', 'eoa20'), ('PW20', 'pw20'), ('PW25', 'pw25'), ('PW30', 'pw30')):
        X[nm] = (OUR['c4'] + pop[tt] + nsc * NP[tt]) * 1e4
    X['c4'] = OUR['c4'] * 1e4; X['R6'] = OUR['R6u'] * 1e4
    X['r6pw20 always'] = (OUR['R6u'] + kc * pop['PW20'] + epsx + nsc * (D.RHO * NP['PW20'] + np.sqrt(1 - D.RHO ** 2) * NV2['PW20'])) * 1e4
    X['r6pw20 rule'] = np.where(disp >= 958, X['r6pw20 always'], X['R6'])
    stk = qL + qI + nsc * eLq + qM + nsc * eMq
    X['swide'] = X['pw25'] + stk
    X['none (A alone)'] = np.full(N, -1e9)
    wg = g['MIX'][sn]
    for a in AL:
        Av = X[a]
        add('mom|' + a, np.column_stack([Av, Av ** 2, (Av < 0).astype(float)]), wg)
        for b in BL:
            add('P|%s|%s' % (a, b), met(np.maximum(Av, X[b])), wg)
        for a2 in AL:
            add('gt|%s|%s' % (a, a2), (Av > X[a2]).astype(float), wg)
        add('dE|' + a, Av - X['swide'], wg)
    # decomposition (private)
    add('priv', np.column_stack([qL, qI, qM, nsc * eLq, nsc * eMq, stk, (stk < 0).astype(float), (qM < 0).astype(float), (qL + qI < 0).astype(float),
                                 X['pw25'] - X['pw20'], pop['PW25'] * 1e4 - pop['PW20'] * 1e4, (X['swide'] < X['pw20']).astype(float)]), wg)
    add('priv_prior', np.column_stack([qL, qI, qM, stk, (stk < 0).astype(float), (qM < 0).astype(float), X['pw25'] - X['pw20']]), wg, pri)

# ---------------- 3. report ----------------
P('MODEL apick_mc.py | tag %s | world skeptic_wc.py (dose_mc prox %s sh %s s %s field %s; lomm05 %s, MCLmm %s%s; rhoS %.2f rhoA %.2f nf %.2f pf %.2f; k N(%.2f, %.2f); c4shift %+.1f) | rank ruler %s | N %d' % (
    TAG, D.PROX, D.PSH, D.PS, D.FS, g['LOMM'], g['MCLP'], (' -> override ' + MALT) if MALT else '', g['RHOS'], g['RHOA'], g['NF'], g['PF'], D.K_MU, D.K_SD, C4SH,
    'on' if RANK else 'off', N))
P('Conditioning (MEASURED read 10-21 17:33 IST): P(c1) %.3f | P(all hard) %.3f | ESS %.0f' % (c1.mean(), hard.mean(), ESS))
# public decomposition
muL, I_, muM, eLp, eMp = g['muL'], g['I'], g['muM'], g['eLp'], g['eMp']
gap = PUB['swide'] - PUB['pw25']
pr = lambda x: (pri * x).sum(); po = lambda x: (wt * x).sum()
P('\n== 1. PUBLIC swide - pw25 gap (e-4): population components vs read noise, prior -> posterior ==')
for nm, x in (('lomm05 pop muL', muL), ('interaction I', I_), ('MCLmm pop muM', muM), ('lomm05 read noise eLp', eLp), ('MCLmm read noise eMp', eMp), ('TOTAL gap', gap)):
    P('  %-24s prior %+5.2f  ->  posterior %+5.2f  (share of posterior gap %4.0f %%)' % (nm, pr(x), po(x), 100 * po(x) / po(gap)))
P('  population part (muL + I + muM): posterior %+.2f = %.0f %% of the gap; noise part %+.2f = %.0f %%' % (
    po(muL + I_ + muM), 100 * po(muL + I_ + muM) / po(gap), po(eLp + eMp), 100 * po(eLp + eMp) / po(gap)))
P('  P(muM > 0) prior %.2f -> post %.2f | P(muL + I > 0) prior %.2f -> post %.2f | P(muL + I + muM > 0) prior %.2f -> post %.2f' % (
    pr(muM > 0), po(muM > 0), pr(muL + I_ > 0), po(muL + I_ > 0), pr(muL + I_ + muM > 0), po(muL + I_ + muM > 0)))
P('  public pw25 - pw30 posterior %+.2f (prior %+.2f) | pos swide %.2f pw25 %.2f pw30 %.2f' % (po(PUB['pw25'] - PUB['pw30']), pr(PUB['pw25'] - PUB['pw30']),
                                                                                   po(ps_), po(p25), po(p30)))
pv = acc['priv']; pp = acc['priv_prior']
P('\n== 2. PRIVATE (scenario mix; pre-read prior -> posterior) ==')
P('  lomm05 qL %+.2f -> %+.2f | I %+.2f -> %+.2f | MCLmm qM %+.2f -> %+.2f | stack total incl. private noise %+.2f -> %+.2f' % (
    pp[0], pv[0], pp[1], pv[1], pp[2], pv[2], pp[3], pv[5]))
P('  P(stack < 0) %.2f -> %.2f | P(MCLmm < 0) %.2f -> %.2f | P(lomm05 + I < 0) post %.2f | private noise sd: lomm05 %.2f, MCLmm %.2f (per draw)' % (
    pp[4], pv[6], pp[5], pv[7], pv[8], g['eLq'].std(), g['eMq'].std()))
P('  dose: pw25 - pw20 private %+.2f (prior %+.2f), population part %+.2f | P(swide < pw20 privately) %.2f' % (pv[9], pp[6], pv[10], pv[11]))

P('\n== 3. A candidates alone: E (sd) | P(A < copy-block, margin<0) | pairwise P(row > col) private ==')
hdr = '                ' + ' '.join('%7s' % a for a in AL)
P(hdr)
for a in AL:
    m = acc['mom|' + a]; sd = np.sqrt(m[1] - m[0] ** 2)
    P('  %-7s E %+6.2f (sd %5.2f) P<0 %.3f | %s | dE vs swide %+.2f' % (a, m[0], sd, m[2], ' '.join('%7.2f' % acc['gt|%s|%s' % (a, a2)][0] if a != a2 else '      -' for a2 in AL),
                                                                    acc['dE|' + a][0]))

res = dict(tag=TAG, ess=float(ESS), pairs={}, alone={}, decomp={})
P('\n== 4. PAIRS {A, B}: E e-4 | silver step / smooth | r150 step / smooth | bronze || d vs A = swide (pts; E) ==')
for b in BL:
    P('  B = %s' % b)
    base = acc['P|swide|%s' % b]
    for a in AL:
        p = acc['P|%s|%s' % (a, b)]
        res['pairs']['%s|%s' % (a, b)] = [float(v) for v in p]
        P('    A=%-6s %+6.2f | %.3f / %.3f | %.3f / %.3f | %.3f || silver %+5.1f / %+5.1f, r150 %+5.1f / %+5.1f, bronze %+5.1f, E %+.2f' % (
            a, p[0], p[1], p[3], p[2], p[5], p[4], 100 * (p[1] - base[1]), 100 * (p[3] - base[3]), 100 * (p[2] - base[2]), 100 * (p[5] - base[5]),
            100 * (p[4] - base[4]), p[0] - base[0]))
P('\n== 5. r6pw20 resubmit mix: B = r6pw20 (if clean, rule) else R6; pair metrics are linear in P(clean) for a fixed A ==')
for pc in (.35, .55, .78):
    P('  P(clean) %.2f:' % pc)
    base = pc * acc['P|swide|r6pw20 rule'] + (1 - pc) * acc['P|swide|R6']
    for a in AL:
        p = pc * acc['P|%s|r6pw20 rule' % a] + (1 - pc) * acc['P|%s|R6' % a]
        res['pairs']['%s|mix%.2f' % (a, pc)] = [float(v) for v in p]
        P('    A=%-6s %+6.2f | %.3f / %.3f | %.3f / %.3f | %.3f || vs swide silver %+5.1f / %+5.1f, r150 %+5.1f / %+5.1f' % (
            a, p[0], p[1], p[3], p[2], p[5], p[4], 100 * (p[1] - base[1]), 100 * (p[3] - base[3]), 100 * (p[2] - base[2]), 100 * (p[5] - base[5])))
for a in AL:
    m = acc['mom|' + a]; res['alone'][a] = dict(E=float(m[0]), sd=float(np.sqrt(m[1] - m[0] ** 2)), Pneg=float(m[2]),
                                               gt={a2: float(acc['gt|%s|%s' % (a, a2)][0]) for a2 in AL})
res['decomp'] = dict(gap=float(po(gap)), muL=float(po(muL)), I=float(po(I_)), muM=float(po(muM)), eLp=float(po(eLp)), eMp=float(po(eMp)),
                     pmuM=float(po(muM > 0)), pmuM_prior=float(pr(muM > 0)), priv=[float(v) for v in pv], priv_prior=[float(v) for v in pp])
json.dump(res, open(HERE + '/apick_%s.json' % TAG, 'w'), indent=1)
P('APICK_MC DONE')
LOG.close()
