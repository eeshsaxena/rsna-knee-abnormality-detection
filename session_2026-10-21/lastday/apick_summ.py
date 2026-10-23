# apick_summ.py: summary over apick_<tag>.json variants (MODEL). Writes apick_summ.out. Run with python -B.
import json, glob, os, numpy as np
HERE = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/lastday'
ORDER = ['base', 'base_norank', 'lskep', 'mstack', 'rhoS0', 'nf15', 'pf5', 'flatU', 'PAREA', 'kpess', 'fieldBad', 'fieldGood', 'c4over', 'c4real',
         'mzero', 'worst', 'rhoA0', 'skepall', 'skepallnf2', 'pure', 'base_mneg', 'base_meb', 'worst_meb', 'worstA0_meb', 'flatU_meb', 'PAREA_meb']
SKEP = set(['mzero', 'worst', 'rhoA0', 'skepall', 'skepallnf2', 'pure', 'base_mneg', 'base_meb', 'worst_meb', 'worstA0_meb', 'flatU_meb', 'PAREA_meb'])
R = {}
for t in ORDER:
    f = HERE + '/apick_%s.json' % t
    if os.path.exists(f):
        R[t] = json.load(open(f))
out = open(HERE + '/apick_summ.out', 'w')


def P(s=''):
    print(s); out.write(s + '\n')


AL = ['swide', 'pw25', 'pw30', 'pw20', 'eoa20']
IDX = {'E': 0, 'silv_step': 1, 'r150_step': 2, 'silv_smooth': 3, 'bronze': 4, 'r150_smooth': 5}
P('apick_summ (MODEL). %d variants. Group S = skeptic-component worlds (stack effect null / negative / not carried to private).' % len(R))
P('\n== 1. decomposition of the public swide - pw25 gap (posterior, e-4) and the private stack ==')
P('  %-12s %s | gap  | muL   I     muM  | noise | P(muM>0) pri->post | priv stack E, P(<0) | P(swide<pw20)' % ('variant', 'grp'))
for t, r in R.items():
    d = r['decomp']; pv = d['priv']
    P('  %-12s %s  | %+.2f | %+.2f %+.2f %+.2f | %+.2f | %.2f -> %.2f | %+.2f, %.2f | %.2f' % (
        t, 'S' if t in SKEP else ' ', d['gap'], d['muL'], d['I'], d['muM'], d['eLp'] + d['eMp'], d['pmuM_prior'], d['pmuM'], pv[5], pv[6], pv[11]))
for B in ('R6', 'r6pw20 rule', 'mix0.55'):
    P('\n== 2. B = %s: pair metrics for A = swide, and alt A minus swide (pts; E e-4) ==' % B)
    P('  %-12s g | swide: silv step/smooth, r150 step/smooth, E | pw25: silv st/sm, r150 st/sm, E | pw30: same | pw20: same | eoa20: same' % 'variant')
    for t, r in R.items():
        p = r['pairs']; sw = np.array(p['swide|%s' % B])
        cells = []
        for a in AL[1:]:
            d = np.array(p['%s|%s' % (a, B)]) - sw
            cells.append('%+4.1f/%+4.1f %+4.1f/%+4.1f %+5.2f' % (100 * d[1], 100 * d[3], 100 * d[2], 100 * d[5], d[0]))
        P('  %-12s %s | %.3f/%.3f %.3f/%.3f %+6.2f | %s' % (t, 'S' if t in SKEP else ' ', sw[1], sw[3], sw[2], sw[5], sw[0], ' | '.join(cells)))
    # counts and regret
    P('  -- counts: rows where swide >= alt (all / as-built / skeptic) and mean / worst (alt - swide), pts --')
    for a in AL[1:]:
        line = '    vs %-6s' % a
        for mn in ('silv_step', 'silv_smooth', 'r150_step', 'r150_smooth', 'bronze', 'E'):
            k = IDX[mn]
            dd = {t: (np.array(r['pairs']['%s|%s' % (a, B)]) - np.array(r['pairs']['swide|%s' % B]))[k] for t, r in R.items()}
            al = np.array(list(dd.values())); sk = np.array([v for t, v in dd.items() if t in SKEP]); ab = np.array([v for t, v in dd.items() if t not in SKEP])
            sc = 1 if mn == 'E' else 100
            line += ' | %s %d/%d (%d/%d, %d/%d) mean %+.2f max %+.2f' % (mn, (al <= 1e-9).sum(), len(al), (ab <= 1e-9).sum(), len(ab), (sk <= 1e-9).sum(), len(sk),
                                                                     sc * al.mean(), sc * al.max())
        P(line)
    P('  -- best A per row (silver step / silver smooth / r150 step / r150 smooth / E) --')
    for t, r in R.items():
        bests = []
        for mn in ('silv_step', 'silv_smooth', 'r150_step', 'r150_smooth', 'E'):
            k = IDX[mn]; v = {a: r['pairs']['%s|%s' % (a, B)][k] for a in AL}; bests.append(max(v, key=v.get))
        P('    %-12s %s' % (t, ' / '.join(bests)))
P('\n== 3. A alone: E, sd, P(<0) per variant ==')
for t, r in R.items():
    P('  %-12s ' % t + ' | '.join('%s %+.2f (%.1f) %.3f' % (a, r['alone'][a]['E'], r['alone'][a]['sd'], r['alone'][a]['Pneg']) for a in AL))
out.close()
