# bpick_summ.py: summarise bpick_<tag>.json over variants (MODEL). Writes bpick_summ.out next to it.
import json, glob, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
out = open(HERE + '/bpick_summ.out', 'w')


def P(s=''):
    print(s); out.write(s + '\n')


fs = sorted(glob.glob(HERE + '/bpick_*.json'))
R = {json.load(open(f))['tag']: json.load(open(f)) for f in fs}
order = ['base'] + [t for t in R if t != 'base']
FREE = ['R6', 'HW', 'c4', 'eoa10', 'eoa20', 'pw20', 'pw25', 'pw30']
ALL = FREE + ['r6pw20 rule']
IDX = {'E': 0, 'silv_step': 1, 'r150_step': 2, 'silv_smooth': 3, 'bronze': 4, 'r150_smooth': 5}
P('MODEL bpick_summ.py | A = swide 57464326 | variants: ' + ', '.join('%s (ESS %.0f)' % (t, R[t]['ess']) for t in order))
for mk in ('silv_step', 'silv_smooth', 'r150_step', 'r150_smooth', 'bronze', 'E'):
    i = IDX[mk]; sc = 1.0 if mk == 'E' else 100.0
    P('\n== %s (pair %s; %s) ==' % (mk, 'e-4' if mk == 'E' else 'pts', 'free B columns, then r6pw20 clean'))
    P('%-12s | A alone | ' % 'variant' + ' | '.join('%7s' % b[:7] for b in ALL) + ' || best free | r6pw20(clean) - R6 | eoa10 - R6 | pw20 - R6')
    for t in order:
        r = R[t]; v = {b: sc * r['B'][b]['pair'][i] for b in ALL}
        bf = max(FREE, key=lambda b: v[b])
        P('%-12s | %7.2f | ' % (t, sc * r['A'][i]) + ' | '.join('%7.2f' % v[b] for b in ALL) + ' || %-6s | %+5.2f | %+5.2f | %+5.2f' % (
            bf, v['r6pw20 rule'] - v['R6'], v['eoa10'] - v['R6'], v['pw20'] - v['R6']))
P('\n== Regret among FREE B (mean / max over variants; pts, E e-4) and rows best ==')
for b in FREE:
    row = []
    for mk in ('silv_step', 'silv_smooth', 'r150_step', 'r150_smooth', 'E'):
        i = IDX[mk]; sc = 1.0 if mk == 'E' else 100.0
        rg = [sc * (max(R[t]['B'][x]['pair'][i] for x in FREE) - R[t]['B'][b]['pair'][i]) for t in order]
        nb = sum(1 for t in order if R[t]['B'][b]['pair'][i] >= max(R[t]['B'][x]['pair'][i] for x in FREE) - 1e-12)
        row.append('%s %.2f/%.2f (%d)' % (mk, np.mean(rg), np.max(rg), nb))
    P('  %-6s ' % b + ' | '.join(row))
P('\n== corr(A, B) private and hedge E[max - A] (base) ==')
for b in ALL:
    x = R['base']['B'][b]
    P('  %-12s corr %.2f | P(B > A) %.2f | E[max - A] %+.2f | given E-OA<0 %+.2f | given MCLmm<0 %+.2f' % (b, x['corr'], x['pBgtA'], x['hedge'], x['hedge_lowE'], x['hedge_negM']))
P('\n== r6pw20 resubmit value vs B = R6, per variant at P(clean) = 1 (multiply by P(clean)) ==')
for t in order:
    r = R[t]; d = [r['B']['r6pw20 rule']['pair'][k] - r['B']['R6']['pair'][k] for k in range(6)]
    P('  %-12s E %+.2f | silver step %+.2f smooth %+.2f | r150 step %+.2f smooth %+.2f | bronze %+.2f' % (
        t, d[0], 100 * d[1], 100 * d[3], 100 * d[2], 100 * d[5], 100 * d[4]))
out.close()
