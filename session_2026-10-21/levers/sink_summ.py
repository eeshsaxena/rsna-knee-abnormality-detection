# sink_summ.py (10-21, Claude, SINK track): compact tables from sink_local.json (local aggregates only). Writes sink_summ.out.
import json, numpy as np, sys
D = json.load(open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/levers/sink_local.json'))
OUT = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/levers/sink_summ.out', 'w')
def P(s=''):
    print(s); OUT.write(s + '\n')
BASES = ['pw30', 'pw25', 'spl', 'swide', 'salt']
EX = ['CVX', 'CVXd', 'SUBH', 'S6up', 'RPF', 'E1', 'BOR', 'C6p', 'C14']
NEST = ['N_A', 'N_B', 'N_C']
RUL = ['g74', 'PUB', 'PUBM', 's-0.83', 'REP']
P('SPUB (RAK trio shift fitted to the 5 public reads) = %+.2f' % D['SPUB'])
P('\n== ladder vs c4 (e-4), public 5.9 / 10.3 / 9.6 / 10.4 / 11.3 ==')
for r in ['G58'] + RUL:
    for nn in ([None] if r == 'G58' else NEST):
        k = r if nn is None else '%s|%s' % (r, nn); lad = D[k]['ladder']
        P('  %-14s ' % k + ' '.join('%s %+6.2f' % (x, lad[x]) for x in ('eoa10', 'eoa20', 'eoa30', 'pw20', 'pw30')))
P('\n== bases vs pw30 and the A22 components (point, e-4): pw25 | spl | swide | salt | LOMM on pw25 | MCLMM on pw25 ==')
for r in ['G58'] + RUL:
    for nn in ([None] if r == 'G58' else NEST):
        k = r if nn is None else '%s|%s' % (r, nn); pt = D[k]['pt']
        P('  %-14s %+6.2f | %+6.2f | %+6.2f | %+6.2f | %+6.2f | %+6.2f' % (k, pt['pw25'], pt['spl'], pt['swide'], pt['salt'], pt['spl'] - pt['pw25'], pt['salt'] - pt['pw25']))
P('\n== marginal of each extra on each base: mean over the 3 nests [min, max], mean boot P>0 (G58: point (P)) ==')
for b in BASES:
    P('-- base %s --' % b)
    P('  %-6s | ' % 'extra' + ' | '.join('%-24s' % r for r in ['G58'] + RUL))
    for x in EX:
        cells = []
        for r in ['G58'] + RUL:
            if r == 'G58':
                m, sd, p = D['G58']['marg'][b + '+' + x]; cells.append('%+6.2f (P %.2f)          ' % (m, p))
            else:
                v = [D['%s|%s' % (r, nn)]['marg'][b + '+' + x] for nn in NEST]
                ms = [a[0] for a in v]; ps = [a[2] for a in v]
                cells.append('%+6.2f [%+5.2f,%+5.2f] %.2f' % (np.mean(ms), min(ms), max(ms), np.mean(ps)))
        P('  %-6s | ' % x + ' | '.join('%-24s' % c for c in cells))
P('\n== interactions I(x, y | base) among extras: mean over nests [min, max] (G58 point) ==')
PX = ['CVX', 'CVXd', 'SUBH', 'S6up', 'RPF', 'C6p', 'C14', 'BOR']
import itertools
for b in BASES:
    P('-- base %s --' % b)
    for x, y in itertools.combinations(PX, 2):
        if {x, y} == {'CVX', 'CVXd'}:
            continue
        k = b + '+' + x + '+' + y; cells = ['G58 %+6.2f' % D['G58']['I'][k][0]]
        for r in RUL:
            v = [D['%s|%s' % (r, nn)]['I'][k] for nn in NEST]; ms = [a[0] for a in v]; sds = [a[1] for a in v]
            cells.append('%s %+5.2f [%+5.2f,%+5.2f] sd %.2f' % (r, np.mean(ms), min(ms), max(ms), np.mean(sds)))
        if max(abs(D['%s|%s' % (r, nn)]['I'][k][0]) for r in RUL for nn in NEST) < 1e-9:
            continue
        P('  %-10s ' % (x + '*' + y) + ' | '.join(cells))
OUT.close()
