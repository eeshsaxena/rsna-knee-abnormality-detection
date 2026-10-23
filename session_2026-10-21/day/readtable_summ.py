# readtable_summ.py (10-21, Claude, DAY track). MODEL summary over readtable_<tag>.json (no new simulation).
# Metric vector order: E, silver_step, r150_step, silver_smooth, bronze_smooth, r150_smooth.
import json, os
HERE = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/day'
TAGS = ['base', 'lskep', 'mzero', 'lskepmzero', 'worst', 'mstack', 'flatU', 'PAREA']
WIN = ['pw30', 'pw25', 'spl', 'swide', 'salt']
L = open(HERE + '/readtable_summ.out', 'w')


def P(s=''):
    print(s); L.write(s + '\n')


f3 = lambda v: '%+.2f %.3f/%.3f b%.3f r%.3f' % (v[0], v[1], v[3], v[4], v[5])
P('MODEL readtable_summ.py | per variant: P(best-id), pair metrics E / silver step / silver smooth / bronze smooth / r150 smooth')
for t in TAGS:
    p = HERE + '/readtable_%s.json' % t
    if not os.path.exists(p):
        continue
    J = json.load(open(p)); M = J['marg']; a = M['all']
    P('\n== %s (lomm05 %s, MCLmm %s)' % (t, J['lomm'], J['mcl']))
    P('   P(best-id) ' + ' / '.join('%s %.3f' % (w, M[w]['P']) for w in WIN) + ' | P(.960 shown) %.2f' % sum(M[w]['P'] * M[w]['P960'] for w in WIN))
    P('   ALL  R1 %s | PF %s | PF-R1 silver %+.1f step / %+.1f smooth, bronze %+.1f, r150 %+.1f' % (
        f3(a['P_R1']), f3(a['P_PF']), 100 * (a['P_PF'][1] - a['P_R1'][1]), 100 * (a['P_PF'][3] - a['P_R1'][3]), 100 * (a['P_PF'][4] - a['P_R1'][4]), 100 * (a['P_PF'][5] - a['P_R1'][5])))
    P('        pw30 fixed %s | pw25 fixed %s' % (f3(a['P_pw30fix']), f3(a['P_pw25fix'])))
    for w in WIN:
        r = M[w]
        P('   %-6s P %.3f | R1 %s | PF %s | PF-R1 silver %+.1f / %+.1f' % (w, r['P'], f3(r['P_R1']), f3(r['P_PF']), 100 * (r['P_PF'][1] - r['P_R1'][1]), 100 * (r['P_PF'][3] - r['P_R1'][3])))
    for nm, sw in (('H1 = PF only if pw30 stays best-id', {'pw30'}), ('H2 = PF if best-id in {pw30, pw25}', {'pw30', 'pw25'})):
        v = [sum(M[w]['P'] * (M[w]['P_PF'][i] if w in sw else M[w]['P_R1'][i]) for w in WIN) for i in range(6)]
        P('   %-38s %s | vs R1 silver %+.1f / %+.1f | vs PF %+.1f / %+.1f' % (nm, f3(v), 100 * (v[1] - a['P_R1'][1]), 100 * (v[3] - a['P_R1'][3]),
                                                                            100 * (v[1] - a['P_PF'][1]), 100 * (v[3] - a['P_PF'][3])))
    for d in ('D957', 'D958', 'D959'):
        r = M[d]
        P('   r6pw20 %s P %.3f | R1 %s | PF %s | if B = r6pw20 anyway: R1 %s' % (d, r['P'], f3(r['P_R1']), f3(r['P_PF']), f3(r['P_R1_Bc'])))
    P('   r6pw20 ERROR -> B = R6: R1 %s | PF %s' % (f3(a['P_R1_err']), f3(a['P_PF_err'])))
    best = []
    for w in WIN:
        r = M[w]
        for k in r:
            if k.startswith('OPT|'):
                on = k[4:]; dv = r[k]; base = r['P_R1']
                best.append((100 * (dv[1] - base[1]), 100 * (dv[3] - base[3]), dv[0] - base[0], w, on, r['OPTsw|' + on]))
    best.sort(reverse=True)
    P('   10-22 options, best 3 (group, option: d silver step / smooth pts, dE e-4, P(new A)): ' + ' ; '.join('%s %s %+.1f/%+.1f %+.2f %.2f' % (b[3], b[4], b[0], b[1], b[2], b[5]) for b in best[:3]))
    P('   10-22 options, worst 2: ' + ' ; '.join('%s %s %+.1f/%+.1f %+.2f %.2f' % (b[3], b[4], b[0], b[1], b[2], b[5]) for b in best[-2:]))
P('READTABLE_SUMM DONE')
L.close()
