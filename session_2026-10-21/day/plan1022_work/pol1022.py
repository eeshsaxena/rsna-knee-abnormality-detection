# pol1022.py: policy table from plan1022_mc_<tag>.json (MODEL; arithmetic on MC outputs only).
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
MIX = {'S0': .6, 'C+': .1, 'C-': .1, 'SITE': .2}
TAGS = ['base', 'lskep', 'kap72', 'mstack', 'PAREA', 'lskepmzero']
POLS = ['P1 refine(R1)', 'P1b refine, no W30 push', 'P2 public-max(R1)', 'P3 hybrid', 'P4 A-freeze public']
out = ['MODEL pol1022.py | per policy vs LOCK (B = r6pw20): dEA e-4 / d silver smooth pts / d silver step pts / d bronze smooth / d r150 smooth / d P(.960) pts / d public e-4 / P(10-22 entry is A)']
for t in TAGS:
    f = os.path.join(HERE, 'plan1022_mc_%s.json' % t)
    if not os.path.exists(f): continue
    J = json.load(open(f))
    if 'P3 hybrid|B=r6pw20|ALL' not in J: continue
    def m(on, cn, bo='B=r6pw20'):
        d = J['%s|%s|%s' % (on, bo, cn)]; return {q: sum(MIX[s] * d[s][q] for s in MIX) for q in d['S0']}
    for cn in ('ALL', 'pw30', 'pw25', 'spl', 'swide', 'salt', 'no960'):
        b = m('LOCK', cn)
        out.append('[%s] %-6s P %.3f LOCK EA %+.2f silver sm %.3f st %.3f bronze %.3f r150 %.3f P(.960) %.3f' % (t, cn, b['P'], b['EA'], b['sS'], b['s19'], b['brS'], b['r150S'], b['p960']))
        for pn in POLS:
            x = m(pn, cn)
            out.append('     %-24s %+.2f / %+.1f / %+.1f / %+.1f / %+.1f / %+.1f / %+.2f / %.2f' % (pn, x['EA'] - b['EA'], 100 * (x['sS'] - b['sS']), 100 * (x['s19'] - b['s19']),
                       100 * (x['brS'] - b['brS']), 100 * (x['r150S'] - b['r150S']), 100 * (x['p960'] - b['p960']), x['dpub'] - b['dpub'], x['newA']))
open(os.path.join(HERE, 'pol1022.out'), 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
