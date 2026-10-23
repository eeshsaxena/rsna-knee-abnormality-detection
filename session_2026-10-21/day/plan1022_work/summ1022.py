# summ1022.py: compact per-class table from plan1022_mc_<tag>.json (MODEL; arithmetic on the MC output only).
import sys, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1] if len(sys.argv) > 1 else 'base'
J = json.load(open(os.path.join(HERE, 'plan1022_mc_%s.json' % tag)))
MIX = {'S0': .6, 'C+': .1, 'C-': .1, 'SITE': .2}
def m(on, bo, cn):
    d = J['%s|%s|%s' % (on, bo, cn)]; return {k: sum(MIX[s] * d[s][k] for s in MIX) for k in d['S0']}
classes = []
for k in J:
    on, bo, cn = k.split('|', 2)
    if cn not in classes: classes.append(cn)
opts = []
for k in J:
    on = k.split('|')[0]
    if on not in opts: opts.append(on)
out = []
for cn in classes:
    b = m('LOCK', 'B=r6pw20', cn)
    out.append('== %-18s P %.3f | LOCK: EA %+.2f PA<0 %.2f | smooth silver %.3f bronze %.3f r150 %.3f | step silver %.3f E[rank] %.0f | P(.960) %.3f dpub %+.2f' % (
        cn, b['P'], b['EA'], b['PAlt'], b['sS'], b['brS'], b['r150S'], b['s19'], b['er19'], b['p960'], b['dpub']))
    for on in opts:
        for bo in ('B=r6pw20', 'B=R6pw25', 'B=R6'):
            if on != 'LOCK' and bo != 'B=r6pw20': continue
            if on == 'LOCK' and bo == 'B=r6pw20': continue
            if '%s|%s|%s' % (on, bo, cn) not in J: continue
            x = m(on, bo, cn)
            out.append('   %-18s %-9s dEA %+.2f | d silver sm %+.1f st %+.1f | d bronze %+.1f | d r150 %+.1f | d E[rank] %+.1f | d P(.960) %+.1f | d pub %+.2f | newA %.2f' % (
                on, bo, x['EA'] - b['EA'], 100 * (x['sS'] - b['sS']), 100 * (x['s19'] - b['s19']), 100 * (x['brS'] - b['brS']), 100 * (x['r150S'] - b['r150S']),
                x['er19'] - b['er19'], 100 * (x['p960'] - b['p960']), x['dpub'] - b['dpub'], x['newA']))
open(os.path.join(HERE, 'summ1022_%s.out' % tag), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
