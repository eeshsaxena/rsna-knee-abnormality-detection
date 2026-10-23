# grid1022.py: cross-variant grid from plan1022_mc_<tag>.json (MODEL; arithmetic on MC outputs only).
# Prints, per class and option, dEA (e-4, private A vs LOCK) / d smooth silver (pts) / d P(.960) (pts) for every variant present.
import json, os, glob
HERE = os.path.dirname(os.path.abspath(__file__))
MIX = {'S0': .6, 'C+': .1, 'C-': .1, 'SITE': .2}
TAGS = ['base', 'lskep', 'mzero', 'mstack', 'kap0', 'kap72', 'PAREA', 'flatU', 'kpess', 'lskepmzero']
J = {t: json.load(open(os.path.join(HERE, 'plan1022_mc_%s.json' % t))) for t in TAGS if os.path.exists(os.path.join(HERE, 'plan1022_mc_%s.json' % t))}
def m(t, on, bo, cn):
    k = '%s|%s|%s' % (on, bo, cn)
    if k not in J[t]: return None
    d = J[t][k]; return {q: sum(MIX[s] * d[s][q] for s in MIX) for q in d['S0']}
CL = ['ALL', 'pw30', 'pw25', 'spl', 'swide', 'salt', 'spl|m<.5', 'spl|m.5-1.5', 'spl|m>1.5', 'swide|m<.5', 'swide|m.5-1.5', 'swide|m>1.5',
      'salt|m<.5', 'salt|m.5-1.5', 'salt|m>1.5', 'no960', 'show960']
OP = [('+spl08', 'B=r6pw20'), ('+swide08', 'B=r6pw20'), ('+spl08+swide08', 'B=r6pw20'), ('+p30L', 'B=r6pw20'), ('+ssafe', 'B=r6pw20'),
      ('+smid', 'B=r6pw20'), ('+p30L+ssafe+smid', 'B=r6pw20'), ('LOCK', 'B=R6pw25')]
out = ['MODEL grid1022.py | variants: ' + ' '.join(J), 'cell = dEA e-4 / d smooth silver pts / d P(.960) pts vs LOCK with B = r6pw20 (scenario MIX)']
for cn in CL:
    b = {t: m(t, 'LOCK', 'B=r6pw20', cn) for t in J}
    out.append('== %-14s P ' % cn + ' '.join('%s %.3f' % (t, b[t]['P']) for t in J if b[t]) + ' | LOCK EA ' + ' '.join('%+.2f' % b[t]['EA'] for t in J if b[t]) +
               ' | silver(sm) ' + ' '.join('%.3f' % b[t]['sS'] for t in J if b[t]) + ' | P(.960) ' + ' '.join('%.2f' % b[t]['p960'] for t in J if b[t]))
    for on, bo in OP:
        cells = []
        for t in J:
            x = m(t, on, bo, cn)
            cells.append('%s %s' % (t, 'n/a' if x is None or b[t] is None else '%+.2f/%+.1f/%+.1f' % (x['EA'] - b[t]['EA'], 100 * (x['sS'] - b[t]['sS']), 100 * (x['p960'] - b[t]['p960']))))
        out.append('   %-18s %-9s %s' % (on, bo, ' | '.join(cells)))
out.append('== ERROR rows (class ALL): value lost if an entry is missing (LOCK minus X) or B falls back to R6')
for on, bo in [('-pw25', 'B=r6pw20'), ('-spl', 'B=r6pw20'), ('-swide', 'B=r6pw20'), ('-salt', 'B=r6pw20'), ('LOCK', 'B=R6')]:
    cells = []
    for t in J:
        x = m(t, on, bo, 'ALL'); b0 = m(t, 'LOCK', 'B=r6pw20', 'ALL')
        cells.append('%s %s' % (t, 'n/a' if x is None else '%+.2f/%+.1f/%+.1f/%+.1f' % (x['EA'] - b0['EA'], 100 * (x['sS'] - b0['sS']), 100 * (x['s19'] - b0['s19']), 100 * (x['p960'] - b0['p960']))))
    out.append('   %-8s %-9s %s   (dEA / d silver smooth / d silver step / d P(.960))' % (on, bo, ' | '.join(cells)))
open(os.path.join(HERE, 'grid1022.out'), 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
