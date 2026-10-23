# skeptic_wc_summ.py (10-21, Claude, DAY SKEPTIC track). MODEL summary over skeptic_wc_<tag>.json (no new simulation).
# P|policy vector: E pair, silver step, silver smooth, bronze smooth, r150 smooth. A|policy: E(A) - pw30. curse: pubW-pub30, privW-priv30,
# pubW-pub25, privW-priv25.
import json, os
HERE = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/day'
TAGS = ['base', 'rhoS0', 'rhoA5', 'rhoA0', 'nf15', 'pf5', 'noise', 'lskep', 'skepall', 'mstack', 'worst', 'worstsplit', 'pure', 'pureneg']
DESC = {'base': 'readtable base', 'rhoS0': 'SITE split: component private effect indep. of public (rhoS 0)',
        'rhoA5': 'rhoA .5 everywhere + rhoS 0', 'nf15': 'component read noise x1.5', 'pf5': 'component prior sd x.5',
        'noise': 'rhoS 0 + nf 1.5 + pf .5', 'lskep': 'lomm05 skeptic N(0,.9)', 'skepall': 'lomm skeptic, MCLmm 0, rhoS 0, nf 1.5',
        'mstack': 'MCLmm builder prior (optimistic)', 'worst': 'lomm05 OAI-only N(-.9,.9), MCLmm 0', 'worstsplit': 'worst + rhoS 0 + nf 1.5',
        'rhoA0': 'component private effect indep. of public everywhere', 'pure': 'pure noise: L = M = I = 0 (stacks = pw25 + read noise)',
        'pureneg': 'L = -0.9 fixed, M = 0, I = 0 (stacks lose for sure)'}
CL = ['ALL', 'pw30', 'pw25', 'spl', 'swide', 'salt', 'stack', 'stack&960', 'stack&no960']
POLS = ['R1', 'R1two {pw30,pw25}', 'PF pw20', 'H1', 'H2', 'H2s (stack only if .960)', 'pw25 fixed', 'R1+P1', 'H2+P1']
SHORT = {'R1': 'R1', 'R1two {pw30,pw25}': 'R1two', 'PF pw20': 'PF', 'H1': 'H1', 'H2': 'H2', 'H2s (stack only if .960)': 'H2s',
         'pw25 fixed': 'pw25fix', 'R1+P1': 'R1+P1', 'H2+P1': 'H2+P1'}
L = open(HERE + '/skeptic_wc_summ.out', 'w')


def P(s=''):
    print(s); L.write(s + '\n')


J = {}
for t in TAGS:
    p = HERE + '/skeptic_wc_%s.json' % t
    if os.path.exists(p):
        J[t] = json.load(open(p))
g = lambda t, k, c: J[t]['%s||%s' % (k, c)]
P('MODEL skeptic_wc_summ.py | variants: ' + ', '.join(J))
P('\n== 1. CURSE (R1 winner): public margin vs private margin over pw30 and over pw25, e-4; P(A_R1 < pw25 / pw20 private) ==')
for t in J:
    c = g(t, 'curse', 'ALL'); s = g(t, 'curse', 'stack')
    P('  %-10s ALL pub %+.2f priv %+.2f (ret %.2f) | vs pw25 pub %+.2f priv %+.2f (ret %.2f) | stack class: pub %+.2f priv %+.2f (ret %.2f) | P(A<pw25) %.2f P(A<pw20) %.2f | %s' % (
        t, c[0], c[1], c[1] / c[0], c[2], c[3], c[3] / c[2], s[0], s[1], s[1] / s[0], g(t, 'lt25|R1', 'ALL')[0], g(t, 'lt20|R1', 'ALL')[0], DESC[t]))
P('\n== 2. WHOLE PLAN by A policy (B = A22 rule): E(A)-pw30 | silver step / smooth (d vs R1 in pts) ==')
for t in J:
    r1 = g(t, 'P|R1', 'ALL')
    P('  %-10s ' % t + ' | '.join('%s %+.2f %.3f/%.3f (%+.1f/%+.1f)' % (SHORT[p], g(t, 'A|' + p, 'ALL')[0], g(t, 'P|' + p, 'ALL')[1], g(t, 'P|' + p, 'ALL')[2],
                                                                     100 * (g(t, 'P|' + p, 'ALL')[1] - r1[1]), 100 * (g(t, 'P|' + p, 'ALL')[2] - r1[2])) for p in POLS))
P('\n== 3. Per class: PF - R1 and H2s - R1 (silver step / smooth pts; E(A) diff) ==')
for t in J:
    P('  %-10s ' % t + ' | '.join('%s PF %+.1f/%+.1f E%+.2f H2s %+.1f/%+.1f' % (
        c, 100 * (g(t, 'P|PF pw20', c)[1] - g(t, 'P|R1', c)[1]), 100 * (g(t, 'P|PF pw20', c)[2] - g(t, 'P|R1', c)[2]),
        g(t, 'A|PF pw20', c)[0] - g(t, 'A|R1', c)[0],
        100 * (g(t, 'P|H2s (stack only if .960)', c)[1] - g(t, 'P|R1', c)[1]), 100 * (g(t, 'P|H2s (stack only if .960)', c)[2] - g(t, 'P|R1', c)[2]))
        for c in ('spl', 'swide', 'salt', 'stack', 'stack&no960')))
P('\n== 3b. H2 - R1, PF - H2, H2+P1 - H2 whole plan (silver step / smooth pts; E(A) diff); E(A_R1) - E(pw25 fixed) ==')
for t in J:
    d = lambda p, q: (100 * (g(t, 'P|' + p, 'ALL')[1] - g(t, 'P|' + q, 'ALL')[1]), 100 * (g(t, 'P|' + p, 'ALL')[2] - g(t, 'P|' + q, 'ALL')[2]),
                      g(t, 'A|' + p, 'ALL')[0] - g(t, 'A|' + q, 'ALL')[0])
    P('  %-10s H2-R1 %+.1f/%+.1f E%+.2f | PF-H2 %+.1f/%+.1f E%+.2f | H2+P1-H2 %+.1f/%+.1f E%+.2f | R1-pw25fix %+.1f/%+.1f E%+.2f' % (
        (t,) + d('H2', 'R1') + d('PF pw20', 'H2') + d('H2+P1', 'H2') + d('R1', 'pw25 fixed')))
P('\n== 4. 10-22 P1 entries (A-eligible): R1+P1 - R1 and H2+P1 - H2, by class: E(A) | silver step / smooth pts ==')
for t in J:
    out = []
    for c in ('ALL', 'pw30', 'spl', 'swide'):
        a = g(t, 'A|R1+P1', c)[0] - g(t, 'A|R1', c)[0]; p = g(t, 'P|R1+P1', c); b = g(t, 'P|R1', c)
        out.append('%s %+.2f %+.1f/%+.1f' % (c, a, 100 * (p[1] - b[1]), 100 * (p[2] - b[2])))
    a = g(t, 'A|H2+P1', 'ALL')[0] - g(t, 'A|H2', 'ALL')[0]; p = g(t, 'P|H2+P1', 'ALL'); b = g(t, 'P|H2', 'ALL')
    out.append('H2 base: ALL %+.2f %+.1f/%+.1f' % (a, 100 * (p[1] - b[1]), 100 * (p[2] - b[2])))
    P('  %-10s ' % t + ' | '.join(out))
P('\n== 5. B check: d silver step / smooth (pts) of each B option vs the A22 B rule, per class (worst over classes in brackets) ==')
for t in J:
    for A in ('R1', 'H2', 'PF'):
        opts = sorted(set(k.split('||')[0].split('|', 2)[2] for k in J[t] if k.startswith('B|%s|' % A)))
        cells = []
        for bn in opts:
            if bn in ('rule', 'r6pw20 always'):
                continue
            ds = [(100 * (g(t, 'B|%s|%s' % (A, bn), c)[1] - g(t, 'B|%s|rule' % A, c)[1]), 100 * (g(t, 'B|%s|%s' % (A, bn), c)[2] - g(t, 'B|%s|rule' % A, c)[2]))
                  for c in CL if '%s||%s' % ('B|%s|%s' % (A, bn), c) in J[t]]
            if not ds:
                continue
            cells.append('%s best %+.1f/%+.1f' % (bn, max(d[0] for d in ds), max(d[1] for d in ds)))
        P('  %-10s A=%-3s ' % (t, A) + ' | '.join(cells))
L.close()
