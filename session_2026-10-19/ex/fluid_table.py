# fluid_table.py: compact markdown tables from the aggregate JSON in box_fluid.log (no per-knee data exists in it).
import json, re
E = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/ex/'
txt = open(E + 'box_fluid.log', encoding='utf-8').read()
RES = json.loads(txt.split('JSON_BEGIN')[1].split('JSON_END')[0])
M = ['D5mean_s1043', 's1044_mix50', 's1045_288']
f = lambda x: '%+.5f' % x
for bn in RES:
    print('\n### %s criterion 1 (raw OAI, macro = label-sum / 12; P = paired bootstrap P>0)' % bn)
    print('| transform | D5 | s1044 | s1045 | per-label E/S/B (D5) |')
    print('|---|---|---|---|---|')
    for t in ['FL3 a-0.1', 'FL3 a+0.1', 'FL3 a+0.2', 'ES2 a-0.1', 'ES2 a+0.1', 'ES2 a+0.2']:
        cells = []
        for m in M:
            v = RES[bn][m]['raw_all'][t]
            if t.startswith('FL3'):
                cells.append('%s (%.2f)' % (f(sum(v['per']) / 12), v['P_FL3sum']))
            else:
                cells.append('%s (%.2f)' % (f(sum(v['per'][:2]) / 12), v['P_ESsum']))
        per = RES[bn][M[0]]['raw_all'][t]['per']
        print('| %s | %s | %s |' % (t, ' | '.join(cells), '/'.join('%+.4f' % x for x in per)))
    print('\n### %s criterion 2a: FL3 on complete-case knees, reweighted to gold74 fluid coupling (macro, P)' % bn)
    print('| scheme | ESS | weighted phi E-S/E-B/S-B | FL3 a-.1 (D5/s1044/s1045) | FL3 a+.1 | FL3 a+.2 |')
    print('|---|---|---|---|---|---|')
    for sn in ('raw_cc', 'R1', 'R2', 'R2hi', 'R2lo'):
        v0 = RES[bn][M[0]]['cc'][sn]
        cols = []
        for t in ('FL3 a-0.1', 'FL3 a+0.1', 'FL3 a+0.2'):
            cols.append(' / '.join('%s (%.2f)' % (f(RES[bn][m]['cc'][sn]['t'][t]['sum3'] / 12), RES[bn][m]['cc'][sn]['t'][t]['P3']) for m in M))
        print('| %s | %.0f | %s | %s |' % (sn, v0['ess'], '/'.join('%+.2f' % x for x in v0['phi']), ' | '.join(cols)))
    print('\n### %s criterion 2b: E+S set, reweighted to gold74 E-S coupling (ES-sum macro, P)' % bn)
    print('| scheme | ESS | weighted phi E-S | ES2 a-.1 | ES2 a+.1 | ES2 a+.2 | FL3 a+.1 (E,S part) |')
    print('|---|---|---|---|---|---|---|')
    for sn in ('raw', 'R1', 'R2', 'R2hi', 'R2lo'):
        v0 = RES[bn][M[0]]['es'][sn]
        cols = []
        for t in ('ES2 a-0.1', 'ES2 a+0.1', 'ES2 a+0.2', 'FL3 a+0.1'):
            cols.append(' / '.join('%s (%.2f)' % (f(RES[bn][m]['es'][sn]['t'][t]['sum2'] / 12), RES[bn][m]['es'][sn]['t'][t]['P2']) for m in M))
        print('| %s | %.0f | %+.2f | %s |' % (sn, v0['ess'], v0['phi'], ' | '.join(cols)))
