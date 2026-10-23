# Summarises cand_oam_box.log (aggregates JSON) into compact markdown tables (macro increments over E-OA(e) alone).
import json
E = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/ex/'
s = open(E + 'cand_oam_box.log', encoding='utf-8').read()
J = json.loads(s.split('JSON_BEGIN')[1].split('JSON_END')[0])
M = ['D5mean_s1043', 's1044_mix50', 's1045_288']
f = lambda v: '%+.5f' % v
for lab in J:
    print('\n### %s' % lab)
    schemes = list(J[lab][M[0]].keys())
    print('\nE-OA alone (macro vs raw), 3 models:')
    for sn in schemes:
        print('| %s | %s | %s |' % (sn, ' / '.join(f(J[lab][m][sn]['EOA0.1'][0]) for m in M), ' / '.join(f(J[lab][m][sn]['EOA0.2'][0]) for m in M)))
    for part in ('BOTH', 'OAside', 'MENside', 'medial_pair', 'lateral_pair', 'BOTH_Epartner'):
        print('\n%s increment over E-OA(e) (macro, D5 / s1044 / s1045; P>0 min..max)' % part)
        print('| scheme | e | a .1 | a .2 | a -.1 (control) | a -.2 |')
        print('|---|---|---|---|---|---|')
        for sn in schemes:
            for e in ('0.1', '0.2'):
                cells = []
                for a in ('+0.1', '+0.2', '-0.1', '-0.2'):
                    r = [J[lab][m][sn]['e%s a%s' % (e, a)][part] for m in M]
                    cells.append('%s (P %.2f-%.2f)' % (' / '.join(f(x[0]) for x in r), min(x[1] for x in r), max(x[1] for x in r)))
                print('| %s | %s | %s |' % (sn, e, ' | '.join(cells)))
    print('\nper-label AUC deltas (x1, not macro) at e .1, a .1 and a -.1, D5 / s1044 / s1045:')
    for sn in schemes:
        for a in ('+0.1', '-0.1'):
            pl = [J[lab][m][sn]['e0.1 a%s' % a]['per_label_x12'] for m in M]
            print('| %s | a %s | %s |' % (sn, a, ' | '.join('%s %s' % (k, '/'.join('%+.4f' % p[k] for p in pl)) for k in ('MOA', 'LOA', 'MM', 'LM'))))
