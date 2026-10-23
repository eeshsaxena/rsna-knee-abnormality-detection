# summ_global.py: compact verdict tables from box_global.log (JSON block, aggregates only) + local_global.json.
import json
E = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/ex/'
txt = open(E + 'box_global.log', encoding='utf-8').read()
R = json.loads(txt.split('JSON_BEGIN')[1].split('JSON_END')[0])
M = list(R)
SC = ['raw', 'raw_B2', 'R1oa3', 'R9', 'R9hi', 'R9lo']
print('== standalone global factor (macro-12 equiv delta vs base, P>0) ==')
for t in ('G9 -0.1', 'G9 +0.1', 'G9 +0.2', 'G12 -0.1', 'G12 +0.1', 'G12 +0.2', 'EOA 0.1', 'EOA 0.2'):
    for sc in SC:
        print('%-9s %-7s ' % (t, sc) + ' | '.join('%+.5f (P %.2f) core %+.5f' % (R[m][sc][t]['macro'], R[m][sc][t]['P'], R[m][sc][t]['core']) for m in M))
print('\n== on top of E-OA .1: increment vs EOA.1 (macro-12 equiv, P_inc), core-7 increment ==')
for t in ('E1+N9 -0.1', 'E1+N9 +0.1', 'E1+N9 +0.2', 'E1+N12 +0.1', 'E1+N12 +0.2'):
    for sc in SC:
        print('%-11s %-7s ' % (t, sc) + ' | '.join('%+.5f (P %.2f) core %+.5f (P %.2f)' % (
            R[m][sc][t]['inc'], R[m][sc][t]['Pinc'], R[m][sc][t]['inc_core'], R[m][sc][t]['Pinc_core']) for m in M))
print('\n== G9 vs EOA.1 (does the global factor beat the group-wise transform?) ==')
for sc in SC:
    print('%-7s ' % sc + ' | '.join('G9+.1 - EOA.1 %+.5f (P %.2f)' % (R[m][sc]['G9 +0.1']['inc'], R[m][sc]['G9 +0.1']['Pinc']) for m in M))
print('\nper-label (ACL, MM, LM, MOA, LOA, PF, Eff, Syn, Bak) mean deltas, E1+N9 +0.1 (non-OA part = global factor):')
for sc in ('raw', 'R9'):
    for m in M:
        print('  %-7s %-13s %s' % (sc, m, R[m][sc]['E1+N9 +0.1']['per']))
