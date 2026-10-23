# summarize3.py: compact cross-variant tables from abc3_primary.json / abc3_variants.json (MODEL outputs).
import json
P = json.load(open('abc3_primary.json')); V = json.load(open('abc3_variants.json'))
R = {'primary': P['primary']}; R.update({k: v for k, v in V.items() if not k.startswith('no O2')})
CS = ['PW25', 'M20_LP30', 'LP30', 'PW35', 'PW40', 'PW45', 'S40', 'S25', 'LPW30', 'S30', 'PW20', 'S20']
print('variant | ' + ' | '.join(CS))
for nm, r in R.items():
    c = r['cand']
    print('%-24s pub ' % nm[:24] + ' '.join('%+6.2f' % (c[v]['pub'][0] * 1e4) for v in CS))
    print('%-24s P>  ' % '' + ' '.join('%6.2f' % c[v]['P_gt'] for v in CS))
    print('%-24s P960' % '' + ' '.join('%6.2f' % c[v]['P960'] for v in CS))
    print('%-24s P958' % '' + ' '.join('%6.2f' % c[v]['P958'] for v in CS))
    print('%-24s prv ' % '' + ' '.join('%+6.2f' % (c[v]['prv'][0] * 1e4) for v in CS))
    print('%-24s Pp> ' % '' + ' '.join('%6.2f' % c[v]['P_prv_gt'] for v in CS))
print()
for nm, r in R.items():
    p = r['cand']['PW30']
    print('%-24s PW30 pop vs c4 %+.2f [%+.2f, %+.2f] vs eoa20 %+.2f [%+.2f, %+.2f] P(priv>c4) %.2f P(priv>eoa20) %.2f | realised overstatement vs c4 %+.2f vs eoa20 %+.2f | pw30->.960 %.2f [%.2f, %.2f]' % (
        nm[:24], p['pop_vs_c4'][0] * 1e4, p['pop_vs_c4'][1][0] * 1e4, p['pop_vs_c4'][1][2] * 1e4, p['pop_vs_eoa20'][0] * 1e4, p['pop_vs_eoa20'][1][0] * 1e4,
        p['pop_vs_eoa20'][1][2] * 1e4, p['P_prv_gt_c4'], p['P_prv_gt_eoa20'], p['realised_curse_vs_c4'] * 1e4, p['realised_curse_vs_eoa20'] * 1e4,
        r['pw30_to_960'][0] * 10, r['pw30_to_960'][1][0] * 10, r['pw30_to_960'][1][2] * 10))
print()
for nm, r in R.items():
    print('%-24s WC ' % nm[:24] + ' | '.join('%s curse %+.2f regret %+.2f Ppick=best %.2f privwin %+.2f' % (k.split(' (')[0] + ('' if '(' not in k else '(' + k.split('(')[1]), d['curse'][0] * 1e4, d['regret'][0] * 1e4, d['P_pick_is_popbest'], d['E_priv_win_vs_c4'] * 1e4) for k, d in r['wc'].items()))
print()
for nm, r in R.items():
    print('%-24s SL ' % nm[:24] + ' | '.join('%s: prv %+.2f up %.2f dn %.2f P960 %.2f' % (k[:14], d['d_prv'][0] * 1e4, d['P_prv_up'], d['P_prv_down'], d['P960']) for k, d in r['slates'].items()))
