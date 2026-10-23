# ranges3.py: primary values + cross-variant ranges for implied2.md tables (reads abc3_primary.json / abc3_variants.json). MODEL.
# Variants exclude 'no O2' (ignores pw30's .959 display) and the MC seed check. Output: ranges3.log.
import json
P = json.load(open('abc3_primary.json')); V = json.load(open('abc3_variants.json'))
R = {'primary': P['primary']}; R.update({k: v for k, v in V.items() if not k.startswith('no O2') and not k.startswith('seed')})
pf = {k: v for k, v in R.items() if 'AREA' not in k}
for v in ('PW25', 'M20_LP30', 'LP30', 'PW35', 'PW40', 'PW45', 'S40', 'PW20', 'S20', 'S30', 'S25', 'LPW30'):
    p = R['primary']['cand'][v]
    rng = lambda key, sub=None, sc=1e4, src=R: (min((x['cand'][v][key][0] if sub is None else x['cand'][v][key]) * sc for x in src.values()),
                                                max((x['cand'][v][key][0] if sub is None else x['cand'][v][key]) * sc for x in src.values()))
    r1 = rng('pub'); r2 = rng('prv'); r3 = rng('P_gt', 1, 1); r4 = rng('P960', 1, 1); r5 = rng('P_prv_gt', 1, 1); r6 = rng('P958', 1, 1)
    q1 = rng('prv', None, 1e4, pf)
    print('%-9s PUB %+.2f [%+.2f, %+.2f] (var %+.2f..%+.2f) P> %.2f (%.2f..%.2f) P960 %.2f (%.2f..%.2f) P958 %.2f (%.2f..%.2f) | PRIV %+.2f [%+.2f, %+.2f] (var %+.2f..%+.2f; P_FT-only %+.2f..%+.2f) P> %.2f (%.2f..%.2f) | POP %+.2f [%+.2f, %+.2f]' % (
        v, p['pub'][0] * 1e4, p['pub'][1][0] * 1e4, p['pub'][1][2] * 1e4, r1[0], r1[1], p['P_gt'], r3[0], r3[1], p['P960'], r4[0], r4[1], p['P958'], r6[0], r6[1],
        p['prv'][0] * 1e4, p['prv'][1][0] * 1e4, p['prv'][1][2] * 1e4, r2[0], r2[1], q1[0], q1[1], p['P_prv_gt'], r5[0], r5[1], p['pop'][0] * 1e4, p['pop'][1][0] * 1e4, p['pop'][1][2] * 1e4))
r = R['primary']
for e in ('c4', 'eoa10', 'eoa20', 'eoa30', 'pw20', 'pw30'):
    x = r['pub_x'][e]; a = r['pub_abs_trunc'][e]
    print('%-6s x %.3f [%.3f, %.3f] | trunc %.5f [%.5f, %.5f] | to .960 (e-4) %.2f [%.2f, %.2f]' % (e, x[0], x[1][0], x[1][2], a[0], a[1][0], a[1][2], (1 - x[0]) * 10, (1 - x[1][2]) * 10, (1 - x[1][0]) * 10))
    print('        variants trunc mean %.5f..%.5f' % (min(v['pub_abs_trunc'][e][0] for v in R.values()), max(v['pub_abs_trunc'][e][0] for v in R.values())))
for k in ('sh', 's', 'gap_c4', 'q', 'phi'):
    print(k, ' '.join('%s %.3f' % (n[:10], v[k][0]) for n, v in R.items()))
for nm in r['wc']:
    print('WC', nm, 'curse %.2f..%.2f regret %.2f..%.2f privwin-fixedPW30 %.2f..%.2f' % (
        min(v['wc'][nm]['curse'][0] for v in R.values()) * 1e4, max(v['wc'][nm]['curse'][0] for v in R.values()) * 1e4,
        min(v['wc'][nm]['regret'][0] for v in R.values()) * 1e4, max(v['wc'][nm]['regret'][0] for v in R.values()) * 1e4,
        min((v['wc'][nm]['E_priv_win_vs_c4'] - v['wc'][nm]['E_pop_fixed_PW30']) for v in R.values()) * 1e4,
        max((v['wc'][nm]['E_priv_win_vs_c4'] - v['wc'][nm]['E_pop_fixed_PW30']) for v in R.values()) * 1e4))
for nm in r['slates']:
    print('SL %-34s prv %+.2f (var %+.2f..%+.2f) Pmove %.2f P960 %.2f (%.2f..%.2f) up %.2f dn %.2f' % (nm, r['slates'][nm]['d_prv'][0] * 1e4,
        min(v['slates'][nm]['d_prv'][0] for v in R.values()) * 1e4, max(v['slates'][nm]['d_prv'][0] for v in R.values()) * 1e4, r['slates'][nm]['P_move'],
        r['slates'][nm]['P960'], min(v['slates'][nm]['P960'] for v in R.values()), max(v['slates'][nm]['P960'] for v in R.values()), r['slates'][nm]['P_prv_up'], r['slates'][nm]['P_prv_down']))
print('pre-read:', {k: P['pre-read check (O1 only)'][k] for k in ('P_outcome_class', 'pw30_minus_eoa20_pct_of_read')})
print('pre-read pw30-eoa20', P['pre-read check (O1 only)']['pub_inc_vs_eoa20']['pw30'])
