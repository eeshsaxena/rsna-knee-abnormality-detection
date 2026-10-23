#!/usr/bin/env python
# summ_sim.py SIM.json: compact tables from sim_pl.py (gains in 1e-4 macro AUC; MODEL numbers).
import sys, json
d = json.load(open(sys.argv[1]))
A = d['agg']
print('worlds per scenario', d['worlds'], '| npop', d['npop'], '| seconds %.0f' % d['seconds'])
cols = [('V224', 'prior2', '2'), ('V224', 'prior2', '4'), ('V224', 'group', '6'), ('V224', 'group', '12'), ('V224', 'group', '24'),
        ('V224', 'group', '35'), ('V224', 'group+shrink', '35'), ('V224', 'label', '35'), ('V224', 'label+shrink', '35'),
        ('V224', 'pubopt', '-'), ('V224', 'oracle', '-'),
        ('V224F20', 'prior2', '4'), ('V224F20', 'group', '12'), ('V224F20', 'group', '35'), ('V224F20', 'label', '35'),
        ('V224F20', 'oracle', '-')]
for sc, v in A.items():
    print('\n### %s %s' % (sc, json.dumps(d['scenarios'][sc])))
    b = v.get('V224|base_level|-')
    if b:
        print('base level (V224): pub %.4f prv %.4f pop %.4f' % (b['pub'], b['prv'], b['pop']))
    print('%-26s %7s %7s %7s %7s %7s %7s %6s %6s %6s %6s' % ('policy|budget', 'pub', 'prv', 'pop', 'prv_q10', 'P(prv<0)', 'P(pop<0)',
                                                             'acc', 'P957', 'P958', 'P960'))
    for base, pol, B in cols:
        k = f'{base}|{pol}|{B}'
        if k not in v:
            continue
        r = v[k]
        print('%-26s %+7.1f %+7.1f %+7.1f %+7.1f %7.2f %7.2f %6.1f %6.2f %6.2f %6.3f' % (
            k, r['pub'] * 1e4, r['prv'] * 1e4, r['pop'] * 1e4, r['prv_q10'] * 1e4, r['P_prv_lt0'], r['P_pop_lt0'], r['acc'],
            r['P_disp957'], r['P_disp958'], r['P_disp960']))
    for g in ('OA', 'TRAUMA'):
        p = v.get('probe1_' + g)
        if p:
            print('  first probe %-6s n %3d P(win) %.2f | dpub %+.1f sd %.1f | dpop %+.1f sd %.1f | P(pop>0|win) %s P(pop>0|loss) %s' % (
                g, p['n'], p['P_win'], p['dpub_mean'] * 1e4, p['dpub_sd'] * 1e4, p['dpop_mean'] * 1e4, p['dpop_sd'] * 1e4,
                '%.2f' % p['P_pop_gt0_given_win'] if p['P_pop_gt0_given_win'] is not None else '-',
                '%.2f' % p['P_pop_gt0_given_loss'] if p['P_pop_gt0_given_loss'] is not None else '-'))
