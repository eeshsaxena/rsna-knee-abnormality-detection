# skep_apick_counts.py (skeptic of track apick, MODEL re-derivation from apick_<tag>.json; no new sims)
import json, os, numpy as np
HERE = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/lastday'
ORDER = ['base', 'base_norank', 'lskep', 'mstack', 'rhoS0', 'nf15', 'pf5', 'flatU', 'PAREA', 'kpess', 'fieldBad', 'fieldGood', 'c4over', 'c4real',
         'mzero', 'worst', 'rhoA0', 'skepall', 'skepallnf2', 'pure', 'base_mneg', 'base_meb', 'worst_meb', 'worstA0_meb', 'flatU_meb', 'PAREA_meb']
AL = ['swide', 'pw25', 'pw30', 'pw20', 'eoa20']
IDX = {'E': 0, 'silv_step': 1, 'r150_step': 2, 'silv_smooth': 3, 'bronze': 4, 'r150_smooth': 5}
R = {t: json.load(open(HERE + '/apick_%s.json' % t)) for t in ORDER}
print('variants', len(R))
for B in ('R6', 'r6pw20 rule', 'mix0.55'):
    print('B =', B)
    for mn, thr in (('silv_step', .01), ('silv_smooth', .01), ('r150_step', .01), ('r150_smooth', .01), ('E', .5), ('bronze', .01)):
        k = IDX[mn]; n_best = 0; n_within = 0; worst = []
        for t, r in R.items():
            v = {a: r['pairs']['%s|%s' % (a, B)][k] for a in AL}
            best = max(v.values()); gap = best - v['swide']
            n_best += gap <= 1e-12; n_within += gap <= thr + 1e-12; worst.append((gap, t, max(v, key=v.get)))
        worst.sort(reverse=True)
        sc = 1 if mn == 'E' else 100
        print('  %-12s best %2d/26 within(%s) %2d/26 | worst regret %s' % (mn, n_best, thr, n_within, ', '.join('%s %.2f(%s)' % (t, sc * g, a) for g, t, a in worst[:4])))
# --- skeptic-only rows (the as-built rows are near-replicas on the A axis) ---
SKEP = ['mzero', 'worst', 'rhoA0', 'skepall', 'skepallnf2', 'pure', 'base_mneg', 'base_meb', 'worst_meb', 'worstA0_meb', 'flatU_meb', 'PAREA_meb']
print('\nSKEPTIC-COMPONENT ROWS ONLY (12)')
for B in ('R6', 'mix0.55'):
    print('B =', B)
    for mn, thr in (('silv_step', .01), ('silv_smooth', .01), ('r150_step', .01), ('r150_smooth', .01), ('E', .5), ('bronze', .01)):
        k = IDX[mn]; nb = nw = 0
        for t in SKEP:
            r = R[t]; v = {a: r['pairs']['%s|%s' % (a, B)][k] for a in AL}; gap = max(v.values()) - v['swide']
            nb += gap <= 1e-12; nw += gap <= thr + 1e-12
        print('  %-12s best %2d/12 within(%s) %2d/12' % (mn, nb, thr, nw))
