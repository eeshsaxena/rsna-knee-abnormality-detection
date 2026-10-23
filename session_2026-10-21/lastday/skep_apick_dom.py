# skep_apick_dom.py (skeptic of apick; MODEL): rows where one alt A beats swide on ALL of silver step/smooth, r150 step/smooth, E (B = R6),
# plus that row's posterior private stack; and the median (alt - swide) over the 26 rows.
import json, numpy as np
HERE = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/lastday'
ORDER = ['base', 'base_norank', 'lskep', 'mstack', 'rhoS0', 'nf15', 'pf5', 'flatU', 'PAREA', 'kpess', 'fieldBad', 'fieldGood', 'c4over', 'c4real',
         'mzero', 'worst', 'rhoA0', 'skepall', 'skepallnf2', 'pure', 'base_mneg', 'base_meb', 'worst_meb', 'worstA0_meb', 'flatU_meb', 'PAREA_meb']
R = {t: json.load(open(HERE + '/apick_%s.json' % t)) for t in ORDER}
for B in ('R6', 'mix0.55'):
    print('B =', B)
    D = {a: [] for a in ('pw25', 'pw20', 'pw30', 'eoa20')}
    for t, r in R.items():
        sw = np.array(r['pairs']['swide|%s' % B])
        for a in D:
            d = np.array(r['pairs']['%s|%s' % (a, B)]) - sw; D[a].append(d)
            if all(d[i] > 0 for i in (0, 1, 2, 3, 5)):
                print('  dominated: %-12s by %-5s stack %+.2f | d silv %+.1f/%+.1f r150 %+.1f/%+.1f E %+.2f' % (t, a, r['decomp']['priv'][5], 100*d[1], 100*d[3], 100*d[2], 100*d[5], d[0]))
    for a, v in D.items():
        v = np.array(v); print('  median(alt - swide) %-5s silv step %+.2f smooth %+.2f r150 step %+.2f smooth %+.2f bronze %+.2f E %+.2f' % (
            a, 100*np.median(v[:, 1]), 100*np.median(v[:, 3]), 100*np.median(v[:, 2]), 100*np.median(v[:, 5]), 100*np.median(v[:, 4]), np.median(v[:, 0])))
