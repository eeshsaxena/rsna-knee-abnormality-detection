# summarize.py (10-20 r5): compact decision matrix over the sensitivity grid (dec_<tag>.json from decision_mc.py). MODEL.
import json, glob, sys
TAGS = ['base', 'prior_orig', 'prior_ladder', 'prior_null', 'prior_neg', 'noise1.5', 'noise0.7', 'ncorr0.85', 'our_skepS6', 'our_strongS6',
        'our_r6better', 'our_tabReal', 'our_tabOver', 'our_noise1.5', 'our_bigKnobs', 'our_smallKnobs', 'field_fieldBad', 'field_Wbig',
        'field_fieldGood', 's6fdefl0.5', 's6fdefl0.0', 'ladder_noise1.5', 'neg_noise1.5']
DEF = {'O20 eoa20 best-id': 'eoa20+R6', 'O10 eoa10 best-id': 'eoa10+R6', 'O0 no move': 'c4+R6', 'P10+ only eoa10 scored, moved': 'eoa10+R6',
       'P10- only eoa10 scored, no move': 'c4+R6', 'P20+ only eoa20 scored, moved': 'eoa20+R6', 'P20- only eoa20 scored, no move': 'c4+R6',
       'U unread (prior)': 'c4+R6'}
ALT = {'O20 eoa20 best-id': ['eoa10+R6', 'c4+eoa20', 'eoa20+HW', 'eoa20+S6F'], 'O10 eoa10 best-id': ['eoa20+R6', 'c4+eoa10', 'eoa10+HW', 'eoa10+S6F'],
       'O0 no move': ['eoa10+R6', 'c4+eoa10', 'c4+HW', 'c4+S6F'], 'P10+ only eoa10 scored, moved': ['c4+eoa10', 'eoa10+HW', 'eoa10+S6F'],
       'P10- only eoa10 scored, no move': ['eoa10+R6', 'c4+HW', 'c4+S6F'], 'P20+ only eoa20 scored, moved': ['c4+eoa20', 'eoa20+HW', 'eoa20+S6F'],
       'P20- only eoa20 scored, no move': ['c4+HW', 'c4+S6F'], 'U unread (prior)': ['eoa10+R6', 'eoa10+S6F', 'c4+S6F']}
R5 = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/r5'
for o in DEF:
    print(f'\n### {o}  (default {DEF[o]}; columns = silver; [best pair excl. S6F]; deltas vs default in pts)')
    for t in TAGS:
        try: J = json.load(open(f'{R5}/dec_{t}.json'))
        except FileNotFoundError: continue
        O = J['outcomes'][o]; P = O['pairs']; d = P[DEF[o]]['silver']
        nos = sorted([(k, v['silver']) for k, v in P.items() if '+' in k and 'S6F' not in k], key=lambda kv: -kv[1])[0]
        best = sorted([(k, v['silver']) for k, v in P.items() if '+' in k], key=lambda kv: -kv[1])[0]
        alts = ' '.join(f'{a} {100*(P[a]["silver"]-d):+.1f}' for a in ALT[o])
        print(f'{t:16s} P {O["P"]:.2f} d10 {O["d10"][0]*1e4:+.1f} ({O["d10"][2]:.2f}) d20 {O["d20"][0]*1e4:+.1f} ({O["d20"][2]:.2f}) | def {d:.3f} '
              f'r200 {P[DEF[o]]["r200"]:.2f} | best-noS6F {nos[0]} {100*(nos[1]-d):+.1f} | best {best[0]} {100*(best[1]-d):+.1f} | {alts}')
