# merge_ph.py (10-20 ex): POST-HOC LOA-PF pair variants, aggregates from box_curve_PH_*.log JSON.
import json
import numpy as np
E = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/ex/'
RES = {'P_FT': {}, 'P_AREA': {}}
for j in ('PH_FT_A', 'PH_FT_B', 'PH_FT_C', 'PH_AREA'):
    s = open(E + 'box_curve_%s.log' % j, encoding='utf-8').read(); d = json.loads(s.split('JSON_BEGIN')[1].split('JSON_END')[0])
    RES[d['proxy']].update(d['res'])
V = ['PH_L10', 'PH_L15', 'PH_L20', 'PH_PAIR10', 'PH_PAIR15', 'PH_PAIR20', 'PH_PAIRh10', 'PH_PAIRh15', 'PH_PAIRh20', 'PW10', 'PW20', 'S0.15']
GM = ['D5', 's1044', 's1045']
for prox, R in RES.items():
    sch = ['raw', 'R1oa3', 'R3p', 'R3p_hi', 'R3p_lo'] if prox == 'P_FT' else ['raw', 'R1oa3', 'R3p']
    print('######## %s' % prox)
    for v in V:
        for s in sch:
            cells = []; npass = 0
            for m in GM:
                r = R[m][s][v]; ok = r['inc1']['m'] > r['inc1']['sd'] and r['inc2']['m'] > r['inc2']['sd']; npass += ok
                cells.append('%+.5f (%+.5f/%.5f, %+.5f/%.5f)%s' % (r['macro']['m'], r['inc1']['m'], r['inc1']['sd'], r['inc2']['m'], r['inc2']['sd'], '*' if ok else ' '))
            others = [m for m in R if m not in GM]
            oi1 = [R[m][s][v]['inc1']['m'] for m in others if s in R[m]]; oi2 = [R[m][s][v]['inc2']['m'] for m in others if s in R[m]]
            print('  %-10s %-6s %s | pass %d/3 | others(%d) inc1 mean %+.5f (#>0 %d) inc2 mean %+.5f (#>0 %d)' % (
                v, s, ' | '.join(cells), npass, len(oi1), np.mean(oi1), sum(x > 0 for x in oi1), np.mean(oi2), sum(x > 0 for x in oi2)))
