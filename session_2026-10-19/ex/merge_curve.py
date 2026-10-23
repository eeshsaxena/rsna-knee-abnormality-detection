# merge_curve.py (10-20 ex): parse box_curve_*.log JSON (aggregates only) and apply the pre-registered rule (cand_eoa_curve.md sec 0).
import json, re
import numpy as np
E = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/ex/'
R = {}
for j in ('D5', 's1044', 's1045', 'ENS3', 'FOLDS', 'AREA'):
    s = open(E + 'box_curve_%s.log' % j, encoding='utf-8').read()
    R[j] = json.loads(s.split('JSON_BEGIN')[1].split('JSON_END')[0])
GM = ['D5', 's1044', 's1045']
VARS = ['S0.05', 'S0.10', 'S0.15', 'S0.20', 'S0.25', 'S0.30', 'S-0.10', 'LP10', 'LP15', 'LP20', 'RAMP', 'RAMP2', 'PFH', 'PFH2', 'LH',
        'MH_ctrl', 'PW10', 'PW20', 'ORACLE']
REF = ['S0.10', 'S0.20']
get = lambda m, s, src=None: (R[src]['res'][m][s] if src else R[m]['res'][m][s])

print('=== 1. symmetric curve, macro delta vs untransformed (P>0), P_FT, B 400 ===')
for s in ('raw', 'R1oa3', 'R3p', 'R3p_hi', 'R3p_lo'):
    print('-- scheme', s)
    for m in GM + ['ENS3']:
        r = get(m, s); row = ' '.join('%s %+.5f(%.2f)' % (a[1:], r[a]['macro']['m'], r[a]['macro']['P']) for a in ['S-0.10', 'S0.05', 'S0.10', 'S0.15', 'S0.20', 'S0.25', 'S0.30'])
        best = max(['S0.05', 'S0.10', 'S0.15', 'S0.20', 'S0.25', 'S0.30'], key=lambda a: r[a]['macro']['m'])
        print('   %-6s %s | best %s' % (m, row, best))

print('\n=== 2. per-label symmetric curve (AUC units, mean of 400 resamples): M / L / PF ===')
for s in ('raw', 'R1oa3', 'R3p', 'R3p_hi'):
    for m in GM:
        c = get(m, s)['curve_per_label']
        print('   %-6s %-7s ' % (m, s) + ' | '.join('a %s %s' % (a, np.round(c[a], 4).tolist()) for a in ['-0.10', '0.05', '0.10', '0.15', '0.20', '0.25', '0.30']))

print('\n=== 3. refinements: increment over S.1 and S.2 (macro, mean / boot sd), PASS = inc > sd vs BOTH ===')
rule = {}
for v in VARS:
    if v in REF: continue
    rule[v] = {}
    for s in ('R1oa3', 'R3p', 'R3p_hi', 'R3p_lo', 'raw'):
        cells = []; npass = 0
        for m in GM:
            r = get(m, s)[v]; ok = r['inc1']['m'] > r['inc1']['sd'] and r['inc2']['m'] > r['inc2']['sd']; npass += ok
            cells.append('%+.5f/%.5f , %+.5f/%.5f%s' % (r['inc1']['m'], r['inc1']['sd'], r['inc2']['m'], r['inc2']['sd'], '*' if ok else ' '))
        rule[v][s] = npass
        re_ = get('ENS3', s)[v]
        print('   %-8s %-7s %s | pass %d/3 | ENS3 %+.5f , %+.5f' % (v, s, ' | '.join(cells), npass, re_['inc1']['m'], re_['inc2']['m']))
print('\n=== 3b. pre-registered rule: pass on all 3 under BOTH R1oa3 and R3p ===')
for v, d in rule.items():
    cand = d['R1oa3'] == 3 and d['R3p'] == 3
    inc = (d['R1oa3'] == 3) != (d['R3p'] == 3) or (d['R1oa3'] >= 2 and d['R3p'] >= 2 and not cand)
    print('   %-8s R1oa3 %d/3  R3p %d/3  R3p_hi %d/3  R3p_lo %d/3 raw %d/3 -> %s' % (v, d['R1oa3'], d['R3p'], d['R3p_hi'], d['R3p_lo'], d['raw'],
          'CANDIDATE' if cand else ('INCONCLUSIVE-level' if inc else 'no')))

print('\n=== 3c. refinements, macro delta vs untransformed (criterion 1 raw / 2 reweighted) ===')
for v in VARS:
    print('   %-8s ' % v + ' | '.join('%s %s' % (s, ' '.join('%+.5f(%.2f)' % (get(m, s)[v]['macro']['m'], get(m, s)[v]['macro']['P']) for m in GM))
                                    for s in ('raw', 'R1oa3', 'R3p', 'R3p_hi')))

print('\n=== 4. ORACLE b (per label (b_o1, b_o2)) and split-half honesty ===')
for m in GM + ['ENS3']:
    for s in ('R1oa3', 'R3p'):
        r = get(m, s); o = r['ORACLE']
        print('   %-6s %-6s oracle b %s inc S.1 %+.5f S.2 %+.5f' % (m, s, [tuple(round(x, 3) for x in b) for b in o['b']], o['inc1']['m'], o['inc2']['m']))
        for k, v in r['split'].items():
            print('        split %s b %s | inc S.1 %+.5f (sd %.5f P %.2f) | S.2 %+.5f (sd %.5f P %.2f)' % (
                k, [tuple(round(x, 3) for x in b) for b in v['b']], v['inc1']['m'], v['inc1']['sd'], v['inc1']['P'], v['inc2']['m'], v['inc2']['sd'], v['inc2']['P']))

print('\n=== 5. single D5 folds: model rho vs best symmetric a ===')
rows = []
for m in ['f0', 'f1', 'f2', 'f3', 'f4']:
    for s in ('raw', 'R1oa3', 'R3p', 'R3p_hi'):
        r = R['FOLDS']['res'][m][s]; d = r['desc']
        curve = {a: r[a]['macro']['m'] for a in ['S0.05', 'S0.10', 'S0.15', 'S0.20', 'S0.25', 'S0.30']}
        best = max(curve, key=curve.get)
        rows.append((m, s, np.mean(d['rho']), np.mean(d['q']), best, curve))
        print('   %s %-7s rho %.3f q %.3f gap %.3f | best %s | S.1 %+.5f S.15 %+.5f S.2 %+.5f S.25 %+.5f | LP20-S.2 %+.5f PW20-S.2 %+.5f RAMP2-S.1 %+.5f' % (
            m, s, np.mean(d['rho']), np.mean(d['q']), np.mean(d['rho']) - np.mean(d['q']), best, curve['S0.10'], curve['S0.15'], curve['S0.20'], curve['S0.25'],
            r['LP20']['inc2']['m'], r['PW20']['inc2']['m'], r['RAMP2']['inc1']['m']))
for m in GM + ['ENS3']:
    for s in ('R1oa3', 'R3p', 'R3p_hi'):
        d = get(m, s)['desc']; print('   %-6s %-7s rho %.3f q %.3f gap %.3f' % (m, s, np.mean(d['rho']), np.mean(d['q']), np.mean(d['rho']) - np.mean(d['q'])))

print('\n=== 6. P_AREA robustness (B 200): macro delta and increments ===')
for m in GM + ['ENS3']:
    for s in ('raw', 'R1oa3', 'R3p'):
        r = R['AREA']['res'][m][s]
        print('   %-6s %-6s ' % (m, s) + ' | '.join('%s %+.5f' % (a, r[a]['macro']['m']) for a in ['S-0.10', 'S0.10', 'S0.15', 'S0.20', 'S0.25']) +
              ' || inc1/inc2: ' + ' '.join('%s %+.5f/%+.5f' % (v, r[v]['inc1']['m'], r[v]['inc2']['m']) for v in ['LP15', 'LP20', 'RAMP2', 'PFH', 'LH', 'PW20', 'MH_ctrl']))
