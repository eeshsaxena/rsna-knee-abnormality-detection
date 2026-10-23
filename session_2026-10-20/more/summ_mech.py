# summ_mech.py (10-20 more): summarise box_mech_L12.log JSON (aggregates only) for mechanics.md
import json, re
txt = open('box_mech_L12.log', encoding='utf-8').read()
J = json.loads(txt.split('JSON_BEGIN')[1].split('JSON_END')[0])
R = J['res']
print('== Lens 2: placement vs FINAL at the same a (macro, mean / sd), and best-of-curve envelope ==')
for cfg, v in R.items():
    for s in ('R1oa3', 'R3p', 'R3p_hi'):
        r = v['oa'][s]
        for pl, d in r['lens2'].items():
            same = ' '.join('%s:%+.5f/%.5f' % (a, x['same_a']['m'], x['same_a']['sd']) for a, x in d.items())
            print('%-5s %-6s %-6s same-a %s' % (cfg, s, pl, same))
        env = {k: (b[0], round(b[1], 5)) for k, b in r['best'].items()}
        print('%-5s %-6s envelope (best a, max delta) %s' % (cfg, s, env))
print('== Decision rule check: placement or e2 beats FINAL e1 at the SAME a by > sd, all configs, R1oa3 AND R3p ==')
for a in ('0.10', '0.20'):
    for pl in ('V224', 'MEMBER', 'E1STACK'):
        cells = []
        for cfg, v in R.items():
            for s in ('R1oa3', 'R3p'):
                r = v['oa'][s]
                if pl == 'E1STACK':
                    x = r['lens1'][a]['stack_inc']
                elif pl in r['lens2']:
                    x = r['lens2'][pl][a]['same_a']
                else:
                    continue
                cells.append((cfg, s, x['m'], x['sd'], x['m'] > x['sd']))
        npass = sum(c[4] for c in cells)
        print('a %s %-7s pass %d/%d | %s' % (a, pl, npass, len(cells), ' '.join('%s/%s %+.5f(%.5f)' % (c[0], c[1], c[2], c[3]) for c in cells)))
print('== Lens 1: interaction (E-OA gain e2 - e1) and E1 OA-trio / non-OA effects ==')
for cfg, v in R.items():
    for s in ('raw', 'R1oa3', 'R3p', 'R3p_hi'):
        r = v['oa'][s]
        print('%-5s %-6s E1 OA %+.5f (%.5f) | interaction a.1 %+.6f (%.6f) a.2 %+.6f (%.6f)' % (
            cfg, s, r['E1_oa']['m'], r['E1_oa']['sd'], r['lens1']['0.10']['interaction']['m'], r['lens1']['0.10']['interaction']['sd'],
            r['lens1']['0.20']['interaction']['m'], r['lens1']['0.20']['interaction']['sd']))
    print('%-5s non-OA E1 6-label macro %+.5f (%.5f)' % (cfg, v['nonoa_macro']['m'], v['nonoa_macro']['sd']))
