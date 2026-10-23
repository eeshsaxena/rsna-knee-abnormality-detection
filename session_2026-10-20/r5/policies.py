# policies.py (10-20 r5): expected P(silver) / P(bronze+) / P(r<=200) of whole read->pick POLICIES before the read (MODEL),
# from dec_<tag>.json (outcome probabilities x conditional pair values). Outcomes O20 / O10 / O0 partition the full read.
import json, sys
O = ['O20 eoa20 best-id', 'O10 eoa10 best-id', 'O0 no move']
POL = {'R1 (A = best-id, B = R6)': ['eoa20+R6', 'eoa10+R6', 'c4+R6'],
       'R1 but eoa10 as A after a no-move': ['eoa20+R6', 'eoa10+R6', 'eoa10+R6'],
       'R1 but eoa10 as A if eoa20 moves': ['eoa10+R6', 'eoa10+R6', 'c4+R6'],
       'R1 with B = c4 when A moves': ['c4+eoa20', 'c4+eoa10', 'c4+R6'],
       'R1 with B = HW': ['eoa20+HW', 'eoa10+HW', 'c4+HW'],
       'R1 with B = S6F': ['eoa20+S6F', 'eoa10+S6F', 'c4+S6F'],
       'blind eoa10 + R6 (ignore read)': ['eoa10+R6', 'eoa10+R6', 'eoa10+R6'],
       'status quo c4 + R6 (no E-OA)': ['c4+R6', 'c4+R6', 'c4+R6']}
for t in sys.argv[1:]:
    J = json.load(open(f'dec_{t}.json')); po = [J['outcomes'][o]['P'] for o in O]
    print(f'== {t}: P(O20, O10, O0) = ' + ', '.join(f'{p:.2f}' for p in po) + f' (sum {sum(po):.3f})')
    for nm, ps in POL.items():
        v = {m: sum(p * J['outcomes'][o]['pairs'][k][m] for p, o, k in zip(po, O, ps)) for m in ('silver', 'bronze', 'r200', 'E')}
        print(f'   {nm:36s} silver {v["silver"]:.3f} bronze+ {v["bronze"]:.3f} r200 {v["r200"]:.3f} E {v["E"]*1e4:+.2f}e-4')
