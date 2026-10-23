#!/usr/bin/env python
# selftest_commit_gate.py (10-20 ~19:35 IST, Claude, Track C): run the EXACT commit-gate command line of every A22 slate entry
# (val/slate_a22.json) on (i) its simulated commit output (val/sim/, written by gate_val_offline.py = the control's REAL committed dir
# + this notebook's cell 18 replayed on those inputs) and (ii) real committed dirs, with a pre-stated expectation per row. Offline only.
import json, os, sys, subprocess
sys.stdout.reconfigure(encoding='utf-8')
R = 'C:/Users/eeshs/Downloads/rsna-knee'; V = R + '/session_2026-10-20/val'; SIM = V + '/sim'
S = json.load(open(V + '/slate_a22.json', encoding='utf-8'))['entries']
PY = ['py', '-3.12', '-I']
C4T = '{"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker\'s": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}'
R6T = '{"Medial OA": 0.5, "Lateral OA": 0.5, "PF OA": 0.5}'
G21 = 'session_2026-10-20/a21/probe_check_eoa21.py'
def cmd_of(key, out): e = S[key]; return PY + [R + '/' + e['gate_script'], out] + e['gate_args']
def g21(out, a224, table, aoa, mode): return PY + [R + '/' + G21, out, '--a224', str(a224), '--table', table, '--aoa', str(aoa), '--mode', mode]
sim = lambda tag: SIM + '/out_%s_sim' % tag
ROWS = [  # (name, argv, expected 'PASS' / 'FAIL')
    ('pw35 gate on pw35 sim', cmd_of('pw35', sim('v224r6c4pw35j')), 'PASS'),
    ('pw40 gate on pw40 sim', cmd_of('pw40', sim('v224r6c4pw40j')), 'PASS'),
    ('pw45 gate on pw45 sim', cmd_of('pw45', sim('v224r6c4pw45j')), 'PASS'),
    ('r6pw30 gate on R6pw30 sim', cmd_of('r6pw30', sim('v224r6flips6fpw30j')), 'PASS'),
    ('r6e20 gate (A20 probe_check_eoa) on R6e20 sim', cmd_of('r6e20', sim('v224r6flips6feoa20j')), 'PASS'),
    ('a21 pw30 gate on the REAL scored pw30 committed dir', g21(R + '/session_2026-10-20/a21/out_v224r6c4pw30j_v1', 0.25, C4T, 0.3, 'pw'), 'PASS'),
    ('a21 pw30 gate on pw30-replayed-on-c4-inputs sim', g21(sim('v224r6c4pw30j'), 0.25, C4T, 0.3, 'pw'), 'PASS'),
    ('NEG pw35 gate on the REAL c4 committed dir (no E-OA)', cmd_of('pw35', R + '/session_2026-10-18/a18/out_v224r6a25c4j_v1'), 'FAIL'),
    ('NEG r6pw30 gate on the REAL R6 committed dir (no E-OA)', cmd_of('r6pw30', R + '/session_2026-10-15/r6entry/out_v224r6flips6fj_v1'), 'FAIL'),
    ('NEG pw40 gate on pw35 sim (dose swap; only the receipt can tell on 3 rows)', cmd_of('pw40', sim('v224r6c4pw35j')), 'FAIL'),
    ('NEG pw35 gate on pw45 sim (dose swap)', cmd_of('pw35', sim('v224r6c4pw45j')), 'FAIL'),
    ('NEG c4-param pw30 gate on R6pw30 sim (base swap)', g21(sim('v224r6flips6fpw30j'), 0.25, C4T, 0.3, 'pw'), 'FAIL'),
    ('NEG R6-param gate on pw35 sim (base swap)', g21(sim('v224r6c4pw35j'), 0.35, R6T, 0.35, 'pw'), 'FAIL'),
    ('NEG r6pw30 gate with --mode sym on R6pw30 sim (shape swap)', g21(sim('v224r6flips6fpw30j'), 0.35, R6T, 0.3, 'sym'), 'FAIL'),
    ('NEG a21 gate --mode sym on R6e20 sim (A20 block writes no oa_mode -> r6e20 must use the A20 gate)', g21(sim('v224r6flips6feoa20j'), 0.35, R6T, 0.2, 'sym'), 'FAIL'),
]
ok = True
for name, argv, exp in ROWS:
    p = subprocess.run(argv, capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=R)
    last = (p.stdout.strip().splitlines() or [''])[-1]
    got = 'PASS' if 'PROBE_CHECK PASS' in last else ('FAIL' if 'PROBE_CHECK FAIL' in last else 'ERR')
    fails = [l.strip() for l in p.stdout.splitlines() if l.strip().startswith('FAIL')]
    good = got == exp and (exp == 'FAIL' or last.endswith('PASS 17/17'))
    ok = ok and good
    print('%s %-95s expect %s got %s | %s%s' % ('OK  ' if good else 'BAD ', name, exp, got, last, (' | first FAIL: ' + fails[0][:110]) if fails else ''), flush=True)
print('COMMIT_GATE_SELFTEST', 'PASS' if ok else 'FAIL', '%d rows' % len(ROWS))
print('\n# commit-gate command lines (run on the fetched committed output dir <OUT>; each must print PROBE_CHECK PASS 17/17):')
for k, e in S.items():
    q = lambda s: "'" + s + "'" if (' ' in s or '"' in s) else s
    print('%-7s py -3.12 -I %s <OUT> %s' % (k, e['gate_script'], ' '.join(q(x) for x in e['gate_args'])))
