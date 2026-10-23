# selftest_gate.py (10-20, Claude): run the DRAFT commit gate probe_check_e1eoa.py over real commits and the SYNTHETIC E1 fixtures;
# every row has a pre-stated expectation. Read-only on the out dirs; writes selftest/gate_<case>.txt + prints a summary.
import os, subprocess, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.normpath(os.path.join(HERE, '..', '..', '..'))
TB = json.dumps({'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.5, 'Synovitis': 0.8, "Baker's": 0.5, 'Lateral Meniscus': 0.4, 'Fracture': 0.65})
O = lambda *p: os.path.join(R, *p)
E20, E30 = O('session_2026-10-19', 'a20', 'out_v224r6c4eoa20j_v1'), O('session_2026-10-20', 'a21', 'out_v224r6c4eoa30j_v1')
P20, P30 = O('session_2026-10-20', 'a21', 'out_v224r6c4pw20j_v1'), O('session_2026-10-20', 'a21', 'out_v224r6c4pw30j_v1')
C4O = O('session_2026-10-18', 'a18', 'out_v224r6a25c4j_v1')
S20, SP20 = os.path.join(HERE, 'selftest', 'SYNTH_out_v224r6c4eoa20e1j_from_eoa20_v1'), os.path.join(HERE, 'selftest', 'SYNTH_out_v224r6c4pw20e1j_from_pw20_v1')
CASES = [  # name, out dir, e1, aoa, mode, expected
    ('real_eoa20_e1x1', E20, 1, .2, 'sym', 'PASS 18/18'), ('real_eoa30_e1x1', E30, 1, .3, 'sym', 'PASS 18/18'),
    ('real_pw20_e1x1', P20, 1, .2, 'pw', 'PASS 18/18'), ('real_pw30_e1x1', P30, 1, .3, 'pw', 'PASS 18/18'),
    ('real_eoa20_as_e1x2', E20, 2, .2, 'sym', 'FAIL'), ('real_pw20_as_e1x2', P20, 2, .2, 'pw', 'FAIL'),
    ('real_pw20_as_sym', P20, 1, .2, 'sym', 'FAIL'), ('real_eoa20_as_aoa0', E20, 1, 0, 'sym', 'FAIL'),
    ('synth_eoa20e1_e1x2', S20, 2, .2, 'sym', 'PASS 18/18'), ('synth_eoa20e1_as_e1x1', S20, 1, .2, 'sym', 'FAIL'),
    ('synth_eoa20e1_as_pw', S20, 2, .2, 'pw', 'FAIL'),
    ('synth_pw20e1_e1x2', SP20, 2, .2, 'pw', 'PASS 18/18'), ('synth_pw20e1_as_sym', SP20, 2, .2, 'sym', 'FAIL'),
    ('synth_pw20e1_as_e1x1', SP20, 1, .2, 'pw', 'FAIL'),
    ('real_c4_aoa0_e1x1', C4O, 1, 0, 'sym', 'PASS 18/18'), ('real_c4_as_aoa20', C4O, 1, .2, 'sym', 'FAIL'),
    ('real_c4_as_e1x2', C4O, 2, 0, 'sym', 'FAIL')]
allok = True
for name, d, e1, aoa, mode, exp in CASES:
    r = subprocess.run([sys.executable, '-I', os.path.join(HERE, 'probe_check_e1eoa.py'), d, '--a224', '0.25', '--table', TB, '--aoa', str(aoa),
                        '--e1', str(e1), '--mode', mode], capture_output=True, text=True, encoding='utf-8')
    open(os.path.join(HERE, 'selftest', 'gate_%s.txt' % name), 'w', encoding='utf-8').write(r.stdout + r.stderr)
    last = (r.stdout.strip().splitlines() or [''])[-1]; fails = [l.strip()[5:60] for l in r.stdout.splitlines() if l.strip().startswith('FAIL')]
    good = exp in last; allok &= good
    print('%-4s %-24s expect %-10s got %-22s %s' % ('OK' if good else 'BAD', name, exp, last.replace('PROBE_CHECK ', ''), fails[:4]))
print('SELFTEST_GATE', 'PASS' if allok else 'FAIL')
