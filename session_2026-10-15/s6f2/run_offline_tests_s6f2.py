#!/usr/bin/env python
# run_offline_tests_s6f2.py (10-15, Claude subagent): RE-RUNS every offline test of the s6f2 builds against the CURRENT files and
# accepts a test only on its exact verdict (rc 0 and the last non-empty stdout line == '<NAME> PASS n/n'), as s6f/run_offline_tests.py.
# Each run's output is (re)written to tests/<name>.txt. Laptop CPU only (about 6-7 min), -I. Order matters: the two mocks are written
# before test_gate_regression_s6f2 reads them.
#   usage: py -3.12 -I run_offline_tests_s6f2.py  -> OFFLINE_TESTS_S6F2 PASS/FAIL n/m
import os, re, sys, time, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(HERE, 'tests')
SPEC = [(['celldiff_s6f2_proof.py', '--kernel', 'flrm'], 'S6F2_CELLDIFF', 'celldiff_proof_flrm', HERE),
        (['celldiff_s6f2_proof.py', '--kernel', 'flips20'], 'S6F2_CELLDIFF', 'celldiff_proof_flips20', HERE),
        (['test_builder_s6f2.py'], 'TEST_BUILDER_S6F2', 'test_builder_s6f2', T),
        (['neg_celldiff_s6f2.py'], 'NEG_CELLDIFF_S6F2', 'neg_celldiff_s6f2', T),
        (['test_cells_s6f2.py'], 'TEST_CELLS_S6F2', 'test_cells_s6f2', T),
        (['mock_s6f2_output.py', 'flrm'], 'MOCK_S6F2', 'mock_s6f2_output_flrm', T),
        (['mock_s6f2_output.py', 'flips20'], 'MOCK_S6F2', 'mock_s6f2_output_flips20', T),
        (['test_gate_regression_s6f2.py'], 'TEST_GATE_REG_S6F2', 'test_gate_regression_s6f2', T)]
ok_n = 0
for argv, verdict, txt, d in SPEC:
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, '-I', os.path.join(d, argv[0])] + argv[1:], capture_output=True, text=True, timeout=3600,
                           encoding='utf-8', errors='replace')
        out, rc = r.stdout + (('\n[stderr]\n' + r.stderr) if r.stderr.strip() else ''), r.returncode
        stdout = r.stdout
    except Exception as e:
        out, rc, stdout = 'run error: %r' % (e,), None, ''
    with open(os.path.join(d, txt + '.txt'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(out)
    last = ([l for l in stdout.splitlines() if l.strip()] or [''])[-1].strip()
    m = re.fullmatch(r'%s (PASS|FAIL) (\d+)/(\d+)' % re.escape(verdict), last)
    good = rc == 0 and bool(m) and m.group(1) == 'PASS' and m.group(2) == m.group(3)
    ok_n += good
    print('  %s %s rc %s %.0f s: %s' % ('OK  ' if good else 'FAIL', ' '.join(argv), rc, time.time() - t0, last[:120]), flush=True)
print('OFFLINE_TESTS_S6F2 %s %d/%d' % ('PASS' if ok_n == len(SPEC) else 'FAIL', ok_n, len(SPEC)), flush=True)
sys.exit(0 if ok_n == len(SPEC) else 1)
