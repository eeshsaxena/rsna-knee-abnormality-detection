#!/usr/bin/env python
# run_offline_tests.py (10-15 rfix, Claude subagent; finding S6F-R4): RE-RUNS every offline test of the V224-FLIP-S6F build against
# the CURRENT files (kernel, builder, cells, gate) and accepts a test only on its exact verdict: rc 0 and the last non-empty stdout
# line == '<NAME> PASS n/n' (same n). A recorded tests/<t>.txt is never trusted (the old step 0 grepped " PASS " in old files, which
# also matched non-verdict lines such as "...V224S6F_CHECK PASS 31/31" inside a FAILED mock run). Each run's output is (re)written
# to <tests-dir>/<t>.txt. Laptop CPU only (about 5 min), -I. Order matters: mock_s6f_output writes the mock that test_gate_recal reads.
#   usage: py -3.12 -I run_offline_tests.py [--tests-dir DIR] [--spec file.py:VERDICT,...]  -> OFFLINE_TESTS PASS/FAIL n/m
import os, re, sys, time, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = [('test_driver_units.py', 'TEST_DRIVER'), ('test_cells_local.py', 'TEST_CELLS'), ('neg_celldiff.py', 'NEG_CELLDIFF'),
        ('mock_s6f_output.py', 'MOCK_S6F'), ('test_controls.py', 'TEST_CONTROLS'), ('test_gate_recal.py', 'TEST_GATE_RECAL'),
        ('test_commit_step0.py', 'TEST_STEP0')]
ap = argparse.ArgumentParser()
ap.add_argument('--tests-dir', default=os.path.join(HERE, 'tests'))
ap.add_argument('--spec', default=None, help='override: file.py:VERDICT,... (tests of this helper)')
A = ap.parse_args()
spec = [tuple(x.split(':', 1)) for x in A.spec.split(',')] if A.spec else SPEC
ok_n = 0
for fn, verdict in spec:
    path = os.path.join(A.tests_dir, fn)
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, '-I', path], capture_output=True, text=True, timeout=1800)
        out, rc = r.stdout + (('\n[stderr]\n' + r.stderr) if r.stderr.strip() else ''), r.returncode
    except Exception as e:
        out, rc = 'run error: %r' % (e,), None
    with open(os.path.join(A.tests_dir, fn[:-3] + '.txt'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(out)
    last = ([l for l in (r.stdout if rc is not None else '').splitlines() if l.strip()] or [''])[-1].strip()
    m = re.fullmatch(r'%s (PASS|FAIL) (\d+)/(\d+)' % re.escape(verdict), last)
    good = rc == 0 and bool(m) and m.group(1) == 'PASS' and m.group(2) == m.group(3)
    ok_n += good
    print('  %s %s rc %s %.0f s: %s' % ('OK  ' if good else 'FAIL', fn, rc, time.time() - t0, last[:120]), flush=True)
print('OFFLINE_TESTS %s %d/%d' % ('PASS' if ok_n == len(spec) else 'FAIL', ok_n, len(spec)), flush=True)
sys.exit(0 if ok_n == len(spec) else 1)
