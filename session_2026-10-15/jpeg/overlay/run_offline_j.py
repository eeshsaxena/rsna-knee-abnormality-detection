#!/usr/bin/env python
# run_offline_j.py (2026-10-15, Claude subagent): re-runs every offline J check on the CURRENT files and accepts only exact verdict
# lines (last line '<NAME> PASS n/n', rc 0). Laptop CPU, ~3 min per variant. Never reads recorded *.txt verdicts.
#   usage: python run_offline_j.py [--save] [TAG ...]   (default: v224flipj v224flips6fj v224flips20j)   -> J_OFFLINE PASS n/n
#   --save writes each check's full output into r2/ (celldiff_proof_TAG.txt, harness_TAG.txt, neg_j_check_TAG.txt, test_make_j.txt).
#   Review rebuild (10-15): the kernels checked are r2/kernel_TAG (DEC cell with the J1 test-data preflight and the J2 /proc
#   envwatch; make_j with the J3 audit). The r1 kernels next to this script (kernel_TAG) and their *.txt outputs are the
#   pre-review state, kept untouched; do not push them.
import os, re, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
S15 = os.path.normpath(os.path.join(HERE, '..', '..'))
ROOT = os.path.normpath(os.path.join(S15, '..'))
OUTD = os.path.join(HERE, 'r2')
V = {'v224flipj': (os.path.join(ROOT, 'session_2026-10-14', 'v224', 'kernel_v224flip'), os.path.join(ROOT, 'session_2026-10-14', 'v224', 'out_v224flip_v1')),
     'v224flips6fj': (os.path.join(S15, 's6f', 'kernel_v224flips6f'), os.path.join(S15, 's6f', 'out_v224flips6f_v1')),
     'v224flips20j': (os.path.join(S15, 'v224f', 'kernel_v224flips20'), os.path.join(S15, 'v224f', 'out_v224flips20_v1'))}
SAVE = '--save' in sys.argv
tags = [a for a in sys.argv[1:] if a != '--save'] or list(V)
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
R = []


def run(name, args, verdict, save=None):
    p = subprocess.run([sys.executable, '-I'] + args, capture_output=True, text=True, env=env, cwd=HERE)
    if SAVE and save:
        with open(os.path.join(OUTD, save), 'w', encoding='utf-8', newline='\n') as f:
            f.write(p.stdout + p.stderr)
    last = (p.stdout.strip().splitlines() or [''])[-1].strip()
    ok = p.returncode == 0 and re.match(r'^%s PASS (?:[a-z0-9]+ )?(\d+)/\1\b' % verdict, last) is not None
    R.append(ok)
    print('%s %-40s %s' % ('OK  ' if ok else 'FAIL', name, last[:150]), flush=True)


for t in tags:
    base, out = V[t]
    kd = os.path.join(OUTD, 'kernel_' + t)
    run('%s cell-diff proof' % t, ['j_celldiff_proof.py', base, kd], 'J_CELLDIFF', 'celldiff_proof_%s.txt' % t)
    run('%s local harness' % t, ['j_local_harness.py', kd], 'J_HARNESS', 'harness_%s.txt' % t)
    run('%s gate negatives' % t, [os.path.join('tests', 'neg_j_check.py'), t, out], 'NEG_J_CHECK', 'neg_j_check_%s.txt' % t)
run('builder determinism / audit / refusals', [os.path.join('tests', 'test_make_j.py')], 'TEST_MAKE_J', 'test_make_j.txt')
ok = all(R)
print('J_OFFLINE %s %d/%d' % ('PASS' if ok else 'FAIL', sum(R), len(R)), flush=True)
sys.exit(0 if ok else 1)
