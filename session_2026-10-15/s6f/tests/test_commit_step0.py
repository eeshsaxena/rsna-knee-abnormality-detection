#!/usr/bin/env python
# test_commit_step0.py (10-15 rfix, Claude subagent; finding S6F-R4): the pre-push step 0 of gpu_commit_v224flips6f.sh must RE-RUN
# the offline tests and accept only exact verdict lines. Offline, laptop CPU, -I. Fake tests live in a temp dir.
#   S1 run_offline_tests.py on 5 fake tests: FAKE_A PASS 3/3 accepted; rejected: a FAIL verdict after 6 non-verdict " PASS " lines,
#      a stale txt that says PASS while the script now FAILS (the txt is rewritten), PASS 20/21, PASS with rc 1 -> OFFLINE_TESTS FAIL 1/5;
#      a lone good test -> OFFLINE_TESTS PASS 1/1
#   S2 (the bug, for the record) the old step-0 rule `grep -q " PASS " tests/<t>.txt` accepts 4 of those 5 recorded files
#   S3 the script's step 0: no grep of recorded files; run_offline_tests.py and the cell-diff proof both piped into verdict_ok;
#      verdict_ok (executed with bash) accepts 'X PASS 7/7' (also with CRLF) and rejects 'X PASS 6/7', 'X FAIL 20/21' after a
#      ' PASS ' line, 'X PASS 17/1'
#   S4 the helper's list = the 4 original tests + test_controls, test_gate_recal, test_commit_step0, all present
#   usage: py -3.12 -I tests/test_commit_step0.py [--script PATH] [--helper PATH]  -> TEST_STEP0 PASS/FAIL n/m
import os, re, sys, shutil, argparse, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
S6F = os.path.dirname(HERE)
ap = argparse.ArgumentParser()
ap.add_argument('--script', default=os.path.join(S6F, 'gpu_commit_v224flips6f.sh'))
ap.add_argument('--helper', default=os.path.join(S6F, 'run_offline_tests.py'))
A = ap.parse_args()
T = os.path.join(HERE, '_tmp_step0')
shutil.rmtree(T, ignore_errors=True)
os.makedirs(T)
R = []


def chk(name, ok, info=''):
    R.append(bool(ok))
    print('  %s %s %s' % ('OK  ' if ok else 'FAIL', name, info), flush=True)


FAKES = {'fake_ok.py': ("print('FAKE_A PASS 3/3')", 0, None),
         'fake_lastfail.py': ("print('  OK   x V224S6F_CHECK PASS 31/31 mode S6F')\n" * 6 + "print('FAKE_B FAIL 20/21')", 1, 'own'),
         'fake_stale.py': ("print('FAKE_C FAIL 1/2')", 1, 'FAKE_C PASS 2/2\n'),
         'fake_partial.py': ("print('FAKE_D PASS 20/21')", 0, 'own'),
         'fake_rc.py': ("print('FAKE_E PASS 1/1')", 1, 'own')}
for fn, (body, rc, txt) in FAKES.items():
    open(os.path.join(T, fn), 'w').write('import sys\n%s\nsys.exit(%d)\n' % (body, rc))
    if txt == 'own':                                          # the recorded output of that script, as a previous run wrote it
        txt = subprocess.run([sys.executable, '-I', os.path.join(T, fn)], capture_output=True, text=True).stdout
    if txt:
        open(os.path.join(T, fn[:-3] + '.txt'), 'w').write(txt)
# ---- S2 (old rule on the recorded files, before any re-run) -----------------------------------------------------------------
old_ok = sorted(fn for fn in FAKES if os.path.isfile(os.path.join(T, fn[:-3] + '.txt')) and ' PASS ' in open(os.path.join(T, fn[:-3] + '.txt')).read())
chk('S2 (record of the bug) old rule grep " PASS " accepts the recorded FAIL / stale / partial / rc-1 files', old_ok ==
    ['fake_lastfail.py', 'fake_partial.py', 'fake_rc.py', 'fake_stale.py'], old_ok)
# ---- S1 -----------------------------------------------------------------------------------------------------------------
spec = 'fake_ok.py:FAKE_A,fake_lastfail.py:FAKE_B,fake_stale.py:FAKE_C,fake_partial.py:FAKE_D,fake_rc.py:FAKE_E'
if os.path.isfile(A.helper):
    r = subprocess.run([sys.executable, '-I', A.helper, '--tests-dir', T, '--spec', spec], capture_output=True, text=True, timeout=300)
    lines = r.stdout.splitlines()
    acc = sorted(l.split()[1] for l in lines if l.strip().startswith('OK '))
    chk('S1 helper: only fake_ok accepted, verdict OFFLINE_TESTS FAIL 1/5, rc 1', r.returncode == 1 and acc == ['fake_ok.py'] and
        lines[-1:] == ['OFFLINE_TESTS FAIL 1/5'], (acc, lines[-1:]))
    chk('S1 helper: the stale PASS txt was re-run and rewritten (now FAKE_C FAIL 1/2)', 'FAKE_C FAIL 1/2' in open(os.path.join(T, 'fake_stale.txt')).read())
    r = subprocess.run([sys.executable, '-I', A.helper, '--tests-dir', T, '--spec', 'fake_ok.py:FAKE_A'], capture_output=True, text=True, timeout=300)
    chk('S1 helper: a lone passing test -> OFFLINE_TESTS PASS 1/1, rc 0', r.returncode == 0 and r.stdout.splitlines()[-1:] == ['OFFLINE_TESTS PASS 1/1'])
    src = open(A.helper, encoding='utf-8').read()
    m = re.search(r'SPEC = \[(.*?)\]\n', src, re.S)
    names = re.findall(r"\('([\w.]+)', '([A-Z0-9_]+)'\)", m.group(1)) if m else []
    want = ['test_driver_units.py', 'test_cells_local.py', 'neg_celldiff.py', 'mock_s6f_output.py', 'test_controls.py', 'test_gate_recal.py',
            'test_commit_step0.py']
    chk('S4 helper list = the 4 original tests + the 3 rfix tests, each file present in tests/', [n for n, _ in names] == want and
        all(os.path.isfile(os.path.join(HERE, n)) for n in want), names)
else:
    chk('S1 helper present', False, A.helper)
# ---- S3 -----------------------------------------------------------------------------------------------------------------
s = open(A.script, encoding='utf-8').read()
step0 = s[s.find('# 0.'):s.find('# 1.')]
code0 = '\n'.join(l for l in step0.splitlines() if not l.lstrip().startswith('#'))
chk('S3 script step 0: no grep of recorded test files; helper re-run with -I and the proof both piped into verdict_ok',
    'grep -q " PASS "' not in code0 and 'tests/$t.txt' not in code0 and
    re.search(r'py -3\.12 -I \$D/run_offline_tests\.py \|.*\| verdict_ok OFFLINE_TESTS \|\|', code0) is not None and
    re.search(r'celldiff_v224flips6f_proof\.py \| verdict_ok S6F_CELLDIFF \|\|', code0) is not None, code0[:0])
fdef = [l for l in s.splitlines() if l.startswith('verdict_ok() {')]
bash = next((b for b in (r'C:\Program Files\Git\bin\bash.exe', shutil.which('bash')) if b and os.path.isfile(b) and 'System32' not in b), None)
if fdef and bash:
    res = []
    for text, want in (('x\r\nOFFLINE_TESTS PASS 7/7\r\n', 0), ('x\nOFFLINE_TESTS PASS 6/7\n', 1),
                       ('  OK   x V224S6F_CHECK PASS 31/31\nOFFLINE_TESTS FAIL 20/21\n', 1), ('OFFLINE_TESTS PASS 17/1\n', 1)):
        r = subprocess.run([bash, '-c', fdef[0] + '\nverdict_ok OFFLINE_TESTS'], input=text.encode(), capture_output=True, timeout=60)
        res.append((r.returncode == 0) == (want == 0))
    chk('S3 verdict_ok (bash): accepts PASS 7/7 (+CRLF); rejects PASS 6/7, FAIL after a " PASS " line, PASS 17/1', all(res), res)
else:
    chk('S3 verdict_ok defined in the script and bash available', False, (bool(fdef), bash))
shutil.rmtree(T, ignore_errors=True)
print('TEST_STEP0 %s %d/%d' % ('PASS' if all(R) else 'FAIL', sum(R), len(R)), flush=True)
sys.exit(0 if all(R) else 1)
