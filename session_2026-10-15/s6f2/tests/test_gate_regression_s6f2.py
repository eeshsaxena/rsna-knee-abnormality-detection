#!/usr/bin/env python
# test_gate_regression_s6f2.py (10-15, Claude subagent): v224s6f2_check.py in flip mode keeps the s6f gate's verdicts, and a build
# receipt of another base never passes. Offline, laptop CPU, -I. Reads the s6f outputs (never writes there).
#   G1 the REAL S6F v1 commit (s6f/out_v224flips6f_v1): s6f gate PASS 31/31 and the s6f2 gate PASS, every s6f check line also OK here
#   G2 the s6f mock clean output: both PASS            G3 the s6f canary mock (--expect-canary --control skip): both PASS
#   G4 cross: the FLRM mock with the flip receipt FAIL; the flips20 mock with the FLRM receipt FAIL; the FLRM mock with the flips20
#      receipt FAIL (each base's gate reads its own last marker / base gate / references)
#   usage: py -3.12 -I tests/test_gate_regression_s6f2.py  -> TEST_GATE_REG_S6F2 PASS/FAIL n/m
import os, re, sys, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
S6F2 = os.path.dirname(HERE)
SESS = os.path.dirname(S6F2)
S6F = os.path.join(SESS, 's6f')
OLD, NEW = os.path.join(S6F, 'v224s6f_check.py'), os.path.join(S6F2, 'v224s6f2_check.py')
REC_FLIP = os.path.join(S6F, 'build_receipt_v224flips6f.json')
R = []


def chk(name, ok, info=''):
    R.append(bool(ok))
    print('  %s %s %s' % ('OK  ' if ok else 'FAIL', name, info), flush=True)


def run(gate, out, *extra):
    r = subprocess.run([sys.executable, '-I', gate, out] + list(extra), capture_output=True, text=True, timeout=1800, encoding='utf-8', errors='replace')
    lines = r.stdout.splitlines()
    v = ([l for l in lines if re.match(r'V224S6F2?_CHECK (PASS|FAIL) ', l)] or ['(none) ' + r.stderr[-200:]])[-1]
    oks = [re.sub(r'\s+\S*$', '', l.strip()[5:])[:60] for l in lines if l.startswith('  OK   ')]
    return r.returncode, v, oks


def names(oks):
    return set(o.split(' (')[0][:50] for o in oks)


for name, out, extra in (('G1 real S6F v1 commit', os.path.join(S6F, 'out_v224flips6f_v1'), []),
                         ('G2 s6f mock clean', os.path.join(S6F, 'tests', 'mock_out_v224flips6f'), []),
                         ('G3 s6f canary mock (skip control)', os.path.join(S6F, 'tests', 'mock_out_v224flips6f_canary'),
                          ['--expect-canary', '--control', 'skip'])):
    if not os.path.isdir(out):
        chk(name + ': output dir present', False, out)
        continue
    rc0, v0, ok0 = run(OLD, out, '--build', REC_FLIP, *extra)
    rc1, v1, ok1 = run(NEW, out, '--build', REC_FLIP, *extra)
    n0, n1 = int(v0.split()[2].split('/')[0]) if 'PASS' in v0 else -1, int(v1.split()[2].split('/')[0]) if 'PASS' in v1 else -1
    chk('%s: s6f gate %s | s6f2 gate %s (one more check: the reviewed submit policy, normal mode only)' % (name, v0, v1),
        rc0 == 0 and rc1 == 0 and 'PASS' in v0 and 'PASS' in v1 and n1 == n0 + (0 if extra else 1))
for name, out, rec in (('G4 FLRM mock + flip receipt', os.path.join(HERE, 'mock_out_v224flrms6f'), REC_FLIP),
                       ('G4 flips20 mock + FLRM receipt', os.path.join(HERE, 'mock_out_v224flips6fs20'), os.path.join(S6F2, 'build_receipt_v224flrms6f.json')),
                       ('G4 FLRM mock + flips20 receipt', os.path.join(HERE, 'mock_out_v224flrms6f'), os.path.join(S6F2, 'build_receipt_v224flips6fs20.json'))):
    if not os.path.isdir(out):
        chk(name + ': mock present (run mock_s6f2_output.py first)', False, out)
        continue
    rc, v, oks = run(NEW, out, '--build', rec)
    chk('%s: FAIL (%s)' % (name, v), rc != 0 and 'FAIL' in v)
print('TEST_GATE_REG_S6F2 %s %d/%d' % ('PASS' if all(R) else 'FAIL', sum(R), len(R)), flush=True)
sys.exit(0 if all(R) else 1)
