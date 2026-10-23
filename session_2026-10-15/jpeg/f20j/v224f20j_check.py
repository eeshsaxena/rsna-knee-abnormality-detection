#!/usr/bin/env python
# v224f20j_check.py (2026-10-14, Claude subagent): commit gate for V224-F20-J (eeshsaxena/rsna-sub-v224f20j), built by
# build_v224f20j.py. Reads a fetched kernel output dir. Prints V224F20J_CHECK PASS/FAIL n/m; exit 0 only on PASS.
#   usage: python v224f20j_check.py OUT [--ref-f20 OUT_V224F20_V1] [--ref-v224 OUT_V224_V1] [--ref-fleet OUT_HFR7RR_V1]
# (1) the full V224-F20 gate, unchanged: session_2026-10-15/v224f/v224f_check.py OUT --wf 0.20 --ref-v224 --ref-fleet (28 checks:
#     v224_check exact on the V224 part, h_check hf, fleet receipt, recomputed blend, ...). The DEC cell's output is before
#     V224_DONE, so v224_check's no-Traceback / no-RuntimeError scan covers it too.
# (2) the DEC cell: exactly one DEC_OK line, before RAPTOR_STAGE_DONE (stage 1), no DEC_FALLBACK; DEC_OK and dec_receipt.json say:
#     wheels found and == PyPI sha, pip rc 0, /tmp/_dec holds exactly the 8 expected entries, versions 2.1.0 / 2.4.0 / 2.6.0
#     imported from /tmp/_dec, pydicom lists pylibjpeg for JPEG Lossless .57 / .70, 3/3 self-test images exact,
#     PYTHONPATH after == /tmp/_dec + ':' + before.
# (3) end-to-end identity with the V224-F20 v1 commit (same 3 public studies, all uncompressed): every stage csv and the final
#     submission within the existing tolerances (1e-6 blends, 1e-3 stage outputs: GPU / compile), byte identity reported per
#     file (expected 8/8, MODEL: the public test has no compressed file, so the decoders are never used).
import os, sys, json, glob, hashlib, argparse, subprocess
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..', '..'))
F_CHECK = os.path.join(ROOT, 'session_2026-10-15', 'v224f', 'v224f_check.py')
ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--ref-f20', default=os.path.join(ROOT, 'session_2026-10-15', 'v224f', 'out_v224f20_v1'))
ap.add_argument('--ref-v224', default=os.path.join(ROOT, 'session_2026-10-14', 'v224', 'out_v224_v1'))
ap.add_argument('--ref-fleet', default=os.path.join(ROOT, 'session_2026-10-13', 'hr7rr', 'out_hfr7rr_v1'))
ap.add_argument('--sample', default=os.path.join(ROOT, '_data', 'sample_submission.csv'))
ap.add_argument('--skip-f-gate', action='store_true', help='negative tests only: skip sub-gate (1)')
A = ap.parse_args()
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
R = []
TOP = ['_libjpeg.cpython-313-x86_64-linux-gnu.so', '_openjpeg.cpython-313-x86_64-linux-gnu.so', 'libjpeg', 'openjpeg', 'pylibjpeg',
       'pylibjpeg-2.1.0.dist-info', 'pylibjpeg_libjpeg-2.4.0.dist-info', 'pylibjpeg_openjpeg-2.6.0.dist-info']
VERS = {'pylibjpeg': '2.1.0', 'libjpeg': '2.4.0', 'openjpeg': '2.6.0'}
PYPI = {'pylibjpeg-2.1.0-py3-none-any.whl', 'pylibjpeg_libjpeg-2.4.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl',
        'pylibjpeg_openjpeg-2.6.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl'}
FILES = [('submission.csv', 1e-6), ('submission_V224.csv', 1e-6), ('submission_H.csv', 1e-6), ('submission_fleetsolo.csv', 1e-3),
         ('raptor_sub.csv', 1e-3), ('raptor_raw.csv', 1e-3), ('raptor224_sub.csv', 1e-3), ('ours_sub.csv', 1e-3)]


def chk(name, ok, info=''):
    R.append(bool(ok))
    print(f'  {"OK  " if ok else "FAIL"} {name} {info}', flush=True)


def P(*a):
    return os.path.join(A.out, *a)


def read_log(d):
    logs = [p for p in glob.glob(os.path.join(d, '*.log')) if 'cufile' not in os.path.basename(p)]
    if not logs:
        return ''
    try:
        return '\n'.join(e.get('data', '') for e in json.load(open(logs[0], encoding='utf-8')))
    except Exception:
        return open(logs[0], encoding='utf-8', errors='replace').read()


def line_json(text, marker):
    for ln in text.splitlines():
        if ln.startswith(marker + ' '):
            try:
                return json.loads(ln[len(marker) + 1:].strip())
            except Exception:
                return None
    return None


print(f'V224F20J_CHECK {A.out}', flush=True)
text = read_log(A.out)
chk('kernel log present', bool(text))
# ---- (1) the V224-F20 gate --------------------------------------------------------------------------------------------------
if not A.skip_f_gate:
    r = subprocess.run([sys.executable, F_CHECK, A.out, '--wf', '0.20', '--ref-v224', A.ref_v224, '--ref-fleet', A.ref_fleet,
                        '--sample', A.sample], capture_output=True, text=True, encoding='utf-8')
    print('\n'.join('    | ' + ln for ln in r.stdout.strip().splitlines()), flush=True)
    chk('V224-F20 gate (v224f_check.py --wf 0.20, refs V224-exact + HFR7-RR) PASS', r.returncode == 0 and 'V224F_CHECK PASS' in r.stdout)
# ---- (2) the DEC cell -------------------------------------------------------------------------------------------------------
n_ok = sum(ln.startswith('DEC_OK ') for ln in text.splitlines())
chk('exactly one DEC_OK line, no DEC_FALLBACK', n_ok == 1 and 'DEC_FALLBACK' not in text, {'DEC_OK lines': n_ok})
p_dec, p_rap = text.find('DEC_OK '), text.find('RAPTOR_STAGE_DONE')
chk('DEC_OK printed before stage 1 (RAPTOR_STAGE_DONE)', 0 <= p_dec < p_rap, {'DEC_OK': p_dec, 'RAPTOR_STAGE_DONE': p_rap})
d = line_json(text, 'DEC_OK') or {}
chk('DEC_OK: ok, tag v224f20j, pip rc 0, versions, JPEG Lossless plugin pylibjpeg, 3/3 self-test exact',
    d.get('ok') is True and d.get('tag') == 'v224f20j' and d.get('pip_rc') == 0 and d.get('versions') == VERS and
    all('pylibjpeg' in v for v in (d.get('plugins') or {'x': []}).values()) and len(d.get('plugins') or {}) == 2 and
    d.get('decode') == {'JLL57_12U': True, 'JLL70_16S': True, 'U_16S': True} and d.get('pydicom') == '3.0.2',
    {k: d.get(k) for k in ('ok', 'seconds', 'versions', 'decode', 'pydicom')})
rc = None
try:
    rc = json.load(open(P('dec_receipt.json'), encoding='utf-8'))
except Exception as e:
    pass
rc = rc or {}
st = rc.get('selftest') or {}
pb = rc.get('pythonpath_before') or ''
chk('dec_receipt.json: ok, 3 PyPI-identical wheels found, /tmp/_dec == the 8 expected entries, decoders imported from /tmp/_dec',
    rc.get('ok') is True and set((rc.get('wheels') or {}).keys()) == PYPI and rc.get('target_files') == TOP and
    st.get('dirs') == {'pylibjpeg': '/tmp/_dec', 'libjpeg': '/tmp/_dec', 'openjpeg': '/tmp/_dec'},
    {'wheels_from': sorted(set(os.path.dirname(p) for p in (rc.get('wheels') or {}).values())), 'target_files': len(rc.get('target_files') or [])})
chk('PYTHONPATH after == /tmp/_dec + ":" + before (before non-empty), same in DEC_OK',
    bool(pb) and rc.get('pythonpath_after') == '/tmp/_dec:' + pb and d.get('pythonpath_after') == rc.get('pythonpath_after'),
    {'before': pb, 'after': rc.get('pythonpath_after')})
# ---- (3) identity with the V224-F20 v1 commit -------------------------------------------------------------------------------
sample = pd.read_csv(A.sample, dtype={'StudyInstanceUID': str})
labels = list(sample.columns)[1:]
n_bytes = 0
for f, tol in FILES:
    a, b = P(f), os.path.join(A.ref_f20, f)
    if not (os.path.isfile(a) and os.path.isfile(b)):
        chk('%s present here and in the V224-F20 v1 commit' % f, False, {'here': os.path.isfile(a), 'ref': os.path.isfile(b)})
        continue
    same_bytes = open(a, 'rb').read() == open(b, 'rb').read()
    n_bytes += same_bytes
    try:
        X = pd.read_csv(a, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
        Y = pd.read_csv(b, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
        cols = [c for c in labels if c in Y.columns]
        ok_ids = X.index.tolist() == Y.index.tolist() and list(X.columns) == list(Y.columns) and len(cols) == len(labels)
        dd = float(np.abs(X.loc[Y.index, cols].values.astype(np.float64) - Y[cols].values.astype(np.float64)).max()) if ok_ids else float('inf')
    except Exception as e:
        ok_ids, dd = False, float('inf')
    chk('%s == V224-F20 v1 commit (ids / columns / order equal, max diff <= %g)' % (f, tol), ok_ids and dd <= tol,
        'max diff %.3g, byte-identical %s' % (dd, same_bytes))
print(f'  info byte-identical to the V224-F20 v1 commit: {n_bytes}/{len(FILES)} files', flush=True)
ref_text = read_log(A.ref_f20)


def secs(t, marker, key=None):
    if key:
        j = line_json(t, marker) or {}
        return j.get(key)
    for ln in t.splitlines():
        if ln.startswith(marker):
            try:
                return float(ln.split()[-1])
            except Exception:
                return None
    return None


print('  info timing (s): DEC cell %s | RAPTOR_STAGE_DONE %s vs ref %s | V224F_DONE total %s vs ref %s' % (
    d.get('seconds'), secs(text, 'RAPTOR_STAGE_DONE seconds'), secs(ref_text, 'RAPTOR_STAGE_DONE seconds'),
    secs(text, 'V224F_DONE', 'total_seconds'), secs(ref_text, 'V224F_DONE', 'total_seconds')), flush=True)
ok = all(R)
print('V224F20J_CHECK %s %d/%d (byte-identical %d/%d vs V224-F20 v1)' % ('PASS' if ok else 'FAIL', sum(R), len(R), n_bytes, len(FILES)), flush=True)
sys.exit(0 if ok else 1)
