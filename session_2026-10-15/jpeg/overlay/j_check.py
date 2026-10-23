#!/usr/bin/env python
# j_check.py (2026-10-15, Claude subagent): commit gate for a J kernel built by make_j.py (FLIP-J, S6F-J, FLIPS-J, ...). Reads the
# fetched output dir of the J GPU commit and of its BASE commit (same code minus the DEC cell, same 3 public studies, all
# uncompressed: the decoders are never used, so every output must equal the base's).
#   usage: python j_check.py OUT --base-out BASE_OUT [--tag TAG] [--sub-tol 1e-6] [--stage-tol 1e-3] [--strict-bytes]
# Prints J_CHECK PASS n/m (exit 0) or J_CHECK FAIL k/m (exit 1). Checks:
#  (A) both kernel logs present; the J log is named after the J slug.
#  (B) DEC loaded: exactly one DEC_OK line, before the first stage marker; no DEC_FALLBACK and no J_UNAVAILABLE anywhere; DEC_OK says
#      ok / tag / pip rc 0 / python 3.13 / pydicom 3.0.2 / decoders 2.1.0, 2.4.0, 2.6.0 / pylibjpeg listed for .57 and .70 / 3 of 3
#      self-test images exact; dec_receipt.json: the 3 PyPI-identical wheels, /tmp/_dec == the 8 expected entries, decoders imported
#      from /tmp/_dec (pydicom not), PYTHONPATH after == /tmp/_dec + ':' + before. The in-process sys.path insert is an info line
#      (the build audit found no in-process DICOM decoding in these kernels) unless --require-inproc.
#      Preflight (review fix J1): ran on the test series dir (n_groups, n_files > 0), ok (no decode process died / timed out).
#  (E) in-run evidence (review fix J2, j_envwatch.py): dec_envwatch.json, written by the DEC cell's /proc watcher, shows every
#      expected stage script (per tag, or --expect-script) as a descendant Python process with /tmp/_dec on its PYTHONPATH and
#      no -I / -E / -S, and NO descendant Python process without it (--envwatch-waive=SCRIPT excludes one script, printed).
#      S6F kernels (base output has s6f_setup_receipt.json): the S6F env PYTHONPATH (setup receipt and the driver's own
#      env_isolation) == /tmp/s6f_dec + ':' + the DEC PYTHONPATH, decoders imported from /tmp/s6f_dec, /tmp/_dec on the driver
#      sys.path.
#  (C) identity with the base commit: submission.csv ids / columns / order equal and max |diff| <= sub-tol (byte identity reported;
#      --strict-bytes requires it); every other CSV of the base output (stage outputs, S6F parts) present and equal within its
#      tolerance (blends 'submission*' sub-tol, stage outputs stage-tol: GPU nondeterminism measured 4.3e-5 on raptor_raw);
#      the multiset of log markers (line-start TOKENS) equal apart from DEC_OK and PROGRESS (same stage / fallback / canary path);
#      no more Traceback / RuntimeError / FALLBACK lines than the base log.
#  (D) info: versions, DEC seconds, stage timings vs the base.
import os, re, sys, glob, json, argparse
from collections import Counter
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # j_envwatch.py (this script runs with -I)
import j_envwatch                                                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--base-out', required=True)
ap.add_argument('--tag', default=None, help='default: from the J log name rsna-sub-TAG.log')
ap.add_argument('--sub-tol', type=float, default=1e-6)
ap.add_argument('--stage-tol', type=float, default=1e-3)
ap.add_argument('--strict-bytes', action='store_true', help='require submission.csv byte-identical to the base output')
ap.add_argument('--tol', action='append', default=[], help='FILE=TOL override for one base CSV (relative path)')
ap.add_argument('--require-inproc', action='store_true', help='also require the in-process sys.path insert of the DEC cell')
ap.add_argument('--expect-script', action='append', default=None,
                help='stage script basename the watcher must have seen (default: j_envwatch.EXPECT[tag])')
ap.add_argument('--envwatch-waive', action='append', default=[],
                help='script basename excluded from the "no descendant without /tmp/_dec" rule (after a main-session look)')
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
STAGE_MARKERS = ('RAPTOR_STAGE_DONE', 'ASSETS_READY', 'WORKER_START', 'OURS_STAGE_COMPLETE', 'R224_STAGE', 'V224_DONE')
VOLATILE = {'DEC_OK', 'PROGRESS'}
RAW_SUBMISSIONS = ('submission_fleetsolo.csv', 'submission_fleet2g.csv')     # raw fleet probabilities (GPU): stage tolerance
TOL = {k: float(v) for k, v in (t.split('=', 1) for t in A.tol)}


def chk(name, ok, info=''):
    R.append(bool(ok))
    print('  %s %s %s' % ('OK  ' if ok else 'FAIL', name, info), flush=True)


def find_log(d):
    logs = [p for p in glob.glob(os.path.join(d, '*.log')) if 'cufile' not in os.path.basename(p)]
    return logs[0] if len(logs) == 1 else (sorted(logs)[0] if logs else None)


def read_log(p):
    if not p:
        return ''
    try:
        return ''.join(e.get('data', '') for e in json.load(open(p, encoding='utf-8')))
    except Exception:
        return open(p, encoding='utf-8', errors='replace').read()


def line_json(text, marker):
    for ln in text.splitlines():
        if ln.startswith(marker + ' '):
            try:
                return json.loads(ln[len(marker) + 1:].strip())
            except Exception:
                return None
    return None


def markers(text):
    return Counter(m for m in re.findall(r'(?m)^([A-Z][A-Z0-9_]{3,})\b', text) if m not in VOLATILE)


def first_pos(text, toks):
    ps = [m.start() for t in toks for m in re.finditer(r'(?m)^%s\b' % t, text)]
    return min(ps) if ps else -1


print('J_CHECK %s vs base %s' % (A.out, A.base_out), flush=True)
# ---- (A) logs --------------------------------------------------------------------------------------------------------------------
jl, bl = find_log(A.out), find_log(A.base_out)
J, B = read_log(jl), read_log(bl)
chk('kernel logs present (J and base)', bool(J) and bool(B), (jl and os.path.basename(jl), bl and os.path.basename(bl)))
m = re.fullmatch(r'rsna-sub-([a-z0-9]+)\.log', os.path.basename(jl or ''))
tag = A.tag or (m.group(1) if m else None)
chk('J log named after the J slug rsna-sub-%s.log (tag ends in j)' % tag, bool(m) and m.group(1) == tag and bool(tag) and tag.endswith('j'),
    os.path.basename(jl or ''))

# ---- (B) DEC loaded --------------------------------------------------------------------------------------------------------------
n_ok = len(re.findall(r'(?m)^DEC_OK ', J))
chk('exactly one DEC_OK line; no DEC_FALLBACK, no J_UNAVAILABLE anywhere in the J log',
    n_ok == 1 and 'DEC_FALLBACK' not in J and 'J_UNAVAILABLE' not in J, {'DEC_OK': n_ok, 'DEC_FALLBACK': J.count('DEC_FALLBACK'),
                                                                        'J_UNAVAILABLE': J.count('J_UNAVAILABLE')})
p_dec, p_stage = first_pos(J, ('DEC_OK',)), first_pos(J, STAGE_MARKERS)
chk('DEC_OK printed before the first stage marker (the DEC cell ran before every decoding stage)', 0 <= p_dec < p_stage,
    {'DEC_OK': p_dec, 'first stage marker': p_stage})
d = line_json(J, 'DEC_OK') or {}
chk('DEC_OK: ok, tag, pip rc 0, python 3.13, pydicom 3.0.2, decoders 2.1.0 / 2.4.0 / 2.6.0, pylibjpeg for .57 + .70, 3/3 exact',
    d.get('ok') is True and d.get('tag') == tag and d.get('pip_rc') == 0 and str(d.get('python', '')).startswith('3.13') and
    d.get('pydicom') == '3.0.2' and d.get('versions') == VERS and len(d.get('plugins') or {}) == 2 and
    all('pylibjpeg' in v for v in (d.get('plugins') or {'x': []}).values()) and
    d.get('decode') == {'JLL57_12U': True, 'JLL70_16S': True, 'U_16S': True},
    {k: d.get(k) for k in ('tag', 'python', 'pydicom', 'numpy', 'versions', 'decode', 'seconds')})
rc = {}
try:
    rc = json.load(open(os.path.join(A.out, 'dec_receipt.json'), encoding='utf-8'))
except Exception:
    pass
st = rc.get('selftest') or {}
pb = rc.get('pythonpath_before') or ''
chk('dec_receipt.json: ok, the 3 PyPI-identical wheels, /tmp/_dec == the 8 expected entries, decoders from /tmp/_dec, pydicom not',
    rc.get('ok') is True and set((rc.get('wheels') or {}).keys()) == PYPI and rc.get('target_files') == TOP and
    st.get('dirs') == {'pylibjpeg': '/tmp/_dec', 'libjpeg': '/tmp/_dec', 'openjpeg': '/tmp/_dec'} and
    bool(st.get('pydicom_dir')) and st.get('pydicom_dir') != '/tmp/_dec' and st.get('versions') == VERS,
    {'wheels_from': sorted(set(os.path.dirname(p) for p in (rc.get('wheels') or {}).values())), 'pydicom_dir': st.get('pydicom_dir')})
chk('PYTHONPATH after == /tmp/_dec + ":" + before (before non-empty), same in DEC_OK',
    bool(pb) and rc.get('pythonpath_after') == '/tmp/_dec:' + pb and d.get('pythonpath_after') == rc.get('pythonpath_after'),
    {'before': pb, 'after': rc.get('pythonpath_after')})
inproc = d.get('sys_path_inserted') is True and rc.get('sys_path_inserted') is True and (rc.get('sys_path_head') or [None])[0] == '/tmp/_dec'
if A.require_inproc:
    chk('in-process: sys.path[0] == /tmp/_dec (DEC_OK + receipt)', inproc, {'head': rc.get('sys_path_head'), 'note': rc.get('sys_path_note')})
else:
    print('  info IN-PROCESS sys.path insert %s %s(the build audit found no in-process DICOM decoding in this kernel)' % (
        inproc, (rc.get('sys_path_note') or '') + ' ' if rc.get('sys_path_note') else ''), flush=True)
print('  info VERSIONS python %s | pydicom %s | numpy %s | decoders %s | DEC seconds %s' % (
    st.get('python'), st.get('pydicom'), st.get('numpy'), st.get('versions'), rc.get('seconds')), flush=True)
pf, rpf = d.get('preflight') or {}, rc.get('preflight') or {}
chk('preflight (J1): ok in DEC_OK + receipt, test series dir scanned (groups / files > 0), every flagged file decoded by a live process',
    pf.get('ok') is True and rpf.get('ok') is True and (pf.get('n_groups') or 0) > 0 and (pf.get('n_files') or 0) > 0 and
    str(rpf.get('series_dir', '')).endswith(('/test_series', '/test_images')) and
    (pf.get('decoded_ok') or 0) + (pf.get('decoded_err') or 0) == (pf.get('n_flagged') or 0),
    {k: pf.get(k) for k in ('n_studies', 'n_groups', 'n_files', 'n_flagged', 'decoded_ok', 'decoded_err', 'seconds')})
print('  info PREFLIGHT probe transfer syntaxes %s | test root %s' % (rpf.get('probe_ts'), rpf.get('test_root')), flush=True)

# ---- (E) in-run evidence: the /proc watcher + the S6F receipts ----------------------------------------------------------------
chk('envwatch started (DEC_OK + receipt)', d.get('envwatch') == 'started' and rc.get('envwatch') == 'started',
    {'DEC_OK': d.get('envwatch'), 'receipt': rc.get('envwatch')})
ew = None
try:
    ew = json.load(open(os.path.join(A.out, 'dec_envwatch.json'), encoding='utf-8'))
except Exception:
    pass
expect = A.expect_script or j_envwatch.EXPECT.get(tag or '', [])
ok_ew, probs_ew, info_ew = j_envwatch.verdict(ew, expect, A.envwatch_waive)
chk('envwatch: every stage script %s seen with /tmp/_dec on PYTHONPATH and no -I / -E / -S; no descendant Python process without it' %
    expect, ok_ew, probs_ew[:4] or info_ew)
if A.envwatch_waive:
    print('  info ENVWATCH WAIVED (main-session decision): %s' % A.envwatch_waive, flush=True)
print('  info ENVWATCH %s' % json.dumps(info_ew)[:900], flush=True)
if os.path.isfile(os.path.join(A.base_out, 's6f_setup_receipt.json')):
    sj, dj = {}, {}
    try:
        sj = json.load(open(os.path.join(A.out, 's6f_setup_receipt.json'), encoding='utf-8'))
        dj = json.load(open(os.path.join(A.out, 's6f', 's6f_driver_receipt.json'), encoding='utf-8'))
    except Exception:
        pass
    want = '/tmp/s6f_dec:' + (rc.get('pythonpath_after') or '?')
    ei = dj.get('env_isolation') or {}
    mods = ei.get('mods') or {}
    chk('S6F setup receipt: S6F_DEC ok, S6F env PYTHONPATH == /tmp/s6f_dec + ":" + the DEC PYTHONPATH (keeps /tmp/_dec)',
        sj.get('dec_ok') is True and sj.get('pythonpath_s6f') == want, {'pythonpath_s6f': sj.get('pythonpath_s6f'), 'want': want})
    chk('S6F driver env_isolation: PYTHONPATH == the same, /tmp/_dec on the driver sys.path, pylibjpeg / libjpeg / openjpeg from '
        '/tmp/s6f_dec', ei.get('PYTHONPATH') == want and '/tmp/_dec' in (ei.get('driver_sys_path') or []) and
        all((mods.get(m) or [None, None])[1] == '/tmp/s6f_dec' for m in ('pylibjpeg', 'libjpeg', 'openjpeg')),
        {'PYTHONPATH': ei.get('PYTHONPATH'), 'mods': {m: mods.get(m) for m in ('pylibjpeg', 'libjpeg', 'openjpeg', 'pydicom')}})


# ---- (C) identity with the base commit -------------------------------------------------------------------------------------------
def compare(a, b, tol, keep_order):
    X = pd.read_csv(a, dtype=str)
    Y = pd.read_csv(b, dtype=str)
    if list(X.columns) != list(Y.columns) or len(X) != len(Y):
        return False, float('inf'), 'columns / rows differ %s vs %s' % ((list(X.columns)[:3], len(X)), (list(Y.columns)[:3], len(Y)))
    key = X.columns[0]
    note = ''
    if not keep_order and X[key].tolist() != Y[key].tolist():
        if sorted(X[key]) != sorted(Y[key]):
            return False, float('inf'), 'id sets differ'
        X, Y = X.sort_values(key, kind='stable').reset_index(drop=True), Y.sort_values(key, kind='stable').reset_index(drop=True)
        note = ' (row order differs; compared by id)'
    if X[key].tolist() != Y[key].tolist():
        return False, float('inf'), 'ids / order differ'
    dmax = 0.0
    for c in X.columns[1:]:
        xn, yn = pd.to_numeric(X[c], errors='coerce'), pd.to_numeric(Y[c], errors='coerce')
        if xn.notna().all() and yn.notna().all():
            dmax = max(dmax, float(np.abs(xn.values.astype(np.float64) - yn.values.astype(np.float64)).max()) if len(xn) else 0.0)
        elif X[c].tolist() != Y[c].tolist():
            return False, float('inf'), 'non-numeric column %s differs' % c
    return dmax <= tol, dmax, note


a, b = os.path.join(A.out, 'submission.csv'), os.path.join(A.base_out, 'submission.csv')
if os.path.isfile(a) and os.path.isfile(b):
    same = open(a, 'rb').read() == open(b, 'rb').read()
    ok, dd, note = compare(a, b, A.sub_tol, keep_order=True)
    chk('submission.csv == the base commit output (ids / columns / order equal, max |diff| <= %g%s)' % (
        A.sub_tol, ', byte-identical required' if A.strict_bytes else ''), ok and (same or not A.strict_bytes),
        'max |diff| %.3g, byte-identical %s%s' % (dd, same, note))
else:
    same = False
    chk('submission.csv present in J and base outputs', False, (os.path.isfile(a), os.path.isfile(b)))
base_csvs = sorted(os.path.relpath(p, A.base_out).replace('\\', '/') for p in glob.glob(os.path.join(A.base_out, '**', '*.csv'), recursive=True))
base_csvs = [f for f in base_csvs if f != 'submission.csv']
n_bytes, bad, rows = 0, [], []
for f in base_csvs:
    a, b = os.path.join(A.out, f), os.path.join(A.base_out, f)
    bn = os.path.basename(f)
    tol = TOL.get(f, A.sub_tol if bn.startswith('submission') and bn not in RAW_SUBMISSIONS else A.stage_tol)
    if not os.path.isfile(a):
        bad.append('%s missing' % f)
        continue
    sb = open(a, 'rb').read() == open(b, 'rb').read()
    n_bytes += sb
    try:
        ok, dd, note = compare(a, b, tol, keep_order=os.path.basename(f).startswith('submission'))
    except Exception as e:
        ok, dd, note = False, float('inf'), type(e).__name__
    rows.append('%s %.2g%s%s' % (f, dd, ' =bytes' if sb else '', note))
    if not ok:
        bad.append('%s max |diff| %.3g > %g%s' % (f, dd, tol, note))
chk('every other CSV of the base output present and equal within tolerance (%d files: rank blends %g, stage outputs %g)' % (
    len(base_csvs), A.sub_tol, A.stage_tol), not bad, bad or '')
print('  info per-file max |diff| (J vs base): %s' % ' | '.join(rows), flush=True)
print('  info byte-identical: submission.csv %s, other CSVs %d/%d' % (same, n_bytes, len(base_csvs)), flush=True)
mj, mb = markers(J), markers(B)
diff = {k: (mj.get(k, 0), mb.get(k, 0)) for k in set(mj) | set(mb) if mj.get(k, 0) != mb.get(k, 0)}
chk('log markers (line-start TOKENS, without DEC_OK / PROGRESS) == the base log: same stage / fallback / canary path',
    not diff, diff or '%d marker kinds' % len(mj))
cnt = lambda t: {k: len(re.findall(k, t)) for k in ('Traceback', 'RuntimeError', 'FALLBACK', 'SKIPPED', 'TIMEOUT', 'CANARY_FIRED')}
cj, cb = cnt(J.replace('DEC_FALLBACK', '')), cnt(B)
chk('no more Traceback / RuntimeError / FALLBACK / SKIPPED / TIMEOUT / CANARY_FIRED mentions than the base log',
    all(cj[k] <= cb[k] for k in cj), {'J': cj, 'base': cb})


# ---- (D) info ----------------------------------------------------------------------------------------------------------------------
def secs(t, marker):
    for ln in t.splitlines():
        if ln.startswith(marker):
            try:
                return float(ln.split()[-1])
            except Exception:
                return None
    return None


def last_time(p):
    try:
        return round(max(e.get('time', 0) for e in json.load(open(p, encoding='utf-8'))), 1)
    except Exception:
        return None


print('  info timing (s): DEC cell %s | RAPTOR_STAGE_DONE %s vs base %s | last log entry %s vs base %s' % (
    rc.get('seconds'), secs(J, 'RAPTOR_STAGE_DONE seconds'), secs(B, 'RAPTOR_STAGE_DONE seconds'), last_time(jl), last_time(bl)), flush=True)
ok = all(R)
print('J_CHECK %s %d/%d tag %s (submission byte-identical %s, other CSVs byte-identical %d/%d)' % (
    'PASS' if ok else 'FAIL', sum(R), len(R), tag, same, n_bytes, len(base_csvs)), flush=True)
sys.exit(0 if ok else 1)
