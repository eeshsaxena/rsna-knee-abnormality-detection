# neg_gate.py (Claude, 2026-10-14): offline tests of v224f20j_check.py. A simulated clean V224-F20-J output = the REAL V224-F20 v1
# commit output + one DEC_OK log entry before stage 1 + dec_receipt.json (the receipt the DEC cell wrote in the CPU check kernel
# when given, else a synthesised one). The clean copy must PASS; each mutation must FAIL.
#   usage: python tests/neg_gate.py [--receipt out_jj_v1/dec_receipt.json]
import os, sys, json, shutil, tempfile, subprocess, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..', '..', '..'))
REF = os.path.join(ROOT, 'session_2026-10-15', 'v224f', 'out_v224f20_v1')
GATE = os.path.join(HERE, '..', 'v224f20j_check.py')
ap = argparse.ArgumentParser()
ap.add_argument('--receipt', default=None)
A = ap.parse_args()
PB = '/kaggle/lib/kagglegym:/kaggle/lib:/kaggle/input/competitions/rsna-knee-abnormality-detection'
if A.receipt:
    rec = json.load(open(A.receipt, encoding='utf-8'))
    src = 'CPU check kernel receipt ' + A.receipt
else:
    rec = {'tag': 'v224f20j', 'ok': True, 'pythonpath_before': PB, 'pip_rc': 0,
           'wheels': {n: '/kaggle/input/rsna-knee-offline-codecs-r09/' + n for n in (
               'pylibjpeg-2.1.0-py3-none-any.whl', 'pylibjpeg_libjpeg-2.4.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl',
               'pylibjpeg_openjpeg-2.6.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl')},
           'target_files': ['_libjpeg.cpython-313-x86_64-linux-gnu.so', '_openjpeg.cpython-313-x86_64-linux-gnu.so', 'libjpeg', 'openjpeg',
                            'pylibjpeg', 'pylibjpeg-2.1.0.dist-info', 'pylibjpeg_libjpeg-2.4.0.dist-info', 'pylibjpeg_openjpeg-2.6.0.dist-info'],
           'selftest': {'pydicom': '3.0.2', 'versions': {'pylibjpeg': '2.1.0', 'libjpeg': '2.4.0', 'openjpeg': '2.6.0'},
                        'dirs': {'pylibjpeg': '/tmp/_dec', 'libjpeg': '/tmp/_dec', 'openjpeg': '/tmp/_dec'},
                        'plugins': {'1.2.840.10008.1.2.4.57': ['pylibjpeg'], '1.2.840.10008.1.2.4.70': ['pylibjpeg']},
                        'decode': {k: {'exact': True} for k in ('JLL57_12U', 'JLL70_16S', 'U_16S')}},
           'pythonpath_after': '/tmp/_dec:' + PB, 'seconds': 4.2}
    src = 'synthesised receipt'
# a CPU-kernel receipt has the CPU kernel's PYTHONPATH; the GPU notebook's will have its own: both are checked for the same form
st = rec['selftest']
short = {k: rec.get(k) for k in ('tag', 'ok', 'why', 'seconds', 'pythonpath_after', 'pip_rc')}
short.update(pydicom=st['pydicom'], versions=st['versions'], plugins=st['plugins'], decode={k: v.get('exact') for k, v in st['decode'].items()})
DEC_LINE = 'DEC_OK ' + json.dumps(short)


def make(mut=None):
    d = tempfile.mkdtemp(prefix='f20j_sim_')
    for f in os.listdir(REF):
        p = os.path.join(REF, f)
        if os.path.isfile(p):
            shutil.copyfile(p, os.path.join(d, f))
    log = os.path.join(d, 'rsna-sub-v224f20.log')
    entries = json.load(open(log, encoding='utf-8'))
    k = next(i for i, e in enumerate(entries) if 'RAPTOR_STAGE_DONE' in e.get('data', '') or '[RAPTOR' in e.get('data', ''))
    k = min(k, next(i for i, e in enumerate(entries) if e.get('stream_name') == 'stdout'))      # before the first stage output
    r = json.loads(json.dumps(rec))
    lines = [DEC_LINE]
    if mut:
        lines, r, k, entries = mut(lines, r, k, entries, d)
    for j, ln in enumerate(lines):
        entries.insert(k + j, {'stream_name': 'stdout', 'time': 1.0, 'data': ln + '\n'})
    os.remove(log)
    json.dump(entries, open(os.path.join(d, 'rsna-sub-v224f20j.log'), 'w', encoding='utf-8'))
    json.dump(r, open(os.path.join(d, 'dec_receipt.json'), 'w'), indent=1)
    return d


def run(name, mut, want_pass, fast=True):
    d = make(mut)
    r = subprocess.run([sys.executable, GATE, d] + (['--skip-f-gate'] if fast else []), capture_output=True, text=True, encoding='utf-8')
    last = [ln for ln in r.stdout.splitlines() if ln.startswith('V224F20J_CHECK ')][-1]
    fails = [ln.strip() for ln in r.stdout.splitlines() if ln.strip().startswith('FAIL')]
    ok = (r.returncode == 0) == want_pass and ('PASS' in last) == want_pass
    print('%s %-58s -> %s %s' % ('OK  ' if ok else 'BAD ', name, last, ('| ' + fails[0][:150]) if fails else ''), flush=True)
    shutil.rmtree(d, ignore_errors=True)
    return ok


def m_fallback(lines, r, k, e, d):
    return [lines[0].replace('DEC_OK ', 'DEC_FALLBACK ')], dict(r, ok=False), k, e


def m_after_raptor(lines, r, k, e, d):
    return lines, r, next(i for i, x in enumerate(e) if 'RAPTOR_STAGE_DONE' in x.get('data', '')) + 1, e


def m_numpy(lines, r, k, e, d):
    r['target_files'] = sorted(r['target_files'] + ['numpy'])
    return lines, r, k, e


def m_sub(lines, r, k, e, d):
    import pandas as pd
    p = os.path.join(d, 'submission.csv'); s = pd.read_csv(p, dtype={'StudyInstanceUID': str}); s.iloc[0, 1] += 0.01; s.to_csv(p, index=False)
    return lines, r, k, e


def m_raw(lines, r, k, e, d):
    import pandas as pd
    p = os.path.join(d, 'raptor_raw.csv'); s = pd.read_csv(p, dtype={'StudyInstanceUID': str}); s.iloc[1, 2] += 0.01; s.to_csv(p, index=False)
    return lines, r, k, e


def m_two(lines, r, k, e, d):
    return lines + lines, r, k, e


def m_pp(lines, r, k, e, d):
    r['pythonpath_after'] = r['pythonpath_before']
    return [lines[0].replace('/tmp/_dec:', '')], r, k, e


def m_selftest(lines, r, k, e, d):
    return [lines[0].replace('"JLL57_12U": true', '"JLL57_12U": false')], r, k, e


def m_wheel(lines, r, k, e, d):
    r['wheels'] = dict(list(r['wheels'].items())[:2])
    return lines, r, k, e


def m_noreceipt(lines, r, k, e, d):
    return lines, {}, k, e


print('neg_gate: base = %s + %s' % (REF, src), flush=True)
res = [run('clean simulated J output (full gate incl. V224-F20 gate)', None, True, fast=False),
       run('clean simulated J output (fast)', None, True)]
for name, m in (('DEC_FALLBACK instead of DEC_OK', m_fallback), ('DEC_OK after stage 1', m_after_raptor),
                ('extra package (numpy) in /tmp/_dec', m_numpy), ('submission.csv cell +.01', m_sub), ('raptor_raw.csv cell +.01', m_raw),
                ('two DEC_OK lines', m_two), ('PYTHONPATH not extended', m_pp), ('self-test JLL57 not exact', m_selftest),
                ('only 2 of 3 wheels found', m_wheel), ('dec_receipt.json missing / empty', m_noreceipt)):
    res.append(run(name, m, False))
res.append(run('submission.csv cell +.01 (full gate)', m_sub, False, fast=False))
print('NEG_GATE %s %d/%d' % ('PASS' if all(res) else 'FAIL', sum(res), len(res)))
