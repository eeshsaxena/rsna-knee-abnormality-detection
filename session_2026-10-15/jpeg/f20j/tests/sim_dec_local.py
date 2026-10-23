# sim_dec_local.py (Claude, 2026-10-14): local (Windows, Python312 + pydicom 3.0.2, NO pylibjpeg) checks of the V224-F20-J DEC
# cell taken verbatim from the built notebook, with only its path line redirected to temp dirs. Positive path = in-kernel only
# (manylinux cp313 wheels). Expected here: every case -> DEC_FALLBACK, PYTHONPATH unchanged, no exception escapes the cell,
# no 'Traceback' / 'RuntimeError' printed, receipt written. Plus the self-test program on the uncompressed control.
import os, sys, io, json, shutil, tempfile, subprocess, contextlib
HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, '..', 'kernel_v224f20j', 'rsna_sub_v224f20j.ipynb')
nb = json.load(open(NB, encoding='utf-8'))
DEC = nb['cells'][2]['source']
LINE = "    INPUT, TARGET, WORK, SLUG = '/kaggle/input', '/tmp/_dec', '/kaggle/working', 'rsna-knee-offline-codecs-r09'"
assert DEC.count(LINE) == 1
res = []


def run(name, setup, expect_ok=False, mutate=None):
    tmp = tempfile.mkdtemp(prefix='decsim_')
    inp, tgt, work = os.path.join(tmp, 'input'), os.path.join(tmp, 'dec'), os.path.join(tmp, 'work')
    os.makedirs(inp); os.makedirs(work)
    setup(inp)
    code = DEC.replace(LINE, "    INPUT, TARGET, WORK, SLUG = %r, %r, %r, 'rsna-knee-offline-codecs-r09'" % (
        inp.replace(os.sep, '/'), tgt.replace(os.sep, '/'), work.replace(os.sep, '/')))
    if mutate:
        code = mutate(code)
    os.environ['PYTHONPATH'] = 'A' + os.pathsep + 'B'
    buf = io.StringIO()
    g = {'__name__': '__sim__'}
    exc = None
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, 'dec_cell', 'exec'), g)
    except BaseException as e:
        exc = repr(e)
    out = buf.getvalue()
    rec = g.get('_DEC_RECEIPT') or {}
    ok = (exc is None and ('DEC_OK' in out) == expect_ok and ('DEC_FALLBACK' in out) == (not expect_ok)
          and os.environ['PYTHONPATH'] == ('A' + os.pathsep + 'B') and 'Traceback' not in out and 'RuntimeError' not in out
          and os.path.isfile(os.path.join(work, 'dec_receipt.json')) and '_dec_main' not in g)
    res.append(ok)
    print('%s %s | exc=%s | out=%s | why=%s' % ('OK  ' if ok else 'FAIL', name, exc, out.strip()[:160], rec.get('why')))
    shutil.rmtree(tmp, ignore_errors=True)


WHEELS = eval(DEC.split('    WHEELS = ', 1)[1].split('\n', 1)[0])


def none(inp):
    pass


def fake_wheels(inp):           # right names, wrong bytes (a tampered / different dataset version)
    d = os.path.join(inp, 'rsna-knee-offline-codecs-r09'); os.makedirs(d)
    for n in WHEELS:
        open(os.path.join(d, n), 'wb').write(b'not the PyPI file')


def two_of_three(inp):
    d = os.path.join(inp, 'datasets', 'glorysing', 'rsna-knee-offline-codecs-r09'); os.makedirs(d)
    for n in list(WHEELS)[:2]:
        open(os.path.join(d, n), 'wb').write(b'x')


run('no dataset mounted', none)
run('tampered wheels (sha != PyPI)', fake_wheels)
run('wheels missing / partial (datasets/*/ layout)', two_of_three)
run('pip failure path (sha check bypassed by a mutated copy; win32 cannot install manylinux)', fake_wheels,
    mutate=lambda c: c.replace("hashlib.sha256(open(p, 'rb').read()).hexdigest() == digest", 'True')
                      .replace("if hashlib.sha256(open(os.path.join(wdir, name), 'rb').read()).hexdigest() != WHEELS[name]:", 'if False:'))
# the self-test program itself, on the uncompressed control only (no pylibjpeg here: import lines stubbed)
TEST = DEC.split("    TEST = r'''", 1)[1].split("'''", 1)[0]
FIX = eval(DEC.split('    FIX = ', 1)[1].split('\n', 1)[0])
tmp = tempfile.mkdtemp(prefix='dectest_')
fxp = os.path.join(tmp, 'fx.json'); json.dump({'U_16S': FIX['U_16S']}, open(fxp, 'w'))
t = TEST.replace('import pydicom, pylibjpeg, libjpeg, openjpeg', 'import pydicom').replace("m3 = ('pylibjpeg', 'libjpeg', 'openjpeg')", "m3 = ('pydicom',)")
r = subprocess.run([sys.executable, '-c', t, fxp], capture_output=True, text=True)
line = next((l for l in r.stdout.splitlines() if l.startswith('DEC_SELFTEST ')), '')
st = json.loads(line[len('DEC_SELFTEST '):]) if line else {}
ok = r.returncode == 0 and st.get('decode', {}).get('U_16S', {}).get('exact') is True
res.append(ok)
print('%s self-test program decodes the uncompressed control exactly (local pydicom %s): %s %s' % (
    'OK  ' if ok else 'FAIL', st.get('pydicom'), st.get('decode'), r.stderr[-300:]))
# and it reports JPEG Lossless as an error (not exact) when no decoder is present
json.dump(FIX, open(fxp, 'w'))
r = subprocess.run([sys.executable, '-c', t, fxp], capture_output=True, text=True)
line = next((l for l in r.stdout.splitlines() if l.startswith('DEC_SELFTEST ')), '')
st = json.loads(line[len('DEC_SELFTEST '):]) if line else {}
d = st.get('decode', {})
ok = r.returncode == 0 and d.get('U_16S', {}).get('exact') is True and 'error' in d.get('JLL70_16S', {}) and 'error' in d.get('JLL57_12U', {})
res.append(ok)
print('%s without decoders the self-test flags both JPEG Lossless images: %s' % ('OK  ' if ok else 'FAIL', json.dumps(d)[:300]))
shutil.rmtree(tmp, ignore_errors=True)
print('SIM_DEC_LOCAL %s %d/%d' % ('PASS' if all(res) else 'FAIL', sum(res), len(res)))
