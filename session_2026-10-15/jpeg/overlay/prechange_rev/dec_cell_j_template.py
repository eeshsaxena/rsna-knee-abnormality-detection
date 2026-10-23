# [eeshsaxena J DECODERS (jpeg/overlay/make_j.py); not in the public notebook] This docker image (Python 3.13) has no JPEG Lossless
# decoder (no pylibjpeg, no gdcm): a JPEG Lossless slice (1.2.840.10008.1.2.4.57 / .70) is zero-filled without a message by Raptor
# r384 / r224 and dropped with its series by R7, our fleet and S6fall (session_2026-10-15/jpeg/RESULT.md, MEASURED on this pin).
# This cell installs pylibjpeg 2.1.0 + pylibjpeg-libjpeg 2.4.0 + pylibjpeg-openjpeg 2.6.0 (cp313 wheels, each sha256 == the
# official PyPI file, from the public dataset glorysing/rsna-knee-offline-codecs-r09) into /tmp/_dec with
# pip --no-index --no-deps --target, and ONLY after a fresh subprocess with /tmp/_dec on its PYTHONPATH imports them and decodes
# the embedded self-test images exactly (JPEG Lossless .70 16-bit signed, .57 12-bit unsigned, an uncompressed control) it
# makes them visible to every later decoding process:
#   os.environ['PYTHONPATH'] = /tmp/_dec + old: every stage process builds its environment from os.environ (Raptor r384 / r224
#   subprocess.run, the R7 controller env and its workers, the fleet workers, the S6fall driver and its workers);
#   sys.path of this notebook process gets /tmp/_dec first (in-process decoding), only while pydicom is not imported yet.
# Uncompressed, implicit VR, RLE and JPEG 2000 data decode byte-identically with and without the decoders (MEASURED:
# eeshsaxena/rsna-v224f20j-check). ANY failure: nothing is changed (= exactly the base notebook), DEC_FALLBACK and J_UNAVAILABLE
# are printed and dec_receipt.json says why; this cell never raises. Versions: DEC_OK line + /kaggle/working/dec_receipt.json.
def _dec_main():
    import os, sys, json, time, glob, shutil, hashlib, platform, importlib, subprocess
    INPUT, TARGET, WORK, SLUG = '/kaggle/input', '/tmp/_dec', '/kaggle/working', 'rsna-knee-offline-codecs-r09'
    PIP_TIMEOUT_S, TEST_TIMEOUT_S = 300, 300
    WHEELS = __WHEELS__
    TOP = __TOP__
    PINS = ['pylibjpeg==2.1.0', 'pylibjpeg-libjpeg==2.4.0', 'pylibjpeg-openjpeg==2.6.0']
    VERSIONS = {'pylibjpeg': '2.1.0', 'libjpeg': '2.4.0', 'openjpeg': '2.6.0'}
    FIX = __FIX__
    TEST = r'''
import os, sys, io, json, base64, hashlib, platform
import numpy as np
import pydicom, pylibjpeg, libjpeg, openjpeg
from pydicom.pixels import get_decoder
from pydicom.uid import UID
fx = json.load(open(sys.argv[1]))
m3 = ('pylibjpeg', 'libjpeg', 'openjpeg')
r = {'python': platform.python_version(), 'pydicom': pydicom.__version__, 'numpy': np.__version__,
     'pydicom_dir': os.path.dirname(os.path.dirname(os.path.abspath(pydicom.__file__))),
     'versions': {m: sys.modules[m].__version__ for m in m3},
     'dirs': {m: os.path.dirname(os.path.dirname(os.path.abspath(sys.modules[m].__file__))) for m in m3},
     'plugins': {u: sorted(get_decoder(UID(u)).available_plugins) for u in ('1.2.840.10008.1.2.4.57', '1.2.840.10008.1.2.4.70')}}
dec = {}
for name, (b64, dtype, shape, digest) in sorted(fx.items()):
    try:
        ds = pydicom.dcmread(io.BytesIO(base64.b64decode(b64)))
        a = ds.pixel_array
        dec[name] = {'ts': str(ds.file_meta.TransferSyntaxUID), 'dtype': str(a.dtype), 'shape': list(a.shape),
                     'exact': bool(str(a.dtype) == dtype and list(a.shape) == shape and
                                   hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() == digest)}
    except Exception as e:
        dec[name] = {'error': '%s: %s' % (type(e).__name__, str(e)[:200])}
r['decode'] = dec
print('DEC_SELFTEST ' + json.dumps(r), flush=True)
'''

    class DecFail(Exception):
        pass

    t0 = time.time()
    had_pp = 'PYTHONPATH' in os.environ
    before = os.environ.get('PYTHONPATH', '')
    after = os.pathsep.join([TARGET] + ([before] if before else []))     # no empty entry (an empty entry would add the cwd)
    path0 = list(sys.path)
    rec = {'tag': '__TAG__', 'ok': False, 'python': platform.python_version(), 'pythonpath_before': before,
           'sys_path_inserted': False}
    try:
        dirs = []
        for pat in (INPUT + '/' + SLUG, INPUT + '/*/' + SLUG, INPUT + '/datasets/*/' + SLUG):
            for d in sorted(glob.glob(pat)):
                if os.path.isdir(d) and d not in dirs:
                    dirs.append(d)
        found = {}
        for name, digest in WHEELS.items():
            for d in dirs:
                for p in [os.path.join(d, name)] + sorted(glob.glob(os.path.join(d, '*', name))):
                    if name not in found and os.path.isfile(p) and hashlib.sha256(open(p, 'rb').read()).hexdigest() == digest:
                        found[name] = p
        rec['dataset_dirs'] = dirs
        rec['wheels'] = found
        if sorted(found) != sorted(WHEELS):
            raise DecFail('PyPI-identical wheels not found: %s' % sorted(set(WHEELS) - set(found)))
        wdir = TARGET + '_wheels'
        for d in (TARGET, wdir):
            shutil.rmtree(d, ignore_errors=True)
        os.makedirs(wdir)
        for name, p in found.items():                              # pip sees ONLY the three sha-checked files
            shutil.copyfile(p, os.path.join(wdir, name))
            if hashlib.sha256(open(os.path.join(wdir, name), 'rb').read()).hexdigest() != WHEELS[name]:
                raise DecFail('copy of %s differs' % name)
        r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', '--target', TARGET,
                            '--find-links', wdir] + PINS, capture_output=True, text=True, timeout=PIP_TIMEOUT_S)
        rec['pip_rc'] = r.returncode
        rec['pip_out'] = (r.stdout + r.stderr)[-800:]
        if r.returncode != 0:
            raise DecFail('pip exit %d' % r.returncode)
        top = sorted(os.listdir(TARGET))
        rec['target_files'] = top
        if top != sorted(TOP):                                    # nothing else (no numpy / pydicom) can shadow the image
            raise DecFail('target dir content differs: %s' % top)
        fxp = TARGET + '_selftest.json'
        with open(fxp, 'w') as f:
            json.dump(FIX, f)
        t = subprocess.run([sys.executable, '-c', TEST, fxp], env=dict(os.environ, PYTHONPATH=after), cwd=wdir,
                           capture_output=True, text=True, timeout=TEST_TIMEOUT_S)
        line = next((ln for ln in t.stdout.splitlines() if ln.startswith('DEC_SELFTEST ')), None)
        if t.returncode != 0 or line is None:
            raise DecFail('self-test exit %s: %s' % (t.returncode, (t.stderr or '')[-400:]))
        st = json.loads(line[len('DEC_SELFTEST '):])
        rec['selftest'] = st
        if st['versions'] != VERSIONS:
            raise DecFail('versions %s' % st['versions'])
        if any(v != TARGET for v in st['dirs'].values()):
            raise DecFail('decoders not imported from %s: %s' % (TARGET, st['dirs']))
        if st['pydicom_dir'] == TARGET:
            raise DecFail('pydicom imported from %s' % TARGET)
        if any('pylibjpeg' not in v for v in st['plugins'].values()):
            raise DecFail('pydicom does not list pylibjpeg for JPEG Lossless: %s' % st['plugins'])
        if sorted(st['decode']) != sorted(FIX) or not all(v.get('exact') is True for v in st['decode'].values()):
            raise DecFail('self-test decode not exact: %s' % st['decode'])
        inproc = not any(m == 'pydicom' or m.startswith('pydicom.') for m in sys.modules)
        try:                                                      # the only changes this cell makes for later cells
            os.environ['PYTHONPATH'] = after
            if inproc and TARGET not in sys.path:
                sys.path.insert(0, TARGET)
                importlib.invalidate_caches()
        except BaseException:
            sys.path[:] = path0
            if had_pp:
                os.environ['PYTHONPATH'] = before
            else:
                os.environ.pop('PYTHONPATH', None)
            raise
        rec['sys_path_inserted'] = bool(inproc and sys.path[:1] == [TARGET])
        if not inproc:
            rec['sys_path_note'] = 'pydicom already imported in this process: in-process sys.path left as it was'
        rec['ok'] = True
    except Exception as e:
        rec['why'] = '%s: %s' % (type(e).__name__, str(e)[:600])
    rec['pythonpath_after'] = os.environ.get('PYTHONPATH', '')
    rec['sys_path_head'] = sys.path[:3]
    rec['seconds'] = round(time.time() - t0, 2)
    try:
        with open(os.path.join(WORK, 'dec_receipt.json'), 'w') as f:
            json.dump(rec, f, indent=1)
    except Exception as e:
        rec['receipt_error'] = '%s: %s' % (type(e).__name__, str(e)[:200])
    short = {k: rec.get(k) for k in ('tag', 'ok', 'why', 'seconds', 'pythonpath_after', 'sys_path_inserted', 'pip_rc', 'python')}
    if rec['ok']:
        st = rec['selftest']
        short.update(pydicom=st['pydicom'], numpy=st['numpy'], versions=st['versions'], plugins=st['plugins'],
                     decode={k: v.get('exact') for k, v in st['decode'].items()})
        print('DEC_OK ' + json.dumps(short), flush=True)
    else:
        print('DEC_FALLBACK ' + json.dumps(short), flush=True)
        print('J_UNAVAILABLE ' + json.dumps({'tag': rec.get('tag'), 'why': rec.get('why'), 'effect': 'base notebook unchanged'}),
              flush=True)
    return rec


try:
    _DEC_RECEIPT = _dec_main()
except Exception as _dec_e:                                         # belt and braces: _dec_main catches everything itself
    _DEC_RECEIPT = {'tag': '__TAG__', 'ok': False, 'why': 'DEC cell: %s' % type(_dec_e).__name__}
    print('DEC_FALLBACK ' + str(_DEC_RECEIPT), flush=True)
    print('J_UNAVAILABLE ' + str(_DEC_RECEIPT), flush=True)
del _dec_main
