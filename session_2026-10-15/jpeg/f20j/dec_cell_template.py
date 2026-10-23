# [eeshsaxena V224-F20-J DECODERS; not in the public notebook] This docker image has no JPEG Lossless decoder (no pylibjpeg,
# no gdcm): a JPEG Lossless slice (1.2.840.10008.1.2.4.57 / .70) is zero-filled without a message by Raptor r384 / r224, and
# dropped with its series by R7 and by our fleet (session_2026-10-15/jpeg/RESULT.md, MEASURED on this docker pin). This cell
# installs pylibjpeg 2.1.0 + pylibjpeg-libjpeg 2.4.0 + pylibjpeg-openjpeg 2.6.0 (cp313 wheels, each sha256 == the official
# PyPI file, taken from the public dataset glorysing/rsna-knee-offline-codecs-r09) into /tmp/_dec with
# pip --no-index --no-deps --target, and prepends /tmp/_dec to os.environ['PYTHONPATH'] ONLY after a fresh subprocess with
# that PYTHONPATH imports them and decodes the embedded self-test images exactly (JPEG Lossless .70 16-bit signed, .57 12-bit
# unsigned, and an uncompressed control). Every stage process inherits it: Raptor r384 / r224 (subprocess.run with this
# environment), R7 (controller env = CODE + PYTHONPATH; its workers get dict(os.environ)), our fleet workers (fleet_dir +
# PYTHONPATH). This notebook process itself is not changed (its sys.path is untouched). Uncompressed, implicit VR, RLE and
# JPEG 2000 data decode byte-identically with and without it (MEASURED: eeshsaxena/rsna-v224f20j-check).
# ANY failure: PYTHONPATH is left untouched (= exactly V224-F20) and DEC_FALLBACK is printed; nothing raises.
def _dec_main():
    import os, sys, json, time, glob, shutil, hashlib, subprocess
    INPUT, TARGET, WORK, SLUG = '/kaggle/input', '/tmp/_dec', '/kaggle/working', 'rsna-knee-offline-codecs-r09'
    WHEELS = __WHEELS__
    PINS = ['pylibjpeg==2.1.0', 'pylibjpeg-libjpeg==2.4.0', 'pylibjpeg-openjpeg==2.6.0']
    VERSIONS = {'pylibjpeg': '2.1.0', 'libjpeg': '2.4.0', 'openjpeg': '2.6.0'}
    TOP = __TOP__
    FIX = __FIX__
    TEST = r'''
import os, sys, io, json, base64, hashlib
import numpy as np
import pydicom, pylibjpeg, libjpeg, openjpeg
from pydicom.pixels import get_decoder
from pydicom.uid import UID
fx = json.load(open(sys.argv[1]))
m3 = ('pylibjpeg', 'libjpeg', 'openjpeg')
r = {'pydicom': pydicom.__version__, 'numpy': np.__version__,
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
    before = os.environ.get('PYTHONPATH', '')
    after = os.pathsep.join([TARGET] + ([before] if before else []))     # no empty entry (an empty entry would add the cwd)
    rec = {'tag': '__TAG__', 'ok': False, 'pythonpath_before': before}
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
                            '--find-links', wdir] + PINS, capture_output=True, text=True, timeout=900)
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
                           capture_output=True, text=True, timeout=600)
        line = next((ln for ln in t.stdout.splitlines() if ln.startswith('DEC_SELFTEST ')), None)
        if t.returncode != 0 or line is None:
            raise DecFail('self-test exit %s: %s' % (t.returncode, (t.stderr or '')[-400:]))
        st = json.loads(line[len('DEC_SELFTEST '):])
        rec['selftest'] = st
        if st['versions'] != VERSIONS:
            raise DecFail('versions %s' % st['versions'])
        if any(v != TARGET for v in st['dirs'].values()):
            raise DecFail('decoders not imported from %s: %s' % (TARGET, st['dirs']))
        if any('pylibjpeg' not in v for v in st['plugins'].values()):
            raise DecFail('pydicom does not list pylibjpeg for JPEG Lossless: %s' % st['plugins'])
        if sorted(st['decode']) != sorted(FIX) or not all(v.get('exact') is True for v in st['decode'].values()):
            raise DecFail('self-test decode not exact: %s' % st['decode'])
        os.environ['PYTHONPATH'] = after                          # the only change this cell makes for later cells
        rec['ok'] = True
    except Exception as e:
        rec['why'] = '%s: %s' % (type(e).__name__, str(e)[:600])
    rec['pythonpath_after'] = os.environ.get('PYTHONPATH', '')
    rec['seconds'] = round(time.time() - t0, 2)
    try:
        with open(os.path.join(WORK, 'dec_receipt.json'), 'w') as f:
            json.dump(rec, f, indent=1)
    except Exception as e:
        rec['receipt_error'] = '%s: %s' % (type(e).__name__, str(e)[:200])
    short = {k: rec.get(k) for k in ('tag', 'ok', 'why', 'seconds', 'pythonpath_after', 'pip_rc')}
    if rec['ok']:
        st = rec['selftest']
        short.update(pydicom=st['pydicom'], versions=st['versions'], plugins=st['plugins'],
                     decode={k: v.get('exact') for k, v in st['decode'].items()})
        print('DEC_OK ' + json.dumps(short), flush=True)
    else:
        print('DEC_FALLBACK ' + json.dumps(short), flush=True)
    return rec


_DEC_RECEIPT = _dec_main()
del _dec_main
