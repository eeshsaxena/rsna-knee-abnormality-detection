# [eeshsaxena J DECODERS (jpeg/overlay/make_j.py); not in the public notebook] This docker image (Python 3.13) has no JPEG Lossless
# decoder (no pylibjpeg, no gdcm): a JPEG Lossless slice (1.2.840.10008.1.2.4.57 / .70) is zero-filled without a message by Raptor
# r384 / r224 and dropped with its series by R7, our fleet and S6fall (session_2026-10-15/jpeg/RESULT.md, MEASURED on this pin).
# This cell installs pylibjpeg 2.1.0 + pylibjpeg-libjpeg 2.4.0 + pylibjpeg-openjpeg 2.6.0 (cp313 wheels, each sha256 == the
# official PyPI file, from the public dataset glorysing/rsna-knee-offline-codecs-r09) into /tmp/_dec with
# pip --no-index --no-deps --target. ONLY after (1) a fresh subprocess with /tmp/_dec on its PYTHONPATH imports them and decodes
# the embedded self-test images exactly (JPEG Lossless .70 16-bit signed, .57 12-bit unsigned, an uncompressed control) and
# (2) a PREFLIGHT on the test data itself passes (the transfer syntax of the first and last file of every series dir is read; every
# file of a series whose probe is not uncompressed / RLE / deflate is decoded with the new decoders in throwaway subprocesses, so
# a native decoder fault or hang on a hidden file happens here, under a time cap, and switches J off instead of hanging or killing
# a stage; a Python exception is fine: pydicom then tries its other plugins, the stages catch the rest as in the base)
# it makes them visible to every later decoding process:
#   os.environ['PYTHONPATH'] = /tmp/_dec + old: every stage process builds its environment from os.environ (Raptor r384 / r224
#   subprocess.run, the R7 controller env and its workers, the fleet workers, the S6fall driver and its workers);
#   sys.path of this notebook process gets /tmp/_dec first (in-process decoding), only while pydicom is not imported yet;
#   a daemon thread (reads /proc only, prints nothing) records the argv / PYTHONPATH of every later Python process in
#   /kaggle/working/dec_envwatch.json: the in-run evidence that each stage process really had /tmp/_dec on its path.
# Uncompressed, implicit VR, RLE and JPEG 2000 data decode byte-identically with and without the decoders (MEASURED:
# eeshsaxena/rsna-v224f20j-check). ANY failure: nothing is changed (= exactly the base notebook), DEC_FALLBACK and J_UNAVAILABLE
# are printed and dec_receipt.json says why; this cell never raises. Versions: DEC_OK line + /kaggle/working/dec_receipt.json.
def _dec_main():
    import os, sys, json, time, glob, shutil, hashlib, platform, importlib, subprocess, threading
    INPUT, TARGET, WORK, SLUG = '/kaggle/input', '/tmp/_dec', '/kaggle/working', 'rsna-knee-offline-codecs-r09'
    PIP_TIMEOUT_S, TEST_TIMEOUT_S = 300, 300
    PRE_TIMEOUT_S, WATCH_EVERY_S, PROC = 600, 1.0, '/proc'
    COMP = 'rsna-knee-abnormality-detection'
    PRE_PROCS = max(1, min(4, os.cpu_count() or 1))
    WHEELS = {'pylibjpeg-2.1.0-py3-none-any.whl': '25df9496a69e64e98c887fddee12a1271e275b5f74ba804f9bf98a08bb80993e', 'pylibjpeg_libjpeg-2.4.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl': 'd2c8da12c79d1f46596b89df94091be01f932d739613e8af93d1d557ac3bab74', 'pylibjpeg_openjpeg-2.6.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl': '3e8ee9e5bd3b62766ca98fc143cd340633e60bb14450162426f32688908ad9e8'}
    TOP = ['_libjpeg.cpython-313-x86_64-linux-gnu.so', '_openjpeg.cpython-313-x86_64-linux-gnu.so', 'libjpeg', 'openjpeg', 'pylibjpeg', 'pylibjpeg-2.1.0.dist-info', 'pylibjpeg_libjpeg-2.4.0.dist-info', 'pylibjpeg_openjpeg-2.6.0.dist-info']
    PINS = ['pylibjpeg==2.1.0', 'pylibjpeg-libjpeg==2.4.0', 'pylibjpeg-openjpeg==2.6.0']
    VERSIONS = {'pylibjpeg': '2.1.0', 'libjpeg': '2.4.0', 'openjpeg': '2.6.0'}
    FIX = {'JLL57_12U': ['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', 'uint16', [40, 56], 'b81f4c9079d9d0c53f1b6dc58f592c69941a7bbd347f0de0e9dbdaa474157f81'], 'JLL70_16S': ['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', 'int16', [40, 56], 'a70f1db4c227d7145c5e314b6d17b6f4d06b7d417d5de02848aff72e98a5a9ca'], 'U_16S': ['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', 'int16', [40, 56], 'a70f1db4c227d7145c5e314b6d17b6f4d06b7d417d5de02848aff72e98a5a9ca']}
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
    # preflight 1/2: transfer syntax of the first + last file of every dir under the series dir (raw file-meta read, no pydicom)
    SCAN = r'''
import os, sys, json, queue, threading
tsdir, out, nthr = sys.argv[1], sys.argv[2], int(sys.argv[3])
PLAIN = {'1.2.840.10008.1.2', '1.2.840.10008.1.2.1', '1.2.840.10008.1.2.2', '1.2.840.10008.1.2.1.99', '1.2.840.10008.1.2.5'}
LONG = (b'OB', b'OD', b'OF', b'OL', b'OV', b'OW', b'SQ', b'SV', b'UC', b'UN', b'UR', b'UT', b'UV')
def ts_of(p):
    try:
        with open(p, 'rb') as f:
            b = f.read(8192)
    except OSError:
        return '?unreadable'
    if len(b) < 132 or b[128:132] != b'DICM':
        return '?no-meta'
    pos = 132
    while pos + 8 <= len(b):
        grp, el = int.from_bytes(b[pos:pos + 2], 'little'), int.from_bytes(b[pos + 2:pos + 4], 'little')
        if grp != 2:
            break
        if b[pos + 4:pos + 6] in LONG:
            ln, v = int.from_bytes(b[pos + 8:pos + 12], 'little'), pos + 12
        else:
            ln, v = int.from_bytes(b[pos + 6:pos + 8], 'little'), pos + 8
        if el == 0x0010:
            return b[v:v + ln].decode('latin-1').rstrip('\x00 ') if v + ln <= len(b) else '?short'
        pos = v + ln
    return '?no-ts'
def ls(d):
    fs, ds = [], []
    for e in sorted(os.scandir(d), key=lambda e: e.name):
        if e.is_dir():
            ds.append(e.path)
        elif e.is_file():
            fs.append(e.path)
    return fs, ds
lock, R = threading.Lock(), {'n_groups': 0, 'n_files': 0, 'probe_ts': {}, 'errors': []}
flagged = []
def group(fs):
    t = [ts_of(fs[0])] + ([ts_of(fs[-1])] if len(fs) > 1 else [])
    with lock:
        R['n_groups'] += 1
        R['n_files'] += len(fs)
        for v in t:
            R['probe_ts'][v] = R['probe_ts'].get(v, 0) + 1
        if any(v not in PLAIN for v in t):
            flagged.extend(fs)
fs0, studies = ls(tsdir)
if fs0:
    group(fs0)
q = queue.Queue()
for s in studies:
    q.put(s)
def work():
    while True:
        try:
            d = q.get_nowait()
        except queue.Empty:
            return
        try:
            todo = [d]
            while todo:
                fs, ds = ls(todo.pop())
                todo += ds
                if fs:
                    group(fs)
        except Exception as e:
            with lock:
                R['errors'].append('%s: %s' % (type(e).__name__, str(e)[:160]))
th = [threading.Thread(target=work, daemon=True) for _ in range(nthr)]
for t in th:
    t.start()
for t in th:
    t.join()
R['n_studies'] = len(studies)
R['flagged'] = sorted(flagged)
with open(out, 'w') as f:
    json.dump(R, f)
print('DEC_SCAN_DONE %d %d %d' % (R['n_groups'], R['n_files'], len(flagged)), flush=True)
'''
    # preflight 2/2: decode every flagged file with the new decoders (a Python exception is counted, a dying process is not ok)
    PRE = r'''
import sys, json, warnings
warnings.simplefilter('ignore')
import pydicom
fs = json.load(open(sys.argv[1]))
n_ok, errs = 0, []
for p in fs:
    try:
        pydicom.dcmread(p).pixel_array
        n_ok += 1
    except Exception as e:
        errs.append('%s: %s' % (type(e).__name__, str(e)[:160]))
print('DEC_PREFLIGHT_DONE ' + json.dumps({'n': len(fs), 'ok': n_ok, 'err': len(errs), 'errs': errs[:3]}), flush=True)
'''

    class DecFail(Exception):
        pass

    def preflight(pf, after, wdir):                                 # fills pf; raises DecFail on anything not proven safe
        t1 = time.time()
        deadline = t1 + PRE_TIMEOUT_S
        pf.update(ok=False, timeout_s=PRE_TIMEOUT_S)
        root = None
        for b in (INPUT + '/competitions/' + COMP, INPUT + '/' + COMP):        # the test-root search of the Raptor scripts
            if os.path.exists(b + '/test.csv'):
                root = b
                break
        for need_dir in (True, False):
            for d, _, f in (os.walk(INPUT) if root is None else ()):
                if time.time() > deadline:
                    raise DecFail('preflight: test-root search timeout')
                if 'test.csv' in f and (not need_dir or os.path.isdir(d + '/test_series') or os.path.isdir(d + '/test_images')):
                    root = d
                    break
        if root is None:
            raise DecFail('preflight: no test root (test.csv) under %s' % INPUT)
        tsdir = root + '/test_series' if os.path.isdir(root + '/test_series') else root + '/test_images'
        pf.update(test_root=root, series_dir=tsdir)
        if not os.path.isdir(tsdir):
            raise DecFail('preflight: no series dir %s' % tsdir)
        pdir = TARGET + '_pf'
        shutil.rmtree(pdir, ignore_errors=True)
        os.makedirs(pdir)
        scan_out = os.path.join(pdir, 'scan.json')
        try:
            s = subprocess.run([sys.executable, '-c', SCAN, tsdir, scan_out, '32'], capture_output=True, text=True, cwd=wdir,
                               timeout=max(1.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            raise DecFail('preflight: header scan timeout (%d s cap)' % PRE_TIMEOUT_S)
        if s.returncode != 0 or not os.path.isfile(scan_out):
            raise DecFail('preflight: header scan exit %s: %s' % (s.returncode, (s.stderr or '')[-300:]))
        with open(scan_out) as f:
            sc = json.load(f)
        fl = sc.pop('flagged')
        pf.update(sc)
        pf['n_flagged'] = len(fl)
        pf['flagged_sample'] = fl[:3]
        if sc['errors']:
            raise DecFail('preflight: header scan errors %s' % sc['errors'][:2])
        pf['decoded_ok'], pf['decoded_err'], pf['errs'] = 0, 0, []
        if fl:
            k = max(1, min(PRE_PROCS, len(fl)))
            procs = []
            try:
                for i in range(k):
                    lp = os.path.join(pdir, 'pre%d.json' % i)
                    with open(lp, 'w') as f:
                        json.dump(fl[i::k], f)
                    fo, fe = open(lp + '.out', 'w'), open(lp + '.err', 'w')
                    procs.append([None, fo, fe, lp, len(fl[i::k])])
                    procs[-1][0] = subprocess.Popen([sys.executable, '-c', PRE, lp], env=dict(os.environ, PYTHONPATH=after),
                                                    cwd=wdir, stdout=fo, stderr=fe, stdin=subprocess.DEVNULL)
                late = False
                for p in procs:
                    try:
                        p[0].wait(timeout=max(0.1, deadline - time.time()))
                    except subprocess.TimeoutExpired:
                        late = True
                        break
            finally:
                for p in procs:
                    if p[0] is not None and p[0].poll() is None:
                        p[0].kill()
                        try:
                            p[0].wait(timeout=30)
                        except Exception:
                            pass
                    p[1].close()
                    p[2].close()
            if late:
                raise DecFail('preflight: decode timeout (%d s cap; a decoder hangs or the compressed share is too large)' %
                              PRE_TIMEOUT_S)
            for p, fo, fe, lp, n in procs:
                out = open(lp + '.out').read()
                line = next((ln for ln in out.splitlines() if ln.startswith('DEC_PREFLIGHT_DONE ')), None)
                r = json.loads(line[len('DEC_PREFLIGHT_DONE '):]) if line else {}
                if p.returncode != 0 or r.get('n') != n:
                    raise DecFail('preflight: decode process exit %s after %s of %d files (native decoder fault on a test file?): %s'
                                  % (p.returncode, r.get('n'), n, open(lp + '.err').read()[-300:]))
                pf['decoded_ok'] += r['ok']
                pf['decoded_err'] += r['err']
                pf['errs'] += r['errs'][:2]
        pf['ok'] = True
        pf['seconds'] = round(time.time() - t1, 2)

    def envwatch(kpid, t_w, path):                                 # daemon thread: /proc reads + one json file, never prints
        import re
        st = {'ok': True, 'kernel_pid': kpid, 'started_unix': round(t_w, 3), 'every_s': WATCH_EVERY_S, 'target': TARGET,
              'sweeps': 0, 'n_python': 0, 'n_scope_python': 0, 'n_scope_bad': 0, 'scripts': {}, 'bad': [], 'records': [],
              'errors': 0}
        seen, ppid = set(), {}
        pyre = re.compile(r'^python[0-9.]*(\.exe)?$', re.I)

        def write():
            tmp = path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(st, f)
            os.replace(tmp, path)

        def stat(pid):
            with open('%s/%s/stat' % (PROC, pid)) as f:
                s = f.read()
            x = s[s.rindex(')') + 2:].split()
            return int(x[1]), x[19]                                    # ppid, starttime

        def argv_of(pid):
            with open('%s/%s/cmdline' % (PROC, pid), 'rb') as f:
                return [a.decode('utf-8', 'replace') for a in f.read().split(b'\0')[:-1]]

        def scope(pid):                                                # True: a descendant of this notebook process
            p, n = pid, 0
            while n < 64:
                if p == kpid:
                    return True
                if p in (0, 1):
                    return False
                if p not in ppid:
                    try:
                        ppid[p] = stat(p)[0]
                    except Exception:
                        return None
                p, n = ppid[p], n + 1
            return None

        def parse(argv):                                               # interpreter options before the script / -c / -m
            flags, i = [], 1
            while i < len(argv):
                a = argv[i]
                if a == '--':
                    i += 1
                    break
                if not a.startswith('-') or a == '-' or a.startswith('--'):
                    break
                j = 1
                while j < len(a):
                    c = a[j]
                    if c in 'IES':
                        flags.append('-' + c)
                    if c in 'cm':
                        rest = a[j + 1:] or (argv[i + 1] if i + 1 < len(argv) else '')
                        return flags, ('-c' if c == 'c' else '-m ' + rest[:60])
                    if c in 'WX':
                        if j + 1 == len(a):
                            i += 1                                     # -W / -X value is the next argument
                        break
                    j += 1
                i += 1
            return flags, (os.path.basename(argv[i]) if i < len(argv) else '')

        try:
            if not os.path.isdir(PROC):
                st.update(ok=False, why='no %s' % PROC)
                write()
                return
            try:
                kargv = argv_of(kpid)
            except Exception:
                kargv = None
            write()
        except Exception:
            return
        while True:
            try:
                ch = False
                for d in os.listdir(PROC):
                    if not d.isdigit() or int(d) == kpid:
                        continue
                    pid = int(d)
                    try:
                        pp, start = stat(pid)
                        argv = argv_of(pid)
                    except Exception:
                        continue
                    key = (pid, start, tuple(argv))
                    if key in seen:
                        continue
                    seen.add(key)
                    ppid[pid] = pp
                    if not argv or argv == kargv or not (pyre.match(os.path.basename(argv[0])) or argv[0] == sys.executable):
                        continue                                       # not Python, or a fork of this process before its exec
                    try:
                        with open('%s/%d/environ' % (PROC, pid), 'rb') as f:
                            env = f.read().split(b'\0')
                        pyp = next((kv[11:].decode('utf-8', 'replace') for kv in env if kv.startswith(b'PYTHONPATH=')), None)
                        envok = True
                    except Exception:
                        pyp, envok = None, False
                    flags, script = parse(argv)
                    sc = scope(pid)
                    has = pyp is not None and TARGET in pyp.split(os.pathsep)
                    bad = bool(flags) or (envok and not has)
                    r = {'pid': pid, 'ppid': pp, 't': round(time.time() - t_w, 1), 'script': script, 'flags': flags,
                         'argv': [a[:120] for a in argv[:6]], 'environ': envok, 'pythonpath': None if pyp is None else pyp[:400],
                         'dec_on_path': has, 'scope': sc}
                    st['n_python'] += 1
                    c = st['scripts'].setdefault(script, {'n': 0, 'scope': 0, 'scope_ok': 0, 'scope_bad': 0})
                    c['n'] += 1
                    if sc is not False:
                        st['n_scope_python'] += 1
                        c['scope'] += 1
                        if bad:
                            st['n_scope_bad'] += 1
                            c['scope_bad'] += 1
                            if len(st['bad']) < 50:
                                st['bad'].append(r)
                        elif envok:
                            c['scope_ok'] += 1
                    if len(st['records']) < 400:
                        st['records'].append(r)
                    ch = True
                st['sweeps'] += 1
                st['last_sweep_s'] = round(time.time() - t_w, 1)
                if ch or st['sweeps'] in (1, 10):
                    write()
            except Exception as e:
                st['errors'] += 1
                st['last_error'] = '%s: %s' % (type(e).__name__, str(e)[:200])
                if st['errors'] >= 1000:
                    st['ok'] = False
                    try:
                        write()
                    except Exception:
                        pass
                    return
            time.sleep(WATCH_EVERY_S)

    t0 = time.time()
    had_pp = 'PYTHONPATH' in os.environ
    before = os.environ.get('PYTHONPATH', '')
    after = os.pathsep.join([TARGET] + ([before] if before else []))     # no empty entry (an empty entry would add the cwd)
    path0 = list(sys.path)
    rec = {'tag': 'v224flips6fj', 'ok': False, 'python': platform.python_version(), 'pythonpath_before': before,
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
        rec['preflight'] = {}
        preflight(rec['preflight'], after, wdir)
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
    if rec['ok']:
        try:
            threading.Thread(target=envwatch, args=(os.getpid(), time.time(), os.path.join(WORK, 'dec_envwatch.json')),
                             name='dec_envwatch', daemon=True).start()
            rec['envwatch'] = 'started'
        except Exception as e:
            rec['envwatch'] = 'not started: %s' % type(e).__name__
    rec['pythonpath_after'] = os.environ.get('PYTHONPATH', '')
    rec['sys_path_head'] = sys.path[:3]
    rec['seconds'] = round(time.time() - t0, 2)
    try:
        with open(os.path.join(WORK, 'dec_receipt.json'), 'w') as f:
            json.dump(rec, f, indent=1)
    except Exception as e:
        rec['receipt_error'] = '%s: %s' % (type(e).__name__, str(e)[:200])
    short = {k: rec.get(k) for k in ('tag', 'ok', 'why', 'seconds', 'pythonpath_after', 'sys_path_inserted', 'pip_rc', 'python',
                                     'envwatch')}
    pf = rec.get('preflight') or {}
    short['preflight'] = {k: pf.get(k) for k in ('ok', 'n_studies', 'n_groups', 'n_files', 'n_flagged', 'decoded_ok', 'decoded_err',
                                                 'seconds')}
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
    _DEC_RECEIPT = {'tag': 'v224flips6fj', 'ok': False, 'why': 'DEC cell: %s' % type(_dec_e).__name__}
    print('DEC_FALLBACK ' + str(_DEC_RECEIPT), flush=True)
    print('J_UNAVAILABLE ' + str(_DEC_RECEIPT), flush=True)
del _dec_main
