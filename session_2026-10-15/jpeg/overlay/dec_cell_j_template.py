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
    _DEC_RECEIPT = {'tag': '__TAG__', 'ok': False, 'why': 'DEC cell: %s' % type(_dec_e).__name__}
    print('DEC_FALLBACK ' + str(_DEC_RECEIPT), flush=True)
    print('J_UNAVAILABLE ' + str(_DEC_RECEIPT), flush=True)
del _dec_main
