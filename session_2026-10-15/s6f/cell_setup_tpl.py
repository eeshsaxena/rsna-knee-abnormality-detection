# [eeshsaxena V224-FLIP-S6F 1/3 SETUP; not in any public notebook] PREREG_10-15 Amendment 3 R4. Every cell above is the submitted
# V224-FLIP notebook (eeshsaxena/rsna-sub-v224flip v1) unchanged and in its order (r384 -> R7 with the flip forced -> r224 ->
# stage 3 -> V224 canary), so submission.csv is complete here. The three S6F cells are APPENDED and only add:
#   1/3 SETUP  freeze the V224-FLIP output (submission.csv -> submission_v224flip.csv, byte copy + sha256); write the S6F driver
#              (the B path of nartaa/rsna-knee-s6fall-solo, sha256-checked) to /tmp/s6f; install the three PyPI-identical cp313
#              JPEG decoder wheels (glorysing/rsna-knee-offline-codecs-r09) into ITS OWN dir /tmp/s6f_dec after a self-test (the
#              reviewed V224-J DEC logic; only the S6F subprocess gets it); build the S6F subprocess environment = this kernel's
#              environment (its site-packages: stock image + R7's pinned timm / pydicom, which V224's cell 6 pip-installed
#              globally) with PYTHONPATH = /tmp/s6f_dec (+ the image's own PYTHONPATH), never an R7 code dir;
#   2/3 RUN    start guard G + hard deadline + process-group kill around ONE driver subprocess;
#   3/3 BLEND  per label .8 rank(V224-FLIP) + .2 rank(S6fall), else the S6F canary (ACL = 0.5 for every row).
# This cell never raises and changes nothing the cells above use (os.environ, sys.path and every V224 file stay as they are).
def _s6f_setup():
    import os, sys, json, time, glob, shutil, base64, hashlib, subprocess
    W, ROOT, DEC, INPUT, SLUG = '/kaggle/working', '/tmp/s6f', '/tmp/s6f_dec', '/kaggle/input', 'rsna-knee-offline-codecs-r09'
    R7_ROOTS = ('/tmp/probe', '/kaggle/temp/probe')
    DRIVER_SHA = '__DRIVER_SHA__'
    DRIVER_B64 = '__DRIVER_B64__'
    WHEELS = __WHEELS__
    PINS = ['pylibjpeg==2.1.0', 'pylibjpeg-libjpeg==2.4.0', 'pylibjpeg-openjpeg==2.6.0']
    VERSIONS = {'pylibjpeg': '2.1.0', 'libjpeg': '2.4.0', 'openjpeg': '2.6.0'}
    TOP = __TOP__
    FIX = __FIX__
    TEST = r'''__DEC_TEST__'''
    sha = lambda b: hashlib.sha256(b).hexdigest()
    rec = {'tag': '__TAG__', 'ok': False, 'dec_ok': False, 'age_h': None}
    env = None
    t0 = time.time()
    try:
        rec['age_h'] = round((t0 - NB_T0) / 3600.0, 4)
    except Exception as e:
        rec['age_h_error'] = type(e).__name__
    try:
        # 1. the V224-FLIP output, frozen before anything of S6F runs
        raw = open(W + '/submission.csv', 'rb').read()
        with open(W + '/submission_v224flip.csv', 'wb') as f:
            f.write(raw)
        if sha(open(W + '/submission_v224flip.csv', 'rb').read()) != sha(raw):
            raise OSError('frozen copy differs from submission.csv')
        rec['base_sha256'] = sha(raw)
        rec['base_bytes'] = len(raw)
        rec['v224_canary'] = dict(globals().get('_canary') or {})
        # 2. the driver
        drv = base64.b64decode(DRIVER_B64)
        if sha(drv) != DRIVER_SHA:
            raise ValueError('embedded S6F driver differs from its build sha')
        shutil.rmtree(ROOT, ignore_errors=True)
        os.makedirs(ROOT + '/work')
        with open(ROOT + '/s6f_driver.py', 'wb') as f:
            f.write(drv)
        if sha(open(ROOT + '/s6f_driver.py', 'rb').read()) != DRIVER_SHA:
            raise OSError('written S6F driver differs')
        rec['driver_sha256'] = DRIVER_SHA
        # 3. the S6F environment (no R7 dir; the image's PYTHONPATH kept after /tmp/s6f_dec)
        env = dict(os.environ)
        stock_pp = [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]
        if any(p.startswith(R7_ROOTS) for p in stock_pp):
            raise ValueError('an R7 dir is on the kernel PYTHONPATH: %s' % stock_pp)
        for k in ('PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONEXECUTABLE'):
            env.pop(k, None)
        if env.get('MPLBACKEND', '').startswith('module://'):      # ipykernel's inline backend: a plain process cannot use it
            env['MPLBACKEND'] = 'Agg'
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['HF_HUB_OFFLINE'] = '1'
        if stock_pp:
            env['PYTHONPATH'] = os.pathsep.join(stock_pp)
        else:
            env.pop('PYTHONPATH', None)
        rec['pythonpath_stock'] = stock_pp
        rec['mplbackend'] = env.get('MPLBACKEND')
        rec['ok'] = True
    except Exception as e:
        rec['why'] = '%s: %s' % (type(e).__name__, str(e)[:600])
        env = None
    # 4. decoders for the S6F subprocess only (the reviewed V224-J DEC cell logic, own target dir, no os.environ change)
    if env is not None:
        after = os.pathsep.join([DEC] + rec['pythonpath_stock'])
        dr = {'target': DEC}
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
                        if name not in found and os.path.isfile(p) and sha(open(p, 'rb').read()) == digest:
                            found[name] = p
            dr['dataset_dirs'] = dirs
            dr['wheels'] = found
            if sorted(found) != sorted(WHEELS):
                raise ValueError('PyPI-identical wheels not found: %s' % sorted(set(WHEELS) - set(found)))
            wdir = DEC + '_wheels'
            for d in (DEC, wdir):
                shutil.rmtree(d, ignore_errors=True)
            os.makedirs(wdir)
            for name, p in found.items():
                shutil.copyfile(p, os.path.join(wdir, name))
                if sha(open(os.path.join(wdir, name), 'rb').read()) != WHEELS[name]:
                    raise ValueError('copy of %s differs' % name)
            r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', '--target', DEC,
                                '--find-links', wdir] + PINS, capture_output=True, text=True, timeout=900, env=env)
            dr['pip_rc'] = r.returncode
            dr['pip_out'] = (r.stdout + r.stderr)[-800:]
            if r.returncode != 0:
                raise ValueError('pip exit %d' % r.returncode)
            top = sorted(os.listdir(DEC))
            dr['target_files'] = top
            if top != sorted(TOP):
                raise ValueError('target dir content differs: %s' % top)
            fxp = DEC + '_selftest.json'
            with open(fxp, 'w') as f:
                json.dump(FIX, f)
            t = subprocess.run([sys.executable, '-c', TEST, fxp], env=dict(env, PYTHONPATH=after), cwd=wdir,
                               capture_output=True, text=True, timeout=600)
            line = next((ln for ln in t.stdout.splitlines() if ln.startswith('DEC_SELFTEST ')), None)
            if t.returncode != 0 or line is None:
                raise ValueError('self-test exit %s: %s' % (t.returncode, (t.stderr or '')[-400:]))
            st = json.loads(line[len('DEC_SELFTEST '):])
            dr['selftest'] = st
            if st['versions'] != VERSIONS:
                raise ValueError('versions %s' % st['versions'])
            if any(v != DEC for v in st['dirs'].values()):
                raise ValueError('decoders not imported from %s: %s' % (DEC, st['dirs']))
            if any('pylibjpeg' not in v for v in st['plugins'].values()):
                raise ValueError('pydicom does not list pylibjpeg for JPEG Lossless: %s' % st['plugins'])
            if sorted(st['decode']) != sorted(FIX) or not all(v.get('exact') is True for v in st['decode'].values()):
                raise ValueError('self-test decode not exact: %s' % st['decode'])
            env['PYTHONPATH'] = after                                # the S6F subprocess environment only
            rec['dec_ok'] = True
        except Exception as e:
            dr['why'] = '%s: %s' % (type(e).__name__, str(e)[:600])
        rec['dec'] = dr
        rec['pythonpath_s6f'] = env.get('PYTHONPATH')
        short = {k: dr.get(k) for k in ('why', 'pip_rc')}
        if rec['dec_ok']:
            short.update(pydicom=dr['selftest']['pydicom'], versions=dr['selftest']['versions'],
                         decode={k: v.get('exact') for k, v in dr['selftest']['decode'].items()})
        print(('S6F_DEC_OK ' if rec['dec_ok'] else 'S6F_DEC_FALLBACK ') + json.dumps(short), flush=True)
    rec['seconds'] = round(time.time() - t0, 2)
    try:
        with open(W + '/s6f_setup_receipt.json', 'w') as f:
            json.dump(rec, f, indent=1)
    except Exception as e:
        rec['receipt_error'] = '%s: %s' % (type(e).__name__, str(e)[:200])
    print(('S6F_SETUP_OK ' if rec['ok'] else 'S6F_SETUP_FAILED ') + json.dumps(
        {k: rec.get(k) for k in ('tag', 'ok', 'why', 'dec_ok', 'age_h', 'base_sha256', 'driver_sha256', 'pythonpath_s6f', 'seconds')}),
        flush=True)
    return rec, env


try:
    _S6F_SETUP, _S6F_ENV = _s6f_setup()
except Exception as _e:                                               # belt and braces: the function catches everything itself
    _S6F_SETUP, _S6F_ENV = {'ok': False, 'why': 'setup cell: %s' % type(_e).__name__}, None
    print('S6F_SETUP_FAILED ' + str(_S6F_SETUP), flush=True)
del _s6f_setup
