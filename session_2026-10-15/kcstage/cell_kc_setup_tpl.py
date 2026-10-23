# [eeshsaxena __TAG__ KC 1/3 SETUP; not in any public notebook] PREREG_10-15 Amendment 6 (KC1) G3. Every cell above is the
# __BASE_NAME__ notebook unchanged and in its order, so submission.csv is complete here (the S6F blend or its canary). The three KC
# cells are APPENDED and only add:
#   1/3 SETUP  freeze the current output (submission.csv -> submission_s6f.csv, byte copy + sha256); find the private KC dataset
#              (__KC_SLUG__) by its manifest sha256, check every file of the manifest (members + code) by sha256 and the frozen
#              KC code by its A6 md5 / sha pins; copy the code to /tmp/kc/code; write the KC driver and the per-GPU serve wrapper
#              (sha256-checked); build the KC subprocess environment = this kernel's environment + PYTHONPATH /tmp/kc/code, then the
#              JPEG decoders already installed by the cells above if present (/tmp/s6f_dec when S6F's DEC passed; /tmp/_dec stays
#              on the inherited PYTHONPATH in a J kernel), never an R7 or S6F code dir;
#   2/3 RUN    start guard G_KC + hard deadline + process-group kill around ONE driver subprocess (side pass, one serve process
#              per T4, fp16, both-canon mean for weak-zone / not-GEOM_OK studies);
#   3/3 BLEND  per label .8 rank(base) + .2 rank(KC leg), else the KC canary (Baker's = 0.5 for every row).
# This cell never raises and changes nothing the cells above use (os.environ, sys.path and every earlier output stay as they are).
def _kc_setup():
    import os, sys, json, time, glob, shutil, base64, hashlib
    W, ROOT, INPUT = '/kaggle/working', '/tmp/kc', '/kaggle/input'
    KC_SLUG = '__KC_SLUG__'
    MANIFEST_SHA = '__MANIFEST_SHA__'
    CODE_MD5 = {'kc_worker.py': '4af60f6ff7b94b3acab9dceb52162a18', 'kc_common.py': '45255ae0e361ef787b290d7510554a9d',
                'fleet_model.py': '897f83b272fbb144a4d3deab773d9064', 'fleet_model_ttr.py': 'bcdf7d153f339962b97d2736d8baed08'}
    CODE_SHA = {'d96_reader_src.py': '30aa72fd972e29cf42c752aa5394a6c88a175f17b98740b1f6b0e0f55553ac40',
                'kc_side_kernel.py': '2979c1915e0cecf2c4792526a5581f93a723fac368eacff18849661dc0bdcb62'}
    DRIVER_SHA, SERVE_SHA = '__DRIVER_SHA__', '__SERVE_SHA__'
    DRIVER_B64 = '__DRIVER_B64__'
    SERVE_B64 = '__SERVE_B64__'
    R7_ROOTS = ('/tmp/probe', '/kaggle/temp/probe', '/tmp/s6f/', '/kaggle/working')
    rec = {'tag': '__TAG__', 'ok': False, 'age_h': None, 'kc_slug': KC_SLUG, 'manifest_sha256': MANIFEST_SHA}
    env = None
    t0 = time.time()

    def sha256(p):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(1 << 22), b''):
                h.update(b)
        return h.hexdigest()

    try:
        rec['age_h'] = round((t0 - NB_T0) / 3600.0, 4)
    except Exception as e:
        rec['age_h_error'] = type(e).__name__
    try:
        # 1. the current output (the S6F blend or the S6F canary, over V224-FLIP), frozen before anything of KC runs
        raw = open(W + '/submission.csv', 'rb').read()
        with open(W + '/submission_s6f.csv', 'wb') as f:
            f.write(raw)
        if hashlib.sha256(open(W + '/submission_s6f.csv', 'rb').read()).hexdigest() != hashlib.sha256(raw).hexdigest():
            raise OSError('frozen copy differs from submission.csv')
        rec['base_sha256'] = hashlib.sha256(raw).hexdigest()
        rec['base_bytes'] = len(raw)
        rec['v224_canary'] = dict(globals().get('_canary') or {})
        s6 = globals().get('_S6F_RECEIPT') or {}
        rec['s6f'] = {'mode': s6.get('mode'), 'fired': list(s6.get('fired') or []),
                      'dec_ok': bool((globals().get('_S6F_SETUP') or {}).get('dec_ok'))}
        # 2. the KC dataset: the dir whose KC_MANIFEST.json has the build's sha256
        name = KC_SLUG.split('/')[-1]
        cands = []
        for pat in (INPUT + '/' + name, INPUT + '/*/' + name, INPUT + '/datasets/*/' + name):
            for d in sorted(glob.glob(pat)):
                for dd in [d] + sorted(glob.glob(d + '/*')):
                    if os.path.isfile(dd + '/KC_MANIFEST.json') and dd not in cands:
                        cands.append(dd)
        kdir = next((d for d in cands if sha256(d + '/KC_MANIFEST.json') == MANIFEST_SHA), None)
        rec['dataset_candidates'] = cands
        if kdir is None:
            raise ValueError('KC dataset with manifest sha %s... not found under %s' % (MANIFEST_SHA[:16], INPUT))
        rec['dataset_dir'] = kdir
        man = json.load(open(kdir + '/KC_MANIFEST.json'))
        files = man['files']
        t_sha = time.time()
        got = {f: sha256(os.path.join(kdir, f)) for f in files}
        rec['sha_seconds'] = round(time.time() - t_sha, 2)
        bad = sorted(f for f in files if got[f] != files[f])
        if bad:
            raise ValueError('KC dataset files differ from the manifest: %s' % bad)
        for f, h in CODE_MD5.items():
            if hashlib.md5(open(os.path.join(kdir, f), 'rb').read()).hexdigest() != h:
                raise ValueError('frozen KC code %s md5 != %s' % (f, h))
        for f, h in CODE_SHA.items():
            if got.get(f) != h:
                raise ValueError('frozen KC code %s sha256 != %s' % (f, h))
        members = [m['file'] for m in man['members']]
        if not members or any(m not in files for m in members):
            raise ValueError('manifest members missing from files')
        rec['members'] = [os.path.join(kdir, m) for m in members]
        rec['member_sha256'] = {m: files[m] for m in members}
        rec['manifest'] = {k: man.get(k) for k in ('kind', 'standin', 'contract_hash', 'members', 'created')}
        # 3. code -> /tmp/kc/code; driver + serve wrapper -> /tmp/kc
        shutil.rmtree(ROOT, ignore_errors=True)
        os.makedirs(ROOT + '/code')
        os.makedirs(ROOT + '/work')
        for f in list(CODE_MD5) + list(CODE_SHA):
            shutil.copyfile(os.path.join(kdir, f), os.path.join(ROOT, 'code', f))
            if sha256(os.path.join(ROOT, 'code', f)) != got[f]:
                raise OSError('copy of %s differs' % f)
        for nm, b64, h in (('kc_driver.py', DRIVER_B64, DRIVER_SHA), ('kc_serve.py', SERVE_B64, SERVE_SHA)):
            b = base64.b64decode(b64)
            if hashlib.sha256(b).hexdigest() != h:
                raise ValueError('embedded %s differs from its build sha' % nm)
            with open(os.path.join(ROOT, nm), 'wb') as f:
                f.write(b)
            if sha256(os.path.join(ROOT, nm)) != h:
                raise OSError('written %s differs' % nm)
        rec['driver_sha256'], rec['serve_sha256'] = DRIVER_SHA, SERVE_SHA
        # 4. the KC environment: inherited + /tmp/kc/code first, then decoders already installed above (if present)
        env = dict(os.environ)
        inh = [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]
        if any(p.startswith(R7_ROOTS) for p in inh):
            raise ValueError('an R7 / S6F / working dir is on the kernel PYTHONPATH: %s' % inh)
        for k in ('PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONEXECUTABLE'):
            env.pop(k, None)
        if env.get('MPLBACKEND', '').startswith('module://'):
            env['MPLBACKEND'] = 'Agg'
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['HF_HUB_OFFLINE'] = '1'
        dec = []
        if rec['s6f']['dec_ok'] and os.path.isdir('/tmp/s6f_dec') and '/tmp/s6f_dec' not in inh:
            dec.append('/tmp/s6f_dec')
        pp = [ROOT + '/code'] + dec + inh
        env['PYTHONPATH'] = os.pathsep.join(pp)
        rec['pythonpath_inherited'] = inh
        rec['pythonpath_kc'] = env['PYTHONPATH']
        rec['dec_dirs'] = [p for p in pp if p in ('/tmp/s6f_dec', '/tmp/_dec')]
        rec['dec_present'] = bool(rec['dec_dirs'])
        rec['ok'] = True
    except Exception as e:
        rec['why'] = '%s: %s' % (type(e).__name__, str(e)[:600])
        env = None
    rec['seconds'] = round(time.time() - t0, 2)
    try:
        with open(W + '/kc_setup_receipt.json', 'w') as f:
            json.dump(rec, f, indent=1)
    except Exception as e:
        rec['receipt_error'] = '%s: %s' % (type(e).__name__, str(e)[:200])
    print(('KC_SETUP_OK ' if rec['ok'] else 'KC_SETUP_FAILED ') + json.dumps(
        {k: rec.get(k) for k in ('tag', 'ok', 'why', 'age_h', 'base_sha256', 'dataset_dir', 'dec_dirs', 'pythonpath_kc', 'sha_seconds',
                                 'seconds')}), flush=True)
    return rec, env


try:
    _KC_SETUP, _KC_ENV = _kc_setup()
except Exception as _e:                                               # belt and braces: the function catches everything itself
    _KC_SETUP, _KC_ENV = {'ok': False, 'why': 'setup cell: %s' % type(_e).__name__}, None
    print('KC_SETUP_FAILED ' + str(_KC_SETUP), flush=True)
del _kc_setup
