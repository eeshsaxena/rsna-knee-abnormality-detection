# rsna-v224f20j-check: private CPU kernel (eeshsaxena; built 2026-10-14 by a Claude subagent with ../build_jj.py). NEVER submitted.
# No GPU, no internet, no model scoring; same docker pin as V224-F20 / V224-F20-J (gcr.io/kaggle-private-byod/python@sha256:2757e0c7...).
# Question: does the V224-F20-J DEC cell (taken VERBATIM from the built notebook rsna_sub_v224f20j.ipynb, sha-asserted) make every
# decoding stage read JPEG Lossless, while every stage output on UNCOMPRESSED data stays byte-identical to V224-F20?
# Method:
#   1. exec the DEC cell in this process, exactly as the notebook does (it sets os.environ['PYTHONPATH'] = /tmp/_dec + old)
#      -> PP0 = PYTHONPATH before (= V224-F20), PPJ = after (= V224-F20-J). Negative tests of the cell in subprocesses.
#   2. fixtures (as rsna-jpeg-check v2): one real public-test study transcoded into U / implicit / JPEG Lossless .70 / .57 /
#      J2K lossless / J2K lossy / RLE / mixed (fake root); "real" root = the 3 public-test studies + 6 train studies, original
#      uncompressed files copied unchanged.
#   3. every stage reader in the environment that stage has in the notebook, with PP0 (OFF) and PPJ (ON):
#        Raptor r384 (stock image): S / SJ          fleet as in HFR7-RR (stock image): S / SJ
#        R7's own pip install (cell 6 command, run under the J environment as in the J notebook), then
#        Raptor r224 + our V224-F20 fleet: R / RJ   R7 worker (cell 7 env: CODE + PYTHONPATH): R7 / R7J
#        plain pydicom (+ the prvsiyan synthetic DICOM fixtures, all uncompressed) in S / SJ / R / RJ
#   4. checks: ON == OFF byte for byte on every uncompressed input (real studies, U fixture, synthetic fixtures); ON recovers JPEG
#      Lossless exactly (== U); lossless syntaxes == U; J2K lossy ON == OFF. Prints V224F20J_CPU PASS / FAIL n/m.
import os, sys, json, glob, time, subprocess, hashlib, shutil, base64, zlib, zipfile, random, traceback, io, contextlib
from pathlib import Path

T0 = time.time()


def log(*a):
    print('[jj %7.1fs]' % (time.time() - T0), *a, flush=True)


WORK = Path('/kaggle/working')
JJ = Path('/tmp/jj')
MODS = JJ / 'mods'
for _p in (JJ, MODS):
    _p.mkdir(parents=True, exist_ok=True)

DEC_SRC = @@DEC_SRC@@
DEC_SHA = @@DEC_SHA@@
NB_SHA = @@NB_SHA@@
JLL_SRC = @@JLL_SRC@@
PROBE_SRC = @@PROBE_SRC@@
FASTREAD_B64 = @@FASTREAD_B64@@
FASTREAD_SHA = @@FASTREAD_SHA@@
R7_SRC_Z = @@R7_SRC_Z@@
R7_SOURCE_SHA = @@R7_SOURCE_SHA@@
R7_CODE_SHA = @@R7_CODE_SHA@@
R7_BUNDLE_ID = '8c7b666e27dc0e8f'
DEC_LINE = "    INPUT, TARGET, WORK, SLUG = '/kaggle/input', '/tmp/_dec', '/kaggle/working', 'rsna-knee-offline-codecs-r09'"
N_TRAIN = 6
VARIANTS = ['U', 'IMPL', 'JLL70', 'JLL57', 'J2KR', 'J2KI', 'RLE', 'MIX']
LOSSLESS = ['IMPL', 'JLL70', 'JLL57', 'J2KR', 'RLE', 'MIX']
RESULT = {'kernel': 'eeshsaxena/rsna-v224f20j-check', 'notebook_sha256': NB_SHA, 'dec_cell_sha256': DEC_SHA,
          'started_utc': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}


def save():
    try:
        json.dump(RESULT, open(WORK / 'v224f20j_cpu_result.json', 'w'), indent=1, default=str)
    except Exception as e:
        log('save failed', e)


def step(name, fn):
    t = time.time()
    try:
        RESULT[name] = fn()
        log('STEP_OK', name, '%.1fs' % (time.time() - t))
    except Exception as e:
        RESULT[name] = {'step_error': '%s: %s' % (type(e).__name__, e), 'tb': traceback.format_exc()[-3000:]}
        log('STEP_FAIL', name, type(e).__name__, str(e)[:300])
    save()
    return RESULT[name]


def find_ds(slug, required):
    inp = Path('/kaggle/input')
    cands = [inp / slug, inp / 'competitions' / slug] + sorted(inp.glob('datasets/*/' + slug)) + sorted(inp.glob('*/' + slug))
    for p in cands:
        if (p / required).exists():
            return p
    return None


def inputs():
    d = {'kaggle_input_top': sorted(os.listdir('/kaggle/input')), 'cpu_count': os.cpu_count(), 'python': sys.version,
         'PYTHONPATH': os.environ.get('PYTHONPATH', '')}
    found = {'COMP': find_ds('rsna-knee-abnormality-detection', 'test.csv'),
             'CODE_SRC': find_ds('rsna-orthofoundation-fold0-epoch7-20261004', 'receipt.json'),
             'M_SRC': find_ds('rsna-knee-compact-r7cnxfsss', 'manifest.json'),
             'FLEET': find_ds('rsna-knee-cnxt-fleet-k11w', 'fleet_common.py'),
             'FIX': find_ds('rsna-knee-synthetic-dicom-fixtures', 'corpus/manifest.csv')}
    d.update({k: (str(v) if v else None) for k, v in found.items()})
    log('INPUTS', json.dumps(d, default=str))
    return d


def write_mods(inp):
    assert hashlib.sha256(DEC_SRC.encode('utf-8')).hexdigest() == DEC_SHA, 'DEC cell text differs from the notebook'
    assert DEC_SRC.count(DEC_LINE) == 1
    (MODS / 'jll.py').write_text(JLL_SRC)
    (MODS / 'probe_jj.py').write_text(PROBE_SRC)
    fr = base64.b64decode(FASTREAD_B64)
    assert hashlib.sha256(fr).hexdigest() == FASTREAD_SHA, 'fastread differs from the notebook'
    (MODS / 'raptor_fastread.py').write_bytes(fr)
    src = json.loads(zlib.decompress(base64.b64decode(R7_SRC_Z)).decode('utf-8'))
    r7 = JJ / 'r7src'
    r7.mkdir(exist_ok=True)
    for name, text in src.items():
        (r7 / name).write_bytes(text.encode('utf-8'))
        assert hashlib.sha256((r7 / name).read_bytes()).hexdigest() == R7_SOURCE_SHA[name], name
    fl = JJ / 'fleet'
    fl.mkdir(exist_ok=True)
    shutil.copy(Path(inp['FLEET']) / 'fleet_common.py', fl / 'fleet_common.py')
    return {'fastread_sha': FASTREAD_SHA[:16], 'r7_sources': sorted(src),
            'fleet_common_sha': hashlib.sha256((fl / 'fleet_common.py').read_bytes()).hexdigest()[:16]}


# ---------------------------------------------------------------------------------------------------------------------
# 1. the DEC cell, verbatim, in this process (as in the notebook) + negative tests in subprocesses
# ---------------------------------------------------------------------------------------------------------------------
PP = {}


def dec_main():
    PP['0'] = os.environ.get('PYTHONPATH', '')
    g = {'__name__': '__main__'}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(DEC_SRC, 'v224f20j_dec_cell', 'exec'), g)
    out = buf.getvalue()
    print(out, end='', flush=True)
    PP['J'] = os.environ.get('PYTHONPATH', '')
    rec = g.get('_DEC_RECEIPT') or {}
    r = {'printed': out.strip()[:3000], 'receipt': rec, 'pp0': PP['0'], 'ppj': PP['J'],
         'ok_line': out.startswith('DEC_OK '), 'names_left': sorted(k for k in g if not k.startswith('__'))}
    log('DEC_MAIN', json.dumps({k: r[k] for k in ('ok_line', 'pp0', 'ppj', 'names_left')}))
    return r


def _run_cell(code, env_pp):
    probe = code + "\nimport os as _o, json as _j\nprint('NEG_PP ' + _j.dumps(_o.environ.get('PYTHONPATH', '')), flush=True)\n"
    r = subprocess.run([sys.executable, '-c', probe], env=dict(os.environ, PYTHONPATH=env_pp), capture_output=True, text=True,
                       timeout=900, cwd=str(JJ))
    pp = next((json.loads(ln[7:]) for ln in r.stdout.splitlines() if ln.startswith('NEG_PP ')), None)
    return {'rc': r.returncode, 'dec_ok': 'DEC_OK ' in r.stdout, 'dec_fallback': 'DEC_FALLBACK ' in r.stdout, 'pp_after': pp,
            'stdout': r.stdout[-1200:], 'stderr': r.stderr[-800:]}


def dec_negatives():
    wheels = eval(DEC_SRC.split('    WHEELS = ', 1)[1].split('\n', 1)[0])
    real_dirs = (RESULT.get('dec_main') or {}).get('receipt', {}).get('wheels') or {}
    out = {}

    def cell(inp, target, work, mutate=None):
        c = DEC_SRC.replace(DEC_LINE, "    INPUT, TARGET, WORK, SLUG = %r, %r, %r, 'rsna-knee-offline-codecs-r09'" % (inp, target, work))
        return mutate(c) if mutate else c
    base = JJ / 'neg'
    shutil.rmtree(base, ignore_errors=True)
    for k in ('missing', 'tampered', 'selftest', 'positive'):
        (base / k / 'in').mkdir(parents=True)
        (base / k / 'work').mkdir(parents=True)
    # tampered: the three real wheels copied, one byte of the libjpeg wheel flipped
    d = base / 'tampered' / 'in' / 'rsna-knee-offline-codecs-r09'
    d.mkdir()
    for n, p in real_dirs.items():
        shutil.copy(p, d / n)
    n_lj = [n for n in wheels if n.startswith('pylibjpeg_libjpeg')][0]
    b = bytearray((d / n_lj).read_bytes()); b[len(b) // 2] ^= 0xFF; (d / n_lj).write_bytes(bytes(b))
    out['tampered'] = _run_cell(cell(str(base / 'tampered' / 'in'), '/tmp/_dec_neg_t', str(base / 'tampered' / 'work')), PP['0'])
    out['missing'] = _run_cell(cell(str(base / 'missing' / 'in'), '/tmp/_dec_neg_m', str(base / 'missing' / 'work')), PP['0'])
    fx = eval(DEC_SRC.split('    FIX = ', 1)[1].split('\n', 1)[0])
    dig = fx['JLL70_16S'][3]
    out['selftest'] = _run_cell(cell('/kaggle/input', '/tmp/_dec_neg_s', str(base / 'selftest' / 'work'),
                                     mutate=lambda c: c.replace(dig, '0' * 64, 1)), PP['0'])
    out['positive'] = _run_cell(cell('/kaggle/input', '/tmp/_dec_pos', str(base / 'positive' / 'work')), PP['0'])
    log('DEC_NEG', json.dumps({k: {kk: v[kk] for kk in ('rc', 'dec_ok', 'dec_fallback', 'pp_after')} for k, v in out.items()}))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# 2. roots
# ---------------------------------------------------------------------------------------------------------------------
def real_root(comp):
    import pandas as pd
    comp = Path(comp)
    root = JJ / 'real'
    shutil.rmtree(root, ignore_errors=True)
    (root / 'test_series').mkdir(parents=True)
    test = pd.read_csv(comp / 'test.csv', dtype={'StudyInstanceUID': str})['StudyInstanceUID'].tolist()
    tser = pd.read_csv(comp / 'test_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
    trs = pd.read_csv(comp / 'train_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
    full = [s for s, g in trs.groupby('StudyInstanceUID') if {'Sagittal', 'Coronal', 'Axial'} <= set(g['Anatomical_Plane'].astype(str))]
    train = sorted(random.Random(1015).sample(sorted(full), N_TRAIN))
    for sid in test:
        shutil.copytree(comp / 'test_series' / sid, root / 'test_series' / sid)
    for sid in train:
        shutil.copytree(comp / 'train_series' / sid, root / 'test_series' / sid)
    ids = test + train
    rows = pd.concat([tser[tser['StudyInstanceUID'].isin(test)], trs[trs['StudyInstanceUID'].isin(train)]], ignore_index=True)
    rows = rows[list(tser.columns)]
    pd.DataFrame({'StudyInstanceUID': ids}).to_csv(root / 'test.csv', index=False)
    rows.to_csv(root / 'test_series.csv', index=False)
    cols = list(pd.read_csv(comp / 'sample_submission.csv', nrows=1).columns)
    df = pd.DataFrame({c: 0.5 for c in cols[1:]}, index=range(len(ids)))
    df.insert(0, cols[0], ids)
    df.to_csv(root / 'sample_submission.csv', index=False)
    nfiles = len(glob.glob(str(root / 'test_series' / '*' / '*' / '*.dcm')))
    out = {'root': str(root), 'test': test, 'train': train, 'series': int(len(rows)), 'files': nfiles}
    log('REAL_ROOT', json.dumps(out))
    return out


def run_probe(args, env, timeout):
    t = time.time()
    try:
        rc = subprocess.run([sys.executable, '-u', str(MODS / 'probe_jj.py')] + args, env=env, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        rc = 'TIMEOUT'
    return {'rc': rc, 'seconds': round(time.time() - t, 1)}


# ---------------------------------------------------------------------------------------------------------------------
# 3. R7 install, exactly as the notebook's R7 cell (its pip subprocess inherits the J environment, as in the J notebook)
# ---------------------------------------------------------------------------------------------------------------------
def r7_install(inp):
    code_src, m_src = Path(inp['CODE_SRC']), Path(inp['M_SRC'])
    code = JJ / 'mainline_code'
    if (code_src / 'cloud_code.zip').is_file():
        with zipfile.ZipFile(code_src / 'cloud_code.zip') as z:
            z.extractall(code)
    else:
        shutil.copytree(code_src / 'cloud_code', code, dirs_exist_ok=True)
    for name, digest in R7_CODE_SHA.items():
        assert hashlib.sha256((code / 'training' / name).read_bytes()).hexdigest() == digest, ('mainline module differs', name)
    wd = JJ / 'r7wheels'
    wd.mkdir(exist_ok=True)
    if (code_src / 'dependencies.zip').is_file():
        with zipfile.ZipFile(code_src / 'dependencies.zip') as z:
            for n in z.namelist():
                if n.endswith('.whl'):
                    (wd / Path(n).name).write_bytes(z.read(n))
    else:
        for p in code_src.rglob('*.whl'):
            shutil.copy(p, wd / p.name)
    wheels = sorted(str(p) for p in wd.glob('*.whl'))
    assert wheels, 'no wheels found'
    pp_at_pip = os.environ.get('PYTHONPATH', '')
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', *wheels], capture_output=True, text=True)
    manifest = json.loads((m_src / 'manifest.json').read_text())
    assert manifest['bundle_id'] == R7_BUNDLE_ID, manifest['bundle_id']
    anchor = code_src / 'anatomy.pt'
    side = m_src / manifest['side_model']
    assert hashlib.sha256(anchor.read_bytes()).hexdigest() == manifest['anchor_weights_sha256'], 'anatomy.pt differs'
    assert hashlib.sha256(side.read_bytes()).hexdigest() == manifest['sha256'][manifest['side_model']], 'side model differs'
    out = {'code': str(code), 'wheels': [Path(w).name for w in wheels], 'pip_rc': r.returncode, 'pip_out': r.stdout[-1500:],
           'pip_err': r.stderr[-800:], 'pythonpath_at_pip': pp_at_pip, 'anchor': str(anchor), 'side': str(side),
           'success_line': 'Successfully installed pydicom-3.0.1 timm-1.0.25' in r.stdout}
    log('R7_INSTALL', json.dumps({k: out[k] for k in ('pip_rc', 'success_line', 'pythonpath_at_pip', 'wheels')}))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# 4. checks
# ---------------------------------------------------------------------------------------------------------------------
CHECKS = []


def chk(name, ok, info=''):
    CHECKS.append({'name': name, 'ok': bool(ok), 'info': info})
    print('  %s %s %s' % ('OK  ' if ok else 'FAIL', name, json.dumps(info, default=str)[:600] if info != '' else ''), flush=True)


def _key(reader, r):
    if not isinstance(r, dict) or 'sha' not in r:
        return ('ERR', (r or {}).get('error') or (r or {}).get('item_exc') if isinstance(r, dict) else repr(r))
    if reader == 'r7':
        return (r.get('sha'), r.get('windows'), r.get('item_error'))
    if reader == 'raptor':
        return (r.get('sha'), r.get('mask_sha'), r.get('mask_sum'))
    if reader == 'fleet':
        return (r.get('sha'), tuple(r.get('views') or []))
    return (r.get('sha'), r.get('ok'), r.get('files'))


def _read(reader, r):
    # a real (uncompressed) study counts only if the stage actually read it: no error, something decoded
    if not isinstance(r, dict) or 'sha' not in r or r.get('sha') is None:
        return False
    if reader == 'r7':
        return r.get('item_error') in (None, '') and (r.get('windows') or 0) > 0 and (r.get('native_series') or 0) > 0
    if reader == 'raptor':
        return (r.get('mask_sum') or 0) > 0
    if reader == 'fleet':
        return (r.get('views_present') or 0) >= 1
    return r.get('ok') == r.get('files') and (r.get('files') or 0) > 0


def verdict():
    ph = {k: RESULT.get('probe_' + k) or {} for k in ('S', 'SJ', 'R', 'RJ', 'R7', 'R7J')}
    dm = RESULT.get('dec_main') or {}
    rec = dm.get('receipt') or {}
    pp0, ppj = dm.get('pp0', ''), dm.get('ppj', '')
    chk('DEC cell (verbatim) in this process: DEC_OK, receipt ok', dm.get('ok_line') and rec.get('ok') is True,
        {k: rec.get(k) for k in ('ok', 'why', 'seconds', 'pip_rc')})
    chk('DEC cell: PYTHONPATH after == /tmp/_dec + pathsep + before (before non-empty)',
        bool(pp0) and ppj == '/tmp/_dec' + os.pathsep + pp0, {'before': pp0, 'after': ppj})
    chk('DEC cell leaves only _DEC_RECEIPT in the notebook namespace', dm.get('names_left') == ['_DEC_RECEIPT'], dm.get('names_left'))
    st = rec.get('selftest') or {}
    chk('DEC self-test: pylibjpeg 2.1.0 / libjpeg 2.4.0 / openjpeg 2.6.0 from /tmp/_dec, JPEG Lossless plugin, 3/3 exact',
        st.get('versions') == {'pylibjpeg': '2.1.0', 'libjpeg': '2.4.0', 'openjpeg': '2.6.0'} and
        all(v == '/tmp/_dec' for v in (st.get('dirs') or {'x': 0}).values()) and
        all('pylibjpeg' in v for v in (st.get('plugins') or {'x': []}).values()) and
        sorted((st.get('decode') or {})) == ['JLL57_12U', 'JLL70_16S', 'U_16S'] and
        all(v.get('exact') is True for v in (st.get('decode') or {}).values()), st)
    neg = RESULT.get('dec_negatives') or {}
    for k, want_ok, want_pp in (('missing', False, pp0), ('tampered', False, pp0), ('selftest', False, pp0),
                                ('positive', True, '/tmp/_dec_pos' + os.pathsep + pp0)):
        x = neg.get(k) or {}
        chk('DEC negative/positive test "%s": %s, PYTHONPATH %s, no exception' % (k, 'DEC_OK' if want_ok else 'DEC_FALLBACK',
                                                                               'set' if want_ok else 'unchanged'),
            x.get('rc') == 0 and x.get('dec_ok') is want_ok and x.get('dec_fallback') is (not want_ok) and x.get('pp_after') == want_pp,
            {kk: x.get(kk) for kk in ('rc', 'dec_ok', 'dec_fallback', 'pp_after')})
    r7i = RESULT.get('r7_install') or {}
    chk('R7 pip install (cell 6 command) under the J environment: rc 0 + "Successfully installed pydicom-3.0.1 timm-1.0.25"',
        r7i.get('pip_rc') == 0 and r7i.get('success_line') is True and r7i.get('pythonpath_at_pip') == ppj,
        {k: r7i.get(k) for k in ('pip_rc', 'success_line', 'pythonpath_at_pip')})
    fx = RESULT.get('fixtures') or {}
    enc = {v: ((fx.get('variants') or {}).get(v) or {}).get('encode') or {} for v in VARIANTS}
    chk('fixtures built: 8 variants, 0 encode errors, lossless variants exact',
        all(enc[v].get('errors') == 0 and enc[v].get('files') for v in VARIANTS) and
        all(enc[v].get('exact') == enc[v].get('files') for v in VARIANTS if v != 'J2KI'), {v: (enc[v].get('files'), enc[v].get('exact'), enc[v].get('errors')) for v in VARIANTS})
    # environments
    for k in ('S', 'SJ', 'R', 'RJ', 'R7', 'R7J'):
        e = (ph[k].get('env') or {})
        on = k.endswith('J')
        pyd = str(e.get('pydicom', '')).split(' @')[0]
        want_pyd = '3.0.2' if k in ('S', 'SJ') else '3.0.1'
        jl = (e.get('pydicom_decoders') or {}).get('1.2.840.10008.1.2.4.70')
        ok = pyd == want_pyd and ((str(e.get('pylibjpeg', '')).startswith('2.1.0 @ /tmp/_dec') and jl == ['pylibjpeg']) if on else
                                  (str(e.get('pylibjpeg', '')).startswith('NO') and jl == []))
        chk('phase %s environment: pydicom %s, JPEG Lossless decoder %s' % (k, want_pyd, 'pylibjpeg from /tmp/_dec' if on else 'none'),
            ok, {'pydicom': pyd, 'pylibjpeg': e.get('pylibjpeg'), '4.70': jl, 'PYTHONPATH': e.get('PYTHONPATH')})
    fake_ids = {v: ((fx.get('variants') or {}).get(v) or {}).get('study') for v in VARIANTS}
    real = RESULT.get('real_root') or {}
    real_ids = (real.get('test') or []) + (real.get('train') or [])
    stages = [('Raptor r384 (stock image)', 'S', 'SJ', 'raptor'), ('Raptor r224 (after R7 pip)', 'R', 'RJ', 'raptor'),
              ('R7 worker (CODE + PYTHONPATH)', 'R7', 'R7J', 'r7'), ('fleet in V224-F20 (after R7 pip)', 'R', 'RJ', 'fleet'),
              ('fleet in the stock image (HFR7-RR)', 'S', 'SJ', 'fleet'), ('plain pydicom, stock', 'S', 'SJ', 'plain'),
              ('plain pydicom, after R7 pip', 'R', 'RJ', 'plain')]
    table = ['| stage | OFF / ON | real uncompressed studies ON == OFF | U fixture ON == OFF | ' +
             ' | '.join('%s ON == U' % v for v in LOSSLESS) + ' | J2KI ON == OFF | OFF loses JPEG Lossless |', '|---' * (6 + len(LOSSLESS)) + '|']
    for name, off, on, rd in stages:
        F0, F1 = ph[off].get(rd + '_fake') or {}, ph[on].get(rd + '_fake') or {}
        R0, R1 = ph[off].get(rd + '_real') or {}, ph[on].get(rd + '_real') or {}
        same_real = [u for u in real_ids if _key(rd, R0.get(u)) == _key(rd, R1.get(u)) and _read(rd, R0.get(u)) and _read(rd, R1.get(u))]
        c_real = len(real_ids) == 9 and len(same_real) == len(real_ids)
        chk('%s: %d/%d real uncompressed studies byte-identical ON vs OFF' % (name, len(same_real), len(real_ids)), c_real,
            {u[-8:]: (_key(rd, R0.get(u)), _key(rd, R1.get(u))) for u in real_ids if u not in same_real})
        u0, u1 = _key(rd, F0.get(fake_ids['U'])), _key(rd, F1.get(fake_ids['U']))
        chk('%s: U fixture byte-identical ON vs OFF' % name, u0 == u1 and u0[0] != 'ERR', {'off': u0, 'on': u1})
        cells = []
        for v in LOSSLESS:
            kv = _key(rd, F1.get(fake_ids[v]))
            ok = kv == u1 and kv[0] != 'ERR'
            cells.append('OK' if ok else 'FAIL')
            chk('%s: %s decodes == U with the fix (ON)' % (name, v), ok, {'on': kv, 'U': u1} if not ok else '')
        j0, j1 = _key(rd, F0.get(fake_ids['J2KI'])), _key(rd, F1.get(fake_ids['J2KI']))
        chk('%s: J2K lossy output unchanged by the fix (ON == OFF)' % name, j0 == j1 and j0[0] != 'ERR', {'off': j0, 'on': j1})
        lost = all(_key(rd, F0.get(fake_ids[v])) != u0 for v in ('JLL70', 'JLL57', 'MIX'))
        chk('%s: OFF (= V224-F20) loses JPEG Lossless (the defect the fix removes)' % name, lost)
        table.append('| %s | %s / %s | %d/%d | %s | %s | %s | %s |' % (name, off, on, len(same_real), len(real_ids), 'OK' if u0 == u1 else 'FAIL',
                                                                  ' | '.join(cells), 'OK' if j0 == j1 else 'FAIL', 'yes' if lost else 'NO'))
    for off, on in (('S', 'SJ'), ('R', 'RJ')):
        s0, s1 = ph[off].get('synthetic') or {}, ph[on].get('synthetic') or {}
        same = [f for f in s0 if s0.get(f) == s1.get(f)]
        chk('synthetic DICOM fixtures (prvsiyan, uncompressed edge cases), plain pydicom %s vs %s: %d/%d identical (sha or error)' % (
            off, on, len(same), len(s0)), len(s0) >= 40 and len(same) == len(s0) and sorted(s0) == sorted(s1),
            {'decoded': sum('sha' in v for v in s1.values()), 'refused': sum('error' in v for v in s1.values())})
    for k in ph:
        bad = [s for s, v in ph[k].items() if (isinstance(v, dict) and 'step_error' in v) or s.startswith('error')]
        chk('phase %s: every probe step ran' % k, bool(ph[k]) and not bad, bad)
    n_ok = sum(c['ok'] for c in CHECKS)
    RESULT['checks'] = CHECKS
    RESULT['verdict'] = 'V224F20J_CPU %s %d/%d' % ('PASS' if n_ok == len(CHECKS) else 'FAIL', n_ok, len(CHECKS))
    md = '\n'.join(table)
    (WORK / 'v224f20j_cpu_table.md').write_text(md + '\n\n' + RESULT['verdict'] + '\n')
    print(md, flush=True)
    print(RESULT['verdict'], flush=True)
    return {'n': len(CHECKS), 'ok': n_ok}


# ---------------------------------------------------------------------------------------------------------------------
def main():
    inp = step('inputs', inputs)
    if not inp.get('COMP'):
        log('no competition data'); return
    step('mods', lambda: write_mods(inp))
    step('dec_main', dec_main)
    if 'J' not in PP:
        PP['J'] = os.environ.get('PYTHONPATH', '')
    step('dec_negatives', dec_negatives)
    env_off = dict(os.environ, PYTHONPATH=PP['0'], PYTHONDONTWRITEBYTECODE='1')
    env_on = dict(os.environ, PYTHONPATH=PP['J'], PYTHONDONTWRITEBYTECODE='1')
    fake = JJ / 'fake'
    common = ['--mods', str(MODS), '--tmp', str(JJ)]
    step('make', lambda: run_probe(['make', '--comp', inp['COMP'], '--out', str(fake)] + common, env_on, 3600))
    try:
        RESULT['fixtures'] = json.load(open(fake / 'fixtures.json'))
    except Exception as e:
        RESULT['fixtures'] = {'error': str(e)}
    real = step('real_root', lambda: real_root(inp['COMP']))
    pargs = ['--fake', str(fake), '--real', str(JJ / 'real'), '--fastread', str(MODS / 'raptor_fastread.py'),
             '--fleet', str(JJ / 'fleet')] + common
    if inp.get('FIX'):
        pargs += ['--synth', inp['FIX']]

    def phase(ph, readers, env, extra=(), roots=('fake,real',)):
        # one probe process per roots group (R7: fake and real in separate processes, so no module state is shared)
        merged = {}
        for rg in roots:
            tag = ph + ('' if rg == 'fake,real' else '_' + rg)
            res = str(JJ / ('res_%s.json' % tag))
            step('run_' + tag, lambda: run_probe(['probe', '--phase', ph, '--readers', readers, '--roots', rg, '--result', res]
                                                 + list(extra) + pargs, env, 5400))
            try:
                part = json.load(open(res))
            except Exception as e:
                part = {'error_' + rg: str(e)}
            sec = dict(merged.get('seconds') or {}, **(part.get('seconds') or {}))
            merged.update(part)
            merged['seconds'] = sec
        RESULT['probe_' + ph] = merged
        save()
    phase('S', 'plain,raptor,fleet', env_off)
    phase('SJ', 'plain,raptor,fleet', env_on)
    # the J notebook's R7 cell runs its pip install with the J environment (os.environ after the DEC cell)
    os.environ['PYTHONPATH'] = PP['J']
    r7 = step('r7_install', lambda: r7_install(inp))
    phase('R', 'plain,raptor,fleet', env_off)
    phase('RJ', 'plain,raptor,fleet', env_on)
    if 'step_error' not in r7:
        extra = ['--r7src', str(JJ / 'r7src'), '--code', r7['code'], '--anchor', r7['anchor'], '--side', r7['side']]
        # notebook R7 cell: env = dict(os.environ, PYTHONPATH=str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', ''), ...)
        phase('R7', 'r7', dict(os.environ, PYTHONPATH=r7['code'] + os.pathsep + PP['0'], PYTHONDONTWRITEBYTECODE='1'), extra, ('fake', 'real'))
        phase('R7J', 'r7', dict(os.environ, PYTHONPATH=r7['code'] + os.pathsep + PP['J'], PYTHONDONTWRITEBYTECODE='1'), extra, ('fake', 'real'))
    step('verdict', verdict)
    RESULT['seconds'] = round(time.time() - T0, 1)
    save()
    log('V224F20J_CPU_DONE', RESULT['seconds'], RESULT.get('verdict'))


main()
