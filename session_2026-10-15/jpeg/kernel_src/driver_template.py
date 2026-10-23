# rsna-jpeg-check: private CPU kernel (eeshsaxena; built 2026-10-14 by a Claude subagent with build_jc.py). NEVER submitted.
# No GPU, no internet, no model scoring; same docker pin as V224-F20 (gcr.io/kaggle-private-byod/python@sha256:2757e0c7...).
# Question: does any V224-F20 stage (Raptor r384 / r224 = the embedded _fastread 'study_lut' reader, R7 = compact_infer's
# mainline native decode, our fleet = fleet_common.build_study) silently drop DICOM slices / series / studies stored as
# JPEG Lossless, JPEG 2000 or RLE, with and without goodpjw2008's decoder wheels (pylibjpeg 2.1.0, libjpeg 2.4.0,
# openjpeg 2.6.0) on PYTHONPATH (the mkhamzanov/rsna-b1-ctrl pattern)?
# Method: one REAL public-test study is transcoded into each transfer syntax (fake competition root under /tmp/jc/fake, new
# UIDs), then every reader runs in 4 environments and is compared with the uncompressed copy:
#   S  = stock image (Raptor r384 runs here in V224; the HFR7-RR fleet ran here)
#   SW = S + decoder wheels on PYTHONPATH
#   R  = after R7's own `pip install --no-index --no-deps` of its wheels (pydicom 3.0.1, timm 1.0.25), as V224 cell 6 does
#        (r224 and the V224-F20 fleet run here); R7 = R + CODE on PYTHONPATH (the R7 worker's env, V224 cell 7)
#   RW / R7W = R / R7 + decoder wheels
# Also: transfer-syntax census of the competition data (every public-test file, first file of every train series, a random
# train file sample) and the CSV headers (any transfer-syntax hint). Outputs: jpeg_check_result.json, jpeg_check_table.md.
import os, sys, json, glob, time, subprocess, hashlib, shutil, base64, zlib, zipfile, collections, random, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

T0 = time.time()


def log(*a):
    print('[jc %7.1fs]' % (time.time() - T0), *a, flush=True)


WORK = Path(os.environ.get('JC_WORK', '/kaggle/working'))       # env overrides exist only for the local dry run
JC = Path(os.environ.get('JC_TMP', '/tmp/jc'))
INPUT = Path(os.environ.get('JC_INPUT', '/kaggle/input'))
MODS = JC / 'mods'
for _p in (JC, MODS):
    _p.mkdir(parents=True, exist_ok=True)

JLL_SRC = @@JLL_SRC@@
PROBE_SRC = @@PROBE_SRC@@
FASTREAD_B64 = @@FASTREAD_B64@@
FASTREAD_SHA = @@FASTREAD_SHA@@
R7_SRC_Z = @@R7_SRC_Z@@
R7_SOURCE_SHA = @@R7_SOURCE_SHA@@
R7_CODE_SHA = @@R7_CODE_SHA@@
R7_BUNDLE_ID = '8c7b666e27dc0e8f'
VARIANTS = ['U', 'IMPL', 'JLL70', 'JLL57', 'J2KR', 'J2KI', 'RLE', 'MIX']
VLABEL = {'U': 'uncompressed explicit LE 1.2.1 (control)', 'IMPL': 'implicit VR LE 1.2', 'JLL70': 'JPEG Lossless SV1 4.70',
          'JLL57': 'JPEG Lossless P14 4.57', 'J2KR': 'JPEG 2000 lossless 4.90', 'J2KI': 'JPEG 2000 lossy 4.91 (cr 20)',
          'RLE': 'RLE Lossless 1.2.5', 'MIX': 'mixed by series: #1/#3/#5 JPEG Lossless 4.70, #2/#4 1.2.1'}
RESULT = {'kernel': 'eeshsaxena/rsna-jpeg-check', 'started_utc': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}


def save():
    try:
        json.dump(RESULT, open(WORK / 'jpeg_check_result.json', 'w'), indent=1, default=str)
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
    inp = INPUT
    cands = [inp / slug, inp / 'competitions' / slug] + sorted(inp.glob('datasets/*/' + slug)) + sorted(inp.glob('*/' + slug))
    for p in cands:
        if (p / required).exists():
            return p
    return None


# ---------------------------------------------------------------------------------------------------------------------
# 0. inputs
# ---------------------------------------------------------------------------------------------------------------------
def inputs():
    d = {'kaggle_input_top': sorted(os.listdir(INPUT)),
         'datasets_dirs': sorted(str(p) for p in INPUT.glob('datasets/*/*')),
         'cpu_count': os.cpu_count(), 'python': sys.version, 'PYTHONPATH': os.environ.get('PYTHONPATH', '')}
    found = {'COMP': find_ds('rsna-knee-abnormality-detection', 'test.csv'),
             'WHEELS': find_ds('rsna-knee-2-5d-convnext-reader', 'pylibjpeg-2.1.0-py3-none-any.whl'),
             'WHEELS313': find_ds('rsna-knee-offline-codecs-r09', 'codec_manifest.json'),
             'CODE_SRC': find_ds('rsna-orthofoundation-fold0-epoch7-20261004', 'receipt.json'),
             'M_SRC': find_ds('rsna-knee-compact-r7cnxfsss', 'manifest.json'),
             'FLEET': find_ds('rsna-knee-cnxt-fleet-k11w', 'fleet_common.py'),
             'K14W': find_ds('rsna-knee-cnxt-fleet-k14w', 'k12w_manifest.json'),
             'FIX': find_ds('rsna-knee-synthetic-dicom-fixtures', 'corpus/manifest.csv'),
             'NARTAA': find_ds('rsna-knee-publication-swa-weights-20261007', 'raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt')}
    d.update({k: (str(v) if v else None) for k, v in found.items()})
    try:
        d['nvidia_smi'] = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=20).stdout[:300]
    except Exception as e:
        d['nvidia_smi'] = 'none (%s)' % type(e).__name__
    log('INPUTS', json.dumps(d, default=str))
    return d


# ---------------------------------------------------------------------------------------------------------------------
# 1. census of the competition data (stock pydicom, file meta only)
# ---------------------------------------------------------------------------------------------------------------------
def census(comp):
    import pandas as pd
    import pydicom
    from pydicom.filereader import read_file_meta_info
    comp = Path(comp)
    out = {'pydicom': pydicom.__version__, 'top': []}
    for e in sorted(os.scandir(comp), key=lambda e: e.name):
        if e.is_dir():
            try:
                n = len(os.listdir(e.path))
            except Exception:
                n = -1
            out['top'].append({'name': e.name, 'dir': True, 'children': n})
        else:
            out['top'].append({'name': e.name, 'dir': False, 'bytes': e.stat().st_size})
    csvs = {}
    for f in sorted(comp.glob('*.csv')):
        try:
            df = pd.read_csv(f, low_memory=False)
            hint = [c for c in df.columns if any(k in c.lower() for k in ('syntax', 'transfer', 'compress', 'jpeg', 'codec', 'encod'))]
            csvs[f.name] = {'rows': int(len(df)), 'columns': list(df.columns)[:40], 'ts_hint_columns': hint,
                            'head': df.head(2).astype(str).to_dict('records') if len(df.columns) <= 12 else None}
            if hint:
                csvs[f.name]['hint_values'] = {c: df[c].astype(str).value_counts().head(10).to_dict() for c in hint}
        except Exception as e:
            csvs[f.name] = {'error': str(e)[:200]}
    out['csvs'] = csvs
    log('CSV', json.dumps(csvs, default=str)[:4000])

    def ts_of(f):
        try:
            return str(read_file_meta_info(f).TransferSyntaxUID)
        except Exception as e:
            return 'ERR ' + type(e).__name__

    # every public-test file
    tdir = comp / 'test_series'
    tf = sorted(glob.glob(str(tdir / '*' / '*' / '*.dcm')))
    with ThreadPoolExecutor(32) as ex:
        tts = list(ex.map(ts_of, tf))
    out['test_files'] = len(tf)
    out['test_ts_per_file'] = dict(collections.Counter(tts))
    per_series = collections.defaultdict(set)
    for f, t in zip(tf, tts):
        per_series[os.path.dirname(f)].add(t)
    out['test_series'] = len(per_series)
    out['test_series_mixed_ts'] = sum(len(v) > 1 for v in per_series.values())
    log('TEST_CENSUS', json.dumps({k: out[k] for k in ('test_files', 'test_ts_per_file', 'test_series', 'test_series_mixed_ts')}))
    # train: first file of every series + a random per-file sample, under a time cap
    trd = None
    for nm in ('train_series', 'train_images', 'train'):
        if (comp / nm).is_dir():
            trd = comp / nm
            break
    out['train_dir'] = str(trd)
    if trd is not None:
        t_cap = time.time() + 9 * 60
        series_dirs = []
        for st in sorted(os.listdir(trd)):
            if time.time() > t_cap:
                break
            p = trd / st
            if p.is_dir():
                series_dirs += [p / se for se in sorted(os.listdir(p))]
        out['train_series_listed'] = len(series_dirs)

        def first_file(sd):
            try:
                fs = sorted(x for x in os.listdir(sd) if x.endswith('.dcm'))
                return (str(sd / fs[0]) if fs else None), [str(sd / x) for x in fs]
            except Exception:
                return None, []
        rng = random.Random(1014)
        with ThreadPoolExecutor(32) as ex:
            lst = list(ex.map(first_file, series_dirs))
        firsts = [a for a, _ in lst if a]
        allf = [x for _, b in lst for x in b]
        out['train_files_listed'] = len(allf)
        sample = rng.sample(allf, min(3000, len(allf)))
        with ThreadPoolExecutor(32) as ex:
            fts = list(ex.map(ts_of, firsts))
            sts = list(ex.map(ts_of, sample))
        out['train_first_file_per_series'] = {'n': len(firsts), 'ts': dict(collections.Counter(fts))}
        out['train_random_file_sample'] = {'n': len(sample), 'ts': dict(collections.Counter(sts))}
        log('TRAIN_CENSUS', json.dumps({k: out.get(k) for k in ('train_dir', 'train_series_listed', 'train_files_listed',
                                                                 'train_first_file_per_series', 'train_random_file_sample')}))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# 2. modules + decoder wheels
# ---------------------------------------------------------------------------------------------------------------------
def write_mods(inp):
    (MODS / 'jll.py').write_text(JLL_SRC)
    (MODS / 'probe_jc.py').write_text(PROBE_SRC)
    fr = base64.b64decode(FASTREAD_B64)
    assert hashlib.sha256(fr).hexdigest() == FASTREAD_SHA, 'fastread differs from V224 cells 3 / 9'
    (MODS / 'raptor_fastread.py').write_bytes(fr)
    src = json.loads(zlib.decompress(base64.b64decode(R7_SRC_Z)).decode('utf-8'))
    r7 = JC / 'r7src'
    r7.mkdir(exist_ok=True)
    for name, text in src.items():
        (r7 / name).write_bytes(text.encode('utf-8'))
        assert hashlib.sha256((r7 / name).read_bytes()).hexdigest() == R7_SOURCE_SHA[name], name
    fl = JC / 'fleet'
    fl.mkdir(exist_ok=True)
    shutil.copy(Path(inp['FLEET']) / 'fleet_common.py', fl / 'fleet_common.py')
    return {'fastread_sha': FASTREAD_SHA[:16], 'r7_sources': sorted(src), 'fleet_common_sha':
            hashlib.sha256((fl / 'fleet_common.py').read_bytes()).hexdigest()[:16]}


# sha256 of the official PyPI wheels (pypi.org JSON API, checked 2026-10-14): a wheel is installed only if it is byte-identical
PYPI_SHA = {
    'pylibjpeg-2.1.0-py3-none-any.whl': '25df9496a69e64e98c887fddee12a1271e275b5f74ba804f9bf98a08bb80993e',
    'pylibjpeg_libjpeg-2.4.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl':
        'd2c8da12c79d1f46596b89df94091be01f932d739613e8af93d1d557ac3bab74',
    'pylibjpeg_openjpeg-2.6.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl':
        '3e8ee9e5bd3b62766ca98fc143cd340633e60bb14450162426f32688908ad9e8'}


def _pip_target(src_dir, dec):
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', '--target', str(dec),
                        '--find-links', str(src_dir), 'pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg'],
                       capture_output=True, text=True)
    chk = subprocess.run([sys.executable, '-c', 'import pylibjpeg, libjpeg, openjpeg, pydicom; print(pydicom.__version__, '
                          'pylibjpeg.__version__, libjpeg.__version__, openjpeg.__version__)'],
                         capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(dec) + os.pathsep + os.environ.get('PYTHONPATH', '')))
    return {'src': str(src_dir), 'pip_rc': r.returncode, 'pip_err': r.stderr[-800:], 'check_rc': chk.returncode,
            'check': chk.stdout.strip(), 'check_err': chk.stderr[-600:], 'files': sorted(os.listdir(dec)) if dec.is_dir() else []}


def install_decoders(inp):
    out = {}
    # (a) goodpjw2008's wheels exactly as mkhamzanov/rsna-b1-ctrl installs them (cp311 / cp312 only), recorded only
    if inp.get('WHEELS'):
        out['goodpjw_cp312'] = _pip_target(inp['WHEELS'], JC / 'decenv_goodpjw')
    # (b) the same three versions as cp313 wheels (glorysing/rsna-knee-offline-codecs-r09), sha-checked against PyPI
    dec = JC / 'decenv'
    if inp.get('WHEELS313'):
        src = Path(inp['WHEELS313'])
        shas = {n: hashlib.sha256((src / n).read_bytes()).hexdigest() if (src / n).is_file() else None for n in PYPI_SHA}
        out['cp313_sha_match_pypi'] = {n: shas[n] == PYPI_SHA[n] for n in PYPI_SHA}
        if all(out['cp313_sha_match_pypi'].values()):
            loc = JC / 'wheels313'
            loc.mkdir(exist_ok=True)
            for n in PYPI_SHA:
                shutil.copy(src / n, loc / n)
            out['cp313'] = _pip_target(loc, dec)
    if not (dec / 'pylibjpeg').is_dir() and (JC / 'decenv_goodpjw' / 'pylibjpeg').is_dir():
        shutil.copytree(JC / 'decenv_goodpjw', dec, dirs_exist_ok=True)
    out['decenv'] = str(dec)
    out['decenv_files'] = sorted(os.listdir(dec)) if dec.is_dir() else []
    out['usable'] = bool((out.get('cp313') or {}).get('check_rc') == 0 or (out.get('goodpjw_cp312') or {}).get('check_rc') == 0)
    log('DECODERS', json.dumps(out)[:3000])
    return out


def env_for(phase, code=None):
    base = os.environ.get('PYTHONPATH', '')
    parts = []
    if phase in ('R7', 'R7W') and code:
        parts.append(str(code))                     # V224 cell 7: the R7 worker gets PYTHONPATH = CODE + existing
    if phase in ('SW', 'RW', 'R7W'):
        parts.append(str(JC / 'decenv'))            # b1-ctrl: PYTHONPATH = decoder target + existing
    if base:
        parts.append(base)
    return dict(os.environ, PYTHONPATH=os.pathsep.join(parts), PYTHONDONTWRITEBYTECODE='1')


def run_probe(args, env, timeout):
    t = time.time()
    try:
        rc = subprocess.run([sys.executable, '-u', str(MODS / 'probe_jc.py')] + args, env=env, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        rc = 'TIMEOUT'
    return {'rc': rc, 'seconds': round(time.time() - t, 1)}


# ---------------------------------------------------------------------------------------------------------------------
# 3. R7 install, exactly as V224 cell 6 (cloud_code copy + sha, wheels pip install into the image)
# ---------------------------------------------------------------------------------------------------------------------
def r7_install(inp):
    code_src, m_src = Path(inp['CODE_SRC']), Path(inp['M_SRC'])
    code = JC / 'mainline_code'
    if (code_src / 'cloud_code.zip').is_file():
        with zipfile.ZipFile(code_src / 'cloud_code.zip') as z:
            z.extractall(code)
    else:
        shutil.copytree(code_src / 'cloud_code', code, dirs_exist_ok=True)
    for name, digest in R7_CODE_SHA.items():
        assert hashlib.sha256((code / 'training' / name).read_bytes()).hexdigest() == digest, ('mainline module differs', name)
    wd = JC / 'r7wheels'
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
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', *wheels], capture_output=True, text=True)
    manifest = json.loads((m_src / 'manifest.json').read_text())
    assert manifest['bundle_id'] == R7_BUNDLE_ID, manifest['bundle_id']
    anchor = code_src / 'anatomy.pt'
    side = m_src / manifest['side_model']
    assert hashlib.sha256(anchor.read_bytes()).hexdigest() == manifest['anchor_weights_sha256'], 'anatomy.pt differs'
    assert hashlib.sha256(side.read_bytes()).hexdigest() == manifest['sha256'][manifest['side_model']], 'side model differs'
    out = {'code': str(code), 'wheels': [Path(w).name for w in wheels], 'pip_rc': r.returncode, 'pip_out': r.stdout[-1200:],
           'pip_err': r.stderr[-800:], 'anchor': str(anchor), 'side': str(side),
           'manifest': {k: manifest.get(k) for k in ('views', 'size', 'slices', 'phys_mm', 'eval_stride', 'side_threshold',
                                                     'side_max_x', 'side_position_mm', 'fluid_stacker')}}
    log('R7_INSTALL', json.dumps(out)[:2500])
    return out


# ---------------------------------------------------------------------------------------------------------------------
# 4. table
# ---------------------------------------------------------------------------------------------------------------------
def _plain(p, v):
    x = (p.get('plain') or {}).get(v)
    if not x:
        return 'n/a'
    s = '%d/%d decoded' % (x['ok'], x['files'])
    if x['ok'] and v != 'U':
        s += ', %d exact' % x['exact_vs_U']
    if x.get('first_error'):
        s += ' | ' + x['first_error'][:90]
    return s


def _raptor(p, v):
    x = (p.get('raptor') or {}).get(v)
    if not x:
        return 'n/a'
    if 'error' in x:
        return 'EXC ' + x['error'][:90]
    if x.get('identical'):
        return 'OK identical (%d/%d slices)' % (x['mask_sum'], x['U_mask_sum'])
    if x.get('lost_slices'):
        return 'SILENT DROP %d/%d slices -> zero-filled (mask %d)' % (x['lost_slices'], x['U_mask_sum'], x['mask_sum'])
    return 'differs: %d/%d slices identical, maxdiff %d' % (x.get('slices_identical', -1), x.get('slices_total', -1), x.get('maxdiff', -1))


def _fleet(p, v):
    x = (p.get('fleet') or {}).get(v)
    if not x:
        return 'n/a'
    if 'error' in x:
        return 'EXC ' + x['error'][:90]
    if x.get('identical'):
        return 'OK identical (%d views)' % x['views_present']
    if x['views_present'] == 0:
        return 'STUDY BLANK (0/%d views; row NaN -> median / coverage gate)' % x.get('U_views_present', -1)
    if x['views_present'] < x.get('U_views_present', 0):
        return 'SILENT DROP %d of %d views (series)' % (x['U_views_present'] - x['views_present'], x['U_views_present'])
    return 'differs: %d views, maxdiff %s' % (x['views_present'], x.get('maxdiff'))


def _r7(p, v):
    x = (p.get('r7') or {}).get(v)
    if not x:
        return 'n/a'
    if 'item_exc' in x:
        return 'EXC ' + x['item_exc'][:90]
    ns = x.get('native_series'); u = ((p.get('r7') or {}).get('U') or {}).get('native_series')
    if x.get('item_error'):
        return 'STUDY FALLBACK (%s); native series %s/%s' % (str(x['item_error'])[:70], ns, u)
    if x.get('identical'):
        return 'OK identical (%d windows, %s series)' % (x['windows'], ns)
    if ns is not None and u is not None and ns < u:
        return 'SILENT DROP %d of %d series (%d vs %d windows)' % (u - ns, u, x.get('windows', -1), x.get('U_windows', -1))
    return 'differs: %s vs %s windows, maxdiff %s, bad slices %s' % (x.get('windows'), x.get('U_windows'), x.get('maxdiff'), x.get('native_bad_slices'))


def table():
    ph = {k: RESULT.get('probe_' + k) or {} for k in ('S', 'SW', 'R', 'RW', 'R7', 'R7W')}
    r7of = {'R': 'R7', 'RW': 'R7W'}
    lines = ['| syntax (fixture study) | phase | plain pydicom | Raptor fastread study_lut (r384 = S/SW, r224 = R/RW) | '
             'R7 compact_infer (R/RW) | fleet fleet_common (V224-F20 = R/RW; HFR7-RR = S/SW) |',
             '|---|---|---|---|---|---|']
    for v in VARIANTS:
        for k in ('S', 'SW', 'R', 'RW'):
            p = ph[k]
            lines.append('| %s | %s | %s | %s | %s | %s |' % (VLABEL[v], k, _plain(p, v), _raptor(p, v),
                                                         _r7(ph[r7of[k]], v) if k in r7of else '-', _fleet(p, v)))
    # U invariance across environments (a decoder-wheel add-on must not move uncompressed data)
    inv = {}
    for rd in ('raptor', 'fleet', 'r7'):
        shas = {k: ((ph[k].get(rd) or {}).get('U') or {}).get('sha') for k in ph if ph[k].get(rd)}
        inv[rd] = shas
    RESULT['U_sha_by_phase'] = inv
    lines.append('')
    lines.append('U-control output sha by phase (equal = the wheels do not change uncompressed data): ' + json.dumps(inv))
    dec = {}
    for k, p in ph.items():
        e = (p.get('env') or {})
        dec[k] = {'pydicom': str(e.get('pydicom', '?')).split(' @')[0], 'pylibjpeg': str(e.get('pylibjpeg', '?')).split(' @')[0],
                  'gdcm': str(e.get('gdcm', '?')).split(' @')[0], 'PIL_features': e.get('PIL_features'),
                  'avail': {ts: (d.get('available') if isinstance(d, dict) else d)
                            for ts, d in (e.get('pydicom_decoders') or {}).items()} if isinstance(e.get('pydicom_decoders'), dict) else e.get('pydicom_decoders')}
    RESULT['decoder_env_by_phase'] = dec
    lines.append('decoders by phase: ' + json.dumps(dec))
    b = {k: {n: (x.get('ts'), x.get('ok'), (x.get('error') or '')[:60]) for n, x in ((p.get('bundled') or {}).items())} for k, p in ph.items()}
    lines.append('pydicom bundled test files (ts, ok, error) by phase: ' + json.dumps(b))
    # per-stage summary in the environment each stage really runs in (V224-F20), and with the decoder wheels
    stages = [('Raptor r384 (stock env)', 'S', 'SW', 'raptor'), ('Raptor r224 (after R7 pip)', 'R', 'RW', 'raptor'),
              ('R7 worker', 'R7', 'R7W', 'r7'), ('fleet in V224-F20 (after R7 pip)', 'R', 'RW', 'fleet'),
              ('fleet in HFR7-RR (stock env)', 'S', 'SW', 'fleet')]
    fmt = {'raptor': _raptor, 'r7': _r7, 'fleet': _fleet}
    summ = {}
    lines += ['', '| stage | env | ' + ' | '.join(VARIANTS) + ' |', '|---|---|' + '---|' * len(VARIANTS)]
    for name, e0, e1, rd in stages:
        for e in (e0, e1):
            cells = []
            for v in VARIANTS:
                c = fmt[rd](ph[e], v)
                cells.append('OK' if c.startswith('OK') else ('lossy-diff' if (v == 'J2KI' and c.startswith('differs')) else c[:40]))
            summ['%s | %s' % (name, e)] = dict(zip(VARIANTS, cells))
            lines.append('| %s | %s | %s |' % (name, e, ' | '.join(cells)))
    RESULT['stage_summary'] = summ
    md = '\n'.join(lines)
    (WORK / 'jpeg_check_table.md').write_text(md)
    print(md, flush=True)
    return {'rows': len(lines)}


# ---------------------------------------------------------------------------------------------------------------------
def main():
    inp = step('inputs', inputs)
    if not inp.get('COMP'):
        log('no competition data'); return
    step('census', lambda: census(inp['COMP']))
    step('mods', lambda: write_mods(inp))
    step('decoders', lambda: install_decoders(inp))
    fake = JC / 'fake'
    common = ['--mods', str(MODS), '--tmp', str(JC)]
    step('make', lambda: run_probe(['make', '--comp', inp['COMP'], '--out', str(fake)] + common, env_for('SW'), 3600))
    try:
        RESULT['fixtures'] = json.load(open(fake / 'fixtures.json'))
    except Exception as e:
        RESULT['fixtures'] = {'error': str(e)}
    save()
    pargs = ['--fake', str(fake), '--fastread', str(MODS / 'raptor_fastread.py'), '--fleet', str(JC / 'fleet')] + common
    if inp.get('FIX'):
        pargs += ['--synth', inp['FIX']]
    for ph in ('S', 'SW'):
        res = str(JC / ('res_%s.json' % ph))
        step('run_' + ph, lambda: run_probe(['probe', '--phase', ph, '--readers', 'raptor,fleet', '--result', res] + pargs,
                                             env_for(ph), 2400))
        try:
            RESULT['probe_' + ph] = json.load(open(res))
        except Exception as e:
            RESULT['probe_' + ph] = {'error': str(e)}
        save()
    r7 = step('r7_install', lambda: r7_install(inp))
    # R / RW = r224 + V224-F20 fleet after R7's pip install (no CODE on their path, as in V224); R7 / R7W = the R7 worker
    # (PYTHONPATH = CODE [+ decoder target]), run even if the R7 install step failed part-way (then its errors are logged)
    for ph, readers in (('R', 'raptor,fleet'), ('RW', 'raptor,fleet'), ('R7', 'r7'), ('R7W', 'r7')):
        if ph.startswith('R7') and 'step_error' in r7:
            continue
        res = str(JC / ('res_%s.json' % ph))
        extra = (['--r7src', str(JC / 'r7src'), '--code', r7['code'], '--anchor', r7['anchor'], '--side', r7['side']]
                 if ph.startswith('R7') else [])
        step('run_' + ph, lambda: run_probe(['probe', '--phase', ph, '--readers', readers, '--result', res] + extra + pargs,
                                             env_for(ph, r7.get('code')), 3600))
        try:
            RESULT['probe_' + ph] = json.load(open(res))
        except Exception as e:
            RESULT['probe_' + ph] = {'error': str(e)}
        save()
    step('table', table)
    RESULT['seconds'] = round(time.time() - T0, 1)
    save()
    log('JPEG_CHECK_DONE', RESULT['seconds'])


main()
