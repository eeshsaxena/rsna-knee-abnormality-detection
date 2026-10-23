#!/usr/bin/env python
# build_v224f20j.py (2026-10-14, Claude subagent): V224-F20-J = the commit-tested V224-F20 notebook + ONE inserted cell (the
# DEC cell, dec_cell_template.py) right before "## Stage 1". Nothing else changes: every other cell is byte-identical and in
# order (proof: deleting the inserted cell re-serialises to the V224-F20 file byte for byte), every sha assert is kept, the
# kernel metadata only adds the public dataset glorysing/rsna-knee-offline-codecs-r09 (the three cp313 decoder wheels).
# The DEC cell enables pylibjpeg / libjpeg / openjpeg for every later stage PROCESS via os.environ['PYTHONPATH'] only after a
# self-test; on any failure it leaves the environment untouched (= V224-F20) and prints DEC_FALLBACK.
# The builder also asserts, from the notebook text, that every decoding stage process inherits os.environ['PYTHONPATH']:
#   Raptor r384 / r224: subprocess.run([...]) without env= (+ fork pool inside); R7: cell env = dict(os.environ, PYTHONPATH=CODE +
#   os.pathsep + os.environ PYTHONPATH) and probe_dual's workers env = dict(os.environ, ...); fleet: workers env = {**os.environ,
#   ..., 'PYTHONPATH': fleet_dir + os.pathsep + os.environ PYTHONPATH}; no cell assigns os.environ['PYTHONPATH'] except ours.
#   usage: python build_v224f20j.py [--base NB --base-sha SHA --tag v224f20j] [--out DIR]
import os, re, ast, json, copy, hashlib, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..', '..'))
F20_DIR = os.path.join(ROOT, 'session_2026-10-15', 'v224f', 'kernel_v224f20')
F20_NB = os.path.join(F20_DIR, 'rsna_sub_v224f20.ipynb')
F20_SHA = '8f8b38e4e97c70b7a0d02f7c5f8d181c11e8b83f26df257f92ea7830cbce44b8'
TEMPLATE = os.path.join(HERE, 'dec_cell_template.py')
FIXTURES = os.path.join(HERE, 'selftest_fixtures.json')
FIXTURES_SHA = '725bd7d8f528f022a201bb0e89031388f1a58ba0b7994747c92cf66fecfd6b34'
JC_DRIVER = os.path.join(HERE, '..', 'kernel_src', 'driver_template.py')         # PYPI_SHA as checked by rsna-jpeg-check v2
JC_RESULT = os.path.join(HERE, '..', 'out_jc_v2', 'jpeg_check_result.json')       # MEASURED pip --target listing (v2)
CODEC_DS = 'glorysing/rsna-knee-offline-codecs-r09'
PYPI_SHA = {
    'pylibjpeg-2.1.0-py3-none-any.whl': '25df9496a69e64e98c887fddee12a1271e275b5f74ba804f9bf98a08bb80993e',
    'pylibjpeg_libjpeg-2.4.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl':
        'd2c8da12c79d1f46596b89df94091be01f932d739613e8af93d1d557ac3bab74',
    'pylibjpeg_openjpeg-2.6.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl':
        '3e8ee9e5bd3b62766ca98fc143cd340633e60bb14450162426f32688908ad9e8'}

ap = argparse.ArgumentParser()
ap.add_argument('--base', default=F20_NB)
ap.add_argument('--base-sha', default=F20_SHA)
ap.add_argument('--base-meta', default=None, help='default: kernel-metadata.json next to --base')
ap.add_argument('--tag', default='v224f20j')
ap.add_argument('--out', default=HERE)
ap.add_argument('--allow-no-fleet', action='store_true', help='base without our fleet stage (e.g. V224-FLIP-RM): skip the fleet asserts')
A = ap.parse_args()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def dumps(nb):
    return json.dumps(nb, ensure_ascii=False, separators=(',', ':'))     # the V224 family serialisation


def src(c):
    s = c['source']
    return ''.join(s) if isinstance(s, list) else s


# ---- inputs -----------------------------------------------------------------------------------------------------------------
RAW = open(A.base, 'rb').read()
assert sha(RAW) == A.base_sha, ('base notebook changed', sha(RAW))
NB = json.loads(RAW.decode('utf-8'))
assert dumps(NB).encode('utf-8') == RAW, 'serialisation does not reproduce the base file'
C = NB['cells']
META = json.load(open(A.base_meta or os.path.join(os.path.dirname(A.base), 'kernel-metadata.json'), encoding='utf-8'))
fx_raw = open(FIXTURES, 'rb').read()
assert sha(fx_raw) == FIXTURES_SHA, 'self-test fixtures changed'
FIX = json.loads(fx_raw)
assert sorted(FIX) == ['JLL57_12U', 'JLL70_16S', 'U_16S']
drv = open(JC_DRIVER, encoding='utf-8').read()
for n, h in PYPI_SHA.items():
    assert ("'%s'" % h) in drv and n in drv, ('PyPI sha differs from the rsna-jpeg-check driver', n)
jc = json.load(open(JC_RESULT, encoding='utf-8'))
TOP = jc['decoders']['cp313']['files']
assert jc['decoders']['cp313']['check'] == '3.0.2 2.1.0 2.4.0 2.6.0' and all(jc['decoders']['cp313_sha_match_pypi'].values())
assert TOP == sorted(TOP) and len(TOP) == 8 and not any(t.startswith(('numpy', 'pydicom', 'PIL', 'torch')) for t in TOP), TOP

# ---- the decode-relevant structure of the base: find cells by content, not by index ---------------------------------------
idx = {}
for i, c in enumerate(C):
    s = src(c)
    if c['cell_type'] == 'markdown' and s.startswith('## Stage 1'):
        idx['md_stage1'] = i
    if s.startswith('%%writefile /tmp/raptor_run.py'):
        idx['r384_src'] = i
    if s.startswith('%%writefile /tmp/raptor224_run.py'):
        idx['r224_src'] = i
    if "subprocess.run([sys.executable, '-u', '/tmp/raptor_run.py'], timeout=2 * 3600)" in s:
        idx['r384_run'] = i
    if "subprocess.run([sys.executable, '-u', '/tmp/raptor224_run.py'], timeout=min(_left, 2.5 * 3600))" in s:
        idx['r224_run'] = i
    if s.startswith("ARM = '") and 'SOURCES = {' in s:
        idx['r7_setup'] = i
    if "env = dict(os.environ, PYTHONPATH=str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', ''), PYTHONDONTWRITEBYTECODE='1')" in s:
        idx['r7_run'] = i
    if s.startswith('K2_CFG = {'):
        idx['fleet'] = i
    if s.startswith('# === HF ENV-SNAP'):
        idx['env_snap'] = i
    if s.startswith('# === HF ENV-RESTORE'):
        idx['env_restore'] = i
need = ['md_stage1', 'r384_src', 'r224_src', 'r384_run', 'r224_run', 'r7_setup', 'r7_run', 'fleet', 'env_snap', 'env_restore']
HAS_FLEET = not (A.allow_no_fleet and not any(k in idx for k in ('fleet', 'env_snap', 'env_restore')))
if not HAS_FLEET:
    need = need[:7]
assert all(k in idx for k in need), {k: idx.get(k) for k in need}
assert idx['md_stage1'] < idx['r384_src'] < idx['r384_run'] < idx['r7_setup'] < idx['r7_run'] < idx['r224_run'], idx
if HAS_FLEET:
    assert idx['r224_run'] < idx['env_snap'] < idx['fleet'] < idx['env_restore'], idx
assert all(c['cell_type'] == 'markdown' for c in C[:idx['md_stage1']]), 'a code cell runs before Stage 1'
# Raptor: both scripts run as plain subprocesses (environment inherited), the fastread module is the same in both
for k in ('r384_run', 'r224_run'):
    assert 'env=' not in src(C[idx[k]]), k
b64 = []
for k in ('r384_src', 'r224_src'):
    s = src(C[idx[k]])
    m = re.findall(r'_FASTREAD_B64 = "([A-Za-z0-9+/=]+)"', s)
    assert len(m) == 1, k
    b64.append(m[0])
    assert 'PYTHONPATH' not in s and 'env=' not in s.replace('env=None', ''), ('raptor script touches the environment', k)
assert b64[0] == b64[1], 'r384 / r224 fastread differ'
# R7: the controller env and the worker env both carry os.environ's PYTHONPATH
r7 = src(C[idx['r7_setup']])
vals = {}
for node in ast.parse(r7).body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('SOURCES', 'SOURCE_SHA', 'CODE_SHA'):
        vals[node.targets[0].id] = ast.literal_eval(node.value)
for n, t in vals['SOURCES'].items():
    assert sha(t.encode('utf-8')) == vals['SOURCE_SHA'][n], n
pdual = vals['SOURCES']['probe_dual.py']
assert 'env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gi), OMP_NUM_THREADS="1"' in pdual and 'subprocess.Popen(cmd, env=env,' in pdual
assert "subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', *wheels], check=True)" in r7
assert '], env=env).returncode' in src(C[idx['r7_run']])
# fleet: the workers' env carries os.environ's PYTHONPATH; ENV-SNAP / ENV-RESTORE round-trip os.environ
if HAS_FLEET:
    fl = src(C[idx['fleet']])
    assert "'PYTHONPATH': fleet_dir + os.pathsep + os.environ.get('PYTHONPATH', '')}" in fl
    assert "subprocess.Popen([sys.executable, f'{fleet_dir}/fleet_worker.py'], env=env)" in fl
    assert '_HF_ENV0 = dict(_hf_os.environ)' in src(C[idx['env_snap']]) and '_hf_os.environ[_k] = _v' in src(C[idx['env_restore']])
# every process launch in the base that passes its own env= is one of the three checked above (R7 controller cell, probe_dual's
# workers inside the R7 SOURCES, fleet workers); all others (pip, nvidia-smi, Raptor scripts) inherit os.environ
units = []
for i, c in enumerate(C):
    if c['cell_type'] != 'code':
        continue
    units.append(('cell%d' % i, '\n'.join(ln for ln in src(c).split('\n') if not ln.lstrip().startswith(('%', '!')))))
units += [('R7:' + n, t) for n, t in vals['SOURCES'].items()]
env_calls = []
for name, text in units:
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Call) and any(k.arg == 'env' for k in node.keywords):
            env_calls.append((name, ast.unparse(node.func), ast.unparse([k for k in node.keywords if k.arg == 'env'][0].value)))
want_calls = {('cell%d' % idx['r7_run'], 'subprocess.run', 'env'), ('R7:probe_dual.py', 'subprocess.Popen', 'env')}
if HAS_FLEET:
    want_calls.add(('cell%d' % idx['fleet'], 'subprocess.Popen', 'env'))
extra_calls = [c for c in env_calls if c not in want_calls]
assert want_calls <= set(env_calls) and all(e.startswith(('dict(os.environ,', '{**os.environ')) and 'PYTHONPATH' not in e
                                            for _, _, e in extra_calls), ('process launches with their own env', env_calls)
# nobody else sets the process-wide PYTHONPATH
for i, c in enumerate(C):
    s = src(c)
    assert not re.search(r"environ\[['\"]PYTHONPATH['\"]\]\s*=", s) and "environ.setdefault('PYTHONPATH'" not in s and \
        "putenv('PYTHONPATH'" not in s, ('a base cell sets PYTHONPATH', i)
    for name in ('_dec_main', '_DEC_RECEIPT'):
        assert name not in s, ('name collision', name, i)

# ---- the DEC cell -----------------------------------------------------------------------------------------------------------
tpl = open(TEMPLATE, encoding='utf-8').read()
DEC = tpl.replace('__WHEELS__', repr(PYPI_SHA)).replace('__TOP__', repr(TOP)).replace('__FIX__', repr(FIX)) \
         .replace('__TAG__', A.tag)
for ph in ('__WHEELS__', '__TOP__', '__FIX__', '__TAG__'):
    assert ph not in DEC, ph
compile(DEC, '<dec cell>', 'exec')
assert '—' not in DEC and '–' not in DEC, 'dash'
assert DEC.startswith('# [eeshsaxena V224-F20-J DECODERS')
assert DEC.count("os.environ['PYTHONPATH'] = after") == 1

# ---- assemble: insert before "## Stage 1" -----------------------------------------------------------------------------------
pos = idx['md_stage1']
nb = copy.deepcopy(NB)
nb['cells'] = [copy.deepcopy(c) for c in C[:pos]] + [
    {'cell_type': 'code', 'metadata': {}, 'source': DEC, 'outputs': [], 'execution_count': None}] + [copy.deepcopy(c) for c in C[pos:]]
back = copy.deepcopy(nb); del back['cells'][pos]
assert dumps(back).encode('utf-8') == RAW, 'removing the DEC cell does not give the base notebook'
out_dir = os.path.join(A.out, 'kernel_' + A.tag)
os.makedirs(out_dir, exist_ok=True)
code_file = 'rsna_sub_%s.ipynb' % A.tag
nb_raw = dumps(nb).encode('utf-8')
open(os.path.join(out_dir, code_file), 'wb').write(nb_raw)
meta = copy.deepcopy(META)
assert CODEC_DS not in meta['dataset_sources']
meta.update(id='eeshsaxena/rsna-sub-' + A.tag, title='rsna sub ' + A.tag, code_file=code_file, is_private=True,
            enable_gpu=True, enable_tpu=False, enable_internet=False)
meta['dataset_sources'] = META['dataset_sources'] + [CODEC_DS]
assert {k: v for k, v in meta.items() if k not in ('id', 'title', 'code_file', 'dataset_sources')} == \
    {k: v for k, v in META.items() if k not in ('id', 'title', 'code_file', 'dataset_sources')}, 'metadata differs beyond id / datasets'
with open(os.path.join(out_dir, 'kernel-metadata.json'), 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=1)
receipt = {'tag': A.tag, 'base': os.path.relpath(A.base, ROOT).replace('\\', '/'), 'base_sha256': A.base_sha,
           'notebook_sha256': sha(nb_raw), 'cells': len(nb['cells']), 'dec_cell_index': pos, 'dec_cell_sha256': sha(DEC.encode('utf-8')),
           'dec_cell_chars': len(DEC), 'template_sha256': sha(tpl.encode('utf-8')), 'fixtures_sha256': FIXTURES_SHA,
           'fastread_sha256': sha(__import__('base64').b64decode(b64[0])), 'r7_source_sha': vals['SOURCE_SHA'],
           'r7_code_sha': vals['CODE_SHA'], 'stage_cells_in_base': idx, 'dataset_sources': meta['dataset_sources'],
           'docker_image': meta['docker_image'], 'proof': 'deleting cell %d re-serialises to the base notebook byte for byte' % pos}
json.dump(receipt, open(os.path.join(A.out, 'build_receipt_%s.json' % A.tag), 'w', encoding='utf-8'), indent=1)
with open(os.path.join(A.out, 'celldiff_%s.txt' % A.tag), 'w', encoding='utf-8', newline='\n') as f:
    f.write('# %s = %s (sha %s...) + ONE inserted code cell at index %d (before "## Stage 1"); every other cell byte-identical and in\n'
            '# order (deleting the inserted cell gives the base file byte for byte). Metadata: + dataset %s.\n'
            '#### inserted cell %d (DEC), full text:\n' % (A.tag, receipt['base'], A.base_sha[:16], pos, CODEC_DS, pos))
    f.write(DEC.replace(repr(FIX), "{...3 self-test DICOMs, selftest_fixtures.json sha %s...}" % FIXTURES_SHA[:16]))
print('BUILT', A.tag, os.path.join(out_dir, code_file), 'sha256', sha(nb_raw), 'cells', len(nb['cells']), 'dec cell at', pos,
      'dec sha', receipt['dec_cell_sha256'][:16], 'datasets', meta['dataset_sources'])
