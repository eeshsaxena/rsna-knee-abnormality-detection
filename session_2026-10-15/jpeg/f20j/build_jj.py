#!/usr/bin/env python
# build_jj.py (2026-10-14, Claude subagent): assemble the private CPU kernel eeshsaxena/rsna-v224f20j-check from
# kernel_src/driver_jj.py + kernel_src/probe_jj.py + ../jll.py, with everything decode-relevant taken FROM THE BUILT V224-F20-J
# NOTEBOOK (sha-asserted): the DEC cell text (verbatim), the Raptor _fastread module (base64, equal in the r384 and r224 cells),
# R7's embedded SOURCES / SOURCE_SHA / CODE_SHA. It also asserts that the environment formulas the driver re-creates are the
# notebook's own lines (R7 cell env, R7 pip command, fleet worker env).
# usage: python build_jj.py [--nb kernel_v224f20j/rsna_sub_v224f20j.ipynb] [--out kernel_jj]
import os, re, ast, json, zlib, base64, hashlib, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--nb', default=os.path.join(HERE, 'kernel_v224f20j', 'rsna_sub_v224f20j.ipynb'))
ap.add_argument('--out', default=os.path.join(HERE, 'kernel_jj'))
A = ap.parse_args()
rec = json.load(open(os.path.join(HERE, 'build_receipt_v224f20j.json'), encoding='utf-8'))
raw = open(A.nb, 'rb').read()
NB_SHA = hashlib.sha256(raw).hexdigest()
assert NB_SHA == rec['notebook_sha256'], 'notebook differs from the build receipt'
nb = json.loads(raw.decode('utf-8'))
meta_nb = json.load(open(os.path.join(os.path.dirname(A.nb), 'kernel-metadata.json'), encoding='utf-8'))
S = lambda c: c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
cells = [S(c) for c in nb['cells']]
dec = cells[rec['dec_cell_index']]
assert dec.startswith('# [eeshsaxena V224-F20-J DECODERS') and hashlib.sha256(dec.encode('utf-8')).hexdigest() == rec['dec_cell_sha256']
assert sum(c.startswith('# [eeshsaxena V224-F20-J DECODERS') for c in cells) == 1
b64s = [m.group(1) for c in cells for m in [re.search(r'_FASTREAD_B64 = "([A-Za-z0-9+/=]+)"', c)] if m]
assert len(b64s) == 2 and b64s[0] == b64s[1]
fr_sha = hashlib.sha256(base64.b64decode(b64s[0])).hexdigest()
assert fr_sha == rec['fastread_sha256']
r7cell = [c for c in cells if c.startswith("ARM = '") and 'SOURCES = {' in c]
assert len(r7cell) == 1
vals = {}
for node in ast.parse(r7cell[0]).body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('SOURCES', 'SOURCE_SHA', 'CODE_SHA', 'BUNDLE_ID'):
        vals[node.targets[0].id] = ast.literal_eval(node.value)
assert vals['BUNDLE_ID'] == '8c7b666e27dc0e8f' and vals['SOURCE_SHA'] == rec['r7_source_sha'] and vals['CODE_SHA'] == rec['r7_code_sha']
for k, t in vals['SOURCES'].items():
    assert hashlib.sha256(t.encode('utf-8')).hexdigest() == vals['SOURCE_SHA'][k], k
r7z = base64.b64encode(zlib.compress(json.dumps(vals['SOURCES']).encode('utf-8'), 9)).decode('ascii')
# the environment formulas the driver re-creates must be the notebook's own lines
allsrc = '\n'.join(cells)
for line in ("env = dict(os.environ, PYTHONPATH=str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', ''), PYTHONDONTWRITEBYTECODE='1')",
             "subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', *wheels], check=True)",
             "'PYTHONPATH': fleet_dir + os.pathsep + os.environ.get('PYTHONPATH', '')}",
             "subprocess.run([sys.executable, '-u', '/tmp/raptor_run.py'], timeout=2 * 3600)",
             "subprocess.run([sys.executable, '-u', '/tmp/raptor224_run.py'], timeout=min(_left, 2.5 * 3600))"):
    assert allsrc.count(line) == 1, line
tpl = open(os.path.join(HERE, 'kernel_src', 'driver_jj.py'), encoding='utf-8').read()
drv_lines = ("PYTHONPATH=r7['code'] + os.pathsep + PP['0']", "PYTHONPATH=r7['code'] + os.pathsep + PP['J']",
             "subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', *wheels], capture_output=True, text=True)")
for line in drv_lines:
    assert line in tpl, line
rep = {'@@DEC_SRC@@': repr(dec), '@@DEC_SHA@@': repr(rec['dec_cell_sha256']), '@@NB_SHA@@': repr(NB_SHA),
       '@@JLL_SRC@@': repr(open(os.path.join(HERE, '..', 'jll.py'), encoding='utf-8').read()),
       '@@PROBE_SRC@@': repr(open(os.path.join(HERE, 'kernel_src', 'probe_jj.py'), encoding='utf-8').read()),
       '@@FASTREAD_B64@@': repr(b64s[0]), '@@FASTREAD_SHA@@': repr(fr_sha), '@@R7_SRC_Z@@': repr(r7z),
       '@@R7_SOURCE_SHA@@': repr(vals['SOURCE_SHA']), '@@R7_CODE_SHA@@': repr(vals['CODE_SHA'])}
for k, v in rep.items():
    assert tpl.count(k) == 1, k
    tpl = tpl.replace(k, v)
compile(tpl, 'rsna_v224f20j_check.py', 'exec')
compile(open(os.path.join(HERE, 'kernel_src', 'probe_jj.py'), encoding='utf-8').read(), 'probe_jj.py', 'exec')
assert '—' not in tpl
os.makedirs(A.out, exist_ok=True)
open(os.path.join(A.out, 'rsna_v224f20j_check.py'), 'w', encoding='utf-8', newline='\n').write(tpl)
meta = {'id': 'eeshsaxena/rsna-v224f20j-check', 'title': 'rsna v224f20j check', 'code_file': 'rsna_v224f20j_check.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': False, 'enable_tpu': False,
        'enable_internet': False,
        'dataset_sources': ['heliosli/rsna-knee-compact-r7cnxfsss', 'heliosli/rsna-orthofoundation-fold0-epoch7-20261004',
                            'eeshsaxena/rsna-knee-cnxt-fleet-k11w', 'glorysing/rsna-knee-offline-codecs-r09',
                            'prvsiyan/rsna-knee-synthetic-dicom-fixtures'],
        'kernel_sources': [], 'competition_sources': ['rsna-knee-abnormality-detection'], 'model_sources': [],
        'docker_image': meta_nb['docker_image']}
assert set(meta['dataset_sources']) - {'prvsiyan/rsna-knee-synthetic-dicom-fixtures'} <= set(meta_nb['dataset_sources'])
json.dump(meta, open(os.path.join(A.out, 'kernel-metadata.json'), 'w'), indent=1)
k = open(os.path.join(A.out, 'rsna_v224f20j_check.py'), 'rb').read()
print('built', A.out, 'bytes', len(k), 'sha256', hashlib.sha256(k).hexdigest()[:16], 'from notebook', NB_SHA[:16],
      'dec cell', rec['dec_cell_sha256'][:16], 'fastread', fr_sha[:16])
