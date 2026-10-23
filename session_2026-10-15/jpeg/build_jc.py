#!/usr/bin/env python
# build_jc.py (2026-10-14, Claude): assemble the private CPU kernel eeshsaxena/rsna-jpeg-check from
#   kernel_src/driver_template.py + kernel_src/probe_jc.py + jll.py, with the Raptor _fastread module (base64, from V224-F20
#   cells 3 and 9, asserted equal) and R7's embedded SOURCES / SOURCE_SHA / CODE_SHA (V224-F20 cell 6) embedded verbatim.
# usage: python build_jc.py [--out kernel_jc]
import os, re, ast, json, zlib, base64, hashlib, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, '..', 'v224f', 'kernel_v224f20', 'rsna_sub_v224f20.ipynb')
NB_SHA = '8f8b38e4e97c70b7'
DOCKER = 'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f'

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(HERE, 'kernel_jc'))
A = ap.parse_args()

raw = open(NB, 'rb').read()
assert hashlib.sha256(raw).hexdigest().startswith(NB_SHA), 'V224-F20 notebook changed'
nb = json.loads(raw.decode('utf-8'))
S = lambda c: c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
b64s = []
for i in (3, 9):
    m = re.search(r'_FASTREAD_B64 = "([A-Za-z0-9+/=]+)"', S(nb['cells'][i]))
    assert m, i
    b64s.append(m.group(1))
assert b64s[0] == b64s[1], 'r384 / r224 fastread differ'
fr = base64.b64decode(b64s[0])
fr_sha = hashlib.sha256(fr).hexdigest()
tree = ast.parse(S(nb['cells'][6]))
vals = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('SOURCES', 'SOURCE_SHA', 'CODE_SHA', 'BUNDLE_ID'):
        vals[node.targets[0].id] = ast.literal_eval(node.value)
assert vals['BUNDLE_ID'] == '8c7b666e27dc0e8f'
for k, t in vals['SOURCES'].items():
    assert hashlib.sha256(t.encode('utf-8')).hexdigest() == vals['SOURCE_SHA'][k], k
r7z = base64.b64encode(zlib.compress(json.dumps(vals['SOURCES']).encode('utf-8'), 9)).decode('ascii')

tpl = open(os.path.join(HERE, 'kernel_src', 'driver_template.py'), encoding='utf-8').read()
rep = {'@@JLL_SRC@@': repr(open(os.path.join(HERE, 'jll.py'), encoding='utf-8').read()),
       '@@PROBE_SRC@@': repr(open(os.path.join(HERE, 'kernel_src', 'probe_jc.py'), encoding='utf-8').read()),
       '@@FASTREAD_B64@@': repr(b64s[0]), '@@FASTREAD_SHA@@': repr(fr_sha), '@@R7_SRC_Z@@': repr(r7z),
       '@@R7_SOURCE_SHA@@': repr(vals['SOURCE_SHA']), '@@R7_CODE_SHA@@': repr(vals['CODE_SHA'])}
for k, v in rep.items():
    assert tpl.count(k) == 1, k
    tpl = tpl.replace(k, v)
compile(tpl, 'rsna_jpeg_check.py', 'exec')
os.makedirs(A.out, exist_ok=True)
open(os.path.join(A.out, 'rsna_jpeg_check.py'), 'w', encoding='utf-8', newline='\n').write(tpl)
meta = {'id': 'eeshsaxena/rsna-jpeg-check', 'title': 'rsna jpeg check', 'code_file': 'rsna_jpeg_check.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': False, 'enable_tpu': False,
        'enable_internet': False,
        'dataset_sources': ['prvsiyan/rsna-knee-synthetic-dicom-fixtures', 'heliosli/rsna-knee-compact-r7cnxfsss',
                            'nartaa/rsna-knee-publication-swa-weights-20261007',
                            'heliosli/rsna-orthofoundation-fold0-epoch7-20261004',
                            'eeshsaxena/rsna-knee-cnxt-fleet-k11w', 'eeshsaxena/rsna-knee-cnxt-fleet-k14w',
                            'goodpjw2008/rsna-knee-2-5d-convnext-reader', 'glorysing/rsna-knee-offline-codecs-r09'],
        'kernel_sources': [], 'competition_sources': ['rsna-knee-abnormality-detection'], 'model_sources': [],
        'docker_image': DOCKER}
json.dump(meta, open(os.path.join(A.out, 'kernel-metadata.json'), 'w'), indent=1)
k = open(os.path.join(A.out, 'rsna_jpeg_check.py'), 'rb').read()
print('built', A.out, 'bytes', len(k), 'sha256', hashlib.sha256(k).hexdigest()[:16], 'fastread', fr_sha[:16],
      'r7 sources', sorted(vals['SOURCES']))
