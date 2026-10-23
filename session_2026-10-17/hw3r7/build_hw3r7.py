#!/usr/bin/env python
# build_hw3r7.py (2026-10-17, Claude; copy of build_r7late.py on the hw3 base): "hw3 (weights + S6 .3) + R7 late leg" = the team best HW entry (session_2026-10-16/hw/kernel_v224r6hws6fj, 57262115)
# with ONE change: an appended R7 LATE LEG cell (cell_r7late_tpl.py; G_R7 6.6 h, W_S6F .2 = HW's) + heliosli/rsna-knee-compact-r7cnxfsss
# attached. Every base cell byte-identical. PREREG_10-15 Amendment 14.
import json, hashlib, os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..', 'session_2026-10-16', 'hw3', 'kernel_v224r6hws6f3j')
TAG = 'v224r6hw3r7j'
sha = lambda b: hashlib.sha256(b).hexdigest()
meta = json.load(open(os.path.join(SRC, 'kernel-metadata.json'), encoding='utf-8'))
raw = open(os.path.join(SRC, meta['code_file']), 'rb').read(); nb = json.loads(raw)
tpl = open(os.path.join(HERE, '..', '..', 'session_2026-10-16', 'r7late', 'cell_r7late_tpl.py'), encoding='utf-8').read()
cell = tpl.replace('__G_R7_H__', '6.6').replace('__W_S6F__', '0.3')
assert '__' + 'G_R7' not in cell and '__W_S6' + 'F__' not in cell and '—' not in cell
compile(cell, 'r7late_cell', 'exec')
n0 = len(nb['cells'])
nb['cells'].append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': cell})
assert 'heliosli/rsna-knee-compact-r7cnxfsss' not in meta['dataset_sources']
meta['dataset_sources'] = meta['dataset_sources'] + ['heliosli/rsna-knee-compact-r7cnxfsss']
out = os.path.join(HERE, 'kernel_' + TAG); os.makedirs(out, exist_ok=True)
code_file = 'rsna_sub_%s.ipynb' % TAG
json.dump(nb, open(os.path.join(out, code_file), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
meta.update(id='eeshsaxena/rsna-sub-' + TAG, title='rsna sub ' + TAG, code_file=code_file)
json.dump(meta, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=2)
new = open(os.path.join(out, code_file), 'rb').read()
a, b = json.loads(raw), json.loads(new)
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
assert len(b['cells']) == n0 + 1 and all(src(x) == src(y) for x, y in zip(a['cells'], b['cells'][:n0]))
json.dump(dict(tag=TAG, src_sha256=sha(raw), out_sha256=sha(new), cells=len(b['cells']), cell_sha256=sha(cell.encode()), dataset_sources=meta['dataset_sources']),
          open(os.path.join(HERE, 'build_receipt_%s.json' % TAG), 'w'), indent=1)
print('HW3R7_BUILT', TAG, sha(new)[:16], '| base', sha(raw)[:16], '| cells', n0, '->', len(b['cells']))
