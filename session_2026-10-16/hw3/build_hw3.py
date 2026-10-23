#!/usr/bin/env python
# build_hw3.py (2026-10-16, Claude): "R6 + heliosli weights + S6fall .3" = the A11 change (cell 14 weight table) AND the A12 change
# (cell 18 W_S6F .3) on the R6 best kernel, both lines byte-identical to the gated single-change builds. For 10-17 slot 1 ONLY if
# both A11 and A12 read positive (else not submitted). Gate: hw3_check.py.
import json, hashlib, os, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..', 'session_2026-10-15', 'r6entry', 'kernel_v224r6flips6fj')
TAG = 'v224r6hws6f3j'
sha = lambda b: hashlib.sha256(b).hexdigest()
hw = json.load(open(os.path.join(HERE, '..', 'hw', 'build_receipt_v224r6hws6fj.json')))
s6w = json.load(open(os.path.join(HERE, '..', 's6w', 'build_receipt_v224r6s6f3j.json')))
meta = json.load(open(os.path.join(SRC, 'kernel-metadata.json'), encoding='utf-8'))
raw = open(os.path.join(SRC, meta['code_file']), 'rb').read(); nb = json.loads(raw)
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
for cell, rec in ((14, hw), (18, s6w)):
    s = src(nb['cells'][cell]); assert s.count(rec['old']) == 1; nb['cells'][cell]['source'] = s.replace(rec['old'], rec['new'])
out = os.path.join(HERE, 'kernel_' + TAG); os.makedirs(out, exist_ok=True)
code_file = 'rsna_sub_%s.ipynb' % TAG
json.dump(nb, open(os.path.join(out, code_file), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
meta.update(id='eeshsaxena/rsna-sub-' + TAG, title='rsna sub ' + TAG, code_file=code_file)
json.dump(meta, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=2)
new = open(os.path.join(out, code_file), 'rb').read()
a, b = json.loads(raw), json.loads(new)
diff = [i for i, (x, y) in enumerate(zip(a['cells'], b['cells'])) if src(x) != src(y)]
assert diff == [14, 18]
for i in diff:
    la, lb = src(a['cells'][i]).splitlines(), src(b['cells'][i]).splitlines(); assert len(la) == len(lb) and sum(x != y for x, y in zip(la, lb)) == 1
json.dump(dict(tag=TAG, src_sha256=sha(raw), out_sha256=sha(new), changed=[14, 18]), open(os.path.join(HERE, 'build_receipt_%s.json' % TAG), 'w'), indent=1)
print('HW3_BUILT', TAG, sha(new)[:16], '| cells 14 + 18, 1 line each')
