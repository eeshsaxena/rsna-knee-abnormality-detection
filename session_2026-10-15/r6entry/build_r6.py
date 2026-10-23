#!/usr/bin/env python
# build_r6.py (2026-10-15, Claude): R6 entry = the J twin of the team best (rsna-sub-v224flips6fj v1, J_CHECK 16/16) with
# ONE change: heliosli R7 bundle 8c7b666e -> R6 bundle fe055406 (our private copy eeshsaxena/rsna-knee-r6-fe055406-v1 of
# heliosli/rsna-knee-r6-frozen-fe055406 version 1). Exactly 2 code lines + 1 metadata source + id / title / code_file change.
import json, hashlib, os, sys, copy
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'jpeg', 'overlay', 'r2', 'kernel_v224flips6fj')
OUT = os.path.join(HERE, 'kernel_v224r6flips6fj')
nb = json.load(open(os.path.join(SRC, 'rsna_sub_v224flips6fj.ipynb'), encoding='utf-8'))
meta = json.load(open(os.path.join(SRC, 'kernel-metadata.json'), encoding='utf-8'))
src_sha = hashlib.sha256(open(os.path.join(SRC, 'rsna_sub_v224flips6fj.ipynb'), 'rb').read()).hexdigest()
REPL = [("BUNDLE_ID = '8c7b666e27dc0e8f'", "BUNDLE_ID = 'fe055406fad6a12f'"),
        ("C_SLUG = 'rsna-knee-compact-r7cnxfsss'", "C_SLUG = 'rsna-knee-r6-fe055406-v1'")]
hits = {a: 0 for a, _ in REPL}
for c in nb['cells']:
    if c['cell_type'] != 'code':
        continue
    s = ''.join(c['source'])
    for a, b in REPL:
        n = s.count(a)
        if n:
            hits[a] += n; s = s.replace(a, b)
    c['source'] = s
assert all(v == 1 for v in hits.values()), hits
ds = meta['dataset_sources']
assert ds.count('heliosli/rsna-knee-compact-r7cnxfsss') == 1
meta['dataset_sources'] = ['eeshsaxena/rsna-knee-r6-fe055406-v1' if d == 'heliosli/rsna-knee-compact-r7cnxfsss' else d for d in ds]
meta.update(id='eeshsaxena/rsna-sub-v224r6flips6fj', title='rsna sub v224r6flips6fj', code_file='rsna_sub_v224r6flips6fj.ipynb')
os.makedirs(OUT, exist_ok=True)
p = os.path.join(OUT, meta['code_file'])
json.dump(nb, open(p, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
json.dump(meta, open(os.path.join(OUT, 'kernel-metadata.json'), 'w'), indent=2)
rec = dict(src=SRC, src_sha256=src_sha, out=p, out_sha256=hashlib.sha256(open(p, 'rb').read()).hexdigest(), replaced=hits,
           dataset_sources=meta['dataset_sources'])
json.dump(rec, open(os.path.join(HERE, 'build_receipt_v224r6flips6fj.json'), 'w'), indent=1)
print(json.dumps(rec, indent=1))
