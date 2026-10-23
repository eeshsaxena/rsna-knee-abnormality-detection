#!/usr/bin/env python
# build_combo.py (2026-10-16, Claude): pre-built COMBINATION entries for after the 10-16 reads, made by applying an already-gated
# one-change edit to another already-gated kernel, at the raw-text level (every other byte unchanged):
#   r6kc : KC1 kernel (kcstage/kernel_v224s6fkcj, V224S6FKC_CHECK 30/30) + the R6 swap of r6entry/build_r6.py (R6_CHECK 19/19):
#          exactly the 2 code lines BUNDLE_ID / C_SLUG + the metadata source heliosli R7 -> our R6 copy.
# Usage: python build_combo.py r6kc   -> session_2026-10-16/combo/kernel_v224r6s6fkcj/ + build_receipt_v224r6s6fkcj.json
import os, sys, json, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
S15 = os.path.join(HERE, '..', '..', 'session_2026-10-15')
sha = lambda b: hashlib.sha256(b).hexdigest()
R6_REPL = [("BUNDLE_ID = '8c7b666e27dc0e8f'", "BUNDLE_ID = 'fe055406fad6a12f'"),
           ("C_SLUG = 'rsna-knee-compact-r7cnxfsss'", "C_SLUG = 'rsna-knee-r6-fe055406-v1'")]
R6_META = ('heliosli/rsna-knee-compact-r7cnxfsss', 'eeshsaxena/rsna-knee-r6-fe055406-v1')
COMBOS = {'r6kc': dict(src=os.path.join(S15, 'kcstage', 'kernel_v224s6fkcj'), src_sha='3cff2339f5db426a', tag='v224r6s6fkcj',
                       repl=R6_REPL, meta=R6_META)}
k = sys.argv[1]
C = COMBOS[k]
meta = json.load(open(os.path.join(C['src'], 'kernel-metadata.json'), encoding='utf-8'))
raw = open(os.path.join(C['src'], meta['code_file']), 'rb').read()
assert sha(raw).startswith(C['src_sha']), ('source kernel changed', sha(raw)[:16])
txt = raw.decode('utf-8')
for a, b in C['repl']:
    n = txt.count(a)
    assert n == 1, (a, n)
    assert txt.count(b) == 0, ('already applied', b)
    txt = txt.replace(a, b)
new = txt.encode('utf-8')
json.loads(new)                                                  # still a valid notebook
# proof: the byte diff is exactly the replaced spans
d = sum(1 for x, y in zip(raw, new) if x != y)
ds = meta['dataset_sources']
assert ds.count(C['meta'][0]) == 1 and C['meta'][1] not in ds
meta['dataset_sources'] = [C['meta'][1] if s == C['meta'][0] else s for s in ds]
code_file = 'rsna_sub_%s.ipynb' % C['tag']
meta.update(id='eeshsaxena/rsna-sub-%s' % C['tag'], title='rsna sub %s' % C['tag'], code_file=code_file)
out = os.path.join(HERE, 'kernel_%s' % C['tag'])
os.makedirs(out, exist_ok=True)
open(os.path.join(out, code_file), 'wb').write(new)
json.dump(meta, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=1)
rec = dict(combo=k, tag=C['tag'], src=os.path.abspath(C['src']), src_sha256=sha(raw), out_sha256=sha(new), replaced=C['repl'],
           len_equal=len(raw) == len(new), bytes_differing_if_same_len=d if len(raw) == len(new) else None,
           dataset_sources=meta['dataset_sources'])
json.dump(rec, open(os.path.join(HERE, 'build_receipt_%s.json' % C['tag']), 'w'), indent=1)
print('COMBO_BUILT', k, C['tag'], 'sha', sha(new)[:16], '| from', sha(raw)[:16], '| sources', meta['dataset_sources'])
