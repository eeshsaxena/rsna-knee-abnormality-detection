#!/usr/bin/env python
# build_hw.py (2026-10-16, Claude): "R6 + heliosli per-finding weights + S6fall" = the team best R6 entry (r6entry/kernel_v224r6flips6fj,
# 57248084 = .958) with ONE change: the stage-3 per-finding Raptor weights W_RAPTOR_C replaced by heliosli's public
# "blend-mine-in" v1 table (Medial OA .5, Lateral OA .2, PF OA .6, Synovitis .8, Baker's .35; every other finding .5 as before).
# PREREG_10-15 Amendment 11 (Eesh approved 10-16 ~14:05: one-time exception to "never public-tuned weights", as the final-pick hedge).
import json, hashlib, os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..', 'session_2026-10-15', 'r6entry', 'kernel_v224r6flips6fj')
TAG = 'v224r6hws6fj'
sha = lambda b: hashlib.sha256(b).hexdigest()
meta = json.load(open(os.path.join(SRC, 'kernel-metadata.json'), encoding='utf-8'))
raw = open(os.path.join(SRC, meta['code_file']), 'rb').read()
nb = json.loads(raw)
OLD = "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.5, 'PF OA': 0.5}   # per-finding Raptor weight (ours = 1 - it); every other finding uses W_RAPTOR"
NEW = ("W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, \"Baker's\": 0.35}   "
       "# [hw] heliosli blend-mine-in v1 per-finding Raptor weights (PREREG_10-15 A11); ours = 1 - it; every other finding uses W_RAPTOR")
hits = []
for i, c in enumerate(nb['cells']):
    s = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
    if OLD in s:
        hits.append(i); c['source'] = s.replace(OLD, NEW)
assert hits == [14], hits
out = os.path.join(HERE, 'kernel_' + TAG); os.makedirs(out, exist_ok=True)
code_file = 'rsna_sub_%s.ipynb' % TAG
json.dump(nb, open(os.path.join(out, code_file), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
meta.update(id='eeshsaxena/rsna-sub-' + TAG, title='rsna sub ' + TAG, code_file=code_file)
json.dump(meta, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=2)
new = open(os.path.join(out, code_file), 'rb').read()
a, b = json.loads(raw), json.loads(new)
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
diff = [i for i, (x, y) in enumerate(zip(a['cells'], b['cells'])) if src(x) != src(y)]
assert diff == [14] and len(a['cells']) == len(b['cells'])
la, lb = src(a['cells'][14]).splitlines(), src(b['cells'][14]).splitlines()
assert len(la) == len(lb) and sum(x != y for x, y in zip(la, lb)) == 1
rec = dict(tag=TAG, src_sha256=sha(raw), out_sha256=sha(new), changed_cell=14, old=OLD, new=NEW, dataset_sources=meta['dataset_sources'])
json.dump(rec, open(os.path.join(HERE, 'build_receipt_%s.json' % TAG), 'w'), indent=1)
print('HW_BUILT', TAG, sha(new)[:16], '| from', sha(raw)[:16], '| 1 line in cell 14')
