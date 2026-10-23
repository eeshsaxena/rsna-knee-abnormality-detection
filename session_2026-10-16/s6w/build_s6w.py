#!/usr/bin/env python
# build_s6w.py (2026-10-16, Claude): "R6 + S6fall .3" = the team best R6 entry (57248084, .958) with ONE change: the S6F blend weight
# W_S6F .2 -> .3 (submission = .7 pct(V224-FLIP-R6) + .3 pct(S6fall)). PREREG_10-15 Amendment 12 (Eesh: "reach .959 tonight, use whatever").
import json, hashlib, os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..', 'session_2026-10-15', 'r6entry', 'kernel_v224r6flips6fj')
TAG = 'v224r6s6f3j'
sha = lambda b: hashlib.sha256(b).hexdigest()
meta = json.load(open(os.path.join(SRC, 'kernel-metadata.json'), encoding='utf-8'))
raw = open(os.path.join(SRC, meta['code_file']), 'rb').read(); nb = json.loads(raw)
OLD = "    W_S6F, COV_MIN, CANARY_COL = 0.2, 0.98, 'ACL'"
NEW = "    W_S6F, COV_MIN, CANARY_COL = 0.3, 0.98, 'ACL'   # [s6w] PREREG_10-15 A12: S6fall weight .2 -> .3 (one change vs R6 57248084)"
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
hits = []
for i, c in enumerate(nb['cells']):
    s = src(c)
    if OLD in s:
        assert s.count(OLD) == 1; hits.append(i); c['source'] = s.replace(OLD, NEW)
assert hits == [18], hits
out = os.path.join(HERE, 'kernel_' + TAG); os.makedirs(out, exist_ok=True)
code_file = 'rsna_sub_%s.ipynb' % TAG
json.dump(nb, open(os.path.join(out, code_file), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
meta.update(id='eeshsaxena/rsna-sub-' + TAG, title='rsna sub ' + TAG, code_file=code_file)
json.dump(meta, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=2)
new = open(os.path.join(out, code_file), 'rb').read()
a, b = json.loads(raw), json.loads(new)
diff = [i for i, (x, y) in enumerate(zip(a['cells'], b['cells'])) if src(x) != src(y)]
la, lb = src(a['cells'][18]).splitlines(), src(b['cells'][18]).splitlines()
assert diff == [18] and len(la) == len(lb) and sum(x != y for x, y in zip(la, lb)) == 1
json.dump(dict(tag=TAG, src_sha256=sha(raw), out_sha256=sha(new), changed_cell=18, old=OLD, new=NEW), open(os.path.join(HERE, 'build_receipt_%s.json' % TAG), 'w'), indent=1)
print('S6W_BUILT', TAG, sha(new)[:16], '| from', sha(raw)[:16], '| 1 line in cell 18')
