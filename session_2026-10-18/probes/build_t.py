#!/usr/bin/env python
# build_t.py (2026-10-18, Claude): PREREG_10-15 Amendment 17 table probes = the team best HW kernel (session_2026-10-16/hw/kernel_v224r6hws6fj,
# 57262115) with exactly ONE line changed: the W_RAPTOR_C per-finding table in cell 14. Usage: python build_t.py {pf50|bk50|lm40|fr65|c4}
import json, hashlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..', 'session_2026-10-16', 'hw', 'kernel_v224r6hws6fj')
HW = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, "Baker's": 0.35}
T = {'pf50': ('v224r6hwpf50j', {**HW, 'PF OA': 0.5}),
     'bk50': ('v224r6hwbk50j', {**HW, "Baker's": 0.5}),
     'lm40': ('v224r6hwlm40j', {**HW, 'Lateral Meniscus': 0.4}),
     'fr65': ('v224r6hwfr65j', {**HW, 'Fracture': 0.65}),
     'c4':   ('v224r6hwc4j', {**HW, 'PF OA': 0.5, "Baker's": 0.5, 'Lateral Meniscus': 0.4, 'Fracture': 0.65})}
OLD = "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, \"Baker's\": 0.35}   # [hw] heliosli blend-mine-in v1 per-finding Raptor weights (PREREG_10-15 A11); ours = 1 - it; every other finding uses W_RAPTOR"
p = sys.argv[1]; tag, table = T[p]; cell = 14
NEW = 'W_RAPTOR_C = %s   # [probe A17 %s] per-finding Raptor weights (HW table %s); ours = 1 - it; every other finding uses W_RAPTOR' % (
    repr(table), p, ', '.join('%s %g->%g' % (k, HW.get(k, 0.5), v) for k, v in table.items() if HW.get(k, 0.5) != v))
assert eval(NEW.split('   #')[0].split('=', 1)[1]) == table
sha = lambda b: hashlib.sha256(b).hexdigest()
meta = json.load(open(os.path.join(SRC, 'kernel-metadata.json'), encoding='utf-8'))
raw = open(os.path.join(SRC, meta['code_file']), 'rb').read(); nb = json.loads(raw)
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
s = src(nb['cells'][cell]); assert s.count(OLD) == 1, (p, s.count(OLD)); nb['cells'][cell]['source'] = s.replace(OLD, NEW)
out = os.path.join(HERE, 'kernel_' + tag); os.makedirs(out, exist_ok=True)
code_file = 'rsna_sub_%s.ipynb' % tag
json.dump(nb, open(os.path.join(out, code_file), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
meta.update(id='eeshsaxena/rsna-sub-' + tag, title='rsna sub ' + tag, code_file=code_file)
json.dump(meta, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=2)
newb = open(os.path.join(out, code_file), 'rb').read()
a, b = json.loads(raw), json.loads(newb)
diff = [i for i, (x, y) in enumerate(zip(a['cells'], b['cells'])) if src(x) != src(y)]
la, lb = src(a['cells'][cell]).splitlines(), src(b['cells'][cell]).splitlines()
assert diff == [cell] and len(a['cells']) == len(b['cells']) and len(la) == len(lb) and sum(x != y for x, y in zip(la, lb)) == 1
json.dump(dict(probe=p, tag=tag, table=table, src_sha256=sha(raw), out_sha256=sha(newb), cell=cell, old=OLD, new=NEW), open(os.path.join(HERE, 'build_receipt_%s.json' % tag), 'w'), indent=1)
print('PROBE_BUILT', p, tag, sha(newb)[:16], '| base', sha(raw)[:16], '| 1 line in cell', cell, '|', NEW[:120])
