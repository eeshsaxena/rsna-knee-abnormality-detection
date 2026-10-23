#!/usr/bin/env python
# build_a18.py (2026-10-18, Claude): PREREG_10-15 Amendment 18 = probes on the team best a25 kernel
# (session_2026-10-17/probes/kernel_v224r6hwa25j, 57316395 = HW + A224 .25) with exactly ONE line changed.
# Usage: python build_a18.py {pf50|bk50|lm40|fr65|c4|a15}
import json, hashlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..', 'session_2026-10-17', 'probes', 'kernel_v224r6hwa25j')
HW = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, "Baker's": 0.35}
T = {'pf50': {**HW, 'PF OA': 0.5}, 'bk50': {**HW, "Baker's": 0.5}, 'lm40': {**HW, 'Lateral Meniscus': 0.4},
     'fr65': {**HW, 'Fracture': 0.65}, 'c4': {**HW, 'PF OA': 0.5, "Baker's": 0.5, 'Lateral Meniscus': 0.4, 'Fracture': 0.65}}
OLD_T = "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, \"Baker's\": 0.35}"
OLD_A = "A224 = 0.25   # [probe A16 a25] share of the Raptor slot given to the 224-crop model (.35 -> .25)"
p = sys.argv[1]; cell = 14
sha = lambda b: hashlib.sha256(b).hexdigest()
meta = json.load(open(os.path.join(SRC, 'kernel-metadata.json'), encoding='utf-8'))
raw = open(os.path.join(SRC, meta['code_file']), 'rb').read(); nb = json.loads(raw)
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
s = src(nb['cells'][cell]); lines = s.split('\n')
if p == 'a15':
    tag = 'v224r6a15j'; idx = [i for i, l in enumerate(lines) if l == OLD_A]; assert len(idx) == 1, ('A224 line', len(idx))
    lines[idx[0]] = "A224 = 0.15   # [probe A18 a15] share of the Raptor slot given to the 224-crop model (.25 -> .15)"
    table = HW
else:
    tag = 'v224r6a25%sj' % p; table = T[p]
    idx = [i for i, l in enumerate(lines) if l.startswith(OLD_T)]; assert len(idx) == 1, ('table line', len(idx))
    lines[idx[0]] = 'W_RAPTOR_C = %s   # [probe A18 %s on a25] per-finding Raptor weights (HW table %s); ours = 1 - it; every other finding uses W_RAPTOR' % (
        repr(table), p, ', '.join('%s %g->%g' % (k, HW.get(k, 0.5), v) for k, v in table.items() if HW.get(k, 0.5) != v))
    assert eval(lines[idx[0]].split('   #')[0].split('=', 1)[1]) == table
nb['cells'][cell]['source'] = '\n'.join(lines)
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
json.dump(dict(probe=p, tag=tag, table=table, a224=(0.15 if p == 'a15' else 0.25), src_sha256=sha(raw), out_sha256=sha(newb), cell=cell),
          open(os.path.join(HERE, 'build_receipt_%s.json' % tag), 'w'), indent=1)
print('PROBE_BUILT', p, tag, sha(newb)[:16], '| base a25', sha(raw)[:16], '| 1 line in cell', cell)
