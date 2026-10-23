#!/usr/bin/env python
# build_a19.py (2026-10-18, Claude): PREREG_10-15 Amendment 19 = the team best c4 kernel (session_2026-10-18/a18/kernel_v224r6a25c4j,
# 57343435) with ONLY the cell-14 W_RAPTOR_C table line changed. Usage: python build_a19.py {c5|c4pfct|pf10|ef65|fr80}
import json, hashlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'a18', 'kernel_v224r6a25c4j')
C4 = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}
T = {'c5': {**C4, 'PF OA': 0.1, 'Contusion': 0.2, 'Effusion': 0.65, 'Fracture': 0.8},
     'c4pfct': {**C4, 'PF OA': 0.1, 'Contusion': 0.2},
     'pf10': {**C4, 'PF OA': 0.1}, 'ef65': {**C4, 'Effusion': 0.65}, 'fr80': {**C4, 'Fracture': 0.8},
     'fr90': {**C4, 'Fracture': 0.9}, 'fr100': {**C4, 'Fracture': 1.0}, 'c2': {**C4, 'Fracture': 0.9, 'PF OA': 0.1}, 'c2max': {**C4, 'Fracture': 1.0, 'PF OA': 0.1},
     'c3max': {**C4, 'Fracture': 1.0, 'PF OA': 0.1, 'Lateral OA': 0.1}}
p = sys.argv[1]; table = T[p]; tag = 'v224r6c4%sj' % p; cell = 14
sha = lambda b: hashlib.sha256(b).hexdigest()
meta = json.load(open(os.path.join(SRC, 'kernel-metadata.json'), encoding='utf-8'))
raw = open(os.path.join(SRC, meta['code_file']), 'rb').read(); nb = json.loads(raw)
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
lines = src(nb['cells'][cell]).split('\n')
idx = [i for i, l in enumerate(lines) if l.startswith('W_RAPTOR_C = ')]; assert len(idx) == 1, len(idx)
assert eval(lines[idx[0]].split('   #')[0].split('=', 1)[1]) == C4, 'base table is not c4'
lines[idx[0]] = 'W_RAPTOR_C = %s   # [probe A19 %s on c4] per-finding Raptor weights (c4 table %s); ours = 1 - it; every other finding uses W_RAPTOR' % (
    repr(table), p, ', '.join('%s %g->%g' % (k, C4.get(k, 0.5), v) for k, v in table.items() if C4.get(k, 0.5) != v))
assert eval(lines[idx[0]].split('   #')[0].split('=', 1)[1]) == table
nb['cells'][cell]['source'] = '\n'.join(lines)
out = os.path.join(HERE, 'kernel_' + tag); os.makedirs(out, exist_ok=True); cf = 'rsna_sub_%s.ipynb' % tag
json.dump(nb, open(os.path.join(out, cf), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
meta.update(id='eeshsaxena/rsna-sub-' + tag, title='rsna sub ' + tag, code_file=cf)
json.dump(meta, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=2)
newb = open(os.path.join(out, cf), 'rb').read(); a, b = json.loads(raw), json.loads(newb)
diff = [i for i, (x, y) in enumerate(zip(a['cells'], b['cells'])) if src(x) != src(y)]
la, lb = src(a['cells'][cell]).splitlines(), src(b['cells'][cell]).splitlines()
assert diff == [cell] and len(a['cells']) == len(b['cells']) and len(la) == len(lb) and sum(x != y for x, y in zip(la, lb)) == 1
json.dump(dict(probe=p, tag=tag, table=table, src_sha256=sha(raw), out_sha256=sha(newb), cell=cell), open(os.path.join(HERE, 'build_receipt_%s.json' % tag), 'w'), indent=1)
print('PROBE_BUILT', p, tag, sha(newb)[:16], '| base c4', sha(raw)[:16], '| 1 line in cell 14')
