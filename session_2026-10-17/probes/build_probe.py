#!/usr/bin/env python
# build_probe.py (2026-10-17, Claude): PREREG_10-15 Amendment 16 probes = the team best HW kernel (session_2026-10-16/hw/kernel_v224r6hws6fj,
# 57262115) with exactly ONE line changed. Usage: python build_probe.py {s615|wr45|a25}
import json, hashlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..', 'session_2026-10-16', 'hw', 'kernel_v224r6hws6fj')
P = {'s615': (18, 'v224r6hws6f15j', "    W_S6F, COV_MIN, CANARY_COL = 0.2, 0.98, 'ACL'",
              "    W_S6F, COV_MIN, CANARY_COL = 0.15, 0.98, 'ACL'   # [probe A16 s615] S6fall weight .2 -> .15"),
     'wr45': (14, 'v224r6hwwr45j', "W_RAPTOR, W_OURS = 0.5, 0.5",
              "W_RAPTOR, W_OURS = 0.45, 0.55   # [probe A16 wr45] Raptor .45 / ours .55 for the findings outside W_RAPTOR_C"),
     'a25': (14, 'v224r6hwa25j', "A224 = 0.35   # [s5] share of the Raptor slot given to the 224-crop model; fixed before any run",
             "A224 = 0.25   # [probe A16 a25] share of the Raptor slot given to the 224-crop model (.35 -> .25)")}
cell, tag, old, new = P[sys.argv[1]]
sha = lambda b: hashlib.sha256(b).hexdigest()
meta = json.load(open(os.path.join(SRC, 'kernel-metadata.json'), encoding='utf-8'))
raw = open(os.path.join(SRC, meta['code_file']), 'rb').read(); nb = json.loads(raw)
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
s = src(nb['cells'][cell]); assert s.count(old) == 1, (sys.argv[1], s.count(old)); nb['cells'][cell]['source'] = s.replace(old, new)
out = os.path.join(HERE, 'kernel_' + tag); os.makedirs(out, exist_ok=True)
code_file = 'rsna_sub_%s.ipynb' % tag
json.dump(nb, open(os.path.join(out, code_file), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
meta.update(id='eeshsaxena/rsna-sub-' + tag, title='rsna sub ' + tag, code_file=code_file)
json.dump(meta, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=2)
newb = open(os.path.join(out, code_file), 'rb').read()
a, b = json.loads(raw), json.loads(newb)
diff = [i for i, (x, y) in enumerate(zip(a['cells'], b['cells'])) if src(x) != src(y)]
la, lb = src(a['cells'][cell]).splitlines(), src(b['cells'][cell]).splitlines()
assert diff == [cell] and len(la) == len(lb) and sum(x != y for x, y in zip(la, lb)) == 1
json.dump(dict(probe=sys.argv[1], tag=tag, src_sha256=sha(raw), out_sha256=sha(newb), cell=cell, old=old, new=new), open(os.path.join(HERE, 'build_receipt_%s.json' % tag), 'w'), indent=1)
print('PROBE_BUILT', sys.argv[1], tag, sha(newb)[:16], '| base', sha(raw)[:16], '| 1 line in cell', cell)
