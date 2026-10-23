# Fills G74P (gold74 pairwise 2x2 counts over the 9 OAI-covered labels, local aggregates) into box_global_template.py
# -> box_global_run.py (piped via ssh stdin). Only aggregate counts go TO the box; nothing comes back but aggregates.
import json
E = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/ex/'
res = json.load(open(E + 'local_coupling.json'))['pairs']
g74p = {k: v['g74']['counts'] for k, v in res.items()}
assert len(g74p) == 36
src = open(E + 'box_global_template.py', encoding='utf-8').read()
marker = "H = os.path.expanduser('~/oai_build')"
assert marker in src
src = src.replace(marker, marker + '\nG74P = ' + json.dumps(g74p), 1)
open(E + 'box_global_run.py', 'w', encoding='utf-8').write(src)
print('ok', len(g74p))
