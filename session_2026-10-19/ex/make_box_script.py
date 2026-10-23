# Fills G74 (gold74 2x2 counts per pair, local aggregates) into the box template -> box_coupling_run.py (piped via ssh stdin).
import json
E = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/ex/'
res = json.load(open(E + 'local_coupling.json'))
g74 = {k: v['g74']['counts'] for k, v in res['pairs'].items()}
src = open(E + 'box_coupling_template.py', encoding='utf-8').read()
marker = "H = os.path.expanduser('~/oai_build')"
assert marker in src
src = src.replace(marker, marker + '\nG74 = ' + json.dumps(g74), 1)
open(E + 'box_coupling_run.py', 'w', encoding='utf-8').write(src)
print('pairs', len(g74))
