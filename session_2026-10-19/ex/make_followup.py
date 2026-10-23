# Fills G74P (gold74 pairwise 2x2 counts, local aggregates) and G74OA (gold74 3-label OA cells, as in skeptic box_reweight.py)
# into box_followup_template.py -> box_followup_run.py (piped via ssh stdin).
import json
E = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/ex/'
res = json.load(open(E + 'local_coupling.json'))['pairs']
g74p = {k: v['g74']['counts'] for k, v in res.items()}
G74OA = {'000': 33, '001': 12, '010': 3, '011': 2, '100': 4, '101': 8, '110': 2, '111': 7}   # M/L/PF, gold74 complete (71)
src = open(E + 'box_followup_template.py', encoding='utf-8').read()
marker = "H = os.path.expanduser('~/oai_build')"
assert marker in src
src = src.replace(marker, marker + '\nG74P = ' + json.dumps(g74p) + '\nG74OA = ' + json.dumps(G74OA), 1)
open(E + 'box_followup_run.py', 'w', encoding='utf-8').write(src)
print('ok', len(g74p))
