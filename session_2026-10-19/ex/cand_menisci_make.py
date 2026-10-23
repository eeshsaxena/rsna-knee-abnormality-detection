# Fills G74P (gold74 pairwise 2x2 counts, local aggregates from local_coupling.json) into cand_menisci_box_template.py
# -> cand_menisci_box_run.py (piped via ssh stdin; nothing is written on the box).
import json
E = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/ex/'
res = json.load(open(E + 'local_coupling.json'))['pairs']
g74p = {k: v['g74']['counts'] for k, v in res.items()}
src = open(E + 'cand_menisci_box_template.py', encoding='utf-8').read()
marker = "H = os.path.expanduser('~/oai_build')"
assert marker in src
src = src.replace(marker, marker + '\nG74P = ' + json.dumps(g74p), 1)
open(E + 'cand_menisci_box_run.py', 'w', encoding='utf-8').write(src)
print('ok', len(g74p), g74p['Medial Meniscus | Lateral Meniscus'])
