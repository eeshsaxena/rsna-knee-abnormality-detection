# Fills G74F (gold74 fluid aggregates: 3-label E/S/B cell counts on the 67 complete studies + pairwise 2x2 counts from
# local_coupling.json) into box_fluid_template.py -> box_fluid_run.py (piped via ssh stdin). Aggregates only.
import json
import numpy as np, pandas as pd
R = 'C:/Users/eeshs/Downloads/rsna-knee/'
E = R + 'session_2026-10-19/ex/'
FL = ['Effusion', 'Synovitis', "Baker's"]
g = pd.read_csv(R + 'session_2026-10-18/next/g58/gold74_eval_set.csv')
d = g[FL].dropna().astype(int); k = (d.values * [4, 2, 1]).sum(1)
cells = {format(i, '03b'): int((k == i).sum()) for i in range(8)}
res = json.load(open(E + 'local_coupling.json'))['pairs']
pairs = {p: res[p]['g74']['counts'] for p in ('Effusion | Synovitis', "Effusion | Baker's", "Synovitis | Baker's")}
# cross-check pairwise counts against the csv
for p, c in pairs.items():
    a, b = p.split(' | '); dd = g[[a, b]].dropna().astype(int)
    cc = [int(((dd[a] == x) & (dd[b] == y)).sum()) for x in (0, 1) for y in (0, 1)]
    assert cc == c, (p, cc, c)
G74F = {'cells': cells, 'pairs': pairs, 'n_complete': int(len(d))}
print(json.dumps(G74F))
src = open(E + 'box_fluid_template.py', encoding='utf-8').read()
marker = "H = os.path.expanduser('~/oai_build')"
assert marker in src
src = src.replace(marker, marker + '\nG74F = ' + json.dumps(G74F), 1)
open(E + 'box_fluid_run.py', 'w', encoding='utf-8').write(src)
print('ok')
