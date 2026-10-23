# Fills G74P (gold74 pairwise 2x2 counts), G74OA (gold74 3-label OA cells, skeptic) and G74C5 (gold74 5-label joint cell counts,
# order MOA, LOA, PF, MM, LM, bit i = label i) into cand_oam_box_template.py -> cand_oam_box_run.py. RSNA aggregates only.
import json
import numpy as np, pandas as pd
E = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/ex/'
G = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58/gold74_eval_set.csv'
res = json.load(open(E + 'local_coupling.json'))['pairs']
g74p = {k: v['g74']['counts'] for k, v in res.items()}
G74OA = {'000': 33, '001': 12, '010': 3, '011': 2, '100': 4, '101': 8, '110': 2, '111': 7}
LAB = ['Medial OA', 'Lateral OA', 'PF OA', 'Medial Meniscus', 'Lateral Meniscus']
g = pd.read_csv(G, dtype={'StudyInstanceUID': str})
d = g[LAB].dropna().values.astype(int)
c5 = np.bincount((d * (2 ** np.arange(5))).sum(1), minlength=32).tolist()
# sanity: 3-label OA marginal of the 5-label table vs skeptic G74OA (different completeness, so only printed)
oa = g[LAB[:3]].dropna().values.astype(int); k3 = np.bincount(oa[:, 0] * 4 + oa[:, 1] * 2 + oa[:, 2], minlength=8)
print('gold74 complete5 n', len(d), '| OA3 cells recomputed', dict(zip([format(i, '03b') for i in range(8)], k3.tolist())), 'vs skeptic', G74OA)
src = open(E + 'cand_oam_box_template.py', encoding='utf-8').read()
marker = "H = os.path.expanduser('~/oai_build')"
assert marker in src
src = src.replace(marker, marker + '\nG74P = ' + json.dumps(g74p) + '\nG74OA = ' + json.dumps(G74OA) + '\nG74C5 = ' + json.dumps(c5), 1)
open(E + 'cand_oam_box_run.py', 'w', encoding='utf-8').write(src)
print('ok pairs', len(g74p), 'c5 sum', sum(c5))
