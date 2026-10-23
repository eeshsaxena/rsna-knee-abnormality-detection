# merge_split.py (10-20 r5, Claude): add the gold58 approx studies (data_g58, split g58a) to the OAI rebuild (data_r5, split r5)
# as symlinked study dirs + concatenated CSVs, so ONE R6 run (one compile, no GPU gap) scores gold58a first, then the OAI
# knees in the stratified order. Writes ~/r5_oai/ids_r5.json. Prints counts only.
import json, os
import pandas as pd
H = os.path.expanduser('~'); R = H + '/r5_oai'
g_ids = json.load(open(R + '/ids_g58.json')); o_ids = json.load(open(R + '/ids_oai_all.json'))
gs = pd.read_csv(R + '/data_g58/g58a_series.csv', dtype=str); os_ = pd.read_csv(R + '/data_r5/r5_series.csv', dtype=str)
built = set(os_.StudyInstanceUID)
for u in g_ids:
    dst = f'{R}/data_r5/r5_series/{u}'
    if not os.path.islink(dst):
        os.symlink(f'{R}/data_g58/g58a_series/{u}', dst)
S = pd.concat([gs, os_], ignore_index=True)
S.to_csv(R + '/data_r5/r5_series.csv', index=False)
ids = [u for u in g_ids if u in set(gs.StudyInstanceUID)] + [k for k in o_ids if k in built]
pd.DataFrame({'StudyInstanceUID': ids}).to_csv(R + '/data_r5/r5.csv', index=False)
json.dump(ids, open(R + '/ids_r5.json', 'w'))
print('merged: gold58a studies', sum(not u.startswith('oai_') for u in ids), '| OAI knees', sum(u.startswith('oai_') for u in ids),
      '| series rows', len(S))
