# box_inv.py (10-21 lastday, Claude). BOX ONLY, read-only, CPU, prints AGGREGATES ONLY (file names, row counts, column
# names, fold counts). Never prints an id, a row or a per-knee value. Writes nothing.
# Run: ssh training@192.168.10.30 'nice -n 19 ~/miniconda3/envs/rsna/bin/python -I -' < box_inv.py
import os, glob, json
import pandas as pd
H = os.path.expanduser('~')
f = pd.read_csv(H + '/oai_build/union_t123v00/folds5_oai.csv', dtype=str)
print('folds5_oai.csv cols', list(f.columns), 'rows', len(f))
idc = [c for c in f.columns if 'uid' in c.lower() or c.lower() in ('id', 'kid', 'studyinstanceuid')]
fc = [c for c in f.columns if 'fold' in c.lower()]
print('id-like cols', idc, 'fold cols', fc)
if idc and fc:
    oai = f[idc[0]].str.startswith('oai_')
    for c in fc:
        print(' fold col', c, '| OAI rows fold counts', f.loc[oai, c].value_counts().sort_index().to_dict(),
              '| RSNA rows fold counts', f.loc[~oai, c].value_counts().sort_index().to_dict())
lab = pd.read_csv(H + '/oai_build/union_t123v00/labels_frac75oai_T123v00.csv', dtype=str)
print('labels_frac75oai_T123v00 cols', list(lab.columns)[:20], 'rows', len(lab))
for d in sorted(glob.glob(H + '/fleet_runs/*oai*')):
    if not os.path.isdir(d):
        continue
    for o in sorted(glob.glob(d + '/oof_*.csv')):
        x = pd.read_csv(o, dtype={'StudyInstanceUID': str})
        ic = 'StudyInstanceUID' if 'StudyInstanceUID' in x.columns else x.columns[0]
        n_oai = int(x[ic].astype(str).str.startswith('oai_').sum())
        print('%-70s rows %5d oai %5d ncols %d' % (o.replace(H, '~'), len(x), n_oai, x.shape[1]))
ex = pd.read_csv(sorted(glob.glob(H + '/fleet_runs/f0_s1043_ttr_frac75oaiT123v00/oof_*raw.csv'))[0], nrows=2)
print('oof cols', list(ex.columns))
for p in sorted(glob.glob(H + '/oai_build/gold74_preds/*.csv')):
    x = pd.read_csv(p, dtype=str)
    print('%-80s rows %d cols %d' % (p.replace(H, '~'), len(x), x.shape[1]))
import re
for dd in ('/r5_oai', '/oai_build/teacher', '/oai_build/teacher2'):
    L = sorted(os.listdir(H + dd)); ok = [n for n in L if not re.search(r'oai_[0-9]', n)]
    print(dd, 'n', len(L), 'names(no-id)', ok[:40])
