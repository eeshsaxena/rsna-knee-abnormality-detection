# box_probe_cols.py (10-20 ~22:50 IST, Claude, STACK). Box CPU, read-only, piped via ssh stdin into python -I.
# Prints ONLY: per member file whether the 5 needed label columns exist + counts; label-file column presence; n of the
# common knee set. Never prints an id, a row, a value or a per-knee anything. Writes nothing.
import os
import pandas as pd
H = os.path.expanduser('~/oai_build'); R5 = os.path.expanduser('~/r5_oai')
NEED = ['Medial OA', 'Lateral OA', 'PF OA', 'Medial Meniscus', 'Lateral Meniscus']
F = {'R6': R5 + '/run_r6/worker0/predictions.csv', 'S6': R5 + '/s6_r5/out/predictions.csv', 'D5': H + '/teacher/teacher_D5_mean_v00.csv',
     's1044': H + '/teacher2/f0_s1044_ttr_mix50__raw__A__gpu.csv', 's1045': H + '/teacher2/f0_s1045_ttr288frac75__raw__A__gpu.csv'}
ids = None
for k, f in F.items():
    d = pd.read_csv(f, dtype={'StudyInstanceUID': str})
    d = d[d['StudyInstanceUID'].str.startswith('oai_')].drop_duplicates('StudyInstanceUID')
    have = [c in d.columns for c in NEED]
    ok = d.dropna(subset=[c for c in NEED if c in d.columns]).StudyInstanceUID
    ids = set(ok) if ids is None else ids & set(ok)
    print('member %-6s has %s | n oai rows %d' % (k, have, len(ok)))
T = pd.read_csv(H + '/oai_labels_T123.csv', nrows=0)
print('T123 has MM / LM columns:', 'Medial Meniscus' in T.columns, 'Lateral Meniscus' in T.columns)
print('n common knees (all 5 members, all 5 cols) %d' % len(ids))
