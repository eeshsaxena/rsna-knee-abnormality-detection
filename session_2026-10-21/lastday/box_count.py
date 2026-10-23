# box_count.py (10-21 lastday, Claude). BOX ONLY, read-only, CPU, AGGREGATES ONLY (counts, id SHAPES with every digit
# replaced by N). Never prints an id, a row or a per-knee value. Writes nothing.
# Question: do any OAI knees with MOAKS expert reads exist that NO OAI-trained model (T12v00 / T123v00 / PL / swinT / r50rad /
# allv) saw in training? Those would be the only valid held-out OAI ruler for an OAI-trained OA head.
import os, re, glob
import pandas as pd
H = os.path.expanduser('~')
shape = lambda s: re.sub(r'[0-9]', 'N', s)
def ids_of(p, col=None):
    x = pd.read_csv(p, dtype=str, usecols=lambda c: c in ('StudyInstanceUID', 'kid'))
    c = 'StudyInstanceUID' if 'StudyInstanceUID' in x.columns else 'kid'
    return set(x[c].astype(str))
src = ids_of(H + '/oai_build/oai_label_sources.csv')
print('label_sources kids', len(src), 'shapes', pd.Series([shape(s) for s in src]).value_counts().head(5).to_dict())
T = ids_of(H + '/oai_build/oai_labels_T123.csv'); print('oai_labels_T123 ids', len(T), 'in src', len(T & src))
uni = {}
for d in sorted(glob.glob(H + '/oai_build/union*')):
    if not os.path.isdir(d):
        continue
    for p in sorted(glob.glob(d + '/*.csv')):
        try:
            s = ids_of(p)
        except Exception as e:
            print(' skip', p.replace(H, '~'), type(e).__name__); continue
        o = {i for i in s if i.startswith('oai_')}
        uni[p] = o
        print('%-70s rows-ids %5d oai %5d shapes %s | oai in src %d' % (p.replace(H, '~'), len(s), len(o),
              pd.Series([shape(i) for i in o]).value_counts().head(3).to_dict() if o else {}, len(o & src)))
v00 = set().union(*[o for p, o in uni.items() if 'union_t123v00' in p or 'union_t12v00' in p])
allo = set().union(*uni.values())
print('OAI ids in V00 unions', len(v00), '| in ANY union', len(allo))
print('label-source kids NOT in V00 unions', len(src - v00), '| NOT in any union', len(src - allo))
for nm, p in (('R6 served', H + '/r5_oai/run_r6/worker0/predictions.csv'), ('S6 served', H + '/r5_oai/s6_r5/out/predictions.csv'),
              ('D5 teacher', H + '/oai_build/teacher/teacher_D5_mean_v00.csv')):
    s = {i for i in ids_of(p) if i.startswith('oai_')}
    print('%-10s oai %5d | in V00 union %5d | NOT in V00 union %5d | NOT in any union %5d' % (nm, len(s), len(s & v00), len(s - v00), len(s - allo)))
