#!/usr/bin/env python
# build_pl4_labels.py (2026-10-11, Claude; box CPU only): PROBE4 OAI label tables = T123v00 MOAKS labels (UNCHANGED)
# + pseudo-label (PL) soft targets for the 3 labels OAI never supervises (MCL, Contusion, Fracture).
# Teacher = set D member of the SAME fold: for training fold F the PL comes from f{F}_s1043_ttr_frac75 (raw best, pass A,
# CPU, ~/oai_build/teacher/f{F}_s1043_ttr_frac75__raw__A__shard{0..3}of4.csv). That member never trained on RSNA fold F,
# so fold F's OOF stays leak-free (a 5-member mean teacher would carry fold-F RSNA labels into the OAI targets).
# Out: ~/oai_build/labels_v/oai_labels_T123PL_f{F}_v00.csv (F = 0..4) + pl4_labels_receipt.json.
import os, sys, json, glob, hashlib
import numpy as np, pandas as pd
H = os.path.expanduser('~')
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
PL = ['MCL', 'Contusion', 'Fracture']
SRC = f'{H}/oai_build/labels_v/oai_labels_T123_v00.csv'; SRC_MD5 = '45bb6f3b4ca284873a78f5565edfd007'
TD, OUT = f'{H}/oai_build/teacher', f'{H}/oai_build/labels_v'
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
assert md5(SRC) == SRC_MD5, 'T123 v00 label table changed'
base = pd.read_csv(SRC, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
assert list(base.columns) == T and base[PL].isna().all().all()
base = base[base.notna().any(axis=1)]
rec = dict(src=SRC, src_md5=SRC_MD5, n_oai=len(base), files={})
for F in range(5):
    sh = sorted(glob.glob(f'{TD}/f{F}_s1043_ttr_frac75__raw__A__shard*of4.csv'))
    assert len(sh) == 4, sh
    t = pd.concat([pd.read_csv(p, dtype={'StudyInstanceUID': str}) for p in sh]).set_index('StudyInstanceUID')
    assert t.index.is_unique and set(base.index) <= set(t.index), 'teacher misses OAI ids'
    t = t.loc[base.index, PL].astype(float)
    assert np.isfinite(t.to_numpy()).all() and ((t >= 0) & (t <= 1)).all().all()
    o = base.copy(); o[PL] = t.to_numpy()
    nine = [c for c in T if c not in PL]
    assert np.array_equal(o[nine].to_numpy(float), base[nine].to_numpy(float), equal_nan=True)
    p = f'{OUT}/oai_labels_T123PL_f{F}_v00.csv'
    o[T].to_csv(p + '.part', float_format='%.6g'); os.replace(p + '.part', p)
    rec['files'][F] = dict(path=p, md5=md5(p), teacher=[os.path.basename(s) for s in sh],
                           pl_mean={c: round(float(t[c].mean()), 4) for c in PL},
                           pl_ge50={c: round(float((t[c] >= .5).mean()), 4) for c in PL})
    print(F, json.dumps(rec['files'][F]['pl_mean']), json.dumps(rec['files'][F]['pl_ge50']), flush=True)
json.dump(rec, open(f'{OUT}/pl4_labels_receipt.json', 'w'), indent=1)
print('PL4 LABELS OK', len(base), 'knees x 5 folds')
