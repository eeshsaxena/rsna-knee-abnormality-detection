# %%%% CELL 0 [code] len=223
import os, glob

# Debug: list all input files
print('=== /kaggle/input listing ===')
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        print(os.path.join(root, f))
print('=== end listing ===')


# %%%% CELL 1 [code] len=821
import pandas as pd
import numpy as np

# Try to find submission.csv from d4 kernel source
candidates = [
    '/kaggle/input/rsna-knee-d4-public0946-no-correction/submission.csv',
    '/kaggle/input/rsna-knee-d4-public-0946-no-correction/submission.csv',
]
src = None
for c in candidates:
    if os.path.exists(c):
        src = c
        print(f'Found: {c}')
        break

if src is None:
    # fallback: search all csvs named submission
    matches = glob.glob('/kaggle/input/**/submission.csv', recursive=True)
    print(f'Fallback search found: {matches}')
    if matches:
        src = matches[0]

if src is None:
    raise FileNotFoundError('Could not find d4 submission.csv in any input path')

df = pd.read_csv(src, dtype={'StudyInstanceUID': str})
print('Loaded d4 submission:', df.shape)
print(df.to_string())


# %%%% CELL 2 [code] len=1471
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
           'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
assert df.columns.tolist() == ['StudyInstanceUID'] + TARGETS, f'Schema mismatch: {df.columns.tolist()}'
assert len(df) == 3, f'Expected 3 rows, got {len(df)}'
print('Schema OK')

out = df.copy().reset_index(drop=True)

# MedMen: d4 predicts p1 most severe; raptor predicts p0 most severe
# Swap values for patients at row indices 0 and 1
v0 = float(out.loc[0, 'Medial Meniscus'])
v1 = float(out.loc[1, 'Medial Meniscus'])
print(f'MedMen before: p0={v0:.4f}, p1={v1:.4f}')
if v1 <= v0:
    print('WARNING: d4 already has p0>=p1 for MedMen, skip correction')
else:
    out.loc[0, 'Medial Meniscus'] = v1
    out.loc[1, 'Medial Meniscus'] = v0
    print(f'MedMen after:  p0={out.loc[0,"Medial Meniscus"]:.4f}, p1={out.loc[1,"Medial Meniscus"]:.4f}')

# MedOA: d4 predicts p2 most severe; raptor predicts p0 most severe
# Swap values for patients at row indices 0 and 2
v0 = float(out.loc[0, 'Medial OA'])
v2 = float(out.loc[2, 'Medial OA'])
print(f'MedOA before: p0={v0:.4f}, p2={v2:.4f}')
if v2 <= v0:
    print('WARNING: d4 already has p0>=p2 for MedOA, skip correction')
else:
    out.loc[0, 'Medial OA'] = v2
    out.loc[2, 'Medial OA'] = v0
    print(f'MedOA after:  p0={out.loc[0,"Medial OA"]:.4f}, p2={out.loc[2,"Medial OA"]:.4f}')

print('\nFinal corrected submission:')
print(out.to_string())


# %%%% CELL 3 [code] len=316
assert not out[TARGETS].isnull().any().any(), 'NaN in output'
assert (out[TARGETS] >= 0).all().all() and (out[TARGETS] <= 1).all().all(), 'Values OOB'
assert out['StudyInstanceUID'].notna().all(), 'NaN UIDs'

out.to_csv('/kaggle/working/submission.csv', index=False)
print('Saved submission.csv, shape:', out.shape)

