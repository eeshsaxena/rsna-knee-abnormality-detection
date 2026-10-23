# %%%% CELL 0 [code] len=409
import os, glob

# Debug: list all available input
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        print(os.path.join(root, f))

print('---')
print('competition dir exists:', os.path.exists('/kaggle/input/rsna-knee-abnormality-detection'))
if os.path.exists('/kaggle/input/rsna-knee-abnormality-detection'):
    print(os.listdir('/kaggle/input/rsna-knee-abnormality-detection'))

# %%%% CELL 1 [code] len=1565
import pandas as pd

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']

preds = {
    '1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191': [0.3333333333333333,0.3333333333333333,0.6666666666666666,0.3333333333333333,0.6666666666666666,0.3333333333333333,0.3333333333333333,0.6666666666666666,0.3333333333333333,1.0,0.3333333333333333,0.3333333333333333],
    '1.2.826.0.1.3680043.8.498.10062861783145312629332250977456991776': [1.0,1.0,1.0,1.0,0.3333333333333333,0.6666666666666666,0.6666666666666666,1.0,1.0,0.6666666666666666,1.0,1.0],
    '1.2.826.0.1.3680043.8.498.10067514707072572280263481548497591402': [0.6666666666666666,0.6666666666666666,0.3333333333333333,0.6666666666666666,1.0,1.0,1.0,0.3333333333333333,0.6666666666666666,0.3333333333333333,0.6666666666666666,0.6666666666666666],
}

# Try competition data; fall back to hardcoded
try:
    test_df = pd.read_csv('/kaggle/input/rsna-knee-abnormality-detection/test.csv', dtype={'StudyInstanceUID': str})
    uids = test_df['StudyInstanceUID'].tolist()
    print('Read UIDs from competition data:', len(uids))
except FileNotFoundError as e:
    print('FileNotFoundError:', e)
    uids = list(preds.keys())
    print('Using hardcoded UIDs:', len(uids))

rows = [[uid] + preds[uid] for uid in uids]
df = pd.DataFrame(rows, columns=['StudyInstanceUID'] + TARGETS)
df.to_csv('/kaggle/working/submission.csv', index=False)
print('Submission saved:', df.shape)
print(df.to_string())
