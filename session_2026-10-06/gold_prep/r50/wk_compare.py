# wk_compare.py: worker pass-A rows (served from DICOM) vs the trainer's OOF rows (from the store) for the same studies.
import glob, os, sys, numpy as np, pandas as pd
H = os.path.expanduser('~'); G = f'{H}/gold_prep/r50'; RUN = sys.argv[1]; TAG = sys.argv[2]
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
z = np.load(f'{G}/wk/out_{TAG}.npz', allow_pickle=False)
print('npz keys', sorted(z.files), '| names', list(z['names']), '| NV', z['NV'].tolist())
sids = [str(s) for s in z['sids']]
for k, n in enumerate(z['names']):
    tag = str(n).split('_')[2]
    oof = pd.read_csv(f'{RUN}/oof_f0_s243_{tag}.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    A = z['PA'][k]; B = z['PB'][k]; P = z['P'][k]; O = oof.loc[sids, T].to_numpy(np.float32)
    print(f'{n}: |PA - OOF| max {np.abs(A - O).max():.2e} mean {np.abs(A - O).mean():.2e} | |PB - PA| max {np.abs(B - A).max():.3f}'
          f' | P == (PA+PB)/2: {np.allclose(P, (A + B) / 2, atol=1e-6)} | finite {np.isfinite(P).all()}')
