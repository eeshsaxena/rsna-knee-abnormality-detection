# make_absz_ref44.py (r7teacher/labels, 2026-10-14, PREREG_TEACHER Amendment A1 FX6; laptop, seconds): per-label |z|
# reference of the 44 REAL R7 hash-fold rows that exist before the full run (commit v1: 8 OOF rows of r7_oof.csv; proof v2:
# the hash-fold model's logits of the 36 proof studies, from the workers' dump.json). Output: r7_absz_ref44.json = for every
# label the 44 sorted |z| values, plus provenance (md5 of every source file, the uid list). sim_r7_oof.py uses it to give the
# simulated teacher realistic logit magnitudes (the pre-A1 DT stand-in never reached the 1e-4 clip, so the dry run could not
# show the clip bug). These 44 rows are the format-check rows already seen on 10-14; no other R7-OOF value is read.
#   usage: python make_absz_ref44.py [--oof_dir ../../oof] [--out r7_absz_ref44.json]
import argparse, hashlib, json, os
import numpy as np, pandas as pd

T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--oof_dir', default=os.path.normpath(os.path.join(HERE, '..', '..', 'oof')))
ap.add_argument('--out', default=os.path.join(HERE, 'r7_absz_ref44.json'))
A = ap.parse_args()
hf = lambda u: int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % 4
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
src, rows = {}, {}
p = os.path.join(A.oof_dir, 'out_commit_v1', 'r7_oof.csv')
o = pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID'); src[os.path.relpath(p, A.oof_dir).replace(os.sep, '/')] = md5(p)
for u in o.index:
    assert int(o.loc[u, 'r7_fold']) == hf(u)
    rows[u] = o.loc[u, ['z|' + c for c in T]].to_numpy(np.float64)
pdir = os.path.join(A.oof_dir, 'out_proof_v2', 'r7oof')
for jd in sorted(os.listdir(pdir)):
    if not jd.startswith('proof_'):
        continue
    rc = json.load(open(os.path.join(pdir, jd, 'receipt.json')))
    assert rc['models'] == ['E_lrprior_s0_f%d.pt' % k for k in range(4)], rc['models']
    dp = os.path.join(pdir, jd, 'dump.json'); src[os.path.relpath(dp, A.oof_dir).replace(os.sep, '/')] = md5(dp)
    for u, v in json.load(open(dp)).items():
        z = np.asarray(v['z'] if isinstance(v, dict) else v, np.float64)
        assert z.shape == (4, 12), z.shape
        rows.setdefault(u, z[hf(u)])                       # the hash-fold model = the one that did NOT train on u
assert len(rows) == 44, len(rows)
Z = np.stack(list(rows.values()))
ref = {c: sorted(float(x) for x in np.abs(Z[:, j])) for j, c in enumerate(T)}
tie = float((np.abs(Z) > np.log((1 - 1e-4) / 1e-4)).mean())
out = dict(note='per-label sorted |z| of the 44 real R7 hash-fold rows (commit v1 OOF 8 + proof v2 36); A1 FX6', n=len(rows),
           sources=src, uids_md5=hashlib.md5('\n'.join(sorted(rows)).encode()).hexdigest(),
           clip_tie_share_all=round(tie, 4), abs_z=ref)
json.dump(out, open(A.out, 'w'), indent=1)
print(f'wrote {A.out}: {len(rows)} rows | share of cells beyond the 1e-4 clip {tie:.3f} | median |z| per label '
      + ' '.join(f'{c} {np.median(ref[c]):.2f}' for c in T))
