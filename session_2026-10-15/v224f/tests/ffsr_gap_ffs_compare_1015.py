# one-off (FFSR gate test 10-15): the replace-mode gap probe applied to the FF-S mock, for comparison.
# (a) the 2 studies of worker 0 swapped on ALL 25 rows (a worker-level misalignment) -> the K14 anchor should catch it;
# (b) swapped on the 5 S rows only -> expected PASS (S rows are unanchored in FF-S too).
import os, sys, json, time, shutil, io, contextlib, subprocess, re
import numpy as np, pandas as pd

V = r'C:\Users\eeshs\Downloads\rsna-knee\session_2026-10-15\v224f'
ROOT = os.path.normpath(os.path.join(V, '..', '..'))
T = os.path.join(V, 'tests'); KD = os.path.join(T, 'standin_v224ffs20'); SRC = os.path.join(T, 'mock_out_v224ffs20')
REF = os.path.join(ROOT, 'session_2026-10-15', 'v224ff', 'out_v224ff20_v1')
SP = os.path.dirname(os.path.abspath(__file__))
B = json.load(open(os.path.join(KD, 'BUILD_V224FS.json')))
NB = json.load(open(os.path.join(KD, 'rsna_sub_v224ffs20.ipynb'), encoding='utf-8'))
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
HF = [src(c) for c in NB['cells'] if src(c).startswith('# [eeshsaxena V224-F-S HF')][0]
sample = pd.read_csv(os.path.join(ROOT, '_data', 'sample_submission.csv'), dtype={'StudyInstanceUID': str})
cols = list(sample.columns); labels = cols[1:]; test_ids = sample['StudyInstanceUID'].tolist()
TG = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
      'Contusion', 'Fracture']


def probe(tag, rows_fn):
    M = os.path.join(SP, 'gap_ffs_' + tag)
    if os.path.isdir(M):
        shutil.rmtree(M)
    shutil.copytree(SRC, M)
    zs = []
    for g in (0, 1):
        z = dict(np.load(os.path.join(M, '_fleet_%d.npz' % g), allow_pickle=False)); nm = [str(x) for x in z['names']]
        if g == 0:
            r = rows_fn(nm); P = z['P'].copy(); P[r] = P[r][:, ::-1, :]; z['P'] = P
        zs.append(z); np.savez(os.path.join(M, '_fleet_%d.npz' % g), **z)
    PP = np.concatenate([z['P'] for z in zs], axis=1).astype(np.float64); ii = np.concatenate([z['sids'] for z in zs]).astype(str)
    so = pd.DataFrame(PP.mean(0), columns=TG, index=ii).reindex(test_ids); so.insert(0, 'StudyInstanceUID', test_ids)
    so.to_csv(os.path.join(M, 'submission_fleetsolo.csv'), index=False)
    logp = os.path.join(M, 'rsna-sub-v224ffs20.log'); E = json.load(open(logp, encoding='utf-8'))
    sv = json.loads(re.search(r'V224F_SAVED (\{.*\})', next(e['data'] for e in E if 'V224F_SAVED' in e['data'])).group(1))
    fr = json.load(open(os.path.join(M, 'fleet_receipt.json')))
    H = pd.read_csv(os.path.join(M, 'submission_V224.csv'), dtype={'StudyInstanceUID': str})
    g = {'NB_T0': time.time() - 600, 'labels': labels, 'test_ids': test_ids, 'sub': H.copy(), '_K2_TARGETS': TG, '_V224F': sv,
         '_canary': {'fired': [], 'r224_used': True}, '_k2_receipt': dict(fr, clean=True)}
    b = io.StringIO()
    with contextlib.redirect_stdout(b):
        exec(compile(HF.replace('/kaggle/working', M.replace('\\', '/')), 'HF-S', 'exec'), g)
    for z_ in g.get('_parts') or []:
        z_.close()
    g.clear()
    new = b.getvalue().strip().splitlines()
    assert [ln.split(' ', 1)[0] for ln in new] == ['V224FS_F2G', 'HF_BLEND_OK', 'HF_COMPLETE', 'V224F_DONE'], new
    k = [i for i, e in enumerate(E) if any(x in e['data'] for x in ('V224FS_F2G', 'HF_BLEND_OK', 'HF_COMPLETE', 'V224F_DONE'))]
    E = E[:k[0]] + [dict(E[k[0]], data=ln + '\n') for ln in new] + E[k[-1] + 1:]
    json.dump(E, open(logp, 'w', encoding='utf-8'))
    d = float(np.abs(pd.read_csv(os.path.join(M, 'submission.csv'))[labels].values - pd.read_csv(os.path.join(SRC, 'submission.csv'))[labels].values).max())
    r = subprocess.run([sys.executable, os.path.join(V, 'v224ffs_check.py'), M, '--wf', '0.2', '--build', os.path.join(KD, 'BUILD_V224FS.json'),
                        '--ref-flip', os.path.join(ROOT, 'session_2026-10-14', 'v224', 'out_v224flip_v1'),
                        '--ref-exact', os.path.join(ROOT, 'session_2026-10-14', 'v224', 'out_v224_v1'), '--ref-fleet', REF],
                       capture_output=True, text=True, encoding='utf-8')
    last = r.stdout.strip().splitlines()[-1]
    k14 = [ln.strip() for ln in r.stdout.splitlines() if 'K14 group mean ==' in ln]
    print(f'FFS-GAP {tag}: submission max |diff| vs clean FF-S mock {d:.4f} | {last} | {k14[:1]}')
    shutil.rmtree(M)


probe('all25', lambda nm: list(range(len(nm))))
probe('s_only', lambda nm: [nm.index(s) for s in B['s']])
