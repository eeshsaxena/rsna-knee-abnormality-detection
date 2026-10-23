# e2e_cells.py (10-15, Claude subagent, kcstage): execute the 3 APPENDED KC cells of a BUILT kernel notebook exactly as written
# (only the path literals '/kaggle/working', '/tmp/kc', '/kaggle/input' redirected into a test dir) on the box CPU, with the real
# driver + serve wrapper (embedded in the SETUP cell, sha-checked), the real stand-in members, and a fake test root of RSNA TRAIN
# studies. The driver gets the test-only overrides KC_TMP_ROOT / KC_OUT_DIR / KC_TEST_ROOT / KC_DEVICE=cpu / KC_WORKERS /
# KC_WORKER_THREADS through the environment (on Kaggle none is set; the gate requires receipt overrides == {}).
#   python e2e_cells.py --nb KERNEL.ipynb --ds DS_DIR --root FAKEROOT --work WORKDIR --case clean|guard|s6fcanary [--age-h 6.0]
import os, sys, io, json, time, shutil, argparse, hashlib
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--nb', required=True); ap.add_argument('--ds', required=True); ap.add_argument('--root', required=True)
ap.add_argument('--work', required=True); ap.add_argument('--case', default='clean', choices=('clean', 'guard', 's6fcanary'))
ap.add_argument('--age-h', type=float, default=6.0); ap.add_argument('--threads', type=int, default=3)
A = ap.parse_args()
assert not os.path.exists(A.work), A.work
W, T, INP = A.work + '/working', A.work + '/tmpkc', A.work + '/input'
os.makedirs(W); os.makedirs(INP)
man = json.load(open(A.ds + '/KC_MANIFEST.json'))
os.symlink(A.ds, INP + '/' + man['slug'].split('/')[1])
nb = json.load(open(A.nb, encoding='utf-8'))
cells = [''.join(c['source']) if isinstance(c['source'], list) else c['source'] for c in nb['cells'][-3:]]
assert cells[0].startswith('# [eeshsaxena ') and 'KC 1/3 SETUP' in cells[0] and 'KC 3/3 BLEND' in cells[2]
cells = [c.replace('/kaggle/working', W).replace('/tmp/kc', T).replace('/kaggle/input', INP) for c in cells]
os.environ.update(KC_TMP_ROOT=T, KC_OUT_DIR=W + '/kc', KC_TEST_ROOT=A.root, KC_DEVICE='cpu', KC_WORKERS='2',
                  KC_WORKER_THREADS=str(A.threads), CUDA_VISIBLE_DEVICES='')
ids = pd.read_csv(A.root + '/test.csv', dtype=str)['StudyInstanceUID'].tolist()
LAB = list(pd.read_csv(A.root + '/sample_submission.csv', nrows=0).columns[1:])
rng = np.random.default_rng(5)
base = pd.DataFrame({'StudyInstanceUID': ids})
for c in LAB:
    base[c] = pd.Series(rng.random(len(ids))).rank(pct=True).astype(np.float32).values
if A.case == 's6fcanary':
    base['ACL'] = np.float32(0.5)
base.to_csv(W + '/submission.csv', index=False)
base_text = open(W + '/submission.csv').read()
g = {'__name__': '__main__', 'NB_T0': time.time() - A.age_h * 3600.0, '_canary': {'fired': []},
     '_S6F_SETUP': {'dec_ok': False}, '_S6F_RECEIPT': {'mode': 'V224FLIP_CANARY' if A.case == 's6fcanary' else 'S6F',
                                                         'fired': ['s6f->ACL'] if A.case == 's6fcanary' else []}}
t0 = time.time()
for i, c in enumerate(cells):
    exec(compile(c, '<kc cell %d>' % i, 'exec'), g)
    sys.stdout.flush()
R = []


def chk(name, ok, info=''):
    R.append(bool(ok))
    print('  %s %s %s' % ('OK  ' if ok else 'FAIL', name, info), flush=True)


rec = json.load(open(W + '/kc_receipt.json'))
sub = pd.read_csv(W + '/submission.csv', dtype={'StudyInstanceUID': str})
chk('os.environ PYTHONPATH / sys.path untouched by the cells', os.environ.get('PYTHONPATH') is None and T + '/code' not in sys.path)
if A.case in ('clean', 's6fcanary'):
    chk('mode KC, fired []', rec['mode'] == 'KC' and rec['fired'] == [], (rec['mode'], (rec.get('blend') or {}).get('why')))
    drv = json.load(open(W + '/kc/kc_driver_receipt.json'))
    chk('driver ok, coverage 1.0, overrides recorded (test run)', drv['ok'] and drv['B']['coverage'] == 1.0 and drv['overrides'],
        (drv.get('error'), drv.get('B', {}).get('coverage')))
    chk('side pass: 0 errors, side mismatch 0, both-from-table never served single', drv['side']['errors'] == 0 and
        drv['B']['n_side_mismatch'] == 0 and not drv['B']['serve_both_table_but_single'], drv['side'])
    kp = pd.read_csv(W + '/kc/kc_probs.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    b = pd.read_csv(W + '/submission_s6f.csv', dtype={'StudyInstanceUID': str})
    pct = lambda v: pd.Series(np.asarray(v, np.float64)).rank(method='average', pct=True).values
    exp = b.copy()
    for c in LAB:
        if b[c].nunique() <= 1:
            continue
        exp[c] = (0.8 * pct(b[c].values) + 0.2 * pct(kp.loc[ids, c].values)).astype(np.float32)
    d = float(np.abs(exp[LAB].to_numpy(np.float64) - sub[LAB].to_numpy(np.float64)).max())
    chk('blend == independent recompute (1e-6)', d <= 1e-6, 'max diff %.2e' % d)
    if A.case == 's6fcanary':
        chk('S6F canary column ACL kept exactly 0.5', bool((sub['ACL'] == 0.5).all()))
    dump = drv.get('dump') or {}
    chk('dump: every study, vol sha == served vol sha', isinstance(dump, dict) and len(dump) == len(ids) and
        all(v.get('equal') for v in dump.values()), {k[-8:]: v.get('equal', v.get('error')) for k, v in dump.items()})
    mb = drv['B'].get('microbench')
    print('  info microbench', json.dumps(mb), '| stages', json.dumps({k: v['seconds'] for k, v in drv['stages'].items()}))
    print('  info workers', json.dumps([{k: w.get(k) for k in ('device', 'shard', 'rc', 'closing', 'decode_s_mean', 'gpu_s_mean', 'n_both')}
                                         for w in drv['workers']]))
else:
    chk('guard: KC_SKIPPED start guard, canary Baker\'s 0.5, other 11 columns == base text', rec['mode'] == 'KC_CANARY' and
        rec['fired'] == ["kc->Baker's"] and (rec['run'] or {}).get('status') == 'SKIPPED_GUARD' and bool((sub["Baker's"] == 0.5).all())
        and all(pd.read_csv(io.StringIO(base_text), dtype=str)[c].tolist() == pd.read_csv(W + '/submission.csv', dtype=str)[c].tolist()
                for c in LAB if c != "Baker's"), rec.get('mode'))
print('E2E_CELLS %s %d/%d case %s | %.0f s' % ('PASS' if all(R) else 'FAIL', sum(R), len(R), A.case, time.time() - t0), flush=True)
