# k2_cell_run.py <notebook.ipynb> <fleet dataset dir> <comp dir> <run dir> [kernel_age_h]  (harness only)
# Executes the APPENDED K2 cells (index 31+) of a notebook built by k2/make_k2_kernel.py, exact cell text with the
# /kaggle prefixes rewritten to <run dir>, after a synthetic parent submission.csv (per-label rank pct, test ids in a
# shuffled row order), then k2/k2_check.check(work, expect_fleet='clean', expect_crop='none'). Same harness shims as
# k2/tests/run_scenario.py: parent globals T0/COMP/ROOT + RSNA_COMP_ROOT set here; torch.cuda reports 2 devices inside
# the fleet cell only; workers forced to CPU (FLEET_ALLOW_CPU=1) with the round3 PIL cv2 shim. Prints 'RESULT {json}'.
import os, sys, json, time, shutil, subprocess
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'; os.environ['MKL_NUM_THREADS'] = '2'
from _paths import K2, SHIM
sys.path.insert(0, K2); sys.path.insert(0, SHIM)
import numpy as np, pandas as pd, torch
torch.set_num_threads(2)
import k2_check
TARGETS = k2_check.TARGETS
NB, DS, COMP, R = sys.argv[1:5]
AGE_H = float(sys.argv[5]) if len(sys.argv) > 5 else 7.0
shutil.rmtree(R, ignore_errors=True)
INP, WORK = f'{R}/input', f'{R}/working'
os.makedirs(WORK); os.makedirs(f'{INP}/datasets/eeshsaxena')
shutil.copytree(DS, f'{INP}/datasets/eeshsaxena/rsna-knee-cnxt-fleet')
ids = pd.read_csv(f'{COMP}/test.csv', dtype=str).StudyInstanceUID.tolist()
rng = np.random.default_rng(7)
par = pd.DataFrame(rng.uniform(0, 1, (len(ids), 12)), columns=TARGETS); par.insert(0, 'StudyInstanceUID', ids)
par = par.sample(frac=1, random_state=3).reset_index(drop=True)
par[TARGETS] = par[TARGETS].rank(pct=True)
par.to_csv(f'{WORK}/submission.csv', index=False)
nb = json.load(open(NB, encoding='utf-8'))
cells = [(c['metadata'].get('k2', '?'), ''.join(c['source'])) for c in nb['cells'][31:]]


def redirect(src):
    src = src.replace('/kaggle/input', INP).replace('/kaggle/working', WORK)
    bad = [l for l in src.splitlines() if '/kaggle/' in l and not l.lstrip().startswith('#')]
    assert not bad, bad[:3]
    return src


os.environ['RSNA_COMP_ROOT'] = COMP
_Popen = subprocess.Popen


def _popen(cmd, *a, env=None, **k):
    env = dict(env or os.environ); env['CUDA_VISIBLE_DEVICES'] = '-1'; env['OMP_NUM_THREADS'] = '2'
    env['PYTHONPATH'] = env.get('PYTHONPATH', '') + os.pathsep + SHIM; env['PYTHONWARNINGS'] = 'ignore'
    env['FLEET_ALLOW_CPU'] = '1'; env['PYTHONDONTWRITEBYTECODE'] = '1'
    return _Popen(cmd, *a, env=env, **k)


subprocess.Popen = _popen
g = {'__name__': '__main__', 'T0': time.time() - AGE_H * 3600, 'COMP': COMP, 'ROOT': COMP}
_cuda = (torch.cuda.is_available, torch.cuda.device_count, torch.cuda.empty_cache)
for name, src in cells:
    if name == 'k2_fleet_stage':
        torch.cuda.is_available = lambda: True; torch.cuda.device_count = lambda: 2; torch.cuda.empty_cache = lambda: None
    try:
        exec(compile(redirect(src), name, 'exec'), g)
    finally:
        torch.cuda.is_available, torch.cuda.device_count, torch.cuda.empty_cache = _cuda
res, info = k2_check.check(WORK, 'clean', 'none')
for n, c, d in res:
    print(f'{"PASS" if c else "FAIL"}  {n}  {d}')
rec = g.get('_k2_receipt') or {}
print('RESULT', json.dumps(dict(ok=all(c for _, c, _ in res), passed=sum(c for _, c, _ in res), checks=len(res),
                                why=str(rec.get('why'))[:200], n_keep=rec.get('n_keep'), models=rec.get('models'),
                                names=rec.get('names'), manifest=rec.get('manifest'), work=WORK), default=str))
