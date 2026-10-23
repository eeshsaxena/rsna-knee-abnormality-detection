# t_gt2_saveep: (a) --save_epochs never changes training: the same TTR fold-0 run (2 epochs) with and without
# --save_epochs 1-2 gives identical top-level ckpt tensors, cv, epoch and OOF bytes; the args differ ONLY in
# 'save_epochs'. (b) gt2_read.py on the synthetic runs (ref = t_b2's default-path fold-0 run, ttr = t_b2_ttr_train's
# TTR fold-0 run): exit 0 (both closing lines present), score == macro AUC recomputed here from the best OOF files,
# verdict consistent with the PASS/KILL/MIDDLE rule; a log without the closing line -> exit 3 (PROVISIONAL).
import os, sys, json, glob, shutil, subprocess, hashlib
sys.dont_write_bytecode = True
from _paths import CODE, DRY, OUT, PY
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__))
C64 = f'{OUT}/code64_new'
env = dict(os.environ, PYTHONWARNINGS='ignore', PYTHONDONTWRITEBYTECODE='1')
BASE = ['--store', f'{DRY}/store_64_normal', '--labels', f'{DRY}/fake_labels_no77.csv', '--folds', f'{DRY}/folds5.csv',
        '--exclude', f'{DRY}/fake_gold.csv', '--gold', f'{DRY}/fake_gold.csv', '--bs', '1', '--accum', '1', '--workers', '0',
        '--device', 'cpu', '--seed', '42', '--backbone', 'convnext_atto.d2_in1k', '--model', 'ttr', '--aug', 'heavy',
        '--train_slices', '16', '--ema', '0.9995', '--fold', '0', '--eval_labels', f'{DRY}/fake_v4.csv', '--epochs', '2']
dirs = {}
for k, extra in (('plain', []), ('saveep', ['--save_epochs', '1-2'])):
    d = f'{OUT}/gt2_{k}'; shutil.rmtree(d, ignore_errors=True)
    r = subprocess.run([PY, '-W', 'ignore', 'run_trainer.py', C64] + BASE + extra + ['--out', d], capture_output=True,
                       text=True, env=env, cwd=HERE)
    assert r.returncode == 0, r.stderr[-1500:]
    dirs[k] = d
top = lambda d: sorted(f for f in os.listdir(d) if f != 'epochs')
assert top(dirs['plain']) == top(dirs['saveep'])
for f in top(dirs['plain']):
    a, b = f'{dirs["plain"]}/{f}', f'{dirs["saveep"]}/{f}'
    if f.endswith('.pt'):
        ca, cb = torch.load(a, weights_only=False), torch.load(b, weights_only=False)
        assert all(torch.equal(ca['model'][q], cb['model'][q]) for q in ca['model']) and ca['cv'] == cb['cv'] and ca['epoch'] == cb['epoch']
        diff = {q for q in set(ca['args']) | set(cb['args']) if ca['args'].get(q) != cb['args'].get(q)}
        assert diff == {'save_epochs'}, diff
    else:
        assert open(a, 'rb').read() == open(b, 'rb').read(), f
print(f'(a) --save_epochs 1-2 vs none: {top(dirs["plain"])} identical tensors/cv/epoch/OOF bytes; args differ only in save_epochs')

ref, ttr = f'{OUT}/b2_f0_new', f'{OUT}/ttr_f0'
r = subprocess.run([PY, '-W', 'ignore', f'{CODE}/gt2_read.py', '--ttr', ttr, '--ref', ref, '--eval_labels', f'{DRY}/fake_v4.csv',
                    '--json', f'{OUT}/gt2.json'], capture_output=True, text=True, env=env)
print(r.stdout[-1800:])
assert r.returncode == 0, r.stderr[-1500:]
js = json.load(open(f'{OUT}/gt2.json'))
ev = pd.read_csv(f'{DRY}/fake_v4.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
TG = list(ev.columns)


def best(d):
    v = []
    for t in ('raw', 'ema'):
        o = pd.read_csv(glob.glob(f'{d}/oof_f0_s*_{t}.csv')[0], dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
        Y = ev.loc[o.index, TG].to_numpy(float); P = o[TG].to_numpy(float); a = []
        for j in range(12):
            ok = np.isfinite(Y[:, j]); yb = Y[ok, j] >= .5
            if ok.sum() and 0 < yb.mean() < 1:
                a.append(roc_auc_score(yb, P[ok, j]))
        v.append(np.mean(a))
    return max(v)


d = best(ttr) - best(ref)
assert abs(js['delta'] - d) < 1e-12, (js['delta'], d)
g_ok = not np.isfinite(js['gold58']).all() or js['gold58'][0] >= js['gold58'][1]
exp = 'KILL' if d <= 0 else ('PASS' if d >= 0.008 and js['worst_label'][0] >= -0.02 and g_ok
                             else 'MIDDLE')
assert js['verdict'] == exp and js['final'] is True, (js['verdict'], exp)
lg = f'{OUT}/gt2_prov.log'
open(lg, 'w').write(''.join(l for l in open(f'{ttr}.log') if not l.startswith('{"raw"')))
r2 = subprocess.run([PY, '-W', 'ignore', f'{CODE}/gt2_read.py', '--ttr', ttr, '--ttr_log', lg, '--ref', ref, '--eval_labels',
                     f'{DRY}/fake_v4.csv'], capture_output=True, text=True, env=env)
assert r2.returncode == 3 and 'PROVISIONAL' in r2.stdout, r2.stdout[-500:]
print(f'(b) gt2_read: delta {d:+.4f} == recomputed, verdict {js["verdict"]} == rule; no closing line -> exit 3 PROVISIONAL')
print('T_GT2_SAVEEP OK')
