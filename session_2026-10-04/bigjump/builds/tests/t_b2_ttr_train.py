# t_b2_ttr_train: the TTR recipe end to end through the bigjump trainer (round3 64 px store, convnext_atto, CPU),
# then serving through the bigjump worker. Runs:
#   ttr_f0   --model ttr --aug heavy --train_slices 16 --ema 0.9995 --epochs 3 --save_epochs 2-3, fold 0
#   ttr_all  --model ttr --aug heavy --train_slices 16 --fold -1 --epochs 3 --save_epochs 1-3
#   f1_save  the family-1 model with only --save_epochs 1-2 (opt-in on the default model)
#   guards   --model ttr --f2 1, --save_epochs 3-1, --model ttr --seed 42 (BJ-5) must refuse
# TTR runs use --seed 1042 (BJ-5: TTR seeds 1000-1999). TTR-6: both ckpts carry ttr_size_t == 224 (int64; the EMA
# copy never averages it) and a TTR state_dict does NOT strict-load into a plain KneeFleetNet.
# Checks: file names (top level unchanged + epochs/), ckpt args (model_mod, ttr_size, aug, aug_cfg, drop_path .2,
# dropout .3, ema, train_slices, model_sha == the worker's TTR sha), per-epoch ckpt cv == that epoch's logged val AUC,
# the top-level best/last file == the epochs/ snapshot of the same epoch (tensors), the trainer's OOF of the best epoch ==
# the epochs/ OOF of that epoch (bytes).
# Serving (64 px fake competition comp_big, 24 studies, CPU, harness relabels the CPU-trained ckpts fp16):
#   bigjump pack (--tag f0_s1042ttr re-keys the TTR fold-0 member; the others keep their own keys) ->
#   bigjump worker serves [TTR f0, TTR all-data ep2 snapshot, F1 f0, F1 all-data]: every row finite, TTR members
#   built as KneeFleetNetTTR; the F1 members' predictions == the K2 worker's (k2/code, what K2 ships) on an F1-only
#   dataset packed by the K2 pack tool (max |diff| 0, same NaN pattern); the K2 worker REFUSES a TTR member (rc != 0:
#   the K2 cell then restores the parent, never mis-serves).
import os, re, sys, json, glob, shutil, subprocess, hashlib
from _paths import CODE, K2C, DRY, OUT, PY, SHIM, stage64, relabel_fp16
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__))
C64 = stage64(CODE, f'{OUT}/code64_new', ['fleet_common.py', 'fleet_model.py', 'fleet_model_f2.py', 'fleet_model_ttr.py',
                                         'fleet_aug.py', 'fleet_train.py', 'fleet_worker.py', 'pack_fleet_ds.py', 'tta_bench.py'])
K64 = stage64(K2C, f'{OUT}/code64_k2', ['fleet_common.py', 'fleet_model.py', 'fleet_model_f2.py', 'fleet_worker.py',
                                       'pack_fleet_ds.py'])
BASE = ['--store', f'{DRY}/store_64_normal', '--labels', f'{DRY}/fake_labels_no77.csv', '--folds', f'{DRY}/folds5.csv',
        '--exclude', f'{DRY}/fake_gold.csv', '--gold', f'{DRY}/fake_gold.csv', '--bs', '1', '--accum', '1',
        '--workers', '0', '--device', 'cpu', '--seed', '42', '--backbone', 'convnext_atto.d2_in1k']
TTR = ['--model', 'ttr', '--aug', 'heavy', '--train_slices', '16', '--ema', '0.9995', '--seed', '1042']   # BJ-5
F0 = ['--fold', '0', '--eval_labels', f'{DRY}/fake_v4.csv']
env = dict(os.environ, PYTHONWARNINGS='ignore', PYTHONDONTWRITEBYTECODE='1')
sys.path.insert(0, C64)


def train(name, argv, ok=True):
    d = f'{OUT}/ttr_{name}'
    shutil.rmtree(d, ignore_errors=True)
    r = subprocess.run([PY, '-W', 'ignore', 'run_trainer.py', C64] + BASE + argv + ['--out', d],
                       capture_output=True, text=True, env=env, cwd=HERE)
    open(f'{d}.log', 'w').write(r.stdout + r.stderr)
    assert (r.returncode == 0) == ok, (name, r.returncode, r.stderr[-1500:])
    return d, r.stdout, r.stderr


def wsha(mod):
    import ast
    tree = ast.parse(open(f'{C64}/fleet_worker.py', encoding='utf-8').read())
    node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_model_sha'][0]
    ns = {}; exec(compile(ast.Module(body=[node], type_ignores=[]), 'w', 'exec'), ns)
    return ns['_model_sha'](mod)


TTR_SHA = wsha('fleet_model_ttr')
res = {}
# ---------------- ttr_f0
d, out, _ = train('f0', TTR + F0 + ['--epochs', '3', '--save_epochs', '2-3'])
assert 'BJ opt-in: model ttr (ttr_size 224) | aug heavy | drop_path 0.2 dropout 0.3 | train_slices 16 | ema 0.9995' in out, out[:600]
top = sorted(os.listdir(d)); ep = sorted(os.listdir(f'{d}/epochs'))
assert [f for f in top if f.endswith('.pt')] and all(re.fullmatch(r'f0_s1042_(raw|ema)_ttr224-64x24_cv\d+_best\.pt', f)
                                                    for f in top if f.endswith('.pt')), top
assert ep == sorted([f'f0_s1042_{t}_ttr224-64x24_ep{e}_best.pt' for t in ('raw', 'ema') for e in (2, 3)]
                    + [f'oof_f0_s1042_{t}_ep{e}.csv' for t in ('raw', 'ema') for e in (2, 3)]), ep
vals = {(int(m[1]) + 1, m[2]): float(m[3]) for m in re.finditer(r'^ep (\d+) (raw|ema) .*?val AUC vs \S+ ([\d.]+)', out, re.M)}
for t in ('raw', 'ema'):
    for e in (2, 3):
        ck = torch.load(f'{d}/epochs/f0_s1042_{t}_ttr224-64x24_ep{e}_best.pt', weights_only=False)
        assert ck['epoch'] == e - 1 and abs(ck['cv'] - vals[(e, t)]) < 6e-5, (t, e, ck['cv'], vals[(e, t)])
        a = ck['args']
    best = torch.load(glob.glob(f'{d}/f0_s1042_{t}_ttr224-*_best.pt')[0], weights_only=False)
    tb = best['model']['ttr_size_t']                                          # TTR-6
    assert tb.dtype == torch.int64 and int(tb) == 224, (t, tb)
    be = best['epoch'] + 1
    if be >= 2:                                      # best epoch inside the saved range: same tensors, same OOF bytes
        snap = torch.load(f'{d}/epochs/f0_s1042_{t}_ttr224-64x24_ep{be}_best.pt', weights_only=False)
        assert all(torch.equal(best['model'][k], snap['model'][k]) for k in best['model'])
        assert open(f'{d}/oof_f0_s1042_{t}.csv', 'rb').read() == open(f'{d}/epochs/oof_f0_s1042_{t}_ep{be}.csv', 'rb').read()
    res[f'f0_{t}_best_epoch'] = be
want = dict(model_mod='fleet_model_ttr', ttr_size=224, aug='heavy', drop_path=0.2, dropout=0.3, ema=0.9995,
            train_slices=16, model_sha=TTR_SHA, save_epochs='2-3', mirror_aug=0.5, epochs=3, fold=0, seed=1042)
bad = {k: (a.get(k), v) for k, v in want.items() if a.get(k) != v}
assert not bad and a['aug_cfg']['rot'] == 15.0, bad
print(f'[ttr_f0] top {top}\n[ttr_f0] epochs/ {ep}\n[ttr_f0] args ok: ' + ', '.join(f'{k}={a[k]}' for k in want)
      + f'; per-epoch cv == log; best epochs {res}')
# ---------------- ttr_all
d2, out2, _ = train('all', TTR + ['--fold', '-1', '--epochs', '3', '--save_epochs', '1-3'])
ep2 = sorted(os.listdir(f'{d2}/epochs'))
assert ep2 == sorted(f'f-1_s1042_{t}_ttr224-64x24_ep{e}_best.pt' for t in ('raw', 'ema') for e in (1, 2, 3)), ep2
for t in ('raw', 'ema'):
    a1 = torch.load(f'{d2}/f-1_s1042_{t}_ttr224-64x24_ep3_best.pt', weights_only=False)
    a2 = torch.load(f'{d2}/epochs/f-1_s1042_{t}_ttr224-64x24_ep3_best.pt', weights_only=False)
    assert all(torch.equal(a1['model'][k], a2['model'][k]) for k in a1['model']) and a1['args'] == a2['args']
    e1 = torch.load(f'{d2}/epochs/f-1_s1042_{t}_ttr224-64x24_ep1_best.pt', weights_only=False)
    assert not all(torch.equal(a1['model'][k], e1['model'][k]) for k in a1['model']), 'ep1 == ep3?'
print(f'[ttr_all] epochs/ {ep2}; top-level ep3 == epochs/ep3 (tensors, args); ep1 differs')
# ---------------- f1 + save_epochs only
d3, out3, _ = train('f1save', F0 + ['--epochs', '2', '--save_epochs', '1-2'])
a3 = torch.load(glob.glob(f'{d3}/epochs/f0_s42_ema_cnxt64x24_ep1_best.pt')[0], weights_only=False)['args']
assert a3.get('model_mod') is None and a3['model_sha'] == wsha('fleet_model') and a3['save_epochs'] == '1-2' \
    and a3['drop_path'] == 0.1 and a3['dropout'] == 0.2 and a3['aug'] == 'light'
print(f'[f1_save] family-1 model + --save_epochs: {sorted(os.listdir(d3 + "/epochs"))}; model_sha family 1, drop_path .1 dropout .2')
# ---------------- guards
_, _, e1 = train('guard_f2', TTR + F0 + ['--f2', '1', '--seed', '142', '--epochs', '1'], ok=False)
_, _, e2 = train('guard_save', F0 + ['--epochs', '1', '--save_epochs', '3-1'], ok=False)
_, _, e3 = train('guard_seed', ['--model', 'ttr'] + F0 + ['--epochs', '1'], ok=False)          # BJ-5: BASE seed 42
assert 'not with --f2 1' in e1 and 'want a-b' in e2 and 'seeds 1000-1999' in e3
print('[guards] --model ttr --f2 1 refused; --save_epochs 3-1 refused; --model ttr --seed 42 refused (BJ-5)')
# ---------------- TTR-6: a TTR state_dict does not strict-load into a plain KneeFleetNet
from fleet_model import KneeFleetNet
sd = torch.load(glob.glob(f'{d}/f0_s1042_ema_ttr224-*_best.pt')[0], weights_only=False)
ka = sd['args']
plain = KneeFleetNet(backbone=ka['backbone'], n_views=ka['n_views'], n_slices=ka['n_slices'], dim=ka['dim'],
                     n_contrast=ka['n_contrast'], contrast_emb=ka['contrast_emb'], pretrained=False, drop_path=0.0)
try:
    plain.load_state_dict(sd['model']); refused = False
except RuntimeError as ex:
    refused = 'ttr_size_t' in str(ex)
miss, unexp = plain.load_state_dict(sd['model'], strict=False)
assert refused and not miss and unexp == ['ttr_size_t'], (refused, miss, unexp)
print(f'[TTR-6] TTR ckpt into KneeFleetNet: strict load refused (unexpected {unexp}); raw/ema ttr_size_t int64 224')

# ---------------- serving
MEM = f'{OUT}/ttr_members'
shutil.rmtree(MEM, ignore_errors=True)
ttr_f0 = relabel_fp16(glob.glob(f'{d}/f0_s1042_ema_ttr224-*_best.pt')[0], f'{MEM}/a/{os.path.basename(glob.glob(f"{d}/f0_s1042_ema_ttr224-*_best.pt")[0])}')
ttr_all = relabel_fp16(f'{d2}/epochs/f-1_s1042_ema_ttr224-64x24_ep2_best.pt', f'{MEM}/a/f-1_s1042_ema_ttr224-64x24_ep2_best.pt')
f1_f0 = relabel_fp16(glob.glob(f'{OUT}/b2_f0_new/f0_s42_ema_*.pt')[0], f'{MEM}/b/{os.path.basename(glob.glob(f"{OUT}/b2_f0_new/f0_s42_ema_*.pt")[0])}')
f1_all = relabel_fp16(glob.glob(f'{OUT}/b2_all_new/f-1_s42_ema_*.pt')[0], f'{MEM}/b/{os.path.basename(glob.glob(f"{OUT}/b2_all_new/f-1_s42_ema_*.pt")[0])}')


def run(cmd, e=None):
    r = subprocess.run([PY, '-W', 'ignore'] + cmd, capture_output=True, text=True, env=e or env)
    return r.returncode, r.stdout + r.stderr


def worker(ds, npz):
    e = dict(env, FLEET_DIR=ds, FLEET_OUT=npz, RSNA_COMP_ROOT=f'{DRY}/comp_big', FLEET_ALLOW_CPU='1',
             CUDA_VISIBLE_DEVICES='-1', FLEET_SHARD='0', FLEET_NSHARD='1', OMP_NUM_THREADS='2', FLEET_MICROBENCH='0',
             PYTHONPATH=SHIM + os.pathsep + env.get('PYTHONPATH', ''))
    return run([f'{ds}/fleet_worker.py'], e)


DS_MIX, DS_F1K2, DS_BAD = f'{OUT}/ttr_ds_mixed', f'{OUT}/ttr_ds_f1_k2', f'{OUT}/ttr_ds_ttr_in_k2'
rc, log = run([f'{C64}/pack_fleet_ds.py', '--out', DS_MIX, '--store', f'{DRY}/store_64_normal',
               '--tag', 'f0_s1042ttr,=,=,=', ttr_f0, ttr_all, f1_f0, f1_all])
assert rc == 0, log[-1500:]
print('[pack] ' + '\n[pack] '.join(l[:200] for l in log.splitlines() if l.startswith('member')))
man = json.load(open(f'{DS_MIX}/fleet_manifest.json'))
assert man['checkpoints'][0].startswith('f0_s1042ttr_ema_ttr224-') and 'fleet_model_ttr.py' in os.listdir(DS_MIX)
assert man['checkpoints'][1].startswith('f-1_s1042_ema_ttr224-')
assert all(set(v) == {'original', 'run_dir', 'sha256', 'model_mod', 'model_sha'} for v in man['sources'].values()), man['sources']   # FIX-5
rc, log = worker(DS_MIX, f'{OUT}/ttr_P_mixed.npz'); open(f'{OUT}/ttr_worker_mixed.log', 'w').write(log)
assert rc == 0, log[-2500:]
z = np.load(f'{OUT}/ttr_P_mixed.npz'); names = [str(x) for x in z['names']]
mline = [l for l in log.splitlines() if 'members:' in l][0]
assert mline.count('KneeFleetNetTTR') == 2 and mline.count('[KneeFleetNet]') == 2, mline
ok_rows = z['NV'] > 0
assert ok_rows.sum() >= 20 and np.isfinite(z['P'][:, ok_rows]).all() and int(z['n_keep']) == 4
print(f'[worker] bigjump worker on {len(z["sids"])} studies: {mline.split("members: ")[1][:300]}; n_keep {int(z["n_keep"])}; '
      f'rows finite {int(ok_rows.sum())}')
rc, log = run([f'{K64}/pack_fleet_ds.py', '--out', DS_F1K2, '--store', f'{DRY}/store_64_normal', f1_f0, f1_all])
assert rc == 0, log[-800:]
rc, log = worker(DS_F1K2, f'{OUT}/ttr_P_f1_k2.npz'); assert rc == 0, log[-2000:]
zk = np.load(f'{OUT}/ttr_P_f1_k2.npz'); nk = [str(x) for x in zk['names']]
assert list(zk['sids']) == list(z['sids'])
for n in nk:
    dd = float(np.nanmax(np.abs(z['P'][names.index(n)] - zk['P'][nk.index(n)])))
    same_nan = bool((np.isnan(z['P'][names.index(n)]) == np.isnan(zk['P'][nk.index(n)])).all())
    print(f'[worker] family-1 member {n}: bigjump worker vs K2 worker max|diff| {dd} (NaN pattern equal {same_nan})')
    assert dd == 0.0 and same_nan
    res[f'f1_vs_k2_{n}'] = dd
shutil.rmtree(DS_BAD, ignore_errors=True); shutil.copytree(DS_MIX, DS_BAD)
shutil.copy2(f'{K64}/fleet_worker.py', f'{DS_BAD}/fleet_worker.py')
rc, log = worker(DS_BAD, f'{OUT}/ttr_P_bad.npz')
why = [l for l in log.splitlines() if 'AssertionError' in l or 'unknown model_mod' in l]
print(f'[worker] K2 worker on a TTR member: rc {rc} | {why[-1][:160] if why else log[-200:]}')
assert rc != 0 and why
res.update(ttr_sha=TTR_SHA, names=names)
json.dump(res, open(f'{OUT}/t_b2_ttr_train.json', 'w'), indent=1)
print('T_B2_TTR_TRAIN OK')
