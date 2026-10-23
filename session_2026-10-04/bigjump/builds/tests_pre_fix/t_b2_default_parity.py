# t_b2_default_parity: the bigjump trainer with NO opt-in flag is the running trainers bit for bit, end to end.
# Same argv, separate processes, round3 64 px synthetic store (6 studies), convnext_atto (resnet18 for the --f2 case):
#   vs ORIG (round3/fleet == box ~/fleet, family 1, md5-checked):  fold-0 (2 ep, eval_labels + gold), all-data
#        (--fold -1 --epochs 3 --stop_epoch 1), fold-0 T6 budget variant (--train_slices 16, light aug)
#   vs F2   (fleet2/code == box ~/fleet2_code, md5-checked):        the same three + a --f2 1 run (resnet18, seed 142)
# Compared: ckpt file names, every state_dict tensor (torch.equal), ckpt args dict (==), cv, epoch, OOF csv bytes, and
# the metric lines of the logs (timings stripped).
import os, re, sys, json, subprocess, hashlib
from _paths import CODE, ORIG, F2, DRY, OUT, PY, stage64
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
C = {'orig': stage64(ORIG, f'{OUT}/code64_orig', ['fleet_common.py', 'fleet_model.py', 'fleet_train.py']),
     'f2': stage64(F2, f'{OUT}/code64_f2', ['fleet_common.py', 'fleet_model.py', 'fleet_model_f2.py', 'fleet_train.py']),
     'new': stage64(CODE, f'{OUT}/code64_new', ['fleet_common.py', 'fleet_model.py', 'fleet_model_f2.py',
                                               'fleet_model_ttr.py', 'fleet_aug.py', 'fleet_train.py'])}
BASE = ['--store', f'{DRY}/store_64_normal', '--labels', f'{DRY}/fake_labels_no77.csv', '--folds', f'{DRY}/folds5.csv',
        '--exclude', f'{DRY}/fake_gold.csv', '--gold', f'{DRY}/fake_gold.csv', '--bs', '1', '--accum', '1',
        '--workers', '0', '--device', 'cpu', '--seed', '42']
AT = ['--backbone', 'convnext_atto.d2_in1k']
RUNS = {'f0': (['orig', 'f2'], AT + ['--fold', '0', '--eval_labels', f'{DRY}/fake_v4.csv', '--epochs', '2']),
        'all': (['orig', 'f2'], AT + ['--fold', '-1', '--epochs', '3', '--stop_epoch', '1']),
        't6': (['orig', 'f2'], AT + ['--fold', '0', '--eval_labels', f'{DRY}/fake_v4.csv', '--epochs', '1',
                                     '--train_slices', '16']),
        'f2r18': (['f2'], ['--f2', '1', '--backbone', 'resnet18.a1_in1k', '--seed', '142', '--lr', '3e-4', '--fold', '0',
                           '--eval_labels', f'{DRY}/fake_v4.csv', '--epochs', '1'])}
METRIC = re.compile(r'(val AUC|gold74|^labels|^\{|^ep \d+ (raw|ema))')
env = dict(os.environ, PYTHONWARNINGS='ignore', PYTHONDONTWRITEBYTECODE='1')


def train(tag, run, argv):
    d = f'{OUT}/b2_{run}_{tag}'
    r = subprocess.run([PY, '-W', 'ignore', 'run_trainer.py', C[tag]] + BASE + argv + ['--out', d],
                       capture_output=True, text=True, env=env, cwd=HERE)
    open(f'{d}.log', 'w').write(r.stdout + r.stderr)
    assert r.returncode == 0, (run, tag, r.stderr[-2000:])
    lines = [re.sub(r'\| epoch [0-9.]+ min', '', l) for l in r.stdout.splitlines() if METRIC.search(l)]
    return d, lines, r.stdout


def compare(d1, d2):
    f1, f2 = sorted(os.listdir(d1)), sorted(os.listdir(d2))
    assert f1 == f2, (f1, f2)
    n_t = 0
    for f in f1:
        p1, p2 = f'{d1}/{f}', f'{d2}/{f}'
        if f.endswith('.pt'):
            a, b = torch.load(p1, weights_only=False), torch.load(p2, weights_only=False)
            assert list(a['model']) == list(b['model'])
            for k in a['model']:
                n_t += 1
                assert torch.equal(a['model'][k], b['model'][k]), (f, k)
            assert a['args'] == b['args'], (f, {k: (a['args'].get(k), b['args'].get(k))
                                                for k in set(a['args']) | set(b['args']) if a['args'].get(k) != b['args'].get(k)})
            assert str(a['cv']) == str(b['cv']) and a['epoch'] == b['epoch'], f
        else:
            assert hashlib.sha256(open(p1, 'rb').read()).digest() == hashlib.sha256(open(p2, 'rb').read()).digest(), f
    return f1, n_t


res = {}
for run, (refs, argv) in RUNS.items():
    dn, ln, outn = train('new', run, argv)
    assert 'BJ opt-in' not in outn, 'default path printed the opt-in line'
    for ref in refs:
        dr, lr, _ = train(ref, run, argv)
        files, n_t = compare(dr, dn)
        assert lr == ln, (lr, ln)
        res[f'{run}_vs_{ref}'] = dict(files=files, tensors_equal=n_t, metric_lines=len(ln))
        print(f'[{run} vs {ref}] files {files}')
        print(f'[{run} vs {ref}] {n_t} state_dict tensors torch.equal; ckpt args == ; cv/epoch == ; OOF bytes == ; '
              f'{len(ln)} metric log lines ==')
    for l in ln[:6]:
        print('    ', l[:170])
json.dump(res, open(f'{OUT}/t_b2_default_parity.json', 'w'), indent=1)
print('T_B2_DEFAULT_PARITY OK: no opt-in flag -> bit-identical to round3/fleet (~/fleet) and fleet2/code (~/fleet2_code)')
