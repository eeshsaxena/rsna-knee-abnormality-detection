# box_parity.py (box, CPU): B2 default-path parity in the BOX environment (torch 2.5.1, timm 1.0.29, real cv2 4.14):
# the bigjump trainer with no opt-in flag vs byte copies of ~/fleet/fleet_train.py (family 1, running) and
# ~/fleet2_code/fleet_train.py, same argv, round3 64 px synthetic store (copied to ~/bigjump_build/dry), convnext_atto
# (resnet18 for --f2), timm pretrained -> False (harness). Compared: ckpt names, every state_dict tensor (torch.equal),
# ckpt args, cv, epoch, OOF bytes, metric log lines. Also the TTR path runs (1 epoch) in this env.
import os, re, sys, json, shutil, subprocess, hashlib
H = os.path.expanduser('~'); BJ = f'{H}/bigjump_build'
OUT = f'{BJ}/out/parity'; DRY = f'{BJ}/dry'
os.makedirs(OUT, exist_ok=True)
import torch
RUNNER = f'{OUT}/run_trainer.py'
open(RUNNER, 'w').write('''import os, sys, runpy
import torch, timm
torch.set_num_threads(4)
_cm = timm.create_model
def _create_model(*a, **k):
    k['pretrained'] = False
    return _cm(*a, **k)
timm.create_model = _create_model
code = sys.argv[1]
sys.argv = [f'{code}/fleet_train.py'] + sys.argv[2:]
runpy.run_path(f'{code}/fleet_train.py', run_name='__main__')
''')


def stage64(src, dst, files):
    shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
    for f in files:
        s = open(f'{src}/{f}', encoding='utf-8', newline='').read()
        if f == 'fleet_common.py':
            assert s.count('STORE = 320 ') == 1
            s = s.replace('STORE = 320 ', 'STORE = 64 ')
        open(f'{dst}/{f}', 'w', encoding='utf-8', newline='').write(s)
    return dst


C = {'orig': stage64(f'{BJ}/ref_fleet', f'{OUT}/code64_orig', ['fleet_common.py', 'fleet_model.py', 'fleet_train.py']),
     'f2': stage64(f'{BJ}/ref_fleet2', f'{OUT}/code64_f2', ['fleet_common.py', 'fleet_model.py', 'fleet_model_f2.py', 'fleet_train.py']),
     'new': stage64(f'{BJ}/code', f'{OUT}/code64_new', ['fleet_common.py', 'fleet_model.py', 'fleet_model_f2.py',
                                                        'fleet_model_ttr.py', 'fleet_aug.py', 'fleet_train.py'])}
BASE = ['--store', f'{DRY}/store_64_normal', '--labels', f'{DRY}/fake_labels_no77.csv', '--folds', f'{DRY}/folds5.csv',
        '--exclude', f'{DRY}/fake_gold.csv', '--gold', f'{DRY}/fake_gold.csv', '--bs', '1', '--accum', '1',
        '--workers', '0', '--device', 'cpu', '--seed', '42']
AT = ['--backbone', 'convnext_atto.d2_in1k']
RUNS = {'f0': (['orig', 'f2'], AT + ['--fold', '0', '--eval_labels', f'{DRY}/fake_v4.csv', '--epochs', '2']),
        'all': (['orig', 'f2'], AT + ['--fold', '-1', '--epochs', '3', '--stop_epoch', '1']),
        't6': (['orig', 'f2'], AT + ['--fold', '0', '--eval_labels', f'{DRY}/fake_v4.csv', '--epochs', '1', '--train_slices', '16']),
        'f2r18': (['f2'], ['--f2', '1', '--backbone', 'resnet18.a1_in1k', '--seed', '142', '--lr', '3e-4', '--fold', '0',
                           '--eval_labels', f'{DRY}/fake_v4.csv', '--epochs', '1'])}
METRIC = re.compile(r'(val AUC|gold74|^labels|^\{|^ep \d+ (raw|ema))')
env = dict(os.environ, PYTHONWARNINGS='ignore', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')


def train(tag, run, argv):
    d = f'{OUT}/{run}_{tag}'
    shutil.rmtree(d, ignore_errors=True)
    r = subprocess.run([sys.executable, '-W', 'ignore', RUNNER, C[tag]] + BASE + argv + ['--out', d],
                       capture_output=True, text=True, env=env)
    open(f'{d}.log', 'w').write(r.stdout + r.stderr)
    assert r.returncode == 0, (run, tag, r.stderr[-2000:])
    return d, [re.sub(r'\| epoch [0-9.]+ min', '', l) for l in r.stdout.splitlines() if METRIC.search(l)], r.stdout


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
            assert a['args'] == b['args'], f
            assert str(a['cv']) == str(b['cv']) and a['epoch'] == b['epoch'], f
        else:
            assert hashlib.sha256(open(p1, 'rb').read()).digest() == hashlib.sha256(open(p2, 'rb').read()).digest(), f
    return f1, n_t


for run, (refs, argv) in RUNS.items():
    dn, ln, outn = train('new', run, argv)
    assert 'BJ opt-in' not in outn
    for ref in refs:
        dr, lr, _ = train(ref, run, argv)
        files, n_t = compare(dr, dn)
        assert lr == ln, (lr, ln)
        print(f'PASS [{run} vs {ref}] {files}: {n_t} tensors torch.equal, args/cv/epoch ==, OOF bytes ==, {len(ln)} metric lines ==',
              flush=True)
d, _, out = train('new', 'ttr', AT + ['--fold', '0', '--eval_labels', f'{DRY}/fake_v4.csv', '--epochs', '1', '--model', 'ttr',
                                      '--aug', 'heavy', '--train_slices', '16', '--ema', '0.9995', '--save_epochs', '1-1',
                                      '--seed', '1042'])                                    # BJ-5: TTR seeds 1000-1999
print('PASS [ttr path in the box env] ' + [l for l in out.splitlines() if l.startswith('BJ opt-in')][0][:200] + ' | files '
      + str(sorted(os.listdir(d))) + ' ' + str(sorted(os.listdir(d + '/epochs'))))
print('BOXPARITY PASS')
