#!/usr/bin/env python3
"""t_k11_pick.py (box, CPU, read-only on ~/fleet_runs; writes only ~/k11_test/pickt; 2026-10-09, Claude): K11-MREC picker
tests on ONE real placed E2 fold. A temp root holds symlinks to the real run dir + log and a (possibly edited) COPY of its
Modal receipt; the picker (k11/box/k5_pick.py, no --copy) must PICK with the real receipt and FAIL, naming the cause, for:
no receipt, verify != OK, a wrong closing line, a receipt that does not list the picked file, a wrong run name, epoch_oof
!= 32. Also: the same fold without --modal_receipt_folds falls back to the box mtime rule (reported, not a crash).
  ~/miniconda3/envs/kmimic/bin/python ~/k11_test/k11/box/t_k11_pick.py F          last line: TK11PICK PASS|FAIL n/m
"""
import json, os, shutil, subprocess, sys

H = os.path.expanduser('~')
F = int(sys.argv[1])
RUN = f'f{F}_s1046_ttr288frac75'
SRC = f'{H}/fleet_runs'
PICK = f'{H}/k11_test/k11/box/k5_pick.py'
PY = sys.executable
res = []


def ok(name, cond, detail=''):
    res.append(bool(cond))
    print(f'{"PASS" if cond else "FAIL"}  {name}  {str(detail)[:240]}', flush=True)


def run(root, folds_arg=True):
    cmd = [PY, '-W', 'ignore', PICK, '--root', root, '--seed', '1046', '--suffix', 'ttr288frac75', '--folds', str(F),
           '--ttr_size', '288', '--train_labels', 'labels_frac75_logit.csv', '--key_tag', 't288f75', '--no_kaggle_receipt',
           '--tag', 'raw', '--min_epochs', '16'] + (['--modal_receipt_folds', str(F)] if folds_arg else [])
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2'))
    return r.returncode, r.stdout + r.stderr


def root_with(edit, name='r'):
    d = f'{H}/k11_test/pickt/{name}'
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    os.symlink(f'{SRC}/{RUN}', f'{d}/{RUN}'); os.symlink(f'{SRC}/{RUN}.log', f'{d}/{RUN}.log')
    rec = json.load(open(f'{SRC}/modal_e2_{RUN}_receipt.json'))
    if edit is not None:
        edit(rec)
        json.dump(rec, open(f'{d}/modal_e2_{RUN}_receipt.json', 'w'), indent=1)
    return d


rc, out = run(root_with(lambda r: None, 'good'))
ok('real receipt: PICK, source modal, Modal-filled line', rc == 0 and f'PICK fold {F}' in out and 'Modal-filled run' in out, out[-240:])
cases = [('no receipt', None, 'no Modal receipt'),
         ('verify != OK', lambda r: r.update(verify=['md5:x']), 'verify'),
         ('closing line differs', lambda r: r.update(closing_line='{"raw": 0.1, "ema": 0.1}'), 'closing_line'),
         ('picked file not listed', lambda r: r.update(ckpts=[dict(c, file='other.pt') for c in r['ckpts']]), 'does not list'),
         ('args not OK', lambda r: r['ckpts'][0].update(args={'seed': 1}), 'does not list'),
         ('wrong run name', lambda r: r.update(run='f9_s1046_ttr288frac75'), 'run '),
         ('epoch_oof 31', lambda r: r.update(epoch_oof=31), 'epoch_oof')]
for i, (name, edit, needle) in enumerate(cases):
    rc, out = run(root_with(edit, f'n{i}'))
    ok(f'negative "{name}": FAIL naming the cause', rc == 1 and f'FAIL fold {F}' in out and needle in out, out[-240:])
rc, out = run(root_with(lambda r: None, 'nofold'), folds_arg=False)
ok('without --modal_receipt_folds: the box mtime rule applies (a copied run fails it or passes it, never crashes)',
   rc in (0, 1) and 'Traceback' not in out, out[-240:])
shutil.rmtree(f'{H}/k11_test/pickt', ignore_errors=True)
n = sum(res)
print(f'TK11PICK {"PASS" if n == len(res) else "FAIL"} {n}/{len(res)}')
