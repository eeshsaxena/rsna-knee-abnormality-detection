# dummy_train.py (kaggle_frac75c copy, unchanged from kaggle_frac75b): DRY-TEST stand-in for fleet_train.py (never shipped). Same CLI, same log line formats, same output
# names; no GPU, no model. Prints the REAL contract (imports the shipped fleet_common.py from its cwd) and the REAL
# fold counts of the store it is given. Behaviour knobs (env):
#   DUMMY_SPEED s/iteration (sleep) | DUMMY_PRINT_EVERY progress line period (real trainer: 200)
#   DUMMY_HANG_AFTER_EP  after this epoch's evals, sleep forever (watchdog test)
#   DUMMY_IGNORE_TERM_FOLDS  comma list of folds whose main process ignores SIGTERM (SIGKILL escalation test)
#   DUMMY_BAD  'contract' -> wrong tensor_sha in the contract line | 'counts' -> val count off by one
import os, sys, json, time, signal, argparse, subprocess, hashlib, inspect
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
for a in ('--store', '--labels', '--folds', '--exclude', '--eval_labels', '--gold', '--out', '--model', '--aug',
          '--save_epochs', '--backbone'):
    ap.add_argument(a, default='')
for a in ('--fold', '--seed', '--epochs', '--train_slices', '--workers'):
    ap.add_argument(a, type=int, default=0)
for a in ('--drop_path', '--dropout', '--ema'):
    ap.add_argument(a, type=float, default=0.0)
A = ap.parse_args()
if str(A.fold) in os.environ.get('DUMMY_IGNORE_TERM_FOLDS', '').split(','):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(100000)', A.out])   # a "DataLoader worker" (argv: out dir, for per-test stray checks)
sys.path.insert(0, os.getcwd())
import fleet_common as fc
SPEED = float(os.environ.get('DUMMY_SPEED', 0.05))
EVERY = int(os.environ.get('DUMMY_PRINT_EVERY', 200))
HANG = int(os.environ.get('DUMMY_HANG_AFTER_EP', -1))
BAD = os.environ.get('DUMMY_BAD', '')

lab = pd.read_csv(A.labels, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
folds = pd.read_csv(A.folds, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['fold']
excl = set(pd.read_csv(A.exclude, dtype={'StudyInstanceUID': str}).StudyInstanceUID)
held = set(folds.index[folds < 0]) | excl
have = {s for s in lab.index if os.path.exists(f'{A.store}/{s}.npz')}
ids = [s for s in lab.index if s in folds.index and s in have]
tr = [s for s in ids if s not in held and folds[s] != A.fold]
va = [s for s in ids if s not in held and folds[s] == A.fold]
nva = len(va) + (1 if BAD == 'counts' else 0)
print(f'labels {len(lab)} | store has {len(have)} | held-out (fold<0 or --exclude) {len(held)} | '
      f'train {len(tr)} | val {nva} | all_data False | device cuda', flush=True)
con = dict(fc.CONTRACT)
if BAD == 'contract':
    con['tensor_sha'] = '0000000000000000'
print('contract', con, '| select_sha', fc.SELECT_SHA, '| jpeg_fp', fc.jpeg_fingerprint(), flush=True)
import fleet_model as _fm, fleet_model_ttr as _tt
msha = hashlib.sha256((chr(10).join(inspect.getsource(_tt).splitlines()) + chr(10) +
                       chr(10).join(inspect.getsource(_fm).splitlines())).encode()).hexdigest()[:16]
print(f'BJ opt-in: model ttr (ttr_size 224) | aug heavy | drop_path {A.drop_path} dropout {A.dropout} | train_slices '
      f'{A.train_slices} | ema {A.ema} | epochs {A.epochs} | save_epochs {A.save_epochs} | name tag ttr224- | '
      f'model_sha {msha}', flush=True)

import torch
box = json.load(open(os.path.join(os.path.dirname(A.labels), 'box_facts.json')))
SD = {k: torch.zeros(s, dtype=getattr(torch, d)) for k, s, d in box['state_dict']}
args = dict(fold=A.fold, seed=A.seed, epochs=A.epochs, save_epochs=A.save_epochs, dummy=True,
            train_labels=os.path.basename(A.labels))
os.makedirs(A.out, exist_ok=True)
a_, _, b_ = A.save_epochs.partition('-')
SE = (int(a_), int(b_ or a_)) if A.save_epochs else None
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
           'Synovitis', "Baker's", 'Contusion', 'Fracture']


def save(tag, cv, ep):
    for q in [q for q in os.listdir(A.out) if q.startswith(f'f{A.fold}_s{A.seed}_{tag}_')]:
        os.remove(f'{A.out}/{q}')
    torch.save({'model': SD, 'args': args, 'cv': cv, 'epoch': ep},
               f'{A.out}/f{A.fold}_s{A.seed}_{tag}_ttr224-320x24_cv{int(cv * 1e4)}_best.pt')
    pd.DataFrame(np.random.rand(len(va), 12), columns=TARGETS, index=pd.Index(va, name='StudyInstanceUID')).to_csv(
        f'{A.out}/oof_f{A.fold}_s{A.seed}_{tag}.csv')


n_it = len(tr) // 2
best = {'raw': -1.0, 'ema': -1.0}
for ep in range(A.epochs):
    t0 = time.time()
    for it in range(n_it):
        time.sleep(SPEED)
        if it % EVERY == 0:
            print(f'ep {ep} it {it}/{n_it} loss 0.5000 {(time.time() - t0) / (it + 1):.2f}s/it', flush=True)
    print(f'ep {ep} TRAIN-ONLY {(time.time() - t0) / 60:.1f} min (G0 timing)', flush=True)
    for tag in ('raw', 'ema'):
        cv = 0.80 + 0.003 * ep + (0.001 if tag == 'raw' else 0.0)
        print(f'ep {ep} {tag} | gold74 0.9000 (12 labels) | gold58 0.9000 (12 labels) | val AUC vs '
              f'llm_labels_v4_blend.csv {cv:.4f} (12 labels) | vs train labels 0.9000 | epoch '
              f'{(time.time() - t0) / 60:.1f} min', flush=True)
        if SE and SE[0] <= ep + 1 <= SE[1]:
            os.makedirs(f'{A.out}/epochs', exist_ok=True)
            torch.save({'model': SD, 'args': args, 'cv': cv, 'epoch': ep},
                       f'{A.out}/epochs/f{A.fold}_s{A.seed}_{tag}_ttr224-320x24_ep{ep + 1}_best.pt')
            pd.DataFrame(np.random.rand(len(va), 12), columns=TARGETS, index=pd.Index(va, name='StudyInstanceUID')).to_csv(
                f'{A.out}/epochs/oof_f{A.fold}_s{A.seed}_{tag}_ep{ep + 1}.csv')         # as fleet_train.save_epoch
        if cv > best[tag]:
            best[tag] = cv
            save(tag, cv, ep)
    if ep == HANG:
        while True:
            time.sleep(1)
print(json.dumps(best), flush=True)
child.kill()
