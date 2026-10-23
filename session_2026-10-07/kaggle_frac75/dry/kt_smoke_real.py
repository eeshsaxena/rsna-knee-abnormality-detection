# kt_smoke_real.py (kaggle_frac75): box CPU smoke of the REAL fleet_train.py (the code dataset's copy == box
# ~/bigjump_build/code) launched with the notebook's own recipe_args() (the exact Kaggle command line, incl. --labels
# <LBL>/labels_frac75_logit.csv), the notebook's HF cache (bundled pretrained bytes, HF_HUB_OFFLINE=1), the LBL csv copies
# and a 14-file symlinked store. Only additions to the command: --device cpu --min_cover 0 --time_epochs 1.
#   run 1: fold 0, E 17, --save_epochs 15-17 (the expected real plan): exact argv, ckpt args vs J1's ckpt args
#   run 2: fold 1, E 2, --save_epochs 1-2: the epochs/ save path (ckpt + OOF) with the frac75 labels
# Checks: the trainer accepts every flag and the frac75 csv, its header passes header_problems() (contract, select_sha,
# jpeg_fp, model_sha, counts), progress/eval/closing lines parse, ckpts load with the box state_dict keys, ckpt args
# train_labels == labels_frac75_logit.csv and differ from J1's only where the Kaggle plan / CPU / tiny store make them,
# OOF rows == val ids. Never touches the GPU (CUDA_VISIBLE_DEVICES='', --device cpu); ~/fleet_store is only read.
import os, sys, json, shutil, subprocess, time, glob
import pandas as pd

H = os.path.expanduser('~')
KT = f'{H}/kt_frac75'
NB = f'{KT}/kernel/rsna-train-ttr-frac75-f01/rsna_train_ttr_frac75_f01.ipynb'
DSRC = f'{KT}/ds/rsna-knee-fleet-train'
DLSRC = f'{KT}/ds_labels'
FRAC = 'labels_frac75_logit.csv'
R = f'{KT}/smoke'
RES = {}


def check(name, cond, info=''):
    RES[name] = bool(cond)
    print(('PASS ' if cond else 'FAIL ') + name + (f'  [{info}]' if info else ''), flush=True)


shutil.rmtree(R, ignore_errors=True)
os.makedirs(f'{R}/kaggle/working')
os.environ['KT_ROOT'] = R
ns = {}
cells = [''.join(c['source']) for c in json.load(open(NB))['cells'] if c['cell_type'] == 'code']
exec(compile(cells[0], 'lib', 'exec'), ns)
box = json.load(open(f'{DSRC}/box_facts.json'))
exp = json.load(open(f'{DSRC}/kt_expect.json'))

# store: 8 train (folds 2-4), per val fold 0 and 1: 2 Effusion-positive + 2 negative (vs v4) so a val AUC exists, 2 gold
L = f'{H}/rsna_train'
lab = pd.read_csv(f'{DLSRC}/{FRAC}', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
folds = pd.read_csv(f'{L}/folds5.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['fold']
sel = pd.read_csv(f'{L}/llm_labels_v4_blend.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
gold = list(pd.read_csv(f'{L}/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID)
ids = sorted(s for s in lab.index if s in folds.index and s not in set(gold) and folds[s] >= 0)
tr = [s for s in ids if folds[s] >= 2][:8]
val = {}
for F in (0, 1):
    vf = [s for s in ids if folds[s] == F and s in sel.index]
    val[F] = [s for s in vf if sel.loc[s, 'Effusion'] >= 0.5][:2] + [s for s in vf if sel.loc[s, 'Effusion'] < 0.5][:2]
pick = tr + val[0] + val[1] + gold[:2]
STORE = f'{R}/tmp/fleet_store'
os.makedirs(STORE)
for s in pick:
    os.symlink(f'{H}/fleet_store/{s}.npz', f'{STORE}/{s}.npz')
LBL = f'{R}/tmp/kt_labels'
os.makedirs(LBL)
for f in ns['LABEL_FILES'] + ['box_facts.json']:
    shutil.copyfile(f'{DSRC}/{f}', f'{LBL}/{f}')
shutil.copyfile(f'{DLSRC}/{FRAC}', f'{LBL}/{FRAC}')
CODE = f'{R}/tmp/kt_code'
os.makedirs(CODE)
for f in ns['CODE_FILES']:
    shutil.copyfile(f'{DSRC}/{f}', f'{CODE}/{f}')
check('code == box ~/bigjump_build/code (the J1 trainer)', all(
    open(f'{CODE}/{f}', 'rb').read() == open(f'{H}/bigjump_build/code/{f}', 'rb').read() for f in ns['CODE_FILES']))
ns['install_hf_cache'](DSRC, exp)
ns['STORE'] = STORE                                   # what cell 4 does on Kaggle
import torch
J1A = torch.load(glob.glob(f'{H}/fleet_runs/f0_s1043_ttr_mix50/f0_s1043_raw_*_best.pt')[0], map_location='cpu',
                 weights_only=False)['args']
keys = [k for k, _, _ in box['state_dict']]


def run(F, E):
    out = f'{R}/kaggle/working/f{F}_s1043_ttr_frac75'
    logp = f'{out}.log'
    se = ns['save_epochs_arg'](E)
    cmd = [sys.executable, f'{CODE}/fleet_train.py'] + ns['recipe_args'](F, E, se, out, LBL, 2) + \
          ['--device', 'cpu', '--min_cover', '0', '--time_epochs', '1']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
               HF_HOME=ns['HF_HOME'], HF_HUB_OFFLINE='1')
    env.pop('KT_ROOT', None)
    print(' '.join(cmd), flush=True)
    t = time.time()
    with open(logp, 'w') as lf:
        rc = subprocess.run(cmd, cwd=CODE, stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=1800).returncode
    txt = open(logp).read()
    print(txt[-2500:], flush=True)
    tag = f'run F{F} E{E}'
    check(f'{tag}: real trainer rc 0 with the notebook recipe_args (+cpu/min_cover/time_epochs)', rc == 0,
          f'rc {rc}, {time.time() - t:.0f} s')
    a = cmd[2:-6]
    check(f'{tag}: argv uses --labels .../{FRAC}, seed 1043, save_epochs {se}, no mix50 train labels',
          a[a.index('--labels') + 1] == f'{LBL}/{FRAC}' and a[a.index('--seed') + 1] == '1043' and
          a[a.index('--save_epochs') + 1] == se and a[a.index('--epochs') + 1] == str(E) and 'labels_mix50' not in ' '.join(a))
    cnt = ns['expected_counts'](LBL, STORE, F)
    pb = ns['header_problems'](ns['parse_header'](txt), cnt, box)
    check(f'{tag}: header == box contract/select/jpeg/model_sha + counts (from the frac75 file)', pb == [], pb)
    check(f'{tag}: progress line parses (RE_IT)', len(ns['parse_its'](txt)) >= 1, ns['parse_its'](txt))
    pr = ns['progress'](txt)
    check(f'{tag}: eval lines parse (RE_EV, raw + ema)', {t for _, t, _, _ in pr['evals']} == {'raw', 'ema'} and
          pr['last_ep_done'] == 0 and len(pr['train_only_min']) == 1, pr)
    check(f'{tag}: closing line', ns['closing_line'](txt) is not None, txt.splitlines()[-1][:120])
    pts = sorted(os.path.relpath(os.path.join(r, f), out) for r, _, fs in os.walk(out) for f in fs if f.endswith('.pt'))
    ok, diffs = [], set()
    for p in pts:
        ck = torch.load(f'{out}/{p}', map_location='cpu', weights_only=False)
        a2 = ck['args']
        ok.append(list(ck['model'].keys()) == keys and a2['epochs'] == E and a2['save_epochs'] == se and a2['seed'] == 1043
                  and a2['fold'] == F and a2['train_labels'] == FRAC and a2['sel_labels'] == 'llm_labels_v4_blend.csv'
                  and a2['model_sha'] == box['ckpt_args']['model_sha'])
        diffs |= {k for k in set(a2) | set(J1A) if repr(a2.get(k)) != repr(J1A.get(k))}
    allowed = {'epochs', 'save_epochs', 'train_labels', 'amp', 'decode_probe'} | ({'fold'} if F else set())
    check(f'{tag}: ckpts load with box keys; args train_labels {FRAC}; vs J1 args differ only in {sorted(allowed)}',
          len(pts) >= 2 and all(ok) and diffs <= allowed and {'train_labels', 'epochs', 'save_epochs'} <= diffs,
          (pts, sorted(diffs)))
    oofs = sorted(os.path.relpath(os.path.join(r, f), out) for r, _, fs in os.walk(out) for f in fs if f.startswith('oof_'))
    o = pd.read_csv(f'{out}/oof_f{F}_s1043_raw.csv', dtype={'StudyInstanceUID': str})
    check(f'{tag}: OOF rows == val ids', sorted(o.StudyInstanceUID) == sorted(val[F]), oofs)
    return pts, oofs


pts, oofs = run(0, 17)
check('run F0 E17: E 17 > time_epochs 1, so no epochs/ files (save range 15-17)', not any('/' in p for p in pts + oofs),
      pts + oofs)
pts, oofs = run(1, 2)
check('run F1 E2: best raw/ema + epochs/ep1 raw/ema ckpts and OOFs', sum('epochs/' in p for p in pts) == 2 and
      sum('epochs/' in p for p in oofs) == 2 and len(pts) == 4, pts + oofs)
print(f'\nSMOKE: {sum(RES.values())}/{len(RES)} PASS', flush=True)
sys.exit(0 if all(RES.values()) else 1)
