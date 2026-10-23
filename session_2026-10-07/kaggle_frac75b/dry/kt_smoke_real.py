# kt_smoke_real.py A|B (kaggle_frac75b): box CPU smoke of the REAL fleet_train.py (the code dataset's copy == box
# ~/bigjump_build/code) launched with the notebook's own recipe_args() (the exact Kaggle command line, incl. --labels
# <LBL>/labels_frac75_logit.csv and the per-trainer --seed), the notebook's HF cache (bundled pretrained bytes,
# HF_HUB_OFFLINE=1), the LBL csv copies and a symlinked 18-file store. Only additions to the command: --device cpu
# --min_cover 0 --time_epochs 1. One run per (fold, seed) of the notebook:
#   A: fold 2 s1043 E 21 (--save_epochs 19-21, the expected plan) + fold 3 s1043 E 2 (epochs/ save path)
#   B: fold 0 s1044 E 21 (the expected plan; ckpt args also paired with ~/fleet_runs/f0_s1044_ttr_mix50) + fold 4 s1043 E 2
# Checks: the trainer accepts every flag and the frac75 csv, its header passes header_problems() (contract, select_sha,
# jpeg_fp, model_sha, counts), progress/eval/closing lines parse, ckpts load with the box state_dict keys, ckpt args
# seed/fold/train_labels right and differ from J1's only where the plan / CPU / tiny store make them, OOF rows == val ids.
# Never touches the GPU (CUDA_VISIBLE_DEVICES='', --device cpu); ~/fleet_store and ~/fleet_runs are only read.
import os, sys, json, shutil, subprocess, time, glob
import pandas as pd

H = os.path.expanduser('~')
KT = f'{H}/kt_frac75b'
KEY = sys.argv[1]
SLUG, PLAN = {'A': ('rsna-train-ttr-frac75-f23', [(2, 1043, 21), (3, 1043, 2)]),
              'B': ('rsna-train-ttr-frac75-f4x', [(0, 1044, 21), (4, 1043, 2)])}[KEY]
NB = f'{KT}/kernel/{SLUG}/{SLUG.replace("-", "_")}.ipynb'
DSRC = f'{KT}/ds/rsna-knee-fleet-train'
DLSRC = f'{KT}/ds_labels'
FRAC = 'labels_frac75_logit.csv'
R = f'{KT}/smoke_{KEY}'
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
SEEDS = ns['CFG']['SEED']
check(f'notebook {SLUG}: trainers {[(F, s) for F, s, _ in PLAN]} == NB_PLAN (GPU order may differ)',
      sorted(SEEDS.items()) == sorted((F, s) for F, s, _ in PLAN), SEEDS)

# store: 8 train ids from folds outside the notebook, per val fold 2 Effusion-positive + 2 negative (vs v4), 2 gold
L = f'{H}/rsna_train'
lab = pd.read_csv(f'{DLSRC}/{FRAC}', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
folds = pd.read_csv(f'{L}/folds5.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['fold']
sel = pd.read_csv(f'{L}/llm_labels_v4_blend.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
gold = list(pd.read_csv(f'{L}/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID)
ids = sorted(s for s in lab.index if s in folds.index and s not in set(gold) and folds[s] >= 0)
NBF = [F for F, _, _ in PLAN]
tr = [s for s in ids if folds[s] not in NBF][:8]
val = {}
for F in NBF:
    vf = [s for s in ids if folds[s] == F and s in sel.index]
    val[F] = [s for s in vf if sel.loc[s, 'Effusion'] >= 0.5][:2] + [s for s in vf if sel.loc[s, 'Effusion'] < 0.5][:2]
pick = tr + sum((val[F] for F in NBF), []) + gold[:2]
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


def run(F, seed, E):
    run_name = ns['RUN'][F]
    check(f'run F{F} s{seed}: notebook RUN name == f{F}_s{seed}_ttr_frac75', run_name == f'f{F}_s{seed}_ttr_frac75', run_name)
    out = f'{R}/kaggle/working/{run_name}'
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
        rc = subprocess.run(cmd, cwd=CODE, stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=2400).returncode
    txt = open(logp).read()
    print(txt[-2500:], flush=True)
    tag = f'run F{F} s{seed} E{E}'
    check(f'{tag}: real trainer rc 0 with the notebook recipe_args (+cpu/min_cover/time_epochs)', rc == 0,
          f'rc {rc}, {time.time() - t:.0f} s')
    a = cmd[2:-6]
    check(f'{tag}: argv uses --labels .../{FRAC}, --seed {seed}, --fold {F}, save_epochs {se}, no mix50 train labels',
          a[a.index('--labels') + 1] == f'{LBL}/{FRAC}' and a[a.index('--seed') + 1] == str(seed) and
          a[a.index('--fold') + 1] == str(F) and a[a.index('--save_epochs') + 1] == se and
          a[a.index('--epochs') + 1] == str(E) and 'labels_mix50' not in ' '.join(a))
    cnt = ns['expected_counts'](LBL, STORE, F)
    pb = ns['header_problems'](ns['parse_header'](txt), cnt, box)
    check(f'{tag}: header == box contract/select/jpeg/model_sha + counts (from the frac75 file)', pb == [], pb)
    check(f'{tag}: progress line parses (RE_IT)', len(ns['parse_its'](txt)) >= 1, ns['parse_its'](txt))
    pr = ns['progress'](txt)
    check(f'{tag}: eval lines parse (RE_EV, raw + ema)', {t for _, t, _, _ in pr['evals']} == {'raw', 'ema'} and
          pr['last_ep_done'] == 0 and len(pr['train_only_min']) == 1, pr)
    check(f'{tag}: closing line', ns['closing_line'](txt) is not None, txt.splitlines()[-1][:120])
    pts = sorted(os.path.relpath(os.path.join(r, f), out) for r, _, fs in os.walk(out) for f in fs if f.endswith('.pt'))
    ok, diffs, args = [], set(), []
    for p in pts:
        ck = torch.load(f'{out}/{p}', map_location='cpu', weights_only=False)
        a2 = ck['args']
        args.append(a2)
        ok.append(list(ck['model'].keys()) == keys and a2['epochs'] == E and a2['save_epochs'] == se and a2['seed'] == seed
                  and a2['fold'] == F and a2['train_labels'] == FRAC and a2['sel_labels'] == 'llm_labels_v4_blend.csv'
                  and a2['model_sha'] == box['ckpt_args']['model_sha'] and os.path.basename(p).startswith(f'f{F}_s{seed}_'))
        diffs |= {k for k in set(a2) | set(J1A) if repr(a2.get(k)) != repr(J1A.get(k))}
    allowed = {'epochs', 'save_epochs', 'train_labels', 'amp', 'decode_probe'} | ({'fold'} if F else set()) | \
              ({'seed'} if seed != 1043 else set())
    check(f'{tag}: ckpts load with box keys, named f{F}_s{seed}_*; args seed {seed} fold {F} train_labels {FRAC}; vs J1 '
          f'args differ only in {sorted(allowed)}',
          len(pts) >= 2 and all(ok) and diffs <= allowed and {'train_labels', 'epochs', 'save_epochs'} <= diffs,
          (pts, sorted(diffs)))
    pair = sorted(glob.glob(f'{H}/fleet_runs/f{F}_s{seed}_ttr_mix50/f{F}_s{seed}_raw_*_best.pt'))
    if seed != 1043 and pair:                 # the mix50 run this replicate pairs with (Kaggle T4, E 21, workers 3)
        PA = torch.load(pair[0], map_location='cpu', weights_only=False)['args']
        d2 = {k for k in set(args[0]) | set(PA) if repr(args[0].get(k)) != repr(PA.get(k))}
        check(f'{tag}: vs the paired ~/fleet_runs/f{F}_s{seed}_ttr_mix50 ckpt args (E {PA["epochs"]}): differ only in '
              f'train_labels / save_epochs (+ CPU amp, decode_probe)',
              'train_labels' in d2 and d2 <= {'train_labels', 'save_epochs', 'amp', 'decode_probe'} and
              PA['seed'] == seed and PA['fold'] == F and PA['epochs'] == E, sorted(d2))
    oofs = sorted(os.path.relpath(os.path.join(r, f), out) for r, _, fs in os.walk(out) for f in fs if f.startswith('oof_'))
    o = pd.read_csv(f'{out}/oof_f{F}_s{seed}_raw.csv', dtype={'StudyInstanceUID': str})
    check(f'{tag}: OOF rows == val ids', sorted(o.StudyInstanceUID) == sorted(val[F]), oofs)
    return pts, oofs


(Fa, sa, Ea), (Fb, sb, Eb) = PLAN
pts, oofs = run(Fa, sa, Ea)
check(f'run F{Fa} E{Ea}: E {Ea} > time_epochs 1, so no epochs/ files (save range {ns["save_epochs_arg"](Ea)})',
      not any('/' in p for p in pts + oofs), pts + oofs)
pts, oofs = run(Fb, sb, Eb)
check(f'run F{Fb} E{Eb}: best raw/ema + epochs/ep1 raw/ema ckpts and OOFs', sum('epochs/' in p for p in pts) == 2 and
      sum('epochs/' in p for p in oofs) == 2 and len(pts) == 4, pts + oofs)
print(f'\nSMOKE {KEY} ({SLUG}): {sum(RES.values())}/{len(RES)} PASS', flush=True)
sys.exit(0 if all(RES.values()) else 1)
