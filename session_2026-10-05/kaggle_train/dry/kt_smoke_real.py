# kt_smoke_real.py: box CPU smoke of the REAL fleet_train.py launched with the notebook's own recipe_args() (the exact
# Kaggle command line), the notebook's HF cache (bundled pretrained bytes, HF_HUB_OFFLINE=1), the LBL csv copies and a
# 14-file symlinked store. Only additions to the command: --device cpu --min_cover 0 --time_epochs 1 (CPU + tiny store).
# Checks: the trainer accepts every flag, its header passes header_problems() (contract, select_sha, jpeg_fp, model_sha,
# counts), progress/eval/closing lines parse, best + epochs/ ckpts load with the box state_dict keys, OOF rows == val.
# Never touches the GPU (CUDA_VISIBLE_DEVICES='', --device cpu), ~/fleet_store is only read (symlinks).
import os, sys, json, shutil, subprocess, time
import pandas as pd

H = os.path.expanduser('~')
KT = f'{H}/kt_test'
NB = f'{KT}/kernel/rsna-train-ttr-f34/rsna_train_ttr_f34.ipynb'
DSRC = f'{KT}/ds/rsna-knee-fleet-train'
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

# store: 8 train (folds != 3), 4 val (fold 3, mixed Effusion labels so a val AUC exists), 2 gold
L = f'{H}/rsna_train'
lab = pd.read_csv(f'{L}/labels_mix50_logit.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
folds = pd.read_csv(f'{L}/folds5.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['fold']
sel = pd.read_csv(f'{L}/llm_labels_v4_blend.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
gold = list(pd.read_csv(f'{L}/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID)
ids = sorted(s for s in lab.index if s in folds.index and s not in set(gold) and folds[s] >= 0)
tr = [s for s in ids if folds[s] != 3][:8]
v3 = [s for s in ids if folds[s] == 3 and s in sel.index]
pos = [s for s in v3 if sel.loc[s, 'Effusion'] >= 0.5][:2]
neg = [s for s in v3 if sel.loc[s, 'Effusion'] < 0.5][:2]
pick = tr + pos + neg + gold[:2]
STORE = f'{R}/tmp/fleet_store'
os.makedirs(STORE)
for s in pick:
    os.symlink(f'{H}/fleet_store/{s}.npz', f'{STORE}/{s}.npz')
LBL = f'{R}/tmp/kt_labels'
os.makedirs(LBL)
for f in ns['LABEL_FILES'] + ['box_facts.json']:
    shutil.copyfile(f'{DSRC}/{f}', f'{LBL}/{f}')
CODE = f'{R}/tmp/kt_code'
os.makedirs(CODE)
for f in ns['CODE_FILES']:
    shutil.copyfile(f'{DSRC}/{f}', f'{CODE}/{f}')
ns['install_hf_cache'](DSRC, exp)
ns['STORE'] = STORE                                   # what cell 4 does on Kaggle

E, F = 2, 3
out = f'{R}/kaggle/working/f3_s1043_ttr_mix50'
logp = f'{R}/kaggle/working/f3_s1043_ttr_mix50.log'
cmd = [sys.executable, f'{CODE}/fleet_train.py'] + ns['recipe_args'](F, E, ns['save_epochs_arg'](E), out, LBL, 2) + \
      ['--device', 'cpu', '--min_cover', '0', '--time_epochs', '1']
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
           HF_HOME=ns['HF_HOME'], HF_HUB_OFFLINE='1')
env.pop('KT_ROOT', None)
print(' '.join(cmd), flush=True)
t = time.time()
with open(logp, 'w') as lf:
    rc = subprocess.run(cmd, cwd=CODE, stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=1800).returncode
txt = open(logp).read()
print(txt[-3000:], flush=True)
check('real trainer rc 0 with the notebook recipe_args (+cpu/min_cover/time_epochs)', rc == 0, f'rc {rc}, {time.time() - t:.0f} s')
cnt = ns['expected_counts'](LBL, STORE, F)
pb = ns['header_problems'](ns['parse_header'](txt), cnt, box)
check('real trainer header == box contract/select/jpeg/model_sha + counts', pb == [], pb)
its = ns['parse_its'](txt)
check('real trainer progress line parses (RE_IT)', len(its) >= 1 and its[0][0] == 0, its)
pr = ns['progress'](txt)
check('real trainer eval lines parse (RE_EV, raw + ema)', {t for _, t, _, _ in pr['evals']} == {'raw', 'ema'} and
      pr['last_ep_done'] == 0 and len(pr['train_only_min']) == 1, pr)
check('real trainer closing line', ns['closing_line'](txt) is not None, txt.splitlines()[-1][:120])
import torch
keys = [k for k, _, _ in box['state_dict']]
pts = sorted(os.path.relpath(os.path.join(r, f), out) for r, _, fs in os.walk(out) for f in fs if f.endswith('.pt'))
ok = []
for p in pts:
    ck = torch.load(f'{out}/{p}', map_location='cpu', weights_only=False)
    ok.append(list(ck['model'].keys()) == keys and ck['args']['epochs'] == E and ck['args']['save_epochs'] == '1-2'
              and ck['args']['seed'] == 1043 and ck['args']['fold'] == 3 and ck['args']['model_sha'] == box['ckpt_args']['model_sha'])
check('ckpts: best raw/ema + epochs/ep1 raw/ema, box keys + args', len(pts) == 4 and all(ok) and
      sum('/' in p for p in pts) == 2, pts)
oofs = sorted(f for r, _, fs in os.walk(out) for f in fs if f.startswith('oof_'))
o = pd.read_csv(f'{out}/oof_f3_s1043_raw.csv', dtype={'StudyInstanceUID': str}) if 'oof_f3_s1043_raw.csv' in oofs else None
check('OOF csvs written (best + epochs), rows == val ids', o is not None and len(oofs) == 4 and
      sorted(o.StudyInstanceUID) == sorted(pos + neg), oofs)
print(f'\nSMOKE: {sum(RES.values())}/{len(RES)} PASS', flush=True)
sys.exit(0 if all(RES.values()) else 1)
