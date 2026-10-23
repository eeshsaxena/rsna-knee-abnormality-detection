# kt_test/box_prep.py: READ-ONLY facts from the box for the Kaggle fold-3/4 training dataset (CPU only).
import os, sys, json, hashlib, time, glob
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np, pandas as pd, torch
H = os.path.expanduser('~')
sys.path.insert(0, f'{H}/kt_test/code')          # byte-identical copy of ~/bigjump_build/code (checked below)
out = {}
# 0 code copy == box code
for f in ['fleet_train.py', 'fleet_common.py', 'fleet_model.py', 'fleet_model_ttr.py', 'fleet_model_f2.py', 'fleet_aug.py']:
    a = hashlib.sha256(open(f'{H}/bigjump_build/code/{f}', 'rb').read()).hexdigest()
    b = hashlib.sha256(open(f'{H}/kt_test/code/{f}', 'rb').read()).hexdigest()
    assert a == b, f
    out.setdefault('code_sha256', {})[f] = a
# 1 store sample sha (box) vs tar members (laptop copy of the Kaggle prep outputs)
tl = json.load(open(f'{H}/kt_test/tar_listing.json'))
smp = []
for s in tl['sample']:
    h = hashlib.sha256(open(f'{H}/fleet_store/{s["name"]}', 'rb').read()).hexdigest()
    assert h == s['sha256'], (s, h)
    smp.append(dict(name=s['name'], part=s['part'], size=s['size'], sha256=h))
out['store_sample'] = smp
allm = sorted(n for v in tl['members'].values() for n in v)
box = sorted(os.path.basename(p) for p in glob.glob(f'{H}/fleet_store/*.npz'))
assert allm == box, (len(allm), len(box))
out['store_n'] = len(box)
out['store_names_sha'] = hashlib.sha256('\n'.join(box).encode()).hexdigest()
print('store sample ok', len(smp), '| tar members == box store names', len(box), flush=True)
# 2 ckpt facts (J1 fold 0 raw best)
ck = torch.load(f'{H}/fleet_runs/f0_s1043_ttr_mix50/f0_s1043_raw_ttr224-320x24_cv8835_best.pt', map_location='cpu', weights_only=False)
sd = ck['model']
out['state_dict'] = [[k, list(v.shape), str(v.dtype).replace('torch.', '')] for k, v in sd.items()]
a = ck['args']
out['ckpt_args'] = {k: a[k] for k in a if k not in ('aug_cfg',)}
out['ckpt_args']['aug_cfg'] = a.get('aug_cfg')
print('ckpt keys', len(sd), '| model_sha', a['model_sha'], '| decode_probe', a['decode_probe'], flush=True)
# 3 expected counts per fold (same arithmetic as fleet_train.py)
L = f'{H}/rsna_train'
lab = pd.read_csv(f'{L}/labels_mix50_logit.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
folds = pd.read_csv(f'{L}/folds5.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['fold']
excl = set(pd.read_csv(f'{L}/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID)
held = set(folds.index[folds < 0]) | excl
have = {s for s in lab.index if os.path.exists(f'{H}/fleet_store/{s}.npz')}
ids = [s for s in lab.index if s in folds.index and s in have]
cnt = {}
for F in range(5):
    tr = [s for s in ids if s not in held and folds[s] != F]; va = [s for s in ids if s not in held and folds[s] == F]
    cnt[F] = dict(labels=len(lab), store_has=len(have), held=len(held), train=len(tr), val=len(va), iters_per_epoch=len(tr) // 2)
out['expected_counts'] = cnt
print('counts', cnt, flush=True)
# 4 CPU decode benchmark (one core): heavy train path and eval path
import cv2; cv2.setNumThreads(0); torch.set_num_threads(1)
import fleet_common as fc
from fleet_aug import heavy_study
PLANES = [p for p, _ in fc.VIEWS]
np.random.seed(0)
bench = sorted(ids)[100:130]
t0 = time.time()
for s in bench:
    with np.load(f'{H}/fleet_store/{s}.npz') as z:
        fc.check_store_contract(z, s); heavy_study(z, s, 16, PLANES)
th = (time.time() - t0) / len(bench)
t0 = time.time()
for s in bench:
    with np.load(f'{H}/fleet_store/{s}.npz') as z:
        fc.check_store_contract(z, s)
        for j in range(4):
            if f'v{j}_blob' in z.files:
                fc.sample_triplets(fc.decode_stack(z[f'v{j}_blob'], z[f'v{j}_offs']), fc.N_SLICES, 0.0)
te = (time.time() - t0) / len(bench)
out['cpu_bench_box'] = dict(cpu='Xeon Silver 4114 2.2GHz, 1 thread', heavy_s_per_study=round(th, 4), eval_s_per_study=round(te, 4))
print('cpu bench s/study heavy', round(th, 3), 'eval', round(te, 3), flush=True)
json.dump(out, open(f'{H}/kt_test/box_facts.json', 'w'), indent=1)
print('wrote box_facts.json')
