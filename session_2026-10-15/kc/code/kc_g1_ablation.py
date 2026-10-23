# kc_g1_ablation.py -- KC G1 early view ablation on a saved epoch ckpt (BOX_PLAN_CRITIQUE_1015: fold-0 view ablation at
# epoch 3, ~1.2 h into f0, on the box CPU). Same kc_g1 code as the f0 end hook. INFO for main only: it never writes a
# queue gate file (output lines start with EARLY_G1, never PASS / FAIL). Main decides whether a clear FAIL stops f0 early.
#   taskset -c 28-39 nice -n 19 env CUDA_VISIBLE_DEVICES= python kc_g1_ablation.py --ckpt <.../epochs/f0_s2001_raw_..._ep3_best.pt>
#       --store ~/kc_store/store --labels ~/r7t_labels/real/labels_r7t_b.csv --folds ~/rsna_train/folds5.csv
#       --exclude ~/rsna_train/gold74_eval_set.csv --fold 0 --n_max 400 --threads 12 --out ~/kc/gate/kc_f0_ep3_ablation
# CPU cost MEASURED in the C4 smoke: ~6-7 s per study for one 72-image encode on 12 threads (cores 28-39).
import os, sys, json, time, argparse
sys.dont_write_bytecode = True                   # never write .pyc into ~/fast_build/code2
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.expanduser('~/fast_build/code2'))   # fleet_model / fleet_model_ttr (read only)
import kc_common as fc
import kc_g1
from fleet_model_ttr import KneeFleetNetTTR

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--store', required=True)
ap.add_argument('--labels', required=True); ap.add_argument('--folds', required=True)
ap.add_argument('--exclude', default=''); ap.add_argument('--fold', type=int, default=0)
ap.add_argument('--n_max', type=int, default=0); ap.add_argument('--threads', type=int, default=12)
ap.add_argument('--workers', type=int, default=3); ap.add_argument('--out', required=True)
A = ap.parse_args()
torch.set_num_threads(A.threads)
t0 = time.time()
ck = torch.load(A.ckpt, map_location='cpu', weights_only=False)
a = ck['args']
assert a.get('model_mod') == 'kc_ttr' and a.get('kc_contract_hash') == fc.CONTRACT_HASH, (a.get('model_mod'), a.get('kc_contract_hash'))
m = KneeFleetNetTTR(ttr_size=a['ttr_size'], backbone=a['backbone'], dim=a['dim'], contrast_emb=a['contrast_emb'],
                    n_views=a['n_views'], n_slices=a['n_slices'], n_contrast=a['n_contrast'], pretrained=False,
                    drop_path=0.0, dropout=0.0, grad_ckpt=False)
m.load_state_dict(ck['model'])
m.eval()
lab = pd.read_csv(A.labels, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
fo = pd.read_csv(A.folds, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['fold']
ex = set(pd.read_csv(A.exclude, dtype={'StudyInstanceUID': str}).StudyInstanceUID) if A.exclude else set()
ids = [s for s in lab.index if s in fo.index and fo[s] == A.fold and s not in ex and os.path.exists(f'{A.store}/{s}.npz')]
n_all = len(ids)
if A.n_max and len(ids) > A.n_max:
    ids = sorted(np.random.default_rng(1015).choice(ids, A.n_max, replace=False).tolist())


class DS(torch.utils.data.Dataset):                  # = the trainer's eval path (DS.__getitem__, train=False)
    def __len__(self):
        return len(ids)

    def __getitem__(self, i):
        sid = ids[i]; vs, pr, ct = [], [], []
        with np.load(f'{A.store}/{sid}.npz') as z:
            fc.check_store_contract(z, sid)
            for j in range(len(fc.VIEWS)):
                if f'v{j}_blob' in z.files:
                    st = fc.decode_stack(z[f'v{j}_blob'], z[f'v{j}_offs'])
                    vs.append(fc.sample_triplets(st, fc.N_SLICES, 0.0)); pr.append(True); ct.append(int(z[f'v{j}_ctr']))
                else:
                    vs.append(np.zeros((fc.N_SLICES, 3, fc.STORE, fc.STORE), np.uint8)); pr.append(False); ct.append(3)
        y = lab.loc[sid, kc_g1.TARGETS].to_numpy(np.float32)
        return (torch.from_numpy(np.stack(vs)), torch.tensor(pr), torch.tensor(ct),
                torch.arange(fc.N_SLICES)[None].expand(len(fc.VIEWS), -1).clone(), torch.from_numpy(y))


def _wif(_):
    import cv2; cv2.setNumThreads(0)


dl = torch.utils.data.DataLoader(DS(), batch_size=1, shuffle=False, num_workers=A.workers, worker_init_fn=_wif)
P = kc_g1.run_variants(m, dl, 'cpu', False, fc.VIEW_NAMES)
res = kc_g1.metrics(P, lab.loc[ids, kc_g1.TARGETS].to_numpy(np.float64))
res.update(ckpt=A.ckpt, epoch=ck.get('epoch'), n_fold=n_all, n_used=len(ids), s_per_study=(time.time() - t0) / max(len(ids), 1),
           wall_min=(time.time() - t0) / 60)
json.dump(res, open(A.out + '.json', 'w'), indent=1, default=float)
pd.concat({k: pd.DataFrame(v, columns=kc_g1.TARGETS, index=pd.Index(ids, name='StudyInstanceUID')) for k, v in P.items()},
          axis=1).to_csv(A.out + '_probs.csv')
line = (f"EARLY_G1 ablation_pass={res['ablation_pass']} epoch={(ck.get('epoch') or 0) + 1} | {kc_g1.summary_line(res)} | "
        f"n_used {len(ids)}/{n_all} | {res['wall_min']:.1f} min")
open(A.out + '.txt', 'w').write(line + '\n# info only (not a queue gate); main decides on a clear FAIL\n')
print(line, flush=True)
