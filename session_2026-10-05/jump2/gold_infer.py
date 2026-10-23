# gold_infer.py (jump2, 2026-10-05): CPU-only predictions of the fold-0 fleet checkpoints on the 74 gold studies
# (gold74 is --exclude'd from every fleet run, so these are out-of-sample, same status as a train row's OOF).
# Eval path = fleet_train.DS non-train branch (sample_triplets off 0, no aug, pos=None, sigmoid). fp32 on CPU
# (trainer evals in fp16 AMP on GPU -> tiny diffs). Read-only imports from ~/bigjump_build/code
# (run with PYTHONDONTWRITEBYTECODE=1). Writes ~/jump2/gold74_preds/<run>__<tag>.csv only.
import os, sys, time, glob
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.expanduser('~/bigjump_build/code'))
import fleet_common as fc
from fleet_model import KneeFleetNet
from fleet_model_ttr import KneeFleetNetTTR
from sklearn.metrics import roc_auc_score

torch.set_num_threads(int(os.environ.get('NTHR', '12')))
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
H = os.path.expanduser
STORE = H('~/fleet_store'); OUT = H('~/jump2/gold74_preds'); os.makedirs(OUT, exist_ok=True)
gold = pd.read_csv(H('~/rsna_train/gold74_eval_set.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
ids = [s for s in gold.index if os.path.exists(f'{STORE}/{s}.npz')]
g58 = [s for s in ids if gold.loc[s, 'source'] == 'original_gold58']
print('gold ids', len(ids), 'gold58', len(g58), flush=True)

# decode once (uint8), reuse for every checkpoint
X = {}
for s in ids:
    vs, pr, ct = [], [], []
    with np.load(f'{STORE}/{s}.npz') as z:
        fc.check_store_contract(z, s)
        for j in range(len(fc.VIEWS)):
            if f'v{j}_blob' in z.files:
                st = fc.decode_stack(z[f'v{j}_blob'], z[f'v{j}_offs'])
                vs.append(fc.sample_triplets(st, fc.N_SLICES, 0.0)); pr.append(True); ct.append(int(z[f'v{j}_ctr']))
            else:
                vs.append(np.zeros((fc.N_SLICES, 3, fc.STORE, fc.STORE), np.uint8)); pr.append(False); ct.append(3)
    X[s] = (torch.from_numpy(np.stack(vs))[None], torch.tensor(pr)[None], torch.tensor(ct)[None])
print('decoded', len(X), flush=True)


def build(a):
    kw = dict(backbone=a['backbone'], dim=a['dim'], contrast_emb=bool(a['contrast_emb']), n_views=a['n_views'],
              n_slices=a['n_slices'], n_contrast=a['n_contrast'], pretrained=False, drop_path=0.0, dropout=0.0,
              grad_ckpt=False)
    if a.get('model_mod') == 'fleet_model_ttr':
        return KneeFleetNetTTR(ttr_size=a['ttr_size'], **kw)
    return KneeFleetNet(**kw)


def mauc(P, sub):
    k = [ids.index(s) for s in sub]; Y = gold.loc[sub, T].values.astype(float); out = []
    for j in range(12):
        y = Y[:, j] >= 0.5
        if 0 < y.mean() < 1: out.append(roc_auc_score(y, P[k, j]))
    return float(np.mean(out))


ckpts = sys.argv[1:]
for f in ckpts:
    run = f.split('/')[-2]; tag = 'ema' if '_ema_' in os.path.basename(f) else 'raw'
    dst = f'{OUT}/{run}__{tag}.csv'
    if os.path.exists(dst):
        print('skip', dst, flush=True); continue
    t0 = time.time()
    c = torch.load(f, map_location='cpu', weights_only=False)
    m = build(c['args']); m.load_state_dict(c['model'], strict=True)
    m = m.eval().to(memory_format=torch.channels_last)
    P = []
    with torch.no_grad():
        for s in ids:
            x, pr, ct = X[s]
            P.append(torch.sigmoid(m(x, pr, ct).float()).numpy()[0])
    P = np.stack(P)
    pd.DataFrame(P, columns=T, index=pd.Index(ids, name='StudyInstanceUID')).to_csv(dst)
    print(f'{run} {tag} ep{c["epoch"] + 1} cv {c["cv"]:.4f} | gold58 {mauc(P, g58):.4f} gold74 {mauc(P, ids):.4f} '
          f'| {time.time() - t0:.0f}s -> {dst}', flush=True)
print('done', flush=True)
