# soup_eval.py (box CPU, 4 threads, nice 19; report-only experiment, NOT a gate): does a uniform WEIGHT soup of the RSNA-only
# seeds (same pretrained init + same recipe) match their prediction ensemble? If yes, one kernel arm could carry k seeds.
# gold58, training geometry (24 evenly spaced windows, full field 384), fp32 CPU. Seeds' own training-geometry predictions =
# ~/raptor_store/rented/s<seed>/gold58_swa.npy (written by the trainer on the 4090, bf16).
import os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np, pandas as pd, torch, timm
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
torch.set_num_threads(int(os.environ.get('NTHR', 4)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import raptor_ref as R
H = os.path.expanduser('~'); D = 96
SEEDS = [int(x) for x in sys.argv[1:]] or [46, 45, 43, 44]
idx = pd.read_csv(f'{H}/raptor_store/rsna_index.csv', dtype={'id': str, 'mask': str}).set_index('id')
g = pd.read_csv(f'{H}/rsna_train/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}); g = g[g.source == 'original_gold58'].set_index('StudyInstanceUID')
ids = list(g.index); Y = g[R.LAB].values.astype(float)
auc = lambda P: float(np.mean([roc_auc_score(Y[:, j], P[:, j]) for j in range(12)]))
rk = lambda P: np.stack([rankdata(P[:, j]) / len(P) for j in range(12)], 1)
own = {s: np.load(f'{H}/raptor_store/rented/s{s}/gold58_swa.npy') for s in SEEDS}
for s in SEEDS: print(f's{s} own (4090 bf16) gold58 {auc(own[s]):.4f}', flush=True)
for k in (2, len(SEEDS)):
    print(f'prediction ensemble k{k} {SEEDS[:k]}: rank-mean {auc(np.mean([rk(own[s]) for s in SEEDS[:k]], 0)):.4f} | prob-mean {auc(np.mean([own[s] for s in SEEDS[:k]], 0)):.4f}', flush=True)
cks = [torch.load(f'{H}/raptor_store/rented/s{s}/swa.pt', map_location='cpu', weights_only=False)['model'] for s in SEEDS]
def soup(sds):
    return {k: ((sum(sd[k].double() for sd in sds) / len(sds)).to(v.dtype) if v.is_floating_point() else sds[0][k].clone()) for k, v in sds[0].items()}
def wins(sid):
    vol = np.load(idx.at[sid, 'path'], mmap_mode='r'); mask = np.frombuffer(idx.at[sid, 'mask'].encode(), np.uint8) - 48
    cs = R._eval_centers(mask, D, 24)
    x = torch.from_numpy(np.stack([vol[[max(1, min(c, D - 2)) + d for d in (-1, 0, 1)]] for c in cs]).astype(np.float32) / 255.0)
    return ((x - R._MEAN) / R._STD)[None]
for k in (2, len(SEEDS)):
    bb = timm.create_model('coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', pretrained=False, num_classes=0, in_chans=3, img_size=384, global_pool='avg')
    m = R.RaptorClassifier(bb, F_dim=bb.num_features); m.load_state_dict(soup(cks[:k]), strict=True); m.eval()
    t0 = time.time(); P = []
    with torch.no_grad():
        for sid in ids:
            P.append(torch.sigmoid(m(wins(sid)))[0].numpy())
    P = np.array(P); np.save(f'{H}/raptor2/soup_k{k}_gold58.npy', P)
    print(f'WEIGHT SOUP k{k} {SEEDS[:k]}: gold58 {auc(P):.4f} (CPU fp32, {time.time() - t0:.0f}s)', flush=True)
