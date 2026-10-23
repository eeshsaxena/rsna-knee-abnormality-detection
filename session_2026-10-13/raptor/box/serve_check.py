# serve_check.py (box GPU, inside a queue pause window only): score Raptor checkpoints (N's dict format) on gold58 EXACTLY
# as the scored r384 source serves them: model built at img_size 320, R.gpu_windows(vol, mask, K_EVAL=94, 320) on the
# 384 volume (centre crop [32:352]), anatomical mirror TTA, R.infer_pair_probs (fp16 autocast, mean of 2 sigmoids).
# Usage: python serve_check.py --index IDX.csv --ck A.pt [B.pt ...] --out DIR
import os, sys, time, argparse, json
import numpy as np, pandas as pd, torch, timm
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import raptor_ref as R
ap = argparse.ArgumentParser()
ap.add_argument('--index', required=True); ap.add_argument('--ck', nargs='+', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--gold', default='/home/training/rsna_train/gold74_eval_set.csv')
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
idx = pd.read_csv(A.index, dtype={'id': str, 'mask': str}).set_index('id')
g = pd.read_csv(A.gold, dtype={'StudyInstanceUID': str}); g = g[g.source == 'original_gold58'].set_index('StudyInstanceUID')
ids = [s for s in g.index if s in idx.index]; assert len(ids) == 58, len(ids)
Y = g.loc[ids, R.LAB].values.astype(float)
dev = 'cuda'
P = {}
for ckp in A.ck:
    t0 = time.time()
    ck = torch.load(ckp, map_location='cpu', weights_only=False)
    bb = timm.create_model(ck.get('arch', 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'), pretrained=False, num_classes=0, in_chans=3,
                           img_size=R.INPUT_RES, global_pool='avg')
    m = R.RaptorClassifier(bb, F_dim=bb.num_features); m.load_state_dict(ck['model'], strict=True); m.eval().to(dev)
    pr = []
    for sid in ids:
        vol = np.load(idx.at[sid, 'path']); mask = np.frombuffer(idx.at[sid, 'mask'].encode(), np.uint8) - 48
        xw = R.gpu_windows(vol, mask, R.K_EVAL, R.INPUT_RES, dev)
        xm = R.gpu_windows(vol, mask, R.K_EVAL, R.INPUT_RES, dev, mirror=True)
        pr.append(R.infer_pair_probs(m, xw, xm, dev))
    p = np.array(pr); assert np.isfinite(p).all()
    a = [roc_auc_score(Y[:, j], p[:, j]) for j in range(12)]
    name = os.path.basename(os.path.dirname(ckp)) + '/' + os.path.basename(ckp)
    P[name] = p; np.save(os.path.join(A.out, name.replace('/', '__') + '.npy'), p)
    print(f'SERVE {name}: gold58 macro {np.mean(a):.6f} | ' + ' '.join(f'{x:.3f}' for x in a) + f' | {time.time()-t0:.0f}s', flush=True)
    del m; torch.cuda.empty_cache()
if len(P) >= 2:
    k = list(P)
    rk = lambda p: np.stack([rankdata(p[:, j]) / len(p) for j in range(12)], 1)
    b = (rk(P[k[0]]) + rk(P[k[1]])) / 2
    a = [roc_auc_score(Y[:, j], b[:, j]) for j in range(12)]
    rho = [np.corrcoef(rk(P[k[0]])[:, j], rk(P[k[1]])[:, j])[0, 1] for j in range(12)]
    print(f'SERVE rank-mean({k[0]}, {k[1]}): gold58 macro {np.mean(a):.6f} | mean Spearman {np.mean(rho):.3f}', flush=True)
