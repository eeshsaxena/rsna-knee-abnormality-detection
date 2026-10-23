# verify_ckpt.py (gold_prep/r50verify, CPU, nice 19, HF offline): independent re-check of the r50verify dry-run ckpts.
# 1) model_sha recomputed with the worker's own algorithm from the K5 shipped code (~/gold_prep/r50/k5code)
# 2) worker-style construction (pretrained=False, drop_path 0, chunk 48) + strict load
# 3) trainer EVAL path re-implemented from the store npz (not DICOM): forward vs the trainer's OOF rows
# 4) OOF macro AUC vs v4 / mix50 recomputed with the trainer's macro_auc rules (val 6 studies) -> must equal the log
# 5) frozen BN: running stats == timm pretrained (HF cache, offline)
import os, sys, glob, hashlib, inspect, numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 4)))
H = os.path.expanduser('~'); G = f'{H}/gold_prep/r50'; RUN = sys.argv[1] if len(sys.argv) > 1 else f'{G}/dryA'
sys.path.insert(0, f'{G}/k5code')
import fleet_common as fc, fleet_model as _fm, fleet_model_f2 as _f2
from fleet_model_f2 import KneeFleetNetF2
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
src = chr(10).join(inspect.getsource(_f2).splitlines()) + chr(10) + chr(10).join(inspect.getsource(_fm).splitlines())
sha = hashlib.sha256(src.encode()).hexdigest()[:16]
v4 = pd.read_csv(f'{H}/rsna_train/llm_labels_v4_blend.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
mix = pd.read_csv(f'{H}/rsna_train/labels_mix50_logit.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
def macro_auc(P, ids, tab):
    Y = tab.loc[ids, T].to_numpy(np.float64); a = []
    for j in range(len(T)):
        ok = np.isfinite(Y[:, j]) & np.isfinite(P[:, j]); yb = Y[ok, j] >= 0.5
        if ok.sum() and 0 < yb.mean() < 1:
            a.append(roc_auc_score(yb, P[ok, j]))
    return (float(np.mean(a)) if a else float('nan')), len(a)
def load_x(sid):
    vs, pr, ct = [], [], []
    with np.load(f'{G}/store/{sid}.npz') as z:
        fc.check_store_contract(z, sid)
        for j in range(len(fc.VIEWS)):
            if f'v{j}_blob' in z.files:
                st = fc.decode_stack(z[f'v{j}_blob'], z[f'v{j}_offs'])
                vs.append(fc.sample_triplets(st, fc.N_SLICES, 0.0)); pr.append(True); ct.append(int(z[f'v{j}_ctr']))
            else:
                vs.append(np.zeros((fc.N_SLICES, 3, fc.STORE, fc.STORE), np.uint8)); pr.append(False); ct.append(3)
    return torch.from_numpy(np.stack(vs))[None], torch.tensor(pr)[None], torch.tensor(ct)[None]
import timm
ref = timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=0).state_dict()
for tag in ('raw', 'ema'):
    p = sorted(glob.glob(f'{RUN}/f0_s243_{tag}_*_best.pt'))[0]
    ck = torch.load(p, map_location='cpu', weights_only=False); a = ck['args']
    print(f'== {os.path.basename(p)} | model_mod {a.get("model_mod")} | ckpt sha {a.get("model_sha")} | worker-algo sha {sha} '
          f'| match {a.get("model_sha") == sha} | contract == fc {a.get("contract") == fc.CONTRACT} | amp {a.get("amp")} '
          f'| bf16 {a.get("bf16")} | n_slices {a["n_slices"]} train_slices {a.get("train_slices")} | fold {a["fold"]} seed {a["seed"]}')
    kw = dict(backbone=a['backbone'], n_views=a['n_views'], n_slices=a['n_slices'], dim=a['dim'], n_contrast=a['n_contrast'],
              contrast_emb=a['contrast_emb'], pretrained=False, drop_path=0.0, chunk=48)
    m = KneeFleetNetF2(freeze_bn=a['freeze_bn'], channels_last=a.get('channels_last', True), **kw)
    miss, unexp = m.load_state_dict(ck['model'], strict=False)
    print(f'   strict-load miss {len(miss)} unexp {len(unexp)} | norm buffers m {m.m.flatten().tolist()} s {m.s.flatten().tolist()}')
    m = m.eval().to(memory_format=torch.channels_last)
    oof = pd.read_csv(f'{RUN}/oof_f0_s243_{tag}.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    ids = oof.index.tolist(); P = []
    with torch.no_grad():
        for s in ids:
            x, pr, ct = load_x(s); P.append(torch.sigmoid(m(x, pr, ct).float()).numpy()[0])
    P = np.stack(P); O = oof[T].to_numpy(np.float32)
    print(f'   eval-path re-forward vs OOF: max |diff| {np.abs(P - O).max():.2e} over {len(ids)} val studies')
    c4, n4 = macro_auc(O.astype(np.float64), ids, v4); cm, nm = macro_auc(O.astype(np.float64), ids, mix)
    print(f'   OOF macro AUC vs v4 {c4:.4f} ({n4} labels) | vs mix50 {cm:.4f} | ckpt cv {ck["cv"]:.4f}')
    rs = [k for k in ref if k.endswith(('running_mean', 'running_var'))]
    print(f'   BN running stats == timm pretrained: {all(torch.equal(ck["model"]["cnn." + k], ref[k]) for k in rs)} ({len(rs)})')
