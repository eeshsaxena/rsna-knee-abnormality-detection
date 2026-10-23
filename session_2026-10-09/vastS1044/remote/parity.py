# parity.py: environment + contract parity probe, run on the BOX and on the INSTANCE with the same arguments; the
# laptop (vast_setup.sh) compares the two JSON outputs. CPU only (run it with CUDA_VISIBLE_DEVICES= ), writes nothing.
# Everything here is recomputed exactly the way fleet_train.py computes it:
#   contract pixel_sha / tensor_sha, select_sha, jpeg_fp (cv2 build), model_sha (ttr + base source), fold counts,
#   the 4-study decode probe (sorted ids[:4], decode_sha) and a sha of the pretrained ConvNeXt backbone tensors.
import os, sys, json, hashlib, inspect, argparse
ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--store', required=True)
ap.add_argument('--labels', required=True); ap.add_argument('--folds_csv', required=True)
ap.add_argument('--exclude', required=True); ap.add_argument('--folds', nargs='*', type=int, default=[])
ap.add_argument('--no_model', action='store_true')
A = ap.parse_args()
sys.path.insert(0, A.code); os.chdir(A.code)
import numpy as np, pandas as pd, torch, timm, cv2, sklearn
import fleet_common as fc
out = {'versions': {'python': sys.version.split()[0], 'torch': torch.__version__, 'timm': timm.__version__,
                    'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'cv2': cv2.__version__}}
out['contract'] = fc.CONTRACT
out['select_sha'] = fc.SELECT_SHA
out['jpeg_fp'] = fc.jpeg_fingerprint()
import fleet_model as _fm, fleet_model_ttr as _tt                       # == fleet_train._model_sha() for --model ttr
src = chr(10).join(inspect.getsource(_tt).splitlines()) + chr(10) + chr(10).join(inspect.getsource(_fm).splitlines())
out['model_sha'] = hashlib.sha256(src.encode()).hexdigest()[:16]
lab = pd.read_csv(A.labels, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
folds = pd.read_csv(A.folds_csv, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['fold']
excl = set(pd.read_csv(A.exclude, dtype={'StudyInstanceUID': str}).StudyInstanceUID)
held = set(folds.index[folds < 0]) | excl
have = {s for s in lab.index if os.path.exists(f'{A.store}/{s}.npz')}
ids = [s for s in lab.index if s in folds.index and s in have]
out['n'] = {'labels': len(lab), 'store_has': len(have), 'held': len(held), 'ids': len(ids)}
out['fold_counts'] = {str(F): [sum(1 for s in ids if s not in held and folds[s] != F),
                               sum(1 for s in ids if s not in held and folds[s] == F)] for F in A.folds}
probe = sorted(ids)[:4]                                                  # == fleet_train decode_probe ids
dp = []
for s in probe:
    with np.load(f'{A.store}/{s}.npz') as z:
        fc.check_store_contract(z, s)
        dp.append(fc.decode_sha(z))
out['decode_probe'] = dp                                                 # hashes only, no study ids printed
if not A.no_model:
    from fleet_model_ttr import KneeFleetNetTTR
    m = KneeFleetNetTTR(ttr_size=int(os.environ.get('TTR', '288')), backbone='convnext_tiny.fb_in22k_ft_in1k', dim=256, contrast_emb=True,
                        n_views=len(fc.VIEWS), n_slices=fc.N_SLICES, n_contrast=fc.N_CONTRAST, pretrained=True,
                        drop_path=0.2, dropout=0.3, grad_ckpt=False)
    h = hashlib.sha256()
    for k, v in m.cnn.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    out['backbone_sha'] = h.hexdigest()[:16]
    out['state_keys'] = len(m.state_dict())
    out['pretrained_cfg'] = {k: m.cnn.pretrained_cfg.get(k) for k in ('hf_hub_id', 'mean', 'std')}
print(json.dumps(out, sort_keys=True, default=str))
