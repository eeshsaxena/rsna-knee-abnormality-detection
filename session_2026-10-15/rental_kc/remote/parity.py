# parity.py (rental_kc): environment + contract parity probe, run on the BOX (on the packed bundle) and on the INSTANCE with
# the same arguments; the box driver compares the two JSON outputs field by field (D4). CPU only (CUDA_VISIBLE_DEVICES= ),
# HF offline from the bundled cache, writes nothing. Recomputed exactly the way fleet_train_kc.py computes it:
#   kc_common CONTRACT_HASH / SELECT_SHA / jpeg_fingerprint, model_sha of the TTR path (fleet_model_ttr + fleet_model) and
#   of the KC2 f2 path (fleet_model_f2 + fleet_model), the decode probe (sorted store ids[:4], fc.decode_sha) plus a sha of
#   4 fully decoded stacks (cv2 PNG decode), the ConvNeXt-S backbone tensors after the --pretrained_file load (KC-4) and the
#   EffNetV2-S backbone tensors from the bundled HF cache, code md5s, label table row counts. No study id is printed.
import os, sys, json, hashlib, inspect, argparse, glob
ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--store', required=True)
ap.add_argument('--tables', nargs='+', required=True); ap.add_argument('--pretrained', required=True)
ap.add_argument('--no_model', action='store_true')
A = ap.parse_args()
sys.path.insert(0, A.code); os.chdir(A.code)
import numpy as np, pandas as pd, torch, timm, cv2, sklearn, safetensors
import kc_common as fc
out = {'versions': {'python': sys.version.split()[0], 'torch': torch.__version__.split('+')[0], 'timm': timm.__version__,
                    'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'cv2': cv2.__version__,
                    'safetensors': safetensors.__version__}}
out['contract_hash'] = fc.CONTRACT_HASH
out['select_sha'] = fc.SELECT_SHA
out['jpeg_fp'] = fc.jpeg_fingerprint()
import fleet_model as _fm, fleet_model_ttr as _tt, fleet_model_f2 as _f2
base = chr(10).join(inspect.getsource(_fm).splitlines())
out['model_sha_ttr'] = hashlib.sha256((chr(10).join(inspect.getsource(_tt).splitlines()) + chr(10) + base).encode()).hexdigest()[:16]
out['model_sha_f2'] = hashlib.sha256((chr(10).join(inspect.getsource(_f2).splitlines()) + chr(10) + base).encode()).hexdigest()[:16]
out['code_md5'] = {os.path.basename(p): hashlib.md5(open(p, 'rb').read()).hexdigest() for p in sorted(glob.glob(A.code + '/*.py'))}
ids = sorted(f[:-4] for f in os.listdir(A.store) if f.endswith('.npz'))
out['n_store'] = len(ids)
out['tables'] = {os.path.basename(t): int(len(pd.read_csv(t, dtype={'StudyInstanceUID': str}))) for t in A.tables}
dp, full = [], hashlib.sha256()
for s in ids[:4]:
    with np.load(f'{A.store}/{s}.npz') as z:
        fc.check_store_contract(z, s)
        dp.append(fc.decode_sha(z))
        for j in range(len(fc.VIEWS)):
            if f'v{j}_blob' in z.files:
                full.update(np.ascontiguousarray(fc.decode_stack(z[f'v{j}_blob'], z[f'v{j}_offs'])).tobytes())
out['decode_probe'] = dp
out['decode_full_sha'] = full.hexdigest()[:16]


def sd_sha(m):
    h = hashlib.sha256()
    for k, v in m.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


if not A.no_model:
    from safetensors.torch import load_file
    from fleet_model_ttr import KneeFleetNetTTR
    from fleet_model_f2 import KneeFleetNetF2
    out['pretrained_sha256'] = hashlib.sha256(open(A.pretrained, 'rb').read()).hexdigest()[:16]
    m = KneeFleetNetTTR(ttr_size=256, backbone='convnext_small.fb_in22k_ft_in1k', dim=256, contrast_emb=False,
                        n_views=len(fc.VIEWS), n_slices=fc.N_SLICES, n_contrast=fc.N_CONTRAST, pretrained=False,
                        drop_path=0.3, dropout=0.3, grad_ckpt=False)
    miss, unexp = m.cnn.load_state_dict(load_file(A.pretrained), strict=False)
    assert not miss and set(unexp) <= {'head.fc.weight', 'head.fc.bias'}, (miss, unexp)
    out['backbone_sha_cnxs'] = sd_sha(m.cnn); out['state_keys_ttr'] = len(m.state_dict())
    m2 = KneeFleetNetF2(backbone='tf_efficientnetv2_s.in21k_ft_in1k', dim=256, contrast_emb=False, n_views=len(fc.VIEWS),
                        n_slices=fc.N_SLICES, n_contrast=fc.N_CONTRAST, pretrained=True, drop_path=0.2, dropout=0.3,
                        grad_ckpt=False, freeze_bn=True, init_ckpt='', channels_last=True)
    out['backbone_sha_ev2s'] = sd_sha(m2.cnn); out['state_keys_f2'] = len(m2.state_dict())
    out['pretrained_cfg_ev2s'] = {k: m2.cnn.pretrained_cfg.get(k) for k in ('hf_hub_id', 'mean', 'std')}
print(json.dumps(out, sort_keys=True, default=str))
