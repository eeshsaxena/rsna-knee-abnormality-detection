# smoke only: a tiny REAL TTR member from the dry family-1 member (same weights, KneeFleetNetTTR at ttr_size 48)
import sys, os, copy, hashlib, inspect, torch
code, src, dst = sys.argv[1:4]
sys.path.insert(0, code)
import fleet_model as fm, fleet_model_ttr as tt
ck = torch.load(src, map_location='cpu', weights_only=False)
a = dict(ck['args'])
kw = dict(backbone=a['backbone'], n_views=a['n_views'], n_slices=a['n_slices'], dim=a['dim'], n_contrast=a['n_contrast'],
          contrast_emb=a['contrast_emb'], pretrained=False, drop_path=0.0, chunk=48)
m = tt.KneeFleetNetTTR(ttr_size=48, **kw)
miss, unexp = m.load_state_dict(ck['model'], strict=False)
assert list(miss) == ['ttr_size_t'] and not unexp, (miss, unexp)
sha = hashlib.sha256((chr(10).join(inspect.getsource(tt).splitlines()) + chr(10) +
                      chr(10).join(inspect.getsource(fm).splitlines())).encode()).hexdigest()[:16]
a.update(seed=1042, model_mod='fleet_model_ttr', ttr_size=48, model_sha=sha)
ck2 = dict(ck); ck2['args'] = a; ck2['model'] = m.state_dict()
torch.save(ck2, dst)
print('TTR', os.path.basename(dst), 'sha', sha, 'ttr_size_t', int(ck2['model']['ttr_size_t']))
