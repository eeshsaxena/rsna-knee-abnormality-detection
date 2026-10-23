# pack_fleet_ds.py -- laptop side. Build the eeshsaxena/rsna-knee-cnxt-fleet dataset folder:
# fleet_common.py + fleet_model.py + fleet_worker.py (copied from THIS folder, i.e. the files the box
# trained with), the chosen members, fleet_manifest.json and parity_<sid>.npz (the trainer's decode-probe
# studies, from the store). Refuses members that do not match this fleet_common's CONTRACT, bf16/fp32
# members, and two members with the same fold/seed (raw AND ema of one run).
# usage: python pack_fleet_ds.py --out <dir> --store <store dir> [--title T] ckpt1.pt [ckpt2.pt ...]
# fleet2 patch: also ships fleet_model_f2.py (family 2) and checks every member's model_sha against the code being
# shipped, per family, exactly as the fleet2 worker will (a mismatch now fails on the laptop, not in a Kaggle rerun).
import os, sys, json, shutil, argparse, hashlib
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fleet_common as fc
ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True); ap.add_argument('--store', required=True)
ap.add_argument('--id', default='eeshsaxena/rsna-knee-cnxt-fleet'); ap.add_argument('--title', default='rsna-knee-cnxt-fleet')
ap.add_argument('members', nargs='+')
A = ap.parse_args()
keys = ['_'.join(os.path.basename(p).split('_')[:2]) for p in A.members]
assert len(keys) == len(set(keys)), f'duplicate fold/seed members: {sorted(keys)}'
shutil.rmtree(A.out, ignore_errors=True); os.makedirs(A.out)
for f in ('fleet_common.py', 'fleet_model.py', 'fleet_model_f2.py', 'fleet_worker.py'):
    shutil.copy2(f'{HERE}/{f}', A.out)


def _model_sha(mod='fleet_model'):                     # == fleet_worker._model_sha (fleet2)
    import inspect, fleet_model as _fm
    src = chr(10).join(inspect.getsource(_fm).splitlines())
    if mod == 'fleet_model_f2':
        import fleet_model_f2 as _f2
        src = chr(10).join(inspect.getsource(_f2).splitlines()) + chr(10) + src
    return hashlib.sha256(src.encode()).hexdigest()[:16]


probe = set()
for p in A.members:
    a = torch.load(p, map_location='cpu', weights_only=False)['args']
    assert a.get('contract') == fc.CONTRACT, (p, 'contract differs from this fleet_common')
    assert a.get('amp') == 'fp16' and not a.get('bf16', False), (p, a.get('amp'))
    mod = a.get('model_mod', 'fleet_model')
    assert mod in ('fleet_model', 'fleet_model_f2'), (p, 'unknown model_mod', mod)
    assert a.get('model_sha') in (None, _model_sha(mod)), (p, f'{mod} source changed since training', a.get('model_sha'))
    probe |= set(a.get('decode_probe', {}))
    shutil.copy2(p, A.out)
    print('member', os.path.basename(p), '| fold', a['fold'], 'seed', a['seed'], '| sel', a.get('sel_labels'),
          '| train_slices', a.get('train_slices'), '|', mod, a['backbone'], '| train_labels', a.get('train_labels'))
for s in sorted(probe):
    shutil.copy2(f'{A.store}/{s}.npz', f'{A.out}/parity_{s}.npz')   # flat: no sub-folder
json.dump({'checkpoints': [os.path.basename(p) for p in A.members]}, open(f'{A.out}/fleet_manifest.json', 'w'), indent=1)
json.dump({'title': A.title, 'id': A.id, 'licenses': [{'name': 'CC0-1.0'}]}, open(f'{A.out}/dataset-metadata.json', 'w'), indent=1)
sha = hashlib.sha256(open(f'{A.out}/fleet_common.py', 'rb').read()).hexdigest()[:16]
print(f'{A.out}: {len(A.members)} members, parity {sorted(probe)}, fleet_common sha256[:16] {sha}')
