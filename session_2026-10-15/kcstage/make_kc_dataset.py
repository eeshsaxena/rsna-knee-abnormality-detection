#!/usr/bin/env python
# make_kc_dataset.py (10-15, Claude subagent, kcstage): assemble the PRIVATE KC dataset dir the S6F+KC kernel reads
# (A6 G3 / MANIFEST_1015 KC_DATASET): the frozen KC code (kc_worker.py 4af60f6f, kc_common.py 45255ae0, d96_reader_src.py sha 30aa72fd,
# fleet_model.py 897f83b2, fleet_model_ttr.py bcdf7d15, the side kernel kc_side_kernel.py sha 2979c191) + the member ckpts + KC_MANIFEST.json
# (sha256 of every file, the ordered member list) + dataset-metadata.json (for MAIN's `kaggle datasets create -p DIR`; never run here).
# The builder (build_v224s6fkc.py --kc-dir DIR) bakes sha256(KC_MANIFEST.json) into the SETUP cell.
#   python make_kc_dataset.py --out DIR --slug OWNER/NAME --kind kc1|standin --members a.pt[=role] b.pt[=role] [--code kc_src]
# With torch importable, every member is opened and checked like kc_worker.load_members does (model_mod kc_ttr, contract 48a46a0d,
# mirror 0, model_sha == sha(fleet_model_ttr + fleet_model) of the shipped files) and its args are summarised in the manifest.
# RSNA-only: a member whose args name an OAI path or label set is refused (the full OAI-string scan stays MAIN's step before upload).
import os, sys, json, time, shutil, hashlib, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_MD5 = {'kc_worker.py': '4af60f6ff7b94b3acab9dceb52162a18', 'kc_common.py': '45255ae0e361ef787b290d7510554a9d',
            'fleet_model.py': '897f83b272fbb144a4d3deab773d9064', 'fleet_model_ttr.py': 'bcdf7d153f339962b97d2736d8baed08'}
CODE_SHA = {'d96_reader_src.py': '30aa72fd972e29cf42c752aa5394a6c88a175f17b98740b1f6b0e0f55553ac40',
            'kc_side_kernel.py': '2979c1915e0cecf2c4792526a5581f93a723fac368eacff18849661dc0bdcb62'}
CONTRACT_HASH = '48a46a0d63ecfc59'
MODEL_SHA = 'f141aeed136dfc01'                       # sha(fleet_model_ttr.py + fleet_model.py)[:16], MEASURED on the s7 ckpt


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def member_info(path):
    try:
        import torch
    except Exception:
        return {'checked': False, 'why': 'torch not importable here'}
    ck = torch.load(path, map_location='cpu', weights_only=False)
    a = ck['args']
    assert a.get('model_mod') == 'kc_ttr' and a.get('kc_contract_hash') == CONTRACT_HASH and a.get('mirror') == 0, \
        (path, a.get('model_mod'), a.get('kc_contract_hash'), a.get('mirror'))
    assert a.get('model_sha') == MODEL_SHA, (path, a.get('model_sha'))
    blob = json.dumps(a, default=str).lower()
    for bad in ('oai', 'osteoarthritis initiative', 'kmimic/oai'):
        assert bad not in blob, (path, 'member args name %r' % bad)
    sd = ck['model']
    dt = sorted({str(v.dtype).replace('torch.', '') for v in sd.values()})
    return {'checked': True, 'model_mod': a['model_mod'], 'model_sha': a['model_sha'], 'kc_contract_hash': a['kc_contract_hash'],
            'mirror': a['mirror'], 'backbone': a.get('backbone'), 'seed': a.get('seed'), 'fold': a.get('fold'),
            'train_labels': a.get('train_labels'), 'epoch': ck.get('epoch'), 'cv': ck.get('cv'), 'standin': bool(a.get('standin')),
            'state_dtypes': dt, 'n_tensors': len(sd), 'n_params': int(sum(v.numel() for v in sd.values() if v.is_floating_point()))}


ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--slug', required=True)
ap.add_argument('--kind', required=True, choices=('kc1', 'standin'))
ap.add_argument('--members', nargs='+', required=True)
ap.add_argument('--code', default=os.path.join(HERE, 'kc_src'))
ap.add_argument('--title', default=None)
A = ap.parse_args()
assert not os.path.exists(A.out) or not os.listdir(A.out), 'output dir not empty: %s' % A.out
os.makedirs(A.out, exist_ok=True)
files = {}
for f, h in CODE_MD5.items():
    src = os.path.join(A.code, f)
    assert hashlib.md5(open(src, 'rb').read()).hexdigest() == h, (f, 'md5 differs from the A6 freeze')
    shutil.copyfile(src, os.path.join(A.out, f))
for f, h in CODE_SHA.items():
    src = os.path.join(A.code, f)
    assert sha256(src) == h, (f, 'sha256 differs from the pin')
    shutil.copyfile(src, os.path.join(A.out, f))
members = []
for spec in A.members:
    p, role = (spec.split('=', 1) + [''])[:2]
    name = os.path.basename(p)
    assert name not in CODE_MD5 and name not in CODE_SHA and name.endswith('.pt'), name
    info = member_info(p)
    if A.kind == 'kc1':
        assert info.get('checked') and not info.get('standin'), ('a kc1 dataset needs checked, real members', name, info)
    shutil.copyfile(p, os.path.join(A.out, name))
    members.append({'file': name, 'role': role or None, **info})
for f in sorted(os.listdir(A.out)):
    files[f] = sha256(os.path.join(A.out, f))
for m in members:
    m['sha256'] = files[m['file']]
man = {'kind': A.kind, 'standin': A.kind == 'standin', 'slug': A.slug, 'contract_hash': CONTRACT_HASH, 'model_sha': MODEL_SHA,
       'members': members, 'files': files, 'code_md5': CODE_MD5, 'code_sha256': CODE_SHA,
       'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'rsna_only': True,
       'serve': 'kc_serve.py (kcstage) runs kc_worker.py __main__ with the side table of kc_side_kernel.py study(); RAW weights at E*'}
raw = json.dumps(man, indent=1, sort_keys=True).encode('utf-8')
open(os.path.join(A.out, 'KC_MANIFEST.json'), 'wb').write(raw)
owner, name = A.slug.split('/')
json.dump({'title': A.title or name.replace('-', ' '), 'id': A.slug, 'licenses': [{'name': 'CC0-1.0'}], 'isPrivate': True},
          open(os.path.join(A.out, 'dataset-metadata.json'), 'w'), indent=1)
print('KC_DATASET', A.out, 'manifest sha256', hashlib.sha256(raw).hexdigest(), 'members', [m['file'] for m in members],
      'bytes', sum(os.path.getsize(os.path.join(A.out, f)) for f in os.listdir(A.out)))
