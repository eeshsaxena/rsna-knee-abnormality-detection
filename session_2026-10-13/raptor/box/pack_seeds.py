# pack_seeds.py (box CPU): Kaggle dataset folder for the RSNA-only Raptor seeds that PASSED PREREG_RAPTOR_RENTED (G0 + G1 and the
# ensemble legs). Copies ~/raptor_store/rented/s<seed>/swa.pt -> OUT/raptor_rsna_s<seed>_swa.pt (bytes unchanged), scans each
# checkpoint dict for any OAI trace (keys, metadata strings), writes manifest.json (sha256, gold58, epochs) + seeds.json (for
# build_hr7rs.py) + dataset-metadata.json (id eeshsaxena/rsna-knee-raptor-seeds, PRIVATE; main creates the dataset).
# usage: python pack_seeds.py --seeds "43 44 45 46" [--out ~/raptor_store/kaggle_seeds]
import os, json, shutil, hashlib, argparse
import torch
H = os.path.expanduser('~')
ap = argparse.ArgumentParser(); ap.add_argument('--seeds', required=True); ap.add_argument('--out', default=f'{H}/raptor_store/kaggle_seeds')
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
man, seeds = {}, []
for s in A.seeds.split():
    src = f'{H}/raptor_store/rented/s{s}/swa.pt'; dst = f'{A.out}/raptor_rsna_s{s}_swa.pt'
    ck = torch.load(src, map_location='cpu', weights_only=False)
    meta = {k: v for k, v in ck.items() if k != 'model'}
    blob = json.dumps(meta, default=str).lower() + ' '.join(ck['model'].keys()).lower()
    assert 'oai' not in blob, f's{s}: OAI string in the checkpoint dict'
    assert set(ck) == {'model', 'gold_auc', 'aucs', 'src', 'res', 'arch', 'lab', 'epoch', 'swa_over'}, sorted(ck)
    shutil.copyfile(src, dst); h = sha(dst); assert h == sha(src)
    man[os.path.basename(dst)] = dict(sha256=h, bytes=os.path.getsize(dst), seed=int(s), arch=ck['arch'], res=ck['res'],
                                      swa_over=ck['swa_over'], gold58_train_geometry=round(float(ck['gold_auc']), 6),
                                      data='RSNA competition training studies only (4,349; gold58 excluded); no external images or labels',
                                      recipe='nartaa Raptor .949 recipe (Kaggle discussion 746792) minus OAI rows; train_raptor.py')
    seeds.append(dict(file=os.path.basename(dst), sha256=h))
    print('packed', dst, h[:16], 'gold58', round(float(ck['gold_auc']), 4))
json.dump(man, open(f'{A.out}/manifest.json', 'w'), indent=1)
json.dump(seeds, open(f'{A.out}/seeds.json', 'w'), indent=1)
json.dump({'title': 'rsna-knee-raptor-seeds', 'id': 'eeshsaxena/rsna-knee-raptor-seeds', 'licenses': [{'name': 'other'}],
           'isPrivate': True}, open(f'{A.out}/dataset-metadata.json', 'w'), indent=1)
print('OK', len(seeds), 'seeds ->', A.out)
