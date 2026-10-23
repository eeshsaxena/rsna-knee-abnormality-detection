# loader_rate.py -- CPU only: how fast does the trainer's heavy-aug TRAIN DataLoader (bs 2, 10 workers, pin_memory off
# since no CUDA here) deliver batches? Needed rate for grad_ckpt 0 alone ~ 1 / s_per_micro batches/s. No GPU touched.
import os, sys, time, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser('~/fast_build/code2'))
import numpy as np, pandas as pd, torch
import fleet_common as fc
from fleet_aug import heavy_study
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
           'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
H = os.path.expanduser('~')
W = int(sys.argv[1]) if len(sys.argv) > 1 else 10
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
lab = pd.read_csv(f'{H}/rsna_train/labels_mix50_logit.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
folds = pd.read_csv(f'{H}/rsna_train/folds5.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['fold']
ids = [s for s in lab.index if s in folds.index and folds[s] >= 0 and folds[s] != 2]
PLANES = [p for p, _ in fc.VIEWS]


def _wif(_):
    import cv2; cv2.setNumThreads(0)


class DS(torch.utils.data.Dataset):
    def __len__(self):
        return len(ids)
    def __getitem__(self, i):
        sid = ids[i]
        with np.load(f'{H}/fleet_store/{sid}.npz') as z:
            fc.check_store_contract(z, sid)
            x, pr, ct, pos, _ = heavy_study(z, sid, 16, PLANES)
        y = lab.loc[sid, TARGETS].to_numpy(np.float32)
        pos = torch.from_numpy(pos)
        return x, torch.tensor(pr), torch.tensor(ct), pos[None].expand(len(fc.VIEWS), -1).clone(), torch.from_numpy(y)


torch.manual_seed(7)
dl = torch.utils.data.DataLoader(DS(), batch_size=2, shuffle=True, drop_last=True, num_workers=W, worker_init_fn=_wif,
                                 persistent_workers=True)
it = iter(dl)
for _ in range(3 * W):
    next(it)
t0 = time.time()
for _ in range(N):
    next(it)
dt = time.time() - t0
print('LOADER ' + json.dumps(dict(workers=W, batches=N, seconds=round(dt, 2), batches_per_s=round(N / dt, 2),
                                  s_per_batch=round(dt / N, 4), load_avg=os.getloadavg())), flush=True)
