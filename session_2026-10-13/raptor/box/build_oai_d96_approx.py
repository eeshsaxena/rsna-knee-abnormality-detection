# build_oai_d96_approx.py (box CPU only; OAI stays on the box, output dir mode 700): Raptor d96 volumes for the 2,399
# MOAKS-labelled V00 knees, derived from our 320 px fleet store (~/oai_store/oai_<subject>_<R|L>.npz; the raw DICOMs were
# pruned 10-09). Method = approx_vol() of approx_parity.py, validated 10-13 on gold58 (N: approxB .9205 vs exactB .9184):
#   slot 0 Sagittal FS  <- store v3 (SAG_IW_TSE)      slot 1 Sagittal non-FS <- (none for OAI: empty)
#   slot 2 Coronal FS   <- store v2 (COR_MPR)         slot 3 Coronal non-FS  <- (COR_IW_TSE not stored: empty)
#   slot 4 Axial        <- store v0 (AX_MPR)
#   per slot: linspace over the stored stack's [2 %, 98 %] span, 2/98 percentile re-window on the uint8 of the picked
#   slices, INTER_LINEAR 320 -> 384, uint8.
# Output: ~/raptor_store/oai/<kid>.npy (96, 384, 384) uint8 + ~/raptor_store/oai_index.csv (id, path, src, n_valid, mask).
import os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np, pandas as pd, cv2
from multiprocessing import Pool
H = '/home/training'
OUT = f'{H}/raptor_store/oai'
SIZES = [26, 22, 18, 12, 18]; B = np.cumsum([0] + SIZES)
SLOT2VIEW = {0: 3, 1: 1, 2: 2, 3: None, 4: 0}


def decode(z, j):
    if f'v{j}_blob' not in z.files:
        return None
    blob, offs = z[f'v{j}_blob'], z[f'v{j}_offs']
    return np.stack([cv2.imdecode(blob[offs[i]:offs[i + 1]], cv2.IMREAD_GRAYSCALE) for i in range(len(offs) - 1)])


def approx_vol(npz):
    z = np.load(npz)
    vol = np.zeros((96, 384, 384), np.uint8)
    for s, k in enumerate(SIZES):
        j = SLOT2VIEW[s]
        if j is None:
            continue
        st = decode(z, j)
        if st is None:
            continue
        n = len(st); lo, hi = int(n * 0.02), int(n * 0.98) - 1; hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        sel = st[np.minimum(picks, n - 1)].astype(np.float32)
        q2, q98 = np.percentile(sel, [2.0, 98.0])
        for t, a in enumerate(sel):
            aw = np.clip((a - q2) / (q98 - q2 + 1e-6), 0, 1)
            aw = cv2.resize(aw, (384, 384), interpolation=cv2.INTER_LINEAR)
            vol[B[s] + t] = (aw * 255).astype(np.uint8)
    mask = (vol.reshape(96, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask


def one(kid):
    cv2.setNumThreads(1)
    try:
        vol, mask = approx_vol(f'{H}/oai_store/{kid}.npz')
        p = f'{OUT}/{kid}.npy'
        np.save(p + '.tmp.npy', vol); os.replace(p + '.tmp.npy', p)
        return kid, p, int(mask.sum()), ''.join(map(str, mask)), ''
    except Exception as e:
        return kid, '', -1, '', f'{type(e).__name__}: {e}'


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True); os.chmod(f'{H}/raptor_store', 0o700); os.chmod(OUT, 0o700)
    lab = pd.read_csv(f'{H}/oai_build/oai_labels_raptorT1.csv')
    kids = lab.StudyInstanceUID.tolist()
    assert len(kids) == 2399 and all(k.startswith('oai_') and '_v' not in k for k in kids)
    t0 = time.time()
    with Pool(int(os.environ.get('NPROC', 6))) as pool:
        res = pool.map(one, kids, chunksize=8)
    df = pd.DataFrame(res, columns=['id', 'path', 'n_valid', 'mask', 'err']); df['src'] = 'oai'
    print(f'{len(df)} knees in {time.time() - t0:.0f}s | failed {(df.n_valid < 0).sum()} | n_valid',
          df.n_valid.describe().round(1).to_dict(), '| errors', df.err[df.err != ''].str[:60].value_counts().head(5).to_dict())
    p = f'{H}/raptor_store/oai_index.csv'
    df[df.n_valid > 0][['id', 'path', 'src', 'n_valid', 'mask']].to_csv(p, index=False); os.chmod(p, 0o600)
    print('wrote', p)
