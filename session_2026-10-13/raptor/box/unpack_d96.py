# unpack_d96.py (box CPU): Kaggle prep tar parts (one PNG mosaic per study) -> ~/raptor_store/rsna/<sid>.npy (96,384,384)
# uint8, each decoded volume checked against the receipt sha256 of the raw vol bytes, plus an index csv
# (id, path, src, n_valid, mask). Usage: python unpack_d96.py --parts DIR [DIR ...] --receipts CSV [CSV ...] --index OUT.csv
#   [--out ~/raptor_store/rsna] [--compare_dir DIR]  (compare: byte-compare with an existing .npy build, e.g. the box mirror)
import os, sys, io, glob, tarfile, hashlib, argparse, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np, pandas as pd, cv2
from multiprocessing import Pool
ap = argparse.ArgumentParser()
ap.add_argument('--parts', nargs='+', required=True); ap.add_argument('--receipts', nargs='+', required=True)
ap.add_argument('--index', required=True); ap.add_argument('--out', default='/home/training/raptor_store/rsna')
ap.add_argument('--compare_dir', default=''); ap.add_argument('--nproc', type=int, default=6)
A = ap.parse_args()
rec = pd.concat([pd.read_csv(r, dtype={'StudyInstanceUID': str}) for r in A.receipts]).set_index('StudyInstanceUID')
SHA = rec.vol_sha256.to_dict()


def do_tar(tp):
    cv2.setNumThreads(1); out = []
    with tarfile.open(tp) as tf:
        for m in tf:
            sid = m.name[:-4]; buf = np.frombuffer(tf.extractfile(m).read(), np.uint8)
            vol = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED).reshape(96, 384, 384)
            ok = hashlib.sha256(vol.tobytes()).hexdigest() == SHA.get(sid)
            p = f'{A.out}/{sid}.npy'
            np.save(p + '.tmp.npy', vol); os.replace(p + '.tmp.npy', p)
            mask = (vol.reshape(96, -1).sum(1) > 0).astype(np.uint8)
            same = maxd = fracd = ''
            if A.compare_dir and os.path.exists(f'{A.compare_dir}/{sid}.npy'):
                ref = np.load(f'{A.compare_dir}/{sid}.npy'); dd = np.abs(ref.astype(np.int16) - vol.astype(np.int16))
                same, maxd, fracd = int(dd.max() == 0), int(dd.max()), float((dd > 0).mean())
            out.append((sid, p, 'rsna', int(mask.sum()), ''.join(map(str, mask)), int(ok), same, maxd, fracd))
    return out


if __name__ == '__main__':
    os.makedirs(A.out, exist_ok=True)
    tars = sorted(t for d in A.parts for t in glob.glob(f'{d}/raptor_d96_s*_p*.tar'))
    t0 = time.time()
    with Pool(A.nproc) as pool:
        rows = [r for part in pool.map(do_tar, tars, chunksize=1) for r in part]
    df = pd.DataFrame(rows, columns=['id', 'path', 'src', 'n_valid', 'mask', 'sha_ok', 'same_as_compare', 'max_abs_diff', 'frac_px_diff'])
    print(f'{len(tars)} tars, {len(df)} studies in {time.time() - t0:.0f}s | sha ok {df.sha_ok.sum()}/{len(df)} | '
          f'receipt rows {len(rec)} | missing from tars {len(set(rec.index) - set(df.id))}')
    if A.compare_dir:
        c = df[df.same_as_compare != '']
        print(f'compare with {A.compare_dir}: {int((c.same_as_compare == 1).sum())}/{len(c)} byte-identical | max abs diff '
              f'{c.max_abs_diff.max()} | worst study frac of differing pixels {c.frac_px_diff.max()} | mean {c.frac_px_diff.mean()}')
        c.to_csv(A.index + '.compare.csv', index=False, columns=['id', 'same_as_compare', 'max_abs_diff', 'frac_px_diff'])
    assert df.sha_ok.all(), 'sha mismatch'
    df[['id', 'path', 'src', 'n_valid', 'mask']].to_csv(A.index, index=False)
    print('wrote', A.index)
