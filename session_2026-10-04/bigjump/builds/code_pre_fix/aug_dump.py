# aug_dump.py -- G-P0 helper (BIG_JUMP_PLAN sec 4): look at what --aug heavy feeds the TTR model, and time it. CPU only.
#  (1) PNG grid for ONE store study: row 0 = no aug (K=16 slots of the 24-slot sampling, centre channel), then --draws
#      rows of fleet_aug.heavy_study draws (the exact trainer code), per view; written as <out>/aug_<sid>_v<j>.png
#  (2) mirror check on that study: with the pixel aug off, a mirrored draw's sagittal slot k == source slot 23-k with
#      the triplet channels reversed, axial/coronal == W flip (prints PASS/FAIL)
#  (3) per-study CPU cost of the TRAIN item, light (the default fleet_train path, replicated: 24 slots, light affine +
#      gain/bias, mirror .5) vs heavy (heavy_study, K slots), over --n_time store studies, one process, cv2 threads 0
# usage (box): CUDA_VISIBLE_DEVICES= python aug_dump.py --store ~/fleet_store --sid <sid> --out <dir> [--n_time 12]
import os, sys, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, torch
import fleet_common as fc
from fleet_model import mirror_views
from fleet_aug import HEAVY, heavy_study, aug_view_heavy

PLANES = [p for p, _ in fc.VIEWS]
N, V = fc.N_SLICES, len(fc.VIEWS)


def light_item(z):                                    # == fleet_train.DS.__getitem__ train path (default), replicated
    import cv2
    vs, pr = [], []
    for j in range(V):
        if f'v{j}_blob' in z.files:
            st = fc.decode_stack(z[f'v{j}_blob'], z[f'v{j}_offs'])
            t = fc.sample_triplets(st, N, np.random.uniform(-0.5, 0.5))
            H = t.shape[-1]
            Mx = cv2.getRotationMatrix2D((H / 2, H / 2), np.random.uniform(-10, 10), np.random.uniform(0.9, 1.1))
            Mx[:, 2] += np.random.uniform(-0.05, 0.05, 2) * H
            out = np.empty_like(t)
            for s in range(t.shape[0]):
                for c in range(3):
                    out[s, c] = cv2.warpAffine(t[s, c], Mx, (H, H), flags=cv2.INTER_LINEAR, borderValue=0)
            t = np.clip(out.astype(np.float32) * np.random.uniform(0.9, 1.1) + np.random.uniform(-10, 10), 0, 255).astype(np.uint8)
            vs.append(t); pr.append(True)
        else:
            vs.append(np.zeros((N, 3, fc.STORE, fc.STORE), np.uint8)); pr.append(False)
    x = torch.from_numpy(np.stack(vs))
    if np.random.rand() < 0.5:
        x = mirror_views(x[None], PLANES)[0]
    return x


def main(argv=None):
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument('--store', required=True); ap.add_argument('--sid', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--K', type=int, default=16); ap.add_argument('--draws', type=int, default=3)
    ap.add_argument('--n_time', type=int, default=12); ap.add_argument('--cols', type=int, default=8)
    A = ap.parse_args(argv)
    os.makedirs(A.out, exist_ok=True)
    with np.load(f'{A.store}/{A.sid}.npz') as z:
        fc.check_store_contract(z, A.sid)
        rows = []
        np.random.seed(0)
        x0, pr0, _, pos0, _ = heavy_study(z, A.sid, A.K, PLANES, dict(HEAVY, jitter=0.0, view_drop=0.0, mirror=0.0), aug=None)
        rows.append((x0, 'no aug'))
        for d in range(A.draws):
            np.random.seed(100 + d)
            x, pr, ct, pos, info = heavy_study(z, A.sid, A.K, PLANES)
            rows.append((x, f'draw {d} mirror {info["mirror"]} drop {info["drop"]}'))
        for j in range(V):
            if not pr0[j]:
                continue
            tiles = []
            for x, _ in rows:
                cols = [x[j, k, 1].numpy() for k in np.linspace(0, x.shape[1] - 1, A.cols).round().astype(int)]
                tiles.append(np.concatenate(cols, axis=1))
            img = np.concatenate(tiles, axis=0)
            cv2.imwrite(f'{A.out}/aug_{A.sid[-12:]}_v{j}_{PLANES[j]}.png', img)
        print(f'PNG: {A.out}/aug_{A.sid[-12:]}_v*.png | rows: ' + ' / '.join(t for _, t in rows))
        # mirror check, pixel aug off
        full = torch.from_numpy(np.stack([fc.sample_triplets(fc.decode_stack(z[f'v{j}_blob'], z[f'v{j}_offs']), N, 0.0)
                                          if f'v{j}_blob' in z.files else np.zeros((N, 3, fc.STORE, fc.STORE), np.uint8)
                                          for j in range(V)]))
        for seed in range(200):
            np.random.seed(seed)
            xm, prm, _, _, info = heavy_study(z, A.sid, 24, PLANES, dict(HEAVY, jitter=0.0, view_drop=0.0), aug=None)
            if info['mirror']:
                break
        ok = True
        for j, p in enumerate(PLANES):
            if prm[j]:
                for k in range(N):
                    ok &= bool(torch.equal(xm[j, k], full[j, N - 1 - k].flip(0) if p == 'Sagittal' else full[j, k].flip(-1)))
        print(f'MIRROR CHECK {"PASS" if ok else "FAIL"}: sagittal slot k == source slot {N - 1}-k with channels reversed, '
              f'axial/coronal W-flipped (views present {prm})')
    cv2.setNumThreads(0)                              # as fleet_train's DataLoader workers (_wif)
    files = sorted(f for f in os.listdir(A.store) if f.endswith('.npz'))[:A.n_time]
    tl, th = [], []
    for f in files:
        with np.load(f'{A.store}/{f}') as z:
            fc.check_store_contract(z, f)
            t0 = time.perf_counter(); light_item(z); tl.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); heavy_study(z, f[:-4], A.K, PLANES); th.append(time.perf_counter() - t0)
    print(f'TIMING per TRAIN item (1 process, cv2 threads 0 as in the loader workers) over {len(files)} studies: light (24 slots) '
          f'{np.mean(tl):.3f} s (median {np.median(tl):.3f}) | heavy (K={A.K}) {np.mean(th):.3f} s (median {np.median(th):.3f})')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
