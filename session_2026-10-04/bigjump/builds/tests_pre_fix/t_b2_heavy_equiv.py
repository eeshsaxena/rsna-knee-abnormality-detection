# t_b2_heavy_equiv: the --aug heavy data path (fleet_aug.heavy_study) on the round3 64 px store (6 real-format npz).
#  (1) subset-before-aug is the SAME sampling as the default path: with the pixel aug off and slice jitter 0, for 40
#      seeds x 6 studies x K in {16, 24}, heavy_study == default-path mirror_views(full 24-slot tensor)[:, pos] (torch.equal),
#      with the mirror coin, pos and contrast/presence as drawn; both mirror outcomes occur
#  (2) view dropout (forced p 1): exactly one present view is zeroed, present False, contrast 3; other views unchanged
#  (3) mirror semantics on a real study: sagittal slot k of the mirrored tensor == source slot N-1-k with the triplet
#      channels reversed; axial/coronal == W flip of the same slot
#  (4) aug_view_heavy: deterministic under a seed, uint8, shape kept; CoarseDropout holes identical on every slice
#      and channel of a view; identity settings (no affine/intensity/noise/holes) return the input exactly
import os, sys, json
from _paths import CODE, DRY, OUT, SHIM, stage64
C64 = stage64(CODE, f'{OUT}/code64_aug', ['fleet_common.py', 'fleet_model.py', 'fleet_aug.py'])
sys.path.insert(0, SHIM); sys.path.insert(0, C64)
import numpy as np, torch
import fleet_common as fc
from fleet_model import mirror_views
from fleet_aug import HEAVY, heavy_study, aug_view_heavy
assert fc.STORE == 64
STORE = f'{DRY}/store_64_normal'
PLANES = [p for p, _ in fc.VIEWS]
N, V = fc.N_SLICES, len(fc.VIEWS)
sids = sorted(f[:-4] for f in os.listdir(STORE) if f.endswith('.npz'))


def default_full(z):
    vs, pr, ct = [], [], []
    for j in range(V):
        if f'v{j}_blob' in z.files:
            vs.append(fc.sample_triplets(fc.decode_stack(z[f'v{j}_blob'], z[f'v{j}_offs']), N, 0.0))
            pr.append(True); ct.append(int(z[f'v{j}_ctr']))
        else:
            vs.append(np.zeros((N, 3, fc.STORE, fc.STORE), np.uint8)); pr.append(False); ct.append(3)
    return torch.from_numpy(np.stack(vs)), pr, ct


res = dict(cases=0, mirrored=0, plain=0)
cfg0 = dict(HEAVY, jitter=0.0, view_drop=0.0)
for K in (16, 24):
    for seed in range(40):
        for sid in sids:
            with np.load(f'{STORE}/{sid}.npz') as z:
                full, pr0, ct0 = default_full(z)
                np.random.seed(seed)
                x, pr, ct, pos, info = heavy_study(z, sid, K, PLANES, cfg0, aug=None)
            exp = mirror_views(full[None], PLANES)[0] if info['mirror'] else full
            exp = exp[:, torch.from_numpy(pos)]
            assert x.shape == (V, K, 3, fc.STORE, fc.STORE) and len(pos) == K and (np.diff(pos) > 0).all()
            assert torch.equal(x, exp), (K, seed, sid, info)
            assert pr == pr0 and ct == ct0
            res['cases'] += 1; res['mirrored' if info['mirror'] else 'plain'] += 1
print(f'(1) heavy_study(aug off, jitter 0) == mirror_views(full)[:, pos]: {res["cases"]} cases '
      f'({res["mirrored"]} mirrored, {res["plain"]} plain), K 16 and 24')
assert res['mirrored'] > 50 and res['plain'] > 50

nd = 0
for seed in range(20):
    for sid in sids:
        with np.load(f'{STORE}/{sid}.npz') as z:
            full, pr0, ct0 = default_full(z)
            np.random.seed(seed)
            x, pr, ct, pos, info = heavy_study(z, sid, 16, PLANES, dict(cfg0, view_drop=1.0), aug=None)
        if sum(pr0) < 2:
            assert info['drop'] == -1; continue
        j = info['drop']; nd += 1
        assert pr0[j] and not pr[j] and ct[j] == 3 and int(x[j].abs().sum()) == 0
        exp = (mirror_views(full[None], PLANES)[0] if info['mirror'] else full)[:, torch.from_numpy(pos)]
        keep = [k for k in range(V) if k != j]
        assert torch.equal(x[keep], exp[keep]) and [pr[k] for k in keep] == [pr0[k] for k in keep]
print(f'(2) forced view dropout: {nd} cases, one present view zeroed (present False, contrast 3), others unchanged')

with np.load(f'{STORE}/{sids[0]}.npz') as z:
    full, pr0, _ = default_full(z)
    for seed in range(100):
        np.random.seed(seed)
        x, pr, ct, pos, info = heavy_study(z, sids[0], 24, PLANES, cfg0, aug=None)
        if info['mirror']:
            break
assert info['mirror']
for j, p in enumerate(PLANES):
    if not pr0[j]:
        continue
    for k in range(N):
        if p == 'Sagittal':
            assert torch.equal(x[j, k], full[j, N - 1 - k].flip(0)), (j, k)
        else:
            assert torch.equal(x[j, k], full[j, k].flip(-1)), (j, k)
print(f'(3) mirror on {sids[0][-6:]}: sagittal slot k == source slot {N - 1}-k, channels reversed; axial/coronal W-flipped '
      f'(views present {pr0})')

rng = np.random.default_rng(0)
trip = rng.integers(0, 256, (16, 3, 64, 64), dtype=np.uint8)
np.random.seed(5); o1 = aug_view_heavy(trip)
np.random.seed(5); o2 = aug_view_heavy(trip)
assert o1.dtype == np.uint8 and o1.shape == trip.shape and np.array_equal(o1, o2) and not np.array_equal(o1, trip)
ident = dict(HEAVY, rot=0.0, scale=(1.0, 1.0), shift=0.0, gamma=(1.0, 1.0), gain=(1.0, 1.0), bias=0.0, noise_p=0.0,
             hole_p=0.0)
np.random.seed(1); oi = aug_view_heavy(trip, ident)
assert np.array_equal(oi, trip), 'identity aug must return the input'
white = np.full((16, 3, 64, 64), 255, np.uint8)
nh = 0
for seed in range(30):
    np.random.seed(seed); oh = aug_view_heavy(white, dict(ident, hole_p=1.0))
    m0 = oh[0, 0] == 0
    assert m0.any() and all(np.array_equal(oh[s, c] == 0, m0) for s in range(16) for c in range(3))
    assert set(np.unique(oh)) <= {0, 255}
    nh += 1
np.random.seed(3); on = aug_view_heavy(np.full((16, 3, 64, 64), 128, np.uint8), dict(ident, noise_p=1.0, noise_sigma=6.0))
print(f'(4) aug_view_heavy: deterministic under a seed, uint8, shape kept; identity cfg returns the input; holes '
      f'identical on all 48 slice-channels ({nh} draws); noise sd at sigma<=6: {on.astype(float).std():.2f}')
res.update(dropout_cases=nd, hole_draws=nh)
json.dump(res, open(f'{OUT}/t_b2_heavy_equiv.json', 'w'), indent=1)
print('T_B2_HEAVY_EQUIV OK')
