# fleet_aug.py -- the TTR "heavy" TRAIN-time data path (BIG_JUMP_PLAN sec 3.1, build item B2), imported by
# fleet_train.py only under --aug heavy (the default path never imports it). Pure functions of an open store npz and
# the global numpy RNG (seeded per DataLoader worker by torch), so the tests and the PNG dump call exactly the code the
# trainer runs.
# Order inside one study (all coins drawn BEFORE any pixel work, so only the K kept triplets are augmented):
#   1 per-study mirror coin (p .5)          2 slot subset pos = sorted K of N_SLICES (K = --train_slices, else all)
#   3 view dropout (p .2, only with >= 2 present views, one view)
#   4 per present, kept view: decode (SAME decode as the worker) -> sample_triplets with slice jitter U(-1, 1) ->
#     take the slots of the MIRRORED view that pos selects (sagittal mirror reverses the slot axis, so those are
#     N-1-pos) -> aug_view_heavy on K triplets
#   5 mirror_views on the (V, K) tensor. With aug off and jitter 0 this equals the default path's
#     mirror_views(full 24-slot tensor)[:, pos] exactly (tests/t_b2_heavy_equiv.py).
# aug_view_heavy, ONE draw per view: affine (rot +-15 deg, scale .85-1.15, shift +-8 % of the side; same matrix on every
# slice and channel; bilinear, zero border) -> intensity LUT (gamma .75-1.35 on [0,1], gain .85-1.15, bias +-15) ->
# Gaussian noise (p .3, sigma U(0, 6), per pixel) -> round + clip to uint8 -> CoarseDropout (p .3, 1-3 holes, each
# side U(8 %, 16 %) of the image, the same holes on every slice and channel of the view, filled with 0).
import numpy as np

HEAVY = dict(rot=15.0, scale=(0.85, 1.15), shift=0.08, gamma=(0.75, 1.35), gain=(0.85, 1.15), bias=15.0,
             noise_p=0.3, noise_sigma=6.0, hole_p=0.3, holes=(1, 3), hole_side=(0.08, 0.16),
             jitter=1.0, view_drop=0.2, mirror=0.5)


def aug_view_heavy(trip, cfg=HEAVY):
    """trip uint8 (K, 3, H, W) -> uint8 (K, 3, H, W); one random draw for the whole view."""
    import cv2
    H = trip.shape[-1]
    ang = np.random.uniform(-cfg['rot'], cfg['rot']); sc = np.random.uniform(*cfg['scale'])
    tx, ty = np.random.uniform(-cfg['shift'], cfg['shift'], 2) * H
    Mx = cv2.getRotationMatrix2D((H / 2, H / 2), ang, sc); Mx[:, 2] += (tx, ty)
    out = np.empty_like(trip)
    for s in range(trip.shape[0]):
        for c in range(trip.shape[1]):
            out[s, c] = cv2.warpAffine(trip[s, c], Mx, (H, H), flags=cv2.INTER_LINEAR, borderValue=0)
    gam = np.random.uniform(*cfg['gamma']); g = np.random.uniform(*cfg['gain'])
    b = np.random.uniform(-cfg['bias'], cfg['bias'])
    lut = (255.0 * (np.arange(256, dtype=np.float64) / 255.0) ** gam * g + b).astype(np.float32)
    f = lut[out]                                                   # gamma, gain, bias in one table
    if np.random.rand() < cfg['noise_p']:
        sig = np.random.uniform(0.0, cfg['noise_sigma'])
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))   # seeded from the global (per-worker) stream
        f += rng.standard_normal(f.shape, dtype=np.float32) * np.float32(sig)
    out = np.clip(f + 0.5, 0, 255).astype(np.uint8)                # round half up
    if np.random.rand() < cfg['hole_p']:
        for _ in range(np.random.randint(cfg['holes'][0], cfg['holes'][1] + 1)):
            hh = max(1, int(round(np.random.uniform(*cfg['hole_side']) * H)))
            ww = max(1, int(round(np.random.uniform(*cfg['hole_side']) * H)))
            y0 = np.random.randint(0, H - hh + 1); x0 = np.random.randint(0, H - ww + 1)
            out[:, :, y0:y0 + hh, x0:x0 + ww] = 0
    return out


def heavy_study(z, sid, K, planes, cfg=HEAVY, aug=aug_view_heavy):
    """z: an open store npz (contract already checked by the caller). -> (x uint8 torch (V, K, 3, S, S), present list,
    contrast list, pos int64 numpy (K,), info dict(mirror, drop)). aug=None skips aug_view_heavy (tests only)."""
    import torch
    import fleet_common as fc
    from fleet_model import mirror_views
    N, V = fc.N_SLICES, len(fc.VIEWS)
    K = K if 0 < K < N else N
    mir = bool(np.random.rand() < cfg['mirror'])                                       # 1
    pos = np.sort(np.random.choice(N, K, replace=False)) if K < N else np.arange(N)    # 2
    have_v = [f'v{j}_blob' in z.files for j in range(V)]
    drop = -1
    if sum(have_v) > 1 and np.random.rand() < cfg['view_drop']:                       # 3
        drop = int(np.random.choice([k for k in range(V) if have_v[k]]))
    vs, pr, ct = [], [], []
    for j in range(V):                                                                 # 4
        if have_v[j] and j != drop:
            st = fc.decode_stack(z[f'v{j}_blob'], z[f'v{j}_offs'])
            t = fc.sample_triplets(st, N, np.random.uniform(-cfg['jitter'], cfg['jitter']))
            src = (N - 1 - pos)[::-1] if (mir and planes[j] == 'Sagittal') else pos
            t = np.ascontiguousarray(t[src])
            vs.append(aug(t, cfg) if aug is not None else t); pr.append(True); ct.append(int(z[f'v{j}_ctr']))
        else:
            vs.append(np.zeros((K, 3, fc.STORE, fc.STORE), np.uint8)); pr.append(False); ct.append(3)
    x = torch.from_numpy(np.stack(vs))
    if mir:                                                                            # 5
        x = mirror_views(x[None], planes)[0]
    return x, pr, ct, pos.astype(np.int64), dict(mirror=mir, drop=drop)
