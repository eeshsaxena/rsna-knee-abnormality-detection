# fleet_common.py -- ONE source of truth for the fleet's input arithmetic.   (round3/fleet CONSOLIDATED)
# Imported by (a) the Kaggle CPU prep kernel that builds the training store,
# (b) the V100 trainer, (c) fleet_worker.py at inference. Ship this exact file inside
# the checkpoint dataset; never re-type it (aastik's S13/S14/B7/S19 bugs were all
# train/serve drift from re-typed constants or re-ordered steps).
#
# Faithful to aastikrajan15's public inference contract (intel/aastik_ours_script.py):
#   4 views [(Axial,FS),(Sagittal,nonFS),(Coronal,FS),(Sagittal,FS)], first decodable
#   series per slot, slices ordered by ImagePositionPatient . normal, <=32 kept by
#   linspace (header-first when longer), one 0.5/99.5 percentile window per series,
#   physical mm centre crop per axis (crop_xy + crop_clamp), uint8 + JPEG q92 round trip,
#   24 centres by linspace, 2.5D triplet (c-1,c,c+1) as RGB, ImageNet normalisation.
# Deliberate differences (allowed because we train from scratch, train==test):
#   - crop on the native pixel grid then resize once to STORE (aastik: resize FOV->512, crop, ->320);
#   - crop_mm 140 (aastik's ckpt value is private; 130 default / 150 "shipped" in his comments);
#   - NO --norm_p80 landmark gain (aastik L650/L771, applied L802-803). Absent on BOTH sides, so no
#     parity risk; if it is ever added it must be ONE function here, applied after decode_stack on
#     the uncropped stored stack, on train and serve alike (his B7 lesson).
#   - NO RescaleSlope/Intercept and NO MONOCHROME1 inversion (aastik applies neither; a constant
#     positive slope cancels in the per-series percentile window).
#
# round3 consolidated fixes (FLEET_REVIEW.md, IDs C#):
#   C1  series selection: candidates sorted by SeriesInstanceUID; Fluid_Sensitive NaN/missing/str ->
#       Fat_Suppression (== Fluid_Sensitive on 24,371/24,371 train rows); NaN plane -> IOP normal;
#       a still-unknown flag leaves the view ABSENT (never guessed). No-op on the complete train CSV.
#   C2  window bounds as Python floats: float32 math on numpy 1.x AND 2.x (NEP 50 made it float64)
#   C3  sorted glob; pixel-less objects dropped; multi-frame arrays REJECTED (was silent a[..., 0])
#   C4  PixelSpacing VM=1 -> isotropic (was: IndexError -> no crop at all); kr/kc never exceed h/w
#   C5  imencode failure raises
#   C6  STORE_CONTRACT / CONTRACT: constants + sha of the pixel-path source, stamped into every
#       npz, asserted by the trainer and (via ckpt args) by the worker (aastik S14 drift class)
#   C7  parity_check() + jpeg_fingerprint(): inference-side proof that THIS environment re-encodes
#       the stored train bytes (cv2/numpy/pydicom may differ between the CPU prep image and the
#       GPU rerun image)
import glob
import json
import hashlib
import inspect
import numpy as np

VIEWS = [('Axial', 1), ('Sagittal', 0), ('Coronal', 1), ('Sagittal', 1)]
SRC_SLICES = 32      # max stored slices per series
CROP_MM = 140.0      # physical centre crop, per axis
STORE = 64          # stored resolution == model input resolution
JPEG_Q = 92
WINDOW = [0.5, 99.5]  # per-series intensity percentiles
N_SLICES = 24        # slice centres per view
N_CONTRAST = 4       # 0 T1, 1 PD, 2 T2, 3 unknown/absent
PREP_REV = 'r3'      # bump on ANY intentional change that can move a stored pixel
PLANES = ('Sagittal', 'Coronal', 'Axial')        # dominant normal axis x, y, z -> plane


def contrast_class(te, tr):
    # own thresholds (aastik's dataset.contrast_class is private)
    try:
        te = float(te); tr = float(tr)
    except Exception:
        return 3
    if not np.isfinite(te) or not np.isfinite(tr):
        return 3
    if te >= 45:
        return 2
    if tr < 800:
        return 0
    return 1


def order_slices(ds):
    try:
        o = np.array(ds[0].ImageOrientationPatient, float)
        nrm = np.cross(o[:3], o[3:])
        if nrm[np.argmax(np.abs(nrm))] < 0:
            nrm = -nrm
        key = [float(np.dot(np.array(d.ImagePositionPatient, float), nrm)) for d in ds]
    except Exception:
        key = [float(getattr(d, 'InstanceNumber', i) or i) for i, d in enumerate(ds)]
    return [ds[i] for i in np.argsort(key, kind='stable')]


def _spacing(d):
    """(row_mm, col_mm); 0.0 = unknown. C4: a VM=1 PixelSpacing is isotropic (was IndexError -> no crop)."""
    ps = getattr(d, 'PixelSpacing', None)
    if ps is None:
        return 0.0, 0.0
    try:
        vals = [float(v) for v in (ps if hasattr(ps, '__len__') and not isinstance(ps, str) else [ps])]
    except Exception:
        return 0.0, 0.0
    vals = [v if (np.isfinite(v) and v > 0) else 0.0 for v in vals] + [0.0, 0.0]
    r, c = vals[0], vals[1]
    return (r or c), (c or r)


def _crop_resize(a, px_r, px_c):
    # a: float32 [0,1] (H, W). Square crop_mm box in physical space, clamped per axis.
    import cv2
    h, w = a.shape
    kr = int(round(CROP_MM / px_r)) if px_r > 0 else h
    kc = int(round(CROP_MM / px_c)) if px_c > 0 else w
    kr = min(h, max(32, kr)); kc = min(w, max(32, kc))          # C4: never larger than the image
    r0 = (h - kr) // 2; c0 = (w - kc) // 2
    a = a[r0:r0 + kr, c0:c0 + kc]
    interp = cv2.INTER_AREA if (kr >= STORE and kc >= STORE) else cv2.INTER_LINEAR
    return cv2.resize(a, (STORE, STORE), interpolation=interp)


def _as2d(a, d):
    """C3: one 2-D slice per file. A colour (H,W,3|4) secondary capture keeps channel 0 (draft
    behaviour); a multi-frame (F,H,W) object raises so the NEXT candidate series is tried (the
    draft's a[..., 0] silently turned it into an (F,H) garbage image)."""
    if a.ndim == 2:
        return a
    if a.ndim == 3 and int(getattr(d, 'SamplesPerPixel', 1) or 1) > 1 and a.shape[-1] in (3, 4):
        return a[..., 0]
    raise ValueError(f'multi-frame pixel array {a.shape} not supported')


def load_series(sdir, return_bytes=False):
    """-> (stack uint8 (n<=32, STORE, STORE) after the q92 JPEG round trip, contrast class
    [, list of the q92 JPEG buffers]) or None. The prep kernel stores the buffers themselves, so
    decoding them later is bit-identical to what inference sees IN THE SAME cv2 BUILD (C7 checks)."""
    import pydicom, cv2
    fs = sorted(glob.glob(f'{sdir}/*.dcm'))                          # C3 deterministic tie order
    if not fs:
        return None
    if len(fs) <= SRC_SLICES:
        ds = order_slices([d for d in (pydicom.dcmread(f) for f in fs) if 'PixelData' in d])   # C3
        if not ds:
            return None
        arrs = [_as2d(d.pixel_array, d).astype(np.float32) for d in ds]
    else:
        hdr = order_slices([d for d in (pydicom.dcmread(f, stop_before_pixels=True) for f in fs)
                            if 'Rows' in d])                         # C3
        if not hdr:
            return None
        idx = np.linspace(0, len(hdr) - 1, SRC_SLICES).round().astype(int)
        ds = [hdr[i] for i in idx]
        full = [pydicom.dcmread(d.filename) for d in ds]
        arrs = [_as2d(f.pixel_array, f).astype(np.float32) for f in full]
    ctr = contrast_class(getattr(ds[0], 'EchoTime', None), getattr(ds[0], 'RepetitionTime', None))
    lo, hi = np.percentile(np.concatenate([a.ravel() for a in arrs]), WINDOW)
    lo, hi = float(lo), float(hi)                                    # C2: float32 on numpy 1 and 2
    px_r, px_c = _spacing(ds[0])                                     # C4
    out, bufs = [], []
    for a in arrs:
        a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
        a = _crop_resize(a, px_r, px_c)
        u8 = (a * 255.0).astype(np.uint8)
        ok, buf = cv2.imencode('.jpg', u8, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])
        if not ok:
            raise ValueError('imencode failed')                      # C5
        out.append(cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)); bufs.append(buf.ravel())
    return (np.stack(out), ctr, bufs) if return_bytes else (np.stack(out), ctr)


# ---------------- series selection (C1) ----------------
def _flag_col(df, col):
    """0/1 float Series; NaN when the column is missing or the cell is not a recognisable 0/1."""
    import pandas as pd
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    s = df[col]
    if s.dtype == bool:
        return s.astype(float)
    s = s.astype(str).str.strip().str.lower().replace({'true': '1', 'false': '0', 'yes': '1', 'no': '0',
                                                         't': '1', 'f': '0'})
    v = pd.to_numeric(s, errors='coerce').astype(float)
    return v.where(v.isin([0.0, 1.0]))


def _plane_from_header(sdir):
    import pydicom
    fs = sorted(glob.glob(f'{sdir}/*.dcm'))
    if not fs:
        return None
    try:
        d = pydicom.dcmread(fs[len(fs) // 2], stop_before_pixels=True)
        o = np.array(d.ImageOrientationPatient, float)
        return PLANES[int(np.argmax(np.abs(np.cross(o[:3], o[3:]))))]
    except Exception:
        return None


def select_series(study_series_df, img_root, sid, stats=None):
    """-> {view: [SeriesInstanceUID, ...] in try order}. On the complete train CSV this is exactly the
    draft rule (Anatomical_Plane == plane & Fluid_Sensitive == flag) in SeriesInstanceUID order,
    which equals the train CSV row order. A series whose flag is unknown after the Fat_Suppression
    fallback is NOT used: an absent view is a state the model trains on, a guessed contrast is not."""
    df = study_series_df.copy()
    df['SeriesInstanceUID'] = df['SeriesInstanceUID'].astype(str)
    df = df.sort_values('SeriesInstanceUID', kind='stable')
    flag = _flag_col(df, 'Fluid_Sensitive')
    nan_fl = flag.isna()
    flag = flag.where(~nan_fl, _flag_col(df, 'Fat_Suppression'))
    if 'Anatomical_Plane' in df.columns:
        plane = df['Anatomical_Plane'].astype(str).str.strip().str.capitalize()
    else:
        plane = df['SeriesInstanceUID'].map(lambda _: '')
    bad_pl = ~plane.isin(PLANES)
    for i in df.index[bad_pl]:
        p = _plane_from_header(f'{img_root}/{sid}/{df.at[i, "SeriesInstanceUID"]}')
        plane.at[i] = p if p is not None else ''
    if stats is not None:
        stats['flag_from_fatsup'] = stats.get('flag_from_fatsup', 0) + int((nan_fl & flag.notna()).sum())
        stats['flag_unknown'] = stats.get('flag_unknown', 0) + int(flag.isna().sum())
        stats['plane_from_iop'] = stats.get('plane_from_iop', 0) + int((bad_pl & plane.isin(PLANES)).sum())
    return {(pl, fl): df.loc[(plane == pl) & (flag == fl), 'SeriesInstanceUID'].tolist() for pl, fl in VIEWS}


def build_study(study_series_df, img_root, sid, return_bytes=False, stats=None, errs=None):
    """study_series_df: rows of train_series/test_series for ONE study.
    -> {view: (stack, ctr, bufs_or_None, series_uid) or None}. First decodable candidate wins
    (aastik rule). `errs` (list) collects 'view|uid|error' strings for the prep receipt."""
    if len(study_series_df):
        cands = select_series(study_series_df, img_root, sid, stats)
    else:
        cands = {v: [] for v in VIEWS}
    out = {}
    for v in VIEWS:
        got = None
        for uid in cands[v]:
            try:
                r = load_series(f'{img_root}/{sid}/{uid}', return_bytes)
            except Exception as e:
                r = None
                if errs is not None:
                    errs.append(f'{v[0][:3]}{v[1]}|{uid[-8:]}|{type(e).__name__}: {str(e)[:80]}')
            if r is not None:
                got = (r[0], r[1], r[2] if return_bytes else None, uid)
                break
        out[v] = got
    return out


def sample_triplets(stack, n_slices=N_SLICES, off=0.0):
    """stack uint8 (n, H, W) -> uint8 (n_slices, 3, H, W); 2.5D (c-1, c, c+1)."""
    n = len(stack)
    centres = np.clip((np.linspace(0, n - 1, n_slices) + off).round().astype(int), 0, n - 1)
    lo = np.clip(centres - 1, 0, n - 1); hi = np.clip(centres + 1, 0, n - 1)
    return np.stack([stack[lo], stack[centres], stack[hi]], axis=1)


def pack_bufs(bufs):
    """list of q92 JPEG buffers -> (bytes uint8 1-D, offsets int64)."""
    offs = np.cumsum([0] + [len(b) for b in bufs]).astype(np.int64)
    return np.concatenate(bufs).astype(np.uint8), offs


def decode_stack(blob, offs):
    import cv2
    return np.stack([cv2.imdecode(blob[offs[i]:offs[i + 1]], cv2.IMREAD_GRAYSCALE)
                     for i in range(len(offs) - 1)])


# ---------------- contract (C6) and environment parity (C7) ----------------
def _fn_sha(*fns):
    h = hashlib.sha256()
    for f in fns:
        h.update(inspect.getsource(f).replace('\r\n', '\n').encode())
    return h.hexdigest()[:16]


# Everything that can move a STORED pixel (asserted by the trainer against every npz).
STORE_CONTRACT = dict(views=[list(v) for v in VIEWS], src_slices=SRC_SLICES, crop_mm=CROP_MM, store=STORE,
                      jpeg_q=JPEG_Q, window=WINDOW, prep_rev=PREP_REV,
                      pixel_sha=_fn_sha(contrast_class, order_slices, _spacing, _crop_resize, _as2d, load_series))
# + everything between the stored bytes and the model input (asserted by the worker on every ckpt).
CONTRACT = dict(STORE_CONTRACT, n_slices=N_SLICES, n_contrast=N_CONTRAST,
                tensor_sha=_fn_sha(decode_stack, sample_triplets))
STORE_CONTRACT_JSON = json.dumps(STORE_CONTRACT, sort_keys=True)
# Which series fills a slot. Logged (MATCH/MISMATCH) by the worker but not asserted: it is a
# no-op on the complete train CSV and may legitimately be hardened again for the hidden test.
SELECT_SHA = _fn_sha(_flag_col, _plane_from_header, select_series, build_study)


def study_npz_dict(st):
    """build_study(..., return_bytes=True) -> dict for np.savez (loads with allow_pickle=False):
    contract str, v{j}_blob uint8[], v{j}_offs int64[n+1], v{j}_ctr int64 0-d, v{j}_uid str."""
    kw = {'contract': np.array(STORE_CONTRACT_JSON)}
    for j, v in enumerate(VIEWS):
        if st[v] is not None:
            _, ctr, bufs, uid = st[v]
            kw[f'v{j}_blob'], kw[f'v{j}_offs'] = pack_bufs(bufs)
            kw[f'v{j}_ctr'], kw[f'v{j}_uid'] = np.int64(ctr), np.array(str(uid))
    return kw


def check_store_contract(z, where=''):
    """Trainer: refuse a store npz built under another contract (C6)."""
    got = json.loads(str(z['contract'])) if 'contract' in z.files else None
    assert got == STORE_CONTRACT, (f'store contract {where}: {got} != this fleet_common {STORE_CONTRACT} '
                                   f'-- the store was built by a different fleet_common.py')


def jpeg_fingerprint():
    """sha of a fixed q-JPEG encode + decode in THIS cv2 build (log in prep, trainer, worker)."""
    import cv2
    img = (np.add.outer(np.arange(STORE), 3 * np.arange(STORE)) % 251).astype(np.uint8)
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])
    dec = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return hashlib.sha256(buf.tobytes() + dec.tobytes()).hexdigest()[:16]


def decode_sha(z):
    """sha of every stored view DECODED in this cv2 build. The trainer records it for 4 store studies
    (ckpt args decode_probe); the worker recomputes it on the same npz shipped in parity/, which
    proves the T4 decode == the V100 decode of the very bytes the model trained on."""
    h = hashlib.sha256()
    for j in range(len(VIEWS)):
        if f'v{j}_blob' in z.files:
            h.update(decode_stack(z[f'v{j}_blob'], z[f'v{j}_offs']).tobytes())
    return h.hexdigest()[:16]


def parity_check(z, study_series_df, img_root, sid):
    """C7. Rebuild one TRAIN study from DICOM in THIS environment and compare it with the bytes the
    prep kernel stored. -> dict(n_pix, frac_diff, frac_gt8, maxdiff, mismatch)."""
    st = build_study(study_series_df, img_root, sid)
    n = d = g8 = mx = 0; mism = []
    for j, v in enumerate(VIEWS):
        stored = f'v{j}_blob' in z.files
        if stored != (st[v] is not None):
            mism.append(f'v{j} presence stored={stored} rebuilt={st[v] is not None}'); continue
        if not stored:
            continue
        a = decode_stack(z[f'v{j}_blob'], z[f'v{j}_offs']).astype(np.int16)
        b = st[v][0].astype(np.int16)
        if a.shape != b.shape:
            mism.append(f'v{j} shape {a.shape} vs {b.shape}'); continue
        e = np.abs(a - b)
        n += e.size; d += int((e > 0).sum()); g8 += int((e > 8).sum()); mx = max(mx, int(e.max()))
    return dict(n_pix=n, frac_diff=d / max(n, 1), frac_gt8=g8 / max(n, 1), maxdiff=mx, mismatch=mism)
