# kc_common.py -- ONE source of truth for the KC (knee-centred compartment) leg's input arithmetic.
# BOX_PLAN_1015 sec 2.2 C1 + BOX_PLAN_CRITIQUE_1015 (adopted). A clone of fleet_common's role for the KC store:
# imported by (a) kc_build_store.py (box CPU, cuts from the d96 corpus npy), (b) fleet_train_kc.py (V100 trainer),
# (c) kc_worker.py (Kaggle serve: verbatim d96 reader -> header side -> canon -> cut -> members). Ship this exact
# file with the ckpts; never re-type it.
#
# Input = one d96 volume (96, 384, 384) uint8 from the VERBATIM Raptor reader (sha 30aa72fd):
#   SLOTS [(Sagittal, fluid 1, 26), (Sagittal, fluid 0, 22), (Coronal, 1, 18), (Coronal, 0, 12), (Axial, any, 18)],
#   slices sorted by dot(IPP, cross(row, col)) of each series, 140 mm centre crop -> 384 px (PX_MM = 384 / 140).
# Views (6 x 12 source slices, 256 px, lossless PNG):
#   sag_lat  SagFS block, canonical slots 0-55 %, 110 mm box     sag_med  SagFS block, canonical slots 45-100 %, 110 mm
#   sag_pd   SagPD block, central 70 % (15-85 %), 120 mm box      cor_med  CorFS block 10-90 %, 90 mm box at x = -22 mm
#   cor_lat  CorFS block 10-90 %, 90 mm box at x = +22 mm         ax       Axial block 5-95 %, full 140 mm
# Canonical frame = a LEFT knee: sagittal slot 0 = lateral end; coronal / axial image-left = medial.
# Flips (critique: per-slot geometry, never constants): for the series that fills each slot,
#   sagittal block reversed  iff (side == 'R') XOR (normal_x > 0)        [d96 sorts by dot(IPP, n): n_x < 0 -> slot 0 = patient left]
#   coronal / axial flip W   iff (side == 'R') XOR (row_x < 0)           [row_x > 0 -> image-left = patient right]
# A slot geometry that is missing (NaN) takes the MEASURED default (sag n_x < 0, cor / ax row_x > 0 on 458/458 box studies).
# Side rule (training and store): mean_x sign if |mean_x| > 30 mm, else the DICOM tag, else the sign if |mean_x| > 5 mm,
# else U (no flip, left frame). WEAK zone = any rule other than 'mean_x'; GEOM_OK = every present slot's geometry has
# the expected sign with |component| >= 0.7. Serve (A6): a weak-zone or not-GEOM_OK study = mean of the KC
# probabilities under both canons (side L and side R, same geometry); training keeps the rule above.
# present[view] = the d96 block is non-empty (mask = vol slice sum > 0, exactly the corpus rsna_index.csv mask and the
# reader's own mask), on train and serve alike. Contrast: the d96 corpus has no TE / TR, so the KC trainer uses
# contrast_emb 0 and every stored v{j}_ctr is the constant 3 (unknown), identical on train and serve.
import json
import hashlib
import inspect
import numpy as np

# ---- d96 layout (reader sha 30aa72fd) ----
D96_SHA = '30aa72fd972e29cf42c752aa5394a6c88a175f17b98740b1f6b0e0f55553ac40'
D96_IMG = 384
D96_CROP_MM = 140.0
PX_MM = D96_IMG / D96_CROP_MM
D96_B = dict(sagfs=(0, 26), sagpd=(26, 48), corfs=(48, 66), corpd=(66, 78), ax=(78, 96))
BLOCK_PLANE = dict(sagfs='Sagittal', sagpd='Sagittal', corfs='Coronal', corpd='Coronal', ax='Axial')

# ---- KC views: (plane, name, block, (lo_frac, hi_frac), (cy_mm, cx_mm), (h_mm, w_mm)) ----
KC_SPEC = [
    ('Sagittal', 'sag_lat', 'sagfs', (0.0, 0.55), (0.0, 0.0), (110.0, 110.0)),
    ('Sagittal', 'sag_med', 'sagfs', (0.45, 1.0), (0.0, 0.0), (110.0, 110.0)),
    ('Sagittal', 'sag_pd', 'sagpd', (0.15, 0.85), (0.0, 0.0), (120.0, 120.0)),
    ('Coronal', 'cor_med', 'corfs', (0.1, 0.9), (0.0, -22.0), (90.0, 90.0)),
    ('Coronal', 'cor_lat', 'corfs', (0.1, 0.9), (0.0, 22.0), (90.0, 90.0)),
    ('Axial', 'ax', 'ax', (0.05, 0.95), (0.0, 0.0), (140.0, 140.0)),
]
VIEWS = [(p, n) for p, n, *_ in KC_SPEC]          # trainer API: PLANES = [p for p, _ in fc.VIEWS]
VIEW_NAMES = [n for _, n in VIEWS]
MED_VIEWS = ('sag_med', 'cor_med')
LAT_VIEWS = ('sag_lat', 'cor_lat')
SRC_SLICES = 12      # stored slices per view
N_SLICES = 12        # triplet centres per view (= SRC_SLICES: centre i -> stored slice i, triplet (i-1, i, i+1))
STORE = 256          # stored resolution == model input (ttr_size 256 passes through untouched, TTR-5)
PNG_LEVEL = 1
N_CONTRAST = 4
CTR_CONST = 3        # no TE / TR in the d96 corpus: constant contrast class on train and serve
PREP_REV = 'kc1'
SIDE_T_MM = 30.0
SIDE_LAT_MM = 5.0
GEOM_MIN = 0.7
# default geometry (MEASURED on 458/458 box studies): sagittal normal x < 0, coronal / axial row x > 0
GEOM_DEFAULT = dict(sagfs=-1.0, sagpd=-1.0, corfs=1.0, corpd=1.0, ax=1.0)

# ---- anatomical view prior (A6): vbias init, +VBIAS_PRIOR on the listed views, 0 elsewhere ----
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
           'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
VBIAS_PRIOR = 1.0
VBIAS_MAP = {'MCL': MED_VIEWS, 'Medial Meniscus': MED_VIEWS, 'Medial OA': MED_VIEWS,
             'Lateral Meniscus': LAT_VIEWS, 'Lateral OA': LAT_VIEWS,
             'ACL': ('sag_pd',), 'PF OA': ('ax',), "Baker's": ('ax',), 'Effusion': ('ax',)}


def vbias_prior_matrix():
    m = np.zeros((len(TARGETS), len(VIEWS)), np.float32)
    for t, vs in VBIAS_MAP.items():
        for v in vs:
            m[TARGETS.index(t), VIEW_NAMES.index(v)] = VBIAS_PRIOR
    return m


# ---------------- side rule ----------------
def side_rule(mean_x, tag):
    """-> (side 'R' | 'L' | 'U', rule 'mean_x' | 'tag' | 'mean_x_weak' | 'none'). tag: string of DICOM laterality
    letters seen in the side series ('' / 'L' / 'R' / 'LR')."""
    mx = float(mean_x) if mean_x is not None else float('nan')
    t = set(str(tag or '').replace('nan', '').strip().upper()) - {' '}
    if np.isfinite(mx) and abs(mx) > SIDE_T_MM:
        return ('R' if mx < 0 else 'L'), 'mean_x'
    if t in ({'R'}, {'L'}):
        return next(iter(t)), 'tag'
    if np.isfinite(mx) and abs(mx) > SIDE_LAT_MM:
        return ('R' if mx < 0 else 'L'), 'mean_x_weak'
    return 'U', 'none'


def geom_of(row):
    """side-table row (dict-like) -> {block: signed component} with NaN -> MEASURED default. v2 side table columns
    g_sagfs, g_sagpd (sagittal normal x), g_corfs, g_corpd, g_ax (row x)."""
    out = {}
    for b, d in GEOM_DEFAULT.items():
        v = row.get(f'g_{b}', np.nan) if hasattr(row, 'get') else np.nan
        try:
            v = float(v)
        except Exception:
            v = float('nan')
        out[b] = v if np.isfinite(v) else d
    return out


def geom_ok(geom_raw, present_blocks):
    """True iff every present KC block's series has the expected sign with |component| >= GEOM_MIN.
    geom_raw: {block: value or NaN} BEFORE defaulting (NaN on a present block = unknown = not OK)."""
    for b in present_blocks:
        v = geom_raw.get(b, float('nan'))
        if not np.isfinite(v) or abs(v) < GEOM_MIN or np.sign(v) != np.sign(GEOM_DEFAULT[b]):
            return False
    return True


# ---------------- pixel path (everything here can move a STORED pixel) ----------------
def block_flips(side, geom):
    """-> {block: bool flip}. Sagittal: reverse slot order; coronal / axial: flip W. U = left frame."""
    r = side == 'R'
    out = {}
    for b in D96_B:
        g = geom[b]
        out[b] = (r != (g > 0)) if BLOCK_PLANE[b] == 'Sagittal' else (r != (g < 0))
    return out


def canon(vol, flips):
    v = vol.copy()
    for b, (a, e) in D96_B.items():
        if not flips[b]:
            continue
        if BLOCK_PLANE[b] == 'Sagittal':
            v[a:e] = v[a:e][::-1]
        else:
            v[a:e] = v[a:e][:, :, ::-1]
    return v


def sl(a, b, k):
    return np.linspace(a, b - 1, k).round().astype(int)


def box_cut(img, cy_mm, cx_mm, h_mm, w_mm):
    import cv2
    H, W = img.shape
    cy, cx = H / 2 + cy_mm * PX_MM, W / 2 + cx_mm * PX_MM
    h, w = int(round(h_mm * PX_MM)), int(round(w_mm * PX_MM))
    y0, x0 = int(round(cy - h / 2)), int(round(cx - w / 2))
    out = np.zeros((h, w), np.uint8)
    ys, xs = max(0, y0), max(0, x0)
    ye, xe = min(H, y0 + h), min(W, x0 + w)
    out[ys - y0:ye - y0, xs - x0:xe - x0] = img[ys:ye, xs:xe]
    return cv2.resize(out, (STORE, STORE), interpolation=cv2.INTER_AREA if h > STORE else cv2.INTER_LINEAR)


def d96_mask(vol):
    return (vol.reshape(vol.shape[0], -1).sum(1) > 0)


def present_blocks(vol_or_mask):
    m = d96_mask(vol_or_mask) if np.ndim(vol_or_mask) == 3 else np.asarray(vol_or_mask).astype(bool)
    return {b: bool(m[a:e].any()) for b, (a, e) in D96_B.items()}


def kc_cut(vol, side, geom):
    """d96 vol (96, 384, 384) uint8 -> {view: uint8 (12, STORE, STORE) or None (absent block)}."""
    pres = present_blocks(vol)
    v = canon(vol, block_flips(side, geom))
    out = {}
    for _, name, blk, (lo, hi), (cy, cx), (hm, wm) in KC_SPEC:
        if not pres[blk]:
            out[name] = None
            continue
        a, e = D96_B[blk]; n = e - a
        idx = sl(int(n * lo), int(n * hi), SRC_SLICES)
        out[name] = np.stack([box_cut(v[a + i], cy, cx, hm, wm) for i in idx])
    return out


# ---------------- tensor path (stored bytes -> model input) ----------------
def encode_png(img):
    import cv2
    ok, buf = cv2.imencode('.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION), PNG_LEVEL])
    if not ok:
        raise ValueError('png encode failed')
    return buf.ravel()


def pack_bufs(bufs):
    offs = np.cumsum([0] + [len(b) for b in bufs]).astype(np.int64)
    return np.concatenate(bufs).astype(np.uint8), offs


def decode_stack(blob, offs):
    import cv2
    return np.stack([cv2.imdecode(blob[offs[i]:offs[i + 1]], cv2.IMREAD_GRAYSCALE)
                     for i in range(len(offs) - 1)])


def sample_triplets(stack, n_slices=N_SLICES, off=0.0):
    """stack uint8 (n, H, W) -> uint8 (n_slices, 3, H, W); 2.5D (c-1, c, c+1) (= fleet_common.sample_triplets)."""
    n = len(stack)
    centres = np.clip((np.linspace(0, n - 1, n_slices) + off).round().astype(int), 0, n - 1)
    lo = np.clip(centres - 1, 0, n - 1); hi = np.clip(centres + 1, 0, n - 1)
    return np.stack([stack[lo], stack[centres], stack[hi]], axis=1)


def views_to_tensor(views):
    """{view: uint8 (12, S, S) or None} -> (x uint8 (V, N_SLICES, 3, S, S), present bool (V,), ctr int64 (V,)).
    The serve path; the trainer's eval path builds the same arrays from the stored PNG bytes."""
    xs, pr = [], []
    for name in VIEW_NAMES:
        st = views[name]
        if st is None:
            xs.append(np.zeros((N_SLICES, 3, STORE, STORE), np.uint8)); pr.append(False)
        else:
            xs.append(sample_triplets(st, N_SLICES, 0.0)); pr.append(True)
    return np.stack(xs), np.array(pr), np.full(len(VIEWS), CTR_CONST, np.int64)


# ---------------- contract ----------------
def _fn_sha(*fns):
    h = hashlib.sha256()
    for f in fns:
        h.update(inspect.getsource(f).replace('\r\n', '\n').encode())
    return h.hexdigest()[:16]


STORE_CONTRACT = dict(kc_spec=[[p, n, b, list(f), list(c), list(s)] for p, n, b, f, c, s in KC_SPEC],
                      src_slices=SRC_SLICES, store=STORE, png_level=PNG_LEVEL, prep_rev=PREP_REV, d96_sha=D96_SHA[:16],
                      side_t_mm=SIDE_T_MM, side_lat_mm=SIDE_LAT_MM, ctr_const=CTR_CONST,
                      pixel_sha=_fn_sha(block_flips, canon, sl, box_cut, d96_mask, present_blocks, kc_cut, encode_png))
CONTRACT = dict(STORE_CONTRACT, n_slices=N_SLICES, n_contrast=N_CONTRAST,
                tensor_sha=_fn_sha(decode_stack, sample_triplets, views_to_tensor))
STORE_CONTRACT_JSON = json.dumps(STORE_CONTRACT, sort_keys=True)
CONTRACT_HASH = hashlib.sha256(json.dumps(CONTRACT, sort_keys=True).encode()).hexdigest()[:16]
SELECT_SHA = _fn_sha(side_rule, geom_of, geom_ok)
JPEG_Q = None        # (fleet API compatibility; KC stores lossless PNG)
CROP_MM = D96_CROP_MM


def study_npz_dict(views, side, rule, geom_raw, flips, weak, gok):
    kw = {'contract': np.array(STORE_CONTRACT_JSON), 'side': np.array(side), 'rule': np.array(rule),
          'weak': np.int64(weak), 'geom_ok': np.int64(gok),
          'geom': np.array(json.dumps({b: (None if not np.isfinite(v) else float(v)) for b, v in geom_raw.items()})),
          'flips': np.array(json.dumps({b: bool(f) for b, f in flips.items()}))}
    for j, name in enumerate(VIEW_NAMES):
        st = views[name]
        if st is not None:
            kw[f'v{j}_blob'], kw[f'v{j}_offs'] = pack_bufs([encode_png(s) for s in st])
            kw[f'v{j}_ctr'], kw[f'v{j}_uid'] = np.int64(CTR_CONST), np.array(name)
    return kw


def check_store_contract(z, where=''):
    got = json.loads(str(z['contract'])) if 'contract' in z.files else None
    assert got == STORE_CONTRACT, (f'KC store contract {where}: {got} != this kc_common {STORE_CONTRACT}')


def jpeg_fingerprint():
    """sha of a fixed PNG encode + decode in THIS cv2 build (name kept for the fleet trainer API)."""
    import cv2
    img = (np.add.outer(np.arange(STORE), 3 * np.arange(STORE)) % 251).astype(np.uint8)
    buf = encode_png(img)
    dec = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return hashlib.sha256(buf.tobytes() + dec.tobytes()).hexdigest()[:16]


def decode_sha(z):
    h = hashlib.sha256()
    for j in range(len(VIEWS)):
        if f'v{j}_blob' in z.files:
            h.update(decode_stack(z[f'v{j}_blob'], z[f'v{j}_offs']).tobytes())
    return h.hexdigest()[:16]
