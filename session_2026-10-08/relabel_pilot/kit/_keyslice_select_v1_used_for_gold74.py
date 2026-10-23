#!/usr/bin/env python3
"""keyslice_select.py (gold_prep/relabel, 2026-10-06). CPU only. Picks the 2-4 most informative MRI slices per TARGET
label for a study and exports them as windowed 384 px PNGs plus a manifest, for the multimodal relabel pilot
(RESEARCH_TO_TOP_10-06.md lever 2).

WHY ATTENTION. The TTR fleet model (fleet_model.KneeFleetNet + fleet_model_ttr.KneeFleetNetTTR) pools, for each of the
12 targets, the 24 slice triplets of each of the 4 views with a per-target slice softmax `a` (B,V,T,S) and then the 4
views with a per-target view softmax `va` (B,T,V). forward() returns logits only (it does NOT expose a / va). This file
recomputes both from the model's OWN modules (encode, proj, slice_pos, view_emb, ctr_emb, slice_q, slice_k, logt_s,
view_q, view_k, vbias, logt_v, head_w, head_b) in forward_with_attn() below, in exactly the order of
KneeFleetNet.forward(pos=None). The model sources are NOT edited: they are imported from a byte-identical copy in
./vendor (sha256 checked against ckpt args model_sha the same way fleet_worker.py does), and the recomputed logits are
checked against model.forward() (--parity_n studies per ckpt) and against ~/jump2/gold74_preds (fold-0 raw, if present).

MODES
  --mode attn (default): average a and va over the chosen checkpoints (default: the 5 raw K5 members
      f{0..4}_s1043_ttr_mix50). --ckpt_policy all (default; right for gold74, which is --exclude'd from every fleet run)
      or oof (train studies: only the checkpoint whose fold == the study's fold in folds5.csv, so the slice choice never
      comes from a model that trained on that study's report labels; ids listed in --exclude_csv still use all).
      Per target t: views ranked by va[t]; per_target 3 = top-2 slices of the best view (>= --min_gap triplets apart)
      + top-1 slice of the 2nd view (per_target 2 = 1+1, 4 = 2+2). Score = va[t,v] * a[v,t,s].
  --mode rule: fixed anatomy fallback, no model (RULE below; positions are GUESSES, fractions of the ordered stack).
  Union over targets, de-duplicated by (view, stored slice); cap --max_images by rank-robin (every target's rank-1
  pick first, then rank-2, ...), then ordered view by view, slice by slice for the prompt.

SOURCES
  --source store (default, pre-registered for the pilot): the fleet store npz (~/fleet_store/<sid>.npz, 4 views, <= 32
      slices per series, 140 mm physical centre crop, per-series 0.5/99.5 percentile window, 320 px, JPEG q92). It is
      the only image source on the box for all 4,407 train studies, so the pilot sees what the scale-up would see.
      Export = the triplet's centre slice, bicubic 320 -> --size.
  --source dicom: native DICOM of the same series uid (only where ~/kmimic/.../train_series/<sid>/<uid> exists, i.e. the
      58 gold58 studies, NOT the 16 extra); same slice-ordering and <= 32 subsampling as fleet_common.load_series, same
      0.5/99.5 per-series window, NO crop, long side -> --size. Falls back to store per view when missing/mismatched.

OUTPUTS (all under --out, which must be inside ~/gold_prep/ or the laptop scratchpad gold_prep/)
  <out>/png[_<tag>]/<sid>/<k:02d>_<VIEW>_s<slice:02d>of<n:02d>.png
  <out>/manifest_<tag>.csv  one row per exported image (sid, img_idx, png, view, plane, fat_sat, contrast, direction,
                            stack_idx, n_stack, frac, triplet_idx, targets, score, source)
  <out>/attn_<tag>.npz      (attn mode) ids, a (N,V,T,S), va (N,T,V), present (N,V), prob (N,12), n_ckpt (N,)
  <out>/keyslice_<tag>.log

Box usage (CPU only, never the GPU):
  cd ~/gold_prep/relabel && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 nice -n 19 \
    ~/miniconda3/envs/rsna/bin/python keyslice_select.py --ids gold74 --mode attn --tag gold74 --threads 8
"""
import os, sys, re, glob, json, math, time, hashlib, inspect, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
H = os.path.expanduser
ap = argparse.ArgumentParser()
ap.add_argument('--ids', default='gold74', help='gold74 | gold58 | all | path to a csv with StudyInstanceUID')
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--mode', default='attn', choices=['attn', 'rule'])
ap.add_argument('--source', default='store', choices=['store', 'dicom'])
ap.add_argument('--ckpts', nargs='*', default=[f'~/fleet_runs/f{f}_s1043_ttr_mix50/f{f}_s1043_raw_*_best.pt' for f in range(5)])
ap.add_argument('--ckpt_policy', default='all', choices=['all', 'oof'])
ap.add_argument('--folds', default='~/rsna_train/folds5.csv')
ap.add_argument('--exclude_csv', default='~/rsna_train/gold74_eval_set.csv', help='ids that no fleet run trained on')
ap.add_argument('--gold', default='~/rsna_train/gold74_eval_set.csv')
ap.add_argument('--gold58', default='~/kmimic/ids/gold58.csv')
ap.add_argument('--store', default='~/fleet_store')
ap.add_argument('--dicom_root', default='~/kmimic/kaggle/input/rsna-knee-abnormality-detection/train_series')
ap.add_argument('--targets', default='Effusion,Synovitis,Fracture,Contusion,PF OA,Lateral OA')
ap.add_argument('--per_target', type=int, default=3, choices=[2, 3, 4])
ap.add_argument('--min_gap', type=int, default=2, help='min triplet-index gap between 2 picks in one view')
ap.add_argument('--max_images', type=int, default=12)
ap.add_argument('--size', type=int, default=384)
ap.add_argument('--threads', type=int, default=8)
ap.add_argument('--parity_n', type=int, default=2, help='studies per ckpt checked against model.forward()')
ap.add_argument('--parity_ref', default='~/jump2/gold74_preds/f0_s1043_ttr_mix50__raw.csv')
ap.add_argument('--tag', default='gold74')
ap.add_argument('--out', default=HERE)
A = ap.parse_args()

OUT = os.path.realpath(H(A.out))
_ok_roots = [os.path.realpath(H('~/gold_prep'))]
_ok_roots += [os.path.realpath(p) for p in glob.glob(H('~/AppData/Local/Temp/claude/*/*/scratchpad/gold_prep'))]
assert any(OUT == r or OUT.startswith(r + os.sep) for r in _ok_roots), f'--out {OUT} must be inside {_ok_roots}'
os.makedirs(OUT, exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES'] = ''                     # CPU only, always
import numpy as np, pandas as pd
import cv2
cv2.setNumThreads(1)
sys.path.insert(0, os.path.join(HERE, 'vendor'))           # byte-identical copies, never the live training code dirs
import fleet_common as fc                                   # noqa: E402

LOGF = open(os.path.join(OUT, f'keyslice_{A.tag}.log'), 'a')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOGF.write(s + '\n'); LOGF.flush()


T12 = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
       "Baker's", 'Contusion', 'Fracture']
TARGETS = [t.strip() for t in A.targets.split(',') if t.strip()]
assert all(t in T12 for t in TARGETS), TARGETS
TI = [T12.index(t) for t in TARGETS]
VIEW_NAME = ['AX_FS', 'SAG_NONFS', 'COR_FS', 'SAG_FS']        # fc.VIEWS order: (Axial,1) (Sagittal,0) (Coronal,1) (Sagittal,1)
assert [tuple(v) for v in fc.VIEWS] == [('Axial', 1), ('Sagittal', 0), ('Coronal', 1), ('Sagittal', 1)], fc.VIEWS
PROMPT_ORDER = [3, 1, 2, 0]                                 # images are shown Sag FS, Sag non-FS, Cor FS, Ax FS
CONTRAST = {0: 'T1-weighted (approx., from TE/TR)', 1: 'PD-weighted (approx., from TE/TR)',
            2: 'T2-weighted (approx., from TE/TR)', 3: 'weighting unknown'}
# fleet_common.order_slices sorts by ImagePositionPatient . normal with the dominant normal component made positive,
# i.e. along +x (patient left) for sagittal, +y (posterior) for coronal, +z (superior) for axial (DICOM LPS).
DIRECTION = {'Sagittal': 'counted from patient-right toward patient-left', 'Coronal': 'counted from anterior toward posterior',
             'Axial': 'counted from inferior toward superior'}
# RULE fallback: target -> [(view j, fraction of the ordered stack)], GUESS positions. Substitutes when a view is absent.
RULE = {
    'Effusion':   [(3, .50), (3, .40), (0, .75)],
    'Synovitis':  [(0, .75), (0, .60), (3, .50)],
    'Fracture':   [(2, .50), (2, .35), (3, .50)],
    'Contusion':  [(2, .40), (2, .60), (3, .35)],
    'PF OA':      [(0, .65), (0, .55), (3, .50)],
    'Lateral OA': [(2, .50), (2, .60), (1, .50)],
    'Medial OA':  [(2, .50), (2, .40), (1, .50)],
    "Baker's":    [(0, .40), (0, .30), (3, .50)],
}
SUBST = {3: [1, 2, 0], 1: [3, 2, 0], 2: [3, 1, 0], 0: [3, 1, 2]}


def ids_from(spec):
    if spec in ('gold74', 'gold58'):
        g = pd.read_csv(H(A.gold), dtype={'StudyInstanceUID': str})
        if spec == 'gold58':
            g = g[g.source == 'original_gold58']
        return g.StudyInstanceUID.tolist()
    if spec == 'all':
        return sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{H(A.store)}/*.npz'))
    return pd.read_csv(H(spec), dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()


IDS = ids_from(A.ids)
if A.limit:
    IDS = IDS[:A.limit]
log(f'== keyslice_select {time.strftime("%Y-%m-%d %H:%M:%S")} mode {A.mode} source {A.source} ids {A.ids} n {len(IDS)} '
    f'targets {TARGETS} per_target {A.per_target} max_images {A.max_images} size {A.size} out {OUT}')


def load_store(sid):
    """-> dict j -> (stack uint8 (n,320,320), ctr, uid) for present views (the trainer/worker decode path)."""
    out = {}
    with np.load(f'{H(A.store)}/{sid}.npz') as z:
        fc.check_store_contract(z, sid)
        for j in range(len(fc.VIEWS)):
            if f'v{j}_blob' in z.files:
                out[j] = (fc.decode_stack(z[f'v{j}_blob'], z[f'v{j}_offs']), int(z[f'v{j}_ctr']), str(z[f'v{j}_uid']))
    return out


def model_input(views):
    """-> (x uint8 (1,V,S,3,H,W), present (1,V) bool, ctr (1,V) long): fleet_train.DS eval branch (off 0, pos None)."""
    import torch
    vs, pr, ct = [], [], []
    for j in range(len(fc.VIEWS)):
        if j in views:
            vs.append(fc.sample_triplets(views[j][0], fc.N_SLICES, 0.0)); pr.append(True); ct.append(views[j][1])
        else:
            vs.append(np.zeros((fc.N_SLICES, 3, fc.STORE, fc.STORE), np.uint8)); pr.append(False); ct.append(3)
    return torch.from_numpy(np.stack(vs))[None], torch.tensor(pr)[None], torch.tensor(ct)[None]


def triplet_centres(n):
    """sample_triplets centre index (into the stored <= 32 stack) of each of the N_SLICES triplets."""
    return np.clip(np.linspace(0, n - 1, fc.N_SLICES).round().astype(int), 0, n - 1)


# ------------------------------------------------------------------ attention (model's own modules, no source edit)
def model_sha_vendor():
    import fleet_model as _fm, fleet_model_ttr as _tt
    src = chr(10).join(inspect.getsource(_tt).splitlines()) + chr(10) + chr(10).join(inspect.getsource(_fm).splitlines())
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def forward_with_attn(m, views_u8, present, ctr):
    """Same arithmetic as KneeFleetNet.forward(views_u8, present, ctr, pos=None), also returning the per-target slice
    softmax a (B,V,T,S) and view softmax va (B,T,V). Uses only m's own sub-modules / parameters."""
    import torch
    B, V, S = views_u8.shape[:3]
    f = m.encode(views_u8.flatten(0, 2)).float()
    assert S == m.slice_pos.shape[1]
    h = m.proj(f).view(B, V, S, m.dim) + m.slice_pos[None] + m.view_emb[None, :, None]
    if m.ctr_emb is not None and ctr is not None:
        h = h + m.ctr_emb(ctr)[:, :, None]
    sc = 1.0 / math.sqrt(m.dim)
    a = (torch.einsum('td,bvsd->bvts', m.slice_q, m.slice_k(h)) * sc * m.logt_s.exp()).softmax(-1)
    pooled = torch.einsum('bvts,bvsd->bvtd', a, h)
    va = torch.einsum('td,bvtd->btv', m.view_q, m.view_k(pooled)) * sc * m.logt_v.exp() + m.vbias[None]
    pres = present.clone()
    pres[pres.sum(1) == 0] = True
    va = va.masked_fill(~pres[:, None, :], -1e4).softmax(-1)
    z = torch.einsum('btv,bvtd->btd', va, pooled)
    logits = (m.drop(z) * m.head_w[None]).sum(-1) + m.head_b
    return logits, a, va


def run_attention(ids, store_cache):
    import torch
    torch.set_num_threads(A.threads)
    from fleet_model_ttr import KneeFleetNetTTR
    paths = []
    for g in A.ckpts:
        hit = sorted(glob.glob(H(g)))
        assert len(hit) == 1, f'ckpt pattern {g} -> {hit}'
        paths.append(hit[0])
    sha_v = model_sha_vendor()
    folds = pd.read_csv(H(A.folds), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['fold'].to_dict()
    excl = set(pd.read_csv(H(A.exclude_csv), dtype={'StudyInstanceUID': str}).StudyInstanceUID) if A.exclude_csv else set()
    N, V, T, S = len(ids), len(fc.VIEWS), 12, fc.N_SLICES
    a_sum = np.zeros((N, V, T, S), np.float64); va_sum = np.zeros((N, T, V), np.float64)
    p_sum = np.zeros((N, 12), np.float64); n_ck = np.zeros(N, np.int64)
    present = np.zeros((N, V), bool)
    for i, sid in enumerate(ids):
        present[i] = [j in store_cache[sid] for j in range(V)]
    par_max = 0.0
    for p in paths:
        ck = torch.load(p, map_location='cpu', weights_only=False)
        a_ = ck['args']; fold = int(a_['fold'])
        assert a_.get('model_mod') == 'fleet_model_ttr', (p, a_.get('model_mod'))
        assert a_.get('contract') == fc.CONTRACT, (p, 'ckpt contract != vendored fleet_common CONTRACT')
        assert a_.get('model_sha') == sha_v, (p, 'model_sha', a_.get('model_sha'), 'vendor', sha_v)
        kw = dict(backbone=a_['backbone'], n_views=a_['n_views'], n_slices=a_['n_slices'], dim=a_['dim'],
                  n_contrast=a_['n_contrast'], contrast_emb=a_['contrast_emb'], pretrained=False, drop_path=0.0,
                  dropout=0.0, chunk=48)
        m = KneeFleetNetTTR(ttr_size=a_['ttr_size'], **kw)
        m.load_state_dict(ck['model'], strict=True)
        assert int(m.ttr_size_t) == int(a_['ttr_size'])
        m = m.eval().to(memory_format=torch.channels_last)
        del ck
        use = [i for i, sid in enumerate(ids)
               if A.ckpt_policy == 'all' or sid in excl or folds.get(sid) == fold]
        t0 = time.time(); probs = {}
        with torch.no_grad():
            for k, i in enumerate(use):
                x, pr, ct = model_input(store_cache[ids[i]])
                lg, a, va = forward_with_attn(m, x, pr, ct)
                if k < A.parity_n:
                    d = float((lg - m(x, pr, ct)).abs().max())
                    par_max = max(par_max, d)
                    assert d < 1e-4, (p, ids[i], 'forward_with_attn != model.forward', d)
                a_sum[i] += a[0].double().numpy(); va_sum[i] += va[0].double().numpy()
                pr_ = torch.sigmoid(lg.float()).numpy()[0]; p_sum[i] += pr_; n_ck[i] += 1
                probs[ids[i]] = pr_
        log(f'ckpt {os.path.basename(p)} fold {fold} cv{ck_cv(p)} studies {len(use)} {time.time() - t0:.0f}s '
            f'| parity vs forward max|dlogit| {par_max:.2e}')
        ref = H(A.parity_ref)
        if os.path.basename(p).startswith('f0_s1043_raw') and os.path.exists(ref) and probs:
            r = pd.read_csv(ref, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
            com = [s for s in probs if s in r.index]
            if com:
                d = np.abs(np.stack([probs[s] for s in com]) - r.loc[com, T12].to_numpy()).max()
                log(f'parity vs {os.path.basename(ref)} (gold_infer.py, 10-05): {len(com)} studies max|dprob| {d:.2e}')
                assert d < 1e-3, 'attention wrapper probabilities differ from the earlier CPU gold inference'
        del m
    assert (n_ck > 0).all(), 'a study got no checkpoint (oof policy + fold missing?)'
    a_m = a_sum / n_ck[:, None, None, None]; va_m = va_sum / n_ck[:, None, None]; p_m = p_sum / n_ck[:, None]
    np.savez(os.path.join(OUT, f'attn_{A.tag}.npz'), ids=np.array(ids), a=a_m.astype(np.float32),
             va=va_m.astype(np.float32), present=present, prob=p_m.astype(np.float32), n_ckpt=n_ck,
             targets=np.array(T12), views=np.array(VIEW_NAME), ckpts=np.array([os.path.basename(p) for p in paths]))
    return a_m, va_m, present


def ck_cv(p):
    m_ = re.search(r'cv(\d+)', os.path.basename(p))
    return m_.group(1) if m_ else '?'


# ------------------------------------------------------------------ selection
def pick_attn(a, va, present, t):
    """-> [(j, triplet s, score, rank)] for target t (index into T12)."""
    order = [j for j in np.argsort(-va[t]) if present[j]]
    quota = {2: [1, 1], 3: [2, 1], 4: [2, 2]}[A.per_target]
    if len(order) == 1:
        quota = [A.per_target]
    picks, rank = [], 0
    for j, q in zip(order, quota):
        got = []
        for s in np.argsort(-a[j, t]):
            if all(abs(int(s) - g) >= A.min_gap for g in got):
                got.append(int(s))
            if len(got) == q:
                break
        for s in got:
            picks.append((int(j), s, float(va[t, j] * a[j, t, s]), rank)); rank += 1
    return picks


def pick_rule(views, tname):
    picks = []
    for rank, (j, fr) in enumerate(RULE[tname]):
        jj = j if j in views else next((k for k in SUBST[j] if k in views), None)
        if jj is None:
            continue
        s = int(round(fr * (fc.N_SLICES - 1)))
        picks.append((jj, s, 1.0 - 0.1 * rank, rank))
    return picks[:A.per_target]


def select(views, a=None, va=None, present=None):
    """-> list of image dicts (deduped, capped, prompt-ordered)."""
    per_t = {}
    for tname, t in zip(TARGETS, TI):
        per_t[tname] = pick_attn(a, va, present, t) if A.mode == 'attn' else pick_rule(views, tname)
    items = {}                               # (j, stack_idx) -> dict
    maxr = max((len(v) for v in per_t.values()), default=0)
    for r in range(maxr):                    # rank-robin under the cap
        layer = sorted(((tn, pk) for tn, pks in per_t.items() for pk in pks if pk[3] == r), key=lambda x: -x[1][2])
        for tn, (j, s, sc, _) in layer:
            n = len(views[j][0]); c = int(triplet_centres(n)[s])
            key = (j, c)
            if key in items:
                it = items[key]
                if tn not in it['targets']:
                    it['targets'].append(tn)
                it['score'] = max(it['score'], sc)
            elif len(items) < A.max_images:
                items[key] = dict(view_j=j, stack_idx=c, n_stack=n, triplet_idx=s, targets=[tn], score=sc)
    out = sorted(items.values(), key=lambda d: (PROMPT_ORDER.index(d['view_j']), d['stack_idx']))
    return out


# ------------------------------------------------------------------ export
def to_size(img_u8, size):
    h, w = img_u8.shape
    sc = size / max(h, w)
    nh, nw = max(1, int(round(h * sc))), max(1, int(round(w * sc)))
    interp = cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC
    return cv2.resize(img_u8, (nw, nh), interpolation=interp)


_dcm_cache = {}


def dicom_slice(sid, uid, k, n_store):
    """stored-stack index k -> uint8 native-FOV image with the series 0.5/99.5 window, or None (caller falls back)."""
    import pydicom
    key = (sid, uid)
    if key not in _dcm_cache:
        sdir = f'{H(A.dicom_root)}/{sid}/{uid}'
        fs = sorted(glob.glob(f'{sdir}/*.dcm'))
        if not fs:
            _dcm_cache[key] = None
        else:
            try:
                if len(fs) <= fc.SRC_SLICES:
                    ds = fc.order_slices([d for d in (pydicom.dcmread(f) for f in fs) if 'PixelData' in d])
                else:
                    hdr = fc.order_slices([d for d in (pydicom.dcmread(f, stop_before_pixels=True) for f in fs) if 'Rows' in d])
                    idx = np.linspace(0, len(hdr) - 1, fc.SRC_SLICES).round().astype(int)
                    ds = [pydicom.dcmread(hdr[i].filename) for i in idx]
                arrs = [fc._as2d(d.pixel_array, d).astype(np.float32) for d in ds]
                lo, hi = np.percentile(np.concatenate([x.ravel() for x in arrs]), fc.WINDOW)
                _dcm_cache[key] = (arrs, float(lo), float(hi))
            except Exception as e:
                log(f'dicom read fail {sid[-8:]} {uid[-8:]}: {type(e).__name__} {str(e)[:80]}')
                _dcm_cache[key] = None
    got = _dcm_cache[key]
    if got is None or len(got[0]) != n_store:
        return None
    arrs, lo, hi = got
    return (np.clip((arrs[k] - lo) / (hi - lo + 1e-6), 0, 1) * 255.0).astype(np.uint8)


def export(sid, views, items, png_dir):
    rows = []
    d = os.path.join(png_dir, sid); os.makedirs(d, exist_ok=True)
    for k, it in enumerate(items):
        j = it['view_j']; plane, fs_ = fc.VIEWS[j]
        stack, ctr, uid = views[j]
        img, src = None, 'store'
        if A.source == 'dicom':
            img = dicom_slice(sid, uid, it['stack_idx'], len(stack))
            src = 'dicom' if img is not None else 'store_fallback'
        if img is None:
            img = stack[it['stack_idx']]
        img = to_size(img, A.size)
        fn = f'{k:02d}_{VIEW_NAME[j]}_s{it["stack_idx"] + 1:02d}of{it["n_stack"]:02d}.png'
        ok = cv2.imwrite(os.path.join(d, fn), img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        assert ok, fn
        n = it['n_stack']
        rows.append(dict(StudyInstanceUID=sid, img_idx=k, png=f'{os.path.basename(png_dir)}/{sid}/{fn}', view=VIEW_NAME[j],
                         plane=plane, fat_sat=int(fs_), contrast=CONTRAST.get(ctr, CONTRAST[3]), direction=DIRECTION[plane],
                         stack_idx=it['stack_idx'], n_stack=n, frac=round(it['stack_idx'] / max(n - 1, 1), 3),
                         triplet_idx=it['triplet_idx'], targets=';'.join(it['targets']), score=round(it['score'], 5),
                         source=src, height=img.shape[0], width=img.shape[1],
                         bytes=os.path.getsize(os.path.join(d, fn))))
    return rows


def main():
    t0 = time.time()
    cache = {}
    for sid in IDS:
        assert os.path.exists(f'{H(A.store)}/{sid}.npz'), f'{sid} not in store'
        cache[sid] = load_store(sid)
    log(f'decoded {len(cache)} studies from the store in {time.time() - t0:.0f}s; views present per study '
        f'{np.mean([len(v) for v in cache.values()]):.2f}')
    if A.mode == 'attn':
        a_m, va_m, present = run_attention(IDS, cache)
    png_dir = os.path.join(OUT, 'png' if A.source == 'store' and A.mode == 'attn' else f'png_{A.mode}_{A.source}')
    rows = []
    for i, sid in enumerate(IDS):
        if not cache[sid]:
            log(f'{sid}: no views in the store, skipped'); continue
        items = select(cache[sid], *((a_m[i], va_m[i], present[i]) if A.mode == 'attn' else ()))
        rows += export(sid, cache[sid], items, png_dir)
    man = pd.DataFrame(rows)
    mp = os.path.join(OUT, f'manifest_{A.tag}.csv')
    man.to_csv(mp, index=False)
    per = man.groupby('StudyInstanceUID').size()
    cov = {t: float(man.groupby('StudyInstanceUID').targets.apply(lambda s: any(t in x.split(';') for x in s)).mean())
           for t in TARGETS}
    log(f'manifest {mp}: {len(man)} images, {man.StudyInstanceUID.nunique()} studies, images/study min {per.min()} '
        f'mean {per.mean():.2f} max {per.max()}; per view {man.view.value_counts().to_dict()}; '
        f'png bytes total {man.bytes.sum() / 1e6:.1f} MB mean {man.bytes.mean() / 1e3:.1f} KB; '
        f'target coverage {cov}; source {man.source.value_counts().to_dict()}; {time.time() - t0:.0f}s total')


if __name__ == '__main__':
    main()
