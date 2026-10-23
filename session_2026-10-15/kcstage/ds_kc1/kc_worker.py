# kc_worker.py -- KC serve path (BOX_PLAN_1015 C1; Amendment 6). Kaggle GPU kernel stage AND box parity tool:
#   verbatim d96 reader (sha 30aa72fd, text in d96_reader_src.py) -> header side (v1 rule) + per-slot geometry (the d96
#   slot series, 3 headers each) -> kc_common.canon / kc_cut -> 6 views x 12 -> members (fp16 on CUDA, ONE pass, never
#   hflip / mirror) -> mean of member sigmoids. A weak-zone study (side rule != mean_x) or a not-GEOM_OK study is served as
#   the mean of the KC probabilities under BOTH canons (side L and side R, same geometry) (A6).
# Library: study_inputs(sid, rows, series_dir, reader) -> dict(vol, side, rule, weak, geom_raw, geom, gok, views[, views_alt]).
# CLI (probs):  python kc_worker.py --root <comp root> --split test --ckpts m1.pt m2.pt m3.pt --out kc_probs.csv
#               [--sids ids.csv] [--dump_dir d] [--device cuda|cuda:0|cuda:1|cpu] [--procs 4] [--window 8] [--shard K/N]
#               [--decode_only 1]
# KC-R2 (10-15 review fix):
#   --procs P   fork Pool(P) of decoders (study_inputs, forked BEFORE torch / CUDA init and before any thread exists) feeding
#               the one GPU consumer of this process through a bounded, ORDER-PRESERVING window (--window, default 2 x P)
#               of outstanding studies; workers return the views without the d96 volume (only its sha), so RAM stays
#               ~window x 10 MB. --procs 0 = serial (same _decode function, identical rows).
#   --shard K/N this process serves sids[K::N]: two processes (cuda:0 --shard 0/2, cuda:1 --shard 1/2, --procs 2 each on
#               4 vCPUs) = one GPU consumer per T4; concatenate the two outputs.
#   model_sha   every member ckpt's args['model_sha'] must equal sha256(fleet_model_ttr.py + fleet_model.py source)[:16] of
#               the modules THIS process imported (= fleet_train_kc._model_sha / fleet_worker._model_sha 'fleet_model_ttr');
#               both .py files must ship in the KC dataset (MANIFEST_1015.txt KC_DATASET section).
#   a study whose decode or predict raises gets NaN probabilities + kc_err (rows never dropped); the closing line reports
#   coverage, so the kernel cell applies the A6 G3 coverage >= 98 % rule and fills unscored studies with rank(base).
#   --decode_only 1 skips the members (decode timing / decode parity; probabilities NaN).
import os, sys, glob, json, time, hashlib, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kc_common as kc

READER_SHA = '30aa72fd972e29cf42c752aa5394a6c88a175f17b98740b1f6b0e0f55553ac40'
KC_SLOTS = {'sagfs': 0, 'sagpd': 1, 'corfs': 2, 'corpd': 3, 'ax': 4}


def load_reader():
    src = open(os.path.join(HERE, 'd96_reader_src.py')).read()
    assert hashlib.sha256(src.encode()).hexdigest() == READER_SHA, 'd96 reader text drift'
    from concurrent.futures import ThreadPoolExecutor
    g = {'np': np, 'glob': glob, 'ThreadPoolExecutor': ThreadPoolExecutor}
    exec(compile(src, 'raptor_reader', 'exec'), g)
    return g


def series_records(csv_path):
    """rows per study exactly as the corpus builder made them (pd.read_csv default dtypes, CSV order)."""
    ser = pd.read_csv(csv_path)
    ser['StudyInstanceUID'] = ser['StudyInstanceUID'].astype(str); ser['SeriesInstanceUID'] = ser['SeriesInstanceUID'].astype(str)
    return {k: v.to_dict('records') for k, v in ser.groupby('StudyInstanceUID')}


def _hdr(path, tags):
    import pydicom
    try:
        return pydicom.dcmread(path, stop_before_pixels=True, specific_tags=tags)
    except Exception:
        return None


def header_side(sid, rows, sdir):
    """v1 side rule (= kc_side_kernel.py): sagittal FS series with the most files, mean IPP[0], laterality tags."""
    sag = [r for r in rows if r['Anatomical_Plane'] == 'Sagittal']
    fs = [r for r in sag if str(r.get('Fat_Suppression')) in ('1', '1.0', 'True')]
    best = None
    for cand in (fs, sag):
        if cand:
            best = max(cand, key=lambda r: len(glob.glob(f"{sdir}/{sid}/{r['SeriesInstanceUID']}/*.dcm")))
            break
    if best is None:
        return float('nan'), ''
    xs, tags = [], set()
    for f in glob.glob(f"{sdir}/{sid}/{best['SeriesInstanceUID']}/*.dcm"):
        h = _hdr(f, ['ImagePositionPatient', 'Laterality', 'ImageLaterality'])
        if h is None:
            continue
        ipp = getattr(h, 'ImagePositionPatient', None)
        if ipp is not None and len(ipp) == 3:
            xs.append(float(ipp[0]))
        tags |= {str(getattr(h, t, '') or '').strip().upper()[:1] for t in ('Laterality', 'ImageLaterality')} - {''}
    return (float(np.mean(xs)) if xs else float('nan')), ''.join(sorted(tags))


def slot_geometry(sid, rows, sdir, pick):
    """per d96 slot: the series _pick_series_for_slot chooses (verbatim reader function `pick`), geometry from 3 headers."""
    raw, used = {}, set()
    for (plane, fluid, _k), b in zip([("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18), ("Coronal", 0, 12),
                                      ("Axial", -1, 18)], KC_SLOTS):
        raw[b] = float('nan')
        r = pick(rows, plane, fluid, used)
        if r is None:
            continue
        used.add(r['SeriesInstanceUID'])
        fs = sorted(glob.glob(f"{sdir}/{sid}/{r['SeriesInstanceUID']}/*.dcm"))
        vals = []
        for i in sorted({0, len(fs) // 2, len(fs) - 1}) if fs else []:
            h = _hdr(fs[i], ['ImageOrientationPatient'])
            try:
                o = np.array(h.ImageOrientationPatient, float)
                vals.append(float(np.cross(o[:3], o[3:])[0]) if plane == 'Sagittal' else float(o[0]))
            except Exception:
                pass
        if vals and len({np.sign(v) for v in vals}) == 1:
            raw[b] = float(np.mean(vals))
    return raw


def study_inputs(sid, rows, sdir, R, both=None):
    rd = R['_make_reader']()
    vol, _mask = R['build_study'](sid, {sid: rows}, sdir, rd)
    mean_x, tag = header_side(sid, rows, sdir)
    side, rule = kc.side_rule(mean_x, tag)
    graw = slot_geometry(sid, rows, sdir, R['_pick_series_for_slot'])
    geom = {b: (graw[b] if np.isfinite(graw[b]) else kc.GEOM_DEFAULT[b]) for b in kc.GEOM_DEFAULT}
    pres = kc.present_blocks(vol)
    gok = kc.geom_ok(graw, [b for b in ('sagfs', 'sagpd', 'corfs', 'ax') if pres[b]])
    weak = rule != 'mean_x'
    out = dict(sid=sid, vol=vol, mean_x=mean_x, tag=tag, side=side, rule=rule, weak=int(weak), gok=int(gok), geom_raw=graw,
               geom=geom, serve='both' if (weak or not gok) else 'single')
    out['views'] = kc.kc_cut(vol, side, geom)
    if both if both is not None else out['serve'] == 'both':
        out['views_L'] = kc.kc_cut(vol, 'L', geom)
        out['views_R'] = kc.kc_cut(vol, 'R', geom)
    return out


def model_sha():
    """KC-R2: sha256(fleet_model_ttr source + fleet_model source)[:16] of the modules this process imports (= the trainer's
    _model_sha for --model ttr), + their files."""
    import inspect, fleet_model as _fm, fleet_model_ttr as _tt
    src = chr(10).join(inspect.getsource(_tt).splitlines()) + chr(10) + chr(10).join(inspect.getsource(_fm).splitlines())
    return hashlib.sha256(src.encode()).hexdigest()[:16], (_tt.__file__, _fm.__file__)


def load_members(paths, device):
    import torch
    sys.path.insert(0, HERE)
    from fleet_model_ttr import KneeFleetNetTTR
    msha, mfiles = model_sha()
    ms = []
    for p in paths:
        ck = torch.load(p, map_location='cpu', weights_only=False)
        a = ck['args']
        assert a.get('model_mod') == 'kc_ttr' and a.get('kc_contract_hash') == kc.CONTRACT_HASH and a.get('mirror') == 0, \
            (p, a.get('model_mod'), a.get('kc_contract_hash'), a.get('mirror'))
        assert a.get('model_sha') == msha, (p, 'model_sha of the ckpt != the shipped fleet_model_ttr.py + fleet_model.py',
                                            a.get('model_sha'), msha, mfiles)                                    # KC-R2
        m = KneeFleetNetTTR(ttr_size=a['ttr_size'], backbone=a['backbone'], dim=a['dim'], contrast_emb=a['contrast_emb'],
                            n_views=a['n_views'], n_slices=a['n_slices'], n_contrast=a['n_contrast'], pretrained=False,
                            drop_path=0.0, dropout=0.0, grad_ckpt=False)
        m.load_state_dict(ck['model'])
        ms.append(m.to(device).eval().to(memory_format=torch.channels_last))
    return ms


def predict_views(ms, views, device):
    import torch
    x, pr, ct = kc.views_to_tensor(views)
    x = torch.from_numpy(x)[None].to(device); pr = torch.from_numpy(pr)[None].to(device); ct = torch.from_numpy(ct)[None].to(device)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=str(device).startswith('cuda')):   # KC-R2 cuda:N
        ps = [torch.sigmoid(m(x, pr, ct).float()).cpu().numpy()[0] for m in ms]
    return np.mean(ps, 0)


def serve_probs(ms, si, device):
    if si['serve'] == 'both':
        return 0.5 * (predict_views(ms, si['views_L'], device) + predict_views(ms, si['views_R'], device))
    return predict_views(ms, si['views'], device)


_W = {}                                                                     # KC-R2 decode-worker state (set before fork)


def _decode(sid):
    """study_inputs without the volume (only its sha): the unit of work of the serial loop AND the fork-pool workers."""
    try:
        si = study_inputs(sid, _W['SER'].get(sid, []), _W['sdir'], _W['R'])
        si['vol_sha'] = hashlib.sha256(si.pop('vol').tobytes()).hexdigest()[:16]
        si['err'] = ''
    except Exception as e:                                                  # never drops a row (A6 G3 coverage rule)
        si = dict(sid=sid, err=f'decode {type(e).__name__}: {str(e)[:160]}')
    return si


def _winit():
    import cv2
    cv2.setNumThreads(1)


def decoded(sids, window, pool):
    """yields _decode(sid) in sids order; with a pool, at most `window` studies outstanding (bounded RAM)."""
    if pool is None:
        for s in sids:
            yield _decode(s)
        return
    from collections import deque
    q, nxt = deque(), 0
    while nxt < len(sids) and len(q) < window:
        q.append(pool.apply_async(_decode, (sids[nxt],))); nxt += 1
    while q:
        r = q.popleft().get()
        if nxt < len(sids):
            q.append(pool.apply_async(_decode, (sids[nxt],))); nxt += 1
        yield r


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--split', default='test', choices=('test', 'train'))
    ap.add_argument('--ckpts', nargs='*', default=[]); ap.add_argument('--out', required=True)
    ap.add_argument('--sids', default=''); ap.add_argument('--dump_dir', default=''); ap.add_argument('--device', default='cuda')
    ap.add_argument('--procs', type=int, default=0); ap.add_argument('--window', type=int, default=0)       # KC-R2
    ap.add_argument('--shard', default=''); ap.add_argument('--decode_only', type=int, default=0)            # KC-R2
    A = ap.parse_args()
    assert A.ckpts or A.decode_only, '--ckpts needed unless --decode_only 1'
    import cv2
    cv2.setNumThreads(1)
    R = load_reader()
    SER = series_records(f'{A.root}/{A.split}_series.csv')
    sdir = f'{A.root}/{A.split}_series'
    sids = sorted(SER) if not A.sids else list(pd.read_csv(A.sids, dtype={'StudyInstanceUID': str}).StudyInstanceUID)
    if A.shard:
        k, n = (int(v) for v in A.shard.split('/'))
        assert 0 <= k < n, A.shard
        sids = sids[k::n]
    _W.update(R=R, SER=SER, sdir=sdir)
    pool = None
    if A.procs > 0:                                     # fork BEFORE torch / CUDA and before any thread exists
        import multiprocessing as mp
        pool = mp.get_context('fork').Pool(A.procs, initializer=_winit)
    window = A.window or 2 * max(A.procs, 1)
    ms = [] if A.decode_only else load_members(A.ckpts, A.device)
    msha = model_sha() if ms else ('-', ())
    rows, t0, nerr = [], time.time(), 0
    for si in decoded(sids, window, pool):
        sid = si['sid']
        p = np.full(len(kc.TARGETS), np.nan)
        if not si['err'] and ms:
            try:
                p = serve_probs(ms, si, A.device)
            except Exception as e:
                si['err'] = f'predict {type(e).__name__}: {str(e)[:160]}'
        nerr += bool(si['err'])
        rows.append(dict(StudyInstanceUID=sid, **{t: float(v) for t, v in zip(kc.TARGETS, p)}, kc_side=si.get('side', ''),
                         kc_rule=si.get('rule', ''), kc_serve=si.get('serve', ''), kc_vol_sha=si.get('vol_sha', ''),
                         kc_err=si['err']))
        if A.dump_dir and not si['err']:
            os.makedirs(A.dump_dir, exist_ok=True)
            np.savez(f"{A.dump_dir}/{sid}.npz", **kc.study_npz_dict(si['views'], si['side'], si['rule'], si['geom_raw'],
                     kc.block_flips(si['side'], si['geom']), si['weak'], si['gok']))
    if pool is not None:
        pool.close(); pool.join()
    pd.DataFrame(rows).to_csv(A.out, index=False)
    n = max(len(rows), 1)
    print(f'KC worker: {len(rows)} studies, {len(ms)} members, {(time.time() - t0) / n:.3f} s/study | procs {A.procs} '
          f'window {window} shard {A.shard or "-"} device {A.device} | coverage {(len(rows) - nerr) / n:.4f} (err {nerr}) | '
          f'serve both {sum(r["kc_serve"] == "both" for r in rows)} | model_sha {msha[0]} {list(msha[1])}', flush=True)
