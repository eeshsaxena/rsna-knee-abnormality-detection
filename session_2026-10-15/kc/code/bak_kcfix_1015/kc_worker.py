# kc_worker.py -- KC serve path (BOX_PLAN_1015 C1; Amendment 6). Kaggle GPU kernel stage AND box parity tool:
#   verbatim d96 reader (sha 30aa72fd, text in d96_reader_src.py) -> header side (v1 rule) + per-slot geometry (the d96
#   slot series, 3 headers each) -> kc_common.canon / kc_cut -> 6 views x 12 -> members (fp16 on CUDA, ONE pass, never
#   hflip / mirror) -> mean of member sigmoids. A weak-zone study (side rule != mean_x) or a not-GEOM_OK study is served as
#   the mean of the KC probabilities under BOTH canons (side L and side R, same geometry) (A6).
# Library: study_inputs(sid, rows, series_dir, reader) -> dict(vol, side, rule, weak, geom_raw, geom, gok, views[, views_alt]).
# CLI (probs):  python kc_worker.py --root <comp root> --split test --ckpts m1.pt m2.pt m3.pt --out kc_probs.csv
#               [--sids ids.csv] [--dump_dir d] [--device cuda] [--procs 4]
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


def load_members(paths, device):
    import torch
    sys.path.insert(0, HERE)
    from fleet_model_ttr import KneeFleetNetTTR
    ms = []
    for p in paths:
        ck = torch.load(p, map_location='cpu', weights_only=False)
        a = ck['args']
        assert a.get('model_mod') == 'kc_ttr' and a.get('kc_contract_hash') == kc.CONTRACT_HASH and a.get('mirror') == 0, \
            (p, a.get('model_mod'), a.get('kc_contract_hash'), a.get('mirror'))
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
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=(device == 'cuda')):
        ps = [torch.sigmoid(m(x, pr, ct).float()).cpu().numpy()[0] for m in ms]
    return np.mean(ps, 0)


def serve_probs(ms, si, device):
    if si['serve'] == 'both':
        return 0.5 * (predict_views(ms, si['views_L'], device) + predict_views(ms, si['views_R'], device))
    return predict_views(ms, si['views'], device)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--split', default='test', choices=('test', 'train'))
    ap.add_argument('--ckpts', nargs='+', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--sids', default=''); ap.add_argument('--dump_dir', default=''); ap.add_argument('--device', default='cuda')
    A = ap.parse_args()
    import cv2
    cv2.setNumThreads(1)
    R = load_reader()
    SER = series_records(f'{A.root}/{A.split}_series.csv')
    sdir = f'{A.root}/{A.split}_series'
    sids = sorted(SER) if not A.sids else list(pd.read_csv(A.sids, dtype={'StudyInstanceUID': str}).StudyInstanceUID)
    ms = load_members(A.ckpts, A.device)
    rows, t0 = [], time.time()
    for sid in sids:
        si = study_inputs(sid, SER.get(sid, []), sdir, R)
        p = serve_probs(ms, si, A.device)
        rows.append(dict(StudyInstanceUID=sid, **{t: float(v) for t, v in zip(kc.TARGETS, p)}, kc_side=si['side'],
                         kc_rule=si['rule'], kc_serve=si['serve'], kc_vol_sha=hashlib.sha256(si['vol'].tobytes()).hexdigest()[:16]))
        if A.dump_dir:
            os.makedirs(A.dump_dir, exist_ok=True)
            np.savez(f"{A.dump_dir}/{sid}.npz", **kc.study_npz_dict(si['views'], si['side'], si['rule'], si['geom_raw'],
                     kc.block_flips(si['side'], si['geom']), si['weak'], si['gok']))
    pd.DataFrame(rows).to_csv(A.out, index=False)
    print(f'KC worker: {len(rows)} studies, {len(ms)} members, {(time.time() - t0) / max(len(rows), 1):.2f} s/study | '
          f'serve both {sum(r["kc_serve"] == "both" for r in rows)}', flush=True)
