# build_synth.py (10-20 r5, Claude). Box only. Rebuilds DICOM series from the 320 px fleet store so the R6 / S6fall serve path
# (compact_infer: native_cache.prepare -> geometry bank -> SKM-TEA anchors -> px_cache views) can run on them. Needed because the
# OAI V00 raw DICOMs were pruned on 10-09 (raw_deletions.csv); ~/oai_store/<kid>.npz (fleet_common r3) is the only OAI image
# form left. The same rebuild is applied to RSNA gold58 (from ~/fleet_store) to MEASURE what the approximation costs R6.
#
# Store recap (fleet_common.load_series): <= 32 slices per series (linspace subsample when longer, physical order along the
# positive cardinal axis), central 140 mm square crop (clamped to the image), resized to 320 px, 0.5/99.5 % window -> uint8,
# JPEG q92. Views kept: v0 Axial FS, v1 Sagittal non-FS, v2 Coronal FS, v3 Sagittal FS. No geometry is stored.
# Rebuild per stored view (geometry is FABRICATED from per-series metadata):
#   pixels   decoded uint8 x 16 -> uint16 MONOCHROME2 (linear; compact_infer re-quantizes by min / max anyway)
#   spacing  in-plane kr * ps / 320 with kr = min(rows, max(32, round(140 / ps))) (exact for the store crop)
#   slices   stored slice k sits at native index idx_k = linspace(0, n_native - 1, 32).round() (or k when n_native <= 32),
#            position (idx_k - (n_native - 1) / 2) * step along the positive cardinal axis of the plane
#   IOP      OAI: cardinal (Sag 0,1,0/0,0,-1; Cor 1,0,0/0,0,-1; Ax 1,0,0/0,1,0 = the majority IOP of the remaining raw
#            follow-up tarballs, proto_stats.py) with a per-series LEFT-RIGHT MIRROR repair for Cor / Ax: the R6 bundle's own
#            side classifier (side_cls.pt, side_common.side_inputs) scores the canonical slices; score < 0 -> the series is
#            written with the mirrored row direction (-1,0,0) so canonicalization un-mirrors it (one of the two raw COR_MPR
#            tarballs had row direction -x). RSNA: the real IOP of the series rounded to signed cardinal axes.
#   centre   all series of a study share one centre (x0, 0, 0): OAI x0 = -65 mm (right knee) / +65 mm (left knee) (raw
#            follow-up tarballs: right knees at x ~ -60 to -70); RSNA x0 = the real sagittal IPP x median. Cross-series
#            planning offsets (~10-20 mm in the raw sample) are LOST: anchors are projected with this error.
#   side     OAI: no laterality tag (none in the raw OAI headers) -> sagittal IPP x rule + the worker's side check, as real
#            OAI would be served. RSNA: the real single laterality tag copied when the real study has exactly one.
#   step     OAI: SAG_IW_TSE 3.0 mm, AX_MPR 1.5 mm, COR_MPR 1.5 mm (raw follow-up headers: 3.0 / 1.5 / 1.51-1.64). RSNA: the
#            real median IPP step. n_native: OAI n_kept (oai_series_meta.csv); RSNA the real file count.
# Outputs <out>/<split>.csv, <split>_series.csv, <split>_series/<study>/<series>/NNNN.dcm and <out>/build_meta.csv (per
# series build record; stays on the box). Prints AGGREGATES ONLY.
# Usage: python -I build_synth.py --mode oai|rsna --ids IDS.json --out DIR --split NAME [--procs 16] [--views 0,2,3]
import argparse, glob, io, json, os, sys, time, hashlib
sys.dont_write_bytecode = True
from multiprocessing import Pool
import numpy as np, pandas as pd

H = os.path.expanduser('~')
RSNA = H + '/kmimic/kaggle/input/rsna-knee-abnormality-detection'
SIDE_W = H + '/r6g58/bundle/side_cls.pt'
SRC6 = H + '/r6g58/src'
STORE = 320
CROP_MM = 140.0
VIEW_PLANE = {0: ('Axial', 1), 1: ('Sagittal', 0), 2: ('Coronal', 1), 3: ('Sagittal', 1)}
STD_IOP = {'Sagittal': (0, 1, 0, 0, 0, -1), 'Coronal': (1, 0, 0, 0, 0, -1), 'Axial': (1, 0, 0, 0, 1, 0)}
MIR_IOP = {'Coronal': (-1, 0, 0, 0, 0, -1), 'Axial': (-1, 0, 0, 0, 1, 0)}
AXIS = {'Sagittal': 0, 'Coronal': 1, 'Axial': 2}
OAI_STEP = {'SAG_IW_TSE': 3.0, 'AX_MPR': 1.5, 'COR_MPR': 1.5, 'COR_IW_TSE': 3.0}


def decode(z, j):
    import cv2
    if f'v{j}_blob' not in z.files:
        return None
    blob, offs = z[f'v{j}_blob'], z[f'v{j}_offs']
    return np.stack([cv2.imdecode(blob[offs[i]:offs[i + 1]], cv2.IMREAD_GRAYSCALE) for i in range(len(offs) - 1)])


def crop_spacing(rows, cols, ps_r, ps_c):
    kr = min(rows, max(32, int(round(CROP_MM / ps_r)))) if ps_r > 0 else rows
    kc = min(cols, max(32, int(round(CROP_MM / ps_c)))) if ps_c > 0 else cols
    return kr * ps_r / STORE, kc * ps_c / STORE


def round_iop(iop):
    out = []
    for v in (np.asarray(iop[:3], float), np.asarray(iop[3:], float)):
        k = int(np.argmax(np.abs(v))); e = [0, 0, 0]; e[k] = 1 if v[k] > 0 else -1; out += e
    return tuple(out)


def rsna_series_info(uid, sid):
    import pydicom
    fs = sorted(glob.glob(f'{RSNA}/train_series/{uid}/{sid}/*.dcm'))
    hd = [pydicom.dcmread(f, stop_before_pixels=True) for f in fs]
    hd = [d for d in hd if 'Rows' in d]
    d0 = hd[len(hd) // 2]
    iop = np.array(d0.ImageOrientationPatient, float)
    n = np.cross(iop[:3], iop[3:]); n = n if n[np.argmax(np.abs(n))] > 0 else -n
    pos = np.sort([float(np.dot(np.array(d.ImagePositionPatient, float), n)) for d in hd])
    step = float(np.median(np.diff(pos))) if len(pos) > 1 else 3.0
    ps = [float(x) for x in d0.PixelSpacing]
    lat = set()
    for d in hd:
        for t in ('Laterality', 'ImageLaterality'):
            v = str(getattr(d, t, '') or '').strip().upper()
            if v in ('L', 'R', 'LEFT', 'RIGHT'):
                lat.add(v[0])
    xs = [float(d.ImagePositionPatient[0]) for d in hd]
    return dict(n_native=len(hd), step=step, ps_r=ps[0], ps_c=ps[1] if len(ps) > 1 else ps[0], rows=int(d0.Rows),
                cols=int(d0.Columns), iop=round_iop(iop), lat=lat, x_med=float(np.median(xs)),
                te=str(getattr(d0, 'EchoTime', '') or ''), tr=str(getattr(d0, 'RepetitionTime', '') or ''))


def write_series(sdir, stack, plane, iop, sp_r, sp_c, idx, n_native, step, centre, study, sid, frame_uid, te, tr, desc, lat,
                 patient):
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    os.makedirs(sdir, exist_ok=True)
    row = np.array(iop[:3], float); col = np.array(iop[3:], float)
    e = np.zeros(3); e[AXIS[plane]] = 1.0
    n, R, C = stack.shape
    c0 = np.array(centre, float) - row * sp_c * (C - 1) / 2 - col * sp_r * (R - 1) / 2
    seed = hashlib.sha256((study + '/' + sid).encode()).hexdigest()
    for k in range(n):
        pos = (float(idx[k]) - (n_native - 1) / 2.0) * step
        ipp = c0 + e * pos
        fm = FileMetaDataset()
        fm.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
        sop = generate_uid(entropy_srcs=[seed, str(k)])
        fm.MediaStorageSOPInstanceUID = sop
        fm.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset(); ds.file_meta = fm
        ds.SOPClassUID = fm.MediaStorageSOPClassUID; ds.SOPInstanceUID = sop
        ds.Modality = 'MR'; ds.PatientID = patient; ds.StudyInstanceUID = generate_uid(entropy_srcs=[study])
        ds.SeriesInstanceUID = generate_uid(entropy_srcs=[seed]); ds.FrameOfReferenceUID = frame_uid
        ds.SeriesDescription = desc; ds.InstanceNumber = k + 1
        ds.ImagePositionPatient = [f'{v:.4f}' for v in ipp]
        ds.ImageOrientationPatient = [f'{v:.6f}' for v in list(row) + list(col)]
        ds.PixelSpacing = [f'{sp_r:.6f}', f'{sp_c:.6f}']
        ds.SliceThickness = f'{step:.4f}'
        if te: ds.EchoTime = te
        if tr: ds.RepetitionTime = tr
        if lat: ds.Laterality = lat
        ds.Rows, ds.Columns = R, C
        ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.BitsAllocated = 16; ds.BitsStored = 12; ds.HighBit = 11; ds.PixelRepresentation = 0
        ds.PixelData = (stack[k].astype(np.uint16) * 16).tobytes()
        ds.save_as(f'{sdir}/{k:04d}.dcm', enforce_file_format=True)


def one(task):
    import cv2
    cv2.setNumThreads(1)
    mode, study, out, split, views, mirror, meta_rows, store = task
    recs, rows = [], []
    try:
        z = np.load(f'{store}/{study}.npz')
        frame = None
        from pydicom.uid import generate_uid
        frame = generate_uid(entropy_srcs=['frame', study])
        info = {}
        for j in views:
            st = decode(z, j)
            if st is None:
                continue
            sid = str(z[f'v{j}_uid'])
            plane, fs = VIEW_PLANE[j]
            if mode == 'oai':
                m = meta_rows.get(sid)
                if m is None:
                    recs.append(dict(study=study, view=j, status='no meta')); continue
                ps = [float(x) for x in str(m['PixelSpacing']).split('|')]
                inf = dict(n_native=int(m['n_kept']), step=OAI_STEP.get(m['base'], 3.0), ps_r=ps[0], ps_c=ps[-1],
                           rows=int(m['Rows']), cols=int(m['Columns']), te=str(m['EchoTime']), tr=str(m['RepetitionTime']),
                           desc=str(m['base']), flu=int(m['Fluid_Sensitive']), fat=int(m['Fat_Suppression']),
                           plane=str(m['Anatomical_Plane']))
                if inf['plane'] != plane:
                    recs.append(dict(study=study, view=j, status='plane mismatch')); continue
                iop = MIR_IOP[plane] if mirror.get(sid) else STD_IOP[plane]
            else:
                inf = rsna_series_info(study, sid)
                m = meta_rows[sid]
                inf.update(desc='rsna', flu=int(m['Fluid_Sensitive']), fat=int(m['Fat_Suppression']), plane=str(m['Anatomical_Plane']))
                iop = inf['iop']
            info[sid] = (j, st, inf, iop)
        if mode == 'oai':
            x0 = -65.0 if study.endswith('_R') else 65.0
            lat = ''
        else:
            sag = [v[2] for v in info.values() if VIEW_PLANE[v[0]][0] == 'Sagittal']
            x0 = float(np.median([s['x_med'] for s in sag])) if sag else 0.0
            lats = set().union(*[v[2]['lat'] for v in info.values()]) if info else set()
            lat = next(iter(lats)) if len(lats) == 1 else ''
        for sid, (j, st, inf, iop) in info.items():
            nn = inf['n_native']; n = len(st)
            exp = min(nn, 32)
            if nn > 32:
                idx = np.linspace(0, nn - 1, 32).round().astype(int)
            else:
                idx = np.arange(nn)
            status = 'ok'
            if n != exp:
                idx = np.linspace(0, nn - 1, n).round().astype(int); status = 'n mismatch'
            sp_r, sp_c = crop_spacing(inf['rows'], inf['cols'], inf['ps_r'], inf['ps_c'])
            plane = VIEW_PLANE[j][0]
            write_series(f'{out}/{split}_series/{study}/{sid}', st, plane, iop, sp_r, sp_c, idx, nn, inf['step'], (x0, 0.0, 0.0),
                         study, sid, frame, inf['te'], inf['tr'], inf['desc'], lat, study)
            rows.append(dict(StudyInstanceUID=study, SeriesInstanceUID=sid, Fluid_Sensitive=inf['flu'], Fat_Suppression=inf['fat'],
                             Anatomical_Plane=plane))
            recs.append(dict(study=study, view=j, status=status, n=n, n_native=nn, step=inf['step'], sp_r=sp_r, sp_c=sp_c,
                             iop='|'.join(map(str, iop)), mirrored=int(bool(mirror.get(sid))), x0=x0, lat=lat))
    except Exception as e:
        recs.append(dict(study=study, view=-1, status='error ' + type(e).__name__ + ': ' + str(e)[:120]))
    return rows, recs


def side_scores(mode, studies, store, views):
    """per Cor / Ax series: side-classifier score of the canonical slices under the standard IOP (OAI) or the real IOP (RSNA);
    returns {sid: score}. CPU, 16 threads."""
    import torch
    sys.path.insert(0, SRC6)
    import side_common as SC
    torch.set_num_threads(16)
    net = SC.build_side_net(); net.load_state_dict(torch.load(SIDE_W, map_location='cpu', weights_only=True), strict=True)
    net.eval()
    if mode == 'rsna':
        sys.path.insert(0, H + '/r7port/code')
        import training.dicom_geometry as G
    out, batch, keys = {}, [], []

    def flush():
        if not batch:
            return
        xb = torch.from_numpy(np.concatenate(batch)).float().div_(255.)[:, None]
        with torch.inference_mode():
            zz = net((xb - 0.45) / 0.25).float().squeeze(1).numpy()
        i = 0
        for sid, k in keys:
            out[sid] = float(zz[i:i + k].mean()); i += k
        batch.clear(); keys.clear()
    for study, side in studies:
        z = np.load(f'{store}/{study}.npz')
        for j in views:
            plane = VIEW_PLANE[j][0]
            if plane not in ('Coronal', 'Axial'):
                continue
            st = decode(z, j)
            if st is None:
                continue
            sid = str(z[f'v{j}_uid'])
            if mode == 'rsna':
                inf = rsna_series_info(study, sid)
                iop = np.array(inf['iop'], float)
                op = G.canonical_inplane_axes(iop[:3], iop[3:], plane)[2]
                st = np.stack([G._apply_inplane_transform(a, op) for a in st])
                side = side if side in ('L', 'R') else None
            if side == 'R':
                st = st[:, :, ::-1]
            v = SC.side_inputs(st)
            if len(v):
                batch.append(v); keys.append((sid, len(v)))
            if len(batch) >= 64:
                flush()
    flush()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['oai', 'rsna'], required=True)
    ap.add_argument('--ids', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--procs', type=int, default=16)
    ap.add_argument('--views', default='0,2,3')
    ap.add_argument('--no-mirror-fix', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    ids = json.load(open(a.ids))
    views = [int(x) for x in a.views.split(',')]
    store = H + ('/oai_store' if a.mode == 'oai' else '/fleet_store')
    if a.mode == 'oai':
        meta = pd.read_csv(H + '/oai_build/oai_series_meta.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
        meta = meta[meta.StudyInstanceUID.isin(set(ids))]
        sides = [(k, 'R' if k.endswith('_R') else 'L') for k in ids]
    else:
        meta = pd.read_csv(RSNA + '/train_series.csv', dtype=str)
        meta = meta[meta.StudyInstanceUID.isin(set(ids))]
        sides = []
        for u in ids:   # the real study side: single laterality tag, else sagittal IPP x median (decide_study_side)
            lat, xs = set(), []
            for r in meta[meta.StudyInstanceUID == u].itertuples():
                try:
                    inf = rsna_series_info(u, r.SeriesInstanceUID)
                except Exception:
                    continue
                lat |= inf['lat']
                if r.Anatomical_Plane == 'Sagittal':
                    xs.append(inf['x_med'])
            sides.append((u, next(iter(lat)) if len(lat) == 1 else (('L' if np.median(xs) >= 0 else 'R') if xs else None)))
    per = {u: {r['SeriesInstanceUID']: r for r in g.to_dict('records')} for u, g in meta.groupby('StudyInstanceUID')}
    sc = side_scores(a.mode, sides, store, views)
    vals = np.array(list(sc.values()))
    print('SIDE_SCORES', json.dumps({'series': len(vals), 'neg_frac': round(float((vals < 0).mean()), 4) if len(vals) else None,
                                     'abs_lt_2_frac': round(float((np.abs(vals) < 2).mean()), 4) if len(vals) else None,
                                     'q10_50_90': np.round(np.percentile(vals, [10, 50, 90]), 2).tolist() if len(vals) else None}),
          flush=True)
    mirror = {} if (a.no_mirror_fix or a.mode == 'rsna') else {s: v < 0 for s, v in sc.items()}
    os.makedirs(a.out, exist_ok=True)
    tasks = [(a.mode, u, a.out, a.split, views, mirror, per.get(u, {}), store) for u in ids]
    rows, recs = [], []
    with Pool(a.procs) as p:
        for r, c in p.imap(one, tasks, chunksize=2):
            rows += r; recs += c
    S = pd.DataFrame(rows)
    ok_studies = sorted(set(S.StudyInstanceUID)) if len(S) else []
    pd.DataFrame({'StudyInstanceUID': ok_studies}).to_csv(f'{a.out}/{a.split}.csv', index=False)
    S.to_csv(f'{a.out}/{a.split}_series.csv', index=False)
    R = pd.DataFrame(recs)
    if a.mode == 'rsna' and len(R):   # mirror-detector accuracy diagnostic (real IOP canonical -> score should be > 0)
        pass
    R.to_csv(f'{a.out}/build_meta.csv', index=False)
    st = R.status.astype(str).str.split(' ').str[0].value_counts().to_dict()
    vc = R[R.view >= 0].groupby('view').agg(n=('n', 'median'), n_native=('n_native', 'median'), step=('step', 'median'),
                                             sp_r=('sp_r', 'median'), mirrored=('mirrored', 'mean'), count=('n', 'size'))
    print('BUILD_DONE', json.dumps({'studies_in': len(ids), 'studies_out': len(ok_studies), 'series': len(S), 'status': st,
                                    'seconds': round(time.time() - t0, 1)}), flush=True)
    print(vc.round(4).to_string(), flush=True)


if __name__ == '__main__':
    main()
