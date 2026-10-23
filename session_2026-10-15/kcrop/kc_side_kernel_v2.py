# kc_side_kernel_v2.py: Kaggle CPU kernel (no GPU, no internet) for MAIN to push. = kc_side_kernel.py (v1, kept unchanged
# next to this file) + the per-slot series geometry that BOX_PLAN_CRITIQUE_1015 asks for (adopted):
#   v1 part (verbatim): side series = the sagittal fat-suppressed series with the most files, else any sagittal; mean_x =
#     mean IPP[0] over it; side = R if mean_x < -30, L if > 30; else the Laterality / ImageLaterality tag if it names one
#     side; else the mean_x sign if |mean_x| > 5 mm; else U.
#   v2 part (new): replicate the d96 reader's slot fill EXACTLY (verbatim _pick_series_for_slot, SER rows in train_series.csv
#     order as the corpus builder prep_kernel_s0.py made them: pd.read_csv default dtypes, groupby StudyInstanceUID) and,
#     for the series that fills each slot, record its geometry from 3 headers (first / middle / last file in sorted order):
#       g_sagfs, g_sagpd = sagittal normal x (cross(row, col)[0]); g_corfs, g_corpd, g_ax = row direction x (IOP[0]).
#     NaN = no series / no files / unreadable / the 3 headers disagree in sign. geom_ok = every filled KC slot (sagfs, sagpd,
#     corfs, ax) has the expected sign (sag < 0, cor / ax > 0) with |value| >= 0.7.
# The KC store (kc_build_store.py) and the serve worker derive each flip from that slot's geometry XOR the side.
# Output: /kaggle/working/kc_side_train.csv (RSNA metadata only, ~1 MB). Box test: KC_ROOT=<eda root> KC_OUT=<csv> python ...
import os, glob, time
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
import pydicom

ROOT = next((c for c in ('/kaggle/input/competitions/rsna-knee-abnormality-detection',
                         '/kaggle/input/rsna-knee-abnormality-detection') if os.path.exists(c + '/train_series.csv')), None)
ROOT = os.environ.get('KC_ROOT', ROOT)                     # box test only; unset on Kaggle
OUT = os.environ.get('KC_OUT', '/kaggle/working/kc_side_train.csv')
assert ROOT, 'competition root not found'
T_MM, LAT_MM = 30.0, 5.0
TAGS = ['ImagePositionPatient', 'ImageOrientationPatient', 'Laterality', 'ImageLaterality']
# d96 reader slots (verbatim SLOTS of the reader sha 30aa72fd) and the KC block name of each
SLOTS = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18), ("Coronal", 0, 12), ("Axial", -1, 18)]
SLOT_NAME = ['sagfs', 'sagpd', 'corfs', 'corpd', 'ax']
EXPECT = dict(sagfs=-1, sagpd=-1, corfs=1, corpd=1, ax=1)
KC_SLOTS = ('sagfs', 'sagpd', 'corfs', 'ax')


def hdr(path):
    try:
        h = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=TAGS)
    except Exception:
        return None
    ipp = getattr(h, 'ImagePositionPatient', None)
    iop = getattr(h, 'ImageOrientationPatient', None)
    tag = {str(getattr(h, t, '') or '').strip().upper()[:1] for t in ('Laterality', 'ImageLaterality')} - {''}
    nx = None
    if iop is not None and len(iop) == 6:
        nx = float(np.cross(np.array(iop[:3], float), np.array(iop[3:], float))[0])
    return (float(ipp[0]) if ipp is not None and len(ipp) == 3 else None, nx, tag)


# verbatim from the d96 reader (sha 30aa72fd)
def _pick_series_for_slot(rows, plane, fluid, used):
    cands = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get('Fluid_Sensitive', 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def geom_one(path):
    try:
        h = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=['ImageOrientationPatient'])
        o = np.array(h.ImageOrientationPatient, float)
        if o.shape != (6,):
            return None
        return float(np.cross(o[:3], o[3:])[0]), float(o[0])
    except Exception:
        return None


def slot_geom(sid, rows):
    """-> {g_<slot>: value, u_<slot>: last 8 chars of the filling series uid} for the 5 d96 slots."""
    out, used = {}, set()
    for (plane, fluid, _k), name in zip(SLOTS, SLOT_NAME):
        r = _pick_series_for_slot(rows, plane, fluid, used)
        out[f'g_{name}'] = np.nan; out[f'u_{name}'] = ''
        if r is None:
            continue
        used.add(r['SeriesInstanceUID'])
        out[f'u_{name}'] = r['SeriesInstanceUID'][-8:]
        fs = sorted(glob.glob(f"{ROOT}/train_series/{sid}/{r['SeriesInstanceUID']}/*.dcm"))
        if not fs:
            continue
        pick = sorted({0, len(fs) // 2, len(fs) - 1})
        g = [geom_one(fs[i]) for i in pick]
        g = [x for x in g if x is not None]
        if not g:
            continue
        vals = [x[0] if plane == 'Sagittal' else x[1] for x in g]
        if len({np.sign(v) for v in vals}) == 1:
            out[f'g_{name}'] = float(np.mean(vals))
    return out


def study(args):
    sid, rows, rows_csv = args
    sag = [r for r in rows if r['Anatomical_Plane'] == 'Sagittal']
    fs = [r for r in sag if str(r.get('Fat_Suppression')) in ('1', '1.0', 'True')]
    out = dict(StudyInstanceUID=sid, mean_x=np.nan, n=0, sag_nx=np.nan, tag='', side='U', rule='none')
    best = None
    for cand in (fs, sag):
        if cand:
            best = max(cand, key=lambda r: len(glob.glob(f"{ROOT}/train_series/{sid}/{r['SeriesInstanceUID']}/*.dcm")))
            break
    out.update(slot_geom(sid, rows_csv))                                     # v2
    if best is None:
        return out
    files = glob.glob(f"{ROOT}/train_series/{sid}/{best['SeriesInstanceUID']}/*.dcm")
    with ThreadPoolExecutor(16) as ex:
        hs = [h for h in ex.map(hdr, files) if h is not None]
    xs = [h[0] for h in hs if h[0] is not None]
    nxs = [h[1] for h in hs if h[1] is not None]
    tags = set().union(*[h[2] for h in hs]) if hs else set()
    out.update(n=len(files), mean_x=float(np.mean(xs)) if xs else np.nan, sag_nx=float(np.mean(nxs)) if nxs else np.nan,
               tag=''.join(sorted(tags)))
    mx = out['mean_x']
    if np.isfinite(mx) and abs(mx) > T_MM:
        out.update(side='R' if mx < 0 else 'L', rule='mean_x')
    elif tags in ({'R'}, {'L'}):
        out.update(side=next(iter(tags)), rule='tag')
    elif np.isfinite(mx) and abs(mx) > LAT_MM:
        out.update(side='R' if mx < 0 else 'L', rule='mean_x_weak')
    return out


def gok(r):
    for s in KC_SLOTS:
        if r[f'u_{s}'] == '':
            continue                                   # empty slot: nothing to orient
        v = r[f'g_{s}']
        if not np.isfinite(v) or abs(v) < 0.7 or np.sign(v) != EXPECT[s]:
            return 0
    return 1


t0 = time.time()
ser = pd.read_csv(f'{ROOT}/train_series.csv', dtype=str)                          # v1 rows (string dtypes)
ser_csv = pd.read_csv(f'{ROOT}/train_series.csv')                                 # corpus-builder rows (prep_kernel_s0)
ser_csv['StudyInstanceUID'] = ser_csv['StudyInstanceUID'].astype(str)
ser_csv['SeriesInstanceUID'] = ser_csv['SeriesInstanceUID'].astype(str)
SER = {k: v.to_dict('records') for k, v in ser_csv.groupby('StudyInstanceUID')}
jobs = [(k, v.to_dict('records'), SER.get(k, [])) for k, v in ser.groupby('StudyInstanceUID')]
if os.environ.get('KC_ONLY_PRESENT'):                                             # box test: studies with DICOMs only
    jobs = [j for j in jobs if os.path.isdir(f'{ROOT}/train_series/{j[0]}')]
with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(study, jobs))
df = pd.DataFrame(res).sort_values('StudyInstanceUID')
df['geom_ok'] = [gok(r) for r in df.to_dict('records')]
df.to_csv(OUT, index=False)
filled = {s: int((df[f'u_{s}'] != '').sum()) for s in SLOT_NAME}
expect = {s: int(((df[f'u_{s}'] != '') & (np.sign(df[f'g_{s}']) == EXPECT[s]) & (df[f'g_{s}'].abs() >= 0.7)).sum())
          for s in SLOT_NAME}
print('KC_SIDE_V2 studies', len(df), dict(df.side.value_counts()), dict(df.rule.value_counts()),
      '| sag normal x < 0:', int((df.sag_nx < -0.7).sum()), '| slot filled', filled, '| slot expected-geometry', expect,
      '| geom_ok', int(df.geom_ok.sum()), '| seconds', round(time.time() - t0, 1))
