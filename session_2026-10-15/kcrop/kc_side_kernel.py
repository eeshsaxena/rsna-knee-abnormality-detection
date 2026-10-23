# kc_side_kernel.py: Kaggle CPU kernel (no GPU, no internet) for MAIN to push. Header-only pass over the TRAIN studies:
# per study, the knee side from DICOM geometry, i.e. the exact rule the KC serve code will apply to the hidden test set.
# Rule (bliverrigpdoge pack_submit.study_mean_x / side_of, lat_mm 5, plus a tag fallback near the midline):
#   side series = the sagittal fat-suppressed series with the most files, else any sagittal, else none
#   mean_x = mean ImagePositionPatient[0] over that series; side = R if mean_x < -T, L if mean_x > T
#   |mean_x| <= T (T = 30 mm): DICOM Laterality / ImageLaterality tag of the study if it names one side, else the
#   mean_x sign if |mean_x| > 5 mm, else U (unresolved = left frame, no flip)
# MEASURED on the box's 458 train studies (probe_lat.py, 10-15): sagittal normal x < 0 and coronal / axial row x > 0 in
# every series; mean_x side vs report side 111/114, vs tag 215/220; the disagreements sit at |mean_x| 8.8-26.5 mm or are
# tag errors at |mean_x| > 80 mm. Output: /kaggle/working/kc_side_train.csv (RSNA metadata only, a few hundred KB).
import os, glob, time
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
import pydicom

ROOT = next((c for c in ('/kaggle/input/competitions/rsna-knee-abnormality-detection',
                         '/kaggle/input/rsna-knee-abnormality-detection') if os.path.exists(c + '/train_series.csv')), None)
assert ROOT, 'competition root not found'
T_MM, LAT_MM = 30.0, 5.0
TAGS = ['ImagePositionPatient', 'ImageOrientationPatient', 'Laterality', 'ImageLaterality']


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


def study(args):
    sid, rows = args
    sag = [r for r in rows if r['Anatomical_Plane'] == 'Sagittal']
    fs = [r for r in sag if str(r.get('Fat_Suppression')) in ('1', '1.0', 'True')]
    out = dict(StudyInstanceUID=sid, mean_x=np.nan, n=0, sag_nx=np.nan, tag='', side='U', rule='none')
    best = None
    for cand in (fs, sag):
        if cand:
            best = max(cand, key=lambda r: len(glob.glob(f"{ROOT}/train_series/{sid}/{r['SeriesInstanceUID']}/*.dcm")))
            break
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


t0 = time.time()
ser = pd.read_csv(f'{ROOT}/train_series.csv', dtype=str)
jobs = [(k, v.to_dict('records')) for k, v in ser.groupby('StudyInstanceUID')]
with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(study, jobs))
df = pd.DataFrame(res).sort_values('StudyInstanceUID')
df.to_csv('/kaggle/working/kc_side_train.csv', index=False)
print('KC_SIDE studies', len(df), dict(df.side.value_counts()), dict(df.rule.value_counts()),
      '| sag normal x < 0:', int((df.sag_nx < -0.7).sum()), '| seconds', round(time.time() - t0, 1))
