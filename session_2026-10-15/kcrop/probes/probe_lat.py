# KC probe A+B (box CPU only, read-only on the data): DICOM-header laterality / slice-order geometry on the
# box's 458 train studies with DICOMs, vs the side named in the report text (all 4,407). No GPU, no writes
# outside ~/kcrop_probe. Prints aggregate counts only.
import os, re, sys, glob, json, time
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.expanduser('~/eda_data/rsna-knee-abnormality-detection')
OUT = os.path.expanduser('~/kcrop_probe')

RX_R = re.compile(r'rodilla\s+derech|\bderech[ao]\b|right\s+knee|joelho\s+direit|\bdireit[ao]\b', re.I)
RX_L = re.compile(r'rodilla\s+izquierd|\bizquierd[ao]\b|left\s+knee|joelho\s+esquerd|\besquerd[ao]\b', re.I)


def report_side(t):
    t = str(t) if isinstance(t, str) else ''
    r, l = bool(RX_R.search(t)), bool(RX_L.search(t))
    return 'R' if r and not l else 'L' if l and not r else ('B' if r and l else 'U')


def series_geo(sdir):
    import pydicom
    xs, normals, rows, lat, ilat, desc, man = [], [], [], set(), set(), set(), set()
    n = 0
    for f in glob.glob(sdir + '/*.dcm'):
        n += 1
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:
            continue
        ipp = getattr(h, 'ImagePositionPatient', None)
        iop = getattr(h, 'ImageOrientationPatient', None)
        if ipp is not None and len(ipp) == 3:
            xs.append(float(ipp[0]))
        if iop is not None and len(iop) == 6:
            r = np.array(iop[:3], float); c = np.array(iop[3:], float)
            normals.append(np.cross(r, c)); rows.append(r)
        for tag, s in (('Laterality', lat), ('ImageLaterality', ilat), ('SeriesDescription', desc), ('Manufacturer', man)):
            v = getattr(h, tag, None)
            if v not in (None, ''):
                s.add(str(v))
    nm = np.mean(normals, 0) if normals else None
    rw = np.mean(rows, 0) if rows else None
    return dict(n=n, mean_x=float(np.mean(xs)) if xs else None, normal=None if nm is None else [round(float(v), 3) for v in nm],
                row=None if rw is None else [round(float(v), 3) for v in rw], lat=sorted(lat), ilat=sorted(ilat),
                desc=sorted(desc)[:2], man=sorted(man)[:1])


def study_job(args):
    sid, recs = args
    out = {'sid': sid, 'series': []}
    for r in recs:
        sdir = f"{ROOT}/train_series/{sid}/{r['SeriesInstanceUID']}"
        if not os.path.isdir(sdir):
            continue
        g = series_geo(sdir)
        g.update(plane=r['Anatomical_Plane'], fs=r['Fat_Suppression'], fl=r['Fluid_Sensitive'])
        out['series'].append(g)
    return out


def main():
    t0 = time.time()
    tr = pd.read_csv(f'{ROOT}/train.csv', usecols=['StudyInstanceUID', 'Report'])
    tr['side_rep'] = tr['Report'].map(report_side)
    vc = tr['side_rep'].value_counts().to_dict()
    print('REPORT_SIDE all train', len(tr), vc, flush=True)
    ser = pd.read_csv(f'{ROOT}/train_series.csv', dtype=str)
    have = sorted(os.listdir(f'{ROOT}/train_series'))
    jobs = [(s, ser[ser.StudyInstanceUID == s].to_dict('records')) for s in have]
    with ProcessPoolExecutor(max_workers=int(os.environ.get('NP', 6))) as ex:
        res = list(ex.map(study_job, jobs, chunksize=4))
    print('HEADERS', len(res), 'studies', round(time.time() - t0, 1), 's', flush=True)
    json.dump(res, open(f'{OUT}/geo458.json', 'w'))
    rep = dict(zip(tr.StudyInstanceUID, tr.side_rep))
    rows = []
    for s in res:
        sag = [g for g in s['series'] if g['plane'] == 'Sagittal' and g['mean_x'] is not None]
        sag_fs = [g for g in sag if str(g['fs']) in ('1', '1.0', 'True')]
        pick = max(sag_fs or sag, key=lambda g: g['n']) if (sag_fs or sag) else None
        mx = pick['mean_x'] if pick else None
        side_hdr = 'U' if mx is None or abs(mx) <= 5.0 else ('R' if mx < 0 else 'L')
        tags = set(t for g in s['series'] for t in g['lat'] + g['ilat'])
        sag_nx = [np.sign(g['normal'][0]) for g in sag if g['normal']]
        sag_nx_dom = [abs(g['normal'][0]) > 0.7 for g in sag if g['normal']]
        cor = [g for g in s['series'] if g['plane'] == 'Coronal' and g['row']]
        ax = [g for g in s['series'] if g['plane'] == 'Axial' and g['row']]
        rows.append(dict(sid=s['sid'], rep=rep.get(s['sid'], 'U'), hdr=side_hdr, mean_x=mx,
                         tag=','.join(sorted(tags)) or '-',
                         sag_nx=','.join(str(int(v)) for v in sag_nx), sag_ok=all(sag_nx_dom) if sag_nx_dom else None,
                         cor_rx=','.join(str(int(np.sign(g['row'][0]))) for g in cor),
                         ax_rx=','.join(str(int(np.sign(g['row'][0]))) for g in ax),
                         man=(s['series'][0]['man'] or ['-'])[0] if s['series'] else '-'))
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/lat458.csv', index=False)
    print('HDR_SIDE', df.hdr.value_counts().to_dict())
    print('REP_SIDE(458)', df.rep.value_counts().to_dict())
    print('LAT_TAG', df.tag.value_counts().head(6).to_dict())
    both = df[df.rep.isin(['R', 'L']) & df.hdr.isin(['R', 'L'])]
    print('AGREE hdr vs report', int((both.rep == both.hdr).sum()), 'of', len(both))
    tagd = df[df.tag.isin(['R', 'L']) & df.hdr.isin(['R', 'L'])]
    print('AGREE hdr vs tag', int((tagd.tag == tagd.hdr).sum()), 'of', len(tagd))
    tagr = df[df.tag.isin(['R', 'L']) & df.rep.isin(['R', 'L'])]
    print('AGREE tag vs report', int((tagr.tag == tagr.rep).sum()), 'of', len(tagr))
    print('ABS_MEANX quantiles', np.nanpercentile(np.abs(df.mean_x.astype(float)), [1, 5, 10, 25, 50]).round(1).tolist())
    print('SAG normal-x sign', df.sag_nx.value_counts().head(6).to_dict(), '| sag normal ~ x:', df.sag_ok.value_counts().to_dict())
    print('COR row-x sign', df.cor_rx.value_counts().head(6).to_dict())
    print('AX row-x sign', df.ax_rx.value_counts().head(6).to_dict())
    print('MANUF', df.man.value_counts().head(6).to_dict())
    print('DONE', round(time.time() - t0, 1), 's')


if __name__ == '__main__':
    main()
