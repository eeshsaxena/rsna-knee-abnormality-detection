# proto_stats.py (10-20 r5, Claude). Box only, read-only, AGGREGATES ONLY: OAI protocol geometry constants from the few raw
# follow-up tarballs still on the box (V00 raws were pruned 10-09). Prints per series type (SeriesDescription without the
# _RIGHT/_LEFT suffix) counts and medians; no ids, no paths, no per-knee values. Writes nothing.
import glob, io, os, tarfile, collections
import numpy as np, pydicom
H = os.path.expanduser('~')
tars = sorted(glob.glob(H + '/oai_raw/image03/**/*.tar.gz', recursive=True))
print('tarballs', len(tars))
S = collections.defaultdict(lambda: collections.defaultdict(list))
for t in tars:
    try:
        tf = tarfile.open(t, 'r:gz')
    except Exception as e:
        print('open fail', type(e).__name__); continue
    hd = []
    for m in tf.getmembers():
        if not m.isfile() or m.size > 64 << 20: continue
        try:
            d = pydicom.dcmread(io.BytesIO(tf.extractfile(m).read()), stop_before_pixels=True)
        except Exception:
            continue
        if 'ImageOrientationPatient' not in d: continue
        hd.append(d)
    tf.close()
    if not hd: continue
    desc = str(getattr(hd[0], 'SeriesDescription', ''))
    side = 'R' if desc.upper().endswith('RIGHT') else 'L' if desc.upper().endswith('LEFT') else '?'
    base = desc.upper().replace('_RIGHT', '').replace('_LEFT', '')
    iops = collections.Counter(tuple(np.round(np.array(d.ImageOrientationPatient, float), 1)) for d in hd)
    iop = np.array(iops.most_common(1)[0][0])
    nrm = np.cross(iop[:3], iop[3:]); ax = int(np.argmax(np.abs(nrm)))
    keep = [d for d in hd if tuple(np.round(np.array(d.ImageOrientationPatient, float), 1)) == tuple(iop)]
    nv = nrm if nrm[ax] > 0 else -nrm
    pos = np.sort([float(np.dot(np.array(d.ImagePositionPatient, float), nv)) for d in keep])
    st = np.diff(pos)
    ipp = np.array([np.array(d.ImagePositionPatient, float) for d in keep])
    rec = S[base]
    rec['side'].append(side); rec['n'].append(len(keep)); rec['n_other'].append(len(hd) - len(keep))
    rec['iop'].append(tuple(iop)); rec['rows'].append(int(keep[0].Rows)); rec['cols'].append(int(keep[0].Columns))
    rec['ps'].append(float(keep[0].PixelSpacing[0])); rec['thk'].append(float(getattr(keep[0], 'SliceThickness', np.nan) or np.nan))
    rec['sbs'].append(float(getattr(keep[0], 'SpacingBetweenSlices', np.nan) or np.nan))
    rec['step'].append(float(np.median(st)) if len(st) else np.nan)
    rec['lat'].append(str(getattr(keep[0], 'Laterality', '')) + '/' + str(getattr(keep[0], 'ImageLaterality', '')))
    rec['ipp_center_x'].append(float(np.median(ipp[:, 0]))); rec['te'].append(str(getattr(keep[0], 'EchoTime', '')))
    rec['tr'].append(str(getattr(keep[0], 'RepetitionTime', ''))); rec['bits'].append(int(keep[0].BitsStored))
    rec['photo'].append(str(keep[0].PhotometricInterpretation)); rec['slope'].append(str(getattr(keep[0], 'RescaleSlope', '')))
    rec['center_inplane'].append(tuple(np.round(ipp.mean(0) + (iop[:3] * float(keep[0].PixelSpacing[1]) * int(keep[0].Columns)
                                                                 + iop[3:] * float(keep[0].PixelSpacing[0]) * int(keep[0].Rows)) / 2, 0)))
for base, r in sorted(S.items()):
    print('==', base, '| series', len(r['n']))
    for k in ('side', 'n', 'n_other', 'rows', 'cols', 'ps', 'thk', 'sbs', 'step', 'lat', 'te', 'tr', 'bits', 'photo', 'slope', 'iop'):
        v = r[k]
        if isinstance(v[0], float):
            print('   %-8s median %.4f min %.4f max %.4f' % (k, np.nanmedian(v), np.nanmin(v), np.nanmax(v)))
        else:
            print('   %-8s %s' % (k, dict(collections.Counter(v))))
    # side vs sagittal/position sign (aggregate cross-tab)
    print('   side x sign(ipp x median):', dict(collections.Counter((s, 'x>=0' if x >= 0 else 'x<0') for s, x in zip(r['side'], r['ipp_center_x']))))
    print('   side x volume centre (rounded):', dict(collections.Counter((s, c) for s, c in zip(r['side'], r['center_inplane']))))
