# probe_jc.py: worker of the rsna-jpeg-check CPU kernel (2026-10-14, Claude). Read-only on /kaggle/input.
#   make : transcode ONE real public-test study into every fixture transfer syntax (fake competition root)
#   probe: decode every fixture study through each stage's reader in THIS process environment -> JSON
# The parent runs `probe` in 4 environments: S (stock image), SW (stock + decoder wheels on PYTHONPATH),
# R (after R7's pip install of pydicom 3.0.1 / timm 1.0.25), RW (R + decoder wheels).
import os, sys, json, glob, time, io, copy, hashlib, traceback, argparse, shutil, importlib.util, faulthandler
import numpy as np
faulthandler.enable()

T0 = time.time()
UID_PREFIX = '1.2.826.0.1.3680043.8.498.'
VARIANTS = ['U', 'IMPL', 'JLL70', 'JLL57', 'J2KR', 'J2KI', 'RLE', 'MIX']
TS_OF = {'U': '1.2.840.10008.1.2.1', 'IMPL': '1.2.840.10008.1.2', 'JLL70': '1.2.840.10008.1.2.4.70',
         'JLL57': '1.2.840.10008.1.2.4.57', 'J2KR': '1.2.840.10008.1.2.4.90', 'J2KI': '1.2.840.10008.1.2.4.91',
         'RLE': '1.2.840.10008.1.2.5'}
LOSSY = {'J2KI'}
# pydicom's own bundled test files (shipped inside the pydicom wheel), decoded by plain pydicom only
BUNDLED = ['MR_small.dcm', 'MR_small_implicit.dcm', 'MR_small_RLE.dcm', 'MR_small_jp2klossless.dcm',
           'MR_small_jpeg_ls_lossless.dcm', 'JPEG2000.dcm', '693_J2KI.dcm', 'GDCMJ2K_TextGBR.dcm', 'JPEG-lossy.dcm',
           'JPGExtended.dcm', 'SC_rgb_jpeg_dcmtk.dcm', 'SC_rgb_jpeg_gdcm.dcm', 'SC_rgb_rle.dcm',
           'JPEGLSNearLossless_16.dcm', 'CT_small.dcm']
DECODE_TS = ['1.2.840.10008.1.2.1', '1.2.840.10008.1.2', '1.2.840.10008.1.2.5', '1.2.840.10008.1.2.4.50',
             '1.2.840.10008.1.2.4.51', '1.2.840.10008.1.2.4.57', '1.2.840.10008.1.2.4.70', '1.2.840.10008.1.2.4.80',
             '1.2.840.10008.1.2.4.81', '1.2.840.10008.1.2.4.90', '1.2.840.10008.1.2.4.91']


def log(*a):
    print('[probe %6.1fs]' % (time.time() - T0), *a, flush=True)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def err_s(e):
    return '%s: %s' % (type(e).__name__, str(e).replace('\n', ' ')[:300])


# ----------------------------------------------------------------------------------------------------------------
# environment report
# ----------------------------------------------------------------------------------------------------------------
def env_report():
    r = {'python': sys.version.split()[0], 'PYTHONPATH': os.environ.get('PYTHONPATH', '')}
    from importlib import metadata
    for dist in ('torch', 'timm', 'pandas', 'pillow', 'opencv-python', 'opencv-python-headless', 'pydicom', 'python-gdcm',
                 'pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg', 'pylibjpeg-rle', 'imagecodecs', 'pyjpegls'):
        try:
            r['dist_' + dist] = metadata.version(dist)          # installed-distribution versions, no import
        except Exception:
            r['dist_' + dist] = None
    for m in ['pydicom', 'numpy', 'cv2', 'PIL', 'pylibjpeg', 'libjpeg', 'openjpeg', 'gdcm', 'imagecodecs', 'jpeg_ls',
              'pyjpegls', 'rle']:
        try:
            mod = __import__(m)
            r[m] = '%s @ %s' % (getattr(mod, '__version__', '?'), os.path.dirname(getattr(mod, '__file__', '') or '?'))
        except Exception as e:
            r[m] = 'NO (%s)' % type(e).__name__
    try:
        from PIL import features
        r['PIL_features'] = {k: bool(features.check(k)) for k in ('jpg', 'jpg_2000', 'libtiff')}
    except Exception as e:
        r['PIL_features'] = err_s(e)
    dec = {}
    try:
        from pydicom.pixels import get_decoder
        from pydicom.uid import UID
        for ts in DECODE_TS:
            try:
                d = get_decoder(UID(ts))
                dec[ts] = {'name': UID(ts).name, 'available': list(d.available_plugins),
                           'missing': list(d.missing_dependencies)}
            except Exception as e:
                dec[ts] = {'name': ts, 'error': err_s(e)}
    except Exception as e:
        dec = {'error': err_s(e)}
    r['pydicom_decoders'] = dec
    return r


# ----------------------------------------------------------------------------------------------------------------
# make: fixture studies
# ----------------------------------------------------------------------------------------------------------------
def _load_jll(mods):
    spec = importlib.util.spec_from_file_location('jll', os.path.join(mods, 'jll.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _encode_one(job):
    """job = (src path, dst path, syntax, new study, new series, variant, mods) -> record."""
    src, dst, syn, nst, nse, var, mods = job
    import pydicom
    from pydicom.uid import (generate_uid, ImplicitVRLittleEndian, ExplicitVRLittleEndian, JPEGLosslessSV1,
                             JPEGLossless, JPEG2000Lossless, JPEG2000, RLELossless, UID)
    from pydicom.encaps import encapsulate
    rec = {'file': os.path.basename(dst), 'syntax': syn}
    try:
        ds = pydicom.dcmread(src)
        a = ds.pixel_array
        rec['src_ts'] = str(ds.file_meta.TransferSyntaxUID)
        ds.StudyInstanceUID = nst
        ds.SeriesInstanceUID = nse
        sop = generate_uid(UID_PREFIX, entropy_srcs=[str(ds.SOPInstanceUID), var])
        ds.SOPInstanceUID = sop
        ds.file_meta.MediaStorageSOPInstanceUID = sop
        if syn == 'U':
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        elif syn == 'IMPL':
            ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
        elif syn in ('JLL70', 'JLL57'):
            jll = _load_jll(mods)
            bs = int(ds.BitsStored)
            raw = a.astype(np.int64) & ((1 << bs) - 1) if int(ds.PixelRepresentation) == 1 else a.astype(np.int64)
            if raw.min() < 0 or raw.max() >= (1 << bs):
                raise ValueError('pixel values exceed BitsStored %d' % bs)
            ds.PixelData = encapsulate([jll.encode_sv1(raw, bs)])
            ds['PixelData'].VR = 'OB'
            ds['PixelData'].is_undefined_length = True
            ds.file_meta.TransferSyntaxUID = JPEGLosslessSV1 if syn == 'JLL70' else JPEGLossless
        elif syn == 'J2KR':
            ds.compress(JPEG2000Lossless, encoding_plugin='pylibjpeg', generate_instance_uid=False)
        elif syn == 'J2KI':
            ds.compress(JPEG2000, encoding_plugin='pylibjpeg', j2k_cr=[20], generate_instance_uid=False)
        elif syn == 'RLE':
            ds.compress(RLELossless, encoding_plugin='pydicom', generate_instance_uid=False)
        else:
            raise ValueError(syn)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        ds.save_as(dst, enforce_file_format=True)
        # validation in the make environment (decoder wheels present)
        d2 = pydicom.dcmread(dst)
        rec['ts'] = str(d2.file_meta.TransferSyntaxUID)
        b = d2.pixel_array
        rec['shape_ok'] = bool(b.shape == a.shape)
        rec['exact'] = bool(b.shape == a.shape and np.array_equal(b, a))
        rec['maxdiff'] = float(np.abs(b.astype(np.float64) - a.astype(np.float64)).max()) if b.shape == a.shape else -1.0
        rec['bytes'] = os.path.getsize(dst)
        rec['bits_stored'] = int(ds.BitsStored); rec['pixrep'] = int(ds.PixelRepresentation)
    except Exception as e:
        rec['error'] = err_s(e)
        rec['tb'] = traceback.format_exc()[-1500:]
    return rec


def make(a):
    import pandas as pd
    from pydicom.uid import generate_uid
    comp, out = a.comp, a.out
    test = pd.read_csv(os.path.join(comp, 'test.csv'), dtype={'StudyInstanceUID': str})
    tser = pd.read_csv(os.path.join(comp, 'test_series.csv'), dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
    cands = []
    for sid in test['StudyInstanceUID']:
        rows = tser[tser['StudyInstanceUID'] == sid]
        planes = set(rows['Anatomical_Plane'].astype(str))
        nf = sum(len(glob.glob(os.path.join(comp, 'test_series', sid, se, '*.dcm'))) for se in rows['SeriesInstanceUID'])
        cands.append((0 if {'Sagittal', 'Coronal', 'Axial'} <= planes else 1, nf, sid))
    cands.sort()
    log('source study candidates (missing-plane flag, files, uid):', cands)
    src_sid = cands[0][2]
    rows = tser[tser['StudyInstanceUID'] == src_sid].reset_index(drop=True)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, 'test_series'))
    jobs, meta, new_rows = [], {}, []
    # v2: new series UIDs keep the ORIGINAL SeriesInstanceUID sort order (fleet_common.select_series tries candidates in
    # UID order; v1's hash UIDs re-ordered same-view candidates per variant, a fixture artefact, not a decode effect)
    rank = {u: k for k, u in enumerate(sorted(rows['SeriesInstanceUID'].astype(str)))}
    for vi, var in enumerate(VARIANTS):
        nst = generate_uid(UID_PREFIX, entropy_srcs=[src_sid, var])
        series = []
        for i, r in rows.iterrows():
            se = str(r['SeriesInstanceUID'])
            h = int(hashlib.sha256((src_sid + '/' + se + '/' + var).encode()).hexdigest(), 16) % 10 ** 18
            nse = '%s%d.%d%018d' % (UID_PREFIX, 10 + rank[se], vi + 1, h)
            syn = ('JLL70' if i % 2 == 0 else 'U') if var == 'MIX' else var
            fs = sorted(glob.glob(os.path.join(comp, 'test_series', src_sid, se, '*.dcm')))
            series.append({'idx': int(i), 'src': se, 'uid': nse, 'syntax': syn, 'plane': str(r['Anatomical_Plane']),
                           'n_files': len(fs)})
            for f in fs:
                jobs.append((f, os.path.join(out, 'test_series', nst, nse, os.path.basename(f)), syn, nst, nse, var, a.mods))
            nr = dict(r); nr['StudyInstanceUID'] = nst; nr['SeriesInstanceUID'] = nse
            new_rows.append(nr)
        meta[var] = {'study': nst, 'series': series}
    log('encoding', len(jobs), 'files for', len(VARIANTS), 'variants of', src_sid)
    from multiprocessing import Pool
    with Pool(max(1, min(4, os.cpu_count() or 1))) as pool:
        recs = pool.map(_encode_one, jobs, chunksize=4)
    by = {}
    for (f, dst, syn, nst, nse, var, _), rec in zip(jobs, recs):
        by.setdefault(var, []).append(rec)
    summ = {}
    for var, rs in by.items():
        s = {'files': len(rs), 'errors': sum('error' in r for r in rs), 'exact': sum(bool(r.get('exact')) for r in rs),
             'maxdiff': max([r.get('maxdiff', -1.0) for r in rs] or [-1.0]),
             'ts': sorted(set(r.get('ts', '?') for r in rs)), 'bytes': sum(r.get('bytes', 0) for r in rs),
             'bits_stored': sorted(set(r.get('bits_stored', -1) for r in rs)),
             'pixrep': sorted(set(r.get('pixrep', -1) for r in rs)),
             'src_ts': sorted(set(r.get('src_ts', '?') for r in rs)),
             'first_error': next((r['error'] for r in rs if 'error' in r), None),
             'first_tb': next((r['tb'] for r in rs if 'tb' in r), None)}
        summ[var] = s
        meta[var]['encode'] = s
        log('make', var, json.dumps({k: v for k, v in s.items() if k != 'first_tb'}))
    pd.DataFrame({'StudyInstanceUID': [meta[v]['study'] for v in VARIANTS]}).to_csv(os.path.join(out, 'test.csv'), index=False)
    pd.DataFrame(new_rows, columns=list(tser.columns)).to_csv(os.path.join(out, 'test_series.csv'), index=False)
    ss = os.path.join(comp, 'sample_submission.csv')
    if os.path.exists(ss):
        cols = list(pd.read_csv(ss, nrows=1).columns)
        df = pd.DataFrame({c: 0.5 for c in cols[1:]}, index=range(len(VARIANTS)))
        df.insert(0, cols[0], [meta[v]['study'] for v in VARIANTS])
        df.to_csv(os.path.join(out, 'sample_submission.csv'), index=False)
    json.dump({'source_study': src_sid, 'variants': meta, 'ts_of': TS_OF}, open(os.path.join(out, 'fixtures.json'), 'w'), indent=1)
    log('MAKE_DONE', src_sid)


# ----------------------------------------------------------------------------------------------------------------
# probe: the readers
# ----------------------------------------------------------------------------------------------------------------
SLOTS_RAPTOR = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18), ("Coronal", 0, 12), ("Axial", -1, 18)]
IMG_RAPTOR = 384


def probe_plain(fx, fake):
    import pydicom
    out = {}
    ref = {}
    for var in VARIANTS:
        m = fx['variants'][var]
        n = ok = exact = 0; first = None; maxd = 0.0; t0 = time.time()
        for s in m['series']:
            for f in sorted(glob.glob(os.path.join(fake, 'test_series', m['study'], s['uid'], '*.dcm'))):
                key = (s['idx'], os.path.basename(f)); n += 1
                try:
                    arr = pydicom.dcmread(f).pixel_array
                    ok += 1
                    if var == 'U':
                        ref[key] = arr
                    r = ref.get(key)
                    if r is not None and r.shape == arr.shape:
                        dd = float(np.abs(arr.astype(np.float64) - r.astype(np.float64)).max())
                        maxd = max(maxd, dd); exact += int(dd == 0.0)
                except Exception as e:
                    if first is None:
                        first = err_s(e)
        out[var] = {'files': n, 'ok': ok, 'exact_vs_U': exact, 'maxdiff_vs_U': maxd, 'first_error': first,
                    'seconds': round(time.time() - t0, 2)}
        log('plain', var, json.dumps(out[var]))
    return out


def probe_bundled():
    import pydicom
    tdir = os.path.join(os.path.dirname(pydicom.__file__), 'data', 'test_files')   # inside the wheel; no download path
    out = {}
    for name in BUNDLED:
        r = {}
        try:
            p = os.path.join(tdir, name)
            if not os.path.isfile(p):
                raise FileNotFoundError(name)
            d = pydicom.dcmread(p)
            r['ts'] = str(d.file_meta.TransferSyntaxUID)
            r['ts_name'] = d.file_meta.TransferSyntaxUID.name
            arr = d.pixel_array
            r['ok'] = True; r['shape'] = list(arr.shape); r['dtype'] = str(arr.dtype)
        except Exception as e:
            r['ok'] = False; r['error'] = err_s(e)
        out[name] = r
    log('bundled', json.dumps({k: (v.get('ts'), v.get('ok')) for k, v in out.items()}))
    return out


def probe_synth(fixdir):
    import pydicom
    out = {'files': 0, 'ok': 0, 'ts': {}, 'errors': {}}
    for f in sorted(glob.glob(os.path.join(fixdir, 'corpus', 'dicom', '*.dcm'))):
        out['files'] += 1
        try:
            d = pydicom.dcmread(f)
            ts = str(d.file_meta.TransferSyntaxUID)
            out['ts'][ts] = out['ts'].get(ts, 0) + 1
            d.pixel_array
            out['ok'] += 1
        except Exception as e:
            out['errors'][os.path.basename(f)] = err_s(e)[:120]
    log('synthetic fixtures', json.dumps({k: out[k] for k in ('files', 'ok', 'ts')}), 'errors:', len(out['errors']))
    return out


def _ser_records(fake):
    import pandas as pd
    tser = pd.read_csv(os.path.join(fake, 'test_series.csv'))
    tser['StudyInstanceUID'] = tser['StudyInstanceUID'].astype(str)
    tser['SeriesInstanceUID'] = tser['SeriesInstanceUID'].astype(str)
    return tser, {k: v.to_dict('records') for k, v in tser.groupby('StudyInstanceUID')}


def probe_raptor(fx, fake, fastread_path):
    spec = importlib.util.spec_from_file_location('_fastread', fastread_path)
    fr = importlib.util.module_from_spec(spec); spec.loader.exec_module(fr)
    build = fr.make_builder('study_lut', IMG_RAPTOR, SLOTS_RAPTOR, hdr_threads=32, px_threads=16)
    _, SER = _ser_records(fake)
    tsdir = os.path.join(fake, 'test_series')
    out, ref = {}, None
    for var in VARIANTS:
        sid = fx['variants'][var]['study']
        r = {}
        try:
            t0 = time.time()
            vol, mask = build(sid, SER, tsdir)
            r['seconds'] = round(time.time() - t0, 2)
            r['mask_sum'] = int(mask.sum()); r['sha'] = sha(vol)
            if var == 'U':
                ref = (vol, mask)
            if ref is not None:
                rv, rm = ref
                r['U_mask_sum'] = int(rm.sum())
                r['slices_identical'] = int(sum(np.array_equal(vol[i], rv[i]) for i in range(len(vol))))
                r['slices_total'] = int(len(vol))
                r['identical'] = bool(np.array_equal(vol, rv))
                r['maxdiff'] = int(np.abs(vol.astype(np.int16) - rv.astype(np.int16)).max())
                r['lost_slices'] = int(((rm > 0) & (mask == 0)).sum())
        except Exception as e:
            r['error'] = err_s(e); r['tb'] = traceback.format_exc()[-1200:]
        out[var] = r
        log('raptor', var, json.dumps({k: v for k, v in r.items() if k != 'tb'}))
    return out


def probe_fleet(fx, fake, fleet_dir):
    sys.path.insert(0, fleet_dir)
    import fleet_common as fc
    tser, _ = _ser_records(fake)
    img_root = os.path.join(fake, 'test_series')
    out, ref = {}, None
    for var in VARIANTS:
        sid = fx['variants'][var]['study']
        r = {}
        try:
            stats, errs = {}, []
            t0 = time.time()
            st = fc.build_study(tser[tser['StudyInstanceUID'] == sid], img_root, sid, stats=stats, errs=errs)
            r['seconds'] = round(time.time() - t0, 2)
            pres = {'%s%d' % (v[0][:3], v[1]): (st[v] is not None) for v in fc.VIEWS}
            r['views_present'] = int(sum(pres.values())); r['views'] = pres; r['errs'] = errs[:6]
            r['sha'] = sha(np.concatenate([st[v][0].ravel() for v in fc.VIEWS if st[v] is not None])) if r['views_present'] else None
            if var == 'U':
                ref = st
            if ref is not None:
                same, md = 0, 0
                for v in fc.VIEWS:
                    if st[v] is not None and ref[v] is not None and st[v][0].shape == ref[v][0].shape:
                        dd = int(np.abs(st[v][0].astype(np.int16) - ref[v][0].astype(np.int16)).max())
                        md = max(md, dd); same += int(dd == 0 and st[v][3] is not None)
                r['U_views_present'] = int(sum(ref[v] is not None for v in fc.VIEWS))
                r['views_identical'] = same; r['maxdiff'] = md
                r['identical'] = bool(same == r['U_views_present'] == r['views_present'])
        except Exception as e:
            r['error'] = err_s(e); r['tb'] = traceback.format_exc()[-1200:]
        out[var] = r
        log('fleet', var, json.dumps({k: v for k, v in r.items() if k != 'tb'}))
    return out


def probe_r7(fx, fake, a):
    sys.path.insert(0, a.r7src)
    if a.code not in sys.path:
        sys.path.insert(0, a.code)
    import torch
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    import compact_infer as CI
    uids = [fx['variants'][v]['study'] for v in VARIANTS]
    tmp = os.path.join(a.tmp, 'r7tmp'); os.makedirs(tmp, exist_ok=True)
    ds = CI.CStudies(fake, 'test', uids, tmp, a.code, 384, 1, 'v4', 3, a.anchor, 3.5, a.side, 2.0, 0.0, 20.0,
                     extent_meta=True, fluid=True)
    out, ref = {}, None
    for i, var in enumerate(VARIANTS):
        uid = uids[i]
        r = {}
        # (1) the decode step alone: mainline native prepare (the same call CStudies makes), per-series record
        try:
            if ds.native is None:
                ds._setup()
            work = os.path.join(tmp, 'p_' + hashlib.sha256(uid.encode()).hexdigest()[:12])
            shutil.rmtree(work, ignore_errors=True)
            import contextlib
            ds.PI.OneStudySource.current = uid
            with contextlib.redirect_stdout(io.StringIO()):
                ds.native.prepare(ds.data, work, split='test', study_ids=[uid], source_directory=True)
            key = hashlib.sha256(uid.encode()).hexdigest()[:24]
            study = json.load(open(os.path.join(work, 'studies', key + '.json'), encoding='utf-8'))
            r['native_series'] = len(study['series'])
            r['native_slices'] = int(sum(len(s['files']) for s in study['series']))
            r['native_bad_slices'] = int(sum(len(s.get('errors') or []) for s in study['series']))
            r['native_errors'] = [{'series': str(e.get('series'))[-10:], 'error': str(e.get('error'))[:160],
                                   'first_detail': (str((e.get('details') or [{}])[0].get('error'))[:200] if e.get('details') else None)}
                                  for e in study['errors']][:6]
            shutil.rmtree(work, ignore_errors=True)
        except Exception as e:
            r['native_error'] = err_s(e); r['native_tb'] = traceback.format_exc()[-1200:]
        # (2) the full per-study item the R7 worker scores (decode + render + anchors + side check + windows)
        try:
            item = ds[i]
            r['item_error'] = item['error']
            imgs = item['images'].numpy()
            r['windows'] = int(imgs.shape[0]); r['sha'] = sha(imgs) if imgs.size else None
            r['decode_render_anchor_s'] = item.get('seconds')
            r['side'] = item.get('side'); r['anchor'] = item.get('anchor')
            if var == 'U':
                ref = imgs
            if ref is not None:
                r['U_windows'] = int(ref.shape[0])
                r['identical'] = bool(imgs.shape == ref.shape and np.array_equal(imgs, ref))
                r['maxdiff'] = int(np.abs(imgs.astype(np.int16) - ref.astype(np.int16)).max()) if imgs.shape == ref.shape and imgs.size else None
        except Exception as e:
            r['item_exc'] = err_s(e); r['item_tb'] = traceback.format_exc()[-1500:]
        out[var] = r
        log('r7', var, json.dumps({k: v for k, v in r.items() if not k.endswith('_tb')}, default=str))
    return out


def probe(a):
    fx = json.load(open(os.path.join(a.fake, 'fixtures.json')))
    res = {'phase': a.phase, 'env': env_report()}
    log('ENV', json.dumps(res['env'], default=str))
    steps = [('bundled', lambda: probe_bundled())]
    if a.synth:
        steps.append(('synthetic', lambda: probe_synth(a.synth)))
    steps.append(('plain', lambda: probe_plain(fx, a.fake)))
    for rd in a.readers.split(','):
        if rd == 'raptor':
            steps.append(('raptor', lambda: probe_raptor(fx, a.fake, a.fastread)))
        elif rd == 'fleet':
            steps.append(('fleet', lambda: probe_fleet(fx, a.fake, a.fleet)))
        elif rd == 'r7':
            steps.append(('r7', lambda: probe_r7(fx, a.fake, a)))
    for name, fn in steps:
        t = time.time()
        try:
            res[name] = fn()
        except Exception as e:
            res[name] = {'step_error': err_s(e), 'tb': traceback.format_exc()[-2000:]}
            log('STEP_FAIL', name, err_s(e))
        res.setdefault('seconds', {})[name] = round(time.time() - t, 1)
        json.dump(res, open(a.result, 'w'), indent=1, default=str)
    log('PROBE_DONE', a.phase)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['make', 'probe'])
    ap.add_argument('--comp'); ap.add_argument('--out'); ap.add_argument('--mods')
    ap.add_argument('--fake'); ap.add_argument('--phase'); ap.add_argument('--readers', default='')
    ap.add_argument('--result'); ap.add_argument('--fastread'); ap.add_argument('--fleet'); ap.add_argument('--synth')
    ap.add_argument('--r7src'); ap.add_argument('--code'); ap.add_argument('--anchor'); ap.add_argument('--side')
    ap.add_argument('--tmp', default='/tmp/jc')
    a = ap.parse_args()
    if a.mode == 'make':
        make(a)
    else:
        probe(a)
