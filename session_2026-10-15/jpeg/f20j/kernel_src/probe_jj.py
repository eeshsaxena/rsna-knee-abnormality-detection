# probe_jj.py: worker of the private CPU kernel eeshsaxena/rsna-v224f20j-check (2026-10-14, Claude). Read-only on /kaggle/input.
# Derived from ../../kernel_src/probe_jc.py (rsna-jpeg-check v2): same fixture maker and the same stage readers, generalised so
# that each reader runs over a ROOT (fake competition root) and a list of study ids and returns a per-study output sha.
#   make : transcode ONE real public-test study into every fixture transfer syntax (fake competition root), as probe_jc v2
#   probe: run the stage readers over the fixture root and the "real" root (original uncompressed public-test + train studies)
#          in THIS process environment -> JSON. The parent compares phases (with / without the V224-F20-J DEC cell).
import os, sys, json, glob, time, io, hashlib, traceback, argparse, shutil, importlib.util, faulthandler, contextlib
import numpy as np
faulthandler.enable()

T0 = time.time()
UID_PREFIX = '1.2.826.0.1.3680043.8.498.'
VARIANTS = ['U', 'IMPL', 'JLL70', 'JLL57', 'J2KR', 'J2KI', 'RLE', 'MIX']
DECODE_TS = ['1.2.840.10008.1.2.1', '1.2.840.10008.1.2', '1.2.840.10008.1.2.5', '1.2.840.10008.1.2.4.50',
             '1.2.840.10008.1.2.4.57', '1.2.840.10008.1.2.4.70', '1.2.840.10008.1.2.4.80', '1.2.840.10008.1.2.4.90',
             '1.2.840.10008.1.2.4.91']


def log(*a):
    print('[probe %6.1fs]' % (time.time() - T0), *a, flush=True)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def err_s(e):
    return '%s: %s' % (type(e).__name__, str(e).replace('\n', ' ')[:300])


def env_report():
    r = {'python': sys.version.split()[0], 'PYTHONPATH': os.environ.get('PYTHONPATH', '')}
    for m in ['pydicom', 'numpy', 'PIL', 'pylibjpeg', 'libjpeg', 'openjpeg', 'gdcm', 'timm']:
        try:
            mod = __import__(m)
            r[m] = '%s @ %s' % (getattr(mod, '__version__', '?'), os.path.dirname(getattr(mod, '__file__', '') or '?'))
        except Exception as e:
            r[m] = 'NO (%s)' % type(e).__name__
    dec = {}
    try:
        from pydicom.pixels import get_decoder
        from pydicom.uid import UID
        for ts in DECODE_TS:
            try:
                dec[ts] = list(get_decoder(UID(ts)).available_plugins)
            except Exception as e:
                dec[ts] = err_s(e)
    except Exception as e:
        dec = err_s(e)
    r['pydicom_decoders'] = dec
    return r


# ----------------------------------------------------------------------------------------------------------------
# make (verbatim logic of probe_jc v2: new series UIDs keep the original order, MIX = series 1/3/5 JPEG Lossless .70)
# ----------------------------------------------------------------------------------------------------------------
def _load_jll(mods):
    spec = importlib.util.spec_from_file_location('jll', os.path.join(mods, 'jll.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _encode_one(job):
    src, dst, syn, nst, nse, var, mods = job
    import pydicom
    from pydicom.uid import (generate_uid, ImplicitVRLittleEndian, ExplicitVRLittleEndian, JPEGLosslessSV1,
                             JPEGLossless, JPEG2000Lossless, JPEG2000, RLELossless, UID)
    from pydicom.encaps import encapsulate
    rec = {'file': os.path.basename(dst), 'syntax': syn}
    try:
        ds = pydicom.dcmread(src)
        a = ds.pixel_array
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
        d2 = pydicom.dcmread(dst)
        rec['ts'] = str(d2.file_meta.TransferSyntaxUID)
        b = d2.pixel_array
        rec['exact'] = bool(b.shape == a.shape and np.array_equal(b, a))
        rec['maxdiff'] = float(np.abs(b.astype(np.float64) - a.astype(np.float64)).max()) if b.shape == a.shape else -1.0
    except Exception as e:
        rec['error'] = err_s(e)
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
    src_sid = cands[0][2]
    rows = tser[tser['StudyInstanceUID'] == src_sid].reset_index(drop=True)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, 'test_series'))
    jobs, meta, new_rows = [], {}, []
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
            series.append({'idx': int(i), 'uid': nse, 'syntax': syn, 'plane': str(r['Anatomical_Plane']), 'n_files': len(fs)})
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
    for var, rs in by.items():
        s = {'files': len(rs), 'errors': sum('error' in r for r in rs), 'exact': sum(bool(r.get('exact')) for r in rs),
             'maxdiff': max([r.get('maxdiff', -1.0) for r in rs] or [-1.0]), 'ts': sorted(set(r.get('ts', '?') for r in rs)),
             'first_error': next((r['error'] for r in rs if 'error' in r), None)}
        meta[var]['encode'] = s
        log('make', var, json.dumps(s))
    pd.DataFrame({'StudyInstanceUID': [meta[v]['study'] for v in VARIANTS]}).to_csv(os.path.join(out, 'test.csv'), index=False)
    pd.DataFrame(new_rows, columns=list(tser.columns)).to_csv(os.path.join(out, 'test_series.csv'), index=False)
    cols = list(pd.read_csv(os.path.join(comp, 'sample_submission.csv'), nrows=1).columns)
    df = pd.DataFrame({c: 0.5 for c in cols[1:]}, index=range(len(VARIANTS)))
    df.insert(0, cols[0], [meta[v]['study'] for v in VARIANTS])
    df.to_csv(os.path.join(out, 'sample_submission.csv'), index=False)
    json.dump({'source_study': src_sid, 'variants': meta}, open(os.path.join(out, 'fixtures.json'), 'w'), indent=1)
    log('MAKE_DONE', src_sid)


# ----------------------------------------------------------------------------------------------------------------
# readers: per-study output sha over a root
# ----------------------------------------------------------------------------------------------------------------
SLOTS_RAPTOR = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18), ("Coronal", 0, 12), ("Axial", -1, 18)]
IMG_RAPTOR = 384


def _ser(root):
    import pandas as pd
    tser = pd.read_csv(os.path.join(root, 'test_series.csv'))
    tser['StudyInstanceUID'] = tser['StudyInstanceUID'].astype(str)
    tser['SeriesInstanceUID'] = tser['SeriesInstanceUID'].astype(str)
    return tser, {k: v.to_dict('records') for k, v in tser.groupby('StudyInstanceUID')}


def _ids(root):
    import pandas as pd
    return pd.read_csv(os.path.join(root, 'test.csv'), dtype={'StudyInstanceUID': str})['StudyInstanceUID'].tolist()


def probe_plain(root):
    import pydicom
    out = {}
    for sid in _ids(root):
        files = sorted(glob.glob(os.path.join(root, 'test_series', sid, '*', '*.dcm')))
        h = hashlib.sha256(); ok = 0; errs = {}
        for f in files:
            try:
                a = pydicom.dcmread(f).pixel_array
                h.update(os.path.basename(f).encode() + str(a.dtype).encode() + str(a.shape).encode() + np.ascontiguousarray(a).tobytes())
                ok += 1
            except Exception as e:
                h.update(os.path.basename(f).encode() + b'ERR' + type(e).__name__.encode())
                errs[type(e).__name__] = errs.get(type(e).__name__, 0) + 1
        out[sid] = {'files': len(files), 'ok': ok, 'sha': h.hexdigest()[:16], 'errors': errs}
    log('plain', os.path.basename(root), json.dumps({k[-8:]: (v['ok'], v['files'], v['sha']) for k, v in out.items()}))
    return out


def probe_synth(fixdir):
    import pydicom
    out = {}
    for f in sorted(glob.glob(os.path.join(fixdir, 'corpus', 'dicom', '*.dcm'))):
        try:
            d = pydicom.dcmread(f)
            a = d.pixel_array
            out[os.path.basename(f)] = {'ts': str(d.file_meta.TransferSyntaxUID), 'dtype': str(a.dtype), 'shape': list(a.shape), 'sha': sha(a)}
        except Exception as e:
            out[os.path.basename(f)] = {'error': err_s(e)[:200]}
    log('synthetic fixtures', len(out), 'files', sum('sha' in v for v in out.values()), 'decoded')
    return out


def probe_raptor(root, fastread_path):
    spec = importlib.util.spec_from_file_location('_fastread', fastread_path)
    fr = importlib.util.module_from_spec(spec); spec.loader.exec_module(fr)
    build = fr.make_builder('study_lut', IMG_RAPTOR, SLOTS_RAPTOR, hdr_threads=32, px_threads=16)
    _, SER = _ser(root)
    tsdir = os.path.join(root, 'test_series')
    out = {}
    for sid in _ids(root):
        r = {}
        try:
            t0 = time.time()
            vol, mask = build(sid, SER, tsdir)
            r.update(seconds=round(time.time() - t0, 2), mask_sum=int(mask.sum()), slots=int(len(vol)), sha=sha(vol),
                     mask_sha=sha(mask))
        except Exception as e:
            r['error'] = err_s(e)
        out[sid] = r
    log('raptor', os.path.basename(root), json.dumps({k[-8:]: (v.get('sha'), v.get('mask_sum'), v.get('error')) for k, v in out.items()}))
    return out


def probe_fleet(root, fleet_dir):
    if fleet_dir not in sys.path:
        sys.path.insert(0, fleet_dir)
    import fleet_common as fc
    tser, _ = _ser(root)
    img_root = os.path.join(root, 'test_series')
    out = {}
    for sid in _ids(root):
        r = {}
        try:
            stats, errs = {}, []
            t0 = time.time()
            st = fc.build_study(tser[tser['StudyInstanceUID'] == sid], img_root, sid, stats=stats, errs=errs)
            r['seconds'] = round(time.time() - t0, 2)
            pres = ['%s%d' % (v[0][:3], v[1]) for v in fc.VIEWS if st[v] is not None]
            r['views'] = pres; r['views_present'] = len(pres); r['errs'] = errs[:4]
            h = hashlib.sha256()                  # per view: the stack the model sees + its contrast class (not the uid)
            for v in fc.VIEWS:
                if st[v] is not None:
                    h.update(repr(v).encode() + str(st[v][0].dtype).encode() + str(st[v][0].shape).encode())
                    h.update(np.ascontiguousarray(st[v][0]).tobytes() + repr(st[v][1]).encode())
                else:
                    h.update(repr(v).encode() + b'NONE')
            r['sha'] = h.hexdigest()[:16]
            r['series'] = [str(st[v][3]).split('.')[8] if st[v] is not None else None for v in fc.VIEWS]
        except Exception as e:
            r['error'] = err_s(e)
        out[sid] = r
    log('fleet', os.path.basename(root), json.dumps({k[-8:]: (v.get('sha'), v.get('views_present'), v.get('error')) for k, v in out.items()}))
    return out


def probe_r7(root, a, tag):
    for p in (a.code, a.r7src):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    import compact_infer as CI
    uids = _ids(root)
    tmp = os.path.join(a.tmp, 'r7tmp_' + tag); os.makedirs(tmp, exist_ok=True)
    ds = CI.CStudies(root, 'test', uids, tmp, a.code, 384, 1, 'v4', 3, a.anchor, 3.5, a.side, 2.0, 0.0, 20.0,
                     extent_meta=True, fluid=True)
    out = {}
    for i, uid in enumerate(uids):
        r = {}
        try:
            if ds.native is None:
                ds._setup()
            work = os.path.join(tmp, 'p_' + hashlib.sha256(uid.encode()).hexdigest()[:12])
            shutil.rmtree(work, ignore_errors=True)
            ds.PI.OneStudySource.current = uid
            with contextlib.redirect_stdout(io.StringIO()):
                ds.native.prepare(ds.data, work, split='test', study_ids=[uid], source_directory=True)
            key = hashlib.sha256(uid.encode()).hexdigest()[:24]
            study = json.load(open(os.path.join(work, 'studies', key + '.json'), encoding='utf-8'))
            r['native_series'] = len(study['series'])
            r['native_bad_slices'] = int(sum(len(s.get('errors') or []) for s in study['series']))
            r['native_errors'] = len(study['errors'])
            shutil.rmtree(work, ignore_errors=True)
        except Exception as e:
            r['native_error'] = err_s(e)
        try:
            t0 = time.time()
            item = ds[i]
            r['item_error'] = item['error']
            imgs = item['images'].numpy()
            r['windows'] = int(imgs.shape[0]); r['sha'] = sha(imgs) if imgs.size else None
            r['side'] = item.get('side'); r['anchor'] = item.get('anchor')
            r['seconds'] = round(time.time() - t0, 1)
        except Exception as e:
            r['item_exc'] = err_s(e)
        out[uid] = r
        log('r7', tag, uid[-8:], json.dumps({k: v for k, v in r.items()}, default=str)[:400])
    return out


def probe(a):
    res = {'phase': a.phase, 'env': env_report()}
    log('ENV', json.dumps(res['env'], default=str))
    steps = []
    roots = [(nm, rt) for nm, rt in (('fake', a.fake), ('real', a.real)) if nm in a.roots.split(',')]
    if 'plain' in a.readers:
        for nm, rt in roots:
            steps.append(('plain_' + nm, lambda rt=rt: probe_plain(rt)))
        if a.synth:
            steps.append(('synthetic', lambda: probe_synth(a.synth)))
    for rd in a.readers.split(','):
        for nm, rt in roots:
            if rd == 'raptor':
                steps.append(('raptor_' + nm, lambda rt=rt: probe_raptor(rt, a.fastread)))
            elif rd == 'fleet':
                steps.append(('fleet_' + nm, lambda rt=rt: probe_fleet(rt, a.fleet)))
            elif rd == 'r7':
                steps.append(('r7_' + nm, lambda rt=rt, nm=nm: probe_r7(rt, a, a.phase + '_' + nm)))
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
    ap.add_argument('--fake'); ap.add_argument('--real'); ap.add_argument('--phase'); ap.add_argument('--readers', default='')
    ap.add_argument('--result'); ap.add_argument('--fastread'); ap.add_argument('--fleet'); ap.add_argument('--synth')
    ap.add_argument('--r7src'); ap.add_argument('--code'); ap.add_argument('--anchor'); ap.add_argument('--side')
    ap.add_argument('--tmp', default='/tmp/jj')
    ap.add_argument('--roots', default='fake,real')
    a = ap.parse_args()
    if a.mode == 'make':
        make(a)
    else:
        probe(a)
