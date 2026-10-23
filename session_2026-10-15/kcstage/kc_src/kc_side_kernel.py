# rsna-kc-side v2 (kc_side_kernel.py): Kaggle CPU kernel (no GPU, no internet, no datasets), HEADER-ONLY DICOM reads
# (stop_before_pixels, specific_tags). Per TRAIN study (and the visible test studies), written for MAIN to push:
#  (1) the knee side, rule UNCHANGED from v1 (columns StudyInstanceUID..rule keep v1 names and meaning):
#        side series = the sagittal Fat_Suppression series with the most files, else any sagittal, else none;
#        mean_x = mean ImagePositionPatient[0] over it; |mean_x| > 30 mm -> R if < 0 else L ('mean_x');
#        else the study's Laterality / ImageLaterality tag if it names one side ('tag');
#        else the mean_x sign if |mean_x| > 5 mm ('mean_x_weak'); else U ('none').
#  (2) NEW (BOX_PLAN_CRITIQUE_1015): the series that ACTUALLY fills d96 slots 0-4, from the VERBATIM d96 reader
#      (_pick_series_for_slot on Fluid_Sensitive in train_series.csv order, reader text sha 30aa72fd asserted), and for
#      slots 0 SagFS, 1 SagPD, 2 CorFS, 4 Ax (the KC slots) the geometry sign of that series over ALL its files:
#        sagittal: x of the slice normal (row x col); expected < 0 (d96 sorts by IPP.normal -> index 0 = patient-left end)
#        coronal / axial: x of the row direction cosines (IOP[:3]); expected > 0 (image-right = patient-left)
#      flip_k = (side == 'R') XOR (sign_k unexpected): the per-slot canon flip kc_build_store / kc_worker apply.
#      weak = side not from the strong mean_x rule (|mean_x| <= 30 mm, tag, weak or U): served as the mean of both canons.
#      serve = 'both' if weak or any present KC slot has unexpected / undefined geometry, else 'single'.
# Outputs (RSNA metadata only, a few MB): /kaggle/working/kc_side_train.csv, kc_side_test.csv (visible test, if any),
# kc_side_summary.json. Box parser / validator: ~/kcrop_probe/v2/kc_side_validate.py.
# The kernel never raises on unexpected geometry: it records it, prints KC_SIDE_GEO and leaves the gate to the validator.
import os, sys, glob, json, time, hashlib
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
import pydicom

VERSION = 'kc_side v2 2026-10-15'
ROOT = next((c for c in ('/kaggle/input/competitions/rsna-knee-abnormality-detection',
                         '/kaggle/input/rsna-knee-abnormality-detection') if os.path.exists(c + '/train_series.csv')), None)
if ROOT is None and os.path.isdir('/kaggle/input'):
    for d, _, f in os.walk('/kaggle/input'):
        if 'train_series.csv' in f and os.path.isdir(d + '/train_series'):
            ROOT = d; break
ROOT = os.environ.get('KC_ROOT', ROOT)                     # box test only; unset on Kaggle
assert ROOT, 'competition root not found'
OUT = os.environ.get('KC_OUT', '/kaggle/working')
NPROC = int(os.environ.get('KC_PROCS', min(4, os.cpu_count() or 4)))
HDR_THREADS = int(os.environ.get('KC_HDR_THREADS', 16))
LIMIT = int(os.environ.get('KC_LIMIT', 0))                 # box test only: first N studies with DICOMs
T_MM, LAT_MM = 30.0, 5.0
TAGS = ['ImagePositionPatient', 'ImageOrientationPatient', 'Laterality', 'ImageLaterality', 'Manufacturer']
KC_SLOTS = (0, 1, 2, 4)                                    # SagFS, SagPD, CorFS, Ax (slot 3 CorPD is not a KC view)
SLOT_NAME = {0: 'SagFS', 1: 'SagPD', 2: 'CorFS', 3: 'CorPD', 4: 'Ax'}
EXPECT = {0: -1, 1: -1, 2: 1, 4: 1}                        # sag: normal x < 0; cor / ax: row x > 0 (MEASURED 458/458)
MAG_MIN = 0.5                                              # |component| below this: sign undefined (oblique)

# ======================= VERBATIM d96 reader text (prep_kernel_s0.py, sha256 READER_SHA) =======================
READER_SHA = '30aa72fd972e29cf42c752aa5394a6c88a175f17b98740b1f6b0e0f55553ac40'
_READER_SRC = 'IMG = 384\nCROP_MM = 140.0\nSLOTS = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18),\n         ("Coronal", 0, 12), ("Axial", -1, 18)]\nMAXS = sum(s[2] for s in SLOTS)\nHDR_THREADS = 16\nPX_THREADS = 8\n\n\ndef _make_reader():\n    import pydicom, cv2\n    from pydicom.pixel_data_handlers.util import apply_modality_lut\n\n    def _hdr(f):\n        try:\n            h = pydicom.dcmread(f, stop_before_pixels=True)\n            iop = getattr(h, \'ImageOrientationPatient\', None)\n            ipp = getattr(h, \'ImagePositionPatient\', None)\n            if iop is not None and ipp is not None and len(iop) == 6:\n                r = np.array(iop[:3], float); c = np.array(iop[3:], float)\n                n = np.cross(r, c); pos = float(np.dot(np.array(ipp, float), n))\n            else:\n                pos = float(getattr(h, \'InstanceNumber\', 0) or 0)\n            ps = getattr(h, \'PixelSpacing\', None); ps = float(ps[0]) if ps is not None else 0.5\n            return (pos, f, ps, True)\n        except Exception:\n            return (0.0, f, 0.5, False)\n\n    def order_and_meta(sdir):\n        # Header reads are latency-bound on Kaggle\'s mounted competition data (~240k opens on the hidden\n        # set); a thread pool hides that latency. Records and sort are unchanged -> identical ordering.\n        fs = glob.glob(sdir + "/*.dcm")\n        with ThreadPoolExecutor(max_workers=HDR_THREADS) as ex:\n            out = list(ex.map(_hdr, fs))\n        recs = [(pos, f, ps) for pos, f, ps, _ in out]; ps_list = [ps for _, _, ps, ok in out if ok]\n        recs.sort(key=lambda x: x[0])\n        med_ps = float(np.median(ps_list)) if ps_list else 0.5\n        return [(f, ps) for _, f, ps in recs], med_ps\n\n    def read_px(f):\n        d = pydicom.dcmread(f)\n        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)\n        if str(getattr(d, \'PhotometricInterpretation\', \'\')) == \'MONOCHROME1\':\n            a = a.max() - a\n        return a\n\n    def mm_crop_resize(a, ps):\n        h, w = a.shape; cpx = int(round(CROP_MM / max(ps, 1e-3)))\n        cpx = min(cpx, min(h, w)); y0 = (h - cpx) // 2; x0 = (w - cpx) // 2\n        a = a[y0:y0 + cpx, x0:x0 + cpx]\n        return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)\n\n    return order_and_meta, read_px, mm_crop_resize\n\n\ndef _pick_series_for_slot(rows, plane, fluid, used):\n    cands = [r for r in rows if r[\'Anatomical_Plane\'] == plane and r[\'SeriesInstanceUID\'] not in used]\n    if fluid in (0, 1):\n        pref = [r for r in cands if int(r.get(\'Fluid_Sensitive\', 0) or 0) == fluid]\n        if pref:\n            return pref[0]\n    return cands[0] if cands else None\n\n\ndef build_study(sid, ser_records, tsdir, reader):\n    order_and_meta, read_px, mm_crop_resize = reader\n    rows = ser_records.get(sid, [])\n    vol = np.zeros((MAXS, IMG, IMG), np.uint8); idx = 0; used = set()\n    for plane, fluid, k in SLOTS:\n        r = _pick_series_for_slot(rows, plane, fluid, used)\n        if r is None:\n            idx += k; continue\n        used.add(r[\'SeriesInstanceUID\'])\n        files, med_ps = order_and_meta(f"{tsdir}/{sid}/{r[\'SeriesInstanceUID\']}")\n        if not files:\n            idx += k; continue\n        # wide span: the collateral ligaments and lateral meniscus live in the\n        # peripheral slices the old 0.15-0.85 crop threw away. Must match the corpus\n        # the weights were trained on (build_corpus80.py, SPAN).\n        n = len(files); lo, hi = int(n * 0.02), int(n * 0.98) - 1; hi = max(hi, lo)\n        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k\n        sel = [files[min(p, n - 1)] for p in picks]\n        def _px(fp_ps):\n            fp, ps = fp_ps\n            try: return read_px(fp), ps\n            except Exception: return None, med_ps\n        with ThreadPoolExecutor(max_workers=PX_THREADS) as ex:\n            got = list(ex.map(_px, sel))                    # order preserved\n        arrs = [a for a, _ in got]; pss = [ps for _, ps in got]\n        valid = [a for a in arrs if a is not None]\n        if valid:\n            allpx = np.concatenate([a.ravel() for a in valid])\n            loq, hiq = np.percentile(allpx, [2.0, 98.0])\n        else:\n            loq, hiq = 0.0, 1.0\n        for a, ps in zip(arrs, pss):\n            if idx >= MAXS: break\n            if a is None: idx += 1; continue\n            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)\n            aw = mm_crop_resize(aw, ps if ps > 0 else med_ps)\n            vol[idx] = (aw * 255).astype(np.uint8); idx += 1\n        if idx >= MAXS: break\n    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)\n    return vol, mask\n'
assert hashlib.sha256(_READER_SRC.encode()).hexdigest() == READER_SHA, 'reader text drift'
_RNS = dict(np=np, glob=glob, ThreadPoolExecutor=ThreadPoolExecutor)
exec(compile(_READER_SRC, 'raptor_reader', 'exec'), _RNS)
_pick_series_for_slot, SLOTS = _RNS['_pick_series_for_slot'], _RNS['SLOTS']
assert [s[:2] for s in SLOTS] == [('Sagittal', 1), ('Sagittal', 0), ('Coronal', 1), ('Coronal', 0), ('Axial', -1)]
# ===============================================================================================================


def slot_series(rows):
    """Exactly build_study's slot loop: a picked series is marked used even if its folder turns out empty."""
    used, out = set(), []
    for plane, fluid, k in SLOTS:
        r = _pick_series_for_slot(rows, plane, fluid, used)
        if r is not None:
            used.add(r['SeriesInstanceUID'])
        out.append(r)
    return out


def hdr(path):
    try:
        h = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=TAGS)
    except Exception:
        return None
    ipp = getattr(h, 'ImagePositionPatient', None)
    iop = getattr(h, 'ImageOrientationPatient', None)
    tag = {str(getattr(h, t, '') or '').strip().upper()[:1] for t in ('Laterality', 'ImageLaterality')} - {''}
    nx = rx = None; dom = ''
    if iop is not None and len(iop) == 6:
        r = np.array(iop[:3], float); c = np.array(iop[3:], float); n = np.cross(r, c)
        nx, rx, dom = float(n[0]), float(r[0]), 'xyz'[int(np.argmax(np.abs(n)))]
    x = float(ipp[0]) if ipp is not None and len(ipp) == 3 else None
    return (x, nx, rx, dom, tag, str(getattr(h, 'Manufacturer', '') or '').strip())


def read_series(sdir, cache, ex):
    if sdir not in cache:
        files = sorted(glob.glob(sdir + '/*.dcm'))
        cache[sdir] = (files, [h for h in ex.map(hdr, files) if h is not None])
    return cache[sdir]


def geo(files, hs, k):
    """Sign of the slot-k geometry component over all parsed headers of the slot series."""
    comp = [h[1] if k in (0, 1) else h[2] for h in hs]
    comp = [c for c in comp if c is not None]
    d = dict(n=len(files), nh=len(hs), sign=0, frac=np.nan, mag=np.nan, dom='', mx=np.nan)
    xs = [h[0] for h in hs if h[0] is not None]
    if xs:
        d['mx'] = float(np.mean(xs))
    doms = [h[3] for h in hs if h[3]]
    if doms:
        d['dom'] = max(sorted(set(doms)), key=doms.count)
    if comp:
        a = np.array(comp)
        med = float(np.median(a))
        d['mag'] = float(np.median(np.abs(a)))
        if d['mag'] >= MAG_MIN:
            d['sign'] = 1 if med > 0 else -1
            d['frac'] = float(np.mean(np.sign(a) == d['sign']))
    return d


def study(args):
    sid, rows, sdir_root = args
    out = dict(StudyInstanceUID=sid, mean_x=np.nan, n=0, sag_nx=np.nan, tag='', side='U', rule='none')
    picks = slot_series(rows)
    cache = {}
    with ThreadPoolExecutor(HDR_THREADS) as ex:
        # (1) side: v1 rule, unchanged
        sag = [r for r in rows if r['Anatomical_Plane'] == 'Sagittal']
        fs = [r for r in sag if str(r.get('Fat_Suppression')) in ('1', '1.0', 'True')]
        best = None
        for cand in (fs, sag):
            if cand:
                best = max(cand, key=lambda r: len(glob.glob(f"{sdir_root}/{sid}/{r['SeriesInstanceUID']}/*.dcm")))
                break
        tags_all, mans = set(), []
        if best is not None:
            files, hs = read_series(f"{sdir_root}/{sid}/{best['SeriesInstanceUID']}", cache, ex)
            xs = [h[0] for h in hs if h[0] is not None]
            nxs = [h[1] for h in hs if h[1] is not None]
            tags = set().union(*[h[4] for h in hs]) if hs else set()
            out.update(n=len(files), mean_x=float(np.mean(xs)) if xs else np.nan,
                       sag_nx=float(np.mean(nxs)) if nxs else np.nan, tag=''.join(sorted(tags)))
            mx = out['mean_x']
            if np.isfinite(mx) and abs(mx) > T_MM:
                out.update(side='R' if mx < 0 else 'L', rule='mean_x')
            elif tags in ({'R'}, {'L'}):
                out.update(side=next(iter(tags)), rule='tag')
            elif np.isfinite(mx) and abs(mx) > LAT_MM:
                out.update(side='R' if mx < 0 else 'L', rule='mean_x_weak')
            out['side_uid'] = best['SeriesInstanceUID']
        else:
            out['side_uid'] = ''
        # (2) the series in each d96 slot + geometry of the KC slots
        geo_ok, is_r = True, out['side'] == 'R'
        for k, r in enumerate(picks):
            out[f's{k}_uid'] = r['SeriesInstanceUID'] if r is not None else ''
            if k not in KC_SLOTS:
                continue
            if r is None:
                d = dict(n=0, nh=0, sign=0, frac=np.nan, mag=np.nan, dom='', mx=np.nan)
            else:
                files, hs = read_series(f"{sdir_root}/{sid}/{r['SeriesInstanceUID']}", cache, ex)
                d = geo(files, hs, k)
                tags_all |= set().union(*[h[4] for h in hs]) if hs else set()
                mans += [h[5] for h in hs if h[5]]
            present = d['n'] > 0
            ok = (not present) or (d['sign'] == EXPECT[k] and d['frac'] == 1.0)
            geo_ok &= ok
            unexpected = d['sign'] == -EXPECT[k]
            out.update({f's{k}_{c}': d[c] for c in ('n', 'nh', 'sign', 'frac', 'mag', 'dom', 'mx')})
            out[f's{k}_ok'] = int(ok)
            out[f's{k}_flip'] = int(is_r ^ unexpected)
    out['tag_all'] = ''.join(sorted(tags_all))
    out['man'] = max(sorted(set(mans)), key=mans.count) if mans else ''
    out['weak'] = int(out['rule'] != 'mean_x')
    out['geo_ok'] = int(geo_ok)
    out['serve'] = 'single' if (not out['weak'] and geo_ok) else 'both'
    out['nread'] = sum(len(v[0]) for v in cache.values())
    return out


def run(csv_name, sdir_name, out_name):
    ser = pd.read_csv(f'{ROOT}/{csv_name}')                # default dtypes, as the d96 corpus builder / scored main()
    ser['StudyInstanceUID'] = ser['StudyInstanceUID'].astype(str)
    ser['SeriesInstanceUID'] = ser['SeriesInstanceUID'].astype(str)
    SER = {k: v.to_dict('records') for k, v in ser.groupby('StudyInstanceUID')}
    sdir_root = f'{ROOT}/{sdir_name}'
    sids = sorted(SER)
    if LIMIT:
        sids = [s for s in sids if os.path.isdir(f'{sdir_root}/{s}')][:LIMIT] + \
               [s for s in sids if not os.path.isdir(f'{sdir_root}/{s}')]
    jobs = [(s, SER[s], sdir_root) for s in sids]
    t0, res = time.time(), []
    with mp.get_context('fork').Pool(NPROC) as pool:
        for i, r in enumerate(pool.imap_unordered(study, jobs, chunksize=4)):
            res.append(r)
            if (i + 1) % 250 == 0 or i + 1 == len(jobs):
                el = time.time() - t0
                print(f'  [{csv_name}] {i + 1}/{len(jobs)} studies | files read {sum(x["nread"] for x in res)} | '
                      f'{el:.0f} s | eta {el / (i + 1) * (len(jobs) - i - 1):.0f} s', flush=True)
    df = pd.DataFrame(res).sort_values('StudyInstanceUID').reset_index(drop=True)
    first = ['StudyInstanceUID', 'mean_x', 'n', 'sag_nx', 'tag', 'side', 'rule']
    df = df[first + [c for c in df.columns if c not in first]]
    df.to_csv(f'{OUT}/{out_name}', index=False)
    return df, time.time() - t0


def summarise(df, secs):
    has = df.n > 0
    s = dict(version=VERSION, root=ROOT, nproc=NPROC, hdr_threads=HDR_THREADS, seconds=round(secs, 1),
             studies=int(len(df)), studies_with_side_series=int(has.sum()), files_read=int(df.nread.sum()),
             sides=dict(df.side.value_counts()), rules=dict(df.rule.value_counts()),
             weak=int(df.weak.sum()), serve=dict(df.serve.value_counts()), geo_ok=int(df.geo_ok.sum()),
             versions=dict(pydicom=pydicom.__version__, numpy=np.__version__, pandas=pd.__version__,
                           python=sys.version.split()[0]))
    slots = {}
    for k in KC_SLOTS:
        p = df[f's{k}_n'] > 0
        bad = df[p & (df[f's{k}_ok'] == 0)]
        slots[SLOT_NAME[k]] = dict(present=int(p.sum()), expected_sign=int((p & (df[f's{k}_sign'] == EXPECT[k])).sum()),
                                   opposite_sign=int((p & (df[f's{k}_sign'] == -EXPECT[k])).sum()),
                                   undefined_sign=int((p & (df[f's{k}_sign'] == 0)).sum()),
                                   mixed_within_series=int((p & (df[f's{k}_frac'] < 1)).sum()),
                                   not_ok=int(len(bad)), not_ok_sids=bad.StudyInstanceUID.head(20).tolist(),
                                   dom_axis=dict(df.loc[p, f's{k}_dom'].value_counts()))
    s['slots'] = slots
    s['KC_SIDE_GEO'] = 'ALL_EXPECTED' if all(v['not_ok'] == 0 for v in slots.values()) else 'UNEXPECTED'
    return json.loads(json.dumps(s, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))


if __name__ == '__main__':
    print(VERSION, '| root', ROOT, '| procs', NPROC, '| hdr threads', HDR_THREADS, '| pydicom', pydicom.__version__, flush=True)
    os.makedirs(OUT, exist_ok=True)
    summ = {}
    df, secs = run('train_series.csv', 'train_series', 'kc_side_train.csv')
    summ['train'] = summarise(df, secs)
    for tdir in ('test_series', 'test_images'):
        if os.path.exists(f'{ROOT}/test_series.csv') and os.path.isdir(f'{ROOT}/{tdir}'):
            tdf, tsecs = run('test_series.csv', tdir, 'kc_side_test.csv')
            summ['test'] = summarise(tdf, tsecs)
            break
    json.dump(summ, open(f'{OUT}/kc_side_summary.json', 'w'), indent=1)
    for part, s in summ.items():
        print(f'KC_SIDE {part}: studies {s["studies"]} | with side series {s["studies_with_side_series"]} | sides {s["sides"]}'
              f' | rules {s["rules"]} | weak {s["weak"]} | serve {s["serve"]} | files {s["files_read"]} | {s["seconds"]} s')
        for name, v in s['slots'].items():
            print(f'  {name}: present {v["present"]} | expected sign {v["expected_sign"]} | opposite {v["opposite_sign"]}'
                  f' | undefined {v["undefined_sign"]} | mixed {v["mixed_within_series"]} | dom {v["dom_axis"]}')
        print(f'KC_SIDE_GEO {part} {s["KC_SIDE_GEO"]}', flush=True)
