# rsna-raptor-d96-prep-s0: Kaggle CPU kernel (no GPU, no internet). Builds the Raptor "d96" corpus for the TRAIN
# studies: 96 x 384 x 384 uint8 per study, made by the VERBATIM reference reader of the scored Raptor r384 source
# (nartaa .949, = HR7 cell 2 /tmp/raptor_run.py: _make_reader, _pick_series_for_slot, build_study, d96 SLOTS, IMG 384,
# CROP_MM 140). One lossless PNG mosaic (96*384 x 384) per study, ~1 GB tar parts, receipt csv with sha256 of the raw
# vol bytes, manifest json with versions. The mask is NOT stored: build_study derives it from vol (sum > 0).
# Shard = sorted(StudyInstanceUID)[SHARD::NSHARD]. Pattern = eeshsaxena/rsna-fleet-prep-s0 (timeouts, retry pool, receipt).
import os, sys, io, json, time, glob, hashlib, tarfile, shutil, multiprocessing as mp
import numpy as np, pandas as pd

SHARD = 0          # literal per kernel copy
NSHARD = 3
READER_SHA = '30aa72fd972e29cf42c752aa5394a6c88a175f17b98740b1f6b0e0f55553ac40'
ROOT = next((c for c in ('/kaggle/input/competitions/rsna-knee-abnormality-detection',
                         '/kaggle/input/rsna-knee-abnormality-detection') if os.path.exists(c + '/train_series.csv')), None)
if ROOT is None:
    for d, _, f in os.walk('/kaggle/input'):
        if 'train_series.csv' in f and os.path.isdir(d + '/train_series'):
            ROOT = d; break
ROOT = os.environ.get('RAPTOR_ROOT', ROOT)              # box dry run only; unset on Kaggle
assert ROOT, 'competition root not found'
WORK = os.environ.get('RAPTOR_WORK', '/kaggle/working')
NPROC = int(os.environ.get('RAPTOR_NPROC', 4))
PART_BYTES = 1 << 30
TIMEOUT_S = 900
from concurrent.futures import ThreadPoolExecutor

# ======================= VERBATIM from the scored Raptor r384 source (sha256 READER_SHA) =======================
_READER_SRC = 'IMG = 384\nCROP_MM = 140.0\nSLOTS = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18),\n         ("Coronal", 0, 12), ("Axial", -1, 18)]\nMAXS = sum(s[2] for s in SLOTS)\nHDR_THREADS = 16\nPX_THREADS = 8\n\n\ndef _make_reader():\n    import pydicom, cv2\n    from pydicom.pixel_data_handlers.util import apply_modality_lut\n\n    def _hdr(f):\n        try:\n            h = pydicom.dcmread(f, stop_before_pixels=True)\n            iop = getattr(h, \'ImageOrientationPatient\', None)\n            ipp = getattr(h, \'ImagePositionPatient\', None)\n            if iop is not None and ipp is not None and len(iop) == 6:\n                r = np.array(iop[:3], float); c = np.array(iop[3:], float)\n                n = np.cross(r, c); pos = float(np.dot(np.array(ipp, float), n))\n            else:\n                pos = float(getattr(h, \'InstanceNumber\', 0) or 0)\n            ps = getattr(h, \'PixelSpacing\', None); ps = float(ps[0]) if ps is not None else 0.5\n            return (pos, f, ps, True)\n        except Exception:\n            return (0.0, f, 0.5, False)\n\n    def order_and_meta(sdir):\n        # Header reads are latency-bound on Kaggle\'s mounted competition data (~240k opens on the hidden\n        # set); a thread pool hides that latency. Records and sort are unchanged -> identical ordering.\n        fs = glob.glob(sdir + "/*.dcm")\n        with ThreadPoolExecutor(max_workers=HDR_THREADS) as ex:\n            out = list(ex.map(_hdr, fs))\n        recs = [(pos, f, ps) for pos, f, ps, _ in out]; ps_list = [ps for _, _, ps, ok in out if ok]\n        recs.sort(key=lambda x: x[0])\n        med_ps = float(np.median(ps_list)) if ps_list else 0.5\n        return [(f, ps) for _, f, ps in recs], med_ps\n\n    def read_px(f):\n        d = pydicom.dcmread(f)\n        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)\n        if str(getattr(d, \'PhotometricInterpretation\', \'\')) == \'MONOCHROME1\':\n            a = a.max() - a\n        return a\n\n    def mm_crop_resize(a, ps):\n        h, w = a.shape; cpx = int(round(CROP_MM / max(ps, 1e-3)))\n        cpx = min(cpx, min(h, w)); y0 = (h - cpx) // 2; x0 = (w - cpx) // 2\n        a = a[y0:y0 + cpx, x0:x0 + cpx]\n        return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)\n\n    return order_and_meta, read_px, mm_crop_resize\n\n\ndef _pick_series_for_slot(rows, plane, fluid, used):\n    cands = [r for r in rows if r[\'Anatomical_Plane\'] == plane and r[\'SeriesInstanceUID\'] not in used]\n    if fluid in (0, 1):\n        pref = [r for r in cands if int(r.get(\'Fluid_Sensitive\', 0) or 0) == fluid]\n        if pref:\n            return pref[0]\n    return cands[0] if cands else None\n\n\ndef build_study(sid, ser_records, tsdir, reader):\n    order_and_meta, read_px, mm_crop_resize = reader\n    rows = ser_records.get(sid, [])\n    vol = np.zeros((MAXS, IMG, IMG), np.uint8); idx = 0; used = set()\n    for plane, fluid, k in SLOTS:\n        r = _pick_series_for_slot(rows, plane, fluid, used)\n        if r is None:\n            idx += k; continue\n        used.add(r[\'SeriesInstanceUID\'])\n        files, med_ps = order_and_meta(f"{tsdir}/{sid}/{r[\'SeriesInstanceUID\']}")\n        if not files:\n            idx += k; continue\n        # wide span: the collateral ligaments and lateral meniscus live in the\n        # peripheral slices the old 0.15-0.85 crop threw away. Must match the corpus\n        # the weights were trained on (build_corpus80.py, SPAN).\n        n = len(files); lo, hi = int(n * 0.02), int(n * 0.98) - 1; hi = max(hi, lo)\n        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k\n        sel = [files[min(p, n - 1)] for p in picks]\n        def _px(fp_ps):\n            fp, ps = fp_ps\n            try: return read_px(fp), ps\n            except Exception: return None, med_ps\n        with ThreadPoolExecutor(max_workers=PX_THREADS) as ex:\n            got = list(ex.map(_px, sel))                    # order preserved\n        arrs = [a for a, _ in got]; pss = [ps for _, ps in got]\n        valid = [a for a in arrs if a is not None]\n        if valid:\n            allpx = np.concatenate([a.ravel() for a in valid])\n            loq, hiq = np.percentile(allpx, [2.0, 98.0])\n        else:\n            loq, hiq = 0.0, 1.0\n        for a, ps in zip(arrs, pss):\n            if idx >= MAXS: break\n            if a is None: idx += 1; continue\n            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)\n            aw = mm_crop_resize(aw, ps if ps > 0 else med_ps)\n            vol[idx] = (aw * 255).astype(np.uint8); idx += 1\n        if idx >= MAXS: break\n    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)\n    return vol, mask\n'
assert hashlib.sha256(_READER_SRC.encode()).hexdigest() == READER_SHA, 'reader text drift'
exec(compile(_READER_SRC, 'raptor_reader', 'exec'))
# ===============================================================================================================

ser = pd.read_csv(f'{ROOT}/train_series.csv')
ser['StudyInstanceUID'] = ser['StudyInstanceUID'].astype(str); ser['SeriesInstanceUID'] = ser['SeriesInstanceUID'].astype(str)
SER = {k: v.to_dict('records') for k, v in ser.groupby('StudyInstanceUID')}      # as main() of the scored source
sids = sorted(SER)[SHARD::NSHARD]
if os.environ.get('RAPTOR_ALL_PRESENT'):                                         # box dry run: every study with DICOMs
    sids = [s for s in sorted(SER) if os.path.isdir(f'{ROOT}/train_series/{s}')]
TSDIR = f'{ROOT}/train_series'
import cv2, pydicom
VERS = dict(numpy=np.__version__, cv2=cv2.__version__, pydicom=pydicom.__version__, pandas=pd.__version__,
            python=sys.version.split()[0], reader_sha=READER_SHA)
print(f'SHARD 0/{NSHARD} studies {len(sids)} of {len(SER)} | root {ROOT} | {VERS} | cpus {os.cpu_count()}', flush=True)
_RD = None


def _init():
    global _RD
    import cv2
    cv2.setNumThreads(1)
    _RD = _make_reader()


def one(sid):
    try:
        vol, mask = build_study(sid, SER, TSDIR, _RD)
        assert vol.shape == (MAXS, IMG, IMG) and vol.dtype == np.uint8
        ok, buf = cv2.imencode('.png', vol.reshape(MAXS * IMG, IMG), [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        if not ok:
            raise ValueError('png encode failed')
        back = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        assert back is not None and np.array_equal(back.reshape(vol.shape), vol), 'png round trip differs'
        return sid, int(mask.sum()), hashlib.sha256(vol.tobytes()).hexdigest(), '', buf.tobytes()
    except Exception as e:
        return sid, -1, '', f'{type(e).__name__}: {str(e)[:200]}', None


class Parts:
    def __init__(self):
        self.k, self.tf, self.size, self.n, self.files = -1, None, 0, 0, []
    def add(self, name, data):
        if self.tf is None or self.size > PART_BYTES:
            self.close(); self.k += 1; self.size = 0
            self.files.append(f'{WORK}/raptor_d96_s0_p{self.k:02d}.tar')
            self.tf = tarfile.open(self.files[-1], 'w')
        ti = tarfile.TarInfo(name); ti.size = len(data); ti.mtime = 0
        self.tf.addfile(ti, io.BytesIO(data)); self.size += len(data); self.n += 1
    def close(self):
        if self.tf is not None:
            self.tf.close(); self.tf = None


def write_receipt(res):
    pd.DataFrame(res, columns=['StudyInstanceUID', 'n_valid', 'vol_sha256', 'err']).to_csv(
        f'{WORK}/raptor_d96_receipt_0.csv', index=False)


def run(todo, parts, res, t0):
    got = set()
    pool = mp.get_context('fork').Pool(NPROC, initializer=_init)
    it = pool.imap_unordered(one, todo, chunksize=1)
    hung = False
    for _ in range(len(todo)):
        try:
            sid, nv, sha, err, data = it.next(timeout=TIMEOUT_S)
        except mp.TimeoutError:
            hung = True; print('!! no result for', TIMEOUT_S, 's -> terminating pool', flush=True); break
        got.add(sid)
        if data is not None:
            parts.add(f'{sid}.png', data)
        res.append((sid, nv, sha, err))
        if len(res) == 32:
            v = np.array([r[1] for r in res])
            assert (v > 0).mean() >= 0.9, f'first 32 studies: {(v <= 0).sum()} empty/failed: {[r[3][:80] for r in res if r[3]][:5]}'
        if len(res) % 100 == 0:
            write_receipt(res)
            print(len(res), len(sids), f'{(time.time()-t0)/60:.1f} min',
                  f'disk free {shutil.disk_usage(WORK).free/2**30:.1f} GB', flush=True)
    pool.terminate() if hung else pool.close()
    pool.join()
    return [s for s in todo if s not in got]


if __name__ == '__main__':
    t0 = time.time(); res = []; parts = Parts()
    lost = run(sids, parts, res, t0)
    if lost:
        print('retrying', len(lost), 'lost studies in a fresh pool', flush=True)
        lost = run(lost, parts, res, t0)
    res += [(s, -2, '', 'lost twice') for s in lost]
    parts.close(); write_receipt(res)
    r = pd.DataFrame(res, columns=['StudyInstanceUID', 'n_valid', 'vol_sha256', 'err'])
    status = 'OK' if ((r.n_valid > 0).mean() >= 0.995 and len(r) == len(sids)) else 'COVERAGE_SHORT'
    man = dict(status=status, shard=SHARD, nshard=NSHARD, n_studies=len(sids), n_png=parts.n,
               n_valid_mean=float(r.n_valid.clip(lower=0).mean()), versions=VERS,
               layout=dict(IMG=IMG, CROP_MM=CROP_MM, SLOTS=SLOTS, MAXS=MAXS, png='mosaic (MAXS*IMG, IMG) uint8, compression 3'),
               parts={os.path.basename(f): hashlib.sha256(open(f, 'rb').read()).hexdigest() for f in parts.files})
    json.dump(man, open(f'{WORK}/raptor_d96_manifest_0.json', 'w'), indent=1)
    print('done', f'{(time.time()-t0)/60:.1f} min | status', status, '| n_valid', r.n_valid.describe().round(1).to_dict(),
          '| png', parts.n, '| errors', r.err[r.err != ''].str[:60].value_counts().head(8).to_dict(), flush=True)
