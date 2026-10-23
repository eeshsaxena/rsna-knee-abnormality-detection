# make_prep_kernels.py (laptop): writes 3 Kaggle CPU kernel folders prep/k{0,1,2}/ for the Raptor d96 train corpus.
# The DICOM reader is copied VERBATIM (ast source segments) from box/raptor_ref.py = the scored Raptor r384 source
# (HR7 cell 2); the sha256 of the copied text is asserted here and printed by the kernel.
import ast, hashlib, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, '..', 'box', 'raptor_ref.py')
s = open(REF, encoding='utf-8').read()
t = ast.parse(s)
want_fn = ['_make_reader', '_pick_series_for_slot', 'build_study']
want_as = ['IMG', 'CROP_MM', 'SLOTS', 'MAXS', 'HDR_THREADS', 'PX_THREADS']
seg = {}
for n in t.body:
    if isinstance(n, ast.FunctionDef) and n.name in want_fn:
        seg[n.name] = ast.get_source_segment(s, n)
    if isinstance(n, ast.Assign):
        for x in n.targets:
            if getattr(x, 'id', None) in want_as:
                seg[x.id] = ast.get_source_segment(s, n)
assert sorted(seg) == sorted(want_fn + want_as), sorted(seg)
READER = '\n'.join(seg[k] for k in want_as) + '\n\n\n' + '\n\n\n'.join(seg[k] for k in want_fn) + '\n'
READER_SHA = hashlib.sha256(READER.encode()).hexdigest()
print('reader sha256', READER_SHA, len(READER), 'chars')

BODY = r'''# rsna-raptor-d96-prep-s{SHARD}: Kaggle CPU kernel (no GPU, no internet). Builds the Raptor "d96" corpus for the TRAIN
# studies: 96 x 384 x 384 uint8 per study, made by the VERBATIM reference reader of the scored Raptor r384 source
# (nartaa .949, = HR7 cell 2 /tmp/raptor_run.py: _make_reader, _pick_series_for_slot, build_study, d96 SLOTS, IMG 384,
# CROP_MM 140). One lossless PNG mosaic (96*384 x 384) per study, ~1 GB tar parts, receipt csv with sha256 of the raw
# vol bytes, manifest json with versions. The mask is NOT stored: build_study derives it from vol (sum > 0).
# Shard = sorted(StudyInstanceUID)[SHARD::NSHARD]. Pattern = eeshsaxena/rsna-fleet-prep-s0 (timeouts, retry pool, receipt).
import os, sys, io, json, time, glob, hashlib, tarfile, shutil, multiprocessing as mp
import numpy as np, pandas as pd

SHARD = {SHARD}          # literal per kernel copy
NSHARD = 3
READER_SHA = '{READER_SHA}'
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
_READER_SRC = {READER_REPR}
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
print(f'SHARD {SHARD}/{NSHARD} studies {len(sids)} of {len(SER)} | root {ROOT} | {VERS} | cpus {os.cpu_count()}', flush=True)
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
            self.files.append(f'{WORK}/raptor_d96_s{SHARD}_p{self.k:02d}.tar')
            self.tf = tarfile.open(self.files[-1], 'w')
        ti = tarfile.TarInfo(name); ti.size = len(data); ti.mtime = 0
        self.tf.addfile(ti, io.BytesIO(data)); self.size += len(data); self.n += 1
    def close(self):
        if self.tf is not None:
            self.tf.close(); self.tf = None


def write_receipt(res):
    pd.DataFrame(res, columns=['StudyInstanceUID', 'n_valid', 'vol_sha256', 'err']).to_csv(
        f'{WORK}/raptor_d96_receipt_{SHARD}.csv', index=False)


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
    json.dump(man, open(f'{WORK}/raptor_d96_manifest_{SHARD}.json', 'w'), indent=1)
    print('done', f'{(time.time()-t0)/60:.1f} min | status', status, '| n_valid', r.n_valid.describe().round(1).to_dict(),
          '| png', parts.n, '| errors', r.err[r.err != ''].str[:60].value_counts().head(8).to_dict(), flush=True)
'''

for k in range(3):
    d = os.path.join(HERE, f'k{k}'); os.makedirs(d, exist_ok=True)
    code = (BODY.replace('{SHARD}', str(k)).replace('{READER_SHA}', READER_SHA)
            .replace('{READER_REPR}', repr(READER)))
    fn = f'rsna-raptor-d96-prep-s{k}.py'
    open(os.path.join(d, fn), 'w', encoding='utf-8', newline='\n').write(code)
    meta = {"id": f"eeshsaxena/rsna-raptor-d96-prep-s{k}", "title": f"rsna-raptor-d96-prep-s{k}", "code_file": fn,
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": [], "kernel_sources": [],
            "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": [],
            "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"}
    json.dump(meta, open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=1)
    print('wrote', d, hashlib.sha256(code.encode()).hexdigest()[:16])
