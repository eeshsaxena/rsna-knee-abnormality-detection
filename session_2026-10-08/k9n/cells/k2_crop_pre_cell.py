# === K2 CROP STAGE (setup), combined mode only: runs AFTER the K2 fleet stage, BEFORE the verbatim K1 crop cell ===
# The K1 crop cell (crop_fleet_cell.py, unchanged bytes) writes submission.csv, submission_fleetsolo.csv and wXX files
# into CROP_FLEET_WORK and finds its manifest by walking /kaggle/input. Run unguarded after the fleet stage it would
# (a) overwrite the fleet stage's submission.csv / submission_fleetsolo.csv / submission_w20.csv and (b) abort on
# "expected exactly one fleet_manifest.json" (the ConvNeXt fleet dataset ships one too). So this cell:
#   C1 points CROP_FLEET_WORK at a sandbox (/kaggle/working/_k2_crop) FIRST, before anything that could fail;
#   C2 gives the sandbox the PARENT bytes (_k2_parent.csv from the fleet stage) as its submission.csv, so the crop
#      cell's gates (rho vs parent etc.) and its solo file are computed exactly as in K1;
#   C3 pins the crop manifest (the one fleet_manifest.json under /kaggle/input whose JSON has a 'members' list);
#   C4 sets the shared 8.6 h wall and the crop start floor through the cell's own CROP_FLEET_OVERRIDES hook;
#   C5 deletes stale CF-stage files, so a stage that does not finish leaves them absent (fail-loud).
# Any failure here leaves the sandbox without submission.csv, and the crop cell then writes nothing at all.
import os, json, time, shutil
os.environ['CROP_FLEET_WORK'] = '/kaggle/working/_k2_crop'                          # C1 (first statement)
K2C = {'C_W': 0.15, 'CF_WC': 0.15, 'CF_WF': 0.25, 'MIN_START_MIN': 10.0, 'WALL_H': 8.6}
K2C.update(globals().get('K2C_CFG') or {})
_K2C_WORK = '/kaggle/working'
_K2C_SAND = '/kaggle/working/_k2_crop'
_K2C_INPUT = '/kaggle/input'
_K2C_PAR = f'{_K2C_WORK}/_k2_parent.csv'
_cf_receipt = None                                                                   # never a stale crop receipt


def _k2c_log(msg):
    print(f'[k2-crop {time.strftime("%H:%M:%S")}] {msg}', flush=True)


def _k2c_loud_names():
    return ['submission_cropsolo.csv', f'submission_c{int(round(float(K2C["C_W"]) * 100)):02d}.csv', 'submission_cf.csv']


def _k2c_find_crop_manifest():
    """Every fleet_manifest.json under the input root (bounded walk, DICOM trees pruned); keep the crop format."""
    prune = {'test_series', 'train_series', 'test_images', 'train_images', '__pycache__', '.git'}
    hits, seen = [], set()
    base = _K2C_INPUT.rstrip('/').count('/')
    for dp, dns, fns in os.walk(_K2C_INPUT, followlinks=True):
        rp = os.path.realpath(dp)
        if rp in seen:
            dns[:] = []; continue
        seen.add(rp)
        dns[:] = sorted(d for d in dns if d not in prune and dp.rstrip('/').count('/') - base < 6)
        if 'fleet_manifest.json' in fns:
            p = os.path.join(dp, 'fleet_manifest.json')
            try:
                m = json.load(open(p))
            except Exception:
                continue
            if isinstance(m, dict) and isinstance(m.get('members'), list):
                hits.append(os.path.realpath(p))
    hits = sorted(set(hits))
    if len(hits) != 1:
        raise RuntimeError(f'expected exactly one crop manifest (with a members list) under {_K2C_INPUT}, found {hits}')
    return hits[0]


def _k2c_pre():
    rm = []
    for f in _k2c_loud_names():                                                    # C5
        for p in (f'{_K2C_WORK}/{f}', f'{_K2C_WORK}/{f}.tmp'):
            try:
                if os.path.lexists(p):
                    os.remove(p)
            except BaseException as e:
                rm.append(repr(e))
    if rm:
        raise RuntimeError(f'stale CF-stage files could not be removed: {rm}')
    shutil.rmtree(_K2C_SAND, ignore_errors=True)
    os.makedirs(_K2C_SAND, exist_ok=True)
    if not os.path.isfile(_K2C_PAR):
        raise RuntimeError('no _k2_parent.csv from the fleet stage (parent wrote no submission.csv?)')
    man = _k2c_find_crop_manifest()                                                 # C3
    over = {'MANIFEST': man, 'MIN_START_S': float(K2C['MIN_START_MIN']) * 60,      # C4
            'HARD_WALL_S': float(K2C['WALL_H']) * 3600}
    os.environ['CROP_FLEET_OVERRIDES'] = json.dumps(over)
    shutil.copyfile(_K2C_PAR, f'{_K2C_SAND}/submission.csv')                       # C2 (last: arms the crop cell)
    _k2c_log(f'STAGE crop start; sandbox {_K2C_SAND}; manifest {man}; overrides {over}; '
             f'fail-loud files {_k2c_loud_names()}')


_k2c_t0 = time.time()
try:
    _k2c_pre()
except BaseException as _e:
    try:
        os.remove(f'{_K2C_SAND}/submission.csv')
    except BaseException:
        pass
    _k2c_log(f'crop stage disarmed (the crop cell will write nothing): {type(_e).__name__}: {_e}')
