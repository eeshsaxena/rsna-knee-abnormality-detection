# K12-MERGE cell (k12/cells/k12_merge_cell.py, 2026-10-09, Claude). Runs after FN-IDS and BEFORE the fleet cell.
# The K12 fleet = the FN11 fleet dataset (eeshsaxena/rsna-knee-cnxt-fleet-k11w: worker code, 20 members, parity npz,
# fleet_manifest.json) + the D-OAI weights dataset (eeshsaxena/rsna-knee-cnxt-fleet-k12w: ONLY the 5 D-OAI checkpoints +
# k12w_manifest.json). This cell links both into ONE fleet directory /tmp/k12_fleet (symlinks, nothing copied) whose
# fleet_manifest.json is the merged manifest the build wrote into K12M_CFG. Fail loud: the base manifest must equal the
# build's byte for byte (as parsed JSON), the k12w manifest must equal the build's, every D-OAI file's full sha256 must
# equal the k12w manifest, no name may collide. On ANY failure the directory is removed, so the strict fleet cell finds
# no fleet_worker.py ("fleet dataset not attached ... K12-MERGE"), the fleet stage is NOT clean, submission.csv is never
# written, and the N stage refuses (strict). Writes nothing under /kaggle/working.
import os, json, hashlib, shutil, time

_K12_OUT = '/tmp/k12_fleet'
_k12_rec = {'ok': False, 'out': _K12_OUT}


def _k12_log(msg):
    print(f'[k12-merge {time.strftime("%H:%M:%S")}] {msg}', flush=True)


def _k12_find(slug, need):
    for c in (f'/kaggle/input/datasets/eeshsaxena/{slug}', f'/kaggle/input/{slug}'):
        if os.path.isfile(os.path.join(c, need)):
            return c
    return None


def _k12_sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def _k12_merge():
    t0 = time.time()
    for d in (_K12_OUT, _K12_OUT + '.tmp'):
        if os.path.lexists(d):
            shutil.rmtree(d)
    base = _k12_find(K12M_CFG['base_ds'], 'fleet_worker.py')
    add = _k12_find(K12M_CFG['add_ds'], 'k12w_manifest.json')
    if not base:
        raise RuntimeError(f'base fleet dataset eeshsaxena/{K12M_CFG["base_ds"]} not attached (no fleet_worker.py)')
    if not add:
        raise RuntimeError(f'D-OAI weights dataset eeshsaxena/{K12M_CFG["add_ds"]} not attached (no k12w_manifest.json)')
    mb = json.load(open(os.path.join(base, 'fleet_manifest.json')))
    if mb != K12M_CFG['base_manifest']:
        raise RuntimeError(f'{base}/fleet_manifest.json != the build (another dataset version attached?)')
    ma = json.load(open(os.path.join(add, 'k12w_manifest.json')))
    if ma != K12M_CFG['add_manifest']:
        raise RuntimeError(f'{add}/k12w_manifest.json != the build (another dataset version attached?)')
    merged = K12M_CFG['merged_manifest']
    if merged['checkpoints'] != mb['checkpoints'] + ma['checkpoints']:
        raise RuntimeError('merged manifest checkpoints != base + add (build error)')
    base_files = set(os.listdir(base))
    extra = sorted(set(os.listdir(add)) - set(ma['checkpoints']) - {'k12w_manifest.json', 'dataset-metadata.json'})
    if extra:
        raise RuntimeError(f'unexpected files in the D-OAI weights dataset: {extra}')
    for n in ma['checkpoints']:
        if n in base_files:
            raise RuntimeError(f'name collision: {n} is in both datasets')
        s = _k12_sha(os.path.join(add, n))
        if s != ma['sources'][n]['sha256_full']:
            raise RuntimeError(f'{n}: sha256 {s[:16]} != k12w manifest {ma["sources"][n]["sha256_full"][:16]}')
        if merged['sources'][n]['sha256'] != s[:16]:
            raise RuntimeError(f'{n}: merged manifest sha256[:16] != the file')
    tmp = _K12_OUT + '.tmp'
    os.makedirs(tmp)
    for f in sorted(base_files):
        if f == 'fleet_manifest.json' or os.path.isdir(os.path.join(base, f)):
            continue
        os.symlink(os.path.join(base, f), os.path.join(tmp, f))
    for n in ma['checkpoints']:
        os.symlink(os.path.join(add, n), os.path.join(tmp, n))
    with open(os.path.join(tmp, 'fleet_manifest.json'), 'w') as fh:
        json.dump(merged, fh, indent=1)
    os.rename(tmp, _K12_OUT)
    _k12_rec.update(ok=True, base=base, add=add, n_base=len(mb['checkpoints']), n_add=len(ma['checkpoints']),
                    s=round(time.time() - t0, 1))
    _k12_log(f'RESULT OK merged fleet dir {_K12_OUT}: {len(mb["checkpoints"])} base members ({base}) + '
             f'{len(ma["checkpoints"])} D-OAI members ({add}), sha256 verified, {time.time() - t0:.1f} s')


try:
    _k12_merge()
except Exception as _e:                                                  # fail loud through the strict fleet cell
    shutil.rmtree(_K12_OUT, ignore_errors=True); shutil.rmtree(_K12_OUT + '.tmp', ignore_errors=True)
    _k12_rec['why'] = repr(_e)
    _k12_log(f'RESULT FAIL {_e!r}: no merged fleet dir; the strict fleet stage refuses (no submission.csv)')
