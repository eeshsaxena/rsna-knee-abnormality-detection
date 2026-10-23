# [step 3] extract the 12 prep tars to <scratch>/fleet_store (hash each tar while streaming it), then prove the store
# is the box's ~/fleet_store: same 4407 names, 24 sampled files byte-identical (sha256), every labelled study present.
# Scratch = the existing one of /tmp, /kaggle/tmp, /kaggle/temp with the most free space (never /kaggle/working:
# that is the 20 GB output area).
_free = {d: shutil.disk_usage(d).free / 1e9 for d in (P('/tmp'), P('/kaggle/tmp'), P('/kaggle/temp')) if os.path.isdir(d)}
_par = max(_free, key=_free.get)
STORE = f'{_par}/fleet_store'
_free_gb = _free[_par]
_need_gb = sum(os.path.getsize(p) for p in TARS.values()) / 1e9
log(f'disk free GB {({k: round(v, 1) for k, v in _free.items()})}: store -> {STORE} (needs ~{_need_gb:.1f} GB; '
    f'floor {CFG["MIN_FREE_GB"]} GB)')
if _free_gb < CFG['MIN_FREE_GB']:
    abort(f'only {_free_gb:.1f} GB free at {_par} (< {CFG["MIN_FREE_GB"]} GB; the store needs {_need_gb:.1f} GB)')
shutil.rmtree(STORE, ignore_errors=True)
os.makedirs(STORE)
_t = time.time()
with ThreadPoolExecutor(4) as _ex:
    _res = list(_ex.map(lambda n: extract_tar(n, TARS[n], STORE), sorted(TARS)))
_dt = time.time() - _t
_bad = [(n, h) for n, h, _, _ in _res if h != EXP['parts'][n]]
if _bad:
    abort(f'tar sha256 mismatch vs the prep manifests: {_bad}')
_names = sorted(f for f in os.listdir(STORE) if f.endswith('.npz'))
_left = [f for f in os.listdir(STORE) if f.endswith('.part')]
_nm_sha = hashlib.sha256('\n'.join(_names).encode()).hexdigest()
if _left or len(_names) != BOX['store_n'] or sum(r[2] for r in _res) != BOX['store_n'] or _nm_sha != BOX['store_names_sha']:
    abort(f'store names differ from the box: {len(_names)} npz (box {BOX["store_n"]}), members '
          f'{sum(r[2] for r in _res)}, names sha {_nm_sha[:16]} (box {BOX["store_names_sha"][:16]}), partial {_left[:3]}')
_smp_bad = [s['name'] for s in BOX['store_sample'] if sha256_file(f'{STORE}/{s["name"]}') != s['sha256']]
if _smp_bad:
    abort(f'store files differ from the box ~/fleet_store: {_smp_bad}')
import pandas as pd
_lab = pd.read_csv(f'{LBL}/{CFG["LABELS_TRAIN"]}', dtype={'StudyInstanceUID': str}).StudyInstanceUID
_miss = [s for s in _lab if not os.path.exists(f'{STORE}/{s}.npz')]
if len(_miss) != BOX['labelled_missing']:
    abort(f'{len(_miss)} labelled studies missing from the store (box: {BOX["labelled_missing"]}), e.g. {_miss[:3]}')
_gb = sum(r[3] for r in _res) / 1e9
STORE_INFO = dict(dir=STORE, n_npz=len(_names), names_sha=_nm_sha, extract_s=round(_dt, 1), tar_gb=round(_gb, 3),
                  mb_per_s=round(_gb * 1e3 / max(_dt, 1e-6), 1), sample_ok=len(BOX['store_sample']),
                  parts_ok=len(_res), labelled_missing=len(_miss), free_gb_before=round(_free_gb, 1),
                  free_gb_candidates={k: round(v, 1) for k, v in _free.items()})
REC['store'] = STORE_INFO
stage('store', ok=True)
log(f'store OK: {len(_names)} npz from {len(_res)} tars in {_dt:.0f} s ({STORE_INFO["mb_per_s"]} MB/s), '
    f'{len(BOX["store_sample"])}/{len(BOX["store_sample"])} sampled files == box, names sha == box')
