# [step 2] environment: versions, GPUs, CPU, RAM, disk. timm < 1.0 or missing -> install the bundled timm 1.0.29
# wheel (no internet needed), pip from PyPI only as a second fallback (internet is on). Logged either way.
import importlib


def _vt(v):
    return tuple(int(x) for x in re.findall(r'\d+', v)[:3])


try:
    import timm
    _tv = timm.__version__
except Exception:
    _tv = None
if _tv is None or _vt(_tv) < (1, 0):
    _rc = 1
    if WHEEL_OK:                                        # stored as .whl.bin so Kaggle never unpacks it
        os.makedirs(P('/tmp/kt_whl'), exist_ok=True)
        _whl = P('/tmp/kt_whl/') + EXP['timm_wheel'][:-len('.bin')]
        shutil.copyfile(f'{DS}/{EXP["timm_wheel"]}', _whl)
        _rc = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--quiet', _whl]).returncode
    if _rc != 0:
        _rc = subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'timm==1.0.29']).returncode
    warn(f'timm was {_tv}: pip install timm 1.0.29 rc {_rc}')
    importlib.invalidate_caches()
    for _m in [m for m in list(sys.modules) if m == 'timm' or m.startswith('timm.')]:
        del sys.modules[_m]
    import timm
import numpy as np, pandas as pd, torch, cv2, sklearn
VERS = dict(python=sys.version.split()[0], torch=torch.__version__, torch_cuda=torch.version.cuda, timm=timm.__version__,
            numpy=np.__version__, cv2=cv2.__version__, sklearn=sklearn.__version__, pandas=pd.__version__)
if VERS['timm'] != EXP['box_versions']['timm']:
    log(f'note: timm {VERS["timm"]} here vs {EXP["box_versions"]["timm"]} on the box (the state_dict check below decides)')
GPUS = gpu_query()
_mem = {l.split(':')[0]: l.split(':')[1].strip() for l in open('/proc/meminfo') if ':' in l}
_free = {p: round(shutil.disk_usage(p).free / 1e9, 1) for p in (P('/tmp'), WORK, P('/dev/shm')) if os.path.exists(p)}
if _free.get(P('/dev/shm'), 99) < 2:      # DataLoader batches travel through /dev/shm; the probe exercises train batches
    warn(f'/dev/shm has only {_free[P("/dev/shm")]} GB free: the eval loaders (not exercised by the probe) may fail')
ENV = dict(versions=VERS, box_versions=EXP['box_versions'], gpus=GPUS, cpu_count=os.cpu_count(),
           mem_total=_mem.get('MemTotal'), mem_available=_mem.get('MemAvailable'), disk_free_gb=_free)
REC['env'] = ENV
log('versions', json.dumps(VERS))
log('box     ', json.dumps(EXP['box_versions']))
log(f'GPUs {[(g["index"], g["name"], g["mem_used_mib"] + "/" + g["mem_total_mib"] + " MiB") for g in GPUS]} | '
    f'cpus {os.cpu_count()} | RAM {_mem.get("MemTotal")} (avail {_mem.get("MemAvailable")}) | disk free GB {_free}')
if not CFG['SKIP_GPU_CHECK']:
    if len(GPUS) < len(FOLDS):
        abort(f'{len(GPUS)} GPU(s) visible, need {len(FOLDS)} (kernel-metadata machine_shape NvidiaTeslaT4 = T4 x2)')
    if any('T4' not in g['name'] for g in GPUS):
        warn(f'GPU is not a T4: {[g["name"] for g in GPUS]} (timing calibration still applies)')
    if any(float(g['mem_used_mib']) > 1000 for g in GPUS):
        abort(f'GPU memory already in use before training: {GPUS}')
stage('env', ok=True)
