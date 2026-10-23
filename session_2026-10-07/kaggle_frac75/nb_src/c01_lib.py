# [lib] rsna-train-ttr-frac75-f01: helpers only (no GPU, no network, no side effects beyond defining things).
# Trains the frac75 Fracture-target variant of TTR224 on folds 0 and 1 (seed 1043, paired with J1 = box
# ~/fleet_runs/f0_s1043_ttr_mix50), one fold per T4, in parallel. 2026-10-07 copy of rsna-train-ttr-s1044-f01: the box
# recipe with ONLY --labels changed (labels_frac75_logit.csv from dataset eeshsaxena/rsna-knee-frac75-labels, sha256
# baked below); the session is sized to END by ~9.3 h (budget 8.4 h, hard wall 9.2 h) for a ~10 h GPU quota.
# KT_* environment variables exist ONLY for the box dry test; a pushed Kaggle kernel receives none of them.
import os, sys, re, ast, json, math, time, glob, shutil, signal, hashlib, tarfile, datetime, subprocess
from concurrent.futures import ThreadPoolExecutor

T_NB0 = time.time()
KT_ROOT = os.environ.get('KT_ROOT', '')            # '' on Kaggle; the dry test maps /kaggle/... and /tmp/... under it


def P(p):
    return KT_ROOT + p


INPUT, WORK = P('/kaggle/input'), P('/kaggle/working')
STORE, CODE, HF_HOME, PROBE_DIR = P('/tmp/fleet_store'), P('/tmp/kt_code'), P('/tmp/kt_hf'), P('/tmp/kt_probe')
LBL = P('/tmp/kt_labels')                          # verified copies of the 5 label/fold csvs (+ box_facts.json)


def _env(k, d, f=float):
    return f(os.environ.get(k, d))


CFG = dict(
    FOLDS=[0, 1],
    GPU_IDS=os.environ.get('KT_GPU_IDS', '0,1').split(','),   # fold 0 -> GPU0, fold 1 -> GPU1
    SEED=1043, EPOCHS_MAX=24, EPOCHS_WARN=16,
    # Session budget for a ~10 h weekly GPU quota: the WHOLE session must end by ~9.3 h. Measured on Kaggle T4 x2
    # (receipts f34 10-05, s1044 f01/f23 10-06): setup + store + checks + probe end at 0.139-0.144 h, 0.78-0.81 s/it,
    # 27.4-28.3 min/epoch incl. both evals; the slowest run (f34) overran its epoch estimate by +5.6 %; finalize < 1 min.
    BUDGET_H=_env('KT_BUDGET_H', 8.4),      # training is planned to END by this many hours after kernel start
    WALL_H=_env('KT_WALL_H', 9.2),          # hard wall: trainers terminated (SIGTERM, then SIGKILL), outputs kept;
                                            # + quiet-wait <= 2 min + kill grace <= 1 min + finalize -> session end < 9.3 h
    SAVE_LAST=_env('KT_SAVE_LAST', 3, int), # epochs/ ckpts + OOFs for the LAST 3 epochs (gate: last-3 schedule-end read)
    LABELS_TRAIN='labels_frac75_logit.csv',  # the ONE recipe change vs J1 (dataset eeshsaxena/rsna-knee-frac75-labels)
    LABELS_TRAIN_SHA256='5036425cd0d64f27f08acfe6865d668ecf830e11562ed8921c63afa782c2c8c0',   # == box ~/novel/frac75/
    LABELS_TRAIN_MD5='ad7582b817243f8f22e8bcfdd7e3f84a',                                       # (PREREG.md md5)
    EVAL_FRAC=_env('KT_EVAL_FRAC', 0.21),   # eval time / train time per epoch; box J1 log: TRAIN-ONLY 10.1-10.4 min,
                                            # epoch incl. raw+ema eval (val 867 + gold74) 12.3-12.6 min -> 0.21
    STARTUP_S=_env('KT_STARTUP_S', 120),    # trainer start: labels, contract probe of 8 npz, model build, workers
    WORKERS=_env('KT_WORKERS', 3, int),     # DataLoader workers PER trainer (Kaggle T4x2 = 4 vCPU for both)
    PROBE_ITERS=_env('KT_PROBE_ITERS', 400, int),      # trainer prints s/it every 200 it; 400 = window 200 -> 400
    PROBE_TIMEOUT_S=_env('KT_PROBE_TIMEOUT_S', 1500),
    PROBE_POLL_S=_env('KT_PROBE_POLL_S', 5),
    MIN_FREE_GB=_env('KT_MIN_FREE_GB', 20),
    KILL_GRACE_S=_env('KT_KILL_GRACE_S', 60),
    POLL_S=_env('KT_POLL_S', 30),
    STATUS_EVERY_S=_env('KT_STATUS_EVERY_S', 900),
    OUT_LIMIT_GB=_env('KT_OUT_LIMIT_GB', 19),
    SKIP_GPU_CHECK=os.environ.get('KT_SKIP_GPU_CHECK', '') == '1',
    TRAINER=os.environ.get('KT_TRAINER_PY', ''),        # '' = the real fleet_train.py (dry test: a dummy)
)
FOLDS = CFG['FOLDS']
RUN = {F: f'f{F}_s{CFG["SEED"]}_ttr_frac75' for F in FOLDS}      # == the box names ~/fleet_runs/f{F}_s1043_ttr_frac75
CODE_FILES = ['fleet_train.py', 'fleet_common.py', 'fleet_model.py', 'fleet_model_ttr.py', 'fleet_model_f2.py',
              'fleet_aug.py']
LABEL_FILES = ['labels_mix50_logit.csv', 'folds5.csv', 'gold74_eval_set.csv', 'llm_labels_v4_blend.csv']  # code dataset
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
           'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
TAR_RE = re.compile(r'fleet_store_s[01]_p\d\d\.tar')
NPZ_RE = re.compile(r'[0-9][0-9.]*\.npz')


def _proc_start(pid):
    up = float(open('/proc/uptime').read().split()[0])
    st = int(open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()[19]) / os.sysconf('SC_CLK_TCK')
    return time.time() - max(0.0, up - st)


def kernel_t0():
    """Kaggle's 12 h clock starts with the session: the Jupyter kernel process start (the K2 cell's proven method),
    or the container's pid 1 when that started up to 30 min earlier (more conservative), never later than now."""
    if os.environ.get('KT_T0'):
        return float(os.environ['KT_T0']), 'env KT_T0'
    try:
        t_self = min(_proc_start('self'), T_NB0)
    except Exception:
        return T_NB0, 'notebook start'
    try:
        t1 = _proc_start(1)
        if 0 <= t_self - t1 <= 1800:
            return t1, 'container pid 1 start'
    except Exception:
        pass
    return t_self, 'kernel process start'


T0, T0_SRC = kernel_t0()
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))


def ist(t=None):
    return datetime.datetime.fromtimestamp(time.time() if t is None else t, IST).strftime('%m-%d %H:%M IST')


def el():
    return time.time() - T0


def log(*a):
    print(f'[kt {el() / 3600:5.2f}h {ist()}]', *a, flush=True)


REC = dict(kernel='rsna-train-ttr-frac75-f01', status='starting', t0=T0, t0_src=T0_SRC, t0_ist=ist(T0),
           cfg=CFG, runs=RUN, warnings=[], stages={})


def save_receipt():
    os.makedirs(WORK, exist_ok=True)
    REC['updated_ist'] = ist()
    REC['elapsed_h'] = round(el() / 3600, 4)
    tmp = f'{WORK}/run_receipt.json.tmp'
    with open(tmp, 'w') as f:
        json.dump(REC, f, indent=1, default=str)
    os.replace(tmp, f'{WORK}/run_receipt.json')


def warn(m):
    REC['warnings'].append(f'{ist()} {m}')
    log('WARNING', m)
    save_receipt()


def stage(name, **kw):
    REC['stages'][name] = dict(kw, at_h=round(el() / 3600, 4))
    save_receipt()


def abort(m):
    REC['status'] = f'ABORTED: {m}'
    log('ABORT', m)
    save_receipt()
    raise RuntimeError(m)


def sha256_file(p, bs=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(bs), b''):
            h.update(b)
    return h.hexdigest()


def read_text(p):
    try:
        with open(p, errors='replace') as f:
            return f.read()
    except FileNotFoundError:
        return ''


def tail(p, n=25):
    return '\n'.join(read_text(p).splitlines()[-n:])


# ---------------------------------------------------------------- inputs
def walk_find(root, pred, maxdepth=6):
    out, base = [], os.path.abspath(root).rstrip(os.sep).count(os.sep)
    for d, dirs, files in os.walk(root, followlinks=True):
        if os.path.abspath(d).count(os.sep) - base >= maxdepth:
            dirs[:] = []
        for f in files:
            if pred(d, f):
                out.append(os.path.join(d, f))
    return sorted(out)


def locate_inputs():
    """-> (dataset dir, {tar name: path}, {shard: manifest path}). Handles /kaggle/input/<slug>/ and
    /kaggle/input/{datasets,notebooks}/<owner>/<slug>/ (and anything else up to depth 6)."""
    ds = sorted({os.path.realpath(os.path.dirname(p)) for p in
                 walk_find(INPUT, lambda d, f: f == 'kt_manifest.json' and os.path.exists(f'{d}/fleet_train.py'))})
    if len(ds) > 1 and len({open(f'{d}/kt_manifest.json', 'rb').read() for d in ds}) == 1:
        warn(f'code dataset visible at {len(ds)} paths with identical kt_manifest.json (legacy + new mount layout): '
             f'using {ds[0]}')       # every file is still sha256-checked against that manifest below
        ds = ds[:1]
    if len(ds) != 1:
        abort(f'code dataset eeshsaxena/rsna-knee-fleet-train: found {ds} under {INPUT} (want exactly 1)')
    tars, mans = {}, {}
    for p in walk_find(INPUT, lambda d, f: bool(TAR_RE.fullmatch(f))):
        n = os.path.basename(p)
        if n in tars and os.path.realpath(tars[n]) != os.path.realpath(p):
            warn(f'tar {n} found twice: {tars[n]} and {p}; using the first')
        tars.setdefault(n, p)
    for p in walk_find(INPUT, lambda d, f: re.fullmatch(r'prep_manifest_[01]\.json', f) is not None):
        mans.setdefault(int(p[-6]), p)
    return ds[0], tars, mans


def locate_train_labels():
    """-> path of labels_frac75_logit.csv (dataset eeshsaxena/rsna-knee-frac75-labels, any mount layout). Copies at
    several paths are fine only if byte-identical (the sha256 is checked by the caller against the baked value)."""
    name = CFG['LABELS_TRAIN']
    found = walk_find(INPUT, lambda d, f: f == name)
    if not found:
        abort(f'{name} not found under {INPUT}: attach dataset eeshsaxena/rsna-knee-frac75-labels')
    if len({open(p, 'rb').read() for p in found}) != 1:
        abort(f'{name} found at {len(found)} paths with DIFFERENT bytes: {found}')
    if len(found) > 1:
        warn(f'{name} visible at {len(found)} paths (identical bytes): using {found[0]}')
    return found[0]


def labels_variant(b, want_sha256):
    """The box bytes of a csv from the bytes seen here: as is, or after a line-ending rewrite in either direction
    (the box files are CRLF). -> (box bytes, 'exact' | 'CRLF->LF' | 'LF->CRLF') or None."""
    lf = b.replace(bytes([13, 10]), bytes([10]))
    for how, x in (('exact', b), ('CRLF->LF', lf), ('LF->CRLF', lf.replace(bytes([10]), bytes([13, 10])))):
        if hashlib.sha256(x).hexdigest() == want_sha256:
            return x, how
    return None


def only_fracture_differs(frac_b, mix_b):
    """frac75 = mix50 with ONLY the last field (Fracture) of each line replaced: same header, same ids in the same
    order, the 11 other columns byte-identical (the box check, repeated here on the bytes the trainer reads)."""
    a = [l.rsplit(b',', 1) for l in frac_b.split(bytes([10])) if l]
    m = [l.rsplit(b',', 1) for l in mix_b.split(bytes([10])) if l]
    return len(a) == len(m) > 1 and a[0] == m[0] and all(x[0] == y[0] for x, y in zip(a, m))


# ---------------------------------------------------------------- store extraction
class _HashReader:
    def __init__(self, f, h):
        self.f, self.h, self.n = f, h, 0

    def read(self, n=-1):
        b = self.f.read(n)
        self.h.update(b)
        self.n += len(b)
        return b


def extract_tar(name, path, dst):
    """Stream one prep tar into dst, hashing the WHOLE file on the way (one read). Members must be flat
    '<StudyInstanceUID>.npz' names. -> (name, sha256, n_members, bytes)."""
    h = hashlib.sha256()
    n = 0
    with open(path, 'rb') as f:
        r = _HashReader(f, h)
        with tarfile.open(fileobj=r, mode='r|') as t:
            for m in t:
                if not m.isfile():
                    continue
                if not NPZ_RE.fullmatch(m.name):
                    raise ValueError(f'{name}: unexpected member {m.name!r}')
                src = t.extractfile(m)
                tmp = f'{dst}/{m.name}.part'
                with open(tmp, 'wb') as o:
                    shutil.copyfileobj(src, o, 1 << 22)
                os.replace(tmp, f'{dst}/{m.name}')
                n += 1
        while r.read(1 << 22):           # tar end padding: hash the whole file
            pass
    return name, h.hexdigest(), n, r.n


# ---------------------------------------------------------------- fold arithmetic (same as fleet_train.py)
def expected_counts(ds_dir, store, F, labels_file=None):
    import pandas as pd
    lab = pd.read_csv(f'{ds_dir}/{labels_file or CFG["LABELS_TRAIN"]}',          # the file the trainer reads
                      dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    folds = pd.read_csv(f'{ds_dir}/folds5.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['fold']
    excl = set(pd.read_csv(f'{ds_dir}/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID)
    held = set(folds.index[folds < 0]) | excl
    have = {s for s in lab.index if os.path.exists(f'{store}/{s}.npz')}
    ids = [s for s in lab.index if s in folds.index and s in have]
    tr = [s for s in ids if s not in held and folds[s] != F]
    va = [s for s in ids if s not in held and folds[s] == F]
    return dict(labels=len(lab), store_has=len(have), held=len(held), train=len(tr), val=len(va),
                iters_per_epoch=len(tr) // 2)


# ---------------------------------------------------------------- trainer log parsing
RE_IT = re.compile(r'^ep (\d+) it (\d+)/(\d+) loss \S+ ([0-9.]+)s/it', re.M)
RE_LAB = re.compile(r'^labels (\d+) \| store has (\d+) \| held-out \(fold<0 or --exclude\) (\d+) \| train (\d+) \| '
                    r'val (\d+) \| all_data (\w+) \| device (\w+)', re.M)
RE_CON = re.compile(r'^contract (\{.*\}) \| select_sha (\w+) \| jpeg_fp (\w+)\s*$', re.M)
RE_BJ = re.compile(r'^BJ opt-in: .*\| model_sha (\w+)\s*$', re.M)
RE_EV = re.compile(r'^ep (\d+) (raw|ema) \|.*\| val AUC vs \S+ (\S+) \((\d+) labels\) \| vs train labels (\S+) \| '
                   r'epoch ([0-9.]+) min\s*$', re.M)
RE_TO = re.compile(r'^ep (\d+) TRAIN-ONLY ([0-9.]+) min', re.M)


def parse_its(text, ep=0):
    """-> [(it, n_iters, s_per_it_cumulative)] for one epoch, in log order."""
    return [(int(m.group(2)), int(m.group(3)), float(m.group(4))) for m in RE_IT.finditer(text) if int(m.group(1)) == ep]


def sit_window(its):
    """Steady s/it from the LAST two progress lines. The trainer prints (now - epoch_start) / (it + 1), so the
    cumulative time at line `it` is (it + 1) * s; the window drops the first-iteration (cudnn benchmark) cost."""
    if len(its) < 2:
        return None
    (n1, _, s1), (n2, _, s2) = its[-2], its[-1]
    return ((n2 + 1) * s2 - (n1 + 1) * s1) / (n2 - n1)


def parse_header(text):
    """The trainer's start lines -> dict (None for lines not printed yet)."""
    h = dict(labels=None, contract=None, select_sha=None, jpeg_fp=None, model_sha=None)
    m = RE_LAB.search(text)
    if m:
        h['labels'] = dict(labels=int(m.group(1)), store_has=int(m.group(2)), held=int(m.group(3)),
                           train=int(m.group(4)), val=int(m.group(5)), all_data=m.group(6), device=m.group(7))
    m = RE_CON.search(text)
    if m:
        h['contract'] = ast.literal_eval(m.group(1))
        h['select_sha'], h['jpeg_fp'] = m.group(2), m.group(3)
    m = RE_BJ.search(text)
    if m:
        h['model_sha'] = m.group(1)
    return h


def header_problems(h, exp_counts, box):
    """Compare a parsed trainer header with the box contract and the expected fold counts. -> list of problems
    ([] = OK). Lines not printed yet are reported as missing."""
    bad = []
    a = box['ckpt_args']
    if h['contract'] is None:
        bad.append('contract line missing')
    else:
        for k in ('pixel_sha', 'tensor_sha'):
            if h['contract'].get(k) != a['contract'][k]:
                bad.append(f'{k} {h["contract"].get(k)} != box {a["contract"][k]}')
        if json.loads(json.dumps(h['contract'])) != a['contract']:
            bad.append(f'contract {h["contract"]} != box {a["contract"]}')
        if h['select_sha'] != a['select_sha']:
            bad.append(f'select_sha {h["select_sha"]} != box {a["select_sha"]}')
        if h['jpeg_fp'] != a['jpeg_fp']:
            bad.append(f'jpeg_fp {h["jpeg_fp"]} != box {a["jpeg_fp"]}')
    if h['labels'] is None:
        bad.append('labels line missing')
    else:
        for k in ('labels', 'store_has', 'held', 'train', 'val'):
            if h['labels'][k] != exp_counts[k]:
                bad.append(f'{k} {h["labels"][k]} != expected {exp_counts[k]}')
        if h['labels']['all_data'] != 'False':
            bad.append('all_data is not False')
    if h['model_sha'] is None:
        bad.append('BJ opt-in / model_sha line missing')
    elif h['model_sha'] != a['model_sha']:
        bad.append(f'model_sha {h["model_sha"]} != box {a["model_sha"]}')
    return bad


def progress(text):
    """-> dict(evals=[(ep, tag, val, epoch_min)], last_ep_done, train_only_min=[...], best={tag: (val, ep)})."""
    ev = [(int(m.group(1)), m.group(2), m.group(3), float(m.group(6))) for m in RE_EV.finditer(text)]
    best = {}
    for ep, tag, v, _ in ev:
        try:
            fv = float(v)
        except ValueError:
            continue
        if math.isfinite(fv) and (tag not in best or fv > best[tag][0]):
            best[tag] = (fv, ep)
    done = [ep for ep, tag, _, _ in ev if tag == 'ema']
    return dict(evals=ev, last_ep_done=max(done) if done else -1, best=best,
                train_only_min=[float(m.group(2)) for m in RE_TO.finditer(text)])


def closing_line(text):
    """fleet_train.py ends with json.dumps(best) = {"raw": .., "ema": ..}. None when the run did not finish."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return None
    try:
        j = json.loads(lines[-1])
    except Exception:
        return None
    return j if isinstance(j, dict) and set(j) == {'raw', 'ema'} else None


# ---------------------------------------------------------------- epochs
def choose_epochs(s_it, iters, elapsed_s, budget_h=None, eval_frac=None, startup_s=None, emax=None):
    """E = min(EPOCHS_MAX, floor(time left in the budget / per-epoch time incl. eval)), at least 1."""
    budget_h = CFG['BUDGET_H'] if budget_h is None else budget_h
    eval_frac = CFG['EVAL_FRAC'] if eval_frac is None else eval_frac
    startup_s = CFG['STARTUP_S'] if startup_s is None else startup_s
    emax = CFG['EPOCHS_MAX'] if emax is None else emax
    epoch_s = iters * s_it * (1.0 + eval_frac)
    avail_s = budget_h * 3600.0 - elapsed_s - startup_s
    E = int(max(1, min(emax, math.floor(avail_s / epoch_s)))) if epoch_s > 0 else emax
    return dict(E=E, epoch_s=epoch_s, avail_s=avail_s, finish_elapsed_s=elapsed_s + startup_s + E * epoch_s)


def save_epochs_arg(E):
    return f'{max(1, E - CFG["SAVE_LAST"] + 1)}-{E}'   # per-epoch ckpts + OOFs for the LAST 3 epochs (3 x 2 x 113 MB per fold)


# ---------------------------------------------------------------- processes
HF_OFFLINE = ['1']                  # trainers load the bundled pretrained bytes from HF_HOME; '0' only on fallback


def recipe_args(F, E, save_ep, out, lbl_dir, workers):
    L = lbl_dir
    return ['--model', 'ttr', '--aug', 'heavy', '--train_slices', '16', '--drop_path', '0.2', '--dropout', '0.3',
            '--ema', '0.9995', '--epochs', str(E), '--seed', str(CFG['SEED']), '--save_epochs', save_ep,
            '--workers', str(workers),
            '--store', STORE, '--labels', f'{L}/{CFG["LABELS_TRAIN"]}', '--folds', f'{L}/folds5.csv',
            '--exclude', f'{L}/gold74_eval_set.csv', '--eval_labels', f'{L}/llm_labels_v4_blend.csv',
            '--gold', f'{L}/gold74_eval_set.csv', '--fold', str(F), '--out', out]


def launch(F, gpu, E, save_ep, out, logp, lbl_dir, workers):
    os.makedirs(os.path.dirname(logp), exist_ok=True)
    cmd = [sys.executable, CFG['TRAINER'] or f'{CODE}/fleet_train.py'] + recipe_args(F, E, save_ep, out, lbl_dir, workers)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', HF_HOME=HF_HOME, HF_HUB_OFFLINE=HF_OFFLINE[0],
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    lf = open(logp, 'w')
    p = subprocess.Popen(cmd, cwd=CODE, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    p.kt_log, p.kt_cmd = lf, cmd
    log(f'launched fold {F} on GPU {gpu}: pid {p.pid} | log {logp}')
    return p


def stop_group(p, grace):
    """SIGTERM the trainer's whole process group (main + DataLoader workers), SIGKILL after `grace` s.
    -> 'exited' | 'sigterm' | 'sigkill'."""
    how = 'exited'
    if p.poll() is None:
        how = 'sigterm'
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            p.wait(grace)
        except subprocess.TimeoutExpired:
            how = 'sigkill'
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()
    try:
        os.killpg(p.pid, signal.SIGKILL)          # stray workers of a main that already exited
    except (ProcessLookupError, PermissionError):
        pass
    try:
        p.kt_log.close()
    except Exception:
        pass
    return how


def wait_quiet(dirs, quiet_s=15.0, max_wait_s=120.0, poll_s=2.0):
    """Before the hard-wall SIGTERM. fleet_train.save() first DELETES the previous best ckpt of the tag, then torch.save()s
    the new one straight to its final name (no tmp + rename), then writes the OOF csv: a kill inside that window leaves a
    truncated best ckpt (or a ckpt/OOF pair from different epochs). Wait (bounded) until nothing under `dirs` (files AND
    directories: a delete changes the dir mtime) changed for `quiet_s` s. Saves only follow an eval line and take seconds.
    -> seconds waited."""
    t = time.time()
    while True:
        newest = 0.0
        for d in dirs:
            for root, _, fs in os.walk(d):
                for p in [root] + [os.path.join(root, f) for f in fs]:
                    try:
                        newest = max(newest, os.path.getmtime(p))
                    except OSError:
                        pass
        if time.time() - newest >= quiet_s or time.time() - t >= max_wait_s:
            return time.time() - t
        time.sleep(poll_s)


def gpu_query():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return []
    keys = ['index', 'name', 'mem_total_mib', 'mem_used_mib', 'util_pct']
    return [dict(zip(keys, [x.strip() for x in l.split(',')])) for l in out.strip().splitlines() if l.count(',') == 4]


def wait_gpus_free(timeout=120, max_used_mib=1000):
    if CFG['SKIP_GPU_CHECK']:
        return True
    t = time.time()
    while time.time() - t < timeout:
        g = gpu_query()
        if g and all(float(x['mem_used_mib']) < max_used_mib for x in g):
            return True
        time.sleep(3)
    return False


def cpu_times():
    v = [int(x) for x in open('/proc/stat').readline().split()[1:]]
    return sum(v[:8]), v[3] + v[4]


def cpu_busy(a, b):
    return 1.0 - (b[1] - a[1]) / max(1, b[0] - a[0])


MODELCHECK_SRC = r'''
import os, sys, json
sys.path.insert(0, sys.argv[1])
import torch, timm
from fleet_model_ttr import KneeFleetNetTTR
box = json.load(open(sys.argv[2]))
m = KneeFleetNetTTR(ttr_size=224, backbone='convnext_tiny.fb_in22k_ft_in1k', dim=256, contrast_emb=True, n_views=4,
                    n_slices=24, n_contrast=4, pretrained=True, drop_path=0.2, dropout=0.3, grad_ckpt=True)
got = [[k, list(v.shape), str(v.dtype).replace('torch.', '')] for k, v in m.state_dict().items()]
exp = box['state_dict']
res = dict(timm=timm.__version__, torch=torch.__version__, n_keys=len(got), n_keys_box=len(exp), keys_match=got == exp)
if got != exp:
    gs, es = {g[0]: g for g in got}, {e[0]: e for e in exp}
    res['only_here'] = sorted(set(gs) - set(es))[:10]; res['only_box'] = sorted(set(es) - set(gs))[:10]
    res['shape_or_dtype_diff'] = [k for k in gs if k in es and gs[k] != es[k]][:10]
from safetensors.torch import load_file
pre = load_file(sys.argv[3]); cnn = m.cnn.state_dict()
miss = [k for k in cnn if k not in pre]; neq = [k for k in cnn if k in pre and not torch.equal(cnn[k].float(), pre[k].float())]
res.update(cnn_keys=len(cnn), n_missing_in_pretrained=len(miss), n_not_equal=len(neq), missing=miss[:5], not_equal=neq[:5],
           pretrained_equal=(not miss and not neq))
print('KTMODELCHECK ' + json.dumps(res), flush=True)
'''


def install_hf_cache(ds_dir, exp):
    """Put the box's own convnext_tiny.fb_in22k_ft_in1k bytes into an HF hub cache, so timm's pretrained=True
    resolves them offline (HF_HUB_OFFLINE=1) exactly like the box (~/.cache/huggingface/hub)."""
    hf = exp['hf']
    repo = f'{HF_HOME}/hub/models--{hf["repo"].replace("/", "--")}'
    os.makedirs(f'{repo}/refs', exist_ok=True)
    os.makedirs(f'{repo}/snapshots/{hf["revision"]}', exist_ok=True)
    with open(f'{repo}/refs/main', 'w') as f:
        f.write(hf['revision'])
    dst = f'{repo}/snapshots/{hf["revision"]}/model.safetensors'
    shutil.copyfile(f'{ds_dir}/{hf["file"]}', dst)
    assert sha256_file(dst) == hf['sha256'], 'pretrained bytes changed while copying'
    return dst
