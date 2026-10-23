# === K9N N STAGE: appended after the K9 fleet stage (cell 31) by k9n/build_k9n.py (2026-10-08, Claude) ===
# Third leg N = the PUBLIC .949 CoAtNet by nartaa (notebook nartaa/rsna-knee-0949-anatomical-mirror, weights dataset
# nartaa/rsna-knee-publication-swa-weights-20261007). Its inference code runs as a separate CHILD PROCESS (the public cell 4
# reproduced verbatim except the K9N-P patches of k9n/n949/make_child.py; credit: nartaa, built on dreaddevelopment's
# Raptor CoAtNet/MIL recipe and timm), embedded below as base64 (_K9N_CHILD_B64, sha256 pinned in K9N_CFG).
#   S1 STRICT, fail-loud: the FIRST action is to delete submission.csv (the fleet stage's P+F blend; its bytes stay in
#      submission_w50.csv). submission.csv is written again ONLY as the LAST step of a clean N stage. Any failure (fleet
#      stage not clean, wall, weights missing / wrong sha256, child crash or non-zero rc, ids / schema / gates) leaves NO
#      submission.csv: the slot errors instead of silently scoring P+F or the parent. The N fail-loud files
#      (submission_n949.csv, submission_pfn_*.csv) exist only after a clean stage.
#   S2 starts only after the fleet stage finished CLEAN (its workers are gone, both GPUs free) and only in strict mode.
#   S3 one wall: kernel start = min(/proc start, parent T0); the child is killed (whole process group: its fork prep pool
#      too) at kernel start + WALL_H - RESERVE_MIN; the stage is skipped (LOUD) when less than MIN_START_MIN minutes
#      remain, or when no kernel clock is readable at all. N never runs past WALL_H (8.6 h); the 9 h limit keeps >= 0.4 h.
#   S4 the weights file is resolved at the public notebook's own dataset paths and its sha256 asserted here (and again by
#      the verbatim child, per GPU); the child never searches /kaggle/input for it (K9N-P4).
#   S5 gates on the child's npz: ids == the parent's ids, finite, in [0, 1], P == (plain + mirror) / 2 exactly and the
#      mirror view differs from the plain view on every non-empty study, FALLBACK + empty studies <= MAX_FAIL_FRAC
#      (0 when n < GATE_MIN_N, i.e. on the commit), N's own submission file == its rank formula, Spearman(N, parent) mean
#      >= MIN_RHO_MEAN once n >= RHO_MIN_N (label order / decode bug guard).
#   S6 blend: submission.csv = rank_pct(wP r(P) + wF r(F) + wN r(N)) per label, r = average-tie percentile rank, P =
#      _k2_parent.csv, F = submission_fleetsolo.csv (the fleet's raw-prob mean), N = the child's raw probability (mean of
#      the plain and mirrored sigmoid; FALLBACK / empty studies -> column median, rank-neutral). W3 and the variant
#      triples are literals in K9N_CFG (written by build_k9n.py; pushed kernels get no env vars).
import os, sys, json, time, shutil, subprocess, hashlib, traceback, signal, base64, tempfile
import numpy as np, pandas as pd

K9N = {'W3': [0.10, 0.30, 0.60], 'W3_VARIANTS': [[0.30, 0.30, 0.40]], 'WALL_H': 8.6, 'RESERVE_MIN': 2.0,
       'MIN_START_MIN': 20.0, 'N_CENTRAL_MIN': 25.0, 'N_SLOW_MIN': 40.0, 'FLEET_CENTRAL_MIN': 58.0, 'N_HIDDEN': 1322,
       'POLL_S': 2.0, 'MAX_FAIL_FRAC': 0.02, 'GATE_MIN_N': 11, 'RHO_MIN_N': 50, 'MIN_RHO_MEAN': 0.30,
       'WEIGHTS_FILE': 'raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt',
       'WEIGHTS_SHA256': '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef',
       'WEIGHTS_DS': 'nartaa/rsna-knee-publication-swa-weights-20261007',
       'SRC_SHA256': 'bcd864cbf349882e876959b49cb348cbb89b4c62fc204d4e4261049c5e3a79c5', 'CHILD_SHA256': None,
       'TAG': None}
K9N.update(globals().get('K9N_CFG') or {})
_N_WORK = '/kaggle/working'
_N_SUB = f'{_N_WORK}/submission.csv'
_N_PAR = f'{_N_WORK}/_k2_parent.csv'
_N_FSOLO = f'{_N_WORK}/submission_fleetsolo.csv'
_N_FSTAT = f'{_N_WORK}/_k2_stage_fleet.json'
_N_STATUS = f'{_N_WORK}/_k9n_stage.json'
_N_RECEIPT = f'{_N_WORK}/n949_receipt.json'
_N_COUT = f'{_N_WORK}/_n949_sub.csv'           # the child's own (verbatim-format) submission, never submission.csv
_N_CNPZ = f'{_N_WORK}/_n949_probs.npz'
_N_CLOG = f'{_N_WORK}/n949_child.log'
_N_WCANDS = [f'/kaggle/input/{K9N["WEIGHTS_DS"].split("/")[1]}/{K9N["WEIGHTS_FILE"]}',             # = public cell 3
             f'/kaggle/input/datasets/{K9N["WEIGHTS_DS"]}/{K9N["WEIGHTS_FILE"]}']
_N_TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
              'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def _n_log(msg):
    print(f'[k9n {time.strftime("%H:%M:%S")}] {msg}', flush=True)


class _NFail(Exception):
    """The N stage did not finish cleanly: no submission.csv, no N fail-loud file."""


def _n_tname(w):
    return 'submission_pfn_' + ''.join(f'{int(round(float(x) * 100)):02d}' for x in w) + '.csv'


def _n_loud_names():
    names = ['submission_n949.csv']
    for w in [K9N['W3']] + list(K9N['W3_VARIANTS']):
        if _n_tname(w) not in names:
            names.append(_n_tname(w))
    return names


def _n_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def _n_write_atomic(df, dst):
    tmp = dst + '.tmp'; df.to_csv(tmp, index=False); os.replace(tmp, dst)


def _n_remove(names):
    errs = []
    for f in names:
        for p in (f'{_N_WORK}/{f}', f'{_N_WORK}/{f}.tmp'):
            try:
                if os.path.lexists(p):
                    os.remove(p)
            except BaseException as e:
                errs.append(f'{os.path.basename(p)}: {e!r}')
    return errs


def _n_delete_main():
    """S1: delete submission.csv; if the delete fails, truncate it (an empty file cannot score)."""
    try:
        if os.path.lexists(_N_SUB):
            os.remove(_N_SUB)
        return 'deleted' if not os.path.lexists(_N_SUB) else 'STILL PRESENT'
    except BaseException as e:
        try:
            open(_N_SUB, 'w').close()
            return f'delete failed ({e!r}), truncated to 0 bytes'
        except BaseException as e2:
            return f'DELETE AND TRUNCATE FAILED {e!r} {e2!r}'


def _n_kernel_start():
    """Earliest of this process's start (/proc) and the parent's T0 (== the fleet cell's _k2_kernel_start)."""
    starts, now = [], time.time()
    try:
        up = float(open('/proc/uptime').read().split()[0])
        st = int(open('/proc/self/stat').read().rsplit(')', 1)[1].split()[19]) / os.sysconf('SC_CLK_TCK')
        starts.append(now - max(0.0, up - st))
    except Exception:
        pass
    t0p = globals().get('T0')
    if isinstance(t0p, (int, float)) and 1e9 < t0p <= now:
        starts.append(float(t0p))
    return min(starts) if starts else None


def _n_rank(a):
    return pd.DataFrame(np.asarray(a, np.float64)).rank(method='average', pct=True).to_numpy()


def _n_kill(p):
    """Kill the child's whole process group (start_new_session: pgid == pid; the fork prep pool is inside it)."""
    if p is None or p.poll() is not None:
        return 'not running'
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass
    try:
        p.wait(timeout=60)
    except Exception:
        pass
    return f'killed (rc {p.poll()})'


def _n_child_tail(k=60):
    try:
        lines = open(_N_CLOG, errors='replace').read().splitlines()
    except Exception as e:
        return [f'(no child log: {e!r})']
    keep = [l for l in lines if 'Debugger warning' not in l and 'frozen modules' not in l and 'PYDEVD' not in l
            and 'Xfrozen_modules' not in l]
    return keep[-k:]


def _n_stage(t_cell, rec, procs):
    """Run N after a clean fleet stage, gate it, return {file name: DataFrame}. Raises on anything short of clean."""
    w3 = [float(x) for x in K9N['W3']]
    for w in [w3] + [[float(x) for x in v] for v in K9N['W3_VARIANTS']]:
        if len(w) != 3 or min(w) < 0 or abs(sum(w) - 1.0) > 1e-9 or w[2] <= 0:
            raise _NFail(f'weight triple {w} must be 3 non-negative weights summing to 1 with N > 0')
    if not K9N.get('CHILD_SHA256') or not isinstance(globals().get('_K9N_CHILD_B64'), str):
        raise _NFail('no embedded child (K9N_CFG CHILD_SHA256 / _K9N_CHILD_B64 missing: not a build_k9n.py kernel)')
    # ---- S2 the fleet stage must be clean (strict) and its files untouched
    try:
        fst = json.load(open(_N_FSTAT))
    except Exception as e:
        raise _NFail(f'fleet stage status {os.path.basename(_N_FSTAT)} unreadable ({e!r}): N not started')
    rec['fleet_status'] = {k: fst.get(k) for k in ('clean', 'why', 'main_mode', 'stage_min', 'models')}
    if fst.get('clean') is not True:
        raise _NFail(f'fleet stage NOT clean ({fst.get("why")}): N not started (strict: no submission)')
    if fst.get('main_mode') != 'strict':
        raise _NFail(f'fleet MAIN_MODE {fst.get("main_mode")!r} != strict: K9N runs only after a strict fleet stage')
    files = fst.get('files') or {}
    for f in ('submission_fleetsolo.csv',):
        if not os.path.isfile(f'{_N_WORK}/{f}') or files.get(f) != _n_sha(f'{_N_WORK}/{f}'):
            raise _NFail(f'{f} missing or changed since the fleet stage wrote it')
    pub = pd.read_csv(_N_PAR, dtype={'StudyInstanceUID': str})
    fso = pd.read_csv(_N_FSOLO, dtype={'StudyInstanceUID': str})
    if pub.columns.tolist() != ['StudyInstanceUID', *_N_TARGETS] or fso.columns.tolist() != pub.columns.tolist():
        raise _NFail(f'parent / fleetsolo columns {pub.columns.tolist()} / {fso.columns.tolist()}')
    if len(fso) != len(pub) or not pub['StudyInstanceUID'].is_unique or             (fso['StudyInstanceUID'].values != pub['StudyInstanceUID'].values).any():
        raise _NFail('parent ids not unique or fleetsolo row order != parent row order')
    if not (np.isfinite(pub[_N_TARGETS].to_numpy(np.float64)).all() and np.isfinite(fso[_N_TARGETS].to_numpy(np.float64)).all()):
        raise _NFail('parent / fleetsolo has non-finite values')
    n = len(pub); ids_par = pub['StudyInstanceUID'].tolist()
    # ---- S3 one wall
    k0 = _n_kernel_start(); now = time.time()
    if k0 is None:                                                      # not expected (/proc + parent T0): no wall = no N
        raise _NFail('RUNTIME GUARD: no kernel clock readable (/proc start, parent T0): N stage SKIPPED (strict)')
    deadline = k0 + float(K9N['WALL_H']) * 3600 - float(K9N['RESERVE_MIN']) * 60
    left = deadline - now
    fmin = fst.get('stage_min')
    speed = max(1.0, float(fmin) / float(K9N['FLEET_CENTRAL_MIN'])) if isinstance(fmin, (int, float)) and n >= int(K9N['GATE_MIN_N']) else 1.0
    scale = min(1.0, n / float(K9N['N_HIDDEN'])) if n >= int(K9N['GATE_MIN_N']) else 0.05
    est_c, est_s = float(K9N['N_CENTRAL_MIN']) * speed * scale, float(K9N['N_SLOW_MIN']) * speed * scale
    rec.update(kernel_age_h_start=(now - k0) / 3600, left_min=left / 60, fleet_speed=speed,
               est_central_min=est_c, est_slow_min=est_s, n=n)
    _n_log(f'kernel age {(now - k0) / 3600:.2f} h; {left / 60:.1f} min to the '
           f'{K9N["WALL_H"]} h wall minus {K9N["RESERVE_MIN"]} min reserve; {n} studies; N estimate ~{est_c:.0f} min '
           f'central / ~{est_s:.0f} min slow (fleet speed x{speed:.2f}); weights P/F/N {w3}')
    if left < float(K9N['MIN_START_MIN']) * 60:
        raise _NFail(f'RUNTIME GUARD: only {left / 60:.1f} min before the wall (< MIN_START_MIN {K9N["MIN_START_MIN"]}): '
                     f'N stage SKIPPED (strict: no submission.csv)')
    if left < est_s * 60:
        _n_log(f'WARNING: {left / 60:.1f} min left < slow estimate {est_s:.0f} min: the child is KILLED at the wall if it '
               f'has not finished (strict: no submission.csv)')
    # ---- GPUs + host state
    import gc; gc.collect()
    import torch
    n_dev = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if not n_dev:
        raise _NFail('no GPU visible (the published N pipeline is CUDA/fp16 only)')
    torch.cuda.empty_cache()
    try:
        rec['gpu_free_mib'] = [int(torch.cuda.mem_get_info(i)[0] / 2**20) for i in range(n_dev)]
    except Exception:
        rec['gpu_free_mib'] = None
    try:
        import psutil
        rec['host_ram_avail_gb'] = round(psutil.virtual_memory().available / 2**30, 1)
    except Exception:
        rec['host_ram_avail_gb'] = None
    rec['gpus'] = n_dev
    _n_log(f'{n_dev} GPU(s), free MiB {rec["gpu_free_mib"]}, host RAM available {rec["host_ram_avail_gb"]} GB')
    # ---- S4 weights: the public notebook's own dataset paths, sha256 asserted here
    wp = next((p for p in _N_WCANDS if os.path.isfile(p)), None)
    if wp is None:
        raise _NFail(f'weights {K9N["WEIGHTS_FILE"]} not found at {_N_WCANDS} (attach {K9N["WEIGHTS_DS"]})')
    wsha = _n_sha(wp)
    rec.update(weights_path=wp, weights_sha256=wsha)
    if wsha != K9N['WEIGHTS_SHA256']:
        raise _NFail(f'weights sha256 {wsha} != pinned {K9N["WEIGHTS_SHA256"]} ({wp})')
    _n_log(f'weights {wp} sha256 {wsha[:16]}... == pinned')
    # ---- child: decode, verify, run in its own session (process group)
    src = base64.b64decode(globals()['_K9N_CHILD_B64'])
    csha = hashlib.sha256(src).hexdigest()
    rec['child_sha256'] = csha
    if csha != K9N['CHILD_SHA256']:
        raise _NFail(f'embedded child sha256 {csha} != K9N_CFG CHILD_SHA256 {K9N["CHILD_SHA256"]}')
    d = tempfile.mkdtemp(prefix='k9n_child_')
    rec['child_dir'] = d
    child = os.path.join(d, 'n949_child.py')
    with open(child, 'wb') as fh:
        fh.write(src)
    env = {**os.environ, 'N949_WEIGHTS': wp, 'N949_OUT': _N_COUT, 'N949_NPZ': _N_CNPZ}
    t_launch = time.time()
    with open(_N_CLOG, 'w') as lf:
        p = subprocess.Popen([sys.executable, child], cwd=d, env=env, stdout=lf, stderr=subprocess.STDOUT,
                             start_new_session=True)
    procs.append(p)
    _n_log(f'child pid {p.pid} started ({os.path.basename(child)}, sha256 {csha[:16]}...); log {os.path.basename(_N_CLOG)}')
    while True:
        rc = p.poll()
        if rc is not None:
            break
        if time.time() > deadline:
            how = _n_kill(p)
            raise _NFail(f'RUNTIME GUARD: N child still running at the wall ({K9N["WALL_H"]} h - {K9N["RESERVE_MIN"]} min): '
                         f'{how} after {(time.time() - t_launch) / 60:.1f} min (strict: no submission.csv)')
        time.sleep(float(K9N['POLL_S']))
    rec['child_s'] = time.time() - t_launch
    rec['child_rc'] = rc
    if rc != 0:
        raise _NFail(f'N child exited rc {rc} after {rec["child_s"]:.0f} s (see {os.path.basename(_N_CLOG)})')
    # ---- S5 gates on the child's npz
    if not os.path.isfile(_N_CNPZ) or not os.path.isfile(_N_COUT):
        raise _NFail('N child rc 0 but wrote no npz / csv')
    z = np.load(_N_CNPZ, allow_pickle=False)
    meta = json.loads(str(z['meta']))
    rec['child_meta'] = {k: meta.get(k) for k in ('n', 'views', 'torch', 'timm', 'ngpu', 'python', 'seconds',
                                                   'k_eval', 'input_res', 'img', 'maxs', 'fastread')}
    rec['child_fallback'], rec['child_empty'] = meta.get('fallback'), meta.get('empty')
    if meta.get('labels') != _N_TARGETS:
        raise _NFail(f'child label order {meta.get("labels")} != {_N_TARGETS}')
    shas = meta.get('weights_sha256') or []
    if not shas or any(s != K9N['WEIGHTS_SHA256'] for s in shas):
        raise _NFail(f'child weights sha256 {shas} != pinned')
    if (meta.get('k_eval'), meta.get('input_res'), meta.get('img'), meta.get('maxs'), meta.get('fastread')) != \
            (94, 320, 384, 96, 'study_lut'):
        raise _NFail(f'child recipe {meta} != K94 / 320 / 384 / 96 / study_lut')
    ids = [str(x) for x in z['ids']]
    if len(ids) != n or len(set(ids)) != n or set(ids) != set(ids_par):
        raise _NFail(f'child ids ({len(ids)}, unique {len(set(ids))}) != parent ids ({n})')
    PN = np.asarray(z['P'], np.float32); PL = np.asarray(z['P_plain'], np.float32); PM = np.asarray(z['P_mirror'], np.float32)
    if PN.shape != (n, 12) or PL.shape != (n, 12) or PM.shape != (n, 12):
        raise _NFail(f'child shapes P {PN.shape} plain {PL.shape} mirror {PM.shape}')
    if not (np.isfinite(PN).all() and (PN >= 0).all() and (PN <= 1).all()):
        raise _NFail('child P not finite / outside [0, 1]')
    pos = {s: i for i, s in enumerate(ids)}
    fb = set(meta.get('fallback') or []); em = set(meta.get('empty') or [])
    if not (fb | em) <= set(ids):
        raise _NFail('fallback / empty ids not in the child ids')
    bad = sorted(fb | em)
    rec.update(n_fallback=len(fb), n_empty=len(em), fail_frac=len(bad) / max(1, n))
    lim = float(K9N['MAX_FAIL_FRAC']) if n >= int(K9N['GATE_MIN_N']) else 0.0
    if len(bad) / max(1, n) > lim:
        raise _NFail(f'N failed studies {len(bad)}/{n} (fallback {len(fb)}, empty {len(em)}) > {lim:.2f}')
    good = np.array([s not in fb for s in ids]); full = np.array([s not in fb and s not in em for s in ids])
    if not (np.isfinite(PL[good]).all() and np.isfinite(PM[good]).all()
            and np.array_equal(PN[good], np.float32(0.5) * (PL[good] + PM[good]))):
        raise _NFail('child P != (plain + mirror) / 2 on the served rows (the anatomical mirror view did not run as published)')
    diff = np.abs(PL.astype(np.float64) - PM).max(axis=1)
    if full.any() and not (diff[full] > 0).all():
        raise _NFail(f'mirror view == plain view on {int((diff[full] == 0).sum())} non-empty studies (mirror not applied)')
    rec['mirror_maxabs'] = float(diff[full].max()) if full.any() else None
    rec['mirror_meanabs'] = float(np.abs(PL[full].astype(np.float64) - PM[full]).mean()) if full.any() else None
    # N's own verbatim output == its formula (rankpct over the clipped probabilities, test.csv order)
    cs = pd.read_csv(_N_COUT, dtype={'StudyInstanceUID': str})
    if cs.columns.tolist() != ['StudyInstanceUID', *_N_TARGETS] or cs['StudyInstanceUID'].tolist() != ids:
        raise _NFail(f'child csv columns / order {cs.columns.tolist()[:3]}...')
    x = np.clip(PN, 0, 1); rk = x.argsort(0).argsort(0).astype(np.float64) / max(1, n - 1)
    dcs = float(np.abs(cs[_N_TARGETS].to_numpy(np.float64) - rk.astype(np.float32)).max())
    rec['child_csv_vs_formula'] = dcs
    if dcs > 1e-6:
        raise _NFail(f'child csv != its rankpct formula (max|diff| {dcs:.2e})')
    gs = np.asarray(z['gpu_s'], np.float64)
    if np.isfinite(gs).any():
        rec['gpu_s_per_study'] = dict(median=float(np.nanmedian(gs)), max=float(np.nanmax(gs)), n=int(np.isfinite(gs).sum()))
    # ---- S6 blend (parent row order)
    NV = PN.astype(np.float64)[[pos[s] for s in ids_par]]
    badp = np.array([s in fb or s in em for s in ids_par])
    if badp.any():
        med = np.median(NV[~badp], axis=0) if (~badp).any() else np.full(12, 0.5)
        NV[badp] = med[None]                                            # rank-neutral, as the fleet's median fill
    P = pub[_N_TARGETS].to_numpy(np.float64); F = fso[_N_TARGETS].to_numpy(np.float64)
    rP, rF, rN = _n_rank(P), _n_rank(F), _n_rank(NV)
    rho = []
    for j in range(12):
        c = np.corrcoef(rP[:, j], rN[:, j])[0, 1] if n > 2 else np.nan
        rho.append(float(c) if np.isfinite(c) else float('nan'))
    rho_mean = float(np.nanmean(rho)) if np.isfinite(rho).any() else float('nan')
    rec.update(rho_parent_n={t: round(r, 4) for t, r in zip(_N_TARGETS, rho)}, rho_mean=rho_mean)
    if n >= int(K9N['RHO_MIN_N']) and not rho_mean >= float(K9N['MIN_RHO_MEAN']):
        raise _NFail(f'parent~N Spearman mean {rho_mean:.3f} < {K9N["MIN_RHO_MEAN"]} (label order / decode bug?)')
    out = {}
    solo = pd.DataFrame(NV, columns=_N_TARGETS); solo.insert(0, 'StudyInstanceUID', ids_par)
    out['submission_n949.csv'] = solo
    for w in [w3] + [[float(x) for x in v] for v in K9N['W3_VARIANTS']]:
        b = _n_rank(w[0] * rP + w[1] * rF + w[2] * rN)
        df = pd.DataFrame(b, columns=_N_TARGETS); df.insert(0, 'StudyInstanceUID', ids_par)
        out[_n_tname(w)] = df
    out['submission.csv'] = out[_n_tname(w3)].copy()
    for f, df in out.items():                                            # final schema / order / range check
        v = df[_N_TARGETS].to_numpy(np.float64)
        if not (len(df) == n and df.columns.tolist() == pub.columns.tolist()
                and (df['StudyInstanceUID'].values == pub['StudyInstanceUID'].values).all()
                and np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all()):
            raise _NFail(f'{f}: failed the final schema/order/finite/range check')
    assert sorted(out) == sorted(_n_loud_names() + ['submission.csv'])
    return out


def _k9n_entry_inner():
    t_cell = time.time()
    loud = _n_loud_names()
    rec = {'config': {k: v for k, v in K9N.items()}, 'fail_loud_files': loud}
    _n_log(f'STAGE N start; STRICT: submission.csv = rank blend P/F/N {K9N["W3"]} (DELETED now, written last only if '
           f'clean); fail-loud files {loud}; tag {K9N.get("TAG")}')
    errs = _n_remove(loud + [os.path.basename(p) for p in (_N_STATUS, _N_RECEIPT, _N_COUT, _N_CNPZ, _N_CLOG)])
    if os.path.isfile(_N_SUB):
        try:
            rec['fleet_submission_sha256'] = _n_sha(_N_SUB)
        except Exception:
            rec['fleet_submission_sha256'] = None
    rec['delete_first'] = _n_delete_main()                               # S1: before anything else can fail
    _n_log(f'submission.csv {rec["delete_first"]} (the fleet P+F bytes stay in submission_w50.csv)')
    ok, why, procs = False, 'not attempted', []
    try:
        if errs:
            raise _NFail(f'stale N files could not be removed: {errs}')
        if rec['delete_first'] != 'deleted':
            raise _NFail(f'could not delete the fleet submission.csv first: {rec["delete_first"]}')
        outputs = _n_stage(t_cell, rec, procs)
        for f in loud + ['submission.csv']:                              # submission.csv LAST
            _n_write_atomic(outputs[f], f'{_N_WORK}/{f}')
        rec['files'] = {f: _n_sha(f'{_N_WORK}/{f}') for f in loud + ['submission.csv']}
        ok = True
        why = (f'N {rec["n"]} studies (fallback {rec["n_fallback"]}, empty {rec["n_empty"]}), child '
               f'{rec["child_s"] / 60:.1f} min, mirror mean|d| {rec.get("mirror_meanabs") or 0:.4f}, rho(P,N) mean '
               f'{rec["rho_mean"]:.3f}, W3 {K9N["W3"]}')
    except BaseException as e:                                           # incl. KeyboardInterrupt
        if not isinstance(e, _NFail):
            traceback.print_exc()
        why = f'{type(e).__name__}: {e}'
        rm = _n_remove(loud)
        if rm:
            rec['not_deleted'] = rm
        rec['delete_on_fail'] = _n_delete_main()
    finally:
        for p in procs:                                                  # never leave the child (or its pool) running
            rec['child_cleanup'] = _n_kill(p)
        if rec.get('child_dir'):
            shutil.rmtree(rec['child_dir'], ignore_errors=True)
    k0 = _n_kernel_start()
    rec.update(clean=ok, why=why, stage_min=(time.time() - t_cell) / 60,
               kernel_age_h_end=None if k0 is None else (time.time() - k0) / 3600)
    try:
        st = {k: rec.get(k) for k in ('clean', 'why', 'files', 'stage_min', 'kernel_age_h_start', 'kernel_age_h_end',
                                      'left_min', 'weights_sha256', 'child_sha256', 'n_fallback', 'n_empty')}
        st.update(W3=K9N['W3'], W3_VARIANTS=K9N['W3_VARIANTS'], TAG=K9N.get('TAG'), main_mode='strict',
                  fleet_submission_sha256=rec.get('fleet_submission_sha256'))
        tmp = _N_STATUS + '.tmp'
        json.dump(st, open(tmp, 'w'), indent=1, default=str); os.replace(tmp, _N_STATUS)
    except Exception as e:
        _n_log(f'status write error: {e!r}')
    if not ok:
        _n_log(f'STRICT: N stage not clean -> submission.csv {_n_delete_main()} (the submission will error)')
    for line in _n_child_tail():
        print(f'[n949-child] {line}', flush=True)
    for f in ['submission.csv'] + loud:
        p = f'{_N_WORK}/{f}'
        if not os.path.isfile(p):
            print(f'[k9n] {f:<27} ABSENT', flush=True); continue
        try:
            d = pd.read_csv(p, dtype={'StudyInstanceUID': str})
            print(f'[k9n] {f:<27} rows={len(d)} nan={int(d.iloc[:, 1:].isna().sum().sum())}', flush=True)
        except Exception as e:
            print(f'[k9n] {f:<27} UNREADABLE {e!r}', flush=True)
    if rec.get('gpu_s_per_study'):
        _n_log(f'N GPU s/study {rec["gpu_s_per_study"]} on {rec.get("gpus")} GPU(s); child {rec.get("child_s", 0):.0f} s')
    _n_log(f'STAGE N end {rec["stage_min"]:.2f} min; kernel age {rec.get("kernel_age_h_end") or float("nan"):.2f} h')
    _n_log(('RESULT OK N stage clean: ' if ok else
            'RESULT N STAGE NOT CLEAN (STRICT: submission.csv DELETED; N files NOT written): ') + why)
    try:
        print('[k9n] receipt', json.dumps(rec, default=str)[:3000], flush=True)
        json.dump(rec, open(_N_RECEIPT, 'w'), indent=1, default=str)
    except Exception as e:
        _n_log(f'receipt write error (outputs unaffected): {e!r}')
    return rec


def _k9n_entry():
    """Last line of defence: delete submission.csv FIRST, then report."""
    try:
        return _k9n_entry_inner()
    except BaseException as e:
        dm = _n_delete_main()
        try:
            traceback.print_exc(); _n_log(f'UNEXPECTED {e!r} outside the guarded path; submission.csv {dm}')
            _n_remove(_n_loud_names())
        except BaseException:
            pass
        return None


_k9n_receipt = _k9n_entry()
