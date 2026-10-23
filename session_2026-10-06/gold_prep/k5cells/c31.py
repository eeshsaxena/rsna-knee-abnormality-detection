K2_CFG = {'W_MAIN': 0.25, 'W_VARIANTS': [0.3, 0.5], 'WALL_H': 8.6, 'MIN_START_MIN': 20.0, 'MAIN_MODE': 'strict', 'MEMBERS': ['f0_s1043_raw_ttr224-320x24_cv8835_best.pt', 'f1_s1043_raw_ttr224-320x24_cv8926_best.pt', 'f2_s1043_raw_ttr224-320x24_cv8907_best.pt', 'f3_s1043_raw_ttr224-320x24_cv8763_best.pt', 'f4_s1043_raw_ttr224-320x24_cv8928_best.pt'], 'TTA': {'f0_s1043_raw_ttr224-320x24_cv8835_best.pt': 'AB', 'f1_s1043_raw_ttr224-320x24_cv8926_best.pt': 'AB', 'f2_s1043_raw_ttr224-320x24_cv8907_best.pt': 'AB', 'f3_s1043_raw_ttr224-320x24_cv8763_best.pt': 'AB', 'f4_s1043_raw_ttr224-320x24_cv8928_best.pt': 'AB'}}   # literal, written by k2/make_k2_kernel.py (pushed kernels get no env vars)
# === K2 FLEET STAGE: appended after the 0.943 parent (repro_0946 cells 0-30) by k2/make_k2_kernel.py ===
# Port of round3/fleet2 fleet_blend_cell.py (B1-B8 kept: one deadline polled for all workers, GPUs counted,
# views/coverage gates, same model list per worker, every check inside the try, receipt) to ENDGAME_V3 sec 5 item 1:
#   P1 multi-file: submission.csv = P+F at W_MAIN, plus submission_wXX.csv for W_MAIN (its fail-loud twin) and for
#      every W in W_VARIANTS, plus submission_fleetsolo.csv (the fleet's raw-prob mean alone, parent row order).
#   P2 per-file fail-loud: the wXX files and fleetsolo exist ONLY if this stage finished cleanly with the full member
#      count on every worker (n_keep == M, no hard stop, every gate passed, every write done). Otherwise NONE of them
#      exists (stale copies are deleted first, partial writes are deleted) and submission.csv holds the parent bytes,
#      so a slot submitted on a missing file errors instead of silently scoring the parent.
#   P3 one wall: kernel start = min(/proc start of this process, parent T0); no worker starts a new chunk after
#      kernel start + WALL_H (8.6 h); workers are killed GRACE_S after it; the stage starts only if >= MIN_START_MIN
#      minutes remain (round3: 0.35 h margin + 30 min, which threw away a feasible fleet at P 8.05-8.15 h). The wall
#      is the only time guard (P6: never cut a fleet that could still finish before it). Only if NO kernel clock is
#      readable (no /proc start AND no parent T0; not expected, the pinned parent sets T0 in cell 5) the stage gets
#      NOCLOCK_MIN minutes from its own start: a parent at its 8.0 h TIME_BUDGET + 40 min + GRACE_S stays < 9 h.
#   P4 W and every knob are literals (K2_CFG line written above this header by the builder; no env vars on Kaggle).
#   P5 commit-only microbench (<= MICROBENCH_MAX_N test studies): FLEET_MICROBENCH forwards per member per GPU in
#      the worker; ms/member/study and decode s/study go to the log and fleet_receipt.json (G-T inputs).
#   P6 the worker's time valve is disabled (budget 1e12): a valve-cut fleet can never be written under P2, so a cut
#      would only throw away a fleet that might still finish before the wall.
#   P7 sanity gate as in the crop cell: parent~fleet per-label Spearman mean >= MIN_RHO_MEAN on >= RHO_MIN_N studies.
#   P8 hand-off for the optional crop stage: parent bytes -> _k2_parent.csv, status -> _k2_stage_fleet.json.
# K3 patch (scratchpad/k3; blend and solo outputs unchanged when the two new options are unset):
#   K3a MAIN_MODE 'strict': submission.csv = the P+F blend at W_MAIN exactly as in 'blend', but on ANY failure path it
#       is DELETED (as in 'solo'), never restored to the parent bytes: a strict slot errors instead of scoring 0.943.
#       The outer last-defence handler is fail-loud for 'solo' and 'strict' alike (it used to restore the parent).
#   K3b MEMBERS: None = every checkpoint in fleet_manifest.json (as before); a list of manifest file names = run ONLY
#       those. Applied before any worker starts: the workers get a private FLEET_DIR (tempfile dir, outside
#       /kaggle/working) that links every dataset file except the unselected checkpoints and holds a filtered
#       manifest, so the worker's own checks run unchanged on the selected members. A name not in the manifest fails
#       loud (wrong dataset version attached); the workers must report exactly the selected names, and the
#       full-member-count gate uses that filtered count.
#   K3c log + receipt the manifest's per-member sha256[:16] ('sources', written by the bigjump pack tool): a trainer
#       re-saves its best file under the SAME name when a later epoch improves cv in the 5th digit, so only the bytes
#       prove which epoch the commit used (k2_check.py --expect_manifest compares them with the local build).
import os, sys, json, time, shutil, subprocess, hashlib, filecmp, traceback
import numpy as np, pandas as pd

K2 = {'W_MAIN': 0.30, 'W_VARIANTS': [0.20, 0.40], 'WALL_H': 8.6, 'MIN_START_MIN': 20.0, 'NOCLOCK_MIN': 40.0,
      'GRACE_S': 300, 'MIN_COVER': 0.98, 'MIN_VIEWS': 3.0, 'MIN_RHO_MEAN': 0.30, 'RHO_MIN_N': 50, 'GATE_MIN_N': 11,
      'MICROBENCH': 5, 'MICROBENCH_MAX_N': 3, 'MAX_GPUS': 2,
      'MAIN_MODE': 'blend',   # 'solo': submission.csv = fleet alone, DELETED (fail-loud) if the stage is not clean
      #                         'strict' (K3a): submission.csv = P+F at W_MAIN, DELETED (fail-loud) if not clean
      'MEMBERS': None}        # K3b: None = all manifest members; else the list of checkpoint file names to run
K2.update(globals().get('K2_CFG') or {})
_K2_FLEET_DIR = None
for _c in ('/kaggle/input/datasets/eeshsaxena/rsna-knee-cnxt-fleet', '/kaggle/input/rsna-knee-cnxt-fleet'):
    if os.path.exists(_c + '/fleet_worker.py'):
        _K2_FLEET_DIR = _c; break
_K2_WORK = '/kaggle/working'
_K2_SUB = f'{_K2_WORK}/submission.csv'
_K2_PAR = f'{_K2_WORK}/_k2_parent.csv'              # parent bytes, taken before anything else (also for the crop stage)
_K2_STATUS = f'{_K2_WORK}/_k2_stage_fleet.json'
_K2_TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
               'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def _k2_log(msg):
    print(f'[k2-fleet {time.strftime("%H:%M:%S")}] {msg}', flush=True)


def _k2_wname(w):
    return f'submission_w{int(round(float(w) * 100)):02d}.csv'


def _k2_loud_names():
    """Fail-loud files of this stage, in write order (fleetsolo first; submission.csv is written after all of them)."""
    names = ['submission_fleetsolo.csv']
    for w in [K2['W_MAIN']] + list(K2['W_VARIANTS']):
        if _k2_wname(w) not in names:
            names.append(_k2_wname(w))
    return names


class _K2Fail(Exception):
    """The stage did not finish cleanly: no fail-loud file, submission.csv = parent bytes."""


def _k2_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def _k2_copy_atomic(src, dst):
    tmp = dst + '.tmp'; shutil.copyfile(src, tmp); os.replace(tmp, dst)


def _k2_write_atomic(df, dst):
    tmp = dst + '.tmp'; df.to_csv(tmp, index=False); os.replace(tmp, dst)


def _k2_remove(names):
    errs = []
    for f in names:
        for p in (f'{_K2_WORK}/{f}', f'{_K2_WORK}/{f}.tmp'):
            try:
                if os.path.lexists(p):
                    os.remove(p)
            except BaseException as e:
                errs.append(f'{os.path.basename(p)}: {e!r}')
    return errs


def _k2_kernel_start():
    """Earliest of this process's start (/proc) and the parent's T0 (repro_0946 cell 5: T0 = time.time())."""
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


def _k2_rank(a):
    return pd.DataFrame(np.asarray(a, np.float64)).rank(method='average', pct=True).to_numpy()


def _k2_loud_main():
    """K3a: main modes in which submission.csv must never hold the parent bytes after a failure."""
    return K2.get('MAIN_MODE') in ('solo', 'strict')


def _k2_delete_main():
    """K3a: delete submission.csv (fail-loud); if the delete itself fails, truncate it (an empty file cannot score)."""
    try:
        if os.path.lexists(_K2_SUB):
            os.remove(_K2_SUB)
        return 'deleted'
    except BaseException as e:
        try:
            open(_K2_SUB, 'w').close()
            return f'delete failed ({e!r}), truncated to 0 bytes'
        except BaseException as e2:
            return f'DELETE AND TRUNCATE FAILED {e!r} {e2!r}'


def _k2_members_want(man):
    """K3b: validate K2['MEMBERS'] against the attached manifest; returns the selected names in the given order."""
    want, full = K2.get('MEMBERS'), list(man.get('checkpoints') or [])
    if isinstance(want, str) or not isinstance(want, (list, tuple)) or not want \
            or not all(isinstance(x, str) and x for x in want):
        raise _K2Fail(f'MEMBERS must be None or a non-empty list of checkpoint file names, got {want!r}')
    want = list(want)
    if len(set(want)) != len(want):
        raise _K2Fail(f'MEMBERS has duplicate names {want}')
    unknown = [x for x in want if x not in full]
    if unknown:
        raise _K2Fail(f'MEMBERS {unknown} not in the fleet manifest {full} (wrong dataset version attached?)')
    return want


def _k2_member_view(man, want, rec):
    """K3b: private FLEET_DIR holding links to every dataset file except the unselected checkpoints + a filtered
    manifest. Outside /kaggle/working, so the kernel output gets no extra files; removed after the stage."""
    import tempfile
    d = tempfile.mkdtemp(prefix='k2_members_')
    rec['member_dir'] = d
    keep = set(want)
    drop = set(man.get('checkpoints') or []) - keep
    for f in sorted(os.listdir(_K2_FLEET_DIR)):
        src, dst = os.path.join(_K2_FLEET_DIR, f), os.path.join(d, f)
        if f == 'fleet_manifest.json' or f in drop or os.path.isdir(src):
            continue
        try:
            os.symlink(src, dst)
        except OSError:                                                  # no symlink right (Windows test hosts)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
    m2 = {'checkpoints': [c for c in man['checkpoints'] if c in keep], 'member_filter': want,
          'manifest_full': list(man['checkpoints'])}
    if isinstance(man.get('sources'), dict):
        m2['sources'] = {k: v for k, v in man['sources'].items() if k in keep}
    with open(os.path.join(d, 'fleet_manifest.json'), 'w') as fh:
        json.dump(m2, fh, indent=1)
    return d


# ---- K5a begin (scratchpad/k5; additive: this block + the one call line marked K5a; outputs unchanged when no member
# is 'AB'). K5a = per-member test-time augmentation (bigjump B6): fleet_manifest.json sources[<ckpt>]['tta'] in
# {'A', 'AB'} (absent = 'A'); the K5 worker serves 'AB' members as the mean of pass A and pass B (mirror_views + slice
# offset +HALF, HALF = 0.5 - 1e-3) and reports what it ran in its npz key 'tta'. Gate: every worker's modes == the
# attached manifest's, an 'AB' member ran pass B on every row it wrote and an 'A' member never did, HALF as above.
# Receipt: rec['tta'] (modes, half, rows, forwards per member per pass, microbench passes) + one 'TTA' log line.
# K5b: a kernel built with K2_CFG['TTA'] (build_k5.py) also needs the manifest's tta == that map for every member that
# ran, so a K5 kernel that mounts a dataset version without the K5 manifest (the K4 one holds the same J1 bytes) fails.
def _k2_tta(parts, man, names, rec):
    src = man.get('sources') if isinstance(man.get('sources'), dict) else {}
    want = {n: (src[n].get('tta', 'A') if isinstance(src.get(n), dict) else 'A') for n in names}
    rec['tta_expected'] = want
    if any(v not in ('A', 'AB') for v in want.values()):
        raise _K2Fail(f'manifest tta {want}: each must be A or AB')
    exp = K2.get('TTA')
    if exp is not None:
        exp_run = {n: exp.get(n) for n in names}
        rec['tta_kernel'] = exp_run
        if exp_run != want:
            raise _K2Fail(f'kernel expects TTA {exp_run} but the attached manifest says {want} '
                          f'(a dataset version without the K5 TTA manifest is mounted?)')
    rep = [json.loads(str(z['tta'])) for z in parts if 'tta' in z.files]
    if not rep:                                                          # pre-K5 worker (or a smoke stub)
        if any(v != 'A' for v in want.values()):
            raise _K2Fail(f'manifest asks TTA {want} but no worker reports TTA (pre-K5 worker in the dataset?)')
        rec['tta'] = None
        return
    if len(rep) != len(parts):
        raise _K2Fail(f'only {len(rep)} of {len(parts)} workers report TTA')
    fwd, rows = {n: {'A': 0, 'B': 0} for n in names}, 0
    for r in rep:
        if r.get('modes') != want:
            raise _K2Fail(f'worker TTA modes {r.get("modes")} != manifest {want}')
        if 'AB' in want.values() and abs(float(r.get('half', 0.0)) - (0.5 - 1e-3)) > 1e-12:
            raise _K2Fail(f'worker pass B offset {r.get("half")} != 0.5 - 1e-3')
        rows += int(r.get('rows', -1))
        for n in names:
            f = (r.get('fwd') or {}).get(n) or {}
            fwd[n]['A'] += int(f.get('A', -1)); fwd[n]['B'] += int(f.get('B', -1))
    bad = {n: fwd[n] for n in names if fwd[n]['A'] != rows or fwd[n]['B'] != (rows if want[n] == 'AB' else 0)}
    if bad:
        raise _K2Fail(f'TTA forward counts {bad} do not match {rows} rows written and modes {want}')
    rec['tta'] = dict(modes=want, half=rep[0].get('half'), rows=rows, fwd=fwd, bench_passes=rep[0].get('bench_passes'))
    _k2_log(f'TTA modes {want} | half {rep[0].get("half")} | {rows} rows | forwards per member {fwd}')
# ---- K5a end


def _k2_stage(t_cell, rec, procs):
    """Run the fleet, apply every gate, return {file name: DataFrame}. Raises on anything short of a clean stage."""
    w_main = float(K2['W_MAIN']); ws = [float(w) for w in K2['W_VARIANTS']]
    if not all(0.0 < w < 1.0 for w in [w_main] + ws):
        raise _K2Fail(f'blend weights {[w_main] + ws} outside (0, 1)')
    if K2.get('MAIN_MODE') not in ('blend', 'solo', 'strict'):
        raise _K2Fail(f'unknown MAIN_MODE {K2.get("MAIN_MODE")!r}')
    pub = pd.read_csv(_K2_PAR, dtype={'StudyInstanceUID': str})
    if pub.columns.tolist() != ['StudyInstanceUID', *_K2_TARGETS]:
        raise _K2Fail(f'parent columns {pub.columns.tolist()}')
    if not pub['StudyInstanceUID'].is_unique:
        raise _K2Fail('parent has duplicate study ids')
    if not np.isfinite(pub[_K2_TARGETS].to_numpy(np.float64)).all():
        raise _K2Fail('parent has non-finite values')
    n = len(pub)
    if not _K2_FLEET_DIR:
        raise _K2Fail('fleet dataset not attached (no fleet_worker.py under /kaggle/input/.../rsna-knee-cnxt-fleet)')
    man = json.load(open(f'{_K2_FLEET_DIR}/fleet_manifest.json'))
    rec['manifest'] = man.get('checkpoints')
    if isinstance(man.get('sources'), dict):                            # K3c: which BYTES are attached (names can
        rec['manifest_sha16'] = {k: (v or {}).get('sha256') for k, v in man['sources'].items()}   # repeat across epochs)
        _k2_log(f'manifest sha256[:16] {rec["manifest_sha16"]}')
    fleet_dir, want = _K2_FLEET_DIR, None
    if K2.get('MEMBERS') is not None:                                   # K3b, before any worker starts
        want = _k2_members_want(man)
        rec['members_filter'] = want
        fleet_dir = _k2_member_view(man, want, rec)
        _k2_log(f'member filter: running {want} of the manifest {rec["manifest"]}')
    # ---- P3 one wall
    k0 = _k2_kernel_start(); now = time.time()
    if k0 is not None:
        deadline = k0 + float(K2['WALL_H']) * 3600                      # the wall is the guard (P6)
    else:
        deadline = t_cell + float(K2['NOCLOCK_MIN']) * 60               # no clock at all: short backstop (P3)
    left = deadline - now
    rec.update(kernel_age_h=None if k0 is None else (now - k0) / 3600, left_min=left / 60)
    _k2_log(f'kernel age {"?" if k0 is None else f"{(now - k0) / 3600:.2f} h"}; {left / 60:.1f} min to the '
            f'{K2["WALL_H"]} h wall; {n} studies; members {rec["manifest"]}')
    if left < float(K2['MIN_START_MIN']) * 60:
        raise _K2Fail(f'only {left / 60:.1f} min before the wall (< {K2["MIN_START_MIN"]} min): stage not started')
    # ---- B2 GPUs counted
    import gc; gc.collect()
    import torch
    n_dev = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_dev:
        torch.cuda.empty_cache()
    vis = [v.strip() for v in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if v.strip()]
    gpus = (vis or [str(i) for i in range(n_dev)])[:n_dev][:int(K2['MAX_GPUS'])]
    if not gpus:
        raise _K2Fail('no GPU visible')
    try:
        import psutil
        _k2_log(f'host RAM available {psutil.virtual_memory().available / 2**30:.1f} GB, gpus {gpus}')
    except Exception:
        pass
    bench = int(K2['MICROBENCH']) if n <= int(K2['MICROBENCH_MAX_N']) else 0
    rec.update(gpus=gpus, microbench_forwards=bench)
    t_launch = time.time()
    for g, gid in enumerate(gpus):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gid, 'FLEET_DIR': fleet_dir,                     # K3b
               'FLEET_SHARD': str(g), 'FLEET_NSHARD': str(len(gpus)), 'FLEET_BUDGET_S': '1e12',            # P6
               'FLEET_DEADLINE': str(deadline), 'FLEET_OUT': f'{_K2_WORK}/_fleet_{g}.npz',
               'FLEET_MICROBENCH': str(bench),
               'PYTHONPATH': fleet_dir + os.pathsep + os.environ.get('PYTHONPATH', '')}              # B8
        env.pop('FLEET_ALLOW_CPU', None)                                 # W3: never a CPU fleet
        try:
            os.remove(env['FLEET_OUT'])
        except FileNotFoundError:
            pass
        procs.append(subprocess.Popen([sys.executable, f'{fleet_dir}/fleet_worker.py'], env=env))
    while True:                                                          # B1 poll every worker together
        rcs = [p.poll() for p in procs]
        if any(r not in (None, 0) for r in rcs):
            raise _K2Fail(f'worker rc {rcs}')
        if all(r == 0 for r in rcs):
            break
        if time.time() > deadline + float(K2['GRACE_S']):
            raise _K2Fail(f'workers still running {K2["GRACE_S"]} s after the wall, rc {rcs}')
        time.sleep(2)
    rec['workers_s'] = time.time() - t_launch
    parts = [np.load(f'{_K2_WORK}/_fleet_{g}.npz', allow_pickle=False) for g in range(len(procs))]
    names = [str(x) for x in parts[0]['names']]
    for z in parts[1:]:
        if [str(x) for x in z['names']] != names:
            raise _K2Fail('workers loaded different model lists')           # B5
    if want is not None and sorted(names) != sorted(want):              # K3b
        raise _K2Fail(f'workers ran {names}, the member filter selected {want}')
    M = len(want) if want is not None else len(names)                   # K3b: the gate counts the filtered members
    keeps = [int(z['n_keep']) for z in parts]
    stops = [bool(z['hard_stopped']) for z in parts]
    rec.update(models=M, names=names, n_keep=keeps, hard_stopped=stops,
               worker_min=[float(z['minutes']) for z in parts],
               worker_loop_min=[float(z['loop_min']) for z in parts if 'loop_min' in z.files],
               n_fail=[int(z['n_fail']) for z in parts], n_blank=[int(z['n_blank']) for z in parts],
               parity=[json.loads(str(z['parity'])) for z in parts],
               bench=[json.loads(str(z['bench'])) for z in parts if 'bench' in z.files])
    stats = {}
    for z in parts:
        for k, v in json.loads(str(z['stats'])).items():
            stats[k] = stats.get(k, 0) + v
    rec['selection'] = stats
    if 'decode_s' in stats and n:
        rec['decode_s_per_study_per_thread'] = stats['decode_s'] / n
    if any(k != M for k in keeps):
        raise _K2Fail(f'full member count not reached: kept {keeps} of {M}')                       # P2
    if any(stops):
        raise _K2Fail(f'a worker hit the wall before the last study: hard_stopped {stops}')        # P2
    _k2_tta(parts, man, names, rec)                                     # K5a
    P = np.concatenate([z['P'] for z in parts], axis=1)                 # (M, n, 12)
    NV = np.concatenate([z['NV'] for z in parts])
    ids = np.concatenate([z['sids'] for z in parts]).astype(str)
    if sorted(ids.tolist()) != sorted(pub['StudyInstanceUID'].tolist()):
        raise _K2Fail(f'worker study ids != parent ids ({len(ids)} vs {n})')
    vps = float(NV[NV >= 0].mean()) if (NV >= 0).any() else 0.0
    study_ok = np.isfinite(P).all(axis=(0, 2)) & (NV > 0)
    rec.update(views_per_study=vps, coverage=float(study_ok.mean()), median_filled=int((~study_ok).sum()))
    if n >= int(K2['GATE_MIN_N']) and vps < float(K2['MIN_VIEWS']):
        raise _K2Fail(f'views/study {vps:.2f} < {K2["MIN_VIEWS"]}')                                 # B4
    if n >= int(K2['GATE_MIN_N']) and not study_ok.mean() > float(K2['MIN_COVER']):
        raise _K2Fail(f'failed studies {1 - study_ok.mean():.3f} > {1 - float(K2["MIN_COVER"]):.2f}')
    if not study_ok.any():
        raise _K2Fail('no study produced a fleet prediction')
    mean = P.astype(np.float64).mean(0)                                 # raw-prob mean over members
    mean[~study_ok] = np.nan
    med = np.nanmedian(mean, axis=0)
    mean = np.where(np.isfinite(mean), mean, med[None])                 # <2% failed -> column median (rank-neutral)
    fleet = pd.DataFrame(mean, columns=_K2_TARGETS, index=ids).reindex(pub['StudyInstanceUID'])
    F = fleet.to_numpy(np.float64)
    if not (np.isfinite(F).all() and (F >= 0).all() and (F <= 1).all()):
        raise _K2Fail('fleet not finite / outside [0, 1]')
    if n >= int(K2['GATE_MIN_N']) and float(F.std(0).min()) < 1e-9:
        raise _K2Fail('constant fleet column')
    r_pub = _k2_rank(pub[_K2_TARGETS].to_numpy(np.float64)); r_fl = _k2_rank(F)
    rho = []
    for j in range(12):
        c = np.corrcoef(r_pub[:, j], r_fl[:, j])[0, 1] if n > 2 else np.nan
        rho.append(float(c) if np.isfinite(c) else float('nan'))
    rho_mean = float(np.nanmean(rho)) if np.isfinite(rho).any() else float('nan')
    rec.update(rho={t: round(r, 4) for t, r in zip(_K2_TARGETS, rho)}, rho_mean=rho_mean)
    if n >= int(K2['RHO_MIN_N']) and not rho_mean >= float(K2['MIN_RHO_MEAN']):
        raise _K2Fail(f'parent~fleet Spearman mean {rho_mean:.3f} < {K2["MIN_RHO_MEAN"]} (label order / decode bug?)')
    out = {}
    solo = pd.DataFrame(F, columns=_K2_TARGETS); solo.insert(0, 'StudyInstanceUID', pub['StudyInstanceUID'].values)
    out['submission_fleetsolo.csv'] = solo
    for w in [w_main] + ws:
        b = _k2_rank((1.0 - w) * r_pub + w * r_fl)
        df = pd.DataFrame(b, columns=_K2_TARGETS); df.insert(0, 'StudyInstanceUID', pub['StudyInstanceUID'].values)
        out[_k2_wname(w)] = df
    out['submission.csv'] = out[_k2_wname(w_main)].copy()
    if K2.get('MAIN_MODE') == 'solo':                                   # Kaggle scores only submission.csv
        out['submission.csv'] = out['submission_fleetsolo.csv'].copy()
    for f, df in out.items():                                            # final schema / order / range check
        v = df[_K2_TARGETS].to_numpy(np.float64)
        if not (len(df) == n and df.columns.tolist() == pub.columns.tolist()
                and (df['StudyInstanceUID'].values == pub['StudyInstanceUID'].values).all()
                and np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all()):
            raise _K2Fail(f'{f}: failed the final schema/order/finite/range check')
    assert sorted(out) == sorted(_k2_loud_names() + ['submission.csv'])
    return out


def _k2_entry_inner():
    t_cell = time.time()
    loud = _k2_loud_names()
    rec = {'config': dict(K2), 'fleet_dir': _K2_FLEET_DIR, 'fail_loud_files': loud}
    if K2.get('MAIN_MODE') == 'strict':                                 # K3a
        _k2_log(f'STAGE fleet start; fail-loud files {loud}; STRICT: submission.csv = P+F at W {K2["W_MAIN"]} '
                f'(DELETED on any failure); members {K2.get("MEMBERS") or "all in the manifest"}')
    elif K2.get('MAIN_MODE') == 'solo':                                 # K3: log text only (solo outputs unchanged)
        _k2_log(f'STAGE fleet start; fail-loud files {loud}; SOLO: submission.csv = the fleet alone (DELETED on any '
                f'failure); members {K2.get("MEMBERS") or "all in the manifest"}')
    else:
        _k2_log(f'STAGE fleet start; fail-loud files {loud}; submission.csv = P+F at W {K2["W_MAIN"]} (parent on failure)')
    errs = _k2_remove(loud + ['_k2_stage_fleet.json'])                   # P2: a fresh session has none; re-runs might
    if errs:
        _k2_log(f'WARNING could not remove stale files {errs}')
    ok, why, procs = False, 'not attempted', []
    if not os.path.isfile(_K2_SUB):
        why = 'parent wrote no submission.csv: nothing to blend, no file written'
    else:
        try:
            _k2_copy_atomic(_K2_SUB, _K2_PAR)
        except BaseException as e:
            why = f'parent copy failed ({e!r}): stage skipped, submission.csv untouched'
        else:
            try:
                if errs:
                    raise _K2Fail(f'stale fail-loud files could not be removed: {errs}')
                outputs = _k2_stage(t_cell, rec, procs)
                for f in loud + ['submission.csv']:                      # submission.csv LAST
                    _k2_write_atomic(outputs[f], f'{_K2_WORK}/{f}')
                rec['files'] = {f: _k2_sha(f'{_K2_WORK}/{f}') for f in loud + ['submission.csv']}
                ok = True
                why = (f'{rec["models"]} member(s) {rec["names"]}, {len(outputs["submission.csv"])} studies, '
                       f'{rec["median_filled"]} median-filled, views/study {rec["views_per_study"]:.2f}, '
                       f'rho mean {rec["rho_mean"]:.3f}')
            except BaseException as e:                                   # incl. KeyboardInterrupt
                if not isinstance(e, _K2Fail):
                    traceback.print_exc()
                why = f'{type(e).__name__}: {e}'
                rm = _k2_remove(loud)                                    # P2 all-or-nothing for the fail-loud set
                if rm:
                    _k2_log(f'WARNING could not delete {rm}')
                    rec['not_deleted'] = rm
                try:
                    if K2.get('MAIN_MODE') != 'strict' and \
                            not (os.path.isfile(_K2_SUB) and filecmp.cmp(_K2_SUB, _K2_PAR, shallow=False)):
                        _k2_copy_atomic(_K2_PAR, _K2_SUB)                # K3a: strict never restores the parent
                except BaseException as e2:                              # tmp + os.replace: submission.csv is whole
                    _k2_log(f'RESTORE FAILED {e2!r}')
            finally:
                for p in procs:                                          # B1 never leave a worker running
                    if p.poll() is None:
                        p.kill()
                        try:
                            p.wait(timeout=30)
                        except Exception:
                            pass
                if rec.get('member_dir'):                                # K3b private FLEET_DIR
                    shutil.rmtree(rec['member_dir'], ignore_errors=True)
    rec.update(clean=ok, why=why, stage_min=(time.time() - t_cell) / 60)
    try:
        st = {k: rec.get(k) for k in ('clean', 'why', 'files', 'models', 'names', 'stage_min')}
        st.update(W_MAIN=K2['W_MAIN'], W_VARIANTS=K2['W_VARIANTS'], parent='_k2_parent.csv',
                  fleet_solo='submission_fleetsolo.csv' if ok else None,
                  main_mode=K2.get('MAIN_MODE'), members_filter=K2.get('MEMBERS'))   # K3a/K3b (read by k2_check)
        tmp = _K2_STATUS + '.tmp'
        json.dump(st, open(tmp, 'w'), indent=1, default=str); os.replace(tmp, _K2_STATUS)
    except Exception as e:
        _k2_log(f'status write error (the crop stage will treat the fleet as not clean): {e!r}')
    if not ok and K2.get('MAIN_MODE') == 'strict':                      # K3a: never let a strict slot score the parent
        _k2_log(f'STRICT MODE: stage not clean -> submission.csv {_k2_delete_main()} '
                f'(the submission will error, not score the parent)')
    for f in ['submission.csv'] + loud:                                  # one line per output file
        p = f'{_K2_WORK}/{f}'
        if not os.path.isfile(p):
            print(f'[k2-fleet] {f:<27} ABSENT', flush=True); continue
        try:
            d = pd.read_csv(p, dtype={'StudyInstanceUID': str})
            same = os.path.isfile(_K2_PAR) and filecmp.cmp(p, _K2_PAR, shallow=False)
            print(f'[k2-fleet] {f:<27} rows={len(d)} nan={int(d.iloc[:, 1:].isna().sum().sum())}'
                  f'{"  (byte-identical to parent)" if same else ""}', flush=True)
        except Exception as e:
            print(f'[k2-fleet] {f:<27} UNREADABLE {e!r}', flush=True)
    b = [x for x in rec.get('bench', []) if x]
    if b:
        _k2_log(f'MICROBENCH ms/member/study per GPU {b}; decode {rec.get("decode_s_per_study_per_thread", float("nan")):.2f} '
                f's/study per thread; workers {rec.get("workers_s", float("nan")):.1f} s')
    _k2_log(f'STAGE fleet end {rec["stage_min"]:.2f} min')
    if not ok and K2.get('MAIN_MODE') == 'solo':                        # never let a solo read score the parent
        try:
            if os.path.isfile(_K2_SUB):
                os.remove(_K2_SUB)
            _k2_log('SOLO MODE: stage not clean -> submission.csv DELETED (the submission will error, not score the parent)')
        except BaseException as e3:
            _k2_log(f'SOLO MODE delete failed {e3!r}')
    if K2.get('MAIN_MODE') in ('strict', 'solo'):                       # K3: log text only
        _k2_log(('RESULT OK fleet stage clean: ' if ok else
                 f'RESULT FLEET STAGE NOT CLEAN ({K2["MAIN_MODE"].upper()}: submission.csv DELETED; wXX/fleetsolo NOT '
                 f'written): ') + why)
    else:
        _k2_log(('RESULT OK fleet stage clean: ' if ok else
                 'RESULT FLEET STAGE NOT CLEAN (submission.csv = parent bytes; wXX/fleetsolo NOT written): ') + why)
    try:
        print('[k2-fleet] receipt', json.dumps(rec, default=str)[:3000], flush=True)
        json.dump(rec, open(f'{_K2_WORK}/fleet_receipt.json', 'w'), indent=1, default=str)
    except Exception as e:
        _k2_log(f'receipt write error (outputs unaffected): {e!r}')
    return rec


def _k2_entry():
    """Last line of defence: an exception escaping the notebook's cell fails the whole submission."""
    try:
        return _k2_entry_inner()
    except BaseException as e:
        dm = _k2_delete_main() if _k2_loud_main() else None             # K3a: delete FIRST (a raising log cannot skip it)
        try:
            traceback.print_exc(); _k2_log(f'UNEXPECTED {e!r} outside the guarded path')
            _k2_remove(_k2_loud_names())
            if _k2_loud_main():                                          # K3a: solo / strict never score the parent
                _k2_log(f'{K2.get("MAIN_MODE")} mode: submission.csv {dm}')
            elif os.path.isfile(_K2_PAR) and not filecmp.cmp(_K2_SUB, _K2_PAR, shallow=False):
                _k2_copy_atomic(_K2_PAR, _K2_SUB)
        except BaseException:
            pass
        return None


_k2_receipt = _k2_entry()
