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
import os, sys, json, time, shutil, subprocess, hashlib, filecmp, traceback
import numpy as np, pandas as pd

K2 = {'W_MAIN': 0.30, 'W_VARIANTS': [0.20, 0.40], 'WALL_H': 8.6, 'MIN_START_MIN': 20.0, 'NOCLOCK_MIN': 40.0,
      'GRACE_S': 300, 'MIN_COVER': 0.98, 'MIN_VIEWS': 3.0, 'MIN_RHO_MEAN': 0.30, 'RHO_MIN_N': 50, 'GATE_MIN_N': 11,
      'MICROBENCH': 5, 'MICROBENCH_MAX_N': 3, 'MAX_GPUS': 2,
      'MAIN_MODE': 'blend'}   # 'solo': submission.csv = fleet alone, DELETED (fail-loud) if the stage is not clean
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


def _k2_stage(t_cell, rec, procs):
    """Run the fleet, apply every gate, return {file name: DataFrame}. Raises on anything short of a clean stage."""
    w_main = float(K2['W_MAIN']); ws = [float(w) for w in K2['W_VARIANTS']]
    if not all(0.0 < w < 1.0 for w in [w_main] + ws):
        raise _K2Fail(f'blend weights {[w_main] + ws} outside (0, 1)')
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
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gid, 'FLEET_DIR': _K2_FLEET_DIR,
               'FLEET_SHARD': str(g), 'FLEET_NSHARD': str(len(gpus)), 'FLEET_BUDGET_S': '1e12',            # P6
               'FLEET_DEADLINE': str(deadline), 'FLEET_OUT': f'{_K2_WORK}/_fleet_{g}.npz',
               'FLEET_MICROBENCH': str(bench),
               'PYTHONPATH': _K2_FLEET_DIR + os.pathsep + os.environ.get('PYTHONPATH', '')}          # B8
        env.pop('FLEET_ALLOW_CPU', None)                                 # W3: never a CPU fleet
        try:
            os.remove(env['FLEET_OUT'])
        except FileNotFoundError:
            pass
        procs.append(subprocess.Popen([sys.executable, f'{_K2_FLEET_DIR}/fleet_worker.py'], env=env))
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
    M = len(names)
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
                    if not (os.path.isfile(_K2_SUB) and filecmp.cmp(_K2_SUB, _K2_PAR, shallow=False)):
                        _k2_copy_atomic(_K2_PAR, _K2_SUB)
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
    rec.update(clean=ok, why=why, stage_min=(time.time() - t_cell) / 60)
    try:
        st = {k: rec.get(k) for k in ('clean', 'why', 'files', 'models', 'names', 'stage_min')}
        st.update(W_MAIN=K2['W_MAIN'], W_VARIANTS=K2['W_VARIANTS'], parent='_k2_parent.csv',
                  fleet_solo='submission_fleetsolo.csv' if ok else None)
        tmp = _K2_STATUS + '.tmp'
        json.dump(st, open(tmp, 'w'), indent=1, default=str); os.replace(tmp, _K2_STATUS)
    except Exception as e:
        _k2_log(f'status write error (the crop stage will treat the fleet as not clean): {e!r}')
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
        try:
            traceback.print_exc(); _k2_log(f'UNEXPECTED {e!r} outside the guarded path')
            _k2_remove(_k2_loud_names())
            if os.path.isfile(_K2_PAR) and not filecmp.cmp(_K2_SUB, _K2_PAR, shallow=False):
                _k2_copy_atomic(_K2_PAR, _K2_SUB)
        except BaseException:
            pass
        return None


_k2_receipt = _k2_entry()
