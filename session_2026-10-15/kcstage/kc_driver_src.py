#!/usr/bin/env python
# kc_driver.py (10-15, Claude subagent, kcstage; written to /tmp/kc/kc_driver.py by the KC SETUP cell of build_v224s6fkc.py).
# The KC stage of the S6F+KC kernel, run as ONE process in its own session by the KC RUN cell (PREREG_10-15 Amendment 6 G3):
#   1 env     a fresh interpreter with the KC env reports versions / module dirs / GPUs (no CUDA in this process: it forks below)
#   2 inputs  code pins (md5 / sha) of /tmp/kc/code, member files, the test root (as the Raptor / side kernels find it)
#   3 side    the side kernel's OWN study() (kc_side_kernel.py sha 2979c191, imported from /tmp/kc/code) on every test study,
#             fork Pool(SIDE_PROCS), one study per task; a study whose study() raises gets a side_err row (-> unscored);
#             -> OUT/kc_side_test.csv (the side kernel's columns and row order) + OUT/kc_side_summary.json
#   4 serve   one kc_serve.py process per GPU (its own session): --device cuda:K --shard K/N --procs P (kc_worker's own __main__,
#             study_inputs fed by the side table); a worker still running at the stop time is SIGKILLed with its group
#   5 merge   parts -> OUT/kc_probs.csv (every test id once, test.csv order; a missing / failed study = NaN + kc_err), coverage
#   6 dump    commit only (N test studies <= DUMP_MAX): per study the d96 volume, the side row, flips, the views and the served
#             probabilities -> OUT/dump/<sid>.npz, for the box CPU parity run (kc_box_parity.py); vol sha == the served kc_vol_sha
# Never writes submission.csv. Exit 0 with receipt ok whenever kc_probs.csv was written (coverage is the blend cell's call), else 3.
#   usage: python kc_driver.py --stop-at EPOCH --nb-t0 EPOCH --members m1.pt m2.pt [--out DIR] [--device cuda|cpu]
import os, sys, json, time, glob, hashlib, argparse, subprocess, traceback

# test-only overrides (box / laptop harness); on Kaggle none is set and the gate requires the receipt's 'overrides' == {}
OVR = {k: os.environ.get(k) for k in ('KC_TMP_ROOT', 'KC_OUT_DIR', 'KC_TEST_ROOT', 'KC_DEVICE', 'KC_WORKERS', 'KC_WORKER_THREADS',
                                      'KC_DUMP_MAX')}
ROOT_TMP = OVR['KC_TMP_ROOT'] or '/tmp/kc'
CODE = ROOT_TMP + '/code'
PINS_MD5 = {'kc_worker.py': '4af60f6ff7b94b3acab9dceb52162a18', 'kc_common.py': '45255ae0e361ef787b290d7510554a9d',
            'fleet_model.py': '897f83b272fbb144a4d3deab773d9064', 'fleet_model_ttr.py': 'bcdf7d153f339962b97d2736d8baed08'}
PINS_SHA = {'d96_reader_src.py': '30aa72fd972e29cf42c752aa5394a6c88a175f17b98740b1f6b0e0f55553ac40',
            'kc_side_kernel.py': '2979c1915e0cecf2c4792526a5581f93a723fac368eacff18849661dc0bdcb62'}
CONTRACT_HASH = '48a46a0d63ecfc59'
TEST_ROOTS = ('/kaggle/input/competitions/rsna-knee-abnormality-detection', '/kaggle/input/rsna-knee-abnormality-detection')
PROBE = r'''
import os, sys, json
r = {'python': sys.version.split()[0], 'executable': sys.executable, 'PYTHONPATH': os.environ.get('PYTHONPATH'), 'sys_path': sys.path,
     'mods': {}}
for m in ('torch', 'timm', 'pydicom', 'numpy', 'pandas', 'cv2', 'pylibjpeg', 'libjpeg', 'openjpeg', 'kc_common', 'kc_worker'):
    try:
        mod = __import__(m)
        r['mods'][m] = [str(getattr(mod, '__version__', '')), os.path.dirname(os.path.dirname(os.path.abspath(mod.__file__)))
                        if m not in ('kc_common', 'kc_worker') else os.path.dirname(os.path.abspath(mod.__file__))]
    except Exception as e:
        r['mods'][m] = ['MISSING ' + type(e).__name__, None]
try:
    import torch
    r['cuda'] = torch.version.cuda
    r['n_gpu'] = torch.cuda.device_count()
    r['gpus'] = [torch.cuda.get_device_name(i) for i in range(r['n_gpu'])]
except Exception as e:
    r['n_gpu'] = 0
    r['gpu_error'] = type(e).__name__
try:
    import pydicom
    from pydicom.uid import JPEGLosslessSV1, JPEGLossless
    r['jpeg_lossless_plugins'] = {str(u): sorted(pydicom.pixels.decoders.get_decoder(u).available_plugins) for u in (JPEGLosslessSV1, JPEGLossless)}
except Exception as e:
    r['jpeg_lossless_plugins'] = 'n/a %s' % type(e).__name__
bad = [p for p in sys.path + (os.environ.get('PYTHONPATH') or '').split(os.pathsep)
       if p.startswith(('/tmp/probe', '/kaggle/temp/probe', '/kaggle/working', '/tmp/s6f/'))]
r['r7_s6f_or_working_on_path'] = bad
print('KC_PROBE ' + json.dumps(r), flush=True)
'''


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def find_test_root(override=None):
    if override:
        return override
    for c in TEST_ROOTS:
        if os.path.isfile(c + '/test.csv') and os.path.isfile(c + '/test_series.csv'):
            return c
    base = '/kaggle/input'                                              # shallow search only: never walk a DICOM tree
    for d in sorted(glob.glob(base + '/*')) + sorted(glob.glob(base + '/*/*')) + sorted(glob.glob(base + '/*/*/*')):
        if os.path.isfile(d + '/test.csv') and os.path.isfile(d + '/test_series.csv'):
            return d
    raise RuntimeError('no competition test root (test.csv + test_series.csv) under /kaggle/input')


def kill_groups(procs):
    import signal
    out = []
    for p in procs:
        try:
            g = os.getpgid(p.pid)
            os.killpg(g, signal.SIGKILL)
            out.append(g)
        except Exception:
            pass
    return out


def _jsonable(v):
    import numpy as np
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    return None if v is None else str(v)


_KS = None


def _side_init():
    import cv2
    cv2.setNumThreads(1)


def _side_one(job):
    try:
        return _KS.study(job)
    except Exception as e:
        return {'StudyInstanceUID': job[0], 'side_err': '%s: %s' % (type(e).__name__, str(e)[:200])}


def main():
    global _KS
    ap = argparse.ArgumentParser()
    ap.add_argument('--stop-at', type=float, required=True)
    ap.add_argument('--nb-t0', type=float, required=True)
    ap.add_argument('--members', nargs='+', required=True)
    ap.add_argument('--out', default=OVR['KC_OUT_DIR'] or '/kaggle/working/kc')
    ap.add_argument('--tmp', default=ROOT_TMP + '/work')
    ap.add_argument('--root', default=OVR['KC_TEST_ROOT'] or '')
    ap.add_argument('--device', default=OVR['KC_DEVICE'] or 'cuda', choices=('cuda', 'cpu'))
    ap.add_argument('--workers', type=int, default=int(OVR['KC_WORKERS'] or 0), help='0 = one per GPU (cuda) / 2 (cpu tests)')
    ap.add_argument('--procs', type=int, default=2)
    ap.add_argument('--side-procs', type=int, default=4)
    ap.add_argument('--dump-max', type=int, default=int(OVR['KC_DUMP_MAX'] or 8))
    ap.add_argument('--worker-threads', type=int, default=int(OVR['KC_WORKER_THREADS'] or 1))
    A = ap.parse_args()
    T0 = time.time()
    os.makedirs(A.out, exist_ok=True)
    os.makedirs(A.tmp, exist_ok=True)
    rec = {'ok': False, 'error': None, 'stages': {}, 'start_age_h': round((T0 - A.nb_t0) / 3600.0, 4),
           'stop_at_age_h': round((A.stop_at - A.nb_t0) / 3600.0, 4), 'device': A.device, 'members': A.members,
           'overrides': {k: v for k, v in OVR.items() if v}, 'argv': sys.argv[1:], 'pid': os.getpid()}
    rpath = os.path.join(A.out, 'kc_driver_receipt.json')

    def save():
        rec['total_seconds'] = round(time.time() - T0, 2)
        rec['end_age_h'] = round((time.time() - A.nb_t0) / 3600.0, 4)
        with open(rpath + '.tmp', 'w') as f:
            json.dump(rec, f, indent=1, default=str)
        os.replace(rpath + '.tmp', rpath)

    def stage(name, t):
        rec['stages'][name] = {'seconds': round(time.time() - t, 2)}
        save()

    try:
        # ---- 1. env (fresh interpreter, the KC env) ----
        t = time.time()
        p = subprocess.run([sys.executable, '-c', PROBE], capture_output=True, text=True, timeout=600, cwd=A.tmp)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith('KC_PROBE ')), None)
        rec['env'] = json.loads(line[len('KC_PROBE '):]) if line else {'error': (p.stderr or '')[-600:]}
        rec['env']['driver_python'] = sys.executable
        rec['env']['driver_sys_path'] = sys.path
        stage('env', t)
        if rec['env'].get('r7_s6f_or_working_on_path'):
            raise RuntimeError('an R7 / S6F / working dir is on the KC import path: %s' % rec['env']['r7_s6f_or_working_on_path'])
        # ---- 2. inputs ----
        t = time.time()
        pins = {f: md5(os.path.join(CODE, f)) for f in PINS_MD5}
        pins.update({f: sha256(os.path.join(CODE, f)) for f in PINS_SHA})
        rec['inputs'] = {'code_pins': pins}
        bad = [f for f in pins if pins[f] != {**PINS_MD5, **PINS_SHA}[f]]
        if bad:
            raise RuntimeError('KC code files differ from the frozen pins: %s' % bad)
        for m in A.members:
            if not os.path.isfile(m):
                raise RuntimeError('member missing: %s' % m)
        rec['inputs']['member_bytes'] = {os.path.basename(m): os.path.getsize(m) for m in A.members}
        root = find_test_root(A.root)
        sdir = root + '/test_series'
        if not os.path.isdir(sdir):
            raise RuntimeError('no test_series dir under %s (kc_worker reads <root>/test_series)' % root)
        sys.path.insert(0, CODE)
        import numpy as np, pandas as pd
        ids = pd.read_csv(root + '/test.csv', dtype={'StudyInstanceUID': str})['StudyInstanceUID'].astype(str).tolist()
        ser = pd.read_csv(root + '/test_series.csv')                    # default dtypes, as the corpus builder / side kernel
        ser['StudyInstanceUID'] = ser['StudyInstanceUID'].astype(str)
        ser['SeriesInstanceUID'] = ser['SeriesInstanceUID'].astype(str)
        SER = {k: v.to_dict('records') for k, v in ser.groupby('StudyInstanceUID')}
        rec['inputs'].update(root=root, n_test=len(ids), n_series_studies=len(SER), ids_without_series=len(set(ids) - set(SER)))
        stage('inputs', t)
        # ---- 3. side pass: the side kernel's own study() ----
        t = time.time()
        os.environ['KC_ROOT'] = root
        os.environ['KC_OUT'] = A.out
        os.environ['KC_PROCS'] = str(A.side_procs)
        import importlib.util
        spec = importlib.util.spec_from_file_location('kc_side_kernel', os.path.join(CODE, 'kc_side_kernel.py'))
        _KS = importlib.util.module_from_spec(spec)
        sys.modules['kc_side_kernel'] = _KS
        spec.loader.exec_module(_KS)                                    # module level only (its __main__ block does not run)
        sids = sorted(SER)
        jobs = [(s, SER[s], sdir) for s in sids]
        import multiprocessing as mp
        res = []
        with mp.get_context('fork').Pool(max(1, A.side_procs), initializer=_side_init) as pool:
            for r in pool.imap_unordered(_side_one, jobs, chunksize=1):
                res.append(r)
                if time.time() > A.stop_at:
                    raise RuntimeError('stop time reached in the side pass (%d of %d studies)' % (len(res), len(jobs)))
        df = pd.DataFrame(res).sort_values('StudyInstanceUID').reset_index(drop=True)
        first = ['StudyInstanceUID', 'mean_x', 'n', 'sag_nx', 'tag', 'side', 'rule']
        df = df[[c for c in first if c in df.columns] + [c for c in df.columns if c not in first]]
        side_csv = os.path.join(A.out, 'kc_side_test.csv')
        df.to_csv(side_csv, index=False)
        nerr = int(df['side_err'].notna().sum()) if 'side_err' in df.columns else 0
        try:
            summ = _KS.summarise(df[df['side_err'].isna()] if nerr else df, time.time() - t)
        except Exception as e:
            summ = {'error': '%s: %s' % (type(e).__name__, str(e)[:200])}
        summ['side_errors'] = nerr
        json.dump(summ, open(os.path.join(A.out, 'kc_side_summary.json'), 'w'), indent=1, default=str)
        rec['side'] = {'n': len(df), 'errors': nerr, 'csv_sha256': sha256(side_csv), 'version': getattr(_KS, 'VERSION', None),
                       'serve': {str(k): int(v) for k, v in df['serve'].value_counts().items()} if 'serve' in df.columns else None,
                       'rules': {str(k): int(v) for k, v in df['rule'].value_counts().items()} if 'rule' in df.columns else None,
                       'kc_side_geo': summ.get('KC_SIDE_GEO'), 's_per_study': round((time.time() - t) / max(len(df), 1), 4)}
        stage('side', t)
        print('KC_SIDE_DONE ' + json.dumps(rec['side']), flush=True)
        # ---- 4. serve: one kc_serve.py process per GPU ----
        t = time.time()
        n_gpu = int(rec['env'].get('n_gpu') or 0)
        nw = A.workers or (n_gpu if A.device == 'cuda' else 2)
        if A.device == 'cuda' and (n_gpu < 1 or nw > n_gpu):
            raise RuntimeError('no GPU for the KC workers (n_gpu %d, workers %d)' % (n_gpu, nw))
        procs, logs, parts = [], [], []
        for k in range(nw):
            part = os.path.join(A.tmp, 'part%d.csv' % k)
            for f in (part, part + '.timing.json'):
                if os.path.exists(f):
                    os.remove(f)
            dev = 'cuda:%d' % k if A.device == 'cuda' else 'cpu'
            cmd = [sys.executable, '-u', ROOT_TMP + '/kc_serve.py', '--code', CODE, '--side', side_csv, '--timing', part + '.timing.json',
                   '--', '--root', root, '--split', 'test', '--ckpts', *A.members, '--out', part, '--device', dev,
                   '--shard', '%d/%d' % (k, nw), '--procs', str(A.procs)]
            env = dict(os.environ, OMP_NUM_THREADS=str(A.worker_threads), MKL_NUM_THREADS=str(A.worker_threads))
            for v in ('KC_ROOT', 'KC_OUT', 'KC_PROCS'):
                env.pop(v, None)
            lf = open(os.path.join(A.tmp, 'part%d.log' % k), 'w')
            procs.append(subprocess.Popen(cmd, env=env, cwd=A.tmp, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT,
                                          start_new_session=True))
            logs.append(lf)
            parts.append(part)
        rec['workers'] = [{'pid': p.pid, 'device': ('cuda:%d' % k if A.device == 'cuda' else 'cpu'), 'shard': '%d/%d' % (k, nw),
                           'procs': A.procs} for k, p in enumerate(procs)]
        print('KC_SERVE_START ' + json.dumps({'workers': nw, 'procs_each': A.procs, 'members': [os.path.basename(m) for m in A.members],
                                              'n_test': len(ids)}), flush=True)
        timed_out = False
        while any(p.poll() is None for p in procs):
            if time.time() >= A.stop_at:
                timed_out = True
                rec['killed_worker_groups'] = kill_groups([p for p in procs if p.poll() is None])
                for p in procs:
                    try:
                        p.wait(timeout=60)
                    except Exception:
                        pass
                break
            time.sleep(2)
        for lf in logs:
            lf.close()
        for k, p in enumerate(procs):
            w = rec['workers'][k]
            txt = open(os.path.join(A.tmp, 'part%d.log' % k), errors='replace').read()
            w.update(rc=p.returncode, timed_out=timed_out and p.returncode is not None and p.returncode < 0,
                     closing=next((ln for ln in txt.splitlines() if ln.startswith('KC worker:')), None), log_tail=txt[-1500:])
            try:
                tj = json.load(open(parts[k] + '.timing.json'))
                rows = tj.get('rows') or []
                dec = [r[1] for r in rows if isinstance(r[1], (int, float))]
                gpu = [r[2] for r in rows if isinstance(r[2], (int, float))]
                both = sum(1 for r in rows if r[3] == 'both')
                w.update(wall_s=tj.get('wall_s'), n_timed=len(rows), gpu_name=tj.get('gpu_name'), max_mem_alloc_mb=tj.get('max_mem_alloc_mb'),
                         decode_s_mean=round(float(np.mean(dec)), 4) if dec else None,
                         gpu_s_mean=round(float(np.mean(gpu)), 4) if gpu else None,
                         gpu_s_warm_mean=round(float(np.mean(gpu[1:])), 4) if len(gpu) > 1 else None,
                         n_both=both, gok_mismatch=sum(1 for r in rows if r[4]), timing_rows=rows[:50])
            except Exception as e:
                w['timing_error'] = type(e).__name__
        rec['serve_timed_out'] = timed_out
        stage('serve', t)
        # ---- 5. merge ----
        t = time.time()
        import kc_common as kc
        L = list(kc.TARGETS)
        frames = []
        for k, part in enumerate(parts):
            if os.path.isfile(part) and rec['workers'][k]['rc'] == 0:
                f = pd.read_csv(part, dtype={'StudyInstanceUID': str})
                f['kc_part'] = k
                frames.append(f)
        allp = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['StudyInstanceUID'] + L + ['kc_err'])
        if allp['StudyInstanceUID'].duplicated().any():
            raise RuntimeError('a study was served twice')
        allp = allp.set_index('StudyInstanceUID')
        out = pd.DataFrame({'StudyInstanceUID': ids})
        for c in L + ['kc_side', 'kc_rule', 'kc_serve', 'kc_vol_sha', 'kc_err', 'kc_part']:
            out[c] = [allp.at[s, c] if (s in allp.index and c in allp.columns) else (np.nan if c in L else '') for s in ids]
        for c in ('kc_side', 'kc_rule', 'kc_serve', 'kc_vol_sha', 'kc_err'):             # text columns (no float NaN -> str casts)
            out[c] = [('' if (v is None or (isinstance(v, float) and not np.isfinite(v))) else str(v)) for v in out[c]]
        miss = [s for s in ids if s not in allp.index]
        if miss:
            out['kc_err'] = [('not served (worker failed / timed out / no series)' if s in set(miss) else e)
                             for s, e in zip(out['StudyInstanceUID'], out['kc_err'])]
        P = out[L].to_numpy(np.float64)
        ok = np.isfinite(P).all(axis=1)
        cov = float(ok.mean()) if len(ids) else 0.0
        probs_csv = os.path.join(A.out, 'kc_probs.csv')
        out.to_csv(probs_csv + '.tmp', index=False)
        os.replace(probs_csv + '.tmp', probs_csv)
        tbl = df.set_index('StudyInstanceUID')
        sc = out[ok]
        side_mis = [s for s, sd_, ru in zip(sc['StudyInstanceUID'], sc['kc_side'], sc['kc_rule'])
                    if s not in tbl.index or (str(tbl.at[s, 'side']), str(tbl.at[s, 'rule'])) != (str(sd_), str(ru))]
        serve_less = [s for s, sv in zip(sc['StudyInstanceUID'], sc['kc_serve']) if s in tbl.index and str(tbl.at[s, 'serve']) == 'both'
                      and str(sv) != 'both']
        rec['B'] = {'n': len(ids), 'scored': int(ok.sum()), 'n_unscored': int((~ok).sum()), 'coverage': cov,
                    'probs_sha256': sha256(probs_csv), 'side_mismatch': side_mis[:20], 'n_side_mismatch': len(side_mis),
                    'serve_both_table_but_single': serve_less[:20], 'n_serve_both': int((sc['kc_serve'] == 'both').sum()),
                    'errors': out.loc[~ok, 'kc_err'].astype(str).value_counts().head(10).to_dict(), 'labels': L}
        wk = [w for w in rec['workers'] if w.get('n_timed')]
        if wk:
            nt = sum(w['n_timed'] for w in wk)
            rec['B']['microbench'] = {
                'side_s_per_study_wall': rec['side']['s_per_study'],
                'decode_s_per_study_per_proc': round(sum(w['decode_s_mean'] * w['n_timed'] for w in wk if w['decode_s_mean'] is not None) / nt, 4),
                'gpu_s_per_study': round(sum(w['gpu_s_mean'] * w['n_timed'] for w in wk if w['gpu_s_mean'] is not None) / nt, 4),
                'gpu_s_per_study_warm': (round(float(np.mean([w['gpu_s_warm_mean'] for w in wk if w.get('gpu_s_warm_mean') is not None])), 4)
                                         if any(w.get('gpu_s_warm_mean') is not None for w in wk) else None),
                'members': len(A.members), 'workers': len(procs), 'procs_each': A.procs,
                'serve_wall_s': rec['stages']['serve']['seconds'],
                'serve_wall_s_per_study': round(rec['stages']['serve']['seconds'] / max(len(ids), 1), 4),
                'both_frac': round(sum(w.get('n_both', 0) for w in wk) / nt, 4)}
        stage('merge', t)
        # ---- 6. dump (commit only) ----
        t = time.time()
        if 0 < len(ids) <= A.dump_max:
            sys.path.insert(0, ROOT_TMP)
            import kc_serve
            import kc_worker as kw
            table = kc_serve.load_table(side_csv)
            si_fn = kc_serve.make_study_inputs(table, kc)
            R = kw.load_reader()
            ddir = os.path.join(A.out, 'dump')
            os.makedirs(ddir, exist_ok=True)
            dumped = {}
            for s in ids:
                try:
                    si = si_fn(s, SER.get(s, []), sdir, R, both=True)
                    vsha = hashlib.sha256(si['vol'].tobytes()).hexdigest()[:16]
                    row = out.set_index('StudyInstanceUID').loc[s]
                    kw_ = {'vol': si['vol'], 'side': np.array(si['side']), 'rule': np.array(si['rule']), 'serve': np.array(si['serve']),
                           'weak': np.int64(si['weak']), 'gok': np.int64(si['gok']),
                           'geom_raw': np.array(json.dumps({b: (None if not np.isfinite(v) else float(v)) for b, v in si['geom_raw'].items()})),
                           'flips': np.array(json.dumps({b: bool(f) for b, f in kc.block_flips(si['side'], si['geom']).items()})),
                           'side_row': np.array(json.dumps({k_: _jsonable(v) for k_, v in table[s].items()})),
                           'labels': np.array(L), 'served': np.array([float(row[c]) for c in L], np.float64),
                           'served_vol_sha': np.array(str(row['kc_vol_sha'])), 'vol_sha': np.array(vsha),
                           'members': np.array(json.dumps([os.path.basename(m) for m in A.members]))}
                    for nm in ('views', 'views_L', 'views_R'):
                        for vn, st in (si.get(nm) or {}).items():
                            if st is not None:
                                kw_['%s__%s' % (nm, vn)] = st
                    np.savez_compressed(os.path.join(ddir, s + '.npz'), **kw_)
                    dumped[s] = {'vol_sha': vsha, 'served_vol_sha': str(row['kc_vol_sha']), 'equal': vsha == str(row['kc_vol_sha']),
                                 'npz_sha256': sha256(os.path.join(ddir, s + '.npz')), 'serve': si['serve']}
                except Exception as e:
                    dumped[s] = {'error': '%s: %s' % (type(e).__name__, str(e)[:200])}
            rec['dump'] = dumped
        else:
            rec['dump'] = 'skipped (n %d > dump max %d)' % (len(ids), A.dump_max)
        stage('dump', t)
        rec['ok'] = True
    except Exception as e:
        rec['error'] = '%s: %s' % (type(e).__name__, str(e)[:600])
        rec['traceback_tail'] = traceback.format_exc()[-1500:]
    save()
    print('KC_DRIVER_DONE ' + json.dumps({'ok': rec['ok'], 'error': rec['error'], 'coverage': (rec.get('B') or {}).get('coverage'),
                                          'scored': (rec.get('B') or {}).get('scored'), 'n': (rec.get('B') or {}).get('n'),
                                          'seconds': rec.get('total_seconds')}), flush=True)
    sys.exit(0 if rec['ok'] else 3)


if __name__ == '__main__':
    main()
