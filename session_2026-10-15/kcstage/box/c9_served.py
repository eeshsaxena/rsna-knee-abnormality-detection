#!/usr/bin/env python
# c9_served.py (10-15, Claude subagent, kcstage; fix of finding KCS2): A6 G3 C9 probability parity on the SERVED path.
# A6 G3 C9 = "probability parity on 20 eda_data studies (including one SagPD-absent and one weak-zone study), DICOM path vs store
# path, same ckpt: max |dp| <= 1e-3". The A6-frozen tool kc_prob_parity.py (md5 ca9cd0ca) builds its DICOM side with
# kc_worker.study_inputs (header_side + 3-header slot_geometry), which the shipped kernel does NOT run: kc_serve.py replaces
# kc_worker.study_inputs with the side-table version (kc_serve.make_study_inputs, side rows from the side kernel's own study()).
# This wrapper runs kc_prob_parity.py VERBATIM (its own source, exec'd as __main__) with exactly that ONE substitution, so C9
# measures the path that ships:
#   kc_worker.study_inputs <- kc_serve.make_study_inputs(kc_serve.load_table(SIDE_CSV), kc_common)
#   SIDE_CSV = the side kernel's own study() (kc_side_kernel.py sha256 2979c191, called exactly as kc_driver.py does: pandas default
#              dtypes, uids as str, one study per call) on the picked TRAIN studies, then compared on every column with the store's
#              side table (~/kc_store/side_train.csv sha ecc5cd26 = the side kernel's Kaggle output the store was built from).
# Picks (n 20): the first SagPD-absent study (store has no sag_pd blob), the first weak-zone study (store weak == 1, served on both
# canons), then kc_prob_parity's own rest order (rng 1015 permutation of the studies with a store npz). The train store has no
# geometry-only 'both' study (180 'both' == 180 weak, MEASURED 10-15), so none is picked.
# PASS iff kc_prob_parity PASS (max |dp| <= 1e-3, own canon, store vs served DICOM) AND n >= 20 AND >= 1 weak-zone study served on
# both canons AND >= 1 SagPD-absent study AND the fresh side rows == the store side table (every column) AND the substitution was live.
# --path header runs the same tool WITHOUT the substitution (the pre-fix C9; used only as the fail-before control of the tests).
# --side-table FILE serves from FILE instead of the fresh rows (tests only; then PASS is impossible: side_source != fresh).
# Box CPU only (taskset -c 28-39 nice -n 19 ionice -c3 env CUDA_VISIBLE_DEVICES=), fp32. Writes one JSON the gate reads
# (v224s6fkc_check.py --c9, default OUT/c9_served.json) and prints KC_C9_SERVED PASS/FAIL.
#   python c9_served.py --code kcstage_1015/code --serve kcstage_1015/kc_serve.py --kpp kc/code/kc_prob_parity.py \
#          --members m1.pt m2.pt [--manifest KC_MANIFEST.json] --store kc_store/store --dicom eda_data --n 20 --threads 12 --out C9.json
import os, sys, json, time, hashlib, argparse, importlib.util
sys.dont_write_bytecode = True

KPP_MD5 = 'ca9cd0ca7310cf5d6853a50a6fabdd70'
SIDE_SHA = '2979c1915e0cecf2c4792526a5581f93a723fac368eacff18849661dc0bdcb62'
TOL = 1e-3
VERSION = 'c9_served v1 10-15'


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True, help='dir with the pinned KC code (kc_worker, kc_common, fleet models, d96 reader, side kernel)')
ap.add_argument('--serve', required=True, help='the kc_serve.py the kernel ships (its sha256 is recorded; the gate compares it)')
ap.add_argument('--kpp', required=True, help='kc_prob_parity.py (A6 freeze md5 ca9cd0ca)')
ap.add_argument('--members', nargs='+', required=True)
ap.add_argument('--manifest', default=None)
ap.add_argument('--store', required=True)
ap.add_argument('--dicom', required=True)
ap.add_argument('--n', type=int, default=20)
ap.add_argument('--threads', type=int, default=12)
ap.add_argument('--side-ref', default=os.path.expanduser('~/kc_store/side_train.csv'))
ap.add_argument('--side-table', default=None, help='tests only: serve from this table instead of the fresh side-kernel rows')
ap.add_argument('--sids', nargs='*', default=None)
ap.add_argument('--path', default='served', choices=('served', 'header'))
ap.add_argument('--out', required=True)
A = ap.parse_args()
t_all = time.time()
outdir = os.path.dirname(os.path.abspath(A.out))
os.makedirs(outdir, exist_ok=True)
stem = os.path.splitext(os.path.abspath(A.out))[0]
res = {'tool': VERSION, 'path': A.path, 'PASS': False, 'error': None}
try:
    assert md5(A.kpp) == KPP_MD5, ('kc_prob_parity.py md5 != the A6 freeze', md5(A.kpp))
    assert sha256(os.path.join(A.code, 'kc_side_kernel.py')) == SIDE_SHA, 'kc_side_kernel.py sha256 != 2979c191'
    spec = importlib.util.spec_from_file_location('kc_serve', A.serve)
    kcs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kcs)                                       # defs only (its main() runs under __main__ only)
    kcs.check_pins(A.code)                                             # kc_worker 4af60f6f, kc_common 45255ae0, fleet models, d96 30aa72fd
    res.update(kpp_md5=KPP_MD5, serve_sha256=sha256(A.serve), side_kernel_sha256=SIDE_SHA, code_md5=dict(kcs.PINS_MD5),
               d96_sha256=kcs.D96_SHA256)
    # members: ordered, sha256, == the manifest (order and sha) when given
    mem = [{'file': os.path.basename(p), 'sha256': sha256(p)} for p in A.members]
    res['members'] = mem
    res['member_sha256'] = {m['file']: m['sha256'] for m in mem}
    if A.manifest:
        man = json.load(open(A.manifest))
        want = [m['file'] for m in man['members']]
        assert [m['file'] for m in mem] == want, ('member order / names != manifest', [m['file'] for m in mem], want)
        assert all(man['files'][m['file']] == m['sha256'] for m in mem), 'member sha256 != manifest'
        res['manifest_sha256'] = sha256(A.manifest)
        res['manifest_kind'] = man.get('kind')
    # the pinned KC code first on the path; fleet models pre-imported from it (kc_prob_parity also puts other dirs on sys.path)
    sys.path.insert(0, os.path.abspath(A.code))
    import numpy as np, pandas as pd
    import kc_common as kc
    import kc_worker as kw
    import fleet_model, fleet_model_ttr                                # noqa: F401  (load_members / model_sha use these modules)
    for mod in (kc, kw, fleet_model, fleet_model_ttr):
        assert os.path.dirname(os.path.abspath(mod.__file__)) == os.path.abspath(A.code), mod.__file__
    res['model_sha'] = kw.model_sha()[0]
    ROOT = A.dicom if os.path.exists(f'{A.dicom}/train_series.csv') else f'{A.dicom}/rsna-knee-abnormality-detection'
    SDIR = f'{ROOT}/train_series'
    # ---- picks ----
    ip = kc.VIEW_NAMES.index('sag_pd')
    if A.sids:
        pick, rule = list(A.sids), 'given'
    else:
        have = sorted(s for s in os.listdir(SDIR) if os.path.exists(f'{A.store}/{s}.npz'))
        absent, weak = [], []
        for s in have:
            with np.load(f'{A.store}/{s}.npz') as z:
                if not absent and f'v{ip}_blob' not in z.files:
                    absent.append(s)
                elif not weak and int(z['weak']) == 1:
                    weak.append(s)
            if absent and weak:
                break
        rest = [s for s in np.random.default_rng(1015).permutation(have) if s not in absent + weak]
        pick, rule = (absent + weak + rest)[:A.n], 'first SagPD-absent + first weak (sorted store order) + rng 1015 permutation'
    res['picks'], res['pick_rule'] = pick, rule
    # ---- side table: the side kernel's own study() on the picks (as kc_driver.py), vs the store side table ----
    ref = pd.read_csv(A.side_ref, dtype={'StudyInstanceUID': str, 'tag': str}).set_index('StudyInstanceUID')
    res['side_ref'], res['side_ref_sha256'] = A.side_ref, sha256(A.side_ref)
    if A.side_table:
        side_csv, res['side_source'] = A.side_table, 'given'
    else:
        os.environ['KC_ROOT'] = ROOT
        os.environ['KC_OUT'] = outdir
        os.environ['KC_PROCS'] = '1'
        sspec = importlib.util.spec_from_file_location('kc_side_kernel', os.path.join(A.code, 'kc_side_kernel.py'))
        KS = importlib.util.module_from_spec(sspec)
        sys.modules['kc_side_kernel'] = KS
        sspec.loader.exec_module(KS)
        ser = pd.read_csv(f'{ROOT}/train_series.csv')                  # default dtypes, as the corpus builder / side kernel / driver
        ser['StudyInstanceUID'] = ser['StudyInstanceUID'].astype(str)
        ser['SeriesInstanceUID'] = ser['SeriesInstanceUID'].astype(str)
        SERs = {k: v.to_dict('records') for k, v in ser.groupby('StudyInstanceUID')}
        t1, rows_ = time.time(), []
        for s in pick:
            try:
                rows_.append(KS.study((s, SERs[s], SDIR)))
            except Exception as e:
                rows_.append({'StudyInstanceUID': s, 'side_err': '%s: %s' % (type(e).__name__, str(e)[:200])})
        df = pd.DataFrame(rows_).sort_values('StudyInstanceUID').reset_index(drop=True)
        first = ['StudyInstanceUID', 'mean_x', 'n', 'sag_nx', 'tag', 'side', 'rule']
        df = df[[c for c in first if c in df.columns] + [c for c in df.columns if c not in first]]
        side_csv = stem + '_side.csv'
        df.to_csv(side_csv, index=False)
        res['side_source'], res['side_seconds'] = 'fresh', round(time.time() - t1, 1)
    res['side_table'], res['side_table_sha256'] = side_csv, sha256(side_csv)
    mine = pd.read_csv(side_csv, dtype={'StudyInstanceUID': str, 'tag': str}).set_index('StudyInstanceUID')
    mine = mine.loc[[s for s in pick if s in mine.index]]
    diff = {}
    if len(mine) != len(pick) or 'side_err' in mine.columns:
        diff['_rows'] = 'missing rows or side_err'
    r_ = ref.loc[mine.index]
    for c in sorted(set(mine.columns) | set(r_.columns)):
        if c not in mine.columns or c not in r_.columns:
            diff[c] = 'missing'
            continue
        a, b = mine[c], r_[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            x, y = a.to_numpy(np.float64), b.to_numpy(np.float64)
            bad = ~((np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= 1e-9 * np.maximum(1, np.abs(y))))
        else:
            bad = (a.fillna('<NA>').astype(str) != b.fillna('<NA>').astype(str)).to_numpy()
        if bad.any():
            diff[c] = int(bad.sum())
    res['side_equal_ref'], res['side_diff'] = not diff, diff
    # ---- THE substitution (served path), then kc_prob_parity.py verbatim ----
    orig_si = kw.study_inputs
    if A.path == 'served':
        kw.study_inputs = kcs.make_study_inputs(kcs.load_table(side_csv), kc)
        res['substitution'] = 'kc_worker.study_inputs <- kc_serve.make_study_inputs(kc_serve.load_table(side table), kc_common)'
    else:
        res['substitution'] = None
    raw_out = stem + '_kpp.json'
    src = open(A.kpp, encoding='utf-8').read()
    sys.argv = [os.path.abspath(A.kpp), '--ckpts', *A.members, '--store', A.store, '--dicom', A.dicom, '--n', str(len(pick)),
                '--device', 'cpu', '--threads', str(A.threads), '--out', raw_out, '--sids', *pick]
    ns = {'__name__': '__main__', '__file__': os.path.abspath(A.kpp), '__builtins__': __builtins__}
    t2 = time.time()
    exec(compile(src, os.path.abspath(A.kpp), 'exec'), ns)
    res['kpp_seconds'] = round(time.time() - t2, 1)
    live = ns.get('kw') is kw and ns.get('kc') is kc and (kw.study_inputs is not orig_si) == (A.path == 'served')
    res['substitution_live'] = bool(live)
    raw = json.load(open(raw_out))
    res['kpp'] = raw
    import torch
    res['torch'], res['threads'], res['device'] = torch.__version__, A.threads, 'cpu fp32'
    rows = raw.get('rows') or []
    res['n'] = len(rows)
    res['max_dp'] = float(max(r['max_dp'] for r in rows)) if rows else float('inf')
    res['n_weak_both'] = sum(1 for r in rows if int(r['weak']) == 1 and r['serve'] == 'both')
    res['n_weak'] = sum(1 for r in rows if int(r['weak']) == 1)
    res['n_sagpd_absent'] = sum(1 for r in rows if 'sag_pd' in (r.get('absent') or []))
    res['prob_pass'] = bool(rows) and res['max_dp'] <= TOL
    req = {'kpp PASS (max |dp| <= 1e-3)': res['prob_pass'] and raw.get('PASS') is True, 'served path': A.path == 'served',
           'substitution live': res['substitution_live'], 'side rows fresh from the side kernel': res['side_source'] == 'fresh',
           'fresh side rows == the store side table': res['side_equal_ref'], 'n >= 20': res['n'] >= 20 and res['n'] == len(pick),
           '>= 1 weak-zone study served on both canons': res['n_weak_both'] >= 1 and res['n_weak_both'] == res['n_weak'],
           '>= 1 SagPD-absent study': res['n_sagpd_absent'] >= 1}
    res['requirements'] = req
    res['PASS'] = all(req.values())
except Exception as e:
    import traceback
    res['error'] = '%s: %s' % (type(e).__name__, str(e)[:600])
    res['trace'] = traceback.format_exc()[-3000:]
res['seconds'] = round(time.time() - t_all, 1)
json.dump(res, open(A.out, 'w'), indent=1, default=str)
print('KC_C9_SERVED %s path %s n %s max |dp| %s | weak-both %s sagpd-absent %s | side %s == ref %s | live %s | %s s%s' % (
    'PASS' if res['PASS'] else 'FAIL', A.path, res.get('n'), ('%.2e' % res['max_dp']) if isinstance(res.get('max_dp'), float) else None,
    res.get('n_weak_both'), res.get('n_sagpd_absent'), res.get('side_source'), res.get('side_equal_ref'), res.get('substitution_live'),
    res['seconds'], (' | ERROR ' + res['error']) if res['error'] else ''), flush=True)
sys.exit(0 if res['PASS'] else 1)
