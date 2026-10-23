#!/usr/bin/env python3
"""k10n_check.py = k9n_check.py with the K10N kernel key / TAG (k10/make_k10_tools.py, 2026-10-09). k9n_check.py (2026-10-08, Claude): independent gate for a K9N working dir (box A0 or the downloaded Kaggle commit
output). Never imports the cells: every blend is recomputed here (numpy argsort average ranks, not pandas).

  python k9n_check.py --work DIR --k9n_build SP/k9n/build/K9N_BUILD.json --tag a|b [--expect_n clean|fail]
        [k2_check fleet options: --expect_fleet --kaggle_commit --expect_members --expect_manifest --expect_tta
         --allow_interim]  [--expect_n_ranks log949]

Part A (fleet stage, cell 31): k2_check.check runs UNCHANGED on a shadow copy of DIR in which submission.csv is the fleet
  stage's own P+F file (submission_w50.csv; its sha256 must equal the fleet status record of submission.csv) or absent
  when the fleet was not clean. So every K9 gate (members, strict mode, W .50, manifest bytes, TTA, commit receipt) is
  applied exactly as for K9-main. --expect_mode strict and --expect_w_main 0.5 are always passed.
Part B (N stage, cell 32), from the K9N files only:
  config (W3 / variants / tag / child sha / weights sha == K9N_BUILD.json), status, receipt, the child npz (ids == parent
  ids, finite, [0, 1], P == (plain + mirror) / 2 exactly, mirror != plain on every non-empty study, weights sha256 per GPU
  == pinned, recipe K94 / 320 / 384 / 96 / study_lut, fallback + empty within the gate), N's own csv == its rank formula,
  submission_n949.csv == the child's probabilities (median-filled) in parent order, every submission_pfn_*.csv and
  submission.csv == an independent 3-leg rank blend of _k2_parent.csv, submission_fleetsolo.csv and the npz, file sha256
  unchanged since the stage wrote them, the fleet's files untouched by the N stage, submission.csv ABSENT when not clean.
  --expect_n_ranks log949: N's 3 public-study ranks == the public commit log of the author (n949/log949.txt) exactly.
  --kaggle_commit also requires 2 GPUs in the child (2 weights sha256 entries) and 0 fallback / 0 empty.
What this gate cannot see: the hidden rerun itself (its ~1,300-study fallback count, its N runtime, the wall margin, RAM /
VRAM there); it only proves that the code path, the bytes and the arithmetic are right on the studies the output holds.
"""
import argparse, filecmp, glob, hashlib, json, os, shutil, sys, tempfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
sys.path.insert(0, HERE)
import k2_check                                                        # noqa: E402

T = k2_check.TARGETS
LOG949 = f'{HERE}/n949/log949.txt'


def _sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def avg_rank_pct(a):
    """Average-tie percentile rank per column (== pandas rank(method='average', pct=True)), numpy only."""
    a = np.asarray(a, np.float64)
    n = a.shape[0]
    out = np.empty_like(a)
    for j in range(a.shape[1]):
        x = a[:, j]
        o = np.argsort(x, kind='mergesort')
        xs = x[o]
        r = np.empty(n, np.float64)
        i = 0
        while i < n:
            k = i
            while k + 1 < n and xs[k + 1] == xs[i]:
                k += 1
            r[o[i:k + 1]] = (i + k) / 2.0 + 1.0
            i = k + 1
        out[:, j] = r / n
    return out


def tname(w):
    return 'submission_pfn_' + ''.join(f'{int(round(float(x) * 100)):02d}' for x in w) + '.csv'


def log949_ranks():
    """The author's public commit log: N's submission.csv rows for the 3 visible studies."""
    rows = {}
    for e in json.load(open(LOG949, encoding='utf-8')):
        s = e.get('data', '')
        if s.startswith('1.2.826.') and e.get('stream_name') == 'stdout':
            parts = s.split()
            rows[parts[0]] = [float(x) for x in parts[1:13]]
    return rows


def fleet_part(work, a, res, info):
    """Part A: k2_check.check on a shadow dir whose submission.csv is the fleet stage's own file."""
    st = json.load(open(f'{work}/_k2_stage_fleet.json')) if os.path.isfile(f'{work}/_k2_stage_fleet.json') else {}
    sh = tempfile.mkdtemp(prefix='k9n_shadow_')
    try:
        for f in os.listdir(work):
            src = f'{work}/{f}'
            if f == 'submission.csv':
                continue
            if os.path.isdir(src):
                shutil.copytree(src, f'{sh}/{f}')
            else:
                shutil.copy2(src, f'{sh}/{f}')
        fs_sha = (st.get('files') or {}).get('submission.csv')
        if st.get('clean') is True:
            w50 = f'{work}/{k2_check.wname(0.5)}'
            okc = os.path.isfile(w50) and _sha(w50) == fs_sha
            res.append(('A: fleet P+F submission recoverable: submission_w50.csv sha256 == the fleet status record of '
                        'submission.csv', okc, f'{fs_sha}'))
            if os.path.isfile(w50):
                shutil.copy2(w50, f'{sh}/submission.csv')
        em = [x.strip() for x in a.expect_members.split(',') if x.strip()] if a.expect_members else None
        r, inf = k2_check.check(sh, a.expect_fleet, 'none', kaggle_commit=a.kaggle_commit, expect_members=em,
                                expect_mode='strict', expect_manifest=a.expect_manifest, expect_w_main=0.5,
                                expect_tta=k2_check.parse_expect_tta(a.expect_tta), expect_outer_flat=None)
        res.extend((f'A: {n}', g, d) for n, g, d in r)
        info['fleet'] = inf
        if a.expect_tta and a.expect_tta.endswith('.json') and os.path.isfile(a.expect_tta):          # K8-INT
            bj = json.load(open(a.expect_tta))
            if bj.get('k8') and (bj.get('do_not_submit') or bj.get('interim')):
                res.append(('A: K9 build is FINAL (an INTERIM build is NEVER submitted)' +
                            (' [--allow_interim: TEST USE ONLY]' if a.allow_interim else ''), bool(a.allow_interim),
                            f'sets {bj.get("sets")}'))
    finally:
        shutil.rmtree(sh, ignore_errors=True)
    return st


def n_part(work, a, kb, res, info, fst):
    def ok(name, cond, detail=''):
        res.append((f'B: {name}', bool(cond), str(detail)[:300]))

    k = kb['kernels'][f'rsna-sub-k10n-{a.tag}']
    W3, VAR = k['W3'], k['W3_VARIANTS']
    loud = ['submission_n949.csv'] + list(dict.fromkeys(tname(w) for w in [W3] + VAR))
    stp, rcp = f'{work}/_k9n_stage.json', f'{work}/n949_receipt.json'
    st = json.load(open(stp)) if os.path.isfile(stp) else {}
    rec = json.load(open(rcp)) if os.path.isfile(rcp) else {}
    cfg = rec.get('config') or {}
    clean = st.get('clean') is True
    info['n'] = {kk: rec.get(kk) for kk in ('why', 'n', 'stage_min', 'child_s', 'kernel_age_h_start', 'kernel_age_h_end',
                                             'left_min', 'est_central_min', 'est_slow_min', 'fleet_speed', 'gpus',
                                             'gpu_free_mib', 'host_ram_avail_gb', 'n_fallback', 'n_empty', 'rho_mean',
                                             'mirror_meanabs', 'mirror_maxabs', 'gpu_s_per_study', 'child_meta')}
    ok('N status + receipt present (_k9n_stage.json, n949_receipt.json)', bool(st) and bool(rec))
    if a.expect_n != 'auto':
        ok(f'N stage outcome == {a.expect_n}', clean == (a.expect_n == 'clean'), st.get('why'))
    ok(f'N config: W3 {W3}, variants {VAR}, tag K10N-{a.tag}, strict', cfg.get('W3') == W3 and cfg.get('W3_VARIANTS') == VAR
       and cfg.get('TAG') == f'K10N-{a.tag}' and st.get('main_mode') == 'strict', f'{cfg.get("W3")} {cfg.get("TAG")}')
    ok('N config: child / source / weights sha256 == K9N_BUILD.json', cfg.get('CHILD_SHA256') == kb['child_sha256']
       and cfg.get('SRC_SHA256') == kb['src_sha256'] and cfg.get('WEIGHTS_SHA256') == kb['weights_sha256'],
       f'child {str(cfg.get("CHILD_SHA256"))[:16]}')
    sub = f'{work}/submission.csv'
    if not clean:
        for f in loud + ['submission.csv']:
            ok(f'{f} ABSENT (N not clean, strict)', not os.path.exists(f'{work}/{f}'))
        return
    ok('receipt: weights sha256 (cell) == pinned', rec.get('weights_sha256') == kb['weights_sha256'], rec.get('weights_sha256'))
    ok('receipt: embedded child sha256 == pinned', rec.get('child_sha256') == kb['child_sha256'])
    ok('receipt: child rc 0', rec.get('child_rc') == 0, rec.get('child_rc'))
    pub = pd.read_csv(f'{work}/_k2_parent.csv', dtype={'StudyInstanceUID': str})
    fso = pd.read_csv(f'{work}/submission_fleetsolo.csv', dtype={'StudyInstanceUID': str})
    n, ids_par = len(pub), pub['StudyInstanceUID'].tolist()
    z = np.load(f'{work}/_n949_probs.npz', allow_pickle=False)
    meta = json.loads(str(z['meta']))
    ids = [str(x) for x in z['ids']]
    PN, PL, PM = (np.asarray(z[x], np.float32) for x in ('P', 'P_plain', 'P_mirror'))
    ok(f'npz: {n} unique ids == the parent ids, shapes ({n}, 12)', len(ids) == n and len(set(ids)) == n
       and set(ids) == set(ids_par) and PN.shape == PL.shape == PM.shape == (n, 12), f'{len(ids)} ids')
    ok('npz: P finite and in [0, 1]', np.isfinite(PN).all() and (PN >= 0).all() and (PN <= 1).all())
    ok('npz: label order == the 12 targets', meta.get('labels') == T, meta.get('labels'))
    ok('npz: recipe K94 / native 320 / 384 cache / 96 slices / study_lut',
       (meta.get('k_eval'), meta.get('input_res'), meta.get('img'), meta.get('maxs'), meta.get('fastread')) == (94, 320, 384, 96, 'study_lut'))
    shas = meta.get('weights_sha256') or []
    ok(f'npz: weights sha256 per loaded model == pinned ({len(shas)} model(s))',
       bool(shas) and all(s == kb['weights_sha256'] for s in shas) and len(shas) == max(1, int(meta.get('ngpu') or 0)), shas)
    fb, em = set(meta.get('fallback') or []), set(meta.get('empty') or [])
    lim = 0.02 if n >= 11 else 0.0
    ok(f'npz: fallback + empty studies <= {lim:.2f} of {n}', len(fb | em) / max(1, n) <= lim, f'fallback {len(fb)} empty {len(em)}')
    good = np.array([s not in fb for s in ids]); full = np.array([s not in fb and s not in em for s in ids])
    ok('npz: P == (plain + mirror) / 2 exactly on the served rows (anatomical mirror TTA as published)',
       np.isfinite(PL[good]).all() and np.isfinite(PM[good]).all()
       and np.array_equal(PN[good], np.float32(0.5) * (PL[good] + PM[good])))
    dmir = np.abs(PL.astype(np.float64) - PM).max(axis=1)
    ok('npz: mirror view != plain view on every non-empty study (mirror path exercised)',
       full.any() and (dmir[full] > 0).all(), f'max|plain - mirror| per study {np.round(dmir, 4).tolist()[:6]}')
    if a.kaggle_commit:
        ok('commit: child saw 2 GPUs and loaded the model on both', meta.get('ngpu') == 2 and len(shas) == 2, meta.get('ngpu'))
        ok('commit: 0 fallback and 0 empty studies', not fb and not em)
    cs = pd.read_csv(f'{work}/_n949_sub.csv', dtype={'StudyInstanceUID': str})
    x = np.clip(PN, 0, 1); rk = x.argsort(0).argsort(0).astype(np.float64) / max(1, n - 1)
    ok("N's own csv (_n949_sub.csv) == its rankpct formula, test.csv order",
       cs.columns.tolist() == ['StudyInstanceUID', *T] and cs['StudyInstanceUID'].tolist() == ids
       and float(np.abs(cs[T].to_numpy(np.float64) - rk.astype(np.float32)).max()) <= 1e-6)
    if a.expect_n_ranks == 'log949':
        ref = log949_ranks()
        got = {s: cs.set_index('StudyInstanceUID').loc[s, T].to_numpy(np.float64).tolist() for s in ref if s in set(ids)}
        okr = len(ref) == 3 and set(got) == set(ref) and all(np.allclose(got[s], ref[s], atol=1e-6) for s in ref)
        det = '3/3 rows equal'
        if not okr and set(got) == set(ref):                         # which labels flipped, and by how much
            bad = [j for j in range(12) if any(abs(got[s][j] - ref[s][j]) > 1e-6 for s in ref)]
            pos_c = {s: i for i, s in enumerate(ids)}
            pp = {s: PN[pos_c[s]] for s in ref}
            gaps = {T[j]: float(min(abs(float(pp[x][j]) - float(pp[y][j])) for x in ref for y in ref if x < y)) for j in bad}
            det = (f'labels differing {[T[j] for j in bad]}; smallest prob gap among the 3 studies per such label {gaps} '
                   f'(a gap < 2e-3 is fp16 / cudnn noise between images: accept by hand only then; else STOP)')
        ok("N's 3 public-study ranks == the author's public commit log (log949.txt) exactly", okr, det)
    # independent recomputation of every N file (numpy average ranks)
    pos = {s: i for i, s in enumerate(ids)}
    NV = PN.astype(np.float64)[[pos[s] for s in ids_par]]
    badp = np.array([s in fb or s in em for s in ids_par])
    if badp.any():
        NV[badp] = np.median(NV[~badp], axis=0)[None]
    rP = avg_rank_pct(pub[T].to_numpy(np.float64)); rF = avg_rank_pct(fso[T].to_numpy(np.float64)); rN = avg_rank_pct(NV)

    def frame(f):
        d = pd.read_csv(f'{work}/{f}', dtype={'StudyInstanceUID': str})
        g = (d.columns.tolist() == pub.columns.tolist() and len(d) == n and d['StudyInstanceUID'].tolist() == ids_par)
        v = d[T].to_numpy(np.float64) if g else np.full((n, 12), np.nan)
        ok(f'{f}: schema, {n} rows, parent row order, finite, in [0, 1]', g and np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all())
        return v
    for f in loud:
        ok(f'{f} written (N clean)', os.path.isfile(f'{work}/{f}'))
    if all(os.path.isfile(f'{work}/{f}') for f in loud + ['submission.csv']):
        v = frame('submission_n949.csv')
        ok('submission_n949.csv == the child probabilities (median-filled) in parent order', float(np.abs(v - NV).max()) < 1e-12)
        for w in [W3] + VAR:
            v = frame(tname(w))
            ref = avg_rank_pct(w[0] * rP + w[1] * rF + w[2] * rN)
            dd = float(np.abs(v - ref).max())
            ok(f'{tname(w)} == independent 3-leg rank blend P/F/N {w}', dd < 1e-9, f'max|diff| {dd:.2e}')
        frame('submission.csv')
        ok(f'submission.csv == its fail-loud twin {tname(W3)} (P/F/N {W3})', filecmp.cmp(sub, f'{work}/{tname(W3)}', shallow=False))
        files = st.get('files') or {}
        for f in loud + ['submission.csv']:
            ok(f'{f} unchanged since the N stage wrote it (sha256)', files.get(f) == _sha(f'{work}/{f}'))
        ffiles = fst.get('files') or {}
        for f in [x for x in ffiles if x != 'submission.csv']:
            ok(f'fleet file {f} untouched by the N stage (sha256)', os.path.isfile(f'{work}/{f}') and ffiles[f] == _sha(f'{work}/{f}'))
        ok('N status: the fleet submission.csv it deleted == the fleet record', st.get('fleet_submission_sha256') == ffiles.get('submission.csv'))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--k9n_build', required=True, help='K9N_BUILD.json of the build whose kernel ran')
    ap.add_argument('--tag', required=True, choices=('a', 'b'))
    ap.add_argument('--expect_n', default='auto', choices=('auto', 'clean', 'fail'))
    ap.add_argument('--expect_fleet', default='auto', choices=('auto', 'clean', 'fail'))
    ap.add_argument('--kaggle_commit', action='store_true')
    ap.add_argument('--expect_members', default=None)
    ap.add_argument('--expect_manifest', default=None)
    ap.add_argument('--expect_tta', default=None)
    ap.add_argument('--allow_interim', action='store_true')
    ap.add_argument('--expect_n_ranks', default=None, choices=(None, 'log949'))
    ap.add_argument('--skip_fleet', action='store_true', help='TEST ONLY: stubbed fleet (box N-only A0): part A not run')
    ap.add_argument('--json', default=None)
    a = ap.parse_args(argv)
    kb = json.load(open(a.k9n_build))
    res, info = [], {}
    if kb.get('do_not_push'):
        res.append(('K10N build is FINAL (K10N_BUILD.json do_not_push false)', False, 'TEST build: never push / submit'))
    if a.skip_fleet:
        fst = json.load(open(f'{a.work}/_k2_stage_fleet.json')) if os.path.isfile(f'{a.work}/_k2_stage_fleet.json') else {}
        res.append(('A: fleet part SKIPPED (--skip_fleet, stubbed fleet stage: TEST ONLY)', not a.kaggle_commit, ''))
    else:
        fst = fleet_part(a.work, a, res, info)
    n_part(a.work, a, kb, res, info, fst)
    for name, good, detail in res:
        print(f'{"PASS" if good else "FAIL"}  {name}  {detail}')
    allok = all(g for _, g, _ in res)
    print('INFO', json.dumps(info, default=str)[:4000])
    print(f'K10NCHECK {"PASS" if allok else "FAIL"} ({sum(g for _, g, _ in res)}/{len(res)})')
    if a.json:
        json.dump({'pass': allok, 'checks': res, 'info': info}, open(a.json, 'w'), indent=1, default=str)
    return 0 if allok else 1


if __name__ == '__main__':
    sys.exit(main())
