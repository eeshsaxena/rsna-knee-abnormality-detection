#!/usr/bin/env python3
"""fn_check.py (kfn, 2026-10-09, Claude): independent gate for an FN (rsna-sub-k9fn) or N-only (rsna-sub-nonly) working
dir (box A0 or the downloaded Kaggle commit output). Never imports the cells: every blend is recomputed here (numpy
argsort average ranks, k9n_check.avg_rank_pct).

  python fn_check.py --work DIR --fn_build SP/kfn/build/FN_BUILD.json --kernel fn|nonly [--expect_n clean|fail]
        [FN fleet options, as k9n_check / k2_check: --expect_fleet --kaggle_commit --expect_members --expect_manifest
         --expect_tta]  [--expect_n_ranks log949]  [--skip_fleet (TEST ONLY, stubbed fleet)]  [--json OUT]

Part 0 (both): FN-IDS: _fn_ids.csv = the 12 targets all 0.5 (id-only placeholder) == its _fn_ids.json record; NO parent
  output in the dir (_pipeline_stage.csv, _raptor.csv, btk*/btkd* files, _coat_env, _g96_env, _d4_env, g96_input, d4_input,
  diagnostics/): the parent never ran. FN: _k2_parent.csv == the placeholder bytes. N-only: no fleet file at all.
Part A (FN): k9n_check.fleet_part = k2_check UNCHANGED on a shadow dir whose submission.csv is the fleet stage's own
  P+F file (strict, W .50, members, manifest bytes, TTA, commit receipt); + the fleet K2_CFG == FN_BUILD k2_cfg
  (the K9-main literal + RHO_MIN_N 1e9).
Part B (both): N config (W3, variants, TAG, MODE, RHO_REF, child / source / weights sha == FN_BUILD.json), status,
  receipt, the child npz (ids == placeholder ids, finite, [0, 1], P == (plain + mirror) / 2, mirror exercised, weights
  sha per GPU, recipe, fallback + empty), N's own csv == its formula, --expect_n_ranks log949 (the author's 3 public-study
  ranks exactly), every submission_pfn_*.csv and submission.csv == an independent blend P/F/N with P = the constant
  placeholder (weight 0), F = submission_fleetsolo.csv (FN) or absent (N-only, weight 0), N = the npz; files unchanged
  since written; fleet files untouched (FN); submission.csv ABSENT when not clean.
"""
import argparse, filecmp, glob, hashlib, json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
sys.path.insert(0, HERE)
import k2_check, k9n_check                                             # noqa: E402

T = k2_check.TARGETS
avg_rank_pct, tname = k9n_check.avg_rank_pct, k9n_check.tname
SLUG = {'fn': 'rsna-sub-k9fn', 'nonly': 'rsna-sub-nonly'}
PARENT_OUT = ['_pipeline_stage.csv', '_raptor.csv', 'btk_v32_rest_input.npz', '_coat_env', '_g96_env', '_d4_env',
              'g96_input', 'd4_input', 'diagnostics', '_coat_arm.csv', 'btkd_v559_complete.json']
FLEET_OUT = ['_k2_stage_fleet.json', 'fleet_receipt.json', 'submission_fleetsolo.csv', '_k2_parent.csv']


def _sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def part0(work, kern, res, info):
    def ok(name, cond, detail=''):
        res.append((f'0: {name}', bool(cond), str(detail)[:300]))
    idp, ijp = f'{work}/_fn_ids.csv', f'{work}/_fn_ids.json'
    ok('FN-IDS files present (_fn_ids.csv, _fn_ids.json)', os.path.isfile(idp) and os.path.isfile(ijp))
    if os.path.isfile(idp) and os.path.isfile(ijp):
        ph = pd.read_csv(idp, dtype={'StudyInstanceUID': str}); rj = json.load(open(ijp))
        ok('_fn_ids.csv: 12 targets, every value 0.5 (id-only placeholder), unique ids, == its json record',
           ph.columns.tolist() == ['StudyInstanceUID', *T] and len(ph) > 0 and ph['StudyInstanceUID'].is_unique
           and (ph[T].to_numpy(np.float64) == 0.5).all() and rj.get('n') == len(ph) and rj.get('file_sha256') == _sha(idp),
           f'n {len(ph)} json {rj.get("n")}')
        info['ids'] = {k: rj.get(k) for k in ('n', 'root', 'sample_submission', 'ids_sha256')}
    present = [f for f in os.listdir(work) if f in PARENT_OUT or f.startswith(('btkd_', 'btk_'))]
    ok('no parent output in the working dir (the 0.943 parent never ran)', not present, present)
    if kern == 'fn':
        pp = f'{work}/_k2_parent.csv'
        ok('FN: _k2_parent.csv == the FN-IDS placeholder bytes (P is the placeholder)',
           os.path.isfile(pp) and os.path.isfile(idp) and filecmp.cmp(pp, idp, shallow=False))
    else:
        fl = [f for f in os.listdir(work) if f in FLEET_OUT or f.startswith(('_fleet_', 'submission_w'))]
        ok('N-only: no fleet stage file in the working dir', not fl, fl)


def part_a(work, a, kb, res, info):
    fst = k9n_check.fleet_part(work, a, res, info)
    rec = json.load(open(f'{work}/fleet_receipt.json')) if os.path.isfile(f'{work}/fleet_receipt.json') else {}
    cfg = rec.get('config') or {}
    want = dict(kb['k2_cfg'])
    got = {k: cfg.get(k) for k in want}
    res.append(('A: fleet K2_CFG == FN_BUILD k2_cfg (K9-main literal + RHO_MIN_N 1e9)', got == want,
                str({k: (got[k], want[k]) for k in want if got[k] != want[k]})[:300]))
    return fst


def part_b(work, a, kb, res, info, fst, kern):
    def ok(name, cond, detail=''):
        res.append((f'B: {name}', bool(cond), str(detail)[:300]))

    k = kb['kernels'][SLUG[kern]]
    W3, VAR = k['W3'], k['W3_VARIANTS']
    loud = ['submission_n949.csv'] + list(dict.fromkeys(tname(w) for w in [W3] + VAR))
    stp, rcp = f'{work}/_k9n_stage.json', f'{work}/n949_receipt.json'
    st = json.load(open(stp)) if os.path.isfile(stp) else {}
    rec = json.load(open(rcp)) if os.path.isfile(rcp) else {}
    cfg = rec.get('config') or {}
    clean = st.get('clean') is True
    info['n'] = {kk: rec.get(kk) for kk in ('why', 'n', 'stage_min', 'child_s', 'kernel_age_h_start', 'kernel_age_h_end',
                                             'left_min', 'est_central_min', 'est_slow_min', 'gpus', 'gpu_free_mib',
                                             'host_ram_avail_gb', 'n_fallback', 'n_empty', 'rho_mean', 'rho_ref',
                                             'mirror_meanabs', 'gpu_s_per_study', 'child_meta')}
    ok('N status + receipt present (_k9n_stage.json, n949_receipt.json)', bool(st) and bool(rec))
    if a.expect_n != 'auto':
        ok(f'N stage outcome == {a.expect_n}', clean == (a.expect_n == 'clean'), st.get('why'))
    ok(f'N config: W3 {W3}, variants {VAR}, TAG {k["tag"]}, MODE {k["mode"]}, RHO_REF {k["rho_ref"]}, strict',
       cfg.get('W3') == W3 and cfg.get('W3_VARIANTS') == VAR and cfg.get('TAG') == k['tag'] and cfg.get('MODE') == k['mode']
       and cfg.get('RHO_REF') == k['rho_ref'] and st.get('main_mode') == 'strict' and st.get('MODE') == k['mode'],
       f'{cfg.get("W3")} {cfg.get("TAG")} {cfg.get("MODE")} {cfg.get("RHO_REF")}')
    ok('N config: child / source / weights sha256 == FN_BUILD.json', cfg.get('CHILD_SHA256') == kb['child_sha256']
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
    ok('receipt: placeholder sha256 == _fn_ids.csv', rec.get('placeholder_sha256') == _sha(f'{work}/_fn_ids.csv'))
    ok(f'receipt: Spearman reference leg == {"fleet" if kern == "fn" else "N-own-csv"}',
       rec.get('rho_ref') == ('fleet' if kern == 'fn' else 'N-own-csv'), rec.get('rho_ref'))
    pub = pd.read_csv(f'{work}/_fn_ids.csv' if kern == 'nonly' else f'{work}/_k2_parent.csv', dtype={'StudyInstanceUID': str})
    n, ids_par = len(pub), pub['StudyInstanceUID'].tolist()
    if kern == 'fn':
        fso = pd.read_csv(f'{work}/submission_fleetsolo.csv', dtype={'StudyInstanceUID': str})
        F = fso[T].to_numpy(np.float64)
        ok('fleetsolo rows == placeholder ids in order', fso['StudyInstanceUID'].tolist() == ids_par)
    else:
        F = np.zeros((n, 12))
    z = np.load(f'{work}/_n949_probs.npz', allow_pickle=False)
    meta = json.loads(str(z['meta']))
    ids = [str(x) for x in z['ids']]
    PN, PL, PM = (np.asarray(z[x], np.float32) for x in ('P', 'P_plain', 'P_mirror'))
    ok(f'npz: {n} unique ids == the placeholder ids, shapes ({n}, 12)', len(ids) == n and len(set(ids)) == n
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
    ok('npz: P == (plain + mirror) / 2 exactly on the served rows', np.isfinite(PL[good]).all() and np.isfinite(PM[good]).all()
       and np.array_equal(PN[good], np.float32(0.5) * (PL[good] + PM[good])))
    dmir = np.abs(PL.astype(np.float64) - PM).max(axis=1)
    ok('npz: mirror view != plain view on every non-empty study', full.any() and (dmir[full] > 0).all(),
       f'{np.round(dmir, 4).tolist()[:6]}')
    if a.kaggle_commit:
        ok('commit: child saw 2 GPUs and loaded the model on both', meta.get('ngpu') == 2 and len(shas) == 2, meta.get('ngpu'))
        ok('commit: 0 fallback and 0 empty studies', not fb and not em)
    cs = pd.read_csv(f'{work}/_n949_sub.csv', dtype={'StudyInstanceUID': str})
    x = np.clip(PN, 0, 1); rk = x.argsort(0).argsort(0).astype(np.float64) / max(1, n - 1)
    ok("N's own csv (_n949_sub.csv) == its rankpct formula, test.csv order",
       cs.columns.tolist() == ['StudyInstanceUID', *T] and cs['StudyInstanceUID'].tolist() == ids
       and float(np.abs(cs[T].to_numpy(np.float64) - rk.astype(np.float32)).max()) <= 1e-6)
    ok('child ids in test.csv order == the placeholder (test.csv) order', ids == ids_par if kern == 'nonly' else True)
    if a.expect_n_ranks == 'log949':
        ref = k9n_check.log949_ranks()
        got = {s: cs.set_index('StudyInstanceUID').loc[s, T].to_numpy(np.float64).tolist() for s in ref if s in set(ids)}
        okr = len(ref) == 3 and set(got) == set(ref) and all(np.allclose(got[s], ref[s], atol=1e-6) for s in ref)
        det = '3/3 rows equal'
        if not okr and set(got) == set(ref):
            bad = [j for j in range(12) if any(abs(got[s][j] - ref[s][j]) > 1e-6 for s in ref)]
            pos_c = {s: i for i, s in enumerate(ids)}
            pp = {s: PN[pos_c[s]] for s in ref}
            gaps = {T[j]: float(min(abs(float(pp[x_][j]) - float(pp[y_][j])) for x_ in ref for y_ in ref if x_ < y_)) for j in bad}
            det = (f'labels differing {[T[j] for j in bad]}; smallest prob gap among the 3 studies per such label {gaps} '
                   f'(a gap < 2e-3 is fp16 / cudnn noise between images: accept by hand only then; else STOP)')
        ok("N's 3 public-study ranks == the author's public commit log (log949.txt) exactly", okr, det)
    pos = {s: i for i, s in enumerate(ids)}
    NV = PN.astype(np.float64)[[pos[s] for s in ids_par]]
    badp = np.array([s in fb or s in em for s in ids_par])
    if badp.any():
        NV[badp] = np.median(NV[~badp], axis=0)[None]
    rP = avg_rank_pct(pub[T].to_numpy(np.float64)); rF = avg_rank_pct(F); rN = avg_rank_pct(NV)
    ok('placeholder P ranks are constant (every row tied): P cannot move any blend', np.all(rP == rP[0:1]))
    if kern == 'fn' and n > 2:
        rho = [float(np.corrcoef(rF[:, j], rN[:, j])[0, 1]) for j in range(12)]
        info['rho_FN_spearman'] = {t: round(r, 4) for t, r in zip(T, rho)}

    def frame(f):
        d = pd.read_csv(f'{work}/{f}', dtype={'StudyInstanceUID': str})
        g = (d.columns.tolist() == ['StudyInstanceUID', *T] and len(d) == n and d['StudyInstanceUID'].tolist() == ids_par)
        v = d[T].to_numpy(np.float64) if g else np.full((n, 12), np.nan)
        ok(f'{f}: schema, {n} rows, placeholder (test.csv) row order, finite, in [0, 1]',
           g and np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all())
        return v
    for f in loud:
        ok(f'{f} written (N clean)', os.path.isfile(f'{work}/{f}'))
    if all(os.path.isfile(f'{work}/{f}') for f in loud + ['submission.csv']):
        v = frame('submission_n949.csv')
        ok('submission_n949.csv == the child probabilities (median-filled) in placeholder order', float(np.abs(v - NV).max()) < 1e-12)
        for w in [W3] + VAR:
            v = frame(tname(w))
            ref = avg_rank_pct(w[0] * rP + w[1] * rF + w[2] * rN)
            ref2 = avg_rank_pct(w[1] * rF + w[2] * rN)                  # the P leg removed outright
            dd, d2 = float(np.abs(v - ref).max()), float(np.abs(v - ref2).max())
            ok(f'{tname(w)} == independent rank blend P/F/N {w} == the same blend with the P leg removed', dd < 1e-9 and d2 < 1e-9,
               f'max|diff| {dd:.2e} / {d2:.2e}')
        sv = frame('submission.csv')
        ok(f'submission.csv == its fail-loud twin {tname(W3)} (P/F/N {W3})', filecmp.cmp(sub, f'{work}/{tname(W3)}', shallow=False))
        if kern == 'nonly':
            ok('N-only: submission.csv == avg_rank_pct(N) exactly', float(np.abs(sv - rN).max()) < 1e-12)
        files = st.get('files') or {}
        for f in loud + ['submission.csv']:
            ok(f'{f} unchanged since the N stage wrote it (sha256)', files.get(f) == _sha(f'{work}/{f}'))
        if kern == 'fn':
            ffiles = fst.get('files') or {}
            for f in [x for x in ffiles if x != 'submission.csv']:
                ok(f'fleet file {f} untouched by the N stage (sha256)', os.path.isfile(f'{work}/{f}') and ffiles[f] == _sha(f'{work}/{f}'))
            ok('N status: the fleet submission.csv it deleted == the fleet record', st.get('fleet_submission_sha256') == ffiles.get('submission.csv'))
        else:
            ok('N status: the submission.csv it deleted first == the FN-IDS placeholder', st.get('fleet_submission_sha256') == _sha(f'{work}/_fn_ids.csv'))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--fn_build', required=True, help='FN_BUILD.json of the build whose kernel ran')
    ap.add_argument('--kernel', required=True, choices=('fn', 'nonly'))
    ap.add_argument('--expect_n', default='auto', choices=('auto', 'clean', 'fail'))
    ap.add_argument('--expect_fleet', default='auto', choices=('auto', 'clean', 'fail'))
    ap.add_argument('--kaggle_commit', action='store_true')
    ap.add_argument('--expect_members', default=None)
    ap.add_argument('--expect_manifest', default=None)
    ap.add_argument('--expect_tta', default=None)
    ap.add_argument('--allow_interim', action='store_true')
    ap.add_argument('--expect_n_ranks', default=None, choices=(None, 'log949'))
    ap.add_argument('--skip_fleet', action='store_true', help='TEST ONLY: stubbed fleet (box A0): part A not run')
    ap.add_argument('--skip_ids', action='store_true', help='TEST ONLY: a negative that removed the FN-IDS files on purpose')
    ap.add_argument('--json', default=None)
    a = ap.parse_args(argv)
    kb = json.load(open(a.fn_build))
    res, info = [], {}
    if kb.get('do_not_push'):
        res.append(('FN build is FINAL (FN_BUILD.json do_not_push false)', False, 'TEST build: never push / submit'))
    if not a.skip_ids:
        part0(a.work, a.kernel, res, info)
    fst = {}
    if a.kernel == 'fn':
        if a.skip_fleet:
            fst = json.load(open(f'{a.work}/_k2_stage_fleet.json')) if os.path.isfile(f'{a.work}/_k2_stage_fleet.json') else {}
            res.append(('A: fleet part SKIPPED (--skip_fleet, stubbed fleet stage: TEST ONLY)', not a.kaggle_commit, ''))
        else:
            fst = part_a(a.work, a, kb, res, info)
    part_b(a.work, a, kb, res, info, fst, a.kernel)
    for name, good, detail in res:
        print(f'{"PASS" if good else "FAIL"}  {name}  {detail}')
    allok = all(g for _, g, _ in res)
    print('INFO', json.dumps(info, default=str)[:4000])
    print(f'FNCHECK {"PASS" if allok else "FAIL"} ({sum(g for _, g, _ in res)}/{len(res)})')
    if a.json:
        json.dump({'pass': allok, 'checks': res, 'info': info}, open(a.json, 'w'), indent=1, default=str)
    return 0 if allok else 1


if __name__ == '__main__':
    sys.exit(main())
