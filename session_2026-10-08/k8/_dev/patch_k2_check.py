"""_dev/patch_k2_check.py (K8, 2026-10-07): applies the K8-KF + K8-INT edits to k8/k2_check.py (a K7 copy). Run once."""
import os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(HERE, 'k2_check.py')
s = open(p, encoding='utf-8').read()
assert 'K8-KF' not in s, 'already patched'


def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)


rep('''Checks: every expected file exists and every non-expected one is ABSENT''', '''K8 (scratchpad/k8, 2026-10-07):
  K8-KF  --expect_outer_flat X: the KF twin (K2_CFG['OUTER_FLAT'] = X, cells/k2_fleet_cell_kf.py). A 3-study commit (or the
         2-study A0) cannot tell the KF parent from the map parent by the blend recompute (they rank identically there),
         so the gate asserts the config, the status and the receipt: kernel config OUTER_FLAT == X, _k2_stage_fleet.json
         outer_flat == X, MAIN_MODE strict, and (clean) rec['outer_flat'] with flat == X, parent_map == the cell-28 map,
         rebuild_max_abs <= 1e-9, stage3_vs_cell26_max_abs <= 1e-9, changed_labels == the map labels != X. Then an
         INDEPENDENT rebuild from the output's own parent files (btk_v32_rest_input.npz, _raptor.csv,
         diagnostics/btkd_parent_after_rad.npz; download the full output incl. diagnostics/): stage 3 == the cell-26 copy,
         the map rebuild == _k2_parent.csv, and every wXX blend is recomputed against the FLAT parent.
         Without --expect_outer_flat every output must carry NO OUTER_FLAT (config, status, receipt): a KF kernel checked as
         a main kernel FAILS, and a main kernel checked with --expect_outer_flat FAILS.
  K8-INT --expect_tta <K5_BUILD.json> of an INTERIM build (do_not_submit: true; build_k8.py) FAILS the gate unless
         --allow_interim (TEST use only): interim 288 subsets are never submitted (NOVEL_PLAN 10-07 sec 4).
Checks: every expected file exists and every non-expected one is ABSENT''')
rep('''def wname(w):
    return f'submission_w{int(round(float(w) * 100)):02d}.csv'
''', '''def wname(w):
    return f'submission_w{int(round(float(w) * 100)):02d}.csv'


KF_PARENT_MAP = {'ACL': 0.75, 'Medial Meniscus': 0.80, 'Lateral Meniscus': 1.00, 'Lateral OA': 0.75, 'Fracture': 0.75}
KF_DEFAULT_W = 0.60                                       # parent cell 28 _coatnet_weight (parent_audit.md sec 1 row 1)


def kf_independent(work, pub, flat):
    """K8-KF: rebuild the parent final from the output's own cell-28 inputs (independent of the cell's code).
    Returns (P_flat n x 12 in parent row order, details dict); raises ValueError with the reason."""
    ids = pub['StudyInstanceUID'].astype(str).tolist()
    p_npz, p_side = f'{work}/btk_v32_rest_input.npz', f'{work}/_raptor.csv'
    p_rad = f'{work}/diagnostics/btkd_parent_after_rad.npz'
    for p in (p_npz, p_side, p_rad):
        if not os.path.isfile(p):
            raise ValueError(f'{os.path.relpath(p, work)} missing in the output (download the FULL kernel output, '
                             f'incl. diagnostics/)')

    def npz_rows(p):
        z = np.load(p, allow_pickle=False)
        u = [str(x) for x in z['study_uids']]
        if [str(x) for x in z['labels']] != TARGETS or len(set(u)) != len(u) or sorted(u) != sorted(ids):
            raise ValueError(f'{os.path.basename(p)} labels / study ids != the parent')
        pos = {x: i for i, x in enumerate(u)}
        return np.asarray(z['values'], np.float64)[[pos[x] for x in ids]]
    s3, s26 = npz_rows(p_npz), npz_rows(p_rad)
    sd = pd.read_csv(p_side, dtype={'StudyInstanceUID': str})
    if (sd.columns.tolist() != ['StudyInstanceUID', *TARGETS] or not sd['StudyInstanceUID'].is_unique
            or sorted(sd['StudyInstanceUID']) != sorted(ids)):
        raise ValueError('_raptor.csv schema / ids != the parent')
    side = sd.set_index('StudyInstanceUID').loc[ids, TARGETS].to_numpy(np.float64)
    tr = _rank(s3)

    def route(wl):
        w = np.array([wl[t] for t in TARGETS], np.float64)[None, :]
        return _rank((1.0 - w) * tr + w * side)
    par = pub[TARGETS].to_numpy(np.float64)
    full_map = {t: KF_PARENT_MAP.get(t, KF_DEFAULT_W) for t in TARGETS}
    d = dict(stage3_vs_cell26=float(np.abs(s3 - s26).max()),
             rebuild_map_vs_parent=float(np.abs(route(full_map) - par).max()))
    pf = route({t: float(flat) for t in TARGETS})
    d['mars_flat_vs_parent'] = float(np.abs(pf - par).mean())
    d['flat_ranks_differ_from_map'] = bool(np.abs(_rank(pf) - _rank(par)).max() > 0)
    return pf, d
''')
rep('''          expect_mode=None, expect_manifest=None, expect_w_main=None, expect_tta=None):
    res, info = [], {}''', '''          expect_mode=None, expect_manifest=None, expect_w_main=None, expect_tta=None, expect_outer_flat=None):
    res, info = [], {}''')
rep('''    info.update(fleet_clean=clean, fleet_why=st.get('why'), n=n, main_mode=mode, members_filter=want)
''', '''    info.update(fleet_clean=clean, fleet_why=st.get('why'), n=n, main_mode=mode, members_filter=want)
    # ---- K8-KF: the flat outer-map twin is asserted from config + status + receipt, then recomputed independently
    of_cfg, of_st, of_rec = cfg.get('OUTER_FLAT'), st.get('outer_flat'), rec.get('outer_flat')
    if expect_outer_flat is None:
        ok('no KF flat parent: OUTER_FLAT unset in the kernel config, the stage status and the receipt '
           '(a KF kernel needs --expect_outer_flat)', of_cfg is None and of_st is None and of_rec is None,
           f'config {of_cfg} status {of_st} receipt {None if of_rec is None else "present"}')
    else:
        X = float(expect_outer_flat)

        def isx(v):
            return isinstance(v, (int, float)) and not isinstance(v, bool) and abs(float(v) - X) < 1e-9
        ok(f'KF: kernel config OUTER_FLAT == {X}', isx(of_cfg), of_cfg)
        ok(f'KF: _k2_stage_fleet.json outer_flat == {X}', isx(of_st), of_st)
        ok('KF: MAIN_MODE strict (in blend mode a KF failure would restore the bare parent)', mode == 'strict', mode)
        if clean:
            r = of_rec if isinstance(of_rec, dict) else {}
            want_changed = [t for t in TARGETS if abs(KF_PARENT_MAP.get(t, KF_DEFAULT_W) - X) > 1e-12]
            full_map = {t: KF_PARENT_MAP.get(t, KF_DEFAULT_W) for t in TARGETS}
            good = (bool(r) and isx(r.get('flat')) and r.get('parent_map') == full_map
                    and float(r.get('rebuild_max_abs', 1)) <= 1e-9 and float(r.get('stage3_vs_cell26_max_abs', 1)) <= 1e-9
                    and r.get('changed_labels') == want_changed)
            ok(f'KF receipt rec[outer_flat]: flat {X}, parent_map == the cell-28 map, rebuild_max_abs and '
               f'stage3_vs_cell26_max_abs <= 1e-9, changed_labels == {want_changed}', good,
               {k: r.get(k) for k in ('flat', 'rebuild_max_abs', 'stage3_vs_cell26_max_abs', 'changed_labels')} if r else of_rec)
            try:
                pf, kd = kf_independent(work, pub, X)
            except Exception as e:
                ok('KF independent rebuild from the output (btk_v32_rest_input.npz, _raptor.csv, diagnostics/...)', False, e)
            else:
                ok('KF independent: stage 3 == cell 26 copy (<= 1e-9) and the map rebuild == _k2_parent.csv (<= 1e-9)',
                   kd['stage3_vs_cell26'] <= 1e-9 and kd['rebuild_map_vs_parent'] <= 1e-9, kd)
                r_p = _rank(pf)                                         # every wXX below is recomputed vs the FLAT parent
                info['kf'] = kd
''')
rep('''                    help="K5: NAME=A|AB[,NAME=A|AB] for every member, or a K5_BUILD.json path (its 'tta' map)")
    a = ap.parse_args(argv)
    em = [x.strip() for x in a.expect_members.split(',') if x.strip()] if a.expect_members else None
    res, info = check(a.work, a.expect_fleet, a.expect_crop, kaggle_commit=a.kaggle_commit, expect_members=em,
                      expect_mode=a.expect_mode, expect_manifest=a.expect_manifest, expect_w_main=a.expect_w_main,
                      expect_tta=parse_expect_tta(a.expect_tta))
''', '''                    help="K5: NAME=A|AB[,NAME=A|AB] for every member, or a K5_BUILD.json path (its 'tta' map)")
    ap.add_argument('--expect_outer_flat', type=float, default=None,
                    help='K8-KF: the KF twin (0.6); without it any OUTER_FLAT in the output FAILS')
    ap.add_argument('--allow_interim', action='store_true',
                    help='K8-INT: accept an INTERIM build (do_not_submit) for TEST use; never for a submission')
    a = ap.parse_args(argv)
    em = [x.strip() for x in a.expect_members.split(',') if x.strip()] if a.expect_members else None
    res, info = check(a.work, a.expect_fleet, a.expect_crop, kaggle_commit=a.kaggle_commit, expect_members=em,
                      expect_mode=a.expect_mode, expect_manifest=a.expect_manifest, expect_w_main=a.expect_w_main,
                      expect_tta=parse_expect_tta(a.expect_tta), expect_outer_flat=a.expect_outer_flat)
    if a.expect_tta and a.expect_tta.endswith('.json') and os.path.isfile(a.expect_tta):          # K8-INT
        bj = json.load(open(a.expect_tta))
        if bj.get('k8') and (bj.get('do_not_submit') or bj.get('interim')):     # K8 builds only (K7i was a deliberate interim)
            res.append((f'build is FINAL (K5_BUILD.json interim {bj.get("interim")}, do_not_submit '
                        f'{bj.get("do_not_submit")}): an INTERIM build is NEVER submitted' +
                        (' [--allow_interim: TEST USE ONLY]' if a.allow_interim else ''), bool(a.allow_interim),
                        f'sets {bj.get("sets")}'))
''')
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched', p)
