"""_dev/patch_smoke.py (K8, 2026-10-07): add the K8-KF scenarios to the laptop smoke (k8/tests/smoke_k2.py +
run_scenario.py, K7 copies). Run once.
  stage: a KF notebook (make_k2_kernel --outer_flat 0.6, strict, TTR pair, TTA) + refusals (--outer_flat with blend / solo)
  scenarios: kf_ok (map-consistent synthetic cell-28 inputs on the 24-study fake comp, live _coatnet_weight in the
  namespace: clean; k2_check --expect_outer_flat 0.6 PASS; WITHOUT it FAIL; the flat parent ranks differently from the map
  on 24 studies; wXX == the flat-parent blend), kf_perturb (side moved after the parent was built: fail loud, ABSENT),
  kf_missing (no stage-3 npz: fail loud, ABSENT), kf_drift (live map drift in the namespace: fail loud, ABSENT)."""
import os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def patch(path, edits):
    s = open(path, encoding='utf-8').read()
    assert 'K8-KF' not in s, f'{path} already patched'
    for old, new in edits:
        assert s.count(old) == 1, (path, old[:70], s.count(old))
        s = s.replace(old, new)
    open(path, 'w', encoding='utf-8', newline='\n').write(s)
    print('patched', path)


patch(os.path.join(HERE, 'tests', 'smoke_k2.py'), [
    ('''"""K5 laptop CPU smoke (scratchpad/k5) = the K3 smoke below, unchanged, run on the K5 kit''',
     '''"""K8 (scratchpad/k8, 2026-10-07): + K8-KF: a KF notebook (make_k2_kernel --outer_flat 0.6: K2_CFG OUTER_FLAT + the KF cell)
  and its refusals (--outer_flat with blend / solo), scenarios kf_ok / kf_perturb / kf_missing / kf_drift (run_scenario.py).
K5 laptop CPU smoke (scratchpad/k5) = the K3 smoke below, unchanged, run on the K5 kit'''),
    ('''    say('A+ K3 notebooks built: strict W.25, strict + 2-member filter, TTR solo, stub filter, unknown member')
''', '''    say('A+ K3 notebooks built: strict W.25, strict + 2-member filter, TTR solo, stub filter, unknown member')
    # K8-KF: the KF twin notebook + refusals
    mk.main(['--out', f'{S}/nb/rsna-sub-k8-kf', '--slug', 'rsna-sub-k8-kf', '--main_mode', 'strict',
             '--members', f'{tm["f1"]},{tm["ttr"]}', '--tta', f'{tm["f1"]}=A,{tm["ttr"]}=AB', '--outer_flat', '0.6'] + W)
    nbk = json.load(open(f'{S}/nb/rsna-sub-k8-kf/rsna_sub_k8_kf.ipynb', encoding='utf-8'))
    src = ''.join(nbk['cells'][31]['source'])
    kfc = open(f'{K2}/cells/k2_fleet_cell_kf.py', encoding='utf-8').read()
    assert src.split(chr(10), 1)[1] == kfc and "'OUTER_FLAT': 0.6" in src.split(chr(10), 1)[0], 'KF notebook cell 31'
    refused = []
    for mode in ('blend', 'solo'):
        try:
            mk.main(['--out', f'{S}/nb/rsna-sub-k8-kfbad', '--slug', 'rsna-sub-k8-kfbad', '--main_mode', mode,
                     '--outer_flat', '0.6'] + W)
        except AssertionError as e:
            refused.append('strict' in str(e))
    assert refused == [True, True], f'--outer_flat with blend / solo must be refused: {refused}'
    say('A8 K8-KF notebook built (KF cell + OUTER_FLAT 0.6, strict, TTR pair AB); --outer_flat with blend / solo refused')
'''),
    ('''       'strict_tta_expect_ok', 'strict_tta_expect_k4ds']                                                # K5b
''', '''       'strict_tta_expect_ok', 'strict_tta_expect_k4ds',                                                # K5b
       'kf_ok', 'kf_perturb', 'kf_missing', 'kf_drift']                                                 # K8-KF
'''),
])

patch(os.path.join(HERE, 'tests', 'run_scenario.py'), [
    ('''           outer=False, members=None, tta=None)''', '''           outer=False, members=None, tta=None, kf=None)                                 # K8-KF'''),
    ('''    'strict_tta_expect_k4ds': dict(nb='strict_exp', fds='ttr', expect_fleet='fail'),
''', '''    'strict_tta_expect_k4ds': dict(nb='strict_exp', fds='ttr', expect_fleet='fail'),
    # ---- K8-KF: the flat outer-map twin (OUTER_FLAT 0.6) on synthetic, map-consistent parent cell-28 inputs
    'kf_ok':            dict(nb='kf', fds='ttr_tta', members='ttr_pair', tta='pair', kf='ok'),
    'kf_perturb':       dict(nb='kf', fds='ttr_tta', kf='perturb', expect_fleet='fail'),
    'kf_missing':       dict(nb='kf', fds='ttr_tta', kf='missing', expect_fleet='fail'),
    'kf_drift':         dict(nb='kf', fds='ttr_tta', kf='drift', expect_fleet='fail'),
'''),
    ('''if cfg['parent'] == 'normal':
    par.to_csv(f'{WORK}/submission.csv', index=False)''', '''KF_MAP = {t: 0.60 for t in TARGETS}
KF_MAP.update({'ACL': 0.75, 'Medial Meniscus': 0.80, 'Lateral Meniscus': 1.00, 'Lateral OA': 0.75, 'Fracture': 0.75})
if cfg['kf']:                                                # K8-KF: parent = cell-28 map routing of synthetic inputs
    _rk = lambda x: pd.DataFrame(np.asarray(x, np.float64)).rank(method='average', pct=True).to_numpy()
    _ids = par['StudyInstanceUID'].tolist()
    _rg = np.random.default_rng(17)
    _s3 = _rg.normal(0, 1, (len(_ids), 12)); _side = _rg.uniform(0, 1, (len(_ids), 12))
    _w = np.array([KF_MAP[t] for t in TARGETS])[None, :]
    par[TARGETS] = _rk((1.0 - _w) * _rk(_s3) + _w * _side)
    os.makedirs(f'{WORK}/diagnostics')
    if cfg['kf'] != 'missing':
        np.savez(f'{WORK}/btk_v32_rest_input.npz', labels=np.array(TARGETS), study_uids=np.array(_ids), values=_s3)
    np.savez(f'{WORK}/diagnostics/btkd_parent_after_rad.npz', labels=np.array(TARGETS), study_uids=np.array(_ids), values=_s3)
    if cfg['kf'] == 'perturb':
        _j = TARGETS.index('Lateral Meniscus'); _side = _side.copy(); _side[int(np.argmin(_side[:, _j])), _j] = 0.999999
    _sd = pd.DataFrame(_side, columns=TARGETS); _sd.insert(0, 'StudyInstanceUID', _ids)
    _sd.sample(frac=1, random_state=9).to_csv(f'{WORK}/_raptor.csv', index=False)
if cfg['parent'] == 'normal':
    par.to_csv(f'{WORK}/submission.csv', index=False)'''),
    ("""      'strict_exp': f'{S}/nb/rsna-sub-k5-strictexp/rsna_sub_k5_strictexp.ipynb'}[cfg['nb']]          # K5b""",
     """      'strict_exp': f'{S}/nb/rsna-sub-k5-strictexp/rsna_sub_k5_strictexp.ipynb',                   # K5b
      'kf': f'{S}/nb/rsna-sub-k8-kf/rsna_sub_k8_kf.ipynb'}[cfg['nb']]                                  # K8-KF"""),
    ('''if cfg['noclock']:                                           # no parent T0 (Windows has no /proc either): no clock
    del g['T0']''', '''if cfg['noclock']:                                           # no parent T0 (Windows has no /proc either): no clock
    del g['T0']
if cfg['kf']:                                                # K8-KF: the parent's live cell-28 global (exercises the drift check)
    g['_coatnet_weight'] = dict(KF_MAP) if cfg['kf'] != 'drift' else {t: 0.60 for t in TARGETS}'''),
    ('''                               expect_tta=EXPECT_TTA)                                                     # K5''',
     '''                               expect_tta=EXPECT_TTA,                                                     # K5
                               expect_outer_flat=0.6 if cfg['kf'] else None)                               # K8-KF'''),
    ('''if cfg['nb'] in ('strict', 'strict_ttr', 'stubfilter', 'unknown', 'strict_exp') and not cfg['outer']:   # K5b''',
     '''if cfg['nb'] in ('strict', 'strict_ttr', 'stubfilter', 'unknown', 'strict_exp', 'kf') and not cfg['outer']:   # K5b, K8-KF'''),
    ('''if SC == 'strict_tta_oldworker':''', '''if cfg['kf']:                                                # K8-KF
    st_ = json.load(open(f'{WORK}/_k2_stage_fleet.json'))
    extra.append(('K8-KF status + kernel config carry outer_flat 0.6',
                  st_.get('outer_flat') == 0.6 and (rec.get('config') or {}).get('OUTER_FLAT') == 0.6))
    if cfg['kf'] == 'ok':
        r2, _ = k2_check.check(WORK, 'clean', 'none', expect_members=EXPECT_MEMBERS, expect_tta=EXPECT_TTA)
        bad = [n for n, c, _ in r2 if not c]
        extra.append(('K8-KF the same output WITHOUT --expect_outer_flat FAILS (absent check + map-parent recompute)',
                      any('no KF flat parent' in n for n in bad)))
        pf, kd = k2_check.kf_independent(WORK, pd.read_csv(f'{WORK}/_k2_parent.csv', dtype={'StudyInstanceUID': str}), 0.6)
        extra.append((f'K8-KF flat parent ranks differ from the map on {len(pf)} studies (MARS {kd["mars_flat_vs_parent"]:.4f})',
                      kd['flat_ranks_differ_from_map']))
        o = (rec.get('outer_flat') or {})
        extra.append(('K8-KF receipt: rebuild 0, changed labels = the 5 map labels',
                      o.get('rebuild_max_abs') == 0.0 or (o.get('rebuild_max_abs') or 1) <= 1e-12))
    else:
        want_ = {'perturb': 'KF rebuild with the parent map', 'missing': 'KF input', 'drift': 'parent outer map drift'}[cfg['kf']]
        extra.append((f'K8-KF {cfg["kf"]}: why names "{want_}"; no worker was launched',
                      want_ in why and not glob.glob(f'{WORK}/_fleet_*.npz')))
if SC == 'strict_tta_oldworker':'''),
])
