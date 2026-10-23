# dev helper (not shipped): k5/box/k2_box_stage.py = the K4 harness + K5 TTA handling (edited in place, idempotent guard)
import os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = f'{HERE}/box/k2_box_stage.py'
s = open(p, encoding='utf-8', newline='').read()
assert '\r\n' not in s and 'K5' not in s, 'already patched'


def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:70], s.count(old))
    s = s.replace(old, new)


rep('''trainer's OOF csv on the same studies (train/serve parity on real DICOMs). Report: OUT/report.json.
"""''', '''trainer's OOF csv on the same studies (train/serve parity on real DICOMs). Report: OUT/report.json.
K5 (scratchpad/k5): --expect_tta NAME=A|AB[,..] or a K5_BUILD.json path -> k2_check.py --expect_tta (receipt TTA record);
per-member OOF parity uses the worker's PASS A row (npz 'PA') when the npz has one (an AB member's served row is
(A + B) / 2, up to ~.06 from pass A on real studies, MEASURED 10-05), and the mean-of-OOFs check uses the members' pass A
rows; new checks: for every worker npz with PA/PB, P == (PA + PB) / 2 exactly for AB members and P == PA for A members
(PB NaN), and, when the microbench ran, the receipt's bench passes == 2 for AB / 1 for A.
"""''')
rep('''    ap.add_argument('--oof_member', action='append', default=[],''', '''    ap.add_argument('--expect_tta', default=None, help='K5: NAME=A|AB[,..] or a K5_BUILD.json path')
    ap.add_argument('--oof_member', action='append', default=[],''')
rep('''    res, info = k2_check.check(work, expect_manifest=man_p if has_src else None)       # K3c''',
    '''    ett = k2_check.parse_expect_tta(a.expect_tta)                                       # K5
    res, info = k2_check.check(work, expect_manifest=man_p if has_src else None, expect_tta=ett)       # K3c, K5''')
rep('''            for z in zs:
                names = [str(x) for x in z['names']]
                if nm not in names:
                    continue
                for k, sid in enumerate(z['sids']):
                    rows[str(sid)] = z['P'][names.index(nm), k].astype(np.float64)''',
    '''            for z in zs:
                names = [str(x) for x in z['names']]
                if nm not in names:
                    continue
                key = 'PA' if 'PA' in z.files else 'P'                  # K5: pass A row (== P for an A member)
                for k, sid in enumerate(z['sids']):
                    rows[str(sid)] = z[key][names.index(nm), k].astype(np.float64)''')
rep('''            res.append((f'member {nm} vs its trainer OOF on {len(common)} studies: max|diff| {d:.2e} < 2e-2', d < 2e-2 and len(common) > 0, ''))''',
    '''            res.append((f'member {nm} (pass A) vs its trainer OOF on {len(common)} studies: max|diff| {d:.2e} < 2e-2',
                        d < 2e-2 and len(common) > 0, ''))''')
rep('''        if len(oofs) >= 2 and os.path.isfile(f'{work}/submission_fleetsolo.csv'):
            f = pd.read_csv(f'{work}/submission_fleetsolo.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
            common = [s for s in f.index if all(s in o.index for o in oofs.values())]
            if common:
                mo = np.mean([o.loc[common, T].to_numpy(np.float64) for o in oofs.values()], axis=0)
                d = float(np.abs(f.loc[common, T].to_numpy(np.float64) - mo).max())
                info['oof_mean_parity'] = dict(studies=len(common), max_abs=d)
                res.append((f'fleet solo vs raw mean of the {len(oofs)} member OOFs on {len(common)} studies: max|diff| {d:.2e} < 2e-2',
                            d < 2e-2, ''))''',
    '''        tta_on = any('PA' in z.files for z in zs)                       # K5: AB members serve (A + B) / 2
        if len(oofs) >= 2 and (tta_on or os.path.isfile(f'{work}/submission_fleetsolo.csv')):
            if tta_on:                                                  # members' pass A mean vs the OOF mean
                f = pd.DataFrame({s: np.mean([rows_all[nm][s] for nm in oofs], axis=0) for s in
                                  set.intersection(*[set(rows_all[nm]) for nm in oofs])}, index=T).T
            else:
                f = pd.read_csv(f'{work}/submission_fleetsolo.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
            common = [s for s in f.index if all(s in o.index for o in oofs.values())]
            if common:
                mo = np.mean([o.loc[common, T].to_numpy(np.float64) for o in oofs.values()], axis=0)
                d = float(np.abs(f.loc[common, T].to_numpy(np.float64) - mo).max())
                info['oof_mean_parity'] = dict(studies=len(common), max_abs=d, pass_a_rows=tta_on)
                res.append((f'{"mean of the members pass A rows" if tta_on else "fleet solo"} vs raw mean of the '
                            f'{len(oofs)} member OOFs on {len(common)} studies: max|diff| {d:.2e} < 2e-2', d < 2e-2, ''))''')
rep('''    for name, good, detail in res:
        print(f'{"PASS" if good else "FAIL"}  {name}  {detail}', flush=True)''',
    '''    # K5: served row identity per member from the worker npz (all studies), and the microbench pass count
    import glob as _glob
    zs = [np.load(f) for f in sorted(_glob.glob(f'{work}/_fleet_*.npz'))]
    if zs and any('PA' in z.files for z in zs):
        rec_t = (g.get('_k2_receipt') or {}).get('tta') or {}
        modes = rec_t.get('modes') or {}
        dmax, okid = 0.0, bool(modes)
        for z in zs:
            for j, nm in enumerate(str(x) for x in z['names']):
                P_, A_, B_ = z['P'][j], z['PA'][j], z['PB'][j]
                fin = np.isfinite(P_).all(axis=1)
                if modes.get(nm) == 'AB':
                    okid &= bool(np.array_equal(P_[fin], (A_[fin] + B_[fin]) / 2)) and np.isfinite(B_[fin]).all()
                else:
                    okid &= bool(np.array_equal(P_[fin], A_[fin])) and np.isnan(B_).all()
                if fin.any():
                    dmax = max(dmax, float(np.abs(P_[fin].astype(np.float64) - A_[fin]).max()))
        res.append((f'K5 npz identity: AB members P == (PA + PB) / 2 exactly, A members P == PA (PB NaN); modes {modes}; '
                    f'max|served - pass A| {dmax:.3g}', okid, ''))
        b = [x for x in ((g.get('_k2_receipt') or {}).get('bench') or []) if x]
        if b:
            want_bp = {nm: (2 if m == 'AB' else 1) for nm, m in modes.items()}
            res.append((f'K5 microbench timed {want_bp} passes per member', rec_t.get('bench_passes') == want_bp,
                        f'{rec_t.get("bench_passes")} bench {b}'))
    for name, good, detail in res:
        print(f'{"PASS" if good else "FAIL"}  {name}  {detail}', flush=True)''')
rep('''                                   ('why', 'workers_s', 'worker_min', 'worker_loop_min', 'bench',
                                    'decode_s_per_study_per_thread', 'parity', 'selection')},''',
    '''                                   ('why', 'workers_s', 'worker_min', 'worker_loop_min', 'bench',
                                    'decode_s_per_study_per_thread', 'parity', 'selection', 'tta')},''')
import ast
ast.parse(s)
open(p, 'w', encoding='utf-8', newline='').write(s)
print('ok')
