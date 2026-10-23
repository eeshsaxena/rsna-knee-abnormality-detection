#!/usr/bin/env python3
"""k2_box_stage.py (box, harness only): run the APPENDED cells of a built K2 notebook (exact text; /kaggle prefixes
rewritten) over real train studies laid out as a fake test set, with the real fleet / crop datasets and real DICOMs,
then check every output with k2_check.py. The parent is NOT run: its submission.csv is a stand-in (any id-complete
file with the 12 labels; --parent_csv takes a real parent output, e.g. a box mimic run's localLB_parent_preds.csv).

  GPU (when the V100 has room; nothing else changes):
    ~/miniconda3/envs/kmimic/bin/python ~/k2_test/k2/box/k2_box_stage.py --nb NB.ipynb --ids ~/k2_test/gold10.csv \
        --fleet_ds ~/k2_test/ds/rsna-knee-cnxt-fleet [--crop_ds ~/k2_test/ds/rsna-knee-cropaug-members] \
        --out ~/k2_test/runs/stage_gold10 --age_h 7.9 --min_free_gb 6
  CPU (tiny, never touches the GPU: CUDA hidden, workers forced to CPU, 4 threads each, nice 19):
    ... --cpu --ids ~/k2_test/two.csv --out ~/k2_test/runs/stage_cpu2

Checks (k2_check.py): files written / ABSENT per stage outcome, schema, parent row order, finite, [0, 1], every blend
== an independent recomputation, fleet files untouched by the crop stage. Extra: --oof compares the fleet solo with the
trainer's OOF csv on the same studies (train/serve parity on real DICOMs). Report: OUT/report.json.
K5 (scratchpad/k5): --expect_tta NAME=A|AB[,..] or a K5_BUILD.json path -> k2_check.py --expect_tta (receipt TTA record);
per-member OOF parity uses the worker's PASS A row (npz 'PA') when the npz has one (an AB member's served row is
(A + B) / 2, up to ~.06 from pass A on real studies, MEASURED 10-05), and the mean-of-OOFs check uses the members' pass A
rows; new checks: for every worker npz with PA/PB, P == (PA + PB) / 2 exactly for AB members and P == PA for A members
(PB NaN), and, when the microbench ran, the receipt's bench passes == 2 for AB / 1 for A.
K8 (scratchpad/k8, 2026-10-07):
  K8-KF --kf_parent ok|perturb|missing: the A0 never runs the parent, so for the KF twin (OUTER_FLAT) the parent's cell-28
        inputs are SYNTHESIZED for the A0 studies: btk_v32_rest_input.npz (stage 3, random), its cell-26 copy
        diagnostics/btkd_parent_after_rad.npz (identical values), _raptor.csv (side, random) and submission.csv = the
        cell-28 routing rank((1 - w_l) rank(S3) + w_l side) with the parent MAP (ACL .75, MedMen .80, LatMen 1.00,
        LatOA .75, Fracture .75, else .60), so the KF self-check can pass. perturb: one Lateral Meniscus side cell is moved
        to the top AFTER the parent was built (the KF rebuild check must fail -> strict deletes submission.csv);
        missing: no stage-3 npz (KF input missing -> fail loud). --expect_outer_flat X goes to k2_check.check.
"""
import argparse, json, os, shutil, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
K2 = os.path.dirname(HERE)
sys.path.insert(0, K2)
HOME = os.path.expanduser('~')
MIRROR_COMP = f'{HOME}/kmimic/kaggle/input/rsna-knee-abnormality-detection'


def gpu_free_mib():
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=30)
        return [int(x) for x in r.stdout.split()]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nb', required=True); ap.add_argument('--ids', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--fleet_ds', required=True); ap.add_argument('--crop_ds', default=None)
    ap.add_argument('--parent_csv', default=None); ap.add_argument('--age_h', type=float, default=7.9)
    ap.add_argument('--cpu', action='store_true'); ap.add_argument('--gpu', default='0')
    ap.add_argument('--min_free_gb', type=float, default=6.0); ap.add_argument('--oof', default=None)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--kf_parent', default=None, choices=('ok', 'perturb', 'missing'),
                    help='K8-KF: synthesize the parent cell-28 KF inputs (see the header)')
    ap.add_argument('--expect_outer_flat', type=float, default=None, help='K8-KF: k2_check --expect_outer_flat')
    ap.add_argument('--expect_tta', default=None, help='K5: NAME=A|AB[,..] or a K5_BUILD.json path')
    ap.add_argument('--oof_member', action='append', default=[],
                    help='K3: NAME=CSV, compare member NAME (from the worker npz) with its trainer OOF csv (repeatable); '
                         'with 2+ members the fleet solo is also compared with the raw mean of the OOFs')
    a = ap.parse_args()
    if a.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        free = gpu_free_mib()
        idx = int(a.gpu)
        if not free or free[idx] < a.min_free_gb * 1024:
            raise SystemExit(f'GPU {idx} free {free} MiB < --min_free_gb {a.min_free_gb}: training still running? not started')
        os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    os.environ['OMP_NUM_THREADS'] = str(a.threads); os.environ['MKL_NUM_THREADS'] = str(a.threads)
    for k in ('CROP_FLEET_WORK', 'CROP_FLEET_OVERRIDES', 'CROP_FLEET_INPUT', 'FLEET_ALLOW_CPU'):
        os.environ.pop(k, None)
    import numpy as np, pandas as pd
    import torch
    torch.set_num_threads(a.threads)
    import k2_check
    T = k2_check.TARGETS
    out = os.path.abspath(os.path.expanduser(a.out))
    if os.path.exists(out):
        raise SystemExit(f'--out {out} exists (use a fresh dir)')
    comp, inp, work = f'{out}/comp', f'{out}/input', f'{out}/working'
    os.makedirs(f'{comp}/test_series'); os.makedirs(f'{inp}/datasets/eeshsaxena'); os.makedirs(work)
    ids = sorted(pd.read_csv(a.ids, dtype=str)['StudyInstanceUID'].tolist())
    trs = pd.read_csv(f'{MIRROR_COMP}/train_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
    miss = [s for s in ids if not os.path.isdir(f'{MIRROR_COMP}/train_series/{s}')]
    assert not miss, f'no DICOMs on the box for {miss[:3]}'
    pd.DataFrame({'StudyInstanceUID': ids}).to_csv(f'{comp}/test.csv', index=False)
    trs[trs.StudyInstanceUID.isin(ids)].to_csv(f'{comp}/test_series.csv', index=False)
    for s in ids:                                        # read-only symlinks into the mirror (never written)
        os.symlink(f'{MIRROR_COMP}/train_series/{s}', f'{comp}/test_series/{s}')
    os.symlink(os.path.abspath(os.path.expanduser(a.fleet_ds)), f'{inp}/datasets/eeshsaxena/rsna-knee-cnxt-fleet')
    if a.crop_ds:
        os.symlink(os.path.abspath(os.path.expanduser(a.crop_ds)), f'{inp}/datasets/eeshsaxena/rsna-knee-cropaug-members')
    if a.parent_csv:
        par = pd.read_csv(a.parent_csv, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids].reset_index()
        par[T] = par[T].rank(pct=True)
    else:                                                # stand-in parent: uniform ranks (values are irrelevant here)
        par = pd.DataFrame(np.random.default_rng(7).uniform(0, 1, (len(ids), 12)), columns=T)
        par.insert(0, 'StudyInstanceUID', ids); par[T] = par[T].rank(pct=True)
    if a.kf_parent:                                      # K8-KF: parent = the cell-28 map routing of synthetic inputs
        kmap = {'ACL': 0.75, 'Medial Meniscus': 0.80, 'Lateral Meniscus': 1.00, 'Lateral OA': 0.75, 'Fracture': 0.75}
        rg = np.random.default_rng(11)
        s3 = rg.normal(0, 1, (len(ids), 12)); side = rg.uniform(0, 1, (len(ids), 12))
        rk = lambda x: pd.DataFrame(np.asarray(x, np.float64)).rank(method='average', pct=True).to_numpy()
        w = np.array([kmap.get(t, 0.60) for t in T], np.float64)[None, :]
        par = pd.DataFrame(rk((1.0 - w) * rk(s3) + w * side), columns=T); par.insert(0, 'StudyInstanceUID', ids)
        os.makedirs(f'{work}/diagnostics')
        if a.kf_parent != 'missing':
            np.savez(f'{work}/btk_v32_rest_input.npz', labels=np.array(T), study_uids=np.array(ids), values=s3)
        np.savez(f'{work}/diagnostics/btkd_parent_after_rad.npz', labels=np.array(T), study_uids=np.array(ids), values=s3)
        if a.kf_parent == 'perturb':                     # after the parent was built: the rebuild must now disagree
            j = T.index('Lateral Meniscus'); side = side.copy(); side[int(np.argmin(side[:, j])), j] = 0.999999
        sdf = pd.DataFrame(side, columns=T); sdf.insert(0, 'StudyInstanceUID', ids)
        sdf.sample(frac=1, random_state=5).to_csv(f'{work}/_raptor.csv', index=False)
    par = par.sample(frac=1, random_state=3).reset_index(drop=True)          # parent order != test.csv order
    par[['StudyInstanceUID', *T]].to_csv(f'{work}/submission.csv', index=False)
    nb = json.load(open(os.path.expanduser(a.nb), encoding='utf-8'))
    cells = [(c.get('metadata', {}).get('k2', f'cell{i}'), ''.join(c['source'])) for i, c in enumerate(nb['cells'])][31:]
    assert cells and cells[0][0] == 'k2_fleet_stage', [c[0] for c in cells]

    def redirect(src):
        src = src.replace('/kaggle/input', inp).replace('/kaggle/working', work)
        bad = [l for l in src.splitlines() if '/kaggle/' in l and not l.lstrip().startswith('#')]
        assert not bad, bad[:3]
        return src

    os.environ['RSNA_COMP_ROOT'] = comp                  # the parent exports this (repro0946 cell 7)
    _cuda = (torch.cuda.is_available, torch.cuda.device_count, torch.cuda.empty_cache)
    if a.cpu:
        os.environ['CROP_FLEET_ALLOW_CPU'] = '1'
        _Popen = subprocess.Popen

        def _popen(cmd, *pa, env=None, **k):             # fleet workers: CPU only, low priority
            env = dict(env or os.environ); env['CUDA_VISIBLE_DEVICES'] = ''; env['FLEET_ALLOW_CPU'] = '1'
            env['OMP_NUM_THREADS'] = str(a.threads)
            return _Popen(['nice', '-n', '19'] + list(cmd), *pa, env=env, **k)
        subprocess.Popen = _popen
        os.nice(19)
    g = {'__name__': '__main__', 'T0': time.time() - a.age_h * 3600, 'COMP': comp, 'ROOT': comp}
    times = {}
    for name, src in cells:
        if a.cpu and name == 'k2_fleet_stage':           # the cell counts GPUs; on --cpu it sees 1 fake device
            torch.cuda.is_available = lambda: True; torch.cuda.device_count = lambda: 1; torch.cuda.empty_cache = lambda: None
        t0 = time.time()
        try:
            exec(compile(redirect(src), name, 'exec'), g)
        except BaseException as e:
            times[name] = round(time.time() - t0, 1)
            raise SystemExit(f'EXCEPTION ESCAPED cell {name}: {e!r} (this would fail the Kaggle notebook)')
        finally:
            torch.cuda.is_available, torch.cuda.device_count, torch.cuda.empty_cache = _cuda
        times[name] = round(time.time() - t0, 1)
    man_p = os.path.join(os.path.abspath(os.path.expanduser(a.fleet_ds)), 'fleet_manifest.json')
    has_src = os.path.isfile(man_p) and isinstance(json.load(open(man_p)).get('sources'), dict)
    ett = k2_check.parse_expect_tta(a.expect_tta)                                       # K5
    res, info = k2_check.check(work, expect_manifest=man_p if has_src else None, expect_tta=ett,       # K3c, K5
                               expect_outer_flat=a.expect_outer_flat)                                  # K8-KF
    if a.oof and os.path.isfile(f'{work}/submission_fleetsolo.csv'):
        o = pd.read_csv(os.path.expanduser(a.oof), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
        f = pd.read_csv(f'{work}/submission_fleetsolo.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
        common = [s for s in f.index if s in o.index]
        cols = [c for c in T if c in o.columns]
        if common and cols:
            d = float(np.abs(f.loc[common, cols].to_numpy(np.float64) - o.loc[common, cols].to_numpy(np.float64)).max())
            info['oof_parity'] = dict(studies=len(common), max_abs=d)
            res.append((f'fleet solo vs trainer OOF on {len(common)} studies: max|diff| {d:.2e} < 2e-2 (fp16 train eval vs '
                        f'{"fp32 CPU" if a.cpu else "fp16 GPU"} serve)', d < 2e-2, ''))
    if a.oof_member:                                     # K3: per-member train/serve parity from the worker npz
        import glob as _glob
        zs = [np.load(f) for f in sorted(_glob.glob(f'{work}/_fleet_*.npz'))]
        oofs, rows_all = {}, {}
        for spec in a.oof_member:
            nm, _, csv = spec.partition('=')
            o = pd.read_csv(os.path.expanduser(csv), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
            oofs[nm] = o
            rows = {}
            for z in zs:
                names = [str(x) for x in z['names']]
                if nm not in names:
                    continue
                key = 'PA' if 'PA' in z.files else 'P'                  # K5: pass A row (== P for an A member)
                for k, sid in enumerate(z['sids']):
                    rows[str(sid)] = z[key][names.index(nm), k].astype(np.float64)
            rows_all[nm] = rows
            common = [s for s in rows if s in o.index and np.isfinite(rows[s]).all()]
            cols = [c for c in T if c in o.columns]
            d = (float(np.abs(np.stack([rows[s] for s in common]) - o.loc[common, cols].to_numpy(np.float64)).max())
                 if common and len(cols) == 12 else float('nan'))
            info.setdefault('oof_member_parity', {})[nm] = dict(studies=len(common), max_abs=d, csv=csv)
            res.append((f'member {nm} (pass A) vs its trainer OOF on {len(common)} studies: max|diff| {d:.2e} < 2e-2',
                        d < 2e-2 and len(common) > 0, ''))
        tta_on = any('PA' in z.files for z in zs)                       # K5: AB members serve (A + B) / 2
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
                            f'{len(oofs)} member OOFs on {len(common)} studies: max|diff| {d:.2e} < 2e-2', d < 2e-2, ''))
            else:                                        # K4: fold members' OOFs are on disjoint val folds
                info['oof_mean_parity'] = 'skipped: member OOFs share no study (disjoint folds); per-member parity above + npz mean below'
    # K4: fleet solo == the raw mean of EVERY member row the workers returned (serve-side identity, all studies)
    if os.path.isfile(f'{work}/submission_fleetsolo.csv'):
        import glob as _glob
        zs = [np.load(f) for f in sorted(_glob.glob(f'{work}/_fleet_*.npz'))]
        mean_rows = {}
        for z in zs:
            for k, sid in enumerate(z['sids']):
                mean_rows[str(sid)] = z['P'][:, k].astype(np.float64).mean(0)
        f = pd.read_csv(f'{work}/submission_fleetsolo.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
        common = [s for s in f.index if s in mean_rows]
        d = (float(np.abs(f.loc[common, T].to_numpy(np.float64) - np.stack([mean_rows[s] for s in common])).max())
             if common else float('nan'))
        nm = sorted({str(x) for z in zs for x in z['names']})
        info['npz_mean_parity'] = dict(studies=len(common), of=len(f), members=nm, max_abs=d)
        res.append((f'fleet solo == raw mean of the {len(nm)} worker member rows on {len(common)}/{len(f)} studies: '
                    f'max|diff| {d:.2e} < 1e-5', d < 1e-5 and len(common) == len(f) > 0, ''))
    # K5: served row identity per member from the worker npz (all studies), and the microbench pass count
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
        print(f'{"PASS" if good else "FAIL"}  {name}  {detail}', flush=True)
    rep = dict(ok=all(gd for _, gd, _ in res), passed=sum(gd for _, gd, _ in res), checks=len(res), cell_s=times,
               n=len(ids), mode='cpu' if a.cpu else f'gpu {a.gpu}', info=info, nb=a.nb,
               fleet_receipt_keys={k: (g.get('_k2_receipt') or {}).get(k) for k in
                                   ('why', 'workers_s', 'worker_min', 'worker_loop_min', 'bench',
                                    'decode_s_per_study_per_thread', 'parity', 'selection', 'tta')},
               crop=(g.get('_k2c_receipt') or {}))
    json.dump(rep, open(f'{out}/report.json', 'w'), indent=1, default=str)
    print('BOXSTAGE', 'PASS' if rep['ok'] else 'FAIL', f'{rep["passed"]}/{rep["checks"]}', json.dumps(
        {'cell_s': times, 'fleet_stage_min': info.get('fleet_stage_min'), 'workers_s': info.get('workers_s'),
         'crop_cell_min': info.get('crop_cell_min'), 'oof_parity': info.get('oof_parity'),
         'oof_member_parity': info.get('oof_member_parity'), 'oof_mean_parity': info.get('oof_mean_parity'),
         'npz_mean_parity': info.get('npz_mean_parity')},
        default=str), flush=True)
    return 0 if rep['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
