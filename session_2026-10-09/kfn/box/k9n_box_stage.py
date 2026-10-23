#!/usr/bin/env python3
"""k9n_box_stage.py (K9N, box harness only, CPU; 2026-10-08, Claude): run the APPENDED cells of a built K9N notebook
(exact text; /kaggle prefixes rewritten) over real train studies laid out as a fake HIDDEN test set
(input/competitions/rsna-knee-abnormality-detection/{test.csv, test_series.csv, sample_submission.csv, test_series/}),
with the real N weights (sha256 pinned) and, for --fleet real, the real K9 fleet dataset; then gate the output with
k9n_check.py (+ harness-only checks). The parent is NOT run: its submission.csv is a seeded stand-in (shuffled order).

  --fleet real : cells 31 (K5 fleet stage, CPU workers, as k2_box_stage.py --cpu) + 32 (N stage)
  --fleet stub : cell 32 only; the strict fleet stage's outputs are synthesized exactly as the K5 cell writes them
                 (_k2_parent.csv, submission_fleetsolo.csv, submission_w50/w40/w60.csv, submission.csv = w50,
                 _k2_stage_fleet.json with sha256 per file, fleet_receipt.json); --neg fleet_fail writes a NOT clean one
CPU: the N child is the embedded child (sha256 checked against the cell's K9N_CFG) with box-only rewrites applied by the
Popen wrapper into a sibling file: /kaggle paths -> this test dir, CUDA assert -> pass, fp16 CUDA autocast -> CPU fp32,
fork pool nproc -> 4 (Kaggle has 4 CPUs). The shipped child stays CUDA-only; nothing here changes the notebook cells.
Negatives (--neg): missing_weights | bad_sha | fleet_fail | time_skip | wall_kill | child_crash | child_sha. Each must end
with N NOT clean, submission.csv ABSENT, the N fail-loud files ABSENT, the why naming the cause, no child process left.
"""
import argparse, json, os, re, shutil, subprocess, sys, time, hashlib, glob
HERE = os.path.dirname(os.path.abspath(__file__))
K9N = os.path.dirname(HERE)
sys.path.insert(0, K9N)
HOME = os.path.expanduser('~')
MIRROR_COMP = f'{HOME}/kmimic/kaggle/input/rsna-knee-abnormality-detection'
WDS = 'nartaa/rsna-knee-publication-swa-weights-20261007'
WFILE = 'raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt'
NEG_WHY = {'missing_weights': 'not found', 'bad_sha': 'sha256', 'fleet_fail': 'fleet stage NOT clean',
           'time_skip': 'RUNTIME GUARD', 'wall_kill': 'RUNTIME GUARD: N child still running', 'child_crash': 'exited rc',
           'child_sha': 'embedded child sha256'}
CHILD_CPU = [   # (name, old, new): box-only rewrites of the N child (each must match exactly once)
    ('cpu-assert', 'assert str(device).startswith("cuda"), "Qualified inference requires CUDA/fp16"', 'pass   # BOX CPU'),
    ('cpu-autocast', 'with torch.autocast("cuda", dtype=torch.float16):', 'with torch.autocast("cpu", enabled=False):   # BOX CPU fp32'),
    ('nproc', 'nproc = max(4, min(96, (os.cpu_count() or 8) - 8))', 'nproc = 4   # BOX: Kaggle T4x2 has 4 CPUs'),
]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nb', required=True); ap.add_argument('--ids', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--tag', required=True, choices=('a', 'b'))
    ap.add_argument('--k9n_build', required=True, help='K9N_BUILD.json (box copy)')
    ap.add_argument('--weights_dir', required=True)
    ap.add_argument('--fleet', required=True, choices=('real', 'stub'))
    ap.add_argument('--fleet_ds', default=None)
    ap.add_argument('--expect_tta', default=None); ap.add_argument('--expect_manifest', default=None)
    ap.add_argument('--neg', default=None, choices=sorted(NEG_WHY))
    ap.add_argument('--age_h', type=float, default=7.0)
    ap.add_argument('--threads', type=int, default=4); ap.add_argument('--n_threads', type=int, default=8)
    a = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = str(a.threads); os.environ['MKL_NUM_THREADS'] = str(a.threads)
    for k in ('CROP_FLEET_WORK', 'CROP_FLEET_OVERRIDES', 'CROP_FLEET_INPUT', 'FLEET_ALLOW_CPU'):
        os.environ.pop(k, None)
    import numpy as np, pandas as pd
    import torch
    torch.set_num_threads(a.threads)
    import k2_check, k9n_check
    T = k2_check.TARGETS
    out = os.path.abspath(os.path.expanduser(a.out))
    if os.path.exists(out):
        raise SystemExit(f'--out {out} exists (use a fresh dir)')
    inp, work = f'{out}/input', f'{out}/working'
    comp = f'{inp}/competitions/rsna-knee-abnormality-detection'
    os.makedirs(f'{comp}/test_series'); os.makedirs(f'{inp}/datasets/eeshsaxena'); os.makedirs(work)
    ids = sorted(pd.read_csv(a.ids, dtype=str)['StudyInstanceUID'].tolist())
    trs = pd.read_csv(f'{MIRROR_COMP}/train_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
    miss = [s for s in ids if not os.path.isdir(f'{MIRROR_COMP}/train_series/{s}')]
    assert not miss, f'no DICOMs on the box for {miss[:3]}'
    pd.DataFrame({'StudyInstanceUID': ids}).to_csv(f'{comp}/test.csv', index=False)
    trs[trs.StudyInstanceUID.isin(ids)].to_csv(f'{comp}/test_series.csv', index=False)
    ss = pd.DataFrame(0.5, index=range(len(ids)), columns=T); ss.insert(0, 'StudyInstanceUID', ids)
    ss.to_csv(f'{comp}/sample_submission.csv', index=False)
    for s in ids:                                        # read-only symlinks into the mirror (never written)
        os.symlink(f'{MIRROR_COMP}/train_series/{s}', f'{comp}/test_series/{s}')
    if a.fleet == 'real':
        os.symlink(os.path.abspath(os.path.expanduser(a.fleet_ds)), f'{inp}/datasets/eeshsaxena/rsna-knee-cnxt-fleet')
    wd = f'{inp}/datasets/{WDS}'
    os.makedirs(wd)
    wsrc = os.path.abspath(os.path.expanduser(f'{a.weights_dir}/{WFILE}'))
    if a.neg == 'bad_sha':                               # same name, foreign bytes
        with open(f'{wd}/{WFILE}', 'wb') as fh:
            fh.write(b'K9N TEST: not the published checkpoint\n' * 4096)
    elif a.neg != 'missing_weights':
        os.symlink(wsrc, f'{wd}/{WFILE}')
    # ---- stand-in parent (shuffled order != test.csv order)
    par = pd.DataFrame(np.random.default_rng(7).uniform(0, 1, (len(ids), 12)), columns=T)
    par.insert(0, 'StudyInstanceUID', ids); par[T] = par[T].rank(pct=True)
    par = par.sample(frac=1, random_state=3).reset_index(drop=True)
    par[['StudyInstanceUID', *T]].to_csv(f'{work}/submission.csv', index=False)
    nb = json.load(open(os.path.expanduser(a.nb), encoding='utf-8'))
    cells = [(c.get('metadata', {}).get('k2', f'cell{i}'), ''.join(c['source'])) for i, c in enumerate(nb['cells'])][31:]
    assert [c[0] for c in cells] == ['k2_fleet_stage', 'k9n_n949_stage'], [c[0] for c in cells]
    k2cfg = eval(cells[0][1].split('\n', 1)[0][len('K2_CFG = '):].split('   #', 1)[0])     # literal dict (our build)
    ncfg_line = cells[1][1].split('\n', 1)[0]
    ncfg = eval(ncfg_line[len('K9N_CFG = '):].split('   #', 1)[0])
    if a.fleet == 'stub':                                # the strict K5 cell's outputs, synthesized
        cells = cells[1:]
        shutil.copyfile(f'{work}/submission.csv', f'{work}/_k2_parent.csv')
        pub = pd.read_csv(f'{work}/_k2_parent.csv', dtype={'StudyInstanceUID': str})
        rk = lambda x: pd.DataFrame(np.asarray(x, np.float64)).rank(method='average', pct=True).to_numpy()
        F = np.random.default_rng(13).uniform(0.01, 0.99, (len(ids), 12))
        files = {}
        if a.neg == 'fleet_fail':
            os.remove(f'{work}/submission.csv')
            clean, why = False, 'K9N TEST: stub fleet stage not clean'
        else:
            out_f = {'submission_fleetsolo.csv': F}
            for w in [k2cfg['W_MAIN']] + list(k2cfg['W_VARIANTS']):
                out_f[k2_check.wname(w)] = rk((1 - w) * rk(pub[T].to_numpy(np.float64)) + w * rk(F))
            out_f['submission.csv'] = out_f[k2_check.wname(k2cfg['W_MAIN'])]
            for f, v in out_f.items():
                df = pd.DataFrame(v, columns=T); df.insert(0, 'StudyInstanceUID', pub['StudyInstanceUID'].values)
                df.to_csv(f'{work}/{f}', index=False)
                files[f] = sha(open(f'{work}/{f}', 'rb').read())
            clean, why = True, f'K9N TEST: stub fleet stage, {len(ids)} studies'
        json.dump(dict(clean=clean, why=why, files=files, models=10, names=k2cfg.get('MEMBERS'), stage_min=1.0,
                       W_MAIN=k2cfg['W_MAIN'], W_VARIANTS=k2cfg['W_VARIANTS'], parent='_k2_parent.csv',
                       fleet_solo='submission_fleetsolo.csv' if clean else None, main_mode=k2cfg['MAIN_MODE'],
                       members_filter=k2cfg.get('MEMBERS')), open(f'{work}/_k2_stage_fleet.json', 'w'), indent=1)
        json.dump({'config': k2cfg, 'clean': clean, 'why': why}, open(f'{work}/fleet_receipt.json', 'w'), indent=1)
    if a.neg in ('time_skip', 'wall_kill', 'child_sha'):  # test-only edits of the N cell's K9N_CFG literal
        n2 = dict(ncfg)
        if a.neg == 'wall_kill':
            n2['MIN_START_MIN'] = 0.1
        if a.neg == 'child_sha':
            n2['CHILD_SHA256'] = '0' * 64
        src = cells[-1][1]
        cells[-1] = (cells[-1][0], f'K9N_CFG = {n2!r}   # BOX TEST EDIT ({a.neg})\n' + src.split('\n', 1)[1])

    def redirect(src):
        src = src.replace('/kaggle/input', inp).replace('/kaggle/working', work)
        bad = [l for l in src.splitlines() if '/kaggle/' in l and not l.lstrip().startswith('#')]
        assert not bad, bad[:3]
        return src

    os.environ['RSNA_COMP_ROOT'] = comp
    child_seen = []
    _Popen = subprocess.Popen

    def _popen(cmd, *pa, env=None, **k):
        env = dict(env or os.environ); env['CUDA_VISIBLE_DEVICES'] = ''
        cmd = list(cmd)
        if len(cmd) >= 2 and str(cmd[1]).endswith('fleet_worker.py'):          # fleet workers: CPU (k2_box_stage --cpu)
            env['FLEET_ALLOW_CPU'] = '1'; env['OMP_NUM_THREADS'] = str(a.threads)
        elif len(cmd) >= 2 and str(cmd[1]).endswith('n949_child.py'):         # the N child: box-only rewrites
            raw = open(cmd[1], 'rb').read()
            src = raw.decode('utf-8')
            info = {'path': cmd[1], 'sha256': sha(raw), 'expect': ncfg['CHILD_SHA256']}
            for name, old, new in CHILD_CPU:
                assert src.count(old) == 1, f'box rewrite {name}: anchor count {src.count(old)}'
                src = src.replace(old, new)
            src = src.replace('/kaggle/input', inp).replace('/kaggle/working', work)
            if a.neg == 'child_crash':
                anchor = '    t0 = time.time()\n    dev = "cuda" if torch.cuda.is_available() else "cpu"\n'
                assert src.count(anchor) == 1
                src = src.replace(anchor, anchor + "    raise RuntimeError('K9N TEST: injected child crash')\n")
            box = cmd[1][:-3] + '_box.py'
            open(box, 'w', encoding='utf-8').write(src)
            keep = f'{out}/n949_child_box_copy.py'                       # the cell removes its temp dir afterwards
            open(keep, 'w', encoding='utf-8').write(src)
            info['box'] = box; info['keep'] = keep; child_seen.append(info)
            cmd[1] = box
            env['OMP_NUM_THREADS'] = str(a.n_threads); env['MKL_NUM_THREADS'] = str(a.n_threads)
        return _Popen(['nice', '-n', '19'] + cmd, *pa, env=env, **k)
    subprocess.Popen = _popen
    os.nice(19)
    t0 = time.time() - a.age_h * 3600
    g = {'__name__': '__main__', 'T0': t0, 'COMP': comp, 'ROOT': comp}
    times = {}
    _cuda = (torch.cuda.is_available, torch.cuda.device_count, torch.cuda.empty_cache)
    for name, src in cells:
        torch.cuda.is_available = lambda: True; torch.cuda.device_count = lambda: 1; torch.cuda.empty_cache = lambda: None
        if name == 'k9n_n949_stage' and a.neg in ('time_skip', 'wall_kill'):
            wall = float(ncfg.get('WALL_H', 8.6)) * 3600 - float(ncfg.get('RESERVE_MIN', 2.0)) * 60
            left = 10 * 60 if a.neg == 'time_skip' else 75                  # time_skip: 10 min (< 20); wall_kill: 75 s
            g['T0'] = time.time() - wall + left
        t1 = time.time()
        try:
            exec(compile(redirect(src), name, 'exec'), g)
        except BaseException as e:
            times[name] = round(time.time() - t1, 1)
            raise SystemExit(f'EXCEPTION ESCAPED cell {name}: {e!r} (this would fail the Kaggle notebook)')
        finally:
            torch.cuda.is_available, torch.cuda.device_count, torch.cuda.empty_cache = _cuda
        times[name] = round(time.time() - t1, 1)
    subprocess.Popen = _Popen
    # ---- gate
    kb = json.load(open(os.path.expanduser(a.k9n_build)))
    expect_n = 'fail' if a.neg else 'clean'
    expect_fleet = 'fail' if a.neg == 'fleet_fail' else 'clean'
    argv = ['--work', work, '--k9n_build', os.path.expanduser(a.k9n_build), '--tag', a.tag, '--expect_n', expect_n,
            '--expect_fleet', expect_fleet, '--json', f'{out}/k9n_check.json']
    if a.fleet == 'stub':
        argv += ['--skip_fleet']
    else:
        argv += ['--expect_members', ','.join(k2cfg['MEMBERS'])]
        if a.expect_tta:
            argv += ['--expect_tta', a.expect_tta]
        if a.expect_manifest:
            argv += ['--expect_manifest', a.expect_manifest]
    print('k9n_check', ' '.join(argv[2:]), flush=True)
    rc = k9n_check.main(argv)
    chk = json.load(open(f'{out}/k9n_check.json'))
    res = [(f'k9n_check: {n}', bool(gd), d) for n, gd, d in chk['checks']
           if not (kb.get('do_not_push') and n.startswith('K9N build is FINAL'))]
    rec = g.get('_k9n_receipt') or {}
    st = json.load(open(f'{work}/_k9n_stage.json')) if os.path.isfile(f'{work}/_k9n_stage.json') else {}
    if a.neg:
        res.append((f'NEG {a.neg}: why names the cause ({NEG_WHY[a.neg]!r})', NEG_WHY[a.neg] in str(st.get('why')), st.get('why')))
        res.append((f'NEG {a.neg}: submission.csv ABSENT', not os.path.exists(f'{work}/submission.csv'), ''))
        started = bool(child_seen)
        want_started = a.neg in ('wall_kill', 'child_crash')
        res.append((f'NEG {a.neg}: child {"started" if want_started else "never started"}', started == want_started,
                    [c.get('box') for c in child_seen]))
    else:
        res.append(('child launched once with the pinned embedded bytes', len(child_seen) == 1
                    and child_seen[0]['sha256'] == child_seen[0]['expect'] == kb['child_sha256'], child_seen))
    # no child (or fork-pool) process may survive the cell
    time.sleep(2)
    left_procs = []
    for c in child_seen:
        left_procs += [p for p in subprocess.run(['pgrep', '-f', c['box']], capture_output=True, text=True).stdout.split() if p.strip()]
    res.append(('no N child / prep-pool process left after the cell (pgrep on the child path)', not left_procs, left_procs))
    if a.neg == 'child_sha':                             # the tampered config must be caught by the gate as well
        res = [(n, (not gd) if n.startswith('k9n_check: B: N config: child / source / weights sha256') else gd, d)
               for n, gd, d in res]
    # numpy reference vs scipy for the rank used by k9n_check (meta-check of the independent reference)
    from scipy.stats import rankdata
    xx = np.random.default_rng(5).integers(0, 7, (40, 12)).astype(np.float64)
    res.append(('k9n_check.avg_rank_pct == scipy rankdata(average) / n (tie-heavy matrix)',
                np.array_equal(k9n_check.avg_rank_pct(xx), np.column_stack([rankdata(xx[:, j]) / 40 for j in range(12)])), ''))
    if not a.neg and os.path.isfile(f'{work}/submission.csv'):
        # an independent numpy 3-leg reference straight from the files, using scipy ranks (third implementation)
        pub = pd.read_csv(f'{work}/_k2_parent.csv', dtype={'StudyInstanceUID': str})
        fso = pd.read_csv(f'{work}/submission_fleetsolo.csv', dtype={'StudyInstanceUID': str})
        z = np.load(f'{work}/_n949_probs.npz')
        pos = {str(s): i for i, s in enumerate(z['ids'])}
        NV = np.asarray(z['P'], np.float64)[[pos[s] for s in pub['StudyInstanceUID']]]
        rr = lambda M: np.column_stack([rankdata(M[:, j]) / len(M) for j in range(12)])
        w = ncfg['W3']
        ref = rr(w[0] * rr(pub[T].to_numpy(np.float64)) + w[1] * rr(fso[T].to_numpy(np.float64)) + w[2] * rr(NV))
        got = pd.read_csv(f'{work}/submission.csv', dtype={'StudyInstanceUID': str})
        d = float(np.abs(got[T].to_numpy(np.float64) - ref).max())
        res.append((f'submission.csv == scipy-rank 3-leg reference P/F/N {w} (max|diff| {d:.1e})', d < 1e-9
                    and got.columns.tolist() == ['StudyInstanceUID', *T] and len(got) == len(ids)
                    and got['StudyInstanceUID'].tolist() == pub['StudyInstanceUID'].tolist(), ''))
        # mirror unit test on the child's own functions (box copy, CPU): sagittal windows reverse channels, coronal /
        # axial windows flip width, mirror twice == identity, eval_windows(mirror) == gpu_windows(mirror) bit for bit
        import importlib.util
        spec = importlib.util.spec_from_file_location('n949_box_mod', child_seen[0]['keep'])
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        rng = np.random.default_rng(1)
        vol = rng.integers(0, 256, (96, 384, 384), dtype=np.uint8); mask = np.ones(96, np.uint8)
        cs = np.array(m._eval_centers(mask, 96, 94))
        tri = torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs - 1, cs, cs + 1], 1)]))
        mt = m._mirror_tri(tri, cs, 96)
        s_, _ = m._slot_index(96)
        sag = np.isin(s_[cs], (0, 1))
        ok_sag = torch.equal(mt[torch.from_numpy(sag)], tri[torch.from_numpy(sag)].flip(1))
        ok_cor = torch.equal(mt[torch.from_numpy(~sag)], tri[torch.from_numpy(~sag)].flip(-1))
        ok_inv = torch.equal(m._mirror_tri(mt, cs, 96), tri)
        gw = m.gpu_windows(vol, mask, 94, 320, 'cpu', mirror=True).numpy()
        ew = m.eval_windows(vol, mask, 94, 320, mirror=True).numpy()
        gp = m.gpu_windows(vol, mask, 94, 320, 'cpu', mirror=False).numpy()
        res.append((f'mirror unit: {int(sag.sum())} sagittal windows reverse channels, {int((~sag).sum())} coronal/axial '
                    f'flip width, mirror o mirror == id, eval_windows == gpu_windows (mirror) bit-identical, '
                    f'mirror != plain', ok_sag and ok_cor and ok_inv and np.array_equal(gw, ew) and gw.shape == (94, 3, 320, 320)
                    and not np.array_equal(gw, gp) and 0 < sag.sum() < 94, ''))
    for name, good, detail in res:
        print(f'{"PASS" if good else "FAIL"}  {name}  {str(detail)[:240] if not good else ""}', flush=True)
    ok = all(gd for _, gd, _ in res)
    rep = dict(ok=ok, passed=sum(gd for _, gd, _ in res), checks=len(res), cell_s=times, n=len(ids), neg=a.neg,
               fleet=a.fleet, tag=a.tag, why=st.get('why'), n_stage_min=rec.get('stage_min'), child_s=rec.get('child_s'),
               gpu_s_per_study=rec.get('gpu_s_per_study'), mirror_meanabs=rec.get('mirror_meanabs'),
               child_meta=rec.get('child_meta'), k9n_check_rc=rc)
    json.dump(rep, open(f'{out}/report.json', 'w'), indent=1, default=str)
    print('K9NBOX', 'PASS' if ok else 'FAIL', f'{rep["passed"]}/{rep["checks"]}', json.dumps(
        {k: rep[k] for k in ('cell_s', 'n', 'neg', 'fleet', 'tag', 'n_stage_min', 'child_s', 'gpu_s_per_study',
                             'mirror_meanabs')}, default=str), flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
