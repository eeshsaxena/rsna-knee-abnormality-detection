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
    res, info = k2_check.check(work)
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
    for name, good, detail in res:
        print(f'{"PASS" if good else "FAIL"}  {name}  {detail}', flush=True)
    rep = dict(ok=all(gd for _, gd, _ in res), passed=sum(gd for _, gd, _ in res), checks=len(res), cell_s=times,
               n=len(ids), mode='cpu' if a.cpu else f'gpu {a.gpu}', info=info, nb=a.nb,
               fleet_receipt_keys={k: (g.get('_k2_receipt') or {}).get(k) for k in
                                   ('why', 'workers_s', 'worker_min', 'worker_loop_min', 'bench',
                                    'decode_s_per_study_per_thread', 'parity', 'selection')},
               crop=(g.get('_k2c_receipt') or {}))
    json.dump(rep, open(f'{out}/report.json', 'w'), indent=1, default=str)
    print('BOXSTAGE', 'PASS' if rep['ok'] else 'FAIL', f'{rep["passed"]}/{rep["checks"]}', json.dumps(
        {'cell_s': times, 'fleet_stage_min': info.get('fleet_stage_min'), 'workers_s': info.get('workers_s'),
         'crop_cell_min': info.get('crop_cell_min'), 'oof_parity': info.get('oof_parity')}, default=str), flush=True)
    return 0 if rep['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
