#!/usr/bin/env python3
"""k5_parity.py (box, harness only; nothing here ships). K5 worker TTA parity tests on real DICOMs + real checkpoints.

  CPU (never touches the GPU; CUDA hidden, nice 19):
    CUDA_VISIBLE_DEVICES= nice -n 19 ~/miniconda3/envs/kmimic/bin/python -W ignore ~/k5_test/k5/box/k5_parity.py --mode cpu
  GPU (ONE check; run it under the shared lock, this process and every worker it starts cap themselves at 4.0 GiB):
    flock ~/jump2/gpu.lock ~/miniconda3/envs/kmimic/bin/python -W ignore ~/k5_test/k5/box/k5_parity.py --mode gpu

Inputs (read only): old worker = ~/bigjump_build/code/fleet_worker.py (== k4/code, md5 pinned), new worker =
~/k5_test/k5/code (the K5 kit), J1 = ~/k4_test/ckpt/f0_s1043_raw_ttr224-320x24_cv8835_best.pt (sha256 pinned), M1 =
~/fleet_runs/f0_s42/f0_s42_raw_cnxt320x24_cv8810_best.pt (sha256 pinned), DICOM mirror ~/kmimic/.../train_series,
~/fleet_store (tta_bench input), tta_bench.py copied (md5 pinned) from ~/bigjump_build/code into OUT/tb_code (so no
__pycache__ is ever written under ~/bigjump_build). Everything is written under --out (default ~/k5_test/parity).

Tests
 (a) A-only parity, 2 fold-0 val studies (~/k3_test/two.csv), J1 + M1: old worker == new worker with (a1) no tta key
     and (a2) explicit tta A for both: P max|diff| 0, NV / names / sids / n_keep / n_fail / n_blank equal
 (a+) mixed J1 AB + M1 A on the same 2 studies: M1 rows == old M1 rows (0), J1 pass A (npz PA) == old J1 rows (0),
     J1 P == (PA + PB) / 2 exactly (float32), PB != PA; microbench passes {J1: 2, M1: 1}
 (b) J1 AB on 20 fold-0 val studies with DICOMs on the box: worker per-study A+B (P) vs tta_bench.py run on the SAME
     20 studies (its own CLI, --passes A,B, a folds csv where only these 20 are fold 0), CPU fp32 both: max|diff| of A+B,
     A, B (gate <= 2e-4)
 (c) the worker's TTA record: modes, rows, forwards per member per pass, bench passes
 GPU mode: (g1) new worker J1 AB + M1 A and the old worker J1 + M1 on the 20 studies on the GPU (fp16 autocast):
     M1 rows new vs old, J1 PA vs old J1; (g2) J1 A+B / A / B vs ~/jump2/tta/run1/tta_passes.npz (tta_bench, same GPU,
     fp16) on the 20 studies (gate <= 2e-4); (g3) vs the CPU fp32 tta_bench of (b) (noted: fp16 vs fp32)
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
import numpy as np, pandas as pd

H = os.path.expanduser('~')
PYK = f'{H}/miniconda3/envs/kmimic/bin/python'
COMP = f'{H}/kmimic/kaggle/input/rsna-knee-abnormality-detection'
BJ = f'{H}/bigjump_build/code'
OLD_WORKER_MD5 = 'f27d3f1cd305db5602154a18d85e3ede'
TB_MD5 = {'tta_bench.py': '176851b1d241b54a283a1e85e2669e30', 'fleet_common.py': '658ab8103acdf6e4aec029a0f6ee5959',
          'fleet_model.py': '897f83b272fbb144a4d3deab773d9064', 'fleet_model_f2.py': '0122f791c82130d5df292dada522eac1',
          'fleet_model_ttr.py': 'bcdf7d153f339962b97d2736d8baed08'}
J1 = f'{H}/k4_test/ckpt/f0_s1043_raw_ttr224-320x24_cv8835_best.pt'
J1_SHA = 'dbd5bf6a20fa0683f6732e2e79cb11e73d2c4e2ec7fc22f433dec773475f8d2c'
J1_OOF = f'{H}/k4_test/oof/oof_f0_s1043_raw_cv8835.csv'
M1 = f'{H}/fleet_runs/f0_s42/f0_s42_raw_cnxt320x24_cv8810_best.pt'
M1_SHA16 = 'dc7e1495404e4e7a'
RUN1 = f'{H}/jump2/tta/run1/tta_passes.npz'
NJ, NM = os.path.basename(J1), os.path.basename(M1)
CODE = ('fleet_common.py', 'fleet_model.py', 'fleet_model_f2.py', 'fleet_model_ttr.py')
CAP_GB = 4.0
RUNNER = r'''# GPU runner (k5_parity only): cap THIS process at CAP_GB, then run the worker unchanged
import os, sys, runpy, torch
tot = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ['K5_CAP_GB']) * 2**30 / tot))
w = os.environ['FLEET_DIR'] + '/fleet_worker.py'
sys.argv = [w]
runpy.run_path(w, run_name='__main__')
'''


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def make_comp(root, ids):
    os.makedirs(f'{root}/test_series')
    trs = pd.read_csv(f'{COMP}/train_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
    pd.DataFrame({'StudyInstanceUID': ids}).to_csv(f'{root}/test.csv', index=False)
    trs[trs.StudyInstanceUID.isin(ids)].to_csv(f'{root}/test_series.csv', index=False)
    for s in ids:
        assert os.path.isdir(f'{COMP}/train_series/{s}'), s
        os.symlink(f'{COMP}/train_series/{s}', f'{root}/test_series/{s}')


def make_fd(d, worker_src, code_dir, members, tta=None):
    """FLEET_DIR: code copies + the worker + ckpt symlinks + parity npz symlinks + manifest (sources only if tta)."""
    os.makedirs(d)
    for f in CODE:
        shutil.copy2(f'{code_dir}/{f}', f'{d}/{f}')
    shutil.copy2(worker_src, f'{d}/fleet_worker.py')
    import torch
    probe = set()
    for p in members:
        os.symlink(p, f'{d}/{os.path.basename(p)}')
        probe |= set(torch.load(p, map_location='cpu', weights_only=False)['args'].get('decode_probe', {}))
    for s in sorted(probe):
        if os.path.isfile(f'{H}/fleet_store/{s}.npz'):
            os.symlink(f'{H}/fleet_store/{s}.npz', f'{d}/parity_{s}.npz')
    man = {'checkpoints': [os.path.basename(p) for p in members]}
    if tta is not None:
        man['sources'] = {os.path.basename(p): {'tta': t} for p, t in zip(members, tta)}
    json.dump(man, open(f'{d}/fleet_manifest.json', 'w'), indent=1)


def run_worker(tag, fd, comp, out, dev, log, mb=0, threads=4):
    env = {k: v for k, v in os.environ.items() if not k.startswith('FLEET_')}
    env.update(FLEET_DIR=fd, FLEET_OUT=f'{out}/{tag}.npz', RSNA_COMP_ROOT=comp, FLEET_SHARD='0', FLEET_NSHARD='1',
               FLEET_MICROBENCH=str(mb), OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=fd)
    if dev == 'cpu':
        env.update(CUDA_VISIBLE_DEVICES='', FLEET_ALLOW_CPU='1')
        cmd = ['nice', '-n', '19', PYK, '-W', 'ignore', f'{fd}/fleet_worker.py']
    else:
        env.update(CUDA_VISIBLE_DEVICES='0', K5_CAP_GB=str(CAP_GB))
        open(f'{out}/gpu_runner.py', 'w').write(RUNNER)
        cmd = [PYK, '-W', 'ignore', f'{out}/gpu_runner.py']
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    open(log, 'w').write(r.stdout + '\n--- stderr ---\n' + r.stderr)
    assert r.returncode == 0, (tag, r.returncode, r.stderr[-1500:])
    print(f'  ran {tag} on {dev} in {time.time() - t0:.0f} s: ' + ' | '.join(
        l[:150] for l in r.stdout.splitlines() if ' TTA ' in l or ' done ' in l or 'MICROBENCH' in l), flush=True)
    return dict(np.load(f'{out}/{tag}.npz', allow_pickle=False))


def rows(z, name):
    nm = [str(x) for x in z['names']]
    return {str(s): k for k, s in enumerate(z['sids'])}, nm.index(name)


def sub(z, key, name, sids):
    idx, j = rows(z, name)
    return np.stack([z[key][j, idx[s]] for s in sids]).astype(np.float64)


def same_outputs(z0, z1):
    keys = ('sids', 'names', 'NV', 'n_keep', 'n_fail', 'n_blank', 'hard_stopped')
    eq = all(np.array_equal(z0[k], z1[k]) for k in keys)
    nan_eq = bool((np.isnan(z0['P']) == np.isnan(z1['P'])).all())
    d = float(np.nanmax(np.abs(z0['P'].astype(np.float64) - z1['P'].astype(np.float64))))
    return eq and nan_eq and d == 0.0, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('cpu', 'gpu'), required=True)
    ap.add_argument('--out', default=f'{H}/k5_test/parity')
    ap.add_argument('--new_code', default=f'{H}/k5_test/k5/code')
    ap.add_argument('--n20', type=int, default=20)
    ap.add_argument('--tol', type=float, default=2e-4)
    a = ap.parse_args()
    res = []

    def ok(name, cond, detail=''):
        res.append((name, bool(cond), str(detail)[:400]))
        print(f'{"PASS" if cond else "FAIL"}  {name}  {str(detail)[:400]}', flush=True)

    out = a.out
    os.makedirs(out, exist_ok=True)
    # ---- inputs, pinned
    assert md5(f'{BJ}/fleet_worker.py') == OLD_WORKER_MD5, 'old worker changed'
    assert sha256(J1) == J1_SHA, 'J1 bytes'
    assert sha256(M1)[:16] == M1_SHA16, 'M1 bytes'
    for f in CODE:
        assert md5(f'{a.new_code}/{f}') == TB_MD5[f], f'{f}: K5 code differs from bigjump (model_sha contract)'
    two = pd.read_csv(f'{H}/k3_test/two.csv', dtype=str)['StudyInstanceUID'].tolist()
    oof = pd.read_csv(J1_OOF, dtype={'StudyInstanceUID': str})['StudyInstanceUID'].tolist()
    have = [s for s in oof if os.path.isdir(f'{COMP}/train_series/{s}') and os.path.isfile(f'{H}/fleet_store/{s}.npz')]
    ids20 = two + [s for s in sorted(have) if s not in two][:a.n20 - len(two)]
    assert len(ids20) == a.n20 and set(two) <= set(have), (len(ids20), len(have))
    if a.mode == 'cpu':
        R = f'{out}/cpu'
        assert not os.path.exists(R), f'{R} exists (fresh dir per run)'
        os.makedirs(R)
        tbc = f'{R}/tb_code'
        os.makedirs(tbc)
        for f, m in TB_MD5.items():
            shutil.copy2(f'{BJ}/{f}', f'{tbc}/{f}')
            assert md5(f'{tbc}/{f}') == m, f
        make_comp(f'{R}/comp2', two); make_comp(f'{R}/comp20', ids20)
        nw = f'{a.new_code}/fleet_worker.py'
        make_fd(f'{R}/fd_old', f'{BJ}/fleet_worker.py', BJ, [M1, J1])
        make_fd(f'{R}/fd_new_none', nw, a.new_code, [M1, J1])
        make_fd(f'{R}/fd_new_A', nw, a.new_code, [M1, J1], ['A', 'A'])
        make_fd(f'{R}/fd_new_mix', nw, a.new_code, [M1, J1], ['A', 'AB'])
        make_fd(f'{R}/fd_new_j1ab', nw, a.new_code, [J1], ['AB'])
        json.dump(dict(two=two, ids20=ids20), open(f'{R}/ids.json', 'w'), indent=1)
        print(f'CPU parity: 2 studies {[s[-10:] for s in two]}, 20 studies (fold-0 val with DICOMs) in {R}', flush=True)
        z_old = run_worker('old2', f'{R}/fd_old', f'{R}/comp2', R, 'cpu', f'{R}/log_old2.txt')
        z_n0 = run_worker('new2_none', f'{R}/fd_new_none', f'{R}/comp2', R, 'cpu', f'{R}/log_new2_none.txt')
        z_nA = run_worker('new2_A', f'{R}/fd_new_A', f'{R}/comp2', R, 'cpu', f'{R}/log_new2_A.txt')
        z_mx = run_worker('new2_mix', f'{R}/fd_new_mix', f'{R}/comp2', R, 'cpu', f'{R}/log_new2_mix.txt', mb=1)
        # (a)
        g0, d0 = same_outputs(z_old, z_n0)
        ok('(a1) A-only: new worker (no tta key) == old k4 worker, J1 + M1, 2 studies (P max|diff| 0, NV/names/sids/'
           'n_keep/n_fail/n_blank equal)', g0 and int(z_old['n_keep']) == 2 and np.isfinite(z_old['P']).all(), f'max|diff| {d0}')
        g1, d1 = same_outputs(z_old, z_nA)
        ok('(a2) A-only: new worker (explicit tta A for both) == old k4 worker', g1, f'max|diff| {d1}')
        ok('(a3) A-only npz carries no PA/PB (outputs as before + the tta record)',
           'PA' not in z_n0 and 'PB' not in z_nA and 'tta' in z_n0 and 'tta' not in z_old, sorted(z_n0))
        # (a+) mixed
        s2 = [str(x) for x in z_old['sids']]
        dm1 = float(np.abs(sub(z_mx, 'P', NM, s2) - sub(z_old, 'P', NM, s2)).max())
        dja = float(np.abs(sub(z_mx, 'PA', NJ, s2) - sub(z_old, 'P', NJ, s2)).max())
        idx, j = rows(z_mx, NJ)
        pa, pb, p = (np.stack([z_mx[k][j, idx[s]] for s in s2]) for k in ('PA', 'PB', 'P'))
        ident = bool(np.array_equal(p, (pa + pb) / 2)) and p.dtype == np.float32
        ok('(a+) mixed J1 AB + M1 A: M1 rows == old M1 rows', dm1 == 0.0, f'max|diff| {dm1}')
        ok('(a+) mixed: J1 pass A (PA) == old J1 rows', dja == 0.0, f'max|diff| {dja}')
        ok('(a+) mixed: J1 P == (PA + PB) / 2 exactly in float32; PB differs from PA',
           ident and float(np.abs(pb - pa).max()) > 1e-4, f'identity {ident}, max|PB-PA| {float(np.abs(pb - pa).max()):.4f}')
        tm = json.loads(str(z_mx['tta']))
        ok('(c) TTA record (mixed): modes, rows 2, forwards J1 A2 B2 / M1 A2 B0, half .499, bench passes {J1: 2, M1: 1}',
           tm['modes'] == {NM: 'A', NJ: 'AB'} and tm['rows'] == 2 and tm['fwd'] == {NM: {'A': 2, 'B': 0}, NJ: {'A': 2, 'B': 2}}
           and abs(tm['half'] - 0.499) < 1e-12 and tm['bench_passes'] == {NM: 1, NJ: 2}
           and set(json.loads(str(z_mx['bench']))) == {NM, NJ}, tm)
        # (b) 20 studies: worker vs tta_bench (CPU fp32 both)
        z20 = run_worker('new20_j1ab', f'{R}/fd_new_j1ab', f'{R}/comp20', R, 'cpu', f'{R}/log_new20_j1ab.txt')
        allst = sorted(os.path.basename(f)[:-4] for f in os.listdir(f'{H}/fleet_store') if f.endswith('.npz'))
        pd.DataFrame({'StudyInstanceUID': allst, 'fold': [0 if s in set(ids20) else 1 for s in allst]}).to_csv(
            f'{R}/folds_20.csv', index=False)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
        t0 = time.time()
        rb = subprocess.run(['nice', '-n', '19', PYK, '-W', 'ignore', f'{tbc}/tta_bench.py', J1, '--store', f'{H}/fleet_store',
                             '--folds', f'{R}/folds_20.csv', '--fold', '0', '--eval_labels',
                             f'{H}/rsna_train/llm_labels_v4_blend.csv', '--passes', 'A,B', '--device', 'cpu', '--boot', '50',
                             '--threads', '4', '--out', f'{R}/tb20'], env=env, capture_output=True, text=True)
        open(f'{R}/log_tb20.txt', 'w').write(rb.stdout + '\n--- stderr ---\n' + rb.stderr)
        assert rb.returncode == 0, rb.stderr[-1500:]
        print(f'  ran tta_bench (CPU, its own CLI) on the 20 studies in {time.time() - t0:.0f} s', flush=True)
        tb = np.load(f'{R}/tb20/tta_passes.npz')
        tsids = [str(x) for x in tb['sids']]
        ok('(b) tta_bench scored exactly the same 20 studies', sorted(tsids) == sorted(ids20), len(tsids))
        ti = {s: k for k, s in enumerate(tsids)}
        tA = np.stack([tb[f'{NJ}|A'][ti[s]] for s in ids20]); tB = np.stack([tb[f'{NJ}|B'][ti[s]] for s in ids20])
        tAB = (tA + tB) / 2                                                # == tta_bench variants()['A+B'] (float32)
        wP, wA, wB = (sub(z20, k, NJ, ids20) for k in ('P', 'PA', 'PB'))
        dP, dA, dB = (float(np.abs(x - y.astype(np.float64)).max()) for x, y in ((wP, tAB), (wA, tA), (wB, tB)))
        ok(f'(b) J1 A+B worker == tta_bench per study on {len(ids20)} studies (CPU fp32 both): max|diff| <= {a.tol}',
           dP <= a.tol and np.isfinite(wP).all(), f'A+B {dP:.3g} | pass A {dA:.3g} | pass B {dB:.3g}')
        t20 = json.loads(str(z20['tta']))
        ok('(c) TTA record (20 studies, J1 AB): rows 20, forwards A 20 B 20', t20['rows'] == len(ids20)
           and t20['fwd'] == {NJ: {'A': len(ids20), 'B': len(ids20)}} and t20['modes'] == {NJ: 'AB'}, t20)
        o = pd.read_csv(J1_OOF, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
        T = [c for c in o.columns]
        doof = float(np.abs(wA - o.loc[ids20, T].to_numpy(np.float64)).max())
        ok('(info) J1 pass A vs its trainer OOF (GPU fp16 trainer vs CPU fp32 worker) < 2e-2', doof < 2e-2, f'{doof:.3g}')
        ab_vs_a = float(np.abs(wP - wA).max())
        print(f'INFO A+B vs pass A on the 20 studies: max|diff| {ab_vs_a:.4f} (TTA moves probabilities this much)')
        rep = dict(mode='cpu', ids=dict(two=two, ids20=ids20), d_a1=d0, d_a2=d1, d_mix_m1=dm1, d_mix_j1A=dja,
                   d_b=dict(AB=dP, A=dA, B=dB), oof_passA=doof, ab_vs_a=ab_vs_a, tta_mix=tm, tta_20=t20)
    else:
        R = f'{out}/gpu'
        assert not os.path.exists(R), f'{R} exists'
        os.makedirs(R)
        q = subprocess.run(['nvidia-smi', '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True).stdout.splitlines()[0].split(',')
        fr, tot = float(q[0]) / 1024, float(q[1]) / 1024              # no CUDA context in this parent process
        print(f'GPU free {fr:.1f} / {tot:.1f} GB; workers capped at {CAP_GB} GiB each, run one at a time', flush=True)
        assert fr >= CAP_GB + 0.5, 'not enough free GPU memory: refusing (the trainer needs the headroom)'
        make_comp(f'{R}/comp20', ids20)
        nw = f'{a.new_code}/fleet_worker.py'
        make_fd(f'{R}/fd_old', f'{BJ}/fleet_worker.py', BJ, [M1, J1])
        make_fd(f'{R}/fd_new_mix', nw, a.new_code, [M1, J1], ['A', 'AB'])
        zo = run_worker('gpu_old20', f'{R}/fd_old', f'{R}/comp20', R, 'cuda', f'{R}/log_gpu_old20.txt', mb=3)
        zn = run_worker('gpu_mix20', f'{R}/fd_new_mix', f'{R}/comp20', R, 'cuda', f'{R}/log_gpu_mix20.txt', mb=3)
        dm1 = float(np.abs(sub(zn, 'P', NM, ids20) - sub(zo, 'P', NM, ids20)).max())
        dja = float(np.abs(sub(zn, 'PA', NJ, ids20) - sub(zo, 'P', NJ, ids20)).max())
        ok('(g1) GPU: new worker M1 (A) rows == old worker M1 rows (fp16 autocast, same GPU)', dm1 <= 1e-6, f'{dm1:.3g}')
        ok('(g1) GPU: new worker J1 pass A == old worker J1 rows', dja <= 1e-6, f'{dja:.3g}')
        r1 = np.load(RUN1)
        rs = {str(s): k for k, s in enumerate(r1['sids'])}
        rA = np.stack([r1[f'{NJ}|A'][rs[s]] for s in ids20]); rB = np.stack([r1[f'{NJ}|B'][rs[s]] for s in ids20])
        rAB = (rA + rB) / 2
        wP, wA, wB = (sub(zn, k, NJ, ids20) for k in ('P', 'PA', 'PB'))
        dP, dA, dB = (float(np.abs(x - y.astype(np.float64)).max()) for x, y in ((wP, rAB), (wA, rA), (wB, rB)))
        ok(f'(g2) GPU: J1 A+B worker == tta_bench run1 (GPU fp16, ~/jump2/tta/run1) per study: max|diff| <= {a.tol}',
           dP <= a.tol, f'A+B {dP:.3g} | pass A {dA:.3g} | pass B {dB:.3g}')
        cp = f'{out}/cpu/tb20/tta_passes.npz'
        if os.path.isfile(cp):
            tb = np.load(cp); ti = {str(s): k for k, s in enumerate(tb['sids'])}
            tAB = np.stack([(tb[f'{NJ}|A'][ti[s]] + tb[f'{NJ}|B'][ti[s]]) / 2 for s in ids20])
            d32 = float(np.abs(wP - tAB.astype(np.float64)).max())
            print(f'INFO (g3) GPU fp16 worker A+B vs CPU fp32 tta_bench A+B: max|diff| {d32:.3g} (precision, not a gate)')
        else:
            d32 = None
        bo, bn = json.loads(str(zo['bench'])), json.loads(str(zn['bench']))
        tn = json.loads(str(zn['tta']))
        ok('(c) GPU microbench: J1 timed over 2 passes, M1 over 1 (bench_passes), ms/study recorded',
           tn['bench_passes'] == {NM: 1, NJ: 2} and set(bn) == {NM, NJ} and all(v > 0 for v in bn.values()),
           f'old {bo} new {bn}')
        print(f'INFO microbench V100 (shared with a trainer, fp16): old {bo} | new {bn} | J1 AB/A ratio '
              f'{bn[NJ] / bo[NJ]:.2f}', flush=True)
        rep = dict(mode='gpu', ids20=ids20, d_m1=dm1, d_j1A=dja, d_run1=dict(AB=dP, A=dA, B=dB), d_fp16_fp32=d32,
                   bench_old=bo, bench_new=bn, tta=tn)
    allok = all(g for _, g, _ in res)
    rep.update(ok=allok, checks=res)
    json.dump(rep, open(f'{R}/report.json', 'w'), indent=1, default=str)
    print(f'K5PARITY {a.mode.upper()} {"PASS" if allok else "FAIL"} {sum(g for _, g, _ in res)}/{len(res)}', flush=True)
    return 0 if allok else 1


if __name__ == '__main__':
    sys.exit(main())
