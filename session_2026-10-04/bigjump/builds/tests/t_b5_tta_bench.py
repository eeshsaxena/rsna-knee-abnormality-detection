# t_b5_tta_bench: B5 tta_bench.py goes through the WORKER path, end to end on CPU (64 px synthetic, harness only).
#  (1) store built from the 24-study fake competition's test DICOMs exactly as the prep kernel builds it
#      (fleet_common.build_study(return_bytes=True) -> study_npz_dict); synthetic folds / eval labels / gold58 file
#  (2) tta_bench on the TTR fold-0 member and the family-1 fold-0 member (the t_b2 mixed dataset, i.e. what the worker
#      served): pass A == the bigjump worker's P for the same studies, max |diff| 0, both members (10-study run, --n 10)
#  (3) pass B of one study recomputed independently (mirror_views(sample_triplets(stack, 24, +HALF))) == tta_bench's B
#  (4) every variant, bootstrap and gate field present; full 24-study run; --passes A run
#  (5) trainer-path parity: tta_bench pass A on the TTR fold-0 run's own val studies (round3 64 px store) vs the
#      trainer's OOF csv of the same checkpoint (G-P0 on synthetic data): --chunk 32 (the trainer's chunk) max |diff| 0;
#      --chunk 48 (the worker's) reported
#  (6) FIX-1 HALF = .5 - 1e-3: on a 24-slice stack P/N/B centres have no duplicate (exact +-.5 gives 0,2,2,4,..);
#      for every other stack length 1..32 HALF == .5 except the N pass's last slot on even n (n-1, the tie gave n-2)
#  (7) R-2 --passes A,B: per member none / A+B / A+B lesion-max, raw+EMA none / A+B, 2 G-TTA lines
#  (8) R-4 --max_minutes cut before the first study: no crash, summary partial, gate lines say PARTIAL
import os, sys, json, glob, shutil, subprocess
sys.dont_write_bytecode = True
from _paths import DRY, OUT, PY, SHIM
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
C64 = f'{OUT}/code64_new'
DS = f'{OUT}/ttr_ds_mixed'
assert os.path.isfile(f'{OUT}/ttr_P_mixed.npz'), 'run t_b2_ttr_train first'
sys.path.insert(0, SHIM); sys.path.insert(0, C64)
import fleet_common as fc
assert fc.STORE == 64
env = dict(os.environ, PYTHONWARNINGS='ignore', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
           CUDA_VISIBLE_DEVICES='-1', PYTHONPATH=SHIM)
COMP = f'{DRY}/comp_big'
ST = f'{OUT}/b5_store'
shutil.rmtree(ST, ignore_errors=True); os.makedirs(ST)
t = pd.read_csv(f'{COMP}/test.csv', dtype=str); ts = pd.read_csv(f'{COMP}/test_series_base.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
sids = t.StudyInstanceUID.tolist()
for s in sids:
    st = fc.build_study(ts[ts.StudyInstanceUID == s], f'{COMP}/test_series', s, return_bytes=True)
    np.savez(f'{ST}/{s}.npz', **fc.study_npz_dict(st))
TG = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
      "Baker's", 'Contusion', 'Fracture']
rng = np.random.default_rng(0)
lab = pd.DataFrame((rng.uniform(size=(len(sids), 12)) < 0.4).astype(float), columns=TG); lab.insert(0, 'StudyInstanceUID', sids)
lab.to_csv(f'{OUT}/b5_eval.csv', index=False)
pd.DataFrame({'StudyInstanceUID': sids, 'fold': 0}).to_csv(f'{OUT}/b5_folds.csv', index=False)
gold = lab.copy(); gold['source'] = ['original_gold58' if i % 2 else 'extra16' for i in range(len(sids))]
gold.to_csv(f'{OUT}/b5_gold.csv', index=False)
print(f'(1) store from {len(sids)} fake-competition DICOM studies via build_study/study_npz_dict; labels/folds/gold synthetic')
man = json.load(open(f'{DS}/fleet_manifest.json'))['checkpoints']
mem = [f'{DS}/{n}' for n in man if n.startswith(('f0_s1042ttr_', 'f0_s42_'))]
assert len(mem) == 2, man


def bench(out, extra):
    r = subprocess.run([PY, '-W', 'ignore', f'{C64}/tta_bench.py', '--store', ST, '--folds', f'{OUT}/b5_folds.csv', '--fold', '0',
                        '--eval_labels', f'{OUT}/b5_eval.csv', '--gold', f'{OUT}/b5_gold.csv', '--device', 'cpu',
                        '--boot', '200', '--out', out, '--threads', '2'] + extra, capture_output=True, text=True, env=env)
    open(f'{out}.log', 'w', encoding='utf-8').write(r.stdout + r.stderr)
    assert r.returncode == 0, r.stderr[-2500:]
    return r.stdout


o10 = bench(f'{OUT}/b5_n10', ['--n', '10'] + mem)
zb = np.load(f'{OUT}/b5_n10/tta_passes.npz'); zw = np.load(f'{OUT}/ttr_P_mixed.npz')
wn = [str(x) for x in zw['names']]; ws = list(zw['sids']); bs = list(zb['sids'])
res = {}
for p in mem:
    n = os.path.basename(p)
    A = zb[f'{n}|A']; W = zw['P'][wn.index(n)][[ws.index(s) for s in bs]]
    d = float(np.abs(A - W).max())
    print(f'(2) {n}: tta_bench pass A vs bigjump worker P on {len(bs)} studies: max|diff| {d}')
    assert d == 0.0 and np.isfinite(A).all()
    res[f'passA_vs_worker_{n}'] = d
sys.argv = ['x']
import torch
torch.set_num_threads(2)
import tta_bench as tb
m, info = tb.load_member(mem[0], 'cpu')
st = tb.study_stacks(ST, bs[0])
assert tb.HALF == 0.5 - 1e-3 and tb.PASSES['B'] == (tb.HALF, True) and tb.PASSES['N'] == (-tb.HALF, False)
x = torch.from_numpy(np.stack([fc.sample_triplets(g[0], fc.N_SLICES, tb.HALF) if g else np.zeros((24, 3, 64, 64), np.uint8)
                               for g in st])[None])
from fleet_model import mirror_views
x = mirror_views(x, tb.PLANES)
pr = torch.tensor([[g is not None for g in st]]); ct = torch.tensor([[g[1] if g else 3 for g in st]])
with torch.no_grad():
    pB = torch.sigmoid(m(x, pr, ct).float()).numpy()[0]
dB = float(np.abs(pB - zb[f'{info["name"]}|B'][0]).max())
print(f'(3) pass B recomputed independently (mirror_views(sample_triplets(+HALF))) vs tta_bench B: max|diff| {dB}')
assert dB == 0.0
s10 = json.load(open(f'{OUT}/b5_n10/tta_summary.json'))
want = {'none', 'mirror', 'offset', 'A+B', 'A+B lesion-max', 'all5'}
for p in mem:
    assert set(s10['results'][os.path.basename(p)]) == want
assert set(s10['results']['raw+EMA']) == {'none', 'A+B'} and len(s10['gate']) == 2
assert all('val_d' in r and 'p_gt0' in r['val_d'] for k, v in s10['results'].items() for vn, r in v.items()
           if not (vn == 'none' and k != 'raw+EMA'))
print('(4) n=10: variants ' + str(sorted(want)) + ' + raw+EMA; gate lines: ' +
      ' | '.join(l for l in o10.splitlines() if l.startswith('G-TTA'))[:400])
o24 = bench(f'{OUT}/b5_full', mem)
oA = bench(f'{OUT}/b5_passA', ['--passes', 'A'] + mem)
s24 = json.load(open(f'{OUT}/b5_full/tta_summary.json'))
assert s24['n_val'] == 24 and s24['n_gold58'] == 12
print(f'(4) full run: {s24["n_val"]} val + {s24["n_gold58"]} gold58 studies, {s24["minutes"]} min; --passes A run ok')
# (5) trainer OOF parity (TTR fold-0 run of t_b2 on the round3 64 px store, CPU fp32, bs 1)
ck = glob.glob(f'{OUT}/ttr_members/a/f0_s1042_ema_ttr224-*_best.pt')[0]
oof = f'{OUT}/ttr_f0/oof_f0_s1042_ema.csv'
par = {}
for chunk in (32, 48):
    r = subprocess.run([PY, '-W', 'ignore', f'{C64}/tta_bench.py', '--store', f'{DRY}/store_64_normal', '--folds',
                        f'{DRY}/folds5.csv', '--fold', '0', '--exclude', f'{DRY}/fake_gold.csv', '--eval_labels',
                        f'{DRY}/fake_v4.csv', '--device', 'cpu', '--passes', 'A', '--chunk', str(chunk), '--boot', '50',
                        '--out', f'{OUT}/b5_oof_c{chunk}', '--oof', oof, '--', ck],
                       capture_output=True, text=True, env=dict(env, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4'))
    assert r.returncode == 0, r.stderr[-2000:]
    s = json.load(open(f'{OUT}/b5_oof_c{chunk}/tta_summary.json'))
    par[chunk] = list(s['oof_parity'].values())[0]
    print(f'(5) trainer OOF ({os.path.basename(oof)}) vs tta_bench pass A, chunk {chunk}: {par[chunk]}')
assert par[32]['n'] == 2 and par[32]['max_abs'] == 0.0
# (6) FIX-1 HALF vs the exact .5 tie
st24 = np.arange(24, dtype=np.uint8)[:, None, None].repeat(4, 1).repeat(4, 2)
for off in (tb.HALF, -tb.HALF):
    c = fc.sample_triplets(st24, 24, off)[:, 1, 0, 0].astype(int)
    assert (np.diff(c) >= 1).all() and (c == np.arange(24)).all(), (off, c)
c5 = fc.sample_triplets(st24, 24, 0.5)[:, 1, 0, 0].astype(int)
assert (np.diff(c5) == 0).any(), c5                                    # the bug the fix avoids
dif = {}
for n in range(1, 33):
    stn = np.arange(n, dtype=np.uint8)[:, None, None].repeat(2, 1).repeat(2, 2)
    for sg in (1, -1):
        a = fc.sample_triplets(stn, 24, sg * 0.5)[:, 1, 0, 0].astype(int); b = fc.sample_triplets(stn, 24, sg * tb.HALF)[:, 1, 0, 0].astype(int)
        if (a != b).any():
            dif[(n, sg)] = [(int(k), int(a[k]), int(b[k])) for k in np.nonzero(a != b)[0]]
assert all(n == 24 or (sg == -1 and n % 2 == 0 and v == [(23, n - 2, n - 1)]) for (n, sg), v in dif.items()), dif
print(f'(6) HALF {tb.HALF}: n=24 P/N centres 0..23 (exact .5 gave {c5[:6].tolist()}..); other n: == .5 except the N '
      f'pass last slot on even n ({sum(1 for k in dif if k[0] != 24)} cases, n-1 instead of n-2)')
# (7) R-2: --passes A,B gives A+B and the gate
oAB = bench(f'{OUT}/b5_AB', ['--passes', 'A,B'] + mem)
sAB = json.load(open(f'{OUT}/b5_AB/tta_summary.json'))
for p in mem:
    assert set(sAB['results'][os.path.basename(p)]) == {'none', 'A+B', 'A+B lesion-max'}, sAB['results'].keys()
assert set(sAB['results']['raw+EMA']) == {'none', 'A+B'} and len(sAB['gate']) == 2 and not sAB['partial']
for p in mem:                                                           # same numbers as the 5-pass run (A and B)
    n = os.path.basename(p)
    assert sAB['results'][n]['A+B']['val'] == s24['results'][n]['A+B']['val'], n
    assert sAB['gate'][n] == s24['gate'][n], n
gl = [l for l in oAB.splitlines() if l.startswith('G-TTA')]
assert len(gl) == 2, oAB[-800:]
print(f'(7) --passes A,B: variants {sorted(sAB["results"][os.path.basename(mem[0])])}, raw+EMA A+B, gate == 5-pass run: ' + ' | '.join(gl)[:300])
# (8) R-4: a cut before the first study
oC = bench(f'{OUT}/b5_cut', ['--passes', 'A,B', '--max_minutes', '1e-9'] + mem)
sC = json.load(open(f'{OUT}/b5_cut/tta_summary.json'))
assert sC['partial'] and sC['n_val'] == 0 and 'max_minutes' in oC and 'PARTIAL' in oC, oC[-600:]
print(f'(8) --max_minutes cut: partial {sC["partial"]}, n_val {sC["n_val"]}, gate lines flagged PARTIAL; no crash')
res.update(passB=dB, oof_parity=par, half_diff={f'{k[0]},{k[1]}': v for k, v in dif.items()})
json.dump(res, open(f'{OUT}/t_b5_tta_bench.json', 'w'), indent=1)
print('T_B5_TTA_BENCH OK')
