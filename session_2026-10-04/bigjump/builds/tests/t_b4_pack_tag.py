# t_b4_pack_tag: B4, two runs whose checkpoints have IDENTICAL file names packed together by re-keying one of them,
# and the K2 build accepting the two-member fleet.
#  (1) run A = family-1 fold-0 on the base targets (t_b2's b2_f0_new), run B = the same contract on other targets
#      (the f0_s42 vs f0_s42_v4 situation); B's EMA file is placed under A's EMA file NAME in its own folder
#  (2) the K2 pack tool (k2/code) refuses the pair; the bigjump pack tool without --tag refuses it too
#  (3) bigjump pack --tag =,f0_s42v4: both files shipped (bytes == sources), manifest order + sources map
#  (4) refusals: wrong fold/seed in the key, '_' in the suffix, entry count != member count, raw+ema of ONE run
#      re-keyed apart, the same file twice re-keyed apart
#  (5) the K2 notebook (k2/make_k2_kernel.py, fleet-only, cells 0-30 == parent) runs its fleet stage on the re-keyed
#      2-member dataset (24-study fake competition and the 3-study commit cut, 2 workers x 2 members) and
#      k2/k2_check.py passes (fleet clean, every file, blends recomputed); the same with the dataset's worker replaced
#      by the K2 worker (k2/code/fleet_worker.py, what build_k2.py ships): identical fleetsolo bytes
#  (6) real contract: k2/build_k2.py with the REAL interim box checkpoint (k2/_real/ckpt, mix50 EMA) and a same-NAME
#      second variant (same weights, args train_labels llm_labels_v4_blend.csv: a different run) re-keyed
#      PATH=f0_s42v4 -> preflight + pack + both kernels OK; bigjump pack --tag =,f0_s42v4 on the same pair OK
import os, sys, json, glob, shutil, subprocess, hashlib, filecmp
sys.dont_write_bytecode = True
from _paths import CODE, K2C, K2, DRY, OUT, PY, SHIM, stage64, relabel_fp16
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__))
env = dict(os.environ, PYTHONWARNINGS='ignore', PYTHONDONTWRITEBYTECODE='1')
C64 = f'{OUT}/code64_new'; K64 = f'{OUT}/code64_k2'
assert os.path.isfile(f'{C64}/pack_fleet_ds.py') and os.path.isfile(f'{K64}/pack_fleet_ds.py'), 'run t_b2_ttr_train first'
stage64(CODE, C64, ['fleet_common.py', 'fleet_model.py', 'fleet_model_f2.py', 'fleet_model_ttr.py', 'fleet_aug.py',
                    'fleet_train.py', 'fleet_worker.py', 'pack_fleet_ds.py', 'tta_bench.py'])
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()


def run(cmd, e=None, cwd=None):
    r = subprocess.run([PY, '-W', 'ignore'] + cmd, capture_output=True, text=True, env=e or env, cwd=cwd)
    return r.returncode, r.stdout + r.stderr


res = {}
# (1) two runs, identical file names
BASE = ['--store', f'{DRY}/store_64_normal', '--folds', f'{DRY}/folds5.csv', '--exclude', f'{DRY}/fake_gold.csv',
        '--gold', f'{DRY}/fake_gold.csv', '--bs', '1', '--accum', '1', '--workers', '0', '--device', 'cpu', '--seed', '42',
        '--backbone', 'convnext_atto.d2_in1k', '--fold', '0', '--eval_labels', f'{DRY}/fake_v4.csv', '--epochs', '2']
dB = f'{OUT}/b4_runB'
shutil.rmtree(dB, ignore_errors=True)
rc, log = run(['run_trainer.py', C64] + BASE + ['--labels', f'{DRY}/fake_v4.csv', '--min_cover', '0.5', '--out', dB], cwd=HERE)
assert rc == 0, log[-1500:]
a_ema = glob.glob(f'{OUT}/b2_f0_new/f0_s42_ema_*_best.pt')[0]; a_raw = glob.glob(f'{OUT}/b2_f0_new/f0_s42_raw_*_best.pt')[0]
b_ema = glob.glob(f'{dB}/f0_s42_ema_*_best.pt')[0]
M = f'{OUT}/b4_members'; shutil.rmtree(M, ignore_errors=True)
A = relabel_fp16(a_ema, f'{M}/runA/{os.path.basename(a_ema)}')
Araw = relabel_fp16(a_raw, f'{M}/runA/{os.path.basename(a_raw)}')
B = relabel_fp16(b_ema, f'{M}/runB/{os.path.basename(a_ema)}')          # SAME file name as A
assert os.path.basename(A) == os.path.basename(B) and sha(A) != sha(B)
ta, tb = (torch.load(p, weights_only=False)['args']['train_labels'] for p in (A, B))
print(f'(1) run A {os.path.basename(A)} [{ta}] and run B under the SAME name [{tb}]')
# (2) refusals without a re-key
rc, log = run([f'{K64}/pack_fleet_ds.py', '--out', f'{OUT}/b4_k2pack', '--store', f'{DRY}/store_64_normal', A, B])
assert rc != 0 and 'duplicate fold/seed' in log, log[-500:]
rc2, log2 = run([f'{C64}/pack_fleet_ds.py', '--out', f'{OUT}/b4_notag', '--store', f'{DRY}/store_64_normal', A, B])
assert rc2 != 0 and 'duplicate member keys' in log2, log2[-500:]
print('(2) K2 pack tool refuses the pair (duplicate fold/seed); bigjump pack without --tag refuses (duplicate member keys)')
# (3) re-key
DS = f'{OUT}/b4_ds_2members'
rc, log = run([f'{C64}/pack_fleet_ds.py', '--out', DS, '--store', f'{DRY}/store_64_normal', '--tag', '=,f0_s42v4', A, B])
assert rc == 0, log[-1500:]
man = json.load(open(f'{DS}/fleet_manifest.json'))
nA, nB = os.path.basename(A), 'f0_s42v4' + os.path.basename(A)[len('f0_s42'):]
assert man['checkpoints'] == [nA, nB] and sha(f'{DS}/{nA}') == sha(A) and sha(f'{DS}/{nB}') == sha(B)
assert man['sources'][nB]['original'] == os.path.basename(A) and man['sources'][nB]['run_dir'] == 'runB'   # FIX-5
assert man['sources'][nB]['sha256'] == sha(B)[:16] and man['sources'][nA]['sha256'] == sha(A)[:16]
assert ':' not in json.dumps(man['sources']).replace('": ', '') and '/' not in json.dumps(man['sources']), man['sources']
print(f'(3) --tag =,f0_s42v4 -> {man["checkpoints"]} (bytes == sources; sources map records the re-key)')
# (4) refusals with bad re-keys
bad = {'wrong fold': ['--tag', '=,f1_s42v4', A, B], 'wrong seed': ['--tag', '=,f0_s43v4', A, B],
       'underscore suffix': ['--tag', '=,f0_s42_v4', A, B], 'count': ['--tag', 'f0_s42v4', A, B],
       'raw+ema one run': ['--tag', '=,f0_s42raw', A, Araw], 'same file twice': ['--tag', '=,f0_s42x', A, A]}
for k, extra in bad.items():
    rc, log = run([f'{C64}/pack_fleet_ds.py', '--out', f'{OUT}/b4_bad', '--store', f'{DRY}/store_64_normal'] + extra)
    why = [l for l in log.splitlines() if 'AssertionError' in l]
    assert rc != 0 and why, (k, log[-400:])
    print(f'(4) refused [{k}]: {why[-1][:150]}')
res['refusals'] = list(bad)
# (5) the K2 notebook on the 2-member dataset
sys.path.insert(0, K2)
import make_k2_kernel as mk
NB = f'{OUT}/b4_nb/rsna-sub-k2-fleet'
shutil.rmtree(f'{OUT}/b4_nb', ignore_errors=True)
mk.main(['--out', NB, '--slug', 'rsna-sub-k2-fleet'])
nbf = glob.glob(f'{NB}/*.ipynb')[0]
DS_K2W = f'{OUT}/b4_ds_2members_k2worker'
shutil.rmtree(DS_K2W, ignore_errors=True); shutil.copytree(DS, DS_K2W)
shutil.copy2(f'{K2C}/fleet_worker.py', f'{DS_K2W}/fleet_worker.py')
comp3 = f'{OUT}/b4_comp3'
if not os.path.isdir(comp3):                              # the 3-study commit-sized cut of comp_big (as k2 smoke)
    t = pd.read_csv(f'{DRY}/comp_big/test.csv', dtype=str); ts = pd.read_csv(f'{DRY}/comp_big/test_series_base.csv', dtype=str)
    keep = t.StudyInstanceUID.tolist()[:3]
    os.makedirs(f'{comp3}/test_series')
    t[t.StudyInstanceUID.isin(keep)].to_csv(f'{comp3}/test.csv', index=False)
    ts[ts.StudyInstanceUID.isin(keep)].to_csv(f'{comp3}/test_series.csv', index=False)
    for s in keep:
        shutil.copytree(f'{DRY}/comp_big/test_series/{s}', f'{comp3}/test_series/{s}')
comp24 = f'{OUT}/b4_comp24'
if not os.path.isdir(comp24):
    shutil.copytree(f'{DRY}/comp_big', comp24, ignore=shutil.ignore_patterns('test_series_base.csv', 'train_series*'))
    shutil.copy2(f'{DRY}/comp_big/test_series_base.csv', f'{comp24}/test_series.csv')
outs = {}
for tag, ds, comp in (('bj24', DS, comp24), ('bj3', DS, comp3), ('k2w24', DS_K2W, comp24)):
    rc, log = run(['k2_cell_run.py', nbf, ds, comp, f'{OUT}/b4_cell_{tag}', '0.2' if comp == comp3 else '7.0'], cwd=HERE)
    open(f'{OUT}/b4_cell_{tag}.log', 'w', encoding='utf-8').write(log)
    line = [l for l in log.splitlines() if l.startswith('RESULT ')]
    assert rc == 0 and line, (tag, log[-2500:])
    r = json.loads(line[-1][7:]); outs[tag] = r
    fails = [l for l in log.splitlines() if l.startswith('FAIL')]
    print(f'(5) K2 cell [{tag}] k2_check {"PASS" if r["ok"] else "FAIL"} {r["passed"]}/{r["checks"]} | n_keep {r["n_keep"]} '
          f'| models {r["models"]} | manifest {r["manifest"]} | {r["why"][:90]}' + (f' | {fails}' if fails else ''))
    assert r['ok'] and r['n_keep'] == [2, 2] and r['manifest'] == [nA, nB], r
fs1, fs2 = (f'{outs[t]["work"]}/submission_fleetsolo.csv' for t in ('bj24', 'k2w24'))
assert filecmp.cmp(fs1, fs2, shallow=False)
print('(5) K2 worker (what build_k2.py ships) on the same 2-member dataset: submission_fleetsolo.csv byte-identical')
res['k2_cell'] = {k: {q: v[q] for q in ('ok', 'passed', 'checks', 'n_keep', 'manifest')} for k, v in outs.items()}
# (6) real contract
real = sorted(glob.glob(f'{K2}/_real/ckpt/*.pt'))
if real:
    RB = f'{OUT}/b4_real'; shutil.rmtree(RB, ignore_errors=True); os.makedirs(f'{RB}/v4')
    ck = torch.load(real[0], map_location='cpu', weights_only=False)
    assert ck['args']['train_labels'] != 'llm_labels_v4_blend.csv'
    ck['args']['train_labels'] = 'llm_labels_v4_blend.csv'          # a different run (D1 variant), same file name
    B_real = f'{RB}/v4/{os.path.basename(real[0])}'
    torch.save(ck, B_real); del ck
    rc, log = run([f'{K2}/build_k2.py', '--ckpt', real[0], '--ckpt', f'{B_real}=f0_s42v4', '--store', f'{K2}/_real/store',
                   '--out', f'{RB}/build', '--overwrite'])
    open(f'{RB}/build_k2.log', 'w', encoding='utf-8').write(log)
    assert rc == 0 and 'upload folder OK' in log, log[-2000:]
    summ = json.load(open(f'{RB}/build/BUILD_SUMMARY.json'))
    staged = [m['staged'] for m in summ['members']]
    up = summ['upload']['path']
    assert len(staged) == 2 and staged[1].startswith('f0_s42v4_') and all(os.path.isfile(f'{up}/{s}') for s in staged)
    print(f'(6) build_k2.py on the REAL pair (same file name, PATH=f0_s42v4): members {staged} | upload '
          f'{summ["upload"]["mb"]} MB {len(summ["upload"]["files"])} files | kernels {[k["id"] for k in summ["kernels"]]}')
    rc, log = run([f'{CODE}/pack_fleet_ds.py', '--out', f'{RB}/bj_pack', '--store', f'{K2}/_real/store', '--tag', '=,f0_s42v4',
                   real[0], B_real])
    assert rc == 0, log[-1500:]
    print('(6) bigjump pack on the REAL pair: ' + ' | '.join(l[:120] for l in log.splitlines() if l.startswith('member')))
    res['real_build'] = dict(staged=staged, mb=summ['upload']['mb'])
    shutil.rmtree(f'{RB}/build/members', ignore_errors=True)             # keep the scratch small (224 MB of copies)
    shutil.rmtree(f'{RB}/bj_pack', ignore_errors=True)
else:
    print('(6) skipped: k2/_real/ckpt empty')
json.dump(res, open(f'{OUT}/t_b4_pack_tag.json', 'w'), indent=1, default=str)
print('T_B4_PACK_TAG OK')
