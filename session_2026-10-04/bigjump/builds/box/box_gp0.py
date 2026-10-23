# box_gp0.py (box, CPU only): gate G-P0 (BIG_JUMP_PLAN sec 4) on REAL data in the box env, before the TTR launch.
#  (1) the 4 parity studies of the K2 kit (~/fleet_store/<sid>.npz, symlinked; steven_v4 targets; 2 train / 2 val):
#      the bigjump trainer, TTR recipe flags (--model ttr --aug heavy --train_slices 16 --drop_path .2 --dropout .3
#      --ema .9995), REAL pretrained ConvNeXt-T (HF cache, offline), 1 epoch, bs 1, CPU fp32
#  (2) trainer path vs worker path: tta_bench (the worker's member build + input) pass A on the 2 val studies vs the
#      trainer's own OOF csv, raw and EMA: max |diff| at chunk 32 (the trainer's) and 48 (the worker's); finite
#  (3) model_sha of the ckpt == the registry's TTR sha, != family 1 (fa35880fca963652)
#  (4) aug_dump.py on one parity study: PNG grid, mirror check, light vs heavy item timing (12 store studies)
#  (5) tta_bench on the REAL interim family-1 member (k2/_real/ckpt mix50 EMA ep5, copied to ~/bigjump_build/ckpt)
#      through all 5 passes on the first N fold-0 val + N gold58 studies of ~/fleet_store (argv[1], default 3) vs its
#      GPU-trained OOF (~/k2_test/oof_f0_s42_ema_cv8183.csv, read only): the real-data B5 smoke
import os, sys, json, glob, shutil, subprocess, time
H = os.path.expanduser('~'); BJ = f'{H}/bigjump_build'; CODE = f'{BJ}/code'; OUT = f'{BJ}/out/gp0'
N5 = int(sys.argv[1]) if len(sys.argv) > 1 else 3
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(f'{OUT}/store'); os.makedirs(f'{OUT}/members')
import numpy as np, pandas as pd, torch
sids = sorted(open(f'{BJ}/dry/parity_sids.txt').read().split())
assert len(sids) == 4
for s in sids:
    os.symlink(f'{H}/fleet_store/{s}.npz', f'{OUT}/store/{s}.npz')
v4 = pd.read_csv(f'{H}/rsna_train/llm_labels_v4_blend.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
v4.loc[sids].to_csv(f'{OUT}/labels.csv')
pd.DataFrame({'StudyInstanceUID': sids, 'fold': [0, 0, 1, 1]}).to_csv(f'{OUT}/folds.csv', index=False)
env = dict(os.environ, PYTHONWARNINGS='ignore', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
           OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
ok = True


def sh(cmd, log):
    t0 = time.time()
    r = subprocess.run([sys.executable, '-W', 'ignore'] + cmd, capture_output=True, text=True, env=env)
    open(log, 'w').write(r.stdout + r.stderr)
    assert r.returncode == 0, (cmd[0], r.stderr[-2500:])
    return r.stdout, time.time() - t0


out, dt = sh([f'{CODE}/fleet_train.py', '--store', f'{OUT}/store', '--labels', f'{OUT}/labels.csv', '--folds', f'{OUT}/folds.csv',
              '--fold', '0', '--eval_labels', f'{OUT}/labels.csv', '--model', 'ttr', '--aug', 'heavy', '--train_slices', '16',
              '--drop_path', '0.2', '--dropout', '0.3', '--ema', '0.9995', '--epochs', '1', '--bs', '1', '--accum', '1',
              '--workers', '0', '--device', 'cpu', '--seed', '1042', '--min_cover', '0', '--out', f'{OUT}/run'], f'{OUT}/train.log')
print(f'(1) TTR trainer 1 epoch on 2+2 real studies, CPU, {dt:.0f} s: ' + [l for l in out.splitlines() if l.startswith('BJ opt-in')][0])
for l in out.splitlines():
    if l.startswith(('labels ', 'ep 0 ', '{')):
        print('    ' + l[:200])
mem = []
for tag in ('ema', 'raw'):
    p = glob.glob(f'{OUT}/run/f0_s1042_{tag}_ttr224-320x24_*_best.pt')
    assert len(p) == 1, p
    ck = torch.load(p[0], map_location='cpu', weights_only=False)
    ck['args']['amp'] = 'fp16'                                    # harness: CPU-trained, the worker admits fp16 only
    torch.save(ck, f'{OUT}/members/{os.path.basename(p[0])}'); mem.append(f'{OUT}/members/{os.path.basename(p[0])}')
    a = ck['args']
    assert int(ck['model']['ttr_size_t']) == 224 == a['ttr_size']                 # TTR-6
sys.path.insert(0, CODE); sys.dont_write_bytecode = True
import ast
tree = ast.parse(open(f'{CODE}/fleet_worker.py', encoding='utf-8').read())
node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_model_sha'][0]
ns = {}; exec(compile(ast.Module(body=[node], type_ignores=[]), 'w', 'exec'), ns)
c3 = a['model_sha'] == ns['_model_sha']('fleet_model_ttr') != ns['_model_sha']('fleet_model') == 'fa35880fca963652'
ok &= c3
print(f'{"PASS" if c3 else "FAIL"} (3) ckpt model_sha {a["model_sha"]} == TTR registry, != family 1 fa35880fca963652; '
      f'args model_mod {a["model_mod"]} ttr_size {a["ttr_size"]} aug {a["aug"]} train_slices {a["train_slices"]}')
for chunk in (32, 48):
    o, dt = sh([f'{CODE}/tta_bench.py', '--store', f'{OUT}/store', '--folds', f'{OUT}/folds.csv', '--fold', '0',
                '--eval_labels', f'{OUT}/labels.csv', '--device', 'cpu', '--passes', 'A', '--chunk', str(chunk), '--boot', '20',
                '--threads', '2', '--out', f'{OUT}/tb_c{chunk}', '--oof', f'{OUT}/run/oof_f0_s1042_ema.csv',
                f'{OUT}/run/oof_f0_s1042_raw.csv', '--'] + mem, f'{OUT}/tb_c{chunk}.log')
    s = json.load(open(f'{OUT}/tb_c{chunk}/tta_summary.json'))
    par = s['oof_parity']
    z = np.load(f'{OUT}/tb_c{chunk}/tta_passes.npz')
    fin = all(np.isfinite(z[k]).all() for k in z.files if '|' in k)
    c2 = fin and all(v['n'] == 2 for v in par.values()) and (chunk != 32 or all(v['max_abs'] == 0.0 for v in par.values()))
    ok &= c2
    print(f'{"PASS" if c2 else "FAIL"} (2) worker path (tta_bench pass A, chunk {chunk}) vs trainer OOF on 2 real val studies: '
          f'{ {k[:24]: v for k, v in par.items()} } finite {fin} ({dt:.0f} s)')
o, dt = sh([f'{CODE}/aug_dump.py', '--store', f'{H}/fleet_store', '--sid', sids[0], '--out', f'{OUT}/aug', '--n_time', '12'],
           f'{OUT}/aug.log')
print('(4) ' + ' | '.join(l for l in o.splitlines()))
ok &= 'MIRROR CHECK PASS' in o
ck = f'{BJ}/ckpt/f0_s42_ema_cnxt320x24_cv8183_best.pt'
if os.path.isfile(ck):
    o, dt = sh([f'{CODE}/tta_bench.py', '--store', f'{H}/fleet_store', '--folds', f'{H}/rsna_train/folds5.csv', '--fold', '0',
                '--exclude', f'{H}/rsna_train/gold74_eval_set.csv', '--eval_labels', f'{H}/rsna_train/llm_labels_v4_blend.csv',
                '--gold', f'{H}/rsna_train/gold74_eval_set.csv', '--n', str(N5), '--device', 'cpu', '--boot', '200',
                '--threads', '4', '--out', f'{OUT}/tb_real_n{N5}', '--oof', f'{H}/k2_test/oof_f0_s42_ema_cv8183.csv', '--', ck],
               f'{OUT}/tb_real.log')
    s = json.load(open(f'{OUT}/tb_real_n{N5}/tta_summary.json'))
    par = list(s['oof_parity'].values())[0]
    c5 = par['n'] == N5 and par['max_abs'] < 2e-2 and set(s['results'][os.path.basename(ck)]) == \
        {'none', 'mirror', 'offset', 'A+B', 'A+B lesion-max', 'all5'}
    ok &= c5
    print(f'{"PASS" if c5 else "FAIL"} (5) tta_bench, real interim family-1 member, {s["n_val"]} val + {s["n_gold58"]} gold58 '
          f'studies x 5 passes on CPU in {dt:.0f} s; pass A vs its GPU fp16 trainer OOF: {par}')
    for l in o.splitlines():
        if l.startswith(('G-TTA', 'A+B per-label')) or ' none ' in l[40:62] or ' A+B ' in l[40:62]:
            print('    ' + l[:200])
print(f'BOXGP0 {"PASS" if ok else "FAIL"}')
sys.exit(0 if ok else 1)
