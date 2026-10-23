#!/usr/bin/env python
# build_friend_kc.py (2026-10-15, Claude): a training notebook for a TEAMMATE's Kaggle account (2 x T4, internet ON):
# trains 2 extra KC1 all-data members (one per T4, in parallel) with the box code (bundle fleet_train.py shim -> --kc) on the
# RSNA-only KC store. Inputs (private datasets of eeshsaxena, shared with the teammate in the Kaggle UI):
#   eeshsaxena/rsna-knee-kc-store-v1   4 zips -> kc_store_part{0..3}/store/*.npz (+ kc_meta/ in part 0)
#   eeshsaxena/rsna-knee-kc-bundle-v1  code (fast_build/code2), pretrained ConvNeXt-S safetensors, timm HF cache, labels
# The argv is the box queue argv (rental kit members.tsv template) with only seed / out / workers changed, plus --stop_epoch E*
# (the schedule stays 13 epochs, so the E* checkpoint equals a full run's). A watchdog stops both trainers at WALL_H hours so
# the commit always ENDS and Kaggle keeps /kaggle/working. Members come back via this notebook's output -> box -> the same G1.
# usage: python build_friend_kc.py --owner <teammate kaggle username> --estar <E*> [--seeds 2005,2006] [--wall_h 9.0]
import os, json, argparse, textwrap
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--owner', required=True); ap.add_argument('--estar', type=int, required=True)
ap.add_argument('--seeds', default='2005,2006'); ap.add_argument('--wall_h', type=float, default=9.0)
ap.add_argument('--slug', default='rsna-train-kc-friend')
A = ap.parse_args()
s0, s1 = [int(x) for x in A.seeds.split(',')]
assert 5 <= A.estar <= 13
COMMON = ('--kc --model ttr --ttr_size 256 --backbone convnext_small.fb_in22k_ft_in1k --pretrained_file @R@/r7port/pretrained/'
          'convnext_small_in22k_ft_in1k.safetensors --mirror 0 --vbias_prior 1 --aug heavy --train_slices 8 --drop_path 0.3 '
          '--dropout 0.3 --ema 0.9995 --epochs 13 --save_epochs 5-13 --bs 1 --accum 8 --lr 1e-4 --head_lr 1e-3 --wd 0.05 '
          '--workers 2 --fold -1 --store @R@/kc_store/store --folds @R@/rsna_train/folds5.csv --exclude @R@/rsna_train/gold74_eval_set.csv '
          '--gold @R@/rsna_train/gold74_eval_set.csv --grad_ckpt 0')
MEMBERS = [
    dict(key=f's{s0}', gpu=0, out=f'fA_s{s0}_kc256_dread_friend', labels='@R@/kc/labels/labels_dread_fleet.csv'),
    dict(key=f's{s1}', gpu=1, out=f'fA_s{s1}_kc256_r7tb_friend', labels='@R@/r7t_labels/real/labels_r7t_b.csv'),
]
for m in MEMBERS:
    seed = int(m['key'][1:])
    m['argv'] = (f"{COMMON} --seed {seed} --labels {m['labels']} --eval_labels {m['labels']} --stop_epoch {A.estar} "
                 f"--out @R@/fleet_runs/{m['out']}")
CODE_MD5 = json.load(open(os.path.join(HERE, 'upload', 'rsna-knee-kc-bundle-v1', 'BUNDLE.json')))['code_md5']

MD = f"""# KC1 extra members (teammate GPU)

Trains two all-data members of our KC1 knee-centred model (ConvNeXt-S, KC256), one on each T4, for team rsna-knee.
Inputs are private team datasets (RSNA competition data only, no external data). Stops itself at {A.wall_h} h.
Run with: Accelerator GPU T4 x2, Internet ON, then Save Version -> Save & Run All (Commit). Do not edit."""

SETUP = textwrap.dedent(f'''
    # 1. Inputs: link the 4 store parts into one store dir, copy the code, check the code md5s and the RSNA-only rules.
    import os, glob, json, hashlib, re, csv, shutil, subprocess, sys, time
    T0 = time.time()
    R = '/kaggle/working/rsna'
    os.makedirs(R, exist_ok=True)
    def find_dir(name):
        c = [p for p in glob.glob('/kaggle/input/**/' + name, recursive=True) if os.path.isdir(p)]
        assert c, 'missing input dir ' + name
        return c[0]
    st = os.path.join(R, 'kc_store', 'store'); os.makedirs(st, exist_ok=True)
    n = 0
    for k in range(4):
        part = find_dir('kc_store_part%d' % k)
        for p in glob.glob(os.path.join(part, '**', '*.npz'), recursive=True):
            d = os.path.join(st, os.path.basename(p))
            if not os.path.exists(d):
                os.symlink(p, d); n += 1
    print('store npz linked', n, flush=True)
    bundle = os.path.dirname(find_dir('code2'))            # .../fast_build
    broot = os.path.dirname(bundle)
    shutil.copytree(os.path.join(broot, 'fast_build'), os.path.join(R, 'fast_build'), dirs_exist_ok=True)
    for sub in ('r7port', 'hf', 'kc', 'r7t_labels', 'rsna_train'):
        src = find_dir(sub) if not os.path.isdir(os.path.join(broot, sub)) else os.path.join(broot, sub)
        if not os.path.exists(os.path.join(R, sub)):
            os.symlink(src, os.path.join(R, sub))
    CODE_MD5 = {json.dumps(CODE_MD5)}
    bad = [f for f, h in CODE_MD5.items() if hashlib.md5(open(os.path.join(R, 'fast_build', 'code2', f), 'rb').read()).hexdigest() != h]
    assert not bad, 'code md5 mismatch: %s' % bad
    UID = re.compile(r'^1\\.2\\.826\\.0\\.1\\.3680043\\.8\\.498\\.\\d+$')
    sids = sorted(f[:-4] for f in os.listdir(st))
    def ids(p):
        with open(p, newline='') as fh: return [r['StudyInstanceUID'] for r in csv.DictReader(fh)]
    fo = set(ids(os.path.join(R, 'rsna_train/folds5.csv'))) | set(ids(os.path.join(R, 'rsna_train/gold74_eval_set.csv')))
    assert len(sids) == 4407 and all(UID.match(s) for s in sids) and set(sids) == fo, 'store is not the 4,407 RSNA studies'
    for c in ('r7t_labels/real/labels_r7t_b.csv', 'kc/labels/labels_dread_fleet.csv'):
        assert not (set(ids(os.path.join(R, c))) - fo), c
    print('RSNA_ONLY_OK store 4407 == folds5 + gold74; code md5s OK', flush=True)
''').strip()

PIP = textwrap.dedent('''
    # 2. Packages pinned to the box env (torch / torchvision of the image kept), exactly as the rental kit did.
    import subprocess, sys
    pins = [l.strip() for l in open('/kaggle/working/rsna/kit/pins.txt') if l.strip() and not l.lower().startswith(('torch==', 'torchvision=='))] \\
        if __import__('os').path.exists('/kaggle/working/rsna/kit/pins.txt') else None
    if pins is None:
        import glob; pins = [l.strip() for l in open(glob.glob('/kaggle/input/**/kit/pins.txt', recursive=True)[0]) if l.strip()]
    cv = [p for p in pins if p.lower().startswith('opencv')]; rest = [p for p in pins if p not in cv]
    import torch, torchvision
    open('/tmp/cons.txt', 'w').write('torch==%s\\ntorchvision==%s\\n' % (torch.__version__.split('+')[0], torchvision.__version__.split('+')[0]))
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-c', '/tmp/cons.txt'] + rest, check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps'] + cv, check=True)
    print(subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=freeze'], capture_output=True, text=True).stdout[-1500:])
    print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), flush=True)
''').strip()

RUN = textwrap.dedent(f'''
    # 3. Two members in parallel (one per T4) + a watchdog that stops both at {A.wall_h} h so the commit always ends.
    import os, subprocess, time, json, signal, glob, hashlib
    R = '/kaggle/working/rsna'; FR = os.path.join(R, 'fleet_runs'); os.makedirs(FR, exist_ok=True)
    MEMBERS = {json.dumps(MEMBERS)}
    WALL = T0 + {A.wall_h} * 3600
    procs = {{}}
    for m in MEMBERS:
        argv = m['argv'].replace('@R@', R).split()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(m['gpu']), HF_HOME=os.path.join(R, 'hf'), HF_HUB_OFFLINE='1',
                   PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
        log = open(os.path.join(FR, m['out'] + '.log'), 'w')
        procs[m['key']] = (subprocess.Popen([sys.executable, 'fleet_train.py'] + argv, cwd=os.path.join(R, 'fast_build', 'code2'),
                                            env=env, stdout=log, stderr=subprocess.STDOUT), log, m)
        print('LAUNCH', m['key'], 'GPU', m['gpu'], ' '.join(argv), flush=True)
    stopped = []
    while any(p.poll() is None for p, _, _ in procs.values()):
        if time.time() > WALL:
            for k, (p, _, _) in procs.items():
                if p.poll() is None:
                    p.send_signal(signal.SIGTERM); stopped.append(k)
            time.sleep(120)
            for p, _, _ in procs.values():
                if p.poll() is None: p.kill()
            break
        time.sleep(60)
    rec = {{}}
    for k, (p, log, m) in procs.items():
        log.close()
        d = os.path.join(FR, m['out'])
        files = sorted(glob.glob(os.path.join(d, '**', '*'), recursive=True))
        last = open(os.path.join(FR, m['out'] + '.log'), errors='replace').read().rstrip().splitlines()[-1:]
        rec[k] = dict(rc=p.returncode, stopped_by_watchdog=k in stopped, closing=last[0][:200] if last else '',
                      files={{os.path.relpath(f, FR): hashlib.sha256(open(f, 'rb').read()).hexdigest() for f in files if os.path.isfile(f)}})
        print(k, 'rc', p.returncode, 'watchdog', k in stopped, 'files', len(rec[k]['files']), flush=True)
    rec['hours'] = round((time.time() - T0) / 3600, 2)
    json.dump(rec, open(os.path.join(FR, 'friend_receipt.json'), 'w'), indent=1)
    # keep only the outputs (drop the code copy and the store links from /kaggle/working)
    import shutil
    shutil.rmtree(os.path.join(R, 'kc_store'), ignore_errors=True); shutil.rmtree(os.path.join(R, 'fast_build'), ignore_errors=True)
    for sub in ('r7port', 'hf', 'kc', 'r7t_labels', 'rsna_train'):
        p = os.path.join(R, sub)
        if os.path.islink(p): os.unlink(p)
    print('FRIEND_KC_DONE', json.dumps({{k: (v['rc'], v['stopped_by_watchdog']) for k, v in rec.items() if k != 'hours'}}), rec['hours'], 'h', flush=True)
''').strip()

def cell(kind, src):
    c = {'cell_type': kind, 'metadata': {}, 'source': src}
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    return c
nb = {'cells': [cell('markdown', MD), cell('code', SETUP), cell('code', PIP), cell('code', RUN)],
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 4}
out = os.path.join(HERE, 'kernel_friend')
os.makedirs(out, exist_ok=True)
code_file = A.slug.replace('-', '_') + '.ipynb'
json.dump(nb, open(os.path.join(out, code_file), 'w', encoding='utf-8'), indent=1)
meta = dict(id=f'{A.owner}/{A.slug}', title=A.slug.replace('-', ' '), code_file=code_file, language='python', kernel_type='notebook',
            is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=True,
            dataset_sources=['eeshsaxena/rsna-knee-kc-store-v1', 'eeshsaxena/rsna-knee-kc-bundle-v1'], kernel_sources=[],
            competition_sources=[], model_sources=[], machine_shape='NvidiaTeslaT4')
json.dump(meta, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=2)
print('BUILT', out, code_file, '| members', [(m['key'], m['gpu'], m['out']) for m in MEMBERS], '| E*', A.estar, '| wall', A.wall_h)
