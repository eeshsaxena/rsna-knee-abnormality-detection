# kt_dry.py A|B (kaggle_frac75c): box CPU dry test of notebook A (rsna_train_ttr_frac75_s1044_f12.ipynb, folds 1+2 s1044)
# or B (rsna_train_ttr_frac75_s1044_f34.ipynb, folds 3+4 s1044). Never touches the GPU, ~/bigjump_build, ~/fleet_runs,
# ~/novel or ~/rsna_train except READS. Layout under ~/kt_frac75c/dry_<A|B>/<phase>/{kaggle/input,kaggle/working,tmp};
# the notebook code cells run UNCHANGED as one script with KT_ROOT=<phase dir> and a dummy trainer.
# Same test as kaggle_frac75b (80/80 there) with the new NB_PLAN; the wall-path checks use the stricter 11.5 h
# (41400 s, the old push timeout) although the push now uses --timeout 43200 (12 h).
#   unit   lib cell only: s/it parser on the real J1 log + a fake log, header check, progress/closing, E arithmetic with
#          the 10.3 h budget at the measured Kaggle s/it, process-group stop, wait_quiet, tar guard, frac75 byte helpers
#   A      full flow, 2-tar subset (prep tars in both mount layouts, labels dataset in the new layout), real model and
#          pretrained check, probe, E, train (frac75 labels in the argv), receipt
#   B      watchdog: wall ~60 s after launch, dummies hang, the GPU1 trainer ignores SIGTERM -> SIGKILL, E < 16 warning
#   W      the DEFAULT budget/wall (10.3 h / 11.4 h, no override) against a T0 set 11.4 h - 5 min in the past: E floors
#          at 1, the wall cuts training ~5 min after start, the receipt's session end stays < 11.5 h (< --timeout 43200)
#   C      corrupted tar -> abort at the store step     D  wrong tensor_sha printed by the trainer -> probe abort
#   E      wrong val count printed by the trainer -> probe abort
#   F      CRLF folds5.csv + missing wheel + frac75 csv rewritten to LF + labels dataset in the LEGACY layout -> warnings, DONE
#   H      code dataset AND labels dataset each at 2 mount paths (identical bytes) -> warnings, DONE
#   L1     labels dataset not attached -> abort at step 1   L2  one byte of the frac75 csv changed -> sha256 abort
#   L3     two DIFFERENT frac75 csvs attached -> abort      L4  the mix50 csv attached under the frac75 name -> abort
import os, sys, json, time, shutil, hashlib, tarfile, subprocess, signal, glob, re
import pandas as pd

H = os.path.expanduser('~')
KT = f'{H}/kt_frac75c'
KEY = sys.argv[1]
NBK = {'A': ('rsna-train-ttr-frac75-s1044-f12', [(1, 1044), (2, 1044)]),
       'B': ('rsna-train-ttr-frac75-s1044-f34', [(3, 1044), (4, 1044)])}[KEY]
DRY = f'{KT}/dry_{KEY}'
DSRC = f'{KT}/ds/rsna-knee-fleet-train'
DLSRC = f'{KT}/ds_labels'
FRAC = 'labels_frac75_logit.csv'
NB = f'{KT}/kernel/{NBK[0]}/{NBK[0].replace("-", "_")}.ipynb'
PY = f'{H}/miniconda3/envs/rsna/bin/python'
RESULTS = {}
FOLDS = tuple(F for F, _ in NBK[1])
SEEDS = dict(NBK[1])
RUNS = {F: f'f{F}_s{SEEDS[F]}_ttr_frac75' for F in FOLDS}
F0, F1 = FOLDS                                          # GPU0 trainer, GPU1 trainer
TIMEOUT_S = 41400                                       # 11.5 h: stricter than the push --timeout 43200


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def cells():
    nb = json.load(open(NB))
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']


def check(name, cond, info=''):
    RESULTS[name] = bool(cond)
    print(('PASS ' if cond else 'FAIL ') + name + (f'  [{info}]' if info else ''), flush=True)


# ---------------------------------------------------------------- fixture: 10 box store files in 2 tars
lab = pd.read_csv(f'{DLSRC}/{FRAC}', dtype={'StudyInstanceUID': str})
folds = pd.read_csv(f'{H}/rsna_train/folds5.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['fold']
box = json.load(open(f'{DSRC}/box_facts.json'))
probe4 = sorted(box['ckpt_args']['decode_probe'])
ids = sorted(lab.StudyInstanceUID)
pick = list(probe4)
for F in (0, 1, 2, 3, 4):
    pick += [s for s in ids if folds.get(s) == F and s not in pick][:2]
assert len(pick) == 14 and all(os.path.exists(f'{H}/fleet_store/{s}.npz') for s in pick)
FIX = f'{DRY}/fixture'
shutil.rmtree(DRY, ignore_errors=True)
os.makedirs(FIX)
for i, grp in enumerate((pick[:7], pick[7:])):
    with tarfile.open(f'{FIX}/fleet_store_s{i}_p00.tar', 'w') as t:
        for s in grp:
            t.add(f'{H}/fleet_store/{s}.npz', arcname=f'{s}.npz')
parts = {f'fleet_store_s{i}_p00.tar': sha(f'{FIX}/fleet_store_s{i}_p00.tar') for i in (0, 1)}


def counts(F, names):
    l = lab.set_index('StudyInstanceUID')
    ex = set(pd.read_csv(f'{H}/rsna_train/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID)
    hd = set(folds.index[folds < 0]) | ex
    hv = {s for s in l.index if s in names}
    ii = [s for s in l.index if s in folds.index and s in hv]
    tr = [s for s in ii if s not in hd and folds[s] != F]
    va = [s for s in ii if s not in hd and folds[s] == F]
    return dict(labels=len(l), store_has=len(hv), held=len(hd), train=len(tr), val=len(va), iters_per_epoch=len(tr) // 2)


def make_root(phase, corrupt=False, crlf=False, drop_wheel=False, dup_ds=False, lab_layout='new', lab_mode='ok'):
    """lab_layout: 'new' (/kaggle/input/datasets/eeshsaxena/rsna-knee-frac75-labels), 'legacy'
    (/kaggle/input/rsna-knee-frac75-labels), 'both', 'none'. lab_mode: 'ok' | 'lf' (rewritten to LF) | 'flip' (one
    Fracture digit changed) | 'two' (a 2nd, different copy at the legacy path) | 'mix50' (mix50 bytes under the name)."""
    R = f'{DRY}/{phase}'
    ds = f'{R}/kaggle/input/datasets/eeshsaxena/rsna-knee-fleet-train'
    shutil.copytree(DSRC, ds)
    names = sorted(f'{s}.npz' for s in pick)
    bf = json.load(open(f'{ds}/box_facts.json'))
    bf['store_n'] = len(names)
    bf['store_names_sha'] = hashlib.sha256('\n'.join(names).encode()).hexdigest()
    bf['store_sample'] = [dict(name=n, part='x', size=os.path.getsize(f'{H}/fleet_store/{n}'),
                               sha256=sha(f'{H}/fleet_store/{n}')) for n in names]
    bf['expected_counts'] = {str(F): counts(F, set(pick)) for F in range(5)}
    bf['labelled_missing'] = len(lab) - len(pick)
    json.dump(bf, open(f'{ds}/box_facts.json', 'w'))
    ke = json.load(open(f'{ds}/kt_expect.json'))
    ke['parts'] = parts
    json.dump(ke, open(f'{ds}/kt_expect.json', 'w'))
    json.dump({'files': {f: sha(f'{ds}/{f}') for f in sorted(os.listdir(ds))
                         if f not in ('kt_manifest.json', 'dataset-metadata.json')}}, open(f'{ds}/kt_manifest.json', 'w'))
    if dup_ds:                                 # the same dataset visible at a 2nd (legacy) mount path, not a symlink
        shutil.copytree(ds, f'{R}/kaggle/input/rsna-knee-fleet-train')
    for i, sub in ((0, 'notebooks/eeshsaxena/rsna-fleet-prep-s0'), (1, 'rsna-fleet-prep-s1')):   # both layouts
        d = f'{R}/kaggle/input/{sub}'
        os.makedirs(d)
        tn = f'fleet_store_s{i}_p00.tar'
        shutil.copyfile(f'{FIX}/{tn}', f'{d}/{tn}')
        json.dump({'status': 'OK', 'shard': i, 'parts': {tn: parts[tn]}}, open(f'{d}/prep_manifest_{i}.json', 'w'))
        if corrupt and i == 1:
            with open(f'{d}/{tn}', 'r+b') as f:
                f.seek(5000)
                b = f.read(1)
                f.seek(5000)
                f.write(bytes([b[0] ^ 0xFF]))
    fb = open(f'{DLSRC}/{FRAC}', 'rb').read()
    if lab_mode == 'lf':
        fb = fb.replace(bytes([13, 10]), bytes([10]))
    elif lab_mode == 'flip':                   # one character of one Fracture value: '...,0.1234\r\n' -> '...,0.1235'
        i = fb.index(bytes([13, 10]), 2000) - 1
        fb = fb[:i] + (b'1' if fb[i:i + 1] != b'1' else b'2') + fb[i + 1:]
    elif lab_mode == 'mix50':
        fb = open(f'{DSRC}/labels_mix50_logit.csv', 'rb').read()
    dirs = {'new': ['datasets/eeshsaxena/rsna-knee-frac75-labels'], 'legacy': ['rsna-knee-frac75-labels'],
            'both': ['datasets/eeshsaxena/rsna-knee-frac75-labels', 'rsna-knee-frac75-labels'], 'none': []}[lab_layout]
    for sub in dirs:
        os.makedirs(f'{R}/kaggle/input/{sub}')
        open(f'{R}/kaggle/input/{sub}/{FRAC}', 'wb').write(fb)
    if lab_mode == 'two':
        os.makedirs(f'{R}/kaggle/input/rsna-knee-frac75-labels-v2')
        i = fb.index(bytes([13, 10]), 3000) - 1
        open(f'{R}/kaggle/input/rsna-knee-frac75-labels-v2/{FRAC}', 'wb').write(
            fb[:i] + (b'1' if fb[i:i + 1] != b'1' else b'2') + fb[i + 1:])
    if crlf:                                   # Kaggle-side CRLF rewrite of a csv (manifest keeps the box sha)
        b = open(f'{ds}/folds5.csv', 'rb').read()
        open(f'{ds}/folds5.csv', 'wb').write(b.replace(bytes([10]), bytes([13, 10])))
    if drop_wheel:
        os.remove(f'{ds}/timm-1.0.29-py3-none-any.whl.bin')
    os.makedirs(f'{R}/kaggle/working')
    os.makedirs(f'{R}/tmp')
    with open(f'{R}/nb_script.py', 'w') as f:
        for i, c in enumerate(cells()):
            f.write(f'\n# ---------------- notebook code cell {i + 1} ----------------\n{c}\n')
    return R


def run_nb(R, env_extra, timeout=900):
    env = dict(os.environ, KT_ROOT=R, KT_SKIP_GPU_CHECK='1', KT_GPU_IDS='-1,-1', CUDA_VISIBLE_DEVICES='',
               KT_TRAINER_PY=f'{KT}/dry_src/dummy_train.py', KT_PROBE_ITERS='2', KT_PROBE_POLL_S='0.3',
               KT_POLL_S='0.5', KT_STATUS_EVERY_S='5', KT_STARTUP_S='0', DUMMY_PRINT_EVERY='1', DUMMY_SPEED='0.3',
               PYTHONUNBUFFERED='1')
    env.update(env_extra)
    t = time.time()
    with open(f'{R}/nb_run.log', 'w') as lf:
        rc = subprocess.run([PY, f'{R}/nb_script.py'], cwd=f'{R}/kaggle/working', stdout=lf, stderr=subprocess.STDOUT,
                            env=env, timeout=timeout).returncode
    rec = json.load(open(f'{R}/kaggle/working/run_receipt.json'))
    out = open(f'{R}/nb_run.log').read()
    return rc, rec, out, time.time() - t


def no_strays():
    r = subprocess.run(['pgrep', '-u', str(os.getuid()), '-f', f'(dummy_train.py|time.sleep\\(100000\\)).*{DRY}/'],
                       capture_output=True, text=True)
    return r.stdout.strip() == '', r.stdout.strip()


# ---------------------------------------------------------------- unit tests (lib cell only)
U = f'{DRY}/unit'
os.makedirs(f'{U}/kaggle/working')
os.environ['KT_ROOT'] = U
ns = {}
exec(compile(cells()[0], 'lib', 'exec'), ns)
check(f'unit CFG: kernel {NBK[0]}, trainers {NBK[1]}, budget 10.3 h, wall 11.4 h, last-3 saves, frac75 labels',
      ns['REC']['kernel'] == NBK[0] and ns['FOLDS'] == list(FOLDS) and ns['CFG']['SEED'] == SEEDS and
      ns['CFG']['BUDGET_H'] == 10.3 and ns['CFG']['WALL_H'] == 11.4 and ns['CFG']['GPU_IDS'] == ['0', '1'] and
      ns['CFG']['SAVE_LAST'] == 3 and ns['CFG']['LABELS_TRAIN'] == FRAC and ns['RUN'] == RUNS, (ns['RUN'], ns['CFG']['SEED']))
check('unit baked sha256/md5 == the box ~/novel/frac75 csv',
      ns['CFG']['LABELS_TRAIN_SHA256'] == sha(f'{H}/novel/frac75/{FRAC}') ==
      hashlib.sha256(open(f'{DLSRC}/{FRAC}', 'rb').read()).hexdigest() and
      ns['CFG']['LABELS_TRAIN_MD5'] == hashlib.md5(open(f'{H}/novel/frac75/{FRAC}', 'rb').read()).hexdigest())
j1 = open(f'{H}/fleet_runs/f0_s1043_ttr_mix50.log').read()
head = '\n'.join(j1.splitlines()[:6]) + '\n'           # labels, contract, BJ, it 0, it 200, it 400
its = ns['parse_its'](head)
w = ns['sit_window'](its)
check('unit parse_its J1 head (it 0/200/400)', [i[0] for i in its] == [0, 200, 400] and its[0][1] == 1733, its)
check('unit sit_window J1 200->400 = (0.50*401-0.59*201)/200', abs(w - 0.40955) < 1e-9, f'{w:.5f}')
itsf = ns['parse_its'](j1)
check('unit sit_window J1 full ep0 (1400->1600)', abs(ns['sit_window'](itsf) - (0.39 * 1601 - 0.40 * 1401) / 200) < 1e-9,
      f'{ns["sit_window"](itsf):.4f}')
fake = ('noise line\nep 0 it 0/1733 loss 0.68 9.00s/it\nep 0 it 200/1733 loss 0.5 1.20s/it\nwarning: x\n'
        'ep 0 it 400/1733 loss 0.5 1.10s/it\nep 1 it 0/1733 loss 0.5 2.00s/it\n')
wf = ns['sit_window'](ns['parse_its'](fake))
check('unit sit_window fake log (ep1 lines ignored)', abs(wf - (1.10 * 401 - 1.20 * 201) / 200) < 1e-9, f'{wf:.4f}')
check('unit sit_window single line -> None', ns['sit_window'](ns['parse_its']('ep 0 it 0/9 loss 1 3.00s/it')) is None)
kr = json.load(open(f'{H}/fleet_runs/kaggle_s1044_f01_run_receipt.json'))
pl0 = kr['probe']['progress_lines']['0']
kfake = ''.join(f'ep 0 it {a}/{b} loss 0.5 {c:.2f}s/it\n' for a, b, c in pl0)
check('unit sit_window on the s1044 f01 Kaggle probe lines == its receipt s/it', abs(ns['sit_window'](ns['parse_its'](kfake))
      - kr['probe']['s_per_it']['0']) < 1e-9, f'{ns["sit_window"](ns["parse_its"](kfake)):.4f}')
h = ns['parse_header'](head)
c0 = box['expected_counts']['0']
check('unit header J1 == box (fold 0 counts)', ns['header_problems'](h, c0, box) == [], ns['header_problems'](h, c0, box))
c3 = box['expected_counts']['3']
pb = ns['header_problems'](h, c3, box)
check('unit header J1 vs fold-3 counts -> train/val problems', any('train' in p for p in pb) and any('val' in p for p in pb), pb)
hb = ns['parse_header'](head.replace('08ec55bc155be309', '08ec55bc155be30X'))
check('unit header wrong tensor_sha detected', any('tensor_sha' in p for p in ns['header_problems'](hb, c0, box)))
check('unit header missing lines detected', len(ns['header_problems'](ns['parse_header'](''), c0, box)) == 3)
pr = ns['progress'](j1)
check('unit progress J1: 24 epochs, best raw .8836 / ema .8824',
      pr['last_ep_done'] == 23 and abs(pr['best']['raw'][0] - 0.8836) < 1e-9 and abs(pr['best']['ema'][0] - 0.8824) < 1e-9,
      pr['best'])
check('unit closing_line J1', ns['closing_line'](j1) == {'raw': 0.8835985889396399, 'ema': 0.8823996581435246})
check('unit closing_line J1 cut before its last line -> None', ns['closing_line']('\n'.join(j1.splitlines()[:-1])) is None)
ce = ns['choose_epochs']
L0 = 0.141 * 3600                                       # launch elapsed in the s1044 f01/f23 runs (0.139-0.144 h)
tab = {}
for s in (0.7796, 0.7996, 0.8096, 0.85, 1.1, 0.5):
    tab[s] = ce(s, 1733, L0)                            # the notebook defaults: budget 10.3 h, eval .21, startup 120 s
print('E table (budget 10.3 h, launch +0.141 h):', {s: (r['E'], round(r['epoch_s'] / 60, 2), round(r['finish_elapsed_s'] / 3600, 2))
                                                 for s, r in tab.items()}, flush=True)
check('unit E at the measured Kaggle s/it: .78 -> 22, .80 -> 21, .81 -> 21 (= the f34 / s1044 receipts: 22, 21, 21)',
      (tab[0.7796]['E'], tab[0.7996]['E'], tab[0.8096]['E']) == (22, 21, 21))
check('unit E: .85 s/it -> 20, 1.1 s/it -> 15 (< 16 warn), 0.5 s/it -> 24 (cap)',
      (tab[0.85]['E'], tab[1.1]['E'], tab[0.5]['E']) == (20, 15, 24))
check('unit E: .7996 s/it plans finish at 9.96 h (= the s1044 f01 receipt est_finish_h)',
      round(tab[0.7996]['finish_elapsed_s'] / 3600, 2) == 9.96, tab[0.7996]['finish_elapsed_s'] / 3600)
check('unit E: planned finish <= 10.3 h for every s/it', all(r['finish_elapsed_s'] <= 10.3 * 3600 for r in tab.values()))
slow = max(L0 + 120 + r['E'] * r['epoch_s'] * 1.10 for r in tab.values()) / 3600
check('unit E: even +10% slower epochs than planned end before the 11.4 h wall', slow < 11.4, f'{slow:.2f} h')
wall_end = ns['CFG']['WALL_H'] * 3600 + 120 + ns['CFG']['KILL_GRACE_S'] + 120     # quiet-wait max + grace + finalize
check('unit wall path: 11.4 h + quiet-wait 2 min + kill grace 1 min + finalize 2 min < 41400 s (11.5 h; push --timeout 43200)',
      wall_end < TIMEOUT_S, f'{wall_end:.0f} s, margin {TIMEOUT_S - wall_end:.0f} s')
check('unit E: no time left -> 1 (floor)', ce(5.0, 1733, 8.3 * 3600)['E'] == 1)
check('unit save_epochs_arg: last 3', ns['save_epochs_arg'](17) == '15-17' and ns['save_epochs_arg'](24) == '22-24' and
      ns['save_epochs_arg'](2) == '1-2' and ns['save_epochs_arg'](1) == '1-1')
for F in FOLDS:
    rf = ns['recipe_args'](F, 21, '19-21', '/o', '/L', 3)
    check(f'unit recipe_args fold {F}: --seed {SEEDS[F]} (per trainer), --fold {F}',
          rf[rf.index('--seed') + 1] == str(SEEDS[F]) and rf[rf.index('--fold') + 1] == str(F), rf)
ra = ns['recipe_args'](F0, 21, '19-21', '/o', '/L', 3)
check('unit recipe_args == J1 argv except --labels (+ E/seed/save/workers/paths)',
      ra[:16] == ['--model', 'ttr', '--aug', 'heavy', '--train_slices', '16', '--drop_path', '0.2', '--dropout', '0.3',
                  '--ema', '0.9995', '--epochs', '21', '--seed', str(SEEDS[F0])] and ra[ra.index('--labels') + 1] == f'/L/{FRAC}' and
      ra[ra.index('--eval_labels') + 1] == '/L/llm_labels_v4_blend.csv' and ra[ra.index('--exclude') + 1] ==
      '/L/gold74_eval_set.csv' and ra[ra.index('--gold') + 1] == '/L/gold74_eval_set.csv' and
      ra[ra.index('--folds') + 1] == '/L/folds5.csv' and ra[-4:] == ['--fold', str(F0), '--out', '/o'] and '--grad_ckpt' not in ra, ra)
fb = open(f'{DLSRC}/{FRAC}', 'rb').read()
mb = open(f'{DSRC}/labels_mix50_logit.csv', 'rb').read()
want = ns['CFG']['LABELS_TRAIN_SHA256']
lfb = fb.replace(bytes([13, 10]), bytes([10]))
check('unit labels_variant: exact / LF-rewritten / CRLF-doubled -> the box bytes; changed -> None',
      ns['labels_variant'](fb, want) == (fb, 'exact') and ns['labels_variant'](lfb, want) == (fb, 'LF->CRLF') and
      ns['labels_variant'](fb.replace(bytes([10]), bytes([13, 10])), want)[0] == fb and
      ns['labels_variant'](fb[:-3] + b'9' + fb[-2:], want) is None and ns['labels_variant'](mb, want) is None)
i_acl = mb.index(b',', mb.index(bytes([10])) + 1) + 1          # first data row, first label field (ACL)
check('unit only_fracture_differs: frac75 vs mix50 True; an ACL change False; dropped row False',
      ns['only_fracture_differs'](fb, mb) and not ns['only_fracture_differs'](fb[:i_acl] + b'9' + fb[i_acl + 1:], mb) and
      not ns['only_fracture_differs'](fb[:fb.rindex(b'\n', 0, len(fb) - 2) + 1], mb))


def spawn(trap):
    lf = open(f'{U}/sg.log', 'w')
    p = subprocess.Popen(['bash', '-c', (f'trap "" TERM; ' if trap else '') + 'sleep 1000 & sleep 1000 & wait'],
                         stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    p.kt_log = lf
    time.sleep(0.5)
    return p


p = spawn(False)
how = ns['stop_group'](p, 5)
time.sleep(0.3)
left = subprocess.run(['pgrep', '-g', str(p.pid)], capture_output=True, text=True).stdout.strip()
check('unit stop_group SIGTERM path, group empty', how == 'sigterm' and left == '', how)
p = spawn(True)
t = time.time()
how = ns['stop_group'](p, 2)
time.sleep(0.3)
left = subprocess.run(['pgrep', '-g', str(p.pid)], capture_output=True, text=True).stdout.strip()
check('unit stop_group SIGKILL escalation after grace, group empty', how == 'sigkill' and left == '' and time.time() - t >= 2, how)
qd = f'{U}/quiet'                                   # wait_quiet (hard-wall guard against a kill inside torch.save)
os.makedirs(qd)
open(f'{qd}/x.pt', 'w').write('x')
time.sleep(2.3)
wq = ns['wait_quiet']([qd], quiet_s=2, max_wait_s=30, poll_s=0.2)
check('unit wait_quiet: quiet dir -> returns at once', wq < 0.5, f'{wq:.2f} s')
wr = subprocess.Popen(['bash', '-c', f'for i in 1 2 3 4 5; do echo $i >> {qd}/x.pt; sleep 1; done'])
time.sleep(0.3)
wq = ns['wait_quiet']([qd], quiet_s=2, max_wait_s=30, poll_s=0.2)
wr.wait()
check('unit wait_quiet: waits until a writer stopped + quiet_s', 4.5 <= wq <= 8, f'{wq:.2f} s')
time.sleep(2.3)
os.remove(f'{qd}/x.pt')
wq = ns['wait_quiet']([qd], quiet_s=2, max_wait_s=30, poll_s=0.2)
check('unit wait_quiet: a delete (dir mtime) counts as activity', 1.5 <= wq <= 3.5, f'{wq:.2f} s')
os.makedirs(f'{U}/evil')
with tarfile.open(f'{U}/evil.tar', 'w') as tf:
    tf.add(f'{H}/fleet_store/{pick[0]}.npz', arcname=f'../{pick[0]}.npz')
try:
    ns['extract_tar']('evil.tar', f'{U}/evil.tar', f'{U}/evil')
    check('unit extract_tar rejects ../ member', False)
except ValueError as e:
    check('unit extract_tar rejects ../ member', True, str(e)[:80])
n, hx, nm, nb = ns['extract_tar']('s0', f'{FIX}/fleet_store_s0_p00.tar', f'{U}/evil')
check('unit extract_tar hash == file sha256, 7 members', hx == parts['fleet_store_s0_p00.tar'] and nm == 7 and
      nb == os.path.getsize(f'{FIX}/fleet_store_s0_p00.tar'))
del os.environ['KT_ROOT']

# ---------------------------------------------------------------- A: full flow
R = make_root('A')
rc, rec, out, dt = run_nb(R, {})
print(out[-6000:])
check('A notebook exit 0', rc == 0, f'{dt:.0f} s')
check('A status DONE', rec['status'] == 'DONE', rec['status'])
check('A found tars in both layouts', len(rec['inputs']['tars']) == 2 and
      any('/notebooks/' in p for p in rec['inputs']['tars'].values()) and
      any('/kaggle/input/rsna-fleet-prep-s1/' in p for p in rec['inputs']['tars'].values()), rec['inputs']['tars'])
check('A frac75 labels found (new layout), sha256 == baked, exact bytes, LBL copy == box ~/novel/frac75 file',
      '/datasets/eeshsaxena/rsna-knee-frac75-labels/' in rec['inputs']['train_labels_src'] and
      rec['inputs']['train_labels_line_endings'] == 'exact' and
      sha(f'{R}/tmp/kt_labels/{FRAC}') == sha(f'{H}/novel/frac75/{FRAC}') == rec['inputs']['train_labels_sha256'])
check('A store == box files (14 npz, sample sha, names sha)', rec['store']['n_npz'] == 14 and rec['store']['sample_ok'] == 14)
check('A parity: decode probe 4/4 + model keys + pretrained bytes', rec['parity']['decode_probe_match'] == 4 and
      rec['parity']['model_check']['keys_match'] and rec['parity']['model_check']['pretrained_equal'],
      rec['parity']['model_check'])
check('A pretrained from the bundled bytes, HF offline', rec['parity']['hf_offline'] == '1')
check('A fold counts from the frac75 file == box arithmetic', all(rec['parity']['counts'][str(F)] ==
      json.load(open(f'{R}/kaggle/input/datasets/eeshsaxena/rsna-knee-fleet-train/box_facts.json'))['expected_counts'][str(F)]
      for F in FOLDS), rec['parity']['counts'])
check('A probe s/it ~ dummy 0.3', all(abs(v - 0.3) < 0.08 for v in rec['probe']['s_per_it'].values()), rec['probe']['s_per_it'])
check('A E = 24 (budget not binding), save_epochs last 3, receipt records E + save_epochs + labels',
      rec['plan']['E'] == 24 and rec['plan']['save_epochs'] == '22-24' and rec['plan']['train_labels'] == FRAC and
      rec['plan']['budget_h'] == 10.3 and rec['plan']['wall_h'] == 11.4 and
      {int(k): v for k, v in rec['plan']['seeds'].items()} == SEEDS, rec['plan'])
for F in FOLDS:
    x = rec['train']['folds'][str(F)]
    pts = [f for f in x['files'] if f['path'].endswith('.pt')]
    eo = [f for f in x['files'] if '/epochs/oof_' in f['path']]
    check(f'A fold {F}: complete, 2 best + 6 epoch ckpts load with box keys, 6 epoch OOFs',
          x['complete'] and len(x['best_ckpts']) == 2 and len(pts) == 8 and len(eo) == 6 and
          all(f['load_ok'] and f['keys_match'] for f in pts), [f['path'] for f in pts])
    check(f'A fold {F}: every ckpt has args seed {SEEDS[F]} / fold {F}, named {RUNS[F]}/f{F}_s{SEEDS[F]}_...',
          all(f['args_seed'] == SEEDS[F] and f['args_fold'] == F and
              f['path'].startswith(f'{RUNS[F]}/') and f'f{F}_s{SEEDS[F]}_' in f['path'] for f in pts), pts[:1])
    check(f'A fold {F}: outputs in /kaggle/working/{RUNS[F]} + .log',
          os.path.isdir(f'{R}/kaggle/working/{RUNS[F]}') and os.path.exists(f'{R}/kaggle/working/{RUNS[F]}.log'))
    cmd = rec['train']['cmd'][str(F)]
    check(f'A fold {F} cmd = J1 recipe with --labels frac75 and --seed {SEEDS[F]} (+E, save_epochs, workers, paths)',
          '--model ttr --aug heavy --train_slices 16 --drop_path 0.2 --dropout 0.3 --ema 0.9995 --epochs 24 --seed '
          f'{SEEDS[F]} --save_epochs 22-24 --workers 3 --store' in cmd and f'--labels {R}/tmp/kt_labels/{FRAC} --folds' in cmd and
          '--eval_labels' in cmd and 'labels_mix50' not in cmd and
          cmd.endswith(f'--fold {F} --out {R}/kaggle/working/{RUNS[F]}'), cmd)
check('A probe logs copied', all(os.path.exists(f'{R}/kaggle/working/probe/probe_f{F}.log') for F in FOLDS))
ok, s = no_strays()
check('A no stray processes', ok, s)
launch_h = rec['train']['launch_h']
el_launch_s = launch_h * 3600

# ---------------------------------------------------------------- B: watchdog, 60 s wall
R = make_root('B')
rc, rec, out, dt = run_nb(R, dict(KT_WALL_H=str((el_launch_s + 60) / 3600), KT_BUDGET_H=str((el_launch_s + 10) / 3600),
                                  KT_KILL_GRACE_S='5', DUMMY_HANG_AFTER_EP='0', DUMMY_IGNORE_TERM_FOLDS=str(F1)))
print(out[-5000:])
x0, x1 = rec['train']['folds'][str(F0)], rec['train']['folds'][str(F1)]
check('B notebook exit 0, status PARTIAL', rc == 0 and rec['status'].startswith('PARTIAL'), rec['status'])
check('B E < 16 warning logged', rec['plan']['E'] < 16 and any('< 16' in w for w in rec['warnings']), rec['plan']['E'])
check('B wall warning logged', any('hard wall' in w for w in rec['warnings']))
check(f'B fold {F0} (GPU0) SIGTERM, fold {F1} (GPU1) SIGKILL', x0['ended'] == 'sigterm' and x1['ended'] == 'sigkill', (x0['ended'], x1['ended']))
check('B killed ~60 s after launch', abs(rec['train']['hours'] * 3600 - 60) < 25, f'{rec["train"]["hours"] * 3600:.0f} s')
check('B wall: quiet-wait ran before SIGTERM (outputs idle -> ~0 s)', rec['train'].get('wall_quiet_wait_s', 99) < 5,
      rec['train'].get('wall_quiet_wait_s'))
check('B best-so-far kept (2 best ckpts per fold, load OK)', all(len(x['best_ckpts']) == 2 and all(
    f.get('load_ok', True) for f in x['files']) for x in (x0, x1)))
ok, s = no_strays()
check('B no stray processes (workers killed with the group)', ok, s)

# ---------------------------------------------------------------- W: the DEFAULT 10.3 h budget / 11.4 h wall vs a T0 in the past
R = make_root('W')
T0W = time.time() - 11.4 * 3600 + 300         # the default wall (11.4 h after T0) falls ~5 min from now
rc, rec, out, dt = run_nb(R, dict(KT_T0=str(T0W), KT_KILL_GRACE_S='5', DUMMY_HANG_AFTER_EP='0'))
print(out[-3000:])
check('W defaults in force (budget 10.3, wall 11.4), E floored to 1 + warning', rec['cfg']['BUDGET_H'] == 10.3 and
      rec['cfg']['WALL_H'] == 11.4 and rec['plan']['E'] == 1 and any('< 16' in w for w in rec['warnings']), rec['plan'])
check('W the default wall cut training (hard wall warning, PARTIAL, best-so-far kept)', rec['status'].startswith('PARTIAL') and
      any('hard wall 11.4 h' in w for w in rec['warnings']) and
      all(len(rec['train']['folds'][str(F)]['best_ckpts']) == 2 for F in FOLDS), rec['status'])
check('W session end (receipt elapsed) within 11.4 h and 11.5 h (push --timeout 43200 = 12 h)', 11.4 <= rec['elapsed_h'] < TIMEOUT_S / 3600,
      rec['elapsed_h'])
ok, s = no_strays()
check('W no stray processes', ok, s)

# ---------------------------------------------------------------- C/D/E: aborts before training
R = make_root('C', corrupt=True)
rc, rec, out, dt = run_nb(R, {})
check('C corrupted tar -> abort at the store step', rc != 0 and 'tar sha256 mismatch' in rec['status'] and 'store' not in rec['stages'],
      rec['status'][:120])
R = make_root('D')
rc, rec, out, dt = run_nb(R, dict(DUMMY_BAD='contract'))
check('D wrong tensor_sha from the trainer -> probe abort, no training', rc != 0 and 'tensor_sha' in rec['status'] and
      'train' not in rec and not os.path.exists(f'{R}/kaggle/working/{RUNS[F0]}'), rec['status'][:160])
R = make_root('E')
rc, rec, out, dt = run_nb(R, dict(DUMMY_BAD='counts'))
check('E wrong val count from the trainer -> probe abort, no training', rc != 0 and 'val' in rec['status'] and 'train' not in rec,
      rec['status'][:160])
ok, s = no_strays()
check('C/D/E no stray processes', ok, s)
R = make_root('F', crlf=True, drop_wheel=True, lab_layout='legacy', lab_mode='lf')
rc, rec, out, dt = run_nb(R, {})
check('F CRLF folds5 + missing wheel + LF frac75 csv (legacy layout) -> 3 warnings, run DONE on the box bytes',
      rc == 0 and rec['status'] == 'DONE' and any('CRLF' in w for w in rec['warnings']) and
      any('missing or changed' in w for w in rec['warnings']) and any('LF->CRLF' in w for w in rec['warnings']) and
      '/kaggle/input/rsna-knee-frac75-labels/' in rec['inputs']['train_labels_src'] and
      sha(f'{R}/tmp/kt_labels/folds5.csv') == sha(f'{H}/rsna_train/folds5.csv') and
      sha(f'{R}/tmp/kt_labels/{FRAC}') == sha(f'{H}/novel/frac75/{FRAC}'), rec['warnings'])
ok, s = no_strays()
check('F no stray processes', ok, s)

R = make_root('H', dup_ds=True, lab_layout='both')
rc, rec, out, dt = run_nb(R, {})
check('H code + labels datasets each at 2 mount paths (identical) -> 2 warnings, run DONE', rc == 0 and rec['status'] == 'DONE'
      and any('identical kt_manifest' in w for w in rec['warnings']) and any('identical bytes' in w for w in rec['warnings']),
      (rec['status'], rec['warnings']))
ok, s = no_strays()
check('H no stray processes', ok, s)

# ---------------------------------------------------------------- L: the labels dataset guards (abort at step 1)
for ph, kw, frag in (('L1', dict(lab_layout='none'), 'attach dataset eeshsaxena/rsna-knee-frac75-labels'),
                     ('L2', dict(lab_mode='flip'), '!= baked'),
                     ('L3', dict(lab_mode='two'), 'DIFFERENT bytes'),
                     ('L4', dict(lab_mode='mix50'), '!= baked')):
    R = make_root(ph, **kw)
    rc, rec, out, dt = run_nb(R, {})
    check(f'{ph} {kw} -> abort at step 1 ("{frag}"), no store, no training', rc != 0 and frag in rec['status'] and
          'inputs' not in rec['stages'] and 'store' not in rec['stages'] and 'train' not in rec and
          not os.path.exists(f'{R}/tmp/fleet_store'), rec['status'][:170])
ok, s = no_strays()
check('L no stray processes', ok, s)

print(f'\nDRY TEST {KEY} ({NBK[0]}): {sum(RESULTS.values())}/{len(RESULTS)} PASS')
json.dump(RESULTS, open(f'{DRY}/results.json', 'w'), indent=1)
sys.exit(0 if all(RESULTS.values()) else 1)
