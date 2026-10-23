# kt_dry.py: box CPU dry test of rsna_train_ttr_f34.ipynb (never touches the GPU, ~/bigjump_build, ~/fleet_runs or
# ~/rsna_train except READS). Layout under ~/kt_test/dry/<phase>/{kaggle/input,kaggle/working,tmp}; the notebook code
# cells run UNCHANGED as one script with KT_ROOT=<phase dir> and a dummy trainer.
#   unit   lib cell only: s/it parser on the real J1 log + a fake log, header check, progress/closing, E arithmetic,
#          process-group stop (TERM and KILL paths), tar member guard
#   A      full flow, 2-tar subset (both mount layouts), real model/pretrained check, probe, E, train, receipt
#   B      watchdog: wall ~60 s after launch, dummies hang, fold 4 ignores SIGTERM -> SIGKILL, E < 16 warning
#   C      corrupted tar -> abort at the store step     D  wrong tensor_sha printed by the trainer -> probe abort
#   E      wrong val count printed by the trainer -> probe abort
import os, sys, json, time, shutil, hashlib, tarfile, subprocess, signal, glob, re
import pandas as pd

H = os.path.expanduser('~')
KT = f'{H}/kt_test'
DRY = f'{KT}/dry'
DSRC = f'{KT}/ds/rsna-knee-fleet-train'
NB = f'{KT}/kernel/rsna-train-ttr-f34/rsna_train_ttr_f34.ipynb'
PY = f'{H}/miniconda3/envs/rsna/bin/python'
RESULTS = {}


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def cells():
    nb = json.load(open(NB))
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']


def check(name, cond, info=''):
    RESULTS[name] = bool(cond)
    print(('PASS ' if cond else 'FAIL ') + name + (f'  [{info}]' if info else ''), flush=True)


# ---------------------------------------------------------------- fixture: 10 box store files in 2 tars
lab = pd.read_csv(f'{H}/rsna_train/labels_mix50_logit.csv', dtype={'StudyInstanceUID': str})
folds = pd.read_csv(f'{H}/rsna_train/folds5.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['fold']
box = json.load(open(f'{DSRC}/box_facts.json'))
probe4 = sorted(box['ckpt_args']['decode_probe'])
ids = sorted(lab.StudyInstanceUID)
pick = list(probe4)
for F in (3, 4, 1):
    pick += [s for s in ids if folds.get(s) == F and s not in pick][:2]
pick = pick[:10]
assert len(pick) == 10 and all(os.path.exists(f'{H}/fleet_store/{s}.npz') for s in pick)
FIX = f'{DRY}/fixture'
shutil.rmtree(DRY, ignore_errors=True)
os.makedirs(FIX)
for i, grp in enumerate((pick[:5], pick[5:])):
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


def make_root(phase, corrupt=False, crlf=False, drop_wheel=False, dup_ds=False):
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
    r = subprocess.run(['pgrep', '-u', str(os.getuid()), '-f', 'dummy_train.py|time.sleep\\(100000\\)'],
                       capture_output=True, text=True)
    return r.stdout.strip() == '', r.stdout.strip()


# ---------------------------------------------------------------- unit tests (lib cell only)
U = f'{DRY}/unit'
os.makedirs(f'{U}/kaggle/working')
os.environ['KT_ROOT'] = U
ns = {}
exec(compile(cells()[0], 'lib', 'exec'), ns)
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
f1 = open(f'{H}/fleet_runs/f1_s1043_ttr_mix50.log').read()
check('unit closing_line running fold-1 log -> None', ns['closing_line'](f1) is None)
ce = ns['choose_epochs']
r1 = ce(1.1, 1733, 0.4 * 3600, 10.3, 0.21, 120, 24)
r2 = ce(0.8, 1733, 0.4 * 3600, 10.3, 0.21, 120, 24)
r3 = ce(0.5, 1733, 0.4 * 3600, 10.3, 0.21, 120, 24)
r4 = ce(5.0, 1733, 10.0 * 3600, 10.3, 0.21, 120, 24)
check('unit E: 1.1 s/it -> 15 (< 16 warn)', r1['E'] == 15, f'epoch {r1["epoch_s"] / 60:.1f} min, avail {r1["avail_s"] / 3600:.2f} h')
check('unit E: 0.8 s/it -> 21', r2['E'] == 21, f'epoch {r2["epoch_s"] / 60:.1f} min')
check('unit E: 0.5 s/it -> 24 (cap)', r3['E'] == 24)
check('unit E: no time left -> 1 (floor)', r4['E'] == 1)
check('unit E: finish <= budget', r1['finish_elapsed_s'] <= 10.3 * 3600 and r2['finish_elapsed_s'] <= 10.3 * 3600)
check('unit save_epochs_arg', ns['save_epochs_arg'](24) == '23-24' and ns['save_epochs_arg'](1) == '1-1')


class _P:
    pass


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
wr = subprocess.Popen(['bash', '-c', f'for i in $(seq 1 30); do echo $i >> {qd}/y.pt; sleep 0.5; done'])
time.sleep(0.6)
wq = ns['wait_quiet']([qd], quiet_s=2, max_wait_s=3, poll_s=0.2)
wr.kill()
wr.wait()
check('unit wait_quiet: max_wait_s cap honoured', 2.9 <= wq <= 4, f'{wq:.2f} s')
os.makedirs(f'{U}/evil')
with tarfile.open(f'{U}/evil.tar', 'w') as tf:
    tf.add(f'{H}/fleet_store/{pick[0]}.npz', arcname=f'../{pick[0]}.npz')
try:
    ns['extract_tar']('evil.tar', f'{U}/evil.tar', f'{U}/evil')
    check('unit extract_tar rejects ../ member', False)
except ValueError as e:
    check('unit extract_tar rejects ../ member', True, str(e)[:80])
n, hx, nm, nb = ns['extract_tar']('s0', f'{FIX}/fleet_store_s0_p00.tar', f'{U}/evil')
check('unit extract_tar hash == file sha256, 5 members', hx == parts['fleet_store_s0_p00.tar'] and nm == 5 and
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
check('A store == box files (10 npz, sample sha, names sha)', rec['store']['n_npz'] == 10 and rec['store']['sample_ok'] == 10)
check('A parity: decode probe 4/4 + model keys + pretrained bytes', rec['parity']['decode_probe_match'] == 4 and
      rec['parity']['model_check']['keys_match'] and rec['parity']['model_check']['pretrained_equal'],
      rec['parity']['model_check'])
check('A pretrained from the bundled bytes, HF offline', rec['parity']['hf_offline'] == '1')
check('A probe s/it ~ dummy 0.3', all(abs(v - 0.3) < 0.08 for v in rec['probe']['s_per_it'].values()), rec['probe']['s_per_it'])
check('A E = 24 (budget not binding)', rec['plan']['E'] == 24 and rec['plan']['save_epochs'] == '23-24', rec['plan'])
for F in (3, 4):
    x = rec['train']['folds'][str(F)]
    pts = [f for f in x['files'] if f['path'].endswith('.pt')]
    check(f'A fold {F}: complete, 2 best + 4 epoch ckpts load with box keys',
          x['complete'] and len(x['best_ckpts']) == 2 and len(pts) == 6 and all(f['load_ok'] and f['keys_match'] for f in pts),
          [f['path'] for f in pts])
    check(f'A fold {F}: outputs in /kaggle/working/f{F}_s1043_ttr_mix50 + .log',
          os.path.isdir(f'{R}/kaggle/working/f{F}_s1043_ttr_mix50') and os.path.exists(f'{R}/kaggle/working/f{F}_s1043_ttr_mix50.log'))
check('A cmd = box recipe (+E, save_epochs, workers, paths)',
      '--model ttr --aug heavy --train_slices 16 --drop_path 0.2 --dropout 0.3 --ema 0.9995 --epochs 24 --seed 1043 '
      '--save_epochs 23-24 --workers 3 --store' in rec['train']['cmd']['3'] and rec['train']['cmd']['3'].endswith(
          f'--fold 3 --out {R}/kaggle/working/f3_s1043_ttr_mix50'), rec['train']['cmd']['3'])
check('A probe logs copied', all(os.path.exists(f'{R}/kaggle/working/probe/probe_f{F}.log') for F in (3, 4)))
ok, s = no_strays()
check('A no stray processes', ok, s)
launch_h = rec['train']['launch_h']
el_launch_s = launch_h * 3600

# ---------------------------------------------------------------- B: watchdog, 60 s wall
R = make_root('B')
rc, rec, out, dt = run_nb(R, dict(KT_WALL_H=str((el_launch_s + 60) / 3600), KT_BUDGET_H=str((el_launch_s + 10) / 3600),
                                  KT_KILL_GRACE_S='5', DUMMY_HANG_AFTER_EP='0', DUMMY_IGNORE_TERM_FOLDS='4'))
print(out[-5000:])
x3, x4 = rec['train']['folds']['3'], rec['train']['folds']['4']
check('B notebook exit 0, status PARTIAL', rc == 0 and rec['status'].startswith('PARTIAL'), rec['status'])
check('B E < 16 warning logged', rec['plan']['E'] < 16 and any('< 16' in w for w in rec['warnings']), rec['plan']['E'])
check('B wall warning logged', any('hard wall' in w for w in rec['warnings']))
check('B fold 3 SIGTERM, fold 4 SIGKILL', x3['ended'] == 'sigterm' and x4['ended'] == 'sigkill', (x3['ended'], x4['ended']))
check('B killed ~60 s after launch', abs(rec['train']['hours'] * 3600 - 60) < 25, f'{rec["train"]["hours"] * 3600:.0f} s')
check('B wall: quiet-wait ran before SIGTERM (outputs idle -> ~0 s)', rec['train'].get('wall_quiet_wait_s', 99) < 5,
      rec['train'].get('wall_quiet_wait_s'))
check('B best-so-far kept (2 best ckpts per fold, load OK)', all(len(x['best_ckpts']) == 2 and all(
    f.get('load_ok', True) for f in x['files']) for x in (x3, x4)))
ok, s = no_strays()
check('B no stray processes (workers killed with the group)', ok, s)

# ---------------------------------------------------------------- C/D/E: aborts before training
R = make_root('C', corrupt=True)
rc, rec, out, dt = run_nb(R, {})
check('C corrupted tar -> abort at the store step', rc != 0 and 'tar sha256 mismatch' in rec['status'] and 'store' not in rec['stages'],
      rec['status'][:120])
R = make_root('D')
rc, rec, out, dt = run_nb(R, dict(DUMMY_BAD='contract'))
check('D wrong tensor_sha from the trainer -> probe abort, no training', rc != 0 and 'tensor_sha' in rec['status'] and
      'train' not in rec and not os.path.exists(f'{R}/kaggle/working/f3_s1043_ttr_mix50'), rec['status'][:160])
R = make_root('E')
rc, rec, out, dt = run_nb(R, dict(DUMMY_BAD='counts'))
check('E wrong val count from the trainer -> probe abort, no training', rc != 0 and 'val' in rec['status'] and 'train' not in rec,
      rec['status'][:160])
ok, s = no_strays()
check('C/D/E no stray processes', ok, s)
R = make_root('F', crlf=True, drop_wheel=True)
rc, rec, out, dt = run_nb(R, {})
check('F CRLF folds5.csv + missing wheel -> 2 warnings, run DONE on the LF copy', rc == 0 and rec['status'] == 'DONE' and
      any('CRLF' in w for w in rec['warnings']) and any('missing or changed' in w for w in rec['warnings']) and
      sha(f'{R}/tmp/kt_labels/folds5.csv') == sha(f'{H}/rsna_train/folds5.csv'), rec['warnings'])
ok, s = no_strays()
check('F no stray processes', ok, s)

R = make_root('H', dup_ds=True)
rc, rec, out, dt = run_nb(R, {})
check('H dataset at 2 mount paths (identical manifest) -> warning, run DONE', rc == 0 and rec['status'] == 'DONE' and
      any('identical kt_manifest' in w for w in rec['warnings']), (rec['status'], rec['warnings']))
ok, s = no_strays()
check('H no stray processes', ok, s)

print(f'\nDRY TEST: {sum(RESULTS.values())}/{len(RESULTS)} PASS')
json.dump(RESULTS, open(f'{DRY}/results.json', 'w'), indent=1)
sys.exit(0 if all(RESULTS.values()) else 1)
