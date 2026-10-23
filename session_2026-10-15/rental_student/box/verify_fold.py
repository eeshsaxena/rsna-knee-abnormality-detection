# verify_fold.py (BOX, CPU, nice 19; called by fetch_loop.sh): verify one pulled fold BEFORE it is placed in ~/fleet_runs.
# usage: verify_fold.py TMPROOT NAME FOLD --mode canonical|partial [--table labels_r7t_b.csv] [--kref DIR] [--skip k1,k2]
# TMPROOT holds NAME/ (fold dir), NAME.log, NAME.receipt.json, NAME.sha256 exactly as pulled from the instance.
#  canonical: every file listed in NAME.sha256 matches; no unlisted file; closing line; exactly 1 raw + 1 ema best ckpt,
#             2 top-level oof csvs; ckpt args (fold, seed, ttr, slices, epochs, save_epochs, model_sha, train_labels ==
#             sel_labels == the table, grad_ckpt False, amp fp16) and every other recipe key == the box reference run
#             (set D fold 2, ~/fleet_runs/f2_s1043_ttr_frac75) except the intended differences; receipt rc 0, fold, seed,
#             argv == the template, table sha256 == the bundle's
#  partial:   sha256 of the listed files only (a near-deadline pull of a running fold; never a student)
# Prints one JSON line; exit 0 = VERIFY_OK.
import argparse, glob, hashlib, json, os, re, sys
ap = argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('name'); ap.add_argument('fold', type=int)
ap.add_argument('--mode', default='canonical'); ap.add_argument('--table', default='labels_r7t_b.csv')
ap.add_argument('--seed', type=int, default=1047); ap.add_argument('--ttr', type=int, default=224); ap.add_argument('--tsl', type=int, default=16)
ap.add_argument('--epochs', type=int, default=21); ap.add_argument('--model_sha', default='f141aeed136dfc01')
ap.add_argument('--kref', default=os.path.expanduser('~/fleet_runs/f2_s1043_ttr_frac75'))
ap.add_argument('--skip', default=''); ap.add_argument('--argv', default=''); ap.add_argument('--table_sha256', default='')
A = ap.parse_args()
os.nice(19)
R, N, F = A.root, A.name, A.fold
res = dict(run=N, fold=F, mode=A.mode, problems=[])
bad = res['problems']
def sha(p):
    m = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): m.update(b)
    return m.hexdigest()
man = {}
for l in open(f'{R}/{N}.sha256'):
    if l.strip():
        h, p = l.rstrip('\n').split('  ', 1); man[p.lstrip('*')] = h
got = sorted(os.path.relpath(os.path.join(d, f), R) for d, _, fs in os.walk(R) for f in fs)
got = [g for g in got if g != f'{N}.sha256']
for p, h in man.items():
    if not os.path.isfile(f'{R}/{p}'): bad.append(f'missing {p}')
    elif sha(f'{R}/{p}') != h: bad.append(f'sha256 mismatch {p}')
extra = sorted(set(got) - set(man))
if extra: bad.append(f'unlisted files {extra[:5]}')
res['files'] = len(man); res['bytes'] = sum(os.path.getsize(f'{R}/{p}') for p in man if os.path.isfile(f'{R}/{p}'))
if A.mode == 'canonical':
    log = open(f'{R}/{N}.log', errors='replace').read().rstrip().splitlines()
    res['closing'] = log[-1][:120] if log else ''
    if not (log and re.match(r'^\{"raw": [^,]+, "ema": [^}]+\}$', log[-1].strip())): bad.append('no closing line')
    braw = glob.glob(f'{R}/{N}/f{F}_s{A.seed}_raw_*_best.pt'); bema = glob.glob(f'{R}/{N}/f{F}_s{A.seed}_ema_*_best.pt')
    oofs = glob.glob(f'{R}/{N}/oof_f{F}_s{A.seed}_*.csv')
    res.update(best_raw=len(braw), best_ema=len(bema), top_oof=len(oofs),
               epoch_ckpts=len(glob.glob(f'{R}/{N}/epochs/*.pt')), epoch_oof=len(glob.glob(f'{R}/{N}/epochs/oof_*.csv')))
    if len(braw) != 1 or len(bema) != 1: bad.append(f'best ckpts raw {len(braw)} ema {len(bema)}')
    if len(oofs) != 2: bad.append(f'top-level oof csvs {len(oofs)}')
    import torch
    torch.set_num_threads(2)
    refp = sorted(glob.glob(f'{A.kref}/f2_s1043_raw_*_best.pt'))
    ref = torch.load(refp[0], map_location='cpu', weights_only=False)['args'] if refp else None
    SKIP = {'fold', 'epochs', 'save_epochs', 'grad_ckpt', 'seed', 'train_labels', 'sel_labels'} | set(filter(None, A.skip.split(',')))
    ck = []
    for p in sorted(braw + bema):
        a = torch.load(p, map_location='cpu', weights_only=False); g = a['args']
        want = dict(fold=F, seed=A.seed, ttr_size=A.ttr, train_slices=A.tsl, epochs=A.epochs, save_epochs='6-21',
                    model_sha=A.model_sha, train_labels=A.table, sel_labels=A.table, grad_ckpt=False, amp='fp16')
        want = {k: v for k, v in want.items() if k not in set(filter(None, A.skip.split(',')))}   # tests only (CPU mock: amp fp32)
        diff = {k: g.get(k) for k, v in want.items() if g.get(k) != v}
        if diff: bad.append(f'{os.path.basename(p)} args {diff}')
        if ref is None: bad.append('no reference run for the recipe check')
        else:
            common = sorted((set(g) & set(ref)) - SKIP); rd = [k for k in common if g[k] != ref[k]]
            one = sorted((set(g) ^ set(ref)) - SKIP)
            if rd: bad.append(f'{os.path.basename(p)} recipe keys differ from set D f2: {rd}')
            if one: bad.append(f'{os.path.basename(p)} one-sided keys: {one}')
            ck.append(f'{os.path.basename(p)}: epoch {a["epoch"] + 1} cv {a["cv"]:.4f} | recipe {len(common) - len(rd)}/{len(common)} keys == set D f2')
    res['ckpts'] = ck
    rc = json.load(open(f'{R}/{N}.receipt.json'))
    if rc.get('rc') != 0 or rc.get('fold') != F or rc.get('seed') != A.seed: bad.append(f'receipt rc/fold/seed {rc.get("rc")}/{rc.get("fold")}/{rc.get("seed")}')
    if A.argv and rc.get('argv', '').split()[:len(A.argv.split())] != A.argv.split(): bad.append('receipt argv != the template')
    if A.table_sha256 and rc.get('table_sha256') != A.table_sha256: bad.append('receipt table sha256 != the bundle table')
    res.update(gpu=rc.get('gpu_name'), wall_h=round(rc.get('wall_s', 0) / 3600, 2), container=rc.get('container_id'))
res['verdict'] = 'VERIFY_OK' if not bad else 'VERIFY_FAIL'
print(json.dumps(res))
sys.exit(0 if not bad else 1)
