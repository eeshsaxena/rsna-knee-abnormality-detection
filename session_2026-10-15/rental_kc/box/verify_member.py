# verify_member.py (rental_kc, BOX, CPU, nice 19; called by fetch_loop.sh): verify one pulled member BEFORE it is placed.
# usage: verify_member.py TMPROOT KEY --reg members.tsv --bundle BUNDLE.json --parity parity_box.json [--ref f0 ckpt] [--root R]
# TMPROOT holds <out>/ (member dir), <out>.log, <out>.receipt.json, <out>.sha256 (+ <out>.gate_verdict*) exactly as pulled.
# Checks: every file listed in <out>.sha256 matches, no unlisted file; closing line; the expected files (AD: top-level raw + ema
# last-epoch ckpts, raw + ema ckpts and gold74 csvs for every --save_epochs epoch; KC2 f0: best raw + ema, 2 oof csvs, the
# epoch ckpts, the G1 verdict copy); ckpt args == the queue argv (fold, seed, epochs, save_epochs, labels, backbone, lr, ...)
# and the pushed bundle (kc contract hash, model_sha, pretrained sha) and, for the ConvNeXt-S members, every recipe key ==
# the box f0 checkpoint (--ref) except the intended differences (fold, seed, save_epochs, labels, decode_probe ids differ
# never: same store); receipt rc 0, argv == the registry template (paths -> R), labels sha256 == the bundle's table.
# Prints one JSON line; exit 0 = VERIFY_OK. No em dashes.
import argparse, glob, hashlib, json, os, re, sys
ap = argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('key')
ap.add_argument('--reg', required=True); ap.add_argument('--bundle', required=True); ap.add_argument('--parity', required=True)
ap.add_argument('--ref', default=''); ap.add_argument('--root_inst', '--root', dest='rinst', default='/workspace/rsna')
ap.add_argument('--skip', default='')
A = ap.parse_args()
os.nice(19)
reg = {l.split('\t')[0]: l.rstrip('\n').split('\t') for l in open(A.reg) if l.strip()}
B = json.load(open(A.bundle)); PB = json.load(open(A.parity))
k = A.key; _, N, kind, prio, src, gate, tmpl = reg[k]
argv = tmpl.replace('@R@', A.rinst).split()
opt = {argv[i]: (argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith('--') else True) for i in range(len(argv)) if argv[i].startswith('--')}
RT = A.root
res = dict(run=N, key=k, problems=[]); bad = res['problems']
SKIP = set(filter(None, A.skip.split(',')))


def sha(p):
    m = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            m.update(b)
    return m.hexdigest()


man = {}
for l in open(f'{RT}/{N}.sha256'):
    if l.strip():
        h, p = l.rstrip('\n').split('  ', 1); man[p.lstrip('*')] = h
got = sorted(os.path.relpath(os.path.join(d, f), RT) for d, _, fs in os.walk(RT) for f in fs)
got = [g for g in got if g not in (f'{N}.sha256', '.pulled')]
for p, h in man.items():
    if not os.path.isfile(f'{RT}/{p}'):
        bad.append(f'missing {p}')
    elif sha(f'{RT}/{p}') != h:
        bad.append(f'sha256 mismatch {p}')
extra = sorted(set(got) - set(man))
if extra:
    bad.append(f'unlisted files {extra[:5]}')
res['files'] = len(man); res['bytes'] = sum(os.path.getsize(f'{RT}/{p}') for p in man if os.path.isfile(f'{RT}/{p}'))
log = open(f'{RT}/{N}.log', errors='replace').read().rstrip().splitlines()
res['closing'] = log[-1][:120] if log else ''
if not (log and re.match(r'^\{"raw": [^,]+, "ema": [^}]+\}$', log[-1].strip())):
    bad.append('no closing line')
fold, seed, E = int(opt['--fold']), int(opt['--seed']), int(opt['--epochs'])
a, _, b = str(opt.get('--save_epochs', '')).partition('-'); save = list(range(int(a), int(b or a) + 1)) if a else []
md = f'{RT}/{N}'
ep_ck = {(t, e): glob.glob(f'{md}/epochs/f{fold}_s{seed}_{t}_*_ep{e}_best.pt') for t in ('raw', 'ema') for e in save}
miss_ep = [f'{t}{e}' for (t, e), v in ep_ck.items() if len(v) != 1]
gold = [e for e in save for t in ('raw', 'ema') if not os.path.exists(f'{md}/epochs/gold_f{fold}_s{seed}_{t}_ep{e}.csv')]
if miss_ep:
    bad.append(f'epoch ckpts missing {miss_ep[:6]}')
if gold:
    bad.append(f'gold74 epoch csvs missing for epochs {sorted(set(gold))[:6]}')
if fold < 0:
    top = [glob.glob(f'{md}/f{fold}_s{seed}_{t}_*_ep{E}_best.pt') for t in ('raw', 'ema')]
    if any(len(t) != 1 for t in top):
        bad.append('top-level last-epoch raw / ema ckpts')
else:
    top = [glob.glob(f'{md}/f{fold}_s{seed}_{t}_*_cv*_best.pt') for t in ('raw', 'ema')]
    if any(len(t) != 1 for t in top):
        bad.append('best raw / ema ckpts')
    if len(glob.glob(f'{md}/oof_f{fold}_s{seed}_*.csv')) != 2:
        bad.append('top-level oof csvs != 2')
    if '--gate_out' in opt and not os.path.exists(f'{RT}/{N}.gate_verdict'):
        bad.append('KC2 G1 verdict copy missing')
res.update(epoch_ckpts=sum(len(v) for v in ep_ck.values()), top=[os.path.basename(t[0]) for t in top if t])
import torch
torch.set_num_threads(2)
rc = json.load(open(f'{RT}/{N}.receipt.json'))
fb = int(rc.get('grad_ckpt_fallback', 0))
lab = os.path.basename(opt['--labels'])
want = dict(fold=fold, seed=seed, epochs=E, save_epochs=str(opt.get('--save_epochs', '')), train_labels=lab,
            sel_labels=os.path.basename(opt.get('--eval_labels') or opt['--labels']), backbone=opt['--backbone'],
            amp='fp16', kc_contract_hash=B['parity_box']['contract_hash'], mirror=int(opt.get('--mirror', 0)),
            vbias_prior=int(opt.get('--vbias_prior', 0)), train_slices=int(opt['--train_slices']), aug=opt['--aug'],
            drop_path=float(opt['--drop_path']), dropout=float(opt['--dropout']), lr=float(opt['--lr']), head_lr=float(opt['--head_lr']),
            max_steps=0, model_mod='kc_f2' if '--f2' in opt else 'kc_ttr',
            model_sha=PB['model_sha_f2'] if '--f2' in opt else PB['model_sha_ttr'])
if '--f2' not in opt:
    want.update(ttr_size=int(opt['--ttr_size']), ema=float(opt['--ema']), wd=float(opt['--wd']), pretrained_sha256=PB['pretrained_sha256'])
if not fb:
    want['grad_ckpt'] = False
want = {kk: v for kk, v in want.items() if kk not in SKIP}
ref = torch.load(A.ref, map_location='cpu', weights_only=False)['args'] if (A.ref and os.path.exists(A.ref) and '--f2' not in opt) else None
INTENDED = {'fold', 'seed', 'save_epochs', 'train_labels', 'sel_labels', 'decode_probe'} | SKIP   # decode_probe = sha of the first 4 labelled ids (label-table dependent)
ck = []
for p in sorted(sum(ep_ck.values(), []) + [t[0] for t in top if t]):
    c = torch.load(p, map_location='cpu', weights_only=False); g = c['args']
    diff = {kk: g.get(kk) for kk, v in want.items() if g.get(kk) != v}
    if diff:
        bad.append(f'{os.path.basename(p)} args {diff}')
    if ref is not None:
        common = sorted((set(g) & set(ref)) - INTENDED); rd = [kk for kk in common if g[kk] != ref[kk]]
        one = sorted((set(g) ^ set(ref)) - INTENDED)
        if rd:
            bad.append(f'{os.path.basename(p)} recipe keys differ from the box f0: {rd}')
        if one:
            bad.append(f'{os.path.basename(p)} one-sided keys vs the box f0: {one}')
        ck.append(f'{os.path.basename(p)}: epoch {c["epoch"] + 1} | recipe {len(common) - len(rd)}/{len(common)} keys == box f0')
    else:
        ck.append(f'{os.path.basename(p)}: epoch {c["epoch"] + 1} | argv keys {len(want) - len(diff)}/{len(want)}')
res['ckpts'] = ck[-3:] + [f'... {len(ck)} ckpts checked']
res['recipe_ref'] = os.path.basename(A.ref) if ref is not None else ('none (KC2 f2 path: argv keys only)' if '--f2' in opt else 'none')
if rc.get('rc') != 0 or rc.get('key') != k:
    bad.append(f'receipt rc/key {rc.get("rc")}/{rc.get("key")}')
want_argv = ' '.join(argv)
if fb:
    want_argv = re.sub(r' --grad_ckpt 0( |$)', r' --grad_ckpt 1\1', want_argv)
if rc.get('argv', '') != want_argv:
    bad.append('receipt argv != the registry template (queue argv, paths -> instance root)')
if rc.get('table_sha256') != B['tables'].get(lab):
    bad.append(f'receipt labels sha256 != the bundle table {lab}')
res.update(gpu=rc.get('gpu_name'), wall_h=round(rc.get('wall_s', 0) / 3600, 2), grad_ckpt_fallback=fb, container=rc.get('container_id'))
res['verdict'] = 'VERIFY_OK' if not bad else 'VERIFY_FAIL'
print(json.dumps(res))
sys.exit(0 if not bad else 1)
