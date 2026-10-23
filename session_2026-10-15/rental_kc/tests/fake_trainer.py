# fake_trainer.py (rental_kc tests only): stands in for `python fleet_train.py --kc <argv>` in the box mock (no GPU, ~no CPU).
# Mimics the KC trainer's observable contract: progress lines 'ep E it I/M loss L Xs/it', 'ep E TRAIN-ONLY', 'ep E raw | ...'
# lines, the files (AD: top-level last-epoch raw/ema ckpts, epochs/ raw+ema ckpts and gold csvs for --save_epochs; fold 0:
# best raw/ema cv ckpts, 2 oof csvs, epoch oof csvs, --gate_out verdict + json) and the closing line. Ckpt args are built
# from the argv + kc_common / model sources of the code dir exactly as fleet_train_kc.py names them (so verify_member.py is
# exercised). Env: FAKE_EP_S (seconds per epoch, default 1), FAKE_SIT (s/it printed, default 0.25), FAKE_HOLD_SMOKE=1 (an
# --out under /smoke/ sleeps after epoch-0 progress until killed), FAKE_FAIL=1 (exit 1 after epoch 0). No em dashes.
import os, sys, json, time, hashlib, inspect
argv = [a for a in sys.argv[1:] if a != '--kc']
opt = {}
i = 0
while i < len(argv):
    if argv[i].startswith('--'):
        if i + 1 < len(argv) and not argv[i + 1].startswith('--'):
            opt[argv[i]] = argv[i + 1]; i += 2
        else:
            opt[argv[i]] = True; i += 1
    else:
        i += 1
code = os.getcwd(); sys.path.insert(0, code)
import torch, pandas as pd
import kc_common as fc
out = opt['--out']; fold = int(opt['--fold']); seed = int(opt['--seed']); E = int(opt['--epochs'])
a, _, b = opt.get('--save_epochs', '').partition('-'); save = range(int(a), int(b or a) + 1) if a else range(0)
f2 = '--f2' in opt
import fleet_model as _fm
src = chr(10).join(inspect.getsource(_fm).splitlines())
if f2:
    import fleet_model_f2 as _m
else:
    import fleet_model_ttr as _m
model_sha = hashlib.sha256((chr(10).join(inspect.getsource(_m).splitlines()) + chr(10) + src).encode()).hexdigest()[:16]
pre = ''
if opt.get('--pretrained_file'):
    pre = hashlib.sha256(open(opt['--pretrained_file'], 'rb').read()).hexdigest()
lab = os.path.basename(opt['--labels']); sel = os.path.basename(opt.get('--eval_labels') or opt['--labels'])
args = dict(backbone=opt['--backbone'], dim=256, contrast_emb=False, n_slices=fc.N_SLICES, n_views=len(fc.VIEWS), fold=fold, seed=seed,
            amp='fp16', bf16=False, epochs=E, stop_epoch=0, train_slices=int(opt['--train_slices']), sel_labels=sel, train_labels=lab,
            mirror_aug=0.0, model_sha=model_sha, decode_probe={'x': fold}, aug=opt['--aug'], drop_path=float(opt['--drop_path']),
            dropout=float(opt['--dropout']), save_epochs=opt.get('--save_epochs', ''), lr=float(opt['--lr']), head_lr=float(opt['--head_lr']),
            model_mod='kc_f2' if f2 else 'kc_ttr', kc_contract_hash=fc.CONTRACT_HASH, kc_views=fc.VIEW_NAMES, mirror=int(opt.get('--mirror', 0)),
            vbias_prior=int(opt.get('--vbias_prior', 0)), pretrained_file=os.path.basename(opt.get('--pretrained_file', '')),
            pretrained_sha256=pre[:16], max_steps=0, kc_trainer='fleet_train_kc.py')
if not f2:
    args.update(ttr_size=int(opt['--ttr_size']), ema=float(opt['--ema']), wd=float(opt['--wd']))
if opt.get('--grad_ckpt') == '0':
    args['grad_ckpt'] = False
tag = ('kcev2s-' if f2 else f'kcttr{opt.get("--ttr_size", 256)}-') + f'{fc.STORE}x{fc.N_SLICES}'
M = 4333 if fold < 0 else 3466
sit = float(os.environ.get('FAKE_SIT', '0.25')); eps = float(os.environ.get('FAKE_EP_S', '1'))
os.makedirs(out, exist_ok=True)
print(f'KC args: model_mod {args["model_mod"]} | contract {fc.CONTRACT_HASH} | fake trainer', flush=True)
T = ['ACL', 'MCL']
for ep in range(E):
    for it in range(0, M, 200):
        print(f'ep {ep} it {it}/{M} loss 0.3000 {sit:.2f}s/it', flush=True)
        time.sleep(eps / (M / 200))
        if ep == 0 and it >= 1000 and os.environ.get('FAKE_HOLD_SMOKE') == '1' and '/smoke/' in out:
            time.sleep(3600)
    print(f'ep {ep} TRAIN-ONLY {M * sit / 60:.1f} min (G0 timing)', flush=True)
    if os.environ.get('FAKE_FAIL') == '1':
        print('RuntimeError: fake failure', flush=True); sys.exit(1)
    e = ep + 1
    for t in ('raw', 'ema'):
        if e in save:
            os.makedirs(f'{out}/epochs', exist_ok=True)
            pd.DataFrame({'StudyInstanceUID': ['1.2.826.0.1.3680043.8.498.1'], 'ACL': [0.5]}).to_csv(f'{out}/epochs/gold_f{fold}_s{seed}_{t}_ep{e}.csv', index=False)
            torch.save({'model': {}, 'args': args, 'cv': float('nan'), 'epoch': ep}, f'{out}/epochs/f{fold}_s{seed}_{t}_{tag}_ep{e}_best.pt')
            if fold >= 0:
                pd.DataFrame({'StudyInstanceUID': ['1.2.826.0.1.3680043.8.498.1'], 'ACL': [0.5]}).to_csv(f'{out}/epochs/oof_f{fold}_s{seed}_{t}_ep{e}.csv', index=False)
        if fold < 0:
            if e == E:
                torch.save({'model': {}, 'args': args, 'cv': float('nan'), 'epoch': ep}, f'{out}/f{fold}_s{seed}_{t}_{tag}_ep{e}_best.pt')
            print(f'ep {ep} {t} | gold74 0.9000 (12 labels)' + (' | saved' if e == E else '') + (' | epoch saved' if e in save else ''), flush=True)
        else:
            for q in [q for q in os.listdir(out) if q.startswith(f'f{fold}_s{seed}_{t}_')]:
                os.remove(f'{out}/{q}')
            torch.save({'model': {}, 'args': args, 'cv': 0.9 + e / 1000, 'epoch': ep}, f'{out}/f{fold}_s{seed}_{t}_{tag}_cv{9000 + e}_best.pt')
            pd.DataFrame({'StudyInstanceUID': ['1.2.826.0.1.3680043.8.498.1'], 'ACL': [0.5]}).to_csv(f'{out}/oof_f{fold}_s{seed}_{t}.csv', index=False)
            print(f'ep {ep} {t} | gold74 0.9000 (12 labels) | val AUC vs {sel} {0.9 + e / 1000:.4f} (12 labels) | epoch 1.0 min', flush=True)
if opt.get('--gate_out'):
    open(opt['--gate_out'], 'w').write('PASS kc_f0 fake\n# fake trainer\n'); json.dump({'verdict': 'PASS'}, open(opt['--gate_out'] + '.json', 'w'))
print(json.dumps({'raw': -1.0 if fold < 0 else 0.913, 'ema': -1.0 if fold < 0 else 0.912}), flush=True)
