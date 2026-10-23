#!/usr/bin/env python3
"""k10_sim_setup.py (box, CPU; TEST ONLY, never part of a real build; 2026-10-09, Claude; from k9_sim_setup.py). Builds a
SIMULATION root (default ~/k10_test/sim/root, ours) so k10_collect.sh / build_k10*.py / the A0s can run before set E exists:

  set C  f{F}_s1044_ttr288mix50 : SYMLINK to the real ~/fleet_runs run (all 5 finished 10-08)
  set D  f{F}_s1043_ttr_frac75  : SYMLINK to the real run; the 3 frac75 Kaggle receipts + f34 receipt SYMLINKED
  set E  f{F}_s1045_ttr288frac75: SIMULATED from set C fold F (both tags): weights untouched, ckpt args seed 1045, train_labels
         labels_frac75_logit.csv, epochs 21; files renamed _s1044_ -> _s1045_; OOFs copied (same predictions, so the
         picker's OOF rescore == ckpt cv holds); the log = '# SIMULATED' + the set C log with the header '| epochs 24 |' ->
         '| epochs 21 |' (the 'ep 20 raw/ema' lines and the closing line are there); folds 1-4 get a SIMULATED
         vast320_<run>_receipt.json in box_ops.sh verify format (mode canonical, verify OK, closing_line, ckpts args OK).
  --neg_receipts: fold 1 receipt deleted, fold 2 verify "md5:[...]", fold 3 closing_line wrong, fold 4 run name wrong
         (picker negatives).
Nothing under ~/fleet_runs is written. Real runs are only read (symlinks / copies).
  ~/miniconda3/envs/kmimic/bin/python k10_sim_setup.py [--out ~/k10_test/sim/root] [--neg_receipts]
"""
import argparse, glob, hashlib, json, os, shutil, time
import torch

H = os.path.expanduser('~')
FR = f'{H}/fleet_runs'
FRAC75 = 'labels_frac75_logit.csv'


def finished(log):
    lines = [l.strip() for l in open(log, encoding='utf-8', errors='replace') if l.strip()] if os.path.isfile(log) else []
    try:
        d = json.loads(lines[-1]) if lines and lines[-1].startswith('{') else None
    except Exception:
        d = None
    return isinstance(d, dict) and {'raw', 'ema'} <= set(d)


def link(src, dst):
    if os.path.lexists(dst):
        if os.path.isdir(dst) and not os.path.islink(dst):
            shutil.rmtree(dst)
        else:
            os.remove(dst)
    os.symlink(src, dst)


def clear(out, name):
    for x in (f'{out}/{name}', f'{out}/{name}.log'):
        if os.path.islink(x) or os.path.isfile(x):
            os.remove(x)
        elif os.path.isdir(x):
            shutil.rmtree(x)


def sim_e(out, F):
    src = f'{FR}/f{F}_s1044_ttr288mix50'
    name = f'f{F}_s1045_ttr288frac75'
    dst = f'{out}/{name}'
    clear(out, name); os.makedirs(dst)
    t0 = time.time() - 3600
    descr = []
    for k, tag in enumerate(('raw', 'ema')):
        (p,) = glob.glob(f'{src}/f{F}_s1044_{tag}_*_best.pt')
        ck = torch.load(p, map_location='cpu', weights_only=False)
        a = ck['args']
        assert int(a['seed']) == 1044 and int(a['fold']) == F and int(a['ttr_size']) == 288, p
        assert a.get('train_labels') == 'labels_mix50_logit.csv' and int(a['epochs']) == 24, p
        ck['args'] = dict(a, seed=1045, train_labels=FRAC75, epochs=21)
        q = f'{dst}/{os.path.basename(p).replace("_s1044_", "_s1045_")}'
        oq = f'{dst}/oof_f{F}_s1045_{tag}.csv'
        torch.save(ck, q)
        shutil.copyfile(f'{src}/oof_f{F}_s1044_{tag}.csv', oq)
        tc = t0 + 600 * k
        os.utime(q, (tc, tc)); os.utime(oq, (tc + 0.5, tc + 0.5))
        descr.append((os.path.basename(q), int(ck['epoch']) + 1, float(ck['cv'])))
        print(f'  sim E f{F} {tag}: {os.path.basename(q)} cv {ck["cv"]:.6f} epoch {int(ck["epoch"]) + 1} args seed 1045 '
              f'labels {FRAC75} epochs 21', flush=True)
        del ck
    log = open(f'{src}.log', encoding='utf-8', errors='replace').read()
    assert log.count('| epochs 24 |') == 1, 'set C log header'
    log = log.replace('| epochs 24 |', '| epochs 21 |')
    open(f'{dst}.log', 'w', encoding='utf-8').write(
        f'# SIMULATED by k10_sim_setup.py: set C f{F} weights relabelled as set E (s1045, frac75, 21 ep; TEST ONLY)\n' + log)
    return name, dst, sorted(descr)


def receipt(out, name, dst, descr, neg=None):
    lines = [l for l in open(f'{dst}.log', encoding='utf-8', errors='replace').read().splitlines() if l.strip()]
    cl = lines[-1].replace('\r', '')[:120]
    nman = len([f for f in os.listdir(dst)]) + 1
    ck = ' ; '.join(f'{b}: epoch {e} cv {c:.4f} args OK' for b, e, c in descr)
    r = dict(run=name, source_run=name, mode='canonical', vast_instance='SIMULATED', gpu='SIM_RTX_3090', offer='0',
             fetched_at=time.strftime('%Y-%m-%d %H:%M:%S %Z'), files=nman, best_raw=1, best_ema=1, top_oof=2, epoch_oof=0,
             closing_line=cl, ckpts=ck, verify='OK', simulated=True)
    if neg == 'verify':
        r['verify'] = 'md5:[f2 SIMULATED mismatch]'
    elif neg == 'closing':
        r['closing_line'] = '{"raw": 0.1, "ema": 0.1}'
    elif neg == 'run':
        r['run'] = r['source_run'] = name.replace('_s1045_', '_s1046_')
    p = f'{out}/vast320_{name}_receipt.json'
    if os.path.lexists(p):
        os.remove(p)
    if neg != 'missing':
        json.dump(r, open(p, 'w'), indent=1)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{H}/k10_test/sim/root')
    ap.add_argument('--neg_receipts', action='store_true')
    a = ap.parse_args()
    assert os.path.realpath(a.out).startswith(os.path.realpath(f'{H}/k10_test') + '/'), 'the sim root must live under ~/k10_test'
    os.makedirs(a.out, exist_ok=True)
    for rc in ('kaggle_f34_run_receipt.json', 'kaggle_frac75_f01_run_receipt.json', 'kaggle_frac75_f23_run_receipt.json',
               'kaggle_frac75_f4x_run_receipt.json'):
        assert os.path.isfile(f'{FR}/{rc}'), rc
        link(f'{FR}/{rc}', f'{a.out}/{rc}')
    kinds = {}
    for kind, fmt in (('C', 'f{F}_s1044_ttr288mix50'), ('D', 'f{F}_s1043_ttr_frac75')):
        for F in range(5):
            r = fmt.format(F=F)
            assert finished(f'{FR}/{r}.log'), f'{r} is not finished'
            clear(a.out, r)
            link(f'{FR}/{r}', f'{a.out}/{r}'); link(f'{FR}/{r}.log', f'{a.out}/{r}.log')
            kinds[f'{kind}{F}'] = 'REAL (symlink)'
    negs = {1: 'missing', 2: 'verify', 3: 'closing', 4: 'run'} if a.neg_receipts else {}
    for F in range(5):
        name, dst, descr = sim_e(a.out, F)
        kinds[f'E{F}'] = 'SIMULATED (set C weights, s1045 / frac75 / 21 ep args)'
        if F >= 1:
            receipt(a.out, name, dst, descr, negs.get(F))
            kinds[f'vast receipt f{F}'] = 'SIMULATED' + (f' NEGATIVE ({negs[F]})' if F in negs else '')
    for k, v in kinds.items():
        print(f'{k}: {v}')
    print('K10SIM SETUP OK', a.out, json.dumps(kinds))


if __name__ == '__main__':
    main()
