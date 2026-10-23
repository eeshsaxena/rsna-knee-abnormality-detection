#!/usr/bin/env python3
"""k8_sim_setup.py (box, CPU; TEST ONLY, never part of a real build; 2026-10-07, Claude). Builds a SIMULATION root
~/k8_test/sim/root (ours) so k8_collect.sh / build_k8.py / the A0 can be exercised before TTR288 folds 1-4 exist:

  set A    f{F}_s1043_ttr_mix50 (+ .log) and kaggle_f34_run_receipt.json: SYMLINKS to the real ~/fleet_runs entries
           (read only; the picker copies from them into ~/k8_test/.../ckpt).
  set C    f{F}_s1044_ttr288mix50: a SYMLINK to the real ~/fleet_runs run when its log already has the closing line
           (f0 since 10-07 00:09; f1 after ~13:40), else SIMULATED: the set A fold-F run (both tags) with ckpt args
           seed 1043 -> 1044 and ttr_size 224 -> 288 and the model buffer ttr_size_t -> 288 (torch re-save; weights
           untouched: TTR weights do not depend on ttr_size, the worker then SERVES them at 288, so the A0 runtime of the
           sim is the 288 cost), files renamed f{F}_s1044_<tag>_ttr288-..., OOFs copied, ckpt / OOF mtimes 0.5 s apart (one
           save() call), the log = the set A fold log + a leading '# SIMULATED' line (the closing line stays last).
Nothing under ~/fleet_runs is written. A real run that is still training is never touched (it is simulated instead).
  ~/miniconda3/envs/kmimic/bin/python k8_sim_setup.py [--out ~/k8_test/sim/root] [--sim_all]   (--sim_all: simulate f1-f4
  even when a real run has finished; f0 stays real)
"""
import argparse, glob, json, os, shutil, sys, time
import torch

H = os.path.expanduser('~')
FR = f'{H}/fleet_runs'


def finished(log):
    if not os.path.isfile(log):
        return False
    lines = [l.strip() for l in open(log, encoding='utf-8', errors='replace') if l.strip()]
    if not lines or not lines[-1].startswith('{'):
        return False
    try:
        d = json.loads(lines[-1])
    except Exception:
        return False
    return isinstance(d, dict) and {'raw', 'ema'} <= set(d)


def link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def sim_c(out, F):
    src = f'{FR}/f{F}_s1043_ttr_mix50'
    name = f'f{F}_s1044_ttr288mix50'
    dst = f'{out}/{name}'
    shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
    t0 = time.time() - 3600
    for k, tag in enumerate(('raw', 'ema')):
        (p,) = glob.glob(f'{src}/f{F}_s1043_{tag}_*_best.pt')
        ck = torch.load(p, map_location='cpu', weights_only=False)
        a = ck['args']
        assert int(a['seed']) == 1043 and int(a['fold']) == F and int(a['ttr_size']) == 224, (p, a.get('seed'), a.get('ttr_size'))
        ck['args'] = dict(a, seed=1044, ttr_size=288)
        ck['model']['ttr_size_t'] = torch.tensor(288, dtype=torch.int64)
        q = f'{dst}/{os.path.basename(p).replace("_s1043_", "_s1044_").replace("_ttr224-", "_ttr288-")}'
        torch.save(ck, q)
        oq = f'{dst}/oof_f{F}_s1044_{tag}.csv'
        shutil.copyfile(f'{src}/oof_f{F}_s1043_{tag}.csv', oq)
        tc = t0 + 600 * k
        os.utime(q, (tc, tc)); os.utime(oq, (tc + 0.5, tc + 0.5))
        print(f'  sim C f{F} {tag}: {os.path.basename(q)} cv {ck["cv"]:.6f} epoch {int(ck["epoch"]) + 1} '
              f'args seed {ck["args"]["seed"]} ttr {ck["args"]["ttr_size"]} buffer {int(ck["model"]["ttr_size_t"])}', flush=True)
        del ck
    log = open(f'{src}.log', encoding='utf-8', errors='replace').read()
    open(f'{dst}.log', 'w', encoding='utf-8').write(
        f'# SIMULATED by k8_sim_setup.py: set A f{F} weights relabelled seed 1044 / ttr_size 288 (TEST ONLY)\n' + log)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{H}/k8_test/sim/root')
    ap.add_argument('--sim_all', action='store_true')
    a = ap.parse_args()
    assert os.path.realpath(a.out).startswith(os.path.realpath(f'{H}/k8_test') + '/'), 'the sim root must live under ~/k8_test'
    os.makedirs(a.out, exist_ok=True)
    for F in range(5):
        r = f'f{F}_s1043_ttr_mix50'
        link(f'{FR}/{r}', f'{a.out}/{r}'); link(f'{FR}/{r}.log', f'{a.out}/{r}.log')
    link(f'{FR}/kaggle_f34_run_receipt.json', f'{a.out}/kaggle_f34_run_receipt.json')
    kinds = {}
    for F in range(5):
        r = f'f{F}_s1044_ttr288mix50'
        real = finished(f'{FR}/{r}.log') and not (a.sim_all and F > 0)
        if os.path.lexists(f'{a.out}/{r}') and not os.path.islink(f'{a.out}/{r}'):
            shutil.rmtree(f'{a.out}/{r}')
        if real:
            link(f'{FR}/{r}', f'{a.out}/{r}'); link(f'{FR}/{r}.log', f'{a.out}/{r}.log')
            kinds[F] = 'REAL (symlink)'
        else:
            for x in (f'{a.out}/{r}', f'{a.out}/{r}.log'):
                if os.path.islink(x):
                    os.remove(x)
            sim_c(a.out, F); kinds[F] = 'SIMULATED (set A weights, seed 1044, ttr 288)'
    for F, k in kinds.items():
        print(f'set C f{F}: {k}')
    print('K8SIM SETUP OK', a.out, json.dumps(kinds))


if __name__ == '__main__':
    main()
