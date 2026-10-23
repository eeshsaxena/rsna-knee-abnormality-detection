#!/usr/bin/env python3
"""k7_sim_setup.py (box, CPU; TEST ONLY, never part of a real build; 2026-10-06, Claude). Builds a SIMULATION root
~/k7_sim/root (ours) from READ-ONLY copies of real runs, so k7_collect.sh / build_k7.py / the A0 can be exercised
before set B exists and before set C has finished:

  set B sim  f{F}_s1044_ttr_mix50  = the set A run f{F}_s1043_ttr_mix50 (both tags), ckpt args seed 1043 -> 1044 (torch
             re-save, weights untouched), files renamed s1043 -> s1044, OOFs and log copied; three SIMULATED Kaggle run
             receipts kaggle_s1044_{f01,f23,f4}_run_receipt.json (cfg SEED 1044, runs, per-file sha256), like the
             teammate's notebooks will write.
  set C sim  f0_s1044_ttr288mix50  = cp -p of the LIVE TTR288 run's current best raw + ema ckpts and OOFs (sha256 of the
             copy == the source after the copy, else retried), its log + 3 SIMULATED lines (ep E-1 raw / ema and the
             closing {"raw", "ema"} line with the copied ckpts' own cv values), because the real run is still training.

Nothing under ~/fleet_runs is written. python k7_sim_setup.py [--out ~/k7_sim/root]
"""
import argparse, glob, hashlib, json, os, shutil, sys, time
import torch

H = os.path.expanduser('~')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def sim_b(out, F):
    src = f'{H}/fleet_runs/f{F}_s1043_ttr_mix50'
    name = f'f{F}_s1044_ttr_mix50'
    dst = f'{out}/{name}'
    os.makedirs(dst, exist_ok=True)
    files = []
    for tag in ('raw', 'ema'):
        (p,) = glob.glob(f'{src}/f{F}_s1043_{tag}_*_best.pt')
        ck = torch.load(p, map_location='cpu', weights_only=False)
        assert int(ck['args']['seed']) == 1043 and int(ck['args']['fold']) == F
        ck['args'] = dict(ck['args'], seed=1044)
        q = f'{dst}/{os.path.basename(p).replace("_s1043_", "_s1044_")}'
        torch.save(ck, q)
        o = f'{src}/oof_f{F}_s1043_{tag}.csv'
        oq = f'{dst}/oof_f{F}_s1044_{tag}.csv'
        shutil.copyfile(o, oq)
        for x in (q, oq):
            files.append(dict(path=f'{name}/{os.path.basename(x)}', bytes=os.path.getsize(x), sha256=sha(x)))
        del ck
    shutil.copyfile(f'{src}.log', f'{dst}.log')
    return name, files


def receipt(out, folds, entries, tag):
    rec = dict(kernel=f'SIMULATION swapiy/rsna-train-ttr-s1044-{tag}', status='DONE', simulated=True,
               cfg=dict(FOLDS=folds, SEED=1044), runs={str(F): entries[F][0] for F in folds}, warnings=[],
               plan=dict(E=24), train=dict(folds={str(F): dict(run=entries[F][0], complete=True, ended='exited', rc=0,
                                                              closing_line=None, files=entries[F][1],
                                                              progress=dict(epochs_done=24, best={}),
                                                              log=f'/kaggle/working/{entries[F][0]}.log')
                                                 for F in folds}))
    p = f'{out}/kaggle_s1044_{tag}_run_receipt.json'
    json.dump(rec, open(p, 'w'), indent=1)
    return p


def sim_c(out):
    src = f'{H}/fleet_runs/f0_s1044_ttr288mix50'
    dst = f'{out}/f0_s1044_ttr288mix50'
    for attempt in range(5):
        shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
        ok = True
        for p in sorted(glob.glob(f'{src}/f0_s1044_*_best.pt')) + sorted(glob.glob(f'{src}/oof_f0_s1044_*.csv')):
            q = f'{dst}/{os.path.basename(p)}'
            shutil.copy2(p, q)
            if not os.path.isfile(p) or sha(p) != sha(q):
                ok = False
        cks = sorted(glob.glob(f'{dst}/f0_s1044_*_best.pt'))
        if ok and len(cks) == 2:
            break
        print(f'sim C: the live run saved during the copy (attempt {attempt}); retrying in 20 s', flush=True)
        time.sleep(20)
    else:
        sys.exit('sim C: could not get a stable copy')
    cv = {}
    for p in cks:
        ck = torch.load(p, map_location='cpu', weights_only=False)
        tag = os.path.basename(p).split('_')[2]
        cv[tag] = float(ck['cv'])
        E = int(ck['args']['epochs'])
        print(f'sim C {os.path.basename(p)}: cv {ck["cv"]} epoch {int(ck["epoch"]) + 1} ttr_size args {ck["args"]["ttr_size"]} '
              f'buffer {int(ck["model"]["ttr_size_t"])} seed {ck["args"]["seed"]} fold {ck["args"]["fold"]}', flush=True)
        del ck
    log = open(f'{src}.log', encoding='utf-8', errors='replace').read()
    log += (f'ep {E - 1} raw | SIMULATED by k7_sim_setup.py (the real run is still training)\n'
            f'ep {E - 1} ema | SIMULATED by k7_sim_setup.py\n' + json.dumps({'raw': cv['raw'], 'ema': cv['ema']}) + '\n')
    open(f'{dst}.log', 'w', encoding='utf-8').write(log)
    return cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{H}/k7_sim/root')
    a = ap.parse_args()
    assert os.path.realpath(a.out).startswith(os.path.realpath(f'{H}/k7_sim')), 'the sim root must live under ~/k7_sim'
    os.makedirs(a.out, exist_ok=True)
    ent = {F: sim_b(a.out, F) for F in range(5)}
    for tag, folds in (('f01', [0, 1]), ('f23', [2, 3]), ('f4', [4])):
        print('receipt', receipt(a.out, folds, ent, tag))
    print('sim C cv', sim_c(a.out))
    print('K7SIM SETUP OK', a.out)


if __name__ == '__main__':
    main()
