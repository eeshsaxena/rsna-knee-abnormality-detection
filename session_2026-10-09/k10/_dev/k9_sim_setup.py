#!/usr/bin/env python3
"""k9_sim_setup.py (box, CPU; TEST ONLY, never part of a real build; 2026-10-08, Claude; from k8_sim_setup.py). Builds a
SIMULATION root ~/k9_test/sim/root (ours) so k9_collect.sh / build_k9.py / the A0 can be exercised before the last set C
fold (TTR288 f4, ~09:45 10-08) and set D folds 2-4 (kmrsuman Kaggle, ~09:00 10-08) exist:

  set C  f{F}_s1044_ttr288mix50: a SYMLINK to the real ~/fleet_runs run when its log has the closing line, else SIMULATED
         exactly as k8_sim_setup.py (set A fold-F weights, args seed 1044, ttr_size 288, buffer 288, files renamed).
  set D  f{F}_s1043_ttr_frac75: a SYMLINK to the real ~/fleet_runs run when its log has the closing line (f0, f1 = swapiy
         Kaggle E17 since 10-08 04:19), else SIMULATED: the set A fold-F run f{F}_s1043_ttr_mix50 (both tags) with ckpt args
         train_labels -> labels_frac75_logit.csv (seed 1043, ttr 224 and the weights untouched; file names unchanged, as
         the Kaggle trainer names frac75 files exactly like mix50 ones), OOFs copied, the log = the set A fold log + a
         leading '# SIMULATED' line (the closing line stays last).
  receipts: kaggle_frac75_f01_run_receipt.json and kaggle_f34_run_receipt.json are SYMLINKS to the real ones; the kmrsuman
         receipts kaggle_frac75_f23_run_receipt.json / kaggle_frac75_f4x_run_receipt.json are symlinks when the real ones
         exist, else SIMULATED in the real notebooks' format (nb_src c01_lib.py: cfg=CFG with SEED = {fold: seed} per
         trainer, runs = {fold: run name}, train.folds[F].files = [{path, sha256}]): f23 = folds 2, 3 seed 1043; f4x =
         fold 4 seed 1043 + fold 0 seed 1044 (f0_s1044_ttr_frac75, absent here: its entry only proves that a per-fold seed
         map can never vouch for f0_s1043_ttr_frac75).
Nothing under ~/fleet_runs is written. A real run that is still training / an empty placeholder is never touched.
  ~/miniconda3/envs/kmimic/bin/python k9_sim_setup.py [--out ~/k9_test/sim/root]
"""
import argparse, glob, hashlib, json, os, shutil, time
import torch

H = os.path.expanduser('~')
FR = f'{H}/fleet_runs'
FRAC75 = 'labels_frac75_logit.csv'


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


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


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


def sim_run(out, F, kind):
    """kind C: seed 1044 / ttr 288 (k8_sim_setup.sim_c); kind D: train_labels frac75 (seed 1043, ttr 224)."""
    src = f'{FR}/f{F}_s1043_ttr_mix50'
    name = f'f{F}_s1044_ttr288mix50' if kind == 'C' else f'f{F}_s1043_ttr_frac75'
    dst = f'{out}/{name}'
    clear(out, name); os.makedirs(dst)
    t0 = time.time() - 3600
    files = []
    for k, tag in enumerate(('raw', 'ema')):
        (p,) = glob.glob(f'{src}/f{F}_s1043_{tag}_*_best.pt')
        ck = torch.load(p, map_location='cpu', weights_only=False)
        a = ck['args']
        assert int(a['seed']) == 1043 and int(a['fold']) == F and int(a['ttr_size']) == 224, (p, a.get('seed'), a.get('ttr_size'))
        assert a.get('train_labels') == 'labels_mix50_logit.csv', (p, a.get('train_labels'))
        if kind == 'C':
            ck['args'] = dict(a, seed=1044, ttr_size=288)
            ck['model']['ttr_size_t'] = torch.tensor(288, dtype=torch.int64)
            q = f'{dst}/{os.path.basename(p).replace("_s1043_", "_s1044_").replace("_ttr224-", "_ttr288-")}'
            oq = f'{dst}/oof_f{F}_s1044_{tag}.csv'
        else:
            ck['args'] = dict(a, train_labels=FRAC75)
            q = f'{dst}/{os.path.basename(p)}'
            oq = f'{dst}/oof_f{F}_s1043_{tag}.csv'
        torch.save(ck, q)
        shutil.copyfile(f'{src}/oof_f{F}_s1043_{tag}.csv', oq)
        tc = t0 + 600 * k
        os.utime(q, (tc, tc)); os.utime(oq, (tc + 0.5, tc + 0.5))
        files += [q, oq]
        print(f'  sim {kind} f{F} {tag}: {os.path.basename(q)} cv {ck["cv"]:.6f} epoch {int(ck["epoch"]) + 1} args seed '
              f'{ck["args"]["seed"]} ttr {ck["args"]["ttr_size"]} labels {ck["args"]["train_labels"]}', flush=True)
        del ck
    log = open(f'{src}.log', encoding='utf-8', errors='replace').read()
    open(f'{dst}.log', 'w', encoding='utf-8').write(
        f'# SIMULATED by k9_sim_setup.py: set A f{F} weights relabelled as set {kind} (TEST ONLY)\n' + log)
    return name, files


def sim_receipt(out, tag, trainers, sim_files):
    """A kmrsuman-format run receipt (cfg SEED per trainer) for the simulated folds."""
    folds, seeds, runs = {}, {}, {}
    for F, s in trainers:
        run = f'f{F}_s{s}_ttr_frac75'
        seeds[str(F)] = s; runs[str(F)] = run
        ents = [dict(path=f'{run}/{os.path.basename(p)}', sha256=sha(p)) for p in sim_files.get(F, [])]
        if not ents:                                                     # f0_s1044 (not simulated): a decoy entry
            ents = [dict(path=f'{run}/f{F}_s{s}_raw_ttr224-320x24_cv0000_best.pt', sha256='0' * 64)]
        folds[str(F)] = dict(run=run, complete=True, files=ents)
    rec = dict(kernel=f'rsna-train-ttr-frac75-{tag}', status='DONE', simulated=True,
               cfg=dict(FOLDS=[F for F, _ in trainers], SEED=seeds, LABELS_TRAIN=FRAC75), runs=runs,
               train=dict(folds=folds))
    p = f'{out}/kaggle_frac75_{tag}_run_receipt.json'
    if os.path.lexists(p):
        os.remove(p)
    json.dump(rec, open(p, 'w'), indent=1)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{H}/k9_test/sim/root')
    a = ap.parse_args()
    assert os.path.realpath(a.out).startswith(os.path.realpath(f'{H}/k9_test') + '/'), 'the sim root must live under ~/k9_test'
    os.makedirs(a.out, exist_ok=True)
    link(f'{FR}/kaggle_f34_run_receipt.json', f'{a.out}/kaggle_f34_run_receipt.json')
    link(f'{FR}/kaggle_frac75_f01_run_receipt.json', f'{a.out}/kaggle_frac75_f01_run_receipt.json')
    kinds, sim_files = {}, {}
    for kind, fmt in (('C', 'f{F}_s1044_ttr288mix50'), ('D', 'f{F}_s1043_ttr_frac75')):
        for F in range(5):
            r = fmt.format(F=F)
            if finished(f'{FR}/{r}.log'):
                clear(a.out, r)
                link(f'{FR}/{r}', f'{a.out}/{r}'); link(f'{FR}/{r}.log', f'{a.out}/{r}.log')
                kinds[f'{kind}{F}'] = 'REAL (symlink)'
            else:
                _, fs = sim_run(a.out, F, kind)
                if kind == 'D':
                    sim_files[F] = fs
                kinds[f'{kind}{F}'] = 'SIMULATED (set A weights)'
    for tag, trainers in (('f23', [(2, 1043), (3, 1043)]), ('f4x', [(4, 1043), (0, 1044)])):
        real = f'{FR}/kaggle_frac75_{tag}_run_receipt.json'
        d_folds = [F for F, s in trainers if s == 1043]
        if os.path.isfile(real) and not any(F in sim_files for F in d_folds):
            link(real, f'{a.out}/kaggle_frac75_{tag}_run_receipt.json'); kinds[f'receipt {tag}'] = 'REAL (symlink)'
        else:
            assert all(F in sim_files for F in d_folds), (tag, 'mixed real / simulated folds in one receipt', sorted(sim_files))
            sim_receipt(a.out, tag, trainers, sim_files); kinds[f'receipt {tag}'] = 'SIMULATED (kmrsuman format, SEED per fold)'
    for k, v in kinds.items():
        print(f'{k}: {v}')
    print('K9SIM SETUP OK', a.out, json.dumps(kinds))


if __name__ == '__main__':
    main()
