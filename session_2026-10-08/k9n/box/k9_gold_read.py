#!/usr/bin/env python3
"""k9_gold_read.py (box, CPU only; 2026-10-08, Claude; from k8_gold_read.py). The PRE-PUSH expert-label read for K9:
the K9 fleet (set C 5 x TTR288 mix50 s1044 + set D 5 x TTR224 frac75 s1043, equal member mean) vs the K8 fleet (set C +
set A 5 x TTR224 mix50 s1043, the K5 members) on gold58 and gold74. gold74 is --exclude'd from every fleet run (box and
Kaggle trainers), so every fold model is out-of-sample there. Set C is shared, so K9 - K8 isolates the D-for-A swap as
the fleet serves it. The same run also prints the K8 read (C - A, k8_gold_read.py's pre-registered quantity) so one run
answers both.

PRE-REGISTERED (written 10-08 ~05:00 IST, after only D f0 / f1 (gate probe + swapiy) have gold reads; D f2-f4 unseen):
  primary   = pass A (offset 0, no mirror), masked macro AUC (tta_bench.macro_auc), gold58 (original_gold58 rows):
              d58 = AUC58(K9 fleet) - AUC58(K8 fleet), paired study bootstrap (B 2000, seed 59) for the SE.
  changed label = Fracture (frac75 re-mixes only the Fracture target): f58 = Fracture AUC58(K9) - AUC58(K8), reported.
  VERDICT   "K9 TIE RULE ON" when d58 > -.003 AND f58 >= 0 (NOVEL_PLAN sec 5: the tie rule may favour the newest fleet only
            with "no negative gold58 sign on the changed labels"); else "K9 TIE RULE OFF" (pick 1 = the best public read,
            no K9 preference). It never blocks a push (the slot plan does not depend on it).
  secondary = A+B served read (--passes A,B), reported only (a disagreement prints BORDERLINE); gold74 reported, not
              gated; D - A per label reported. gold58 resolves only ~.008 on macro (SE ~.003-.005 for these
              differences): a guard against a clearly worse set on expert labels, not a gain test.
  Valid only at C 5/5 and D 5/5 (every fold's log has its closing line); fewer = PARTIAL / INFO ONLY.
Members = the RAW best checkpoint of each FINISHED run; with --members (default ~/k9_test/members.json when present) the
C and D files must equal that collect's picks (sha256), so the read is about exactly what K9 ships. Set A = the K5 runs.
Predictions: ~/novel/gold_infer5.py (CPU, fp32) writes ~/novel/k9gold/preds/<run>__raw__<pass>.csv; existing files in
~/novel/k8gold/preds and ~/novel/gold74_preds are reused READ-ONLY (only when newer than the checkpoint). Nothing outside
~/novel/k9gold is written; never the GPU (CUDA hidden).

  CUDA_VISIBLE_DEVICES= NTHR=8 nice -n 19 ionice -c3 ~/miniconda3/envs/kmimic/bin/python ~/novel/k9gold/k9_gold_read.py \\
      [--infer] [--passes A|A,B] [--members ~/k9_test/members.json|none]
Output: stdout + ~/novel/k9gold/k9_gold_read_<YYYYmmdd_HHMM>.txt; last line 'K9GOLD <FINAL|PARTIAL> ... K9 TIE RULE <ON|OFF>'.
"""
import argparse, glob, hashlib, json, os, subprocess, sys, time
import numpy as np, pandas as pd

H = os.path.expanduser('~')
OUT = f'{H}/novel/k9gold'
PRED = f'{OUT}/preds'
READ_ONLY = [f'{H}/novel/k8gold/preds', f'{H}/novel/gold74_preds']
FR = f'{H}/fleet_runs'
sys.path.insert(0, f'{H}/novel/code')
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
BAR = -0.003
RUNS = {'A': 'f{F}_s1043_ttr_mix50', 'C': 'f{F}_s1044_ttr288mix50', 'D': 'f{F}_s1043_ttr_frac75'}


def finished(log):
    if not os.path.isfile(log):
        return False
    lines = [l.strip() for l in open(log, encoding='utf-8', errors='replace') if l.strip()]
    try:
        d = json.loads(lines[-1]) if lines and lines[-1].startswith('{') else None
    except Exception:
        d = None
    return isinstance(d, dict) and {'raw', 'ema'} <= set(d)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def members(mjson):
    out = {s: {} for s in RUNS}
    for s, fmt in RUNS.items():
        for F in range(5):
            run = fmt.format(F=F)
            if not finished(f'{FR}/{run}.log'):
                continue
            seed = run.split('_')[1]
            fs = sorted(glob.glob(f'{FR}/{run}/f{F}_{seed}_raw_*_best.pt'))
            assert len(fs) == 1, (run, fs)
            out[s][F] = fs[0]
    if mjson:
        m = json.load(open(mjson))
        pk = {(r['set'], r['fold']): r for r in m['members']}
        for s in ('C', 'D'):
            for F, p in out[s].items():
                r = pk.get((s, F))
                assert r is not None, f'set {s} fold {F} is finished but not in {mjson} (re-run k9_collect.sh)'
                assert os.path.basename(p) == r['file'] and sha(p) == r['sha256'], \
                    f'{p} != the collected pick {r["file"]} (sha256): re-run k9_collect.sh'
        miss = [k for k in pk if k[0] in ('C', 'D') and k[1] not in out[k[0]]]
        assert not miss, f'{mjson} picks {miss} have no finished run here'
    return out


def pred_path(ckpt, ps):
    run = os.path.basename(os.path.dirname(ckpt))
    for d in [PRED] + READ_ONLY:
        q = f'{d}/{run}__raw__{ps}.csv'
        if os.path.isfile(q) and os.path.getmtime(q) > os.path.getmtime(ckpt):
            return q
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--passes', default='A')
    ap.add_argument('--infer', action='store_true')
    ap.add_argument('--members', default=f'{H}/k9_test/members.json')
    ap.add_argument('--boot', type=int, default=2000)
    a = ap.parse_args()
    passes = [p for p in a.passes.split(',') if p]
    assert passes[0] == 'A' and set(passes) <= {'A', 'B'}, passes
    mj = a.members if a.members != 'none' and os.path.isfile(a.members) else None
    M = members(mj)
    os.makedirs(PRED, exist_ok=True)
    lines = []

    def say(s):
        print(s, flush=True); lines.append(s)
    say(f'k9_gold_read {time.strftime("%Y-%m-%d %H:%M:%S")} | members json {mj} | folds A {sorted(M["A"])} C {sorted(M["C"])} '
        f'D {sorted(M["D"])} | passes {passes}')
    todo = sorted({p for s in M for p in M[s].values() for ps in passes if pred_path(p, ps) is None})
    if todo and a.infer:
        for p in todo:                                                  # stale files in OUR dir only
            run = os.path.basename(os.path.dirname(p))
            for ps in passes:
                q = f'{PRED}/{run}__raw__{ps}.csv'
                if os.path.isfile(q) and os.path.getmtime(q) <= os.path.getmtime(p):
                    os.remove(q)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        cmd = [sys.executable, '-W', 'ignore', f'{H}/novel/gold_infer5.py', '--passes', ','.join(passes), '--out', PRED] + todo
        say(f'infer: {len(todo)} checkpoint(s) -> gold_infer5.py ({" ".join(os.path.basename(os.path.dirname(x)) for x in todo)})')
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        for l in r.stdout.splitlines():
            if 'gold58' in l or 'decoded' in l or 'done' in l:
                say('  ' + l)
        assert r.returncode == 0, r.stderr[-2000:]
    elif todo:
        say(f'MISSING predictions for {len(todo)} checkpoint(s) (run with --infer): '
            f'{[os.path.basename(os.path.dirname(x)) for x in todo]}')
    gold = pd.read_csv(f'{H}/rsna_train/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    import tta_bench as tb
    jF = T.index('Fracture')
    res = {}
    for mode in (['A'] + (['AB'] if 'B' in passes else [])):
        P = {}
        for s in M:
            for F, p in M[s].items():
                qa = pred_path(p, 'A')
                qb = pred_path(p, 'B') if mode == 'AB' else None
                if qa is None or (mode == 'AB' and qb is None):
                    continue
                x = pd.read_csv(qa, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[T]
                if mode == 'AB':
                    x = (x + pd.read_csv(qb, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[T]) / 2
                P[(s, F)] = x
        if not P:
            continue
        ids = sorted(set.intersection(*[set(x.index) for x in P.values()]) & set(gold.index))
        g58 = [i for i, u in enumerate(ids) if gold.loc[u, 'source'] == 'original_gold58']
        Y = gold.loc[ids, T].to_numpy(np.float64)
        sets = {s: sorted(k for k in P if k[0] == s) for s in RUNS}
        arr = {k: P[k].loc[ids].to_numpy(np.float64) for k in P}
        mean = {s: np.mean([arr[k] for k in ks], axis=0) for s, ks in sets.items() if ks}
        if sets['A'] and sets['C']:
            mean['K8'] = np.mean([arr[k] for k in sets['A'] + sets['C']], axis=0)
        if sets['C'] and sets['D']:
            mean['K9'] = np.mean([arr[k] for k in sets['C'] + sets['D']], axis=0)
        dfo = sorted(k[1] for k in sets['D'])
        if sets['D'] and len(dfo) < 5:                                  # PARTIAL: fold-matched info (never a decision)
            mean[f'A_folds{dfo}'] = np.mean([arr[('A', F)] for F in dfo if ('A', F) in arr], axis=0)
            mean[f'D_folds{dfo}'] = mean['D']
        rng = np.random.default_rng(59)
        out = {}
        for sub, idx in (('gold58', g58), ('gold74', list(range(len(ids))))):
            sc = {s: tb.macro_auc(m[idx], Y[idx])[0] for s, m in mean.items()}
            o = dict(n=len(idx), auc=sc)
            ii = np.asarray(idx)
            B = [rng.integers(0, len(idx), len(idx)) for _ in range(a.boot)]

            def boot(x, y):
                return np.array([tb.macro_auc(mean[x][ii[b]], Y[ii[b]])[0] - tb.macro_auc(mean[y][ii[b]], Y[ii[b]])[0] for b in B])

            def fr(m_):
                yy = Y[idx, jF]; okm = np.isfinite(yy)
                return float(tb._auc(yy[okm] >= .5, m_[idx][okm, jF]))
            for x, y in (('K9', 'K8'), ('C', 'A'), ('D', 'A')):
                if x in mean and y in mean:
                    dd = boot(x, y)
                    o[f'd_{x}_{y}'] = sc[x] - sc[y]; o[f'se_{x}_{y}'] = float(np.nanstd(dd))
                    o[f'p_{x}_{y}_gt0'] = float(np.nanmean(dd > 0))
                    o[f'fracture_{x}_{y}'] = fr(mean[x]) - fr(mean[y])
            if 'D' in mean and 'A' in mean:
                pl = {}
                for j, t in enumerate(T):
                    yy = Y[idx, j]; okm = np.isfinite(yy)
                    if okm.sum() and 0 < (yy[okm] >= .5).mean() < 1:
                        pl[t] = round(float(tb._auc(yy[okm] >= .5, mean['D'][idx][okm, j]) - tb._auc(yy[okm] >= .5, mean['A'][idx][okm, j])), 4)
                o['per_label_D_minus_A'] = pl
            out[sub] = o
        res[mode] = dict(sets={s: [k[1] for k in ks] for s, ks in sets.items()}, n_ids=len(ids), **out)
        say(f'== pass {mode}: folds A {res[mode]["sets"]["A"]} C {res[mode]["sets"]["C"]} D {res[mode]["sets"]["D"]}, '
            f'{len(ids)} gold studies')
        for sub in ('gold58', 'gold74'):
            o = out[sub]
            s = ' '.join(f'{k} {v:.4f}' for k, v in o['auc'].items())
            for x, y in (('K9', 'K8'), ('C', 'A'), ('D', 'A')):
                if f'd_{x}_{y}' in o:
                    s += (f' | {x} - {y} {o[f"d_{x}_{y}"]:+.4f} (SE {o[f"se_{x}_{y}"]:.4f}, P>0 {o[f"p_{x}_{y}_gt0"]:.2f}; '
                          f'Fracture {o[f"fracture_{x}_{y}"]:+.4f})')
            say(f'  {sub} n {o["n"]}: {s}')
            if 'per_label_D_minus_A' in o:
                say(f'    per label D - A: {o["per_label_D_minus_A"]}')
    nC, nD = len(M['C']), len(M['D'])
    prim = res.get('A', {}).get('gold58', {})
    final = nC == 5 and nD == 5 and len(M['A']) == 5 and res.get('A', {}).get('sets', {}).get('C', []) == [0, 1, 2, 3, 4] \
        and res.get('A', {}).get('sets', {}).get('D', []) == [0, 1, 2, 3, 4]
    if 'd_K9_K8' not in prim:
        verdict, tag = 'NO READ (missing predictions or sets)', 'NOREAD'
    else:
        d, f58 = prim['d_K9_K8'], prim['fracture_K9_K8']
        on = d > BAR and f58 >= 0
        verdict = (f'K9 TIE RULE {"ON" if on else "OFF"}: gold58 pass A K9 - K8 {d:+.4f} (SE {prim["se_K9_K8"]:.4f}) '
                   f'{">" if d > BAR else "<="} {BAR}; Fracture K9 - K8 {f58:+.4f} ({">= 0" if f58 >= 0 else "< 0"})')
        if 'd_C_A' in prim:
            verdict += (f' | K8 read: C - A {prim["d_C_A"]:+.4f} (SE {prim["se_C_A"]:.4f}) -> K8 TIE RULE '
                        f'{"ON" if prim["d_C_A"] > BAR else "OFF"}')
        if 'AB' in res and 'd_K9_K8' in res['AB'].get('gold58', {}):
            dab = res['AB']['gold58']['d_K9_K8']
            verdict += (f' | BORDERLINE: the A+B served read says {dab:+.4f} (the primary pass A read decides)'
                        if (dab > BAR) != (d > BAR) else f' | A+B served read agrees ({dab:+.4f})')
        tag = 'FINAL' if final else f'PARTIAL (C {nC}/5, D {nD}/5: NOT the decision read)'
        if not final:
            verdict = 'INFO ONLY, no verdict before C 5/5 + D 5/5: ' + verdict
    say(f'K9GOLD {tag} | {verdict}')
    os.makedirs(OUT, exist_ok=True)
    p = f'{OUT}/k9_gold_read_{time.strftime("%Y%m%d_%H%M")}.txt'
    open(p, 'w').write('\n'.join(lines) + '\n' + json.dumps(res, indent=1, default=str) + '\n')
    print(f'wrote {p}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
