#!/usr/bin/env python3
"""k8_gold_read.py (box, CPU only; 2026-10-07, Claude). The PRE-PUSH expert-label read that NOVEL_PLAN_10-07 sec 4 requires
before K8 is pushed: the full 5 x TTR288 set (set C, f{0..4}_s1044_ttr288mix50) vs set A (the K5 members
f{0..4}_s1043_ttr_mix50) on gold58 and gold74. gold74 is --exclude'd from every fleet run, so every fold model is
out-of-sample there and a set = the equal mean of its 5 fold members' probabilities (as served).

PRE-REGISTERED (written 10-07 ~13:00 IST, before any 288 fold beyond f0 has a gold read):
  primary read   = pass A (offset 0, no mirror): the convention of the verifier's deciding read (verify_ensemble V4,
                   ~/novel/gold74_preds/*__raw__A.csv); identical for both sets, so the difference is apples to apples.
  secondary      = A+B served ((pass A + pass B) / 2 per member, what K8 ships); reported when --passes A,B ran. It does
                   NOT override the primary verdict; a disagreement is printed as BORDERLINE for the record.
  metric         = macro AUC, masked convention (tta_bench.macro_auc == fleet_train.macro_auc: per label, finite label
                   AND prediction, label >= .5, both classes present), on gold58 (original_gold58 rows) and gold74.
  d58 = AUC58(set C) - AUC58(set A) (5/5 folds each), paired index bootstrap over studies (B 2000, seed 58) for SE.
  BAR (the plan): d58 <= -.003  ->  "TIE RULE OFF": the sec 5 pick-1 tie rule stops favouring K8 (pick 1 = the best
                  public read among the candidates, no K8 preference).  d58 > -.003 -> "TIE RULE ON" (K8 may be pick 1
                  when it reads >= best other - .001). gold58 resolves only ~.008 (SE ~.003-.005 here): this is a guard
                  against a clearly worse 288 set on expert labels, not a gain test. gold74 is reported, not gated.
  Valid only with set C 5/5 (every C fold has its closing line); fewer = PARTIAL (printed, never a decision).

Members = the K8 picks: the RAW best checkpoint of each FINISHED run (log ends with the closing {"raw", "ema"} line).
With --members (default ~/k8_test/members.json when present) the files must equal that collect's picks (sha256), so the
read is about exactly what ships. Predictions: gold_infer5.py (~/novel, unchanged, CPU, fp32) writes
~/novel/k8gold/preds/<run>__raw__<pass>.csv. A prediction is reused only when its mtime is newer than the checkpoint's
(a best file is rewritten while a run trains); existing files in ~/novel/gold74_preds are reused READ-ONLY under the same
rule. Nothing outside ~/novel/k8gold is written; never the GPU (CUDA hidden), nice 19 recommended.

  CUDA_VISIBLE_DEVICES= NTHR=8 nice -n 19 ionice -c3 ~/miniconda3/envs/kmimic/bin/python ~/novel/k8gold/k8_gold_read.py \
      [--infer] [--passes A|A,B] [--members ~/k8_test/members.json|none]
  (--infer computes the missing predictions first: ~10 min per 224 member pass, ~15-20 min per 288 member pass on the
   loaded box; run it after each TTR288 fold lands so only fold 4 remains on 10-08.)
Output: stdout + ~/novel/k8gold/k8_gold_read_<YYYYmmdd_HHMM>.txt; last line 'K8GOLD <FINAL|PARTIAL> ... TIE RULE <ON|OFF>'.
"""
import argparse, glob, hashlib, json, os, subprocess, sys, time
import numpy as np, pandas as pd

H = os.path.expanduser('~')
OUT = f'{H}/novel/k8gold'
PRED = f'{OUT}/preds'
SHARED = f'{H}/novel/gold74_preds'
FR = f'{H}/fleet_runs'
sys.path.insert(0, f'{H}/novel/code')
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
BAR = -0.003


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
    out = {'A': {}, 'C': {}}
    for s, run_fmt in (('A', 'f{F}_s1043_ttr_mix50'), ('C', 'f{F}_s1044_ttr288mix50')):
        for F in range(5):
            run = run_fmt.format(F=F)
            if not finished(f'{FR}/{run}.log'):
                continue
            fs = sorted(glob.glob(f'{FR}/{run}/f{F}_s{run[4:8]}_raw_*_best.pt'))
            assert len(fs) == 1, (run, fs)
            out[s][F] = fs[0]
    if mjson:
        m = json.load(open(mjson))
        pk = {(r['set'], r['fold']): r for r in m['members']}
        for s in out:
            for F, p in out[s].items():
                r = pk.get((s, F))
                assert r is not None, f'set {s} fold {F} is finished but not in {mjson} (re-run k8_collect.sh)'
                assert os.path.basename(p) == r['file'] and sha(p) == r['sha256'], \
                    f'{p} != the collected pick {r["file"]} (sha256): re-run k8_collect.sh'
        miss = [k for k in pk if k[0] in out and k[1] not in out[k[0]]]
        assert not miss, f'{mjson} picks {miss} have no finished run here'
    return out


def pred_path(ckpt, ps):
    run = os.path.basename(os.path.dirname(ckpt))
    for d in (PRED, SHARED):
        q = f'{d}/{run}__raw__{ps}.csv'
        if os.path.isfile(q) and os.path.getmtime(q) > os.path.getmtime(ckpt):
            return q
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--passes', default='A')
    ap.add_argument('--infer', action='store_true')
    ap.add_argument('--members', default=f'{H}/k8_test/members.json')
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
    say(f'k8_gold_read {time.strftime("%Y-%m-%d %H:%M:%S")} | members json {mj} | set A folds {sorted(M["A"])} | '
        f'set C folds {sorted(M["C"])} | passes {passes}')
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
        sets = {s: [k for k in P if k[0] == s] for s in 'AC'}
        mean = {s: np.mean([P[k].loc[ids].to_numpy(np.float64) for k in ks], axis=0) for s, ks in sets.items() if ks}
        if sets['A'] and sets['C']:
            mean['K8'] = np.mean([P[k].loc[ids].to_numpy(np.float64) for k in sets['A'] + sets['C']], axis=0)
            cf = sorted(k[1] for k in sets['C'])
            if len(cf) < 5:                                             # PARTIAL: fold-matched info (never a decision)
                mean[f'A_folds{cf}'] = np.mean([P[('A', F)].loc[ids].to_numpy(np.float64) for F in cf if ('A', F) in P], axis=0)
        rng = np.random.default_rng(58)
        out = {}
        for sub, idx in (('gold58', g58), ('gold74', list(range(len(ids))))):
            sc = {s: tb.macro_auc(m[idx], Y[idx])[0] for s, m in mean.items()}
            o = dict(n=len(idx), auc=sc)
            if 'A' in mean and 'C' in mean:
                B = [rng.integers(0, len(idx), len(idx)) for _ in range(a.boot)]
                ii = np.asarray(idx)
                dd = np.array([tb.macro_auc(mean['C'][ii[b]], Y[ii[b]])[0] - tb.macro_auc(mean['A'][ii[b]], Y[ii[b]])[0]
                               for b in B])
                dk = np.array([tb.macro_auc(mean['K8'][ii[b]], Y[ii[b]])[0] - tb.macro_auc(mean['A'][ii[b]], Y[ii[b]])[0]
                               for b in B])
                o.update(d_C_A=sc['C'] - sc['A'], se_C_A=float(np.nanstd(dd)), p_gt0=float(np.nanmean(dd > 0)),
                         d_K8_A=sc['K8'] - sc['A'], se_K8_A=float(np.nanstd(dk)))
                pl = {}
                for j, t in enumerate(T):
                    yy = Y[idx, j]; ok = np.isfinite(yy)
                    if ok.sum() and 0 < (yy[ok] >= .5).mean() < 1:
                        pl[t] = round(float(tb._auc(yy[ok] >= .5, mean['C'][idx][ok, j]) - tb._auc(yy[ok] >= .5, mean['A'][idx][ok, j])), 4)
                o['per_label_C_minus_A'] = pl
            out[sub] = o
        res[mode] = dict(sets={s: [k[1] for k in ks] for s, ks in sets.items()}, n_ids=len(ids), **out)
        say(f'== pass {mode}: set A folds {res[mode]["sets"]["A"]}, set C folds {res[mode]["sets"]["C"]}, {len(ids)} gold studies')
        for sub in ('gold58', 'gold74'):
            o = out[sub]
            s = ' '.join(f'{k} {v:.4f}' for k, v in o['auc'].items())
            if 'd_C_A' in o:
                s += (f' | C - A {o["d_C_A"]:+.4f} (SE {o["se_C_A"]:.4f}, P>0 {o["p_gt0"]:.2f}) | K8 - A {o["d_K8_A"]:+.4f} '
                      f'(SE {o["se_K8_A"]:.4f})')
            say(f'  {sub} n {o["n"]}: {s}')
            if 'per_label_C_minus_A' in o:
                say(f'    per label C - A: {o["per_label_C_minus_A"]}')
    nC = len(M['C'])
    prim = res.get('A', {}).get('gold58', {})
    final = nC == 5 and len(M['A']) == 5 and len(res.get('A', {}).get('sets', {}).get('C', [])) == 5
    if 'd_C_A' not in prim:
        verdict = 'NO READ (missing predictions)'
        tag = 'NOREAD'
    else:
        d = prim['d_C_A']
        on = d > BAR
        verdict = (f'TIE RULE {"ON" if on else "OFF"}: gold58 pass A C - A {d:+.4f} (SE {prim["se_C_A"]:.4f}) '
                   f'{">" if on else "<="} {BAR}')
        if 'AB' in res and 'd_C_A' in res['AB'].get('gold58', {}):
            dab = res['AB']['gold58']['d_C_A']
            if (dab > BAR) != on:
                verdict += f' | BORDERLINE: the A+B served read says {dab:+.4f} (the primary pass A read decides)'
            else:
                verdict += f' | A+B served read agrees ({dab:+.4f})'
        tag = 'FINAL' if final else f'PARTIAL (set C {nC}/5: NOT the decision read)'
        if not final:                                                   # a k-fold C mean vs the 5-fold A mean is unfair
            verdict = 'INFO ONLY, no verdict before C 5/5 (see the A_folds line for the fold-matched view): ' + verdict
    say(f'K8GOLD {tag} | {verdict}')
    os.makedirs(OUT, exist_ok=True)
    p = f'{OUT}/k8_gold_read_{time.strftime("%Y%m%d_%H%M")}.txt'
    open(p, 'w').write('\n'.join(lines) + '\n' + json.dumps(res, indent=1, default=str) + '\n')
    print(f'wrote {p}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
