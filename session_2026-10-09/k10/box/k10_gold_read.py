#!/usr/bin/env python3
"""k10_gold_read.py (box, CPU only; 2026-10-09, Claude; from k9_gold_read.py). The PRE-PUSH expert-label read for K10:
set E (5 x TTR288 frac75 s1045) vs set C (5 x TTR288 mix50 s1044), and the K10 fleet (E + D) vs the K9 fleet (C + D), on
gold58 and gold74 (gold74 is --exclude'd from every fleet run, so every fold model is out-of-sample there). D is shared,
so K10 - K9 isolates the E-for-C swap as the fleet serves it (equal-member raw-probability mean, as the worker).

PRE-REGISTERED (written 10-09 ~03:45 IST, BEFORE any E checkpoint was picked or inferred. Disclosure: the writer had seen
the E f0 trainer log up to epoch 5 of 21 (val / trainer-gold58 per epoch); no E gold prediction, no E fold-1..4 output and
no final E checkpoint existed. The rule is the K9 rule with K10 / K9 in place of K9 / K8, unchanged in every number.)
  primary   = pass A (offset 0, no mirror), masked macro AUC (tta_bench.macro_auc), gold58 (original_gold58 rows):
              d58 = AUC58(K10 fleet = E + D) - AUC58(K9 fleet = C + D), paired study bootstrap (B 2000, seed 59) for the SE.
  changed label = Fracture (frac75 re-mixes only the Fracture target; the seed changes every label but carries no
              direction): f58 = Fracture AUC58(K10) - AUC58(K9), reported.
  VERDICT   "K10 TIE RULE ON" when d58 > -.003 AND f58 >= 0; else "K10 TIE RULE OFF".
            ON : a K10-family kernel (K10, K10N-a, FN10) may be preferred over its K9-family twin when their public reads are
                 within .001 (final-pick tie rule, research sec 5: newest fleet whose gates passed).
            OFF: no K10 preference: pick by the public read alone. The read never blocks a push (slots are not scarce).
  composition (parent-free vehicle, research L2 / verify_E M10; reported in the same run, decided by the same bar):
            w58 = AUC58(K10W = C + D + E, 15 members) - max(AUC58(K9), AUC58(K10)).
            "FN10-3 PREFERRED" when w58 > -.003 (a third same-family set is a +.0001 to +.0005 MODEL gain that gold58 cannot
            resolve, so only a clear gold58 loss rejects it); else the parent-free vehicle is FN10 when the K10 tie rule is
            ON, else the K9 FN kernel.
  secondary (reported, never gating): gold74; per-label E - C; the A+B served read (--passes A,B: a disagreement prints
            BORDERLINE); E~C, D~C, D~E decorrelation = mean per-label Spearman of fold-matched predictions on gold74 and on
            the members' OOF (same val studies per fold); OOF macro AUC vs the selection labels of (C+D)/2, (E+D)/2,
            (C+D+E)/3 (selection-biased: each checkpoint was picked on these labels; info only).
  gold58 resolves only ~.008 on macro (SE ~.003-.005 for these differences): a guard against a clearly worse set on
  expert labels, not a gain test. Valid only at E 5/5, C 5/5, D 5/5; fewer = PARTIAL / INFO ONLY (no verdict).
Members = the RAW best checkpoint of each FINISHED run under --root; with --members (default ~/k10_test/members.json when
present) the E / D / C files must equal that collect's picks (sha256), so the read is about exactly what K10 ships.
Predictions: ~/novel/gold_infer5.py (CPU fp32) into --pred_dir (default ~/k10_test/gold/preds) + a <csv>.sha256 sidecar
(the checkpoint's sha256); a prediction in --pred_dir is used only when its sidecar == the checkpoint's sha256 (so a
simulation prediction can never stand in for a real E fold). C / D / A predictions in ~/novel/{k9gold,k8gold}/preds and
~/novel/gold74_preds are reused READ-ONLY when newer than the checkpoint (those runs are frozen). Writes only --out.

  CUDA_VISIBLE_DEVICES= NTHR=6 taskset -c 34-39 nice -n 19 ionice -c3 ~/miniconda3/envs/kmimic/bin/python \\
      ~/k10_test/k10/box/k10_gold_read.py [--infer] [--passes A|A,B] [--members ~/k10_test/members.json|none] \\
      [--root ~/fleet_runs] [--pred_dir ~/k10_test/gold/preds] [--out ~/k10_test/gold]
Last line: 'K10GOLD <FINAL|PARTIAL> | K10 TIE RULE <ON|OFF> ... | COMPOSITION ...'.
"""
import argparse, glob, hashlib, json, os, subprocess, sys, time
import numpy as np, pandas as pd

H = os.path.expanduser('~')
READ_ONLY = [f'{H}/novel/k9gold/preds', f'{H}/novel/k8gold/preds', f'{H}/novel/gold74_preds']
sys.path.insert(0, f'{H}/novel/code')
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
BAR = -0.003
RUNS = {'A': 'f{F}_s1043_ttr_mix50', 'C': 'f{F}_s1044_ttr288mix50', 'D': 'f{F}_s1043_ttr_frac75', 'E': 'f{F}_s1045_ttr288frac75'}
FLEETS = {'K9': ('C', 'D'), 'K10': ('E', 'D'), 'K10W': ('C', 'D', 'E')}


def finished(log):
    lines = [l.strip() for l in open(log, encoding='utf-8', errors='replace') if l.strip()] if os.path.isfile(log) else []
    try:
        d = json.loads(lines[-1]) if lines and lines[-1].startswith('{') else None
    except Exception:
        d = None
    return isinstance(d, dict) and {'raw', 'ema'} <= set(d)


_SHA = {}


def sha(p):
    rp = os.path.realpath(p)
    if rp not in _SHA:
        h = hashlib.sha256()
        with open(rp, 'rb') as f:
            for b in iter(lambda: f.read(1 << 24), b''):
                h.update(b)
        _SHA[rp] = h.hexdigest()
    return _SHA[rp]


def members(root, mjson, sets_root):
    out = {s: {} for s in RUNS}
    for s, fmt in RUNS.items():
        r0 = root if s in sets_root else f'{H}/fleet_runs'               # set A always from ~/fleet_runs (K5 members)
        for F in range(5):
            run = fmt.format(F=F)
            if not finished(f'{r0}/{run}.log'):
                continue
            seed = run.split('_')[1]
            fs = sorted(glob.glob(f'{r0}/{run}/f{F}_{seed}_raw_*_best.pt'))
            assert len(fs) == 1, (run, fs)
            out[s][F] = fs[0]
    if mjson:
        m = json.load(open(mjson))
        pk = {(r['set'], r['fold']): r for r in m['members']}
        for s in ('C', 'D', 'E'):
            for F, p in out[s].items():
                r = pk.get((s, F))
                assert r is not None, f'set {s} fold {F} is finished but not in {mjson} (re-run k10_collect.sh)'
                assert os.path.basename(p) == r['file'] and sha(p) == r['sha256'], f'{p} != the collected pick {r["file"]}'
        miss = [k for k in pk if k[1] not in out[k[0]]]
        assert not miss, f'{mjson} picks {miss} have no finished run under {root}'
    return out


def pred_path(ckpt, ps, pred_dir, set_name):
    run = os.path.basename(os.path.dirname(ckpt))
    q = f'{pred_dir}/{run}__raw__{ps}.csv'
    if os.path.isfile(q) and os.path.isfile(q + '.sha256') and open(q + '.sha256').read().strip() == sha(ckpt):
        return q
    if set_name in ('A', 'C', 'D'):                                      # frozen runs: the older read-only caches
        for d in READ_ONLY:
            q = f'{d}/{run}__raw__{ps}.csv'
            if os.path.isfile(q) and os.path.getmtime(q) > os.path.getmtime(ckpt):
                return q
    return None


def spearman_mean(x, y):
    from scipy.stats import spearmanr
    v = [spearmanr(x[:, j], y[:, j]).correlation for j in range(x.shape[1])]
    v = [c for c in v if np.isfinite(c)]
    return float(np.mean(v)) if v else float('nan')


def oof_read(M, mj, sel_csv):
    """Fold-matched OOF of each set (each val study scored by its own fold's member); E~C etc. decorrelation + OOF AUC."""
    if not mj:
        return {}
    m = json.load(open(mj))
    oof = {}
    for r in m['members']:
        if r['set'] in ('C', 'D', 'E') and r.get('oof') and os.path.isfile(r['oof']):
            oof.setdefault(r['set'], []).append(pd.read_csv(r['oof'], dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[T])
    if not all(len(oof.get(s, [])) == 5 for s in ('C', 'D', 'E')):
        return dict(note='OOF read needs the 5 OOF copies of C, D and E')
    cat = {s: pd.concat(v) for s, v in oof.items()}
    ids = sorted(set.intersection(*[set(x.index) for x in cat.values()]))
    X = {s: cat[s].loc[ids].to_numpy(np.float64) for s in cat}
    sel = pd.read_csv(sel_csv, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    Y = sel.loc[ids, T].to_numpy(np.float64)
    import tta_bench as tb
    res = dict(n=len(ids), rho={f'{a}~{b}': spearman_mean(X[a], X[b]) for a, b in (('E', 'C'), ('D', 'C'), ('D', 'E'))})
    res['auc'] = {k: float(tb.macro_auc(np.mean([X[s] for s in v], axis=0), Y)[0]) for k, v in FLEETS.items()}
    res['auc'].update({s: float(tb.macro_auc(X[s], Y)[0]) for s in X})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--passes', default='A')
    ap.add_argument('--infer', action='store_true')
    ap.add_argument('--members', default=f'{H}/k10_test/members.json')
    ap.add_argument('--root', default=f'{H}/fleet_runs', help='run root of sets C / D / E (simulation: ~/k10_test/sim/root)')
    ap.add_argument('--pred_dir', default=f'{H}/k10_test/gold/preds')
    ap.add_argument('--out', default=f'{H}/k10_test/gold')
    ap.add_argument('--sel_csv', default=f'{H}/rsna_train/llm_labels_v4_blend.csv')
    ap.add_argument('--boot', type=int, default=2000)
    a = ap.parse_args()
    for d in (a.out, a.pred_dir):
        assert os.path.realpath(os.path.expanduser(d)).startswith(os.path.realpath(f'{H}/k10_test') + '/'), \
            f'{d}: k10_gold_read writes only under ~/k10_test'
    passes = [p for p in a.passes.split(',') if p]
    assert passes[0] == 'A' and set(passes) <= {'A', 'B'}, passes
    mj = a.members if a.members != 'none' and os.path.isfile(a.members) else None
    M = members(a.root, mj, ('C', 'D', 'E'))
    os.makedirs(a.pred_dir, exist_ok=True); os.makedirs(a.out, exist_ok=True)
    lines = []

    def say(s):
        print(s, flush=True); lines.append(s)
    say(f'k10_gold_read {time.strftime("%Y-%m-%d %H:%M:%S")} | root {a.root} | members json {mj} | folds '
        + ' '.join(f'{s} {sorted(M[s])}' for s in RUNS) + f' | passes {passes} | pred_dir {a.pred_dir}')
    setof = {p: s for s in M for p in M[s].values()}
    todo = sorted({p for s in M for p in M[s].values() for ps in passes if pred_path(p, ps, a.pred_dir, s) is None})
    if todo and a.infer:
        for p in todo:                                                  # stale / foreign files in OUR dir only
            run = os.path.basename(os.path.dirname(p))
            for ps in passes:
                for q in (f'{a.pred_dir}/{run}__raw__{ps}.csv', f'{a.pred_dir}/{run}__raw__{ps}.csv.sha256'):
                    if os.path.isfile(q):
                        os.remove(q)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', NTHR=os.environ.get('NTHR', '6'))
        cmd = [sys.executable, '-W', 'ignore', f'{H}/novel/gold_infer5.py', '--passes', ','.join(passes), '--out', a.pred_dir] + todo
        say(f'infer: {len(todo)} checkpoint(s) -> gold_infer5.py NTHR {env["NTHR"]} '
            f'({" ".join(os.path.basename(os.path.dirname(x)) for x in todo)})')
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        for l in r.stdout.splitlines():
            if 'gold58' in l or 'decoded' in l or 'done' in l:
                say('  ' + l)
        assert r.returncode == 0, r.stderr[-2000:]
        for p in todo:
            run = os.path.basename(os.path.dirname(p))
            for ps in passes:
                q = f'{a.pred_dir}/{run}__raw__{ps}.csv'
                assert os.path.isfile(q), q
                open(q + '.sha256', 'w').write(sha(p) + '\n')
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
                qa = pred_path(p, 'A', a.pred_dir, s)
                qb = pred_path(p, 'B', a.pred_dir, s) if mode == 'AB' else None
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
        for fl, ss in FLEETS.items():
            if all(sets[s] for s in ss):
                mean[fl] = np.mean([arr[k] for s in ss for k in sets[s]], axis=0)
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
            for x, y in (('K10', 'K9'), ('E', 'C'), ('K10W', 'K9'), ('K10W', 'K10')):
                if x in mean and y in mean:
                    dd = boot(x, y)
                    o[f'd_{x}_{y}'] = sc[x] - sc[y]; o[f'se_{x}_{y}'] = float(np.nanstd(dd))
                    o[f'p_{x}_{y}_gt0'] = float(np.nanmean(dd > 0))
                    o[f'fracture_{x}_{y}'] = fr(mean[x]) - fr(mean[y])
            if 'E' in mean and 'C' in mean:
                pl = {}
                for j, t in enumerate(T):
                    yy = Y[idx, j]; okm = np.isfinite(yy)
                    if okm.sum() and 0 < (yy[okm] >= .5).mean() < 1:
                        pl[t] = round(float(tb._auc(yy[okm] >= .5, mean['E'][idx][okm, j]) - tb._auc(yy[okm] >= .5, mean['C'][idx][okm, j])), 4)
                o['per_label_E_minus_C'] = pl
            out[sub] = o
        rho = {}
        for x, y in (('E', 'C'), ('D', 'C'), ('D', 'E')):
            fm = [F for F in range(5) if (x, F) in arr and (y, F) in arr]
            if fm:
                rho[f'{x}~{y}'] = float(np.mean([spearman_mean(arr[(x, F)], arr[(y, F)]) for F in fm]))
        res[mode] = dict(sets={s: [k[1] for k in ks] for s, ks in sets.items()}, n_ids=len(ids), rho_gold74_foldmatched=rho, **out)
        say(f'== pass {mode}: ' + ' '.join(f'{s} {res[mode]["sets"][s]}' for s in RUNS) + f', {len(ids)} gold studies')
        for sub in ('gold58', 'gold74'):
            o = out[sub]
            s = ' '.join(f'{k} {v:.4f}' for k, v in o['auc'].items())
            for x, y in (('K10', 'K9'), ('E', 'C'), ('K10W', 'K9'), ('K10W', 'K10')):
                if f'd_{x}_{y}' in o:
                    s += (f' | {x} - {y} {o[f"d_{x}_{y}"]:+.4f} (SE {o[f"se_{x}_{y}"]:.4f}, P>0 {o[f"p_{x}_{y}_gt0"]:.2f}; '
                          f'Fracture {o[f"fracture_{x}_{y}"]:+.4f})')
            say(f'  {sub} n {o["n"]}: {s}')
            if 'per_label_E_minus_C' in o:
                say(f'    per label E - C: {o["per_label_E_minus_C"]}')
        say(f'  fold-matched gold74 Spearman (mean over labels and folds): {rho}')
    orr = oof_read(M, mj, a.sel_csv)
    res['oof'] = orr
    if orr:
        say(f'  OOF (info, selection-biased): {json.dumps(orr, default=lambda v: round(v, 4))}')
    n = {s: len(M[s]) for s in ('C', 'D', 'E')}
    pa = res.get('A', {})
    prim = pa.get('gold58', {})
    final = all(v == 5 for v in n.values()) and all(pa.get('sets', {}).get(s, []) == [0, 1, 2, 3, 4] for s in ('C', 'D', 'E'))
    if 'd_K10_K9' not in prim:
        verdict, tag = 'NO READ (missing predictions or sets)', 'NOREAD'
    else:
        d, f58 = prim['d_K10_K9'], prim['fracture_K10_K9']
        on = d > BAR and f58 >= 0
        verdict = (f'K10 TIE RULE {"ON" if on else "OFF"}: gold58 pass A K10 - K9 {d:+.4f} (SE {prim["se_K10_K9"]:.4f}) '
                   f'{">" if d > BAR else "<="} {BAR}; Fracture K10 - K9 {f58:+.4f} ({">= 0" if f58 >= 0 else "< 0"})')
        if 'd_E_C' in prim:
            verdict += f' | E - C {prim["d_E_C"]:+.4f} (SE {prim["se_E_C"]:.4f})'
        if 'K10W' in prim['auc']:
            w58 = prim['auc']['K10W'] - max(prim['auc']['K9'], prim['auc']['K10'])
            comp = ('FN10-3 PREFERRED' if w58 > BAR else ('FN10' if on else 'K9 FN (rsna-sub-k9fn)'))
            verdict += f' | COMPOSITION: K10W - max(K9, K10) {w58:+.4f} -> {comp}'
        if 'AB' in res and 'd_K10_K9' in res['AB'].get('gold58', {}):
            dab = res['AB']['gold58']['d_K10_K9']
            verdict += (f' | BORDERLINE: the A+B served read says {dab:+.4f} (the primary pass A read decides)'
                        if (dab > BAR) != (d > BAR) else f' | A+B served read agrees ({dab:+.4f})')
        tag = 'FINAL' if final else f'PARTIAL (C {n["C"]}/5, D {n["D"]}/5, E {n["E"]}/5: NOT the decision read)'
        if not final:
            verdict = 'INFO ONLY, no verdict before C, D, E 5/5: ' + verdict
    if '/sim/' in a.root:
        tag = 'SIMULATION (E = set C weights relabelled; TEST ONLY) ' + tag
    say(f'K10GOLD {tag} | {verdict}')
    p = f'{a.out}/k10_gold_read_{time.strftime("%Y%m%d_%H%M")}.txt'
    open(p, 'w').write('\n'.join(lines) + '\n' + json.dumps(res, indent=1, default=str) + '\n')
    print(f'wrote {p}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
