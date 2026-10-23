#!/usr/bin/env python3
"""k14_gate.py (2026-10-11, Claude: k12_gate.py with the PREREG_K14 bars + T123v00 DO set; orig. from k11_gate.py). The PRE-REGISTERED K12 gate
(Downloads/rsna-knee/session_2026-10-10/PREREG_K14.md sec 4, sha256 98bcedb8..., written 10-11 ~10:40 IST before any
D-OAI fold 1-4 read): does D-OAI (DO) REPLACE set D, or join it, in the FN11 fleet? RSNA rows only.

  B = C + D + E + E2 (FN11, 20)   R = C + DO + E + E2 (REPLACE, 20)   A = C + D + DO + E + E2 (ADD, 25)
  G1 OOF : fold-matched OOF macro AUC vs llm_labels_v4_blend on the RSNA val rows (each set's fold-F member scores the
           fold-F val studies; a fleet row = the equal mean of its sets' rows); rows asserted free of oai ids and disjoint
           from gold74; DO's OOF ids == D's per fold. X passes G1 iff OOF(X) - OOF(B) >= +0.0005 (PREREG_K14).
  G2 gold58: pass A CPU fp32 predictions, equal mean of ALL fleet members, masked macro AUC on original_gold58.
           X passes G2 iff AUC58(X) - AUC58(B) >= -0.0014 (PREREG_K14).
  VERDICT (sec 4): R pass, A fail -> R. Both pass -> R unless OOF(A) - OOF(R) >= +.0003 AND AUC58(A) - AUC58(R) >= 0 -> A.
           Only A pass -> A. Neither -> NO K14 (FN11 stays).
  Tag: FINAL (DO 5/5 real) | NOT GATED (< 5 real DO folds) | SIMULATION (TEST ONLY).
Predictions: C / D read-only caches (as the K10 read), E ~/k10_test/gold/preds, E2 ~/k11_test/gold/preds (sha256
sidecars), DO --pred_dir (default ~/k14_test/gold/preds, written by --infer with sidecars). Writes only under ~/k14_test.
  CUDA_VISIBLE_DEVICES= NTHR=6 taskset -c 34-39 nice -n 19 ionice -c3 ~/miniconda3/envs/kmimic/bin/python \\
      ~/k14_test/k14/box/k14_gate.py [--infer] [--members ~/k14_test/members.json] [--out ~/k14_test/gold]
Last line: 'K14GATE <tag> | R: ... | A: ... | VERDICT ...'.
"""
import argparse, hashlib, json, os, subprocess, sys, time
import numpy as np, pandas as pd

H = os.path.expanduser('~')
sys.path.insert(0, f'{H}/novel/code')
READ_ONLY = [f'{H}/novel/k9gold/preds', f'{H}/novel/k8gold/preds', f'{H}/novel/gold74_preds']
SIDE = {'E': f'{H}/k10_test/gold/preds', 'E2': f'{H}/k11_test/gold/preds'}
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
S_T12 = ['PF OA', 'Lateral OA', 'Synovitis', 'Medial OA', 'Effusion', "Baker's"]
G1_BAR, G2_BAR, ADD_OOF, ADD_58 = 0.0005, -0.0014, 0.0003, 0.0   # PREREG_K14 sec gate
FLEETS = {'B': ('C', 'D', 'E', 'E2'), 'R': ('C', 'DO', 'E', 'E2'), 'A': ('C', 'D', 'DO', 'E', 'E2')}
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


def pred_path(r, pred_dir):
    run = os.path.basename(os.path.dirname(r['src']))
    name = f'{run}__raw__A.csv'
    if r['set'] in ('DO', 'E', 'E2'):
        q = f'{pred_dir if r["set"] == "DO" else SIDE[r["set"]]}/{name}'
        if os.path.isfile(q) and os.path.isfile(q + '.sha256') and open(q + '.sha256').read().strip() == r['sha256']:
            return q
        return None
    for d in READ_ONLY:
        q = f'{d}/{name}'
        if os.path.isfile(q) and os.path.getmtime(q) > os.path.getmtime(r['src']):
            return q
    return None


def spearman_mean(x, y):
    from scipy.stats import spearmanr
    v = [spearmanr(x[:, j], y[:, j]).correlation for j in range(x.shape[1])]
    v = [c for c in v if np.isfinite(c)]
    return float(np.mean(v)) if v else float('nan')


def rd(p):
    return pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[T]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', default=f'{H}/k14_test/members.json')
    ap.add_argument('--infer', action='store_true')
    ap.add_argument('--pred_dir', default=f'{H}/k14_test/gold/preds')
    ap.add_argument('--out', default=f'{H}/k14_test/gold')
    ap.add_argument('--sel_csv', default=f'{H}/rsna_train/llm_labels_v4_blend.csv')
    ap.add_argument('--boot', type=int, default=2000)
    a = ap.parse_args()
    for d in (a.out, a.pred_dir):
        assert os.path.realpath(os.path.expanduser(d)).startswith(os.path.realpath(f'{H}/k14_test') + '/'), d
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.pred_dir, exist_ok=True)
    lines = []

    def say(s):
        print(s, flush=True); lines.append(s)
    mj = json.load(open(a.members))
    assert mj.get('k12') is True, 'not a k14_merge.py members.json'
    M = mj['members']
    by = {}
    for r in M:
        by.setdefault(r['set'], {})[r['fold']] = r
    assert all(sorted(by.get(s, {})) == [0, 1, 2, 3, 4] for s in ('C', 'D', 'E', 'E2'))
    dof = sorted(by.get('DO', {}))
    nsim = sum(bool(r.get('simulated')) for r in by.get('DO', {}).values())
    say(f'k14_gate {time.strftime("%Y-%m-%d %H:%M:%S")} | members {a.members} (generated {mj.get("generated")}, interim '
        f'{mj.get("interim")}, simulated DO folds {nsim}) | DO folds {dof}')
    for r in M:
        assert os.path.isfile(r['src']) and sha(r['src']) == r['sha256'], f'{r["src"]} != members.json sha256'
    todo = [r for r in M if r['set'] == 'DO' and pred_path(r, a.pred_dir) is None]
    if todo and a.infer:
        for r in todo:
            run = os.path.basename(os.path.dirname(r['src']))
            for q in (f'{a.pred_dir}/{run}__raw__A.csv', f'{a.pred_dir}/{run}__raw__A.csv.sha256'):
                if os.path.isfile(q):
                    os.remove(q)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', NTHR=os.environ.get('NTHR', '6'))
        cmd = [sys.executable, '-W', 'ignore', f'{H}/novel/gold_infer5.py', '--passes', 'A', '--out', a.pred_dir] + [r['src'] for r in todo]
        say(f'infer: {len(todo)} DO checkpoint(s) -> gold_infer5.py NTHR {env["NTHR"]}')
        t0 = time.time()
        rr = subprocess.run(cmd, env=env, capture_output=True, text=True)
        for l in rr.stdout.splitlines():
            if 'gold58' in l or 'decoded' in l or 'done' in l:
                say('  ' + l)
        assert rr.returncode == 0, rr.stderr[-2000:]
        for r in todo:
            run = os.path.basename(os.path.dirname(r['src']))
            q = f'{a.pred_dir}/{run}__raw__A.csv'
            assert os.path.isfile(q), q
            open(q + '.sha256', 'w').write(r['sha256'] + '\n')
        say(f'  infer done in {time.time() - t0:.0f} s')
    miss = [r['file'] for r in M if pred_path(r, a.pred_dir) is None]
    import tta_bench as tb
    gold = pd.read_csv(f'{H}/rsna_train/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    res = dict(members=a.members, do_folds=dof, simulated=nsim, prereg_sha256='98bcedb8925e63e155ab2d763f00b5815915421bc466a149b1b3acf0278462bc')
    # ---------------- G1 OOF (RSNA val rows) ----------------
    sel = pd.read_csv(a.sel_csv, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    oof = {s: {F: rd(r['oof']) for F, r in by[s].items()} for s in by}
    rows, perfold, ids_all = {k: [] for k in FLEETS}, {}, []
    for F in dof:
        assert set(oof['DO'][F].index) == set(oof['D'][F].index), f'fold {F}: DO OOF ids != D OOF ids'
        ids = sorted(set.intersection(*[set(oof[s][F].index) for s in ('C', 'D', 'DO', 'E', 'E2')]))
        assert not (set(ids) & set(gold.index)) and not any('oai' in s.lower() for s in ids), f'fold {F}: bad OOF rows'
        ids_all += ids
        for k, ss in FLEETS.items():
            rows[k].append(np.mean([oof[s][F].loc[ids].to_numpy(np.float64) for s in ss], axis=0))
        Yf = sel.loc[ids, T].to_numpy(np.float64)
        perfold[F] = {s: float(tb.macro_auc(oof[s][F].loc[ids].to_numpy(np.float64), Yf)[0]) for s in by}
    Y = sel.loc[ids_all, T].to_numpy(np.float64) if ids_all else None
    o_auc = {k: float(tb.macro_auc(np.concatenate(v), Y)[0]) for k, v in rows.items()} if ids_all else {}
    rho_oof = float(np.mean([spearman_mean(oof['DO'][F].loc[sorted(oof['D'][F].index)].to_numpy(np.float64),
                                           oof['D'][F].loc[sorted(oof['D'][F].index)].to_numpy(np.float64)) for F in dof])) if dof else float('nan')
    res['oof'] = dict(n=len(ids_all), folds=dof, auc=o_auc, per_fold=perfold, rho_DO_D=rho_oof)
    if o_auc:
        say(f'== G1 OOF (folds {dof}, {len(ids_all)} RSNA val studies): ' + ' '.join(f'{k} {v:.5f}' for k, v in o_auc.items())
            + f' | R - B {o_auc["R"] - o_auc["B"]:+.5f} | A - B {o_auc["A"] - o_auc["B"]:+.5f} | A - R {o_auc["A"] - o_auc["R"]:+.5f}'
            + f' | Spearman DO~D (OOF) {rho_oof:.4f}')
        for F in dof:
            say(f'  fold {F}: ' + ' '.join(f'{s} {v:.5f}' for s, v in perfold[F].items())
                + f' | DO - D {perfold[F]["DO"] - perfold[F]["D"]:+.5f}')
        jS = [T.index(t) for t in S_T12]
        pl = {}
        for k in ('R', 'A'):
            X, Bm = np.concatenate(rows[k]), np.concatenate(rows['B'])
            pl[k] = {t: round(float(tb._auc(Y[np.isfinite(Y[:, j]), j] >= .5, X[np.isfinite(Y[:, j]), j])
                                    - tb._auc(Y[np.isfinite(Y[:, j]), j] >= .5, Bm[np.isfinite(Y[:, j]), j])), 4) for j, t in enumerate(T)}
        res['oof']['per_label'] = pl
        say(f'    OOF per label R - B: {pl["R"]}')
    # ---------------- G2 gold58 ----------------
    out = {}
    if miss:
        say(f'MISSING gold predictions for {len(miss)} member(s) (run with --infer): {miss}')
    elif dof:
        P = {(r['set'], r['fold']): rd(pred_path(r, a.pred_dir)) for r in M}
        gids = sorted(set.intersection(*[set(x.index) for x in P.values()]) & set(gold.index))
        g58 = [i for i, u in enumerate(gids) if gold.loc[u, 'source'] == 'original_gold58']
        YG = gold.loc[gids, T].to_numpy(np.float64)
        arr = {k: P[k].loc[gids].to_numpy(np.float64) for k in P}
        mean = {k: np.mean([arr[x] for x in arr if x[0] in ss], axis=0) for k, ss in FLEETS.items()}
        mean.update({s: np.mean([arr[x] for x in arr if x[0] == s], axis=0) for s in by})
        nmem = {k: sum(1 for x in arr if x[0] in ss) for k, ss in FLEETS.items()}
        rng = np.random.default_rng(59)
        for sub, idx in (('gold58', g58), ('gold74', list(range(len(gids))))):
            ii = np.asarray(idx)
            sc = {k: float(tb.macro_auc(m[ii], YG[ii])[0]) for k, m in mean.items()}
            o = dict(n=len(idx), auc=sc, n_members=nmem)
            Bs = [rng.integers(0, len(idx), len(idx)) for _ in range(a.boot)]
            for x, y in (('R', 'B'), ('A', 'B'), ('A', 'R'), ('DO', 'D')):
                dd = np.array([tb.macro_auc(mean[x][ii[b]], YG[ii[b]])[0] - tb.macro_auc(mean[y][ii[b]], YG[ii[b]])[0] for b in Bs])
                o[f'd_{x}_{y}'] = sc[x] - sc[y]; o[f'se_{x}_{y}'] = float(np.nanstd(dd)); o[f'p_{x}_{y}_gt0'] = float(np.nanmean(dd > 0))
                pl = {}
                for j, t in enumerate(T):
                    yy = YG[ii, j]; okm = np.isfinite(yy)
                    if okm.sum() and 0 < (yy[okm] >= .5).mean() < 1:
                        pl[t] = round(float(tb._auc(yy[okm] >= .5, mean[x][ii][okm, j]) - tb._auc(yy[okm] >= .5, mean[y][ii][okm, j])), 4)
                o[f'per_label_{x}_{y}'] = pl
            out[sub] = o
            s = ' '.join(f'{k} {v:.4f}' for k, v in sc.items())
            for x, y in (('R', 'B'), ('A', 'B'), ('A', 'R'), ('DO', 'D')):
                s += f' | {x} - {y} {o[f"d_{x}_{y}"]:+.4f} (SE {o[f"se_{x}_{y}"]:.4f})'
            say(f'== G2 {sub} pass A n {o["n"]} (members {nmem}): {s}')
            say(f'    per label R - B: {o["per_label_R_B"]}')
        fm = [F for F in dof if ('DO', F) in arr]
        out['rho_gold74_DO_D'] = float(np.mean([spearman_mean(arr[('DO', F)], arr[('D', F)]) for F in fm])) if fm else float('nan')
        say(f'  fold-matched gold74 Spearman DO~D: {out["rho_gold74_DO_D"]:.4f}')
        for F in fm:
            say(f'  DO f{F} single-member gold58 {tb.macro_auc(arr[("DO", F)][g58], YG[g58])[0]:.4f} vs D f{F} '
                f'{tb.macro_auc(arr[("D", F)][g58], YG[g58])[0]:.4f}')
    res['gold'] = out
    # ---------------- verdict ----------------
    real = len(dof) - nsim
    tag = 'FINAL (DO 5/5)' if real == 5 and dof == [0, 1, 2, 3, 4] else f'NOT GATED ({real} real DO folds < 5)'
    if nsim:
        tag = f'SIMULATION (TEST ONLY; {nsim} simulated DO folds) ' + tag
    passes, s = {}, []
    for x in ('R', 'A'):
        dO = o_auc[x] - o_auc['B'] if o_auc else float('nan')
        d58 = out['gold58'][f'd_{x}_B'] if 'gold58' in out else float('nan')
        passes[x] = bool(dO >= G1_BAR and d58 >= G2_BAR)
        s.append(f'{x}: dOOF {dO:+.5f} ({"PASS" if dO >= G1_BAR else "FAIL"}) d58 {d58:+.4f} '
                 f'({"PASS" if d58 >= G2_BAR else "FAIL"}) -> {"PASS" if passes[x] else "FAIL"}')
    if 'gold58' not in out or not o_auc:
        verdict = 'VERDICT NONE (missing predictions or folds)'
    else:
        aoof, a58 = o_auc['A'] - o_auc['R'], out['gold58']['d_A_R']
        if passes['R'] and passes['A'] and aoof >= ADD_OOF and a58 >= ADD_58:
            v = 'A (ADD)'
        elif passes['R']:
            v = 'R (REPLACE)'
        elif passes['A']:
            v = 'A (ADD)'
        else:
            v = 'NO K14 (FN11 stays)'
        verdict = f'VERDICT {v} [A - R: OOF {aoof:+.5f} (bar >= +{ADD_OOF}), gold58 {a58:+.4f} (bar >= 0)]'
        if 'NOT GATED' in tag:
            verdict = 'VERDICT NONE (not gated) ' + verdict.split(' ', 1)[1]
    res['verdict'] = dict(tag=tag, R=s[0], A=s[1], verdict=verdict)
    say(f'K14GATE {tag} | {s[0]} | {s[1]} | {verdict}')
    p = f'{a.out}/k14_gate_{time.strftime("%Y%m%d_%H%M")}.txt'
    open(p, 'w').write('\n'.join(lines) + '\n' + json.dumps(res, indent=1, default=str) + '\n')
    print(f'wrote {p}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
