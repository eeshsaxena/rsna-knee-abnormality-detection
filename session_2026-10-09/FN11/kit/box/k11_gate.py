#!/usr/bin/env python3
"""k11_gate.py (box, CPU only; 2026-10-09, Claude; from k10_gold_read.py). The PRE-REGISTERED FN11 gate
(Downloads/rsna-knee/session_2026-10-09/FN11/PREREG_FN11.md sec 3, sha256 5d19d441a8b7a8e4..., written 10-09 ~16:50 IST
before any E2 fold was final): does set E2 (TTR288 frac75 s1046) join the parent-free fleet C + D + E?

  B0 = C + D + E (15 members == FN10-3), B1 = B0 + the ACCEPTED E2 members (k11_merge.py) [B2 = B1 + D2, --with_d2: sec 6]
  G1 OOF : dOOF = macroAUC_OOF(B1) - macroAUC_OOF(B0) >= 0.0000. Fold-matched OOF: a val study of fold F gets, per set,
           the fold-F member's OOF row (the picked RAW checkpoint's OOF csv); a fleet row = the equal mean of its sets' rows.
           Labels ~/rsna_train/llm_labels_v4_blend.csv; masked macro AUC = tta_bench.macro_auc. Rows = the val studies of
           the folds where E2 has an accepted member, intersected over sets, asserted disjoint from gold74. (Selection-biased
           equally for every set: a guard, not an unbiased gain estimate.)
  G2 gold58: d58 = AUC58(B1) - AUC58(B0) >= -0.0005. Pass A (offset 0, no mirror) CPU fp32 predictions
           (~/novel/gold_infer5.py); fleet = equal mean of ALL its members; masked macro AUC on original_gold58 rows.
  VERDICT "E2 JOINS" iff G1 AND G2, else "E2 DOES NOT JOIN" (FN10-3 stays the vehicle; nothing is built).
  Tag: FINAL (E2 5/5) | E2-4 DECISION (members.json e2_ship 4of5) | E2-4 PROVISIONAL (folds 0-3, f4 pending: info, sec 4)
       | E2 NOT GATED (< 4 accepted folds).
  Reported, never gating: gold74 delta, paired study bootstrap SE (B 2000, seed 59), per-label deltas (Fracture), per-set
  and per-fold OOF AUC, E2 vs E per fold, mean per-label Spearman E2~E (fold-matched, gold74 and OOF).
Predictions: C / D from the read-only caches of the K10 read (~/novel/{k9gold,k8gold}/preds, ~/novel/gold74_preds; newer
than the checkpoint), E from ~/k10_test/gold/preds (sidecar == the ckpt sha256), E2 / D2 from --pred_dir (default
~/k11_test/gold/preds; written by --infer with a .sha256 sidecar). Every member's checkpoint sha256 == members.json.
Writes only under ~/k11_test.
  CUDA_VISIBLE_DEVICES= NTHR=6 taskset -c 34-39 nice -n 19 ionice -c3 ~/miniconda3/envs/kmimic/bin/python \\
      ~/k11_test/k11/box/k11_gate.py [--infer] [--members ~/k11_test/members.json] [--out ~/k11_test/gold] [--with_d2]
Last line: 'K11GATE <tag> | G1 dOOF ... | G2 d58 ... | VERDICT ...'.
"""
import argparse, hashlib, json, os, subprocess, sys, time
import numpy as np, pandas as pd

H = os.path.expanduser('~')
sys.path.insert(0, f'{H}/novel/code')
READ_ONLY = [f'{H}/novel/k9gold/preds', f'{H}/novel/k8gold/preds', f'{H}/novel/gold74_preds']
E_PRED = f'{H}/k10_test/gold/preds'
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
G1_BAR, G2_BAR = 0.0, -0.0005
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
    ck = r['src']
    run = os.path.basename(os.path.dirname(ck))
    name = f'{run}__raw__A.csv'
    if r['set'] in ('E2', 'D2', 'E'):
        d = pred_dir if r['set'] != 'E' else E_PRED
        q = f'{d}/{name}'
        if os.path.isfile(q) and os.path.isfile(q + '.sha256') and open(q + '.sha256').read().strip() == r['sha256']:
            return q
        return None
    for d in READ_ONLY:                                                   # C / D: frozen runs, read-only caches
        q = f'{d}/{name}'
        if os.path.isfile(q) and os.path.getmtime(q) > os.path.getmtime(ck):
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
    ap.add_argument('--members', default=f'{H}/k11_test/members.json')
    ap.add_argument('--infer', action='store_true')
    ap.add_argument('--pred_dir', default=f'{H}/k11_test/gold/preds')
    ap.add_argument('--out', default=f'{H}/k11_test/gold')
    ap.add_argument('--sel_csv', default=f'{H}/rsna_train/llm_labels_v4_blend.csv')
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--with_d2', action='store_true')
    ap.add_argument('--sim', action='store_true', help='TEST ONLY: members.json whose E2 = relabelled E picks')
    a = ap.parse_args()
    for d in (a.out, a.pred_dir):
        assert os.path.realpath(os.path.expanduser(d)).startswith(os.path.realpath(f'{H}/k11_test') + '/'), \
            f'{d}: k11_gate writes only under ~/k11_test'
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.pred_dir, exist_ok=True)
    lines = []

    def say(s):
        print(s, flush=True); lines.append(s)
    mj = json.load(open(a.members))
    assert mj.get('k11') is True, 'not a k11_merge.py members.json'
    M = mj['members']
    by = {}
    for r in M:
        by.setdefault(r['set'], {})[r['fold']] = r
    assert all(sorted(by.get(s, {})) == [0, 1, 2, 3, 4] for s in ('C', 'D', 'E')), {s: sorted(by.get(s, {})) for s in by}
    e2f = sorted(by.get('E2', {}))
    say(f'k11_gate {time.strftime("%Y-%m-%d %H:%M:%S")} | members {a.members} (generated {mj.get("generated")}, interim '
        f'{mj.get("interim")}, e2_ship {mj.get("e2_ship")}) | E2 folds {e2f} | D2 folds {sorted(by.get("D2", {}))}'
        + (' | SIMULATION (E2 = relabelled E: TEST ONLY)' if a.sim else ''))
    for r in M:                                                          # every checkpoint == the merged pick
        assert os.path.isfile(r['src']) and sha(r['src']) == r['sha256'], f'{r["src"]} != members.json sha256'
    todo = [r for r in M if r['set'] in ('E2', 'D2') and pred_path(r, a.pred_dir) is None]
    if todo and a.infer:
        assert not a.sim, 'no inference in --sim'
        for r in todo:
            run = os.path.basename(os.path.dirname(r['src']))
            for q in (f'{a.pred_dir}/{run}__raw__A.csv', f'{a.pred_dir}/{run}__raw__A.csv.sha256'):
                if os.path.isfile(q):
                    os.remove(q)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', NTHR=os.environ.get('NTHR', '6'))
        cmd = [sys.executable, '-W', 'ignore', f'{H}/novel/gold_infer5.py', '--passes', 'A', '--out', a.pred_dir] + [r['src'] for r in todo]
        say(f'infer: {len(todo)} checkpoint(s) -> gold_infer5.py NTHR {env["NTHR"]} ({" ".join(r["file"] for r in todo)})')
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
    res = dict(members=a.members, e2_folds=e2f, prereg_sha256='5d19d441a8b7a8e4a4fb2d5aceb2297e6217c4de6c33e68bd180940091a2c79c')
    fleets = {'B0': ('C', 'D', 'E'), 'B1': ('C', 'D', 'E', 'E2')}
    if a.with_d2:
        fleets['B2'] = ('C', 'D', 'E', 'E2', 'D2')
    pairs = [('B1', 'B0')] + ([('B2', 'B1')] if a.with_d2 else [])
    # ---------------- G1: OOF (non-gold rows), fold-matched ----------------
    sel = pd.read_csv(a.sel_csv, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    newset = 'D2' if a.with_d2 else 'E2'
    gf = sorted(by.get(newset, {})) if not a.with_d2 else sorted(set(by.get('D2', {})) & set(e2f))
    oof = {s: {F: rd(r['oof']) for F, r in by[s].items()} for s in by}
    rows, perfold = {k: [] for k in fleets}, {}
    ids_all = []
    for F in gf:
        ids = sorted(set.intersection(*[set(oof[s][F].index) for s in fleets[max(fleets)] if F in oof.get(s, {})]))
        assert not (set(ids) & set(gold.index)), f'fold {F}: OOF rows overlap gold74'
        ids_all += ids
        for k, ss in fleets.items():
            rows[k].append(np.mean([oof[s][F].loc[ids].to_numpy(np.float64) for s in ss], axis=0))
        Yf = sel.loc[ids, T].to_numpy(np.float64)
        perfold[F] = {s: float(tb.macro_auc(oof[s][F].loc[ids].to_numpy(np.float64), Yf)[0]) for s in by if F in oof[s]}
    Y = sel.loc[ids_all, T].to_numpy(np.float64)
    o_auc = {k: float(tb.macro_auc(np.concatenate(v), Y)[0]) for k, v in rows.items()}
    o_set = {s: float(tb.macro_auc(np.concatenate([oof[s][F].loc[sorted(set.intersection(*[set(oof[x][F].index) for x in fleets[max(fleets)] if F in oof.get(x, {})]))].to_numpy(np.float64) for F in gf]), Y)[0]) for s in by}
    rho_oof = float(np.mean([spearman_mean(oof['E2'][F].loc[sorted(set(oof['E2'][F].index) & set(oof['E'][F].index))].to_numpy(np.float64),
                                           oof['E'][F].loc[sorted(set(oof['E2'][F].index) & set(oof['E'][F].index))].to_numpy(np.float64))
                             for F in e2f])) if e2f else float('nan')
    res['oof'] = dict(n=len(ids_all), folds=gf, auc=o_auc, per_set=o_set, per_fold=perfold, rho_E2_E=rho_oof)
    say(f'== G1 OOF (folds {gf}, {len(ids_all)} val studies, labels {os.path.basename(a.sel_csv)}): '
        + ' '.join(f'{k} {v:.5f}' for k, v in o_auc.items()) + ' | per set ' + ' '.join(f'{s} {v:.5f}' for s, v in o_set.items())
        + f' | Spearman E2~E (OOF, fold-matched) {rho_oof:.4f}')
    for F in gf:
        say(f'  fold {F}: ' + ' '.join(f'{s} {v:.5f}' for s, v in perfold[F].items())
            + (f' | E2 - E {perfold[F]["E2"] - perfold[F]["E"]:+.5f}' if 'E2' in perfold[F] else ''))
    # ---------------- G2: gold58 pass A ----------------
    out = {}
    if miss:
        say(f'MISSING gold predictions for {len(miss)} member(s) (run with --infer): {miss}')
    else:
        P = {(r['set'], r['fold']): rd(pred_path(r, a.pred_dir)) for r in M}
        gids = sorted(set.intersection(*[set(x.index) for x in P.values()]) & set(gold.index))
        g58 = [i for i, u in enumerate(gids) if gold.loc[u, 'source'] == 'original_gold58']
        YG = gold.loc[gids, T].to_numpy(np.float64)
        arr = {k: P[k].loc[gids].to_numpy(np.float64) for k in P}
        mean = {k: np.mean([arr[x] for x in arr if x[0] in ss], axis=0) for k, ss in fleets.items()}
        mean.update({s: np.mean([arr[x] for x in arr if x[0] == s], axis=0) for s in by})
        nmem = {k: sum(1 for x in arr if x[0] in ss) for k, ss in fleets.items()}
        jF = T.index('Fracture')
        rng = np.random.default_rng(59)
        for sub, idx in (('gold58', g58), ('gold74', list(range(len(gids))))):
            ii = np.asarray(idx)
            sc = {k: float(tb.macro_auc(m[ii], YG[ii])[0]) for k, m in mean.items()}
            o = dict(n=len(idx), auc=sc, n_members=nmem)
            B = [rng.integers(0, len(idx), len(idx)) for _ in range(a.boot)]
            for x, y in pairs + [('E2', 'E')]:
                if x not in mean or y not in mean:
                    continue
                dd = np.array([tb.macro_auc(mean[x][ii[b]], YG[ii[b]])[0] - tb.macro_auc(mean[y][ii[b]], YG[ii[b]])[0] for b in B])
                o[f'd_{x}_{y}'] = sc[x] - sc[y]
                o[f'se_{x}_{y}'] = float(np.nanstd(dd)); o[f'p_{x}_{y}_gt0'] = float(np.nanmean(dd > 0))
                pl = {}
                for j, t in enumerate(T):
                    yy = YG[ii, j]; okm = np.isfinite(yy)
                    if okm.sum() and 0 < (yy[okm] >= .5).mean() < 1:
                        pl[t] = round(float(tb._auc(yy[okm] >= .5, mean[x][ii][okm, j]) - tb._auc(yy[okm] >= .5, mean[y][ii][okm, j])), 4)
                o[f'per_label_{x}_{y}'] = pl
            out[sub] = o
            s = ' '.join(f'{k} {v:.4f}' for k, v in sc.items())
            for x, y in pairs + [('E2', 'E')]:
                if f'd_{x}_{y}' in o:
                    s += (f' | {x} - {y} {o[f"d_{x}_{y}"]:+.4f} (SE {o[f"se_{x}_{y}"]:.4f}, P>0 {o[f"p_{x}_{y}_gt0"]:.2f}; '
                          f'Fracture {o[f"per_label_{x}_{y}"].get("Fracture", float("nan")):+.4f})')
            say(f'== G2 {sub} pass A n {o["n"]} (members {nmem}): {s}')
            for x, y in pairs:
                say(f'    per label {x} - {y}: {o[f"per_label_{x}_{y}"]}')
        fm = [F for F in e2f if ('E2', F) in arr and ('E', F) in arr]
        rho_g = float(np.mean([spearman_mean(arr[('E2', F)], arr[('E', F)]) for F in fm])) if fm else float('nan')
        out['rho_gold74_E2_E'] = rho_g
        say(f'  fold-matched gold74 Spearman E2~E (mean over labels and folds {fm}): {rho_g:.4f}')
        for F in e2f:
            if ('E2', F) in arr:
                say(f'  E2 f{F} single-member gold58 {tb.macro_auc(arr[("E2", F)][g58], YG[g58])[0]:.4f} vs E f{F} '
                    f'{tb.macro_auc(arr[("E", F)][g58], YG[g58])[0]:.4f}')
    res['gold'] = out
    # ---------------- verdict ----------------
    x, y = pairs[-1]
    dO = o_auc[x] - o_auc[y]
    n_new = len(gf)
    ship = mj.get('e2_ship')
    if n_new < 4:
        tag = f'{newset} NOT GATED ({n_new} accepted folds < 4)'
    elif a.with_d2:
        tag = 'D2 FINAL' if n_new == 5 else 'D2 NOT GATED (needs 5/5)'
    elif n_new == 5:
        tag = 'FINAL (E2 5/5)'
    elif ship == '4of5':
        tag = 'E2-4 DECISION (e2_ship 4of5, PREREG sec 4)'
    else:
        tag = 'E2-4 PROVISIONAL (f4 pending; info only until f4 lands or the 10-10 12:00 IST cutoff, PREREG sec 4)'
    if a.sim:
        tag = 'SIMULATION (TEST ONLY) ' + tag
    g1 = dO >= G1_BAR
    s1 = f'G1 dOOF {x} - {y} {dO:+.5f} ({"PASS" if g1 else "FAIL"}, bar >= {G1_BAR:+.4f})'
    if 'gold58' in out:
        d58 = out['gold58'][f'd_{x}_{y}']
        g2 = d58 >= G2_BAR
        s2 = (f'G2 d58 {d58:+.4f} (SE {out["gold58"][f"se_{x}_{y}"]:.4f}; {"PASS" if g2 else "FAIL"}, bar >= {G2_BAR:+.4f}) | '
              f'info d74 {out["gold74"][f"d_{x}_{y}"]:+.4f} (SE {out["gold74"][f"se_{x}_{y}"]:.4f})')
        verdict = (f'VERDICT {newset} JOINS' if (g1 and g2) else f'VERDICT {newset} DOES NOT JOIN')
        if 'NOT GATED' in tag:
            verdict = 'VERDICT NONE (not gated)'
    else:
        s2, verdict = 'G2 MISSING PREDICTIONS', 'VERDICT NONE (G2 missing: run --infer)'
    res['verdict'] = dict(tag=tag, g1=s1, g2=s2, verdict=verdict)
    say(f'K11GATE {tag} | {s1} | {s2} | {verdict}')
    p = f'{a.out}/k11_gate_{time.strftime("%Y%m%d_%H%M")}.txt'
    open(p, 'w').write('\n'.join(lines) + '\n' + json.dumps(res, indent=1, default=str) + '\n')
    print(f'wrote {p}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
