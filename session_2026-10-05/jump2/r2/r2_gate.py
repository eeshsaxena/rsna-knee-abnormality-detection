#!/usr/bin/env python
# r2_gate.py (jump2 r2, 2026-10-05). CPU only, read-only inputs. Evaluates the PRE-REGISTERED round-2 gate (GATE.md):
# candidate fold-0 run vs J1 (~/fleet_runs/f0_s1043_ttr_mix50: same fold, same recipe, mix50 labels).
#   (a) val AUC vs steven_v4 (best tag, closing JSON)      bar: >= J1 - .003
#   (b) gold58 at the best tag's best epoch (run log)       bar: >= J1 - .005
#   (c) parent-likeness = mean per-label Spearman of the best-tag OOF vs the Raptor proxy on the 867 fold-0 val rows
#                                                           bar: <= J1 - .020
# The J1 numbers are recomputed from J1's files and must match the pre-registered values in GATE.md.
import argparse, json, os, re, sys, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

H = os.path.expanduser
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
ap = argparse.ArgumentParser()
ap.add_argument('--cand', required=True, help='candidate run dir, e.g. ~/fleet_runs/f0_s2043_ttr_r2a50')
ap.add_argument('--ref', default='~/fleet_runs/f0_s1043_ttr_mix50')
ap.add_argument('--proxy', default='~/jump2/r2/raptor_proxy_v5v8.csv')
ap.add_argument('--v4', default='~/rsna_train/llm_labels_v4_blend.csv')
ap.add_argument('--out', default='', help='write <out>.json and <out>.txt (default: none)')
A = ap.parse_args()
# pre-registered J1 values (MEASURED 2026-10-05 from J1's log / OOF; GATE.md)
J1_PRE = {'tag': 'raw', 'val': 0.8836, 'gold58': 0.9132, 'pl': 0.8962}
BAR = {'a': -0.003, 'b': -0.005, 'c': -0.020}
RX = re.compile(r'^ep (\d+) (raw|ema) \| gold74 ([\d.]+).*?gold58 ([\d.]+).*?val AUC vs \S+ ([\d.]+)')


def rd(p):
    return pd.read_csv(H(p), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')


def read_run(d):
    d = H(d).rstrip('/'); name = os.path.basename(d)
    m = re.match(r'^f(\d+)_s(\d+)', name); F, S = int(m.group(1)), int(m.group(2))
    L = [x.strip() for x in open(d + '.log', encoding='utf-8', errors='replace') if x.strip()]
    cl = json.loads(L[-1])
    assert 'raw' in cl and 'ema' in cl, f'{d}.log: no closing JSON'
    tag = 'raw' if cl['raw'] >= cl['ema'] else 'ema'
    ep = pd.DataFrame([g.groups() for g in map(RX.match, L) if g], columns=['ep', 'tag', 'g74', 'g58', 'val'])
    ep = ep.astype({'ep': int, 'g74': float, 'g58': float, 'val': float})
    e = ep[ep.tag == tag]
    hit = e[np.isclose(e.val, round(cl[tag], 4), atol=5e-5)]
    row = (hit if len(hit) else e.loc[[e.val.idxmax()]]).iloc[0]          # first epoch reaching the best (cv > best)
    oof = rd(f'{d}/oof_f{F}_s{S}_{tag}.csv')[T].astype(float)
    labs = [x for x in L if x.startswith('labels ')][:1]
    return {'dir': d, 'fold': F, 'seed': S, 'tag': tag, 'closing': cl, 'val': float(cl[tag]), 'best_ep': int(row.ep),
            'gold58': float(row.g58), 'gold74': float(row.g74), 'n_epochs_logged': int(len(e)), 'oof': oof,
            'labels_line': labs[0] if labs else ''}


def msp(X, Y, rows):
    return float(np.nanmean([spearmanr(X.loc[rows, l], Y.loc[rows, l]).correlation for l in T]))


ref, cand = read_run(A.ref), read_run(A.cand)
assert ref['fold'] == cand['fold'] == 0, 'gate is defined on fold 0'
P = rd(A.proxy)[T].astype(float); v4 = rd(A.v4)[T].astype(float)
rows = sorted(set(ref['oof'].index) & set(cand['oof'].index) & set(P.index))
assert len(rows) == len(ref['oof']) == len(cand['oof']), (len(rows), len(ref['oof']), len(cand['oof']))
for r in (ref, cand):
    r['pl'] = msp(r['oof'], P, rows)
    r['auc_v4_recomputed'] = float(np.mean([roc_auc_score(v4.loc[rows, l] >= .5, r['oof'].loc[rows, l]) for l in T]))
r_mm = msp(ref['oof'], cand['oof'], rows)

# pre-registered J1 values must reproduce
chk = {'tag': ref['tag'] == J1_PRE['tag'], 'val': abs(ref['val'] - J1_PRE['val']) < 6e-5,
       'gold58': abs(ref['gold58'] - J1_PRE['gold58']) < 6e-5, 'pl': abs(ref['pl'] - J1_PRE['pl']) < 6e-5}
crit = {
    'a_valAUC_vs_v4': (cand['val'], ref['val'] + BAR['a'], cand['val'] >= ref['val'] + BAR['a']),
    'b_gold58': (cand['gold58'], ref['gold58'] + BAR['b'], cand['gold58'] >= ref['gold58'] + BAR['b']),
    'c_parentlikeness': (cand['pl'], ref['pl'] + BAR['c'], cand['pl'] <= ref['pl'] + BAR['c']),
}
verdict = 'PASS' if all(v[2] for v in crit.values()) else 'FAIL'

# (d) INFO, not gating: the frontier of rank blends of members we ALREADY have (J1 raw + TTR v4 ema, fold 0).
# A candidate is only new if it beats the blend that has the same parent-likeness (GATE.md sec 4).
front = None
try:
    V = rd('~/fleet_runs/f0_s1042_ttr/oof_f0_s1042_ema.csv')[T].astype(float)
    gt = rd('~/rsna_train/gold74_eval_set.csv'); G58 = [s for s in gt.index if gt.loc[s, 'source'] == 'original_gold58']
    gJ = rd('~/jump2/gold74_preds/f0_s1043_ttr_mix50__raw.csv')[T]; gV = rd('~/jump2/gold74_preds/f0_s1042_ttr__ema.csv')[T]
    from scipy.stats import rankdata
    curve = []
    for w in np.linspace(0, 1, 21):
        au, pl, g5 = [], [], []
        for l in T:
            b = (1 - w) * rankdata(ref['oof'].loc[rows, l]) + w * rankdata(V.loc[rows, l])
            au.append(roc_auc_score(v4.loc[rows, l] >= .5, b)); pl.append(spearmanr(b, P.loc[rows, l]).correlation)
            yg = gt.loc[G58, l].astype(float) > .5
            if 0 < yg.mean() < 1:
                g5.append(roc_auc_score(yg, (1 - w) * rankdata(gJ.loc[G58, l]) + w * rankdata(gV.loc[G58, l])))
        curve.append((w, np.mean(au), np.mean(pl), np.mean(g5)))
    ok = [c for c in curve if c[2] <= cand['pl'] + 1e-9]
    if ok:
        best = max(ok, key=lambda c: c[1])
        front = {'w_ttrv4': best[0], 'auc': best[1], 'pl': best[2], 'gold58': best[3],
                 'd_auc': cand['val'] - best[1], 'd_gold58': cand['gold58'] - best[3]}
except Exception as e:                                   # info only; never blocks the gate
    front = {'error': repr(e)}
lines = [f'R2 GATE {time.strftime("%Y-%m-%d %H:%M")}  candidate {cand["dir"]}  vs  J1 {ref["dir"]}',
         f'  J1 pre-registered values reproduce: {chk}',
         f'  candidate: tag {cand["tag"]} best ep {cand["best_ep"]} (0-based) | closing {cand["closing"]} | {cand["labels_line"][:110]}',
         f'  J1       : tag {ref["tag"]} best ep {ref["best_ep"]} (0-based) | closing {ref["closing"]}']
for k, (x, bar, ok) in crit.items():
    j = {'a_valAUC_vs_v4': ref['val'], 'b_gold58': ref['gold58'], 'c_parentlikeness': ref['pl']}[k]
    op = '<=' if k.startswith('c') else '>='
    lines.append(f'  ({k}) candidate {x:.4f} | J1 {j:.4f} | bar {op} {bar:.4f} | {"PASS" if ok else "FAIL"}')
lines += [f'  info: gold74 cand {cand["gold74"]:.4f} J1 {ref["gold74"]:.4f} | OOF AUC vs v4>=.5 recomputed cand '
          f'{cand["auc_v4_recomputed"]:.4f} J1 {ref["auc_v4_recomputed"]:.4f} | Spearman(cand OOF, J1 OOF) {r_mm:.4f}']
if front and 'error' not in front:
    lines.append(f'  (d INFO) existing-member frontier at PL <= {cand["pl"]:.4f}: J1+TTRv4 rank blend w_TTRv4 '
                 f'{front["w_ttrv4"]:.2f} AUC {front["auc"]:.4f} gold58 {front["gold58"]:.4f} PL {front["pl"]:.4f} | '
                 f'candidate minus frontier: AUC {front["d_auc"]:+.4f} gold58 {front["d_gold58"]:+.4f} '
                 f'({"BEATS" if front["d_auc"] >= .001 and front["d_gold58"] >= 0 else "does not beat"} the frontier; '
                 f'bar AUC +.001 and gold58 >= 0)')
elif front:
    lines.append(f'  (d INFO) frontier not computed: {front["error"]}')
else:
    lines.append('  (d INFO) candidate PL below every J1/TTRv4 blend: frontier undefined (candidate is less parent-like '
                 'than TTR v4 itself)')
lines.append(f'  VERDICT {verdict}')
print('\n'.join(lines), flush=True)
if A.out:
    with open(H(A.out) + '.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    js = {'verdict': verdict, 'criteria': {k: {'cand': v[0], 'bar': v[1], 'pass': bool(v[2])} for k, v in crit.items()},
          'j1_reproduces': chk, 'r_mm_spearman': r_mm, 'frontier_info': front,
          'cand': {k: v for k, v in cand.items() if k != 'oof'}, 'ref': {k: v for k, v in ref.items() if k != 'oof'}}
    with open(H(A.out) + '.json', 'w') as f:
        json.dump(js, f, indent=1, default=float)
sys.exit(0 if verdict == 'PASS' else 1)
