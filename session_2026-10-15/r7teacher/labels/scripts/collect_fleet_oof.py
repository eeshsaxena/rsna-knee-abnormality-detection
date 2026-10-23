# collect_fleet_oof.py (r7teacher/labels, 2026-10-14): box CPU, read only on ~/fleet_runs. Collects the out-of-fold
# predictions of our TTR fleet sets (5 folds each, folds5.csv split, gold74 excluded from every run) into one table per
# set and one fleet-mean table, with sha256 receipts. A set's OOF row for study s comes from the ONE fold model that did
# not train on s (folds5 fold of s). File used per fold: <run>/oof_f{F}_s{seed}_raw.csv (the raw-best checkpoint's OOF,
# the same files k14_gate.py reads as G1 OOF).
# usage: python collect_fleet_oof.py --sets C=s1044_ttr288mix50,D=s1043_ttr_frac75,... --out DIR [--fleet_name NAME]
#        [--gold_sets C=...]  (gold74 pass-A prediction caches, see GOLD_CACHES; used for the gold rows of the fleet mean)
import argparse, hashlib, json, os, sys, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr

H = os.path.expanduser('~')
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
GOLD_CACHES = [f'{H}/novel/k9gold/preds', f'{H}/novel/k8gold/preds', f'{H}/novel/gold74_preds', f'{H}/k10_test/gold/preds',
               f'{H}/k11_test/gold/preds', f'{H}/k14_test/gold/preds', f'{H}/r7t_labels/gold_preds']
ap = argparse.ArgumentParser()
ap.add_argument('--sets', required=True)
ap.add_argument('--runs', default=f'{H}/fleet_runs')
ap.add_argument('--folds', default=f'{H}/rsna_train/folds5.csv')
ap.add_argument('--gold', default=f'{H}/rsna_train/gold74_eval_set.csv')
ap.add_argument('--v4', default=f'{H}/rsna_train/llm_labels_v4_blend.csv')
ap.add_argument('--out', required=True)
ap.add_argument('--fleet_name', default='fleet')
ap.add_argument('--allow_oai', action='store_true')     # only for the SIMULATED teacher (DT); never for a training table
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


rd = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
folds = rd(A.folds)['fold']; gold = rd(A.gold); v4 = rd(A.v4)[T]
gset = set(gold.index)
pool = [s for s in folds.index if folds[s] >= 0 and s not in gset]          # == fleet_train train+val ids (store has all)
print(f'folds5 {len(folds)} | gold74 {len(gold)} | non-gold pool {len(pool)}', flush=True)
sets = [kv.split('=') for kv in A.sets.split(',')]
rec = dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), sets={}, folds_sha256=sha(A.folds), gold_sha256=sha(A.gold))
tabs, gtabs = {}, {}
for name, tag in sets:
    seed = int(tag.split('_')[0][1:])
    if ('oai' in tag.lower()) and not A.allow_oai:
        sys.exit(f'REFUSE {name}={tag}: OAI-trained set (pass --allow_oai only for the simulated teacher)')
    parts, files = [], []
    for F in range(5):
        p = f'{A.runs}/f{F}_{tag}/oof_f{F}_s{seed}_raw.csv'
        assert os.path.isfile(p), f'missing {p}'
        o = rd(p)[T]
        want = sorted(s for s in pool if folds[s] == F)
        assert sorted(o.index) == want, f'{p}: OOF ids != folds5 fold-{F} val ids minus gold74 ({len(o)} vs {len(want)})'
        assert np.isfinite(o.values).all() and (o.values >= 0).all() and (o.values <= 1).all(), p
        parts.append(o); files.append(dict(path=p, sha256=sha(p), rows=len(o)))
    so = pd.concat(parts).loc[pool]
    assert so.index.is_unique and len(so) == len(pool)
    so.to_csv(f'{A.out}/oof_{name}.csv')
    tabs[name] = so
    # gold74 rows: pass-A predictions of the 5 fold members (every fold model is out-of-sample there), equal mean
    gp = []
    for F in range(5):
        run = f'f{F}_{tag}'
        q = next((f'{d}/{run}__raw__A.csv' for d in GOLD_CACHES if os.path.isfile(f'{d}/{run}__raw__A.csv')), None)
        if q: gp.append(rd(q)[T])
    if len(gp) == 5:
        gm = sum(g.loc[gold.index] for g in gp) / 5.0
        gtabs[name] = gm
    # sanity: OOF macro AUC vs v4 (== fleet_train macro_auc convention: label >= .5, both classes present)
    from sklearn.metrics import roc_auc_score
    Y = v4.loc[pool].values
    au = np.mean([roc_auc_score(Y[:, j] >= .5, so.values[:, j]) for j in range(12)])
    rec['sets'][name] = dict(tag=tag, seed=seed, files=files, oof_auc_vs_v4=round(float(au), 5),
                             gold_preds=len(gp) == 5)
    print(f'set {name:3s} {tag:28s} OOF rows {len(so)} | OOF macro AUC vs v4 {au:.5f} | gold74 preds {"5/5" if len(gp) == 5 else f"{len(gp)}/5 (missing)"}', flush=True)

names = [n for n, _ in sets]
fm = sum(tabs[n] for n in names) / len(names)                      # prob-space equal mean (as inside the fleet kernels)
fm.to_csv(f'{A.out}/oof_{A.fleet_name}_mean.csv')
rec['fleet_mean'] = dict(name=A.fleet_name, members=names, path=f'{A.out}/oof_{A.fleet_name}_mean.csv',
                         sha256=sha(f'{A.out}/oof_{A.fleet_name}_mean.csv'))
if all(n in gtabs for n in names):
    gm = sum(gtabs[n] for n in names) / len(names)
    gm.to_csv(f'{A.out}/gold_{A.fleet_name}_mean.csv')
    rec['fleet_gold_mean'] = dict(path=f'{A.out}/gold_{A.fleet_name}_mean.csv', sha256=sha(f'{A.out}/gold_{A.fleet_name}_mean.csv'))
    print(f'fleet gold74 mean written ({len(gm)} rows)')
else:
    print('fleet gold74 mean NOT written (a set lacks gold preds): ' + str([n for n in names if n not in gtabs]))
# pairwise mean per-label Spearman on the pool rows (reference for the student diversity leg)
rho = {}
for i, a in enumerate(names):
    for b in names[i + 1:]:
        rho[f'{a}~{b}'] = round(float(np.mean([spearmanr(tabs[a].values[:, j], tabs[b].values[:, j]).correlation
                                               for j in range(12)])), 4)
rec['pairwise_spearman_oof'] = rho
print('pairwise mean per-label Spearman (OOF pool rows): ' + ' '.join(f'{k} {v}' for k, v in rho.items()))
json.dump(rec, open(f'{A.out}/fleet_oof_receipt_{A.fleet_name}.json', 'w'), indent=1)
print(f'wrote {A.out}/oof_{A.fleet_name}_mean.csv + receipt')
