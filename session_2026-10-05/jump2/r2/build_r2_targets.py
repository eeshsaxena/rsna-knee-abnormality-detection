#!/usr/bin/env python
# build_r2_targets.py (jump2 r2, 2026-10-05). CPU only. Own-OOF pseudo-label round 2 (Archit pattern):
#   target = a * base + (1 - a) * oof_mapped   (probability space, per cell)
# base       = report labels (default steven_v4 = ~/rsna_train/llm_labels_v4_blend.csv; NOT mix50, so the parent teacher
#              does not enter the target directly).
# oof_mapped = out-of-fold predictions of a 5-fold set of our own runs, per fold and per label z-scored on the logit
#              scale, (optionally averaged over several sets), then mapped onto the base label's logit mean/std over the
#              covered training rows (logit mean/std matching, NOT ECDF: steven_v4 has heavy ties).
# Silence-aware variant 'aXXsYY': report-silent cells (verdict UNK in report_labels_v2.csv, or the base tie mode when
#              no verdict file is given) get OOF weight YY/100; spoken cells keep base weight XX/100.
# OOF guarantee: every OOF row must belong to the run's own validation fold (folds5), never to the held-out gold74 set;
#              only runs with a closing JSON line in <run>.log are used (unfinished runs may hold a mid-training OOF csv).
# Gold74 / held-out rows and rows absent from the store keep the base values (never trained on).
# Inputs are read-only. Writes only into --out_dir (refuses ~/rsna_train, ~/fleet_runs, ~/fleet, ~/bigjump_build).
import argparse, glob, hashlib, json, os, re, sys, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr

H = os.path.expanduser
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
ap = argparse.ArgumentParser()
ap.add_argument('--sets', nargs='+', default=['~/fleet_runs/f?_s1043_ttr_mix50'],
                help='one glob of run dirs per OOF set; several sets are averaged in z-logit space')
ap.add_argument('--base', default='~/rsna_train/llm_labels_v4_blend.csv')
ap.add_argument('--folds', default='~/rsna_train/folds5.csv')
ap.add_argument('--exclude', default='~/rsna_train/gold74_eval_set.csv')
ap.add_argument('--store', default='~/fleet_store')
ap.add_argument('--verdicts', default='~/rsna_train/report_labels_v2.csv', help="'' = use the base tie mode as silence")
ap.add_argument('--proxy', nargs='*', default=['~/jump2/r2/raptor_proxy_v5v8.csv'], help='parent proxies (diagnostic)')
ap.add_argument('--ref', nargs='*', default=['~/rsna_train/labels_mix50_logit.csv'], help='reference tables (diagnostic)')
ap.add_argument('--variants', default='a50,a60,a50s75')
ap.add_argument('--out_dir', default='~/jump2/r2')
ap.add_argument('--partial', type=int, default=0, help='1 = dry run: uncovered training rows keep base, files get DRY_')
ap.add_argument('--eps', type=float, default=0.005, help='probability clip before logit (base min is 0.005)')
ap.add_argument('--per_label', type=int, default=1)
A = ap.parse_args()

OUT = os.path.abspath(H(A.out_dir))
for bad in ['~/rsna_train', '~/fleet_runs', '~/fleet', '~/bigjump_build', '~/fleet_store']:
    b = os.path.abspath(H(bad))
    assert not (OUT == b or OUT.startswith(b + os.sep)), f'refusing to write into {b}'
os.makedirs(OUT, exist_ok=True)


def rd(p):
    return pd.read_csv(H(p), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')


def sha(p):
    h = hashlib.sha256()
    with open(H(p), 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()[:16]


def lg(p):
    p = np.clip(np.asarray(p, float), A.eps, 1 - A.eps)
    return np.log(p / (1 - p))


def ex(z):
    return 1.0 / (1.0 + np.exp(-z))


base = rd(A.base)
assert list(base.columns[:12]) == T or set(T) <= set(base.columns), 'base schema'
base = base[T].astype(float)
assert not base.isna().any().any(), 'base has NaN'
folds = rd(A.folds)['fold']
excl = set(rd(A.exclude).index) if A.exclude else set()
held = set(folds.index[folds < 0]) | excl
have = {s for s in base.index if os.path.exists(H(f'{A.store}/{s}.npz'))}
TRAIN = [s for s in base.index if s in folds.index and s not in held and s in have]   # every row any fold trains on
print(f'[r2] base {len(base)} rows | held {len(held)} | store {len(have)} | training rows {len(TRAIN)} | '
      f'folds {folds[TRAIN].value_counts().sort_index().to_dict()}', flush=True)


# ---------------- OOF discovery ----------------
def closing(run_dir):
    lp = run_dir.rstrip('/') + '.log'
    if not os.path.exists(lp):
        return None
    with open(lp, 'rb') as f:
        f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 4096))
        lines = [x.strip() for x in f.read().decode('utf-8', 'replace').splitlines() if x.strip()]
    if not lines:
        return None
    try:
        j = json.loads(lines[-1])
    except Exception:
        return None
    return j if isinstance(j, dict) and 'raw' in j and 'ema' in j else None


RUNS = []   # (set_idx, run_dir, fold, seed, tag, oof_path, closing)
for si, g in enumerate(A.sets):
    dirs = sorted(d for d in glob.glob(H(g)) if os.path.isdir(d))
    for d in dirs:
        m = re.match(r'^f(\d+)_s(\d+)', os.path.basename(d))
        if not m:
            print(f'[r2] skip {d}: name not f<fold>_s<seed>*'); continue
        F, S = int(m.group(1)), int(m.group(2))
        cl = closing(d)
        if cl is None:
            print(f'[r2] skip {d}: no closing JSON line in {d}.log (run unfinished or crashed)'); continue
        tag = 'raw' if cl['raw'] >= cl['ema'] else 'ema'
        p = f'{d}/oof_f{F}_s{S}_{tag}.csv'
        if not os.path.exists(p):
            alt = 'ema' if tag == 'raw' else 'raw'; p2 = f'{d}/oof_f{F}_s{S}_{alt}.csv'
            assert os.path.exists(p2), f'no OOF csv in {d}'
            print(f'[r2] WARN {d}: best tag {tag} has no OOF csv, using {alt}'); tag, p = alt, p2
        RUNS.append((si, d, F, S, tag, p, cl))
        print(f'[r2] set{si} {os.path.basename(d)} fold {F} seed {S} best tag {tag} '
              f'(raw {cl["raw"]:.4f} ema {cl["ema"]:.4f}) oof sha {sha(p)}', flush=True)
assert RUNS, 'no finished runs found'

TS = set(TRAIN)
Z_sets = []        # per set: DataFrame of z-logits (rows covered by the set)
cover_by_set = []
for si in range(len(A.sets)):
    parts = []
    for (s_i, d, F, S, tag, p, cl) in RUNS:
        if s_i != si:
            continue
        o = rd(p)[T].astype(float)
        assert not o.isna().any().any(), f'{p} has NaN'
        bad_held = set(o.index) & held
        assert not bad_held, f'{p}: {len(bad_held)} held-out/gold rows in OOF'
        fo = folds.reindex(o.index)
        assert fo.notna().all(), f'{p}: rows missing from folds file'
        wrong = int((fo != F).sum())
        assert wrong == 0, f'{p}: {wrong} rows NOT in fold {F} -> not out-of-fold'
        want = {s for s in TRAIN if folds[s] == F}
        miss = want - set(o.index)
        if miss:
            print(f'[r2] WARN {p}: {len(miss)} fold-{F} training rows missing from OOF')
        L = lg(o.values)
        Z = (L - L.mean(0)) / L.std(0)          # per-fold, per-label standardisation (fold calibration removed)
        parts.append((F, pd.DataFrame(Z, index=o.index, columns=T)))
    if not parts:
        continue
    # several runs of the same fold inside one set (e.g. two seeds): average them
    byF = {}
    for F, z in parts:
        byF.setdefault(F, []).append(z)
    zs = []
    for F, lst in sorted(byF.items()):
        if len(lst) > 1:
            print(f'[r2] set{si} fold {F}: averaging {len(lst)} runs')
        zz = sum(x.loc[lst[0].index] for x in lst) / len(lst)
        zz = (zz - zz.mean()) / zz.std(ddof=0)
        zs.append(zz)
    Zs = pd.concat(zs)
    assert Zs.index.is_unique, f'set{si}: a study appears in two folds'
    Z_sets.append(Zs); cover_by_set.append(sorted(byF))
    print(f'[r2] set{si}: folds {sorted(byF)} | rows {len(Zs)}', flush=True)

COV = sorted(set.intersection(*[set(z.index) for z in Z_sets]) & TS)
UNCOV = sorted(TS - set(COV))
mode = 'FULL'
if UNCOV:
    if not A.partial:
        print(f'[r2] ABORT: {len(UNCOV)} of {len(TRAIN)} training rows have no OOF (folds covered '
              f'{cover_by_set}). Use --partial 1 only for a dry run.', flush=True)
        sys.exit(2)
    mode = 'PARTIAL'
print(f'[r2] mode {mode} | covered {len(COV)}/{len(TRAIN)} training rows', flush=True)

Z = sum(z.loc[COV] for z in Z_sets) / len(Z_sets)
Z = (Z - Z.mean()) / Z.std(ddof=0)
LB = lg(base.loc[COV].values)
MU, SD = LB.mean(0), LB.std(0)
OOFM = pd.DataFrame(ex(Z.values * SD + MU), index=COV, columns=T)          # oof mapped onto base logit mean/std
OOFRAW = None
if len(Z_sets) == 1 and len(A.sets) == 1:   # raw OOF probabilities, for the Spearman diagnostics only
    _r = pd.concat([rd(p)[T] for (_, d, F, S, tag, p, cl) in RUNS])
    OOFRAW = _r.loc[COV] if _r.index.is_unique else None

# ---------------- silence ----------------
if A.verdicts and os.path.exists(H(A.verdicts)):
    vr = rd(A.verdicts)
    VER = pd.DataFrame({l: vr[l + '__verdict'] for l in T}).reindex(base.index)
    SIL = (VER == 'UNK')                     # missing verdict row -> treated as spoken
    sil_src = f'verdict UNK in {A.verdicts}'
else:
    SIL = pd.DataFrame(False, index=base.index, columns=T)
    for l in T:
        mode_v = base[l].round(4).value_counts().idxmax()
        SIL[l] = base[l].round(4) == mode_v
    sil_src = 'base tie mode (no verdict file)'
print(f'[r2] silence from {sil_src}; silent fraction on covered rows '
      f'{SIL.loc[COV].values.mean():.3f}', flush=True)


def parse_var(v):
    m = re.fullmatch(r'a(\d+)(?:s(\d+))?', v)
    assert m, f'bad variant {v}'
    a = int(m.group(1)) / 100.0
    ws = (1 - a) if m.group(2) is None else int(m.group(2)) / 100.0
    return a, ws


proxies = {os.path.basename(p): rd(p)[T].astype(float) for p in A.proxy if os.path.exists(H(p))}
for p in A.proxy:
    if not os.path.exists(H(p)):
        print(f'[r2] WARN proxy {p} missing')
refs = {os.path.basename(p): rd(p)[T].astype(float) for p in A.ref if os.path.exists(H(p))}


def sp(a, b):
    r = spearmanr(np.asarray(a, float), np.asarray(b, float)).correlation
    return float(r) if np.isfinite(r) else float('nan')


def msp(X, Y, rows):
    return float(np.nanmean([sp(X.loc[rows, l], Y.loc[rows, l]) for l in T]))


manifest = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'mode': mode, 'base': A.base, 'base_sha': sha(A.base),
            'folds_sha': sha(A.folds), 'sets': A.sets,
            'runs': [{'set': s_i, 'dir': d, 'fold': F, 'seed': S, 'tag': tag, 'oof': p, 'oof_sha': sha(p), 'closing': cl}
                     for (s_i, d, F, S, tag, p, cl) in RUNS],
            'n_train_rows': len(TRAIN), 'n_covered': len(COV), 'n_uncovered_keep_base': len(UNCOV),
            'silence': sil_src, 'eps': A.eps, 'mapping': 'per-fold per-label z of logit -> base logit mean/std on covered rows',
            'variants': {}}
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)

# reference parent-likeness on the covered rows (same rows as the targets)
ref_lines = []
for pn, P in proxies.items():
    rows = [s for s in COV if s in P.index]
    ref_lines.append(f'  vs {pn} on {len(rows)} covered rows: base {msp(base, P, rows):.4f}'
                     + ''.join(f' | {rn} {msp(R, P, rows):.4f}' for rn, R in refs.items() if set(rows) <= set(R.index))
                     + f' | OOF {msp(OOFRAW if OOFRAW is not None else OOFM, P, rows):.4f}')
print('[r2] REFERENCE parent-likeness (mean per-label Spearman):'); print('\n'.join(ref_lines))
manifest['reference'] = ref_lines

prefix = 'DRY_' if mode == 'PARTIAL' else ''
for v in [x.strip() for x in A.variants.split(',') if x.strip()]:
    a, ws = parse_var(v)
    W = np.where(SIL.loc[COV].values, ws, 1 - a)          # OOF weight per cell
    tgt = base.copy()
    tgt.loc[COV, T] = (1 - W) * base.loc[COV].values + W * OOFM.values
    assert not tgt.isna().any().any() and (tgt.values >= 0).all() and (tgt.values <= 1).all()
    assert len(tgt) == len(base) and (tgt.index == base.index).all()
    name = f'{prefix}labels_r2_{v}'
    path = f'{OUT}/{name}.csv'
    tgt.reset_index()[['StudyInstanceUID'] + T].to_csv(path, index=False, float_format='%.6f')
    rec = []
    for l in T:
        r = {'label': l, 'n_cov': len(COV), 'silent': float(SIL.loc[COV, l].mean()),
             'mean_base': float(base.loc[COV, l].mean()), 'mean_oofm': float(OOFM[l].mean()),
             'mean_tgt': float(tgt.loc[COV, l].mean()),
             'sp_t_base': sp(tgt.loc[COV, l], base.loc[COV, l]), 'sp_t_oof': sp(tgt.loc[COV, l], OOFM[l])}
        for rn, R in refs.items():
            r[f'sp_t_{rn[:12]}'] = sp(tgt.loc[COV, l], R.loc[COV, l])
        for pn, P in proxies.items():
            rows = [s for s in COV if s in P.index]
            r[f'PL_tgt'] = sp(tgt.loc[rows, l], P.loc[rows, l])
            r[f'PL_base'] = sp(base.loc[rows, l], P.loc[rows, l])
            r[f'PL_oof'] = sp(OOFM.loc[rows, l], P.loc[rows, l])
            for rn, R in refs.items():
                r[f'PL_{rn[:12]}'] = sp(R.loc[rows, l], P.loc[rows, l])
            break                                   # per-label table uses the first proxy
        rec.append(r)
    D = pd.DataFrame(rec).set_index('label')
    macro = D.mean(numeric_only=True).to_dict()
    PLs = {pn: msp(tgt, P, [s for s in COV if s in P.index]) for pn, P in proxies.items()}
    print(f'\n[r2] === {name}  (a={a:.2f} spoken base weight, silent OOF weight {ws:.2f}) -> {path}')
    if A.per_label:
        print(D.round(3).to_string())
    print(f'[r2] MACRO {name}: ' + ' '.join(f'{k} {val:.4f}' for k, val in macro.items() if k != 'n_cov'))
    print(f'[r2] PARENT-LIKENESS of target {name}: ' + ' '.join(f'{pn} {x:.4f}' for pn, x in PLs.items()), flush=True)
    manifest['variants'][v] = {'file': path, 'sha': sha(path), 'a': a, 'w_silent': ws, 'macro': macro,
                               'parent_likeness': PLs, 'per_label': D.reset_index().to_dict('records')}
mp = f'{OUT}/{prefix}r2_manifest_{"_".join(re.sub(r"[^A-Za-z0-9]", "", os.path.basename(s)) for s in A.sets)}.json'
with open(mp, 'w') as f:
    json.dump(manifest, f, indent=1, default=float)
print(f'\n[r2] manifest {mp}\n[r2] done {mode}', flush=True)
