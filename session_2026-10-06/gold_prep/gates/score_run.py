#!/usr/bin/env python3
# score_run.py (2026-10-06 ~22:30 IST, Claude; gold_prep gate scorers). CPU ONLY: CUDA_VISIBLE_DEVICES is forced to ''
# before numpy/torch load, torch is used only to read a checkpoint's metadata (map_location cpu, mmap). Reads run dirs,
# label files and logs; writes nothing except --json / stdout.
"""Score a finished fleet run dir (its <run>.log ends with the trainer's closing line {"raw": .., "ema": ..}) and apply
the pre-registered gate for it.

Per run it reports:
  * best tag (raw / ema): the larger closing value, raw on a tie (= the chains' max(raw, ema));
  * OOF macro AUC vs steven_v4 (llm_labels_v4_blend.csv) on the fold's val studies, recomputed with a VERBATIM copy of
    fleet_train.macro_auc (bigjump_build/code and fast_build/code2 copies are identical, diffed 10-06) and checked
    against (a) the closing line at full precision, (b) the best checkpoint's stored cv, (c) the 4-dp val AUC printed on
    the best epoch's log line;
  * gold58 and gold74 macro AUC from the log line of the best epoch (best epoch = the best checkpoint's 'epoch' field;
    fallback: the epoch whose saved epochs/ OOF equals the best OOF; last fallback: the log walk);
  * mean per-label Spearman of the OOF vs the Raptor proxy (raptor_proxy_v5v8.csv) on the same studies (parent-likeness);
  * mean per-label Spearman vs a reference run's best OOF on the same fold (member diversity; default ref = J1);
  * per-label AUCs (run and ref) and per-label Spearmans.

Gates (pre-registered 10-06, before either run finished; values compared at full precision):
  t288: f0_s1044_ttr288mix50 vs J1: OOF >= .8876
  r50 : f0_s243_r50mix50     vs J1: OOF >= .8786 AND Spearman vs Raptor proxy <= .87 AND gold58 >= .905
                                    AND Spearman vs J1 OOF <= .90

Usage (box):
  CUDA_VISIBLE_DEVICES= nice -n 19 ~/miniconda3/envs/rsna/bin/python ~/gold_prep/gates/score_run.py --gate t288
  ... score_run.py --gate r50
  ... score_run.py f0_s42 f0_s1042_ttr f0_s142_f2v4 --ref f0_s1043_ttr_mix50       (table of finished runs)
  --partial scores a run that is still training (its current best files), marked PROVISIONAL; gate verdict = PENDING.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '2')
import argparse, glob, json, re, sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

H = os.path.expanduser('~')
HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
           'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
J1 = 'f0_s1043_ttr_mix50'
DEFAULTS = dict(
    runs_root=f'{H}/fleet_runs',
    eval_labels=f'{H}/rsna_train/llm_labels_v4_blend.csv',
    folds=f'{H}/rsna_train/folds5.csv',
    gold74=f'{H}/rsna_train/gold74_eval_set.csv',
    gold58=f'{H}/kmimic/ids/gold58.csv',
    proxy=f'{HERE}/raptor_proxy_v5v8.csv',
)
# Pre-registered 10-06 (task text + RESEARCH_TO_TOP_10-06.md lever 3/5). (metric, op, threshold, why)
GATES = {
    't288': dict(run='f0_s1044_ttr288mix50', ref=J1, title='T288: ConvNeXt TTR 288 px, mix50, fold 0, seed 1044',
                 checks=[('oof', '>=', 0.8876, 'best OOF vs steven_v4 >= J1 .8836 + .004')]),
    'r50': dict(run='f0_s243_r50mix50', ref=J1, title='R50: ResNet-50 family probe, mix50, fold 0, seed 243',
                checks=[('oof', '>=', 0.8786, 'best OOF vs steven_v4 >= J1 .8836 - .005'),
                        ('rho_proxy', '<=', 0.87, 'parent-likeness: mean Spearman vs Raptor proxy <= .87'),
                        ('gold58', '>=', 0.905, 'gold58 (best-epoch log line) >= .905'),
                        ('rho_ref', '<=', 0.90, 'diversity: mean Spearman vs J1 OOF <= .90')]),
}


class RunError(Exception):
    pass


# ---------------------------------------------------------------- trainer metric (VERBATIM from fleet_train.py) ----
def macro_auc(P, ids, tab):
    # T9 NaN labels/preds masked per target (draft: NaN label -> negative, NaN pred -> crash)
    Y = tab.loc[ids, TARGETS].to_numpy(np.float64); aucs = []
    for j in range(len(TARGETS)):
        ok = np.isfinite(Y[:, j]) & np.isfinite(P[:, j]); yb = Y[ok, j] >= 0.5
        if ok.sum() and 0 < yb.mean() < 1:
            aucs.append(roc_auc_score(yb, P[ok, j]))
    return (float(np.mean(aucs)) if aucs else float('nan')), len(aucs)


def label_aucs(P, ids, tab):
    """Per-label AUC with the same masking/skip rule as macro_auc (nan where the trainer would skip the label)."""
    Y = tab.loc[ids, TARGETS].to_numpy(np.float64); out = []
    for j in range(len(TARGETS)):
        ok = np.isfinite(Y[:, j]) & np.isfinite(P[:, j]); yb = Y[ok, j] >= 0.5
        out.append(float(roc_auc_score(yb, P[ok, j])) if ok.sum() and 0 < yb.mean() < 1 else float('nan'))
    return out


def spearman_labels(A, B, ids):
    """Per-label Spearman (pandas, as fleet_g1b.py) of two prediction frames on the given studies."""
    return [float(A.loc[ids, c].astype(np.float64).corr(B.loc[ids, c].astype(np.float64), method='spearman'))
            for c in TARGETS]


# ---------------------------------------------------------------------------------------------- file helpers ----
def read_ids(path):
    return set(pd.read_csv(path, dtype={'StudyInstanceUID': str}).StudyInstanceUID)


def read_tab(path):
    return pd.read_csv(path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')


def read_oof(path):
    d = read_tab(path)
    miss = [c for c in TARGETS if c not in d.columns]
    if miss:
        raise RunError(f'{path}: missing columns {miss}')
    # the trainer wrote float32 sigmoid outputs; shortest-repr text -> float32 recovers them exactly
    return d[TARGETS].astype(np.float32)


EP_RE = re.compile(r'^ep (\d+) (raw|ema) \|(.*)$')
NF_RE = re.compile(r'^!! (raw|ema) produced non-finite predictions/AUC at ep (\d+)')


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float('nan')


def parse_log(path):
    """-> (closing dict or None, {tag: {ep: rec}}, non-finite set, n lines)."""
    with open(path, errors='replace') as f:
        lines = [ln.rstrip('\r\n') for ln in f]
    eps = {'raw': {}, 'ema': {}}; nonfin = set()
    for ln in lines:
        m = EP_RE.match(ln)
        if m:
            ep, tag, rest = int(m.group(1)), m.group(2), m.group(3)
            rec = dict(ep=ep, line=ln)
            g = re.search(r'\| gold74 (\S+) \((\d+) labels\)', ln); rec['gold74'] = _num(g.group(1)) if g else float('nan')
            g = re.search(r'\| gold58 (\S+) \((\d+) labels\)', ln); rec['gold58'] = _num(g.group(1)) if g else float('nan')
            g = re.search(r'\| val AUC vs (\S+) (\S+) \((\d+) labels\)', ln)
            rec['sel_name'], rec['cv_str'] = (g.group(1), g.group(2)) if g else (None, None)
            rec['cv'] = _num(rec['cv_str']); rec['n_labels'] = int(g.group(3)) if g else 0
            g = re.search(r'\| vs train labels (\S+)', ln); rec['cv_train'] = _num(g.group(1)) if g else float('nan')
            eps[tag][ep] = rec
            continue
        m = NF_RE.match(ln)
        if m:
            nonfin.add((m.group(1), int(m.group(2))))
    closing = None
    for ln in reversed(lines):
        if not ln.strip():
            continue
        try:
            d = json.loads(ln)
            if isinstance(d, dict) and 'raw' in d and 'ema' in d:
                closing = {k: float(d[k]) for k in ('raw', 'ema')}
        except ValueError:
            pass
        break
    return closing, eps, nonfin, len(lines)


def top_files(run_dir, tag):
    oofs = [p for p in glob.glob(os.path.join(run_dir, f'oof_f*_s*_{tag}.csv'))
            if re.match(rf'^oof_f(-?\d+)_s(\d+)_{tag}\.csv$', os.path.basename(p))]
    ckpts = [p for p in glob.glob(os.path.join(run_dir, f'f*_s*_{tag}_*_best.pt'))
             if re.match(rf'^f(-?\d+)_s(\d+)_{tag}_.*_best\.pt$', os.path.basename(p))]
    return oofs, ckpts


def ckpt_meta(path):
    """Read epoch / cv / args from a trainer checkpoint without materialising the weights (mmap, CPU)."""
    import torch
    torch.set_num_threads(1)
    d = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    a = d.get('args', {}) or {}
    return dict(epoch=int(d['epoch']), cv=float(d['cv']),
                train_labels=a.get('train_labels'), sel_labels=a.get('sel_labels'),
                model_mod=a.get('model_mod', 'fleet_model'), backbone=a.get('backbone'), ttr_size=a.get('ttr_size'),
                seed=a.get('seed'), fold=a.get('fold'), epochs=a.get('epochs'), lr=a.get('lr'), aug=a.get('aug'))


def resolve(run, root):
    p = run if os.path.isabs(run) or run.startswith('.') or os.sep in run.rstrip(os.sep) else os.path.join(root, run)
    return os.path.abspath(os.path.expanduser(p.rstrip('/')))


# ----------------------------------------------------------------------------------------------- core loader ----
def load_run(run_dir, ctx, partial=False):
    """Load a run's best-tag OOF + metadata. Raises RunError when the run is not scorable."""
    name = os.path.basename(run_dir); log = run_dir + '.log'
    if not os.path.isdir(run_dir):
        raise RunError(f'{name}: run dir {run_dir} does not exist (not started?)')
    if not os.path.isfile(log):
        raise RunError(f'{name}: log {log} missing')
    closing, eps, nonfin, nlines = parse_log(log)
    finished = closing is not None
    if not finished and not partial:
        last = max([max(e) for e in eps.values() if e], default=None)
        raise RunError(f'{name}: NOT FINISHED (no closing {{"raw","ema"}} line; last evaluated epoch {last}). '
                       f'Use --partial for a provisional read.')
    R = dict(name=name, dir=run_dir, finished=finished, closing=closing, notes=[], tags={})
    for tag in ('raw', 'ema'):
        T = dict()
        for attempt in range(3):     # a live trainer replaces its best files (remove + save): retry a torn read
            try:
                oofs, ckpts = top_files(run_dir, tag)
                if len(oofs) != 1:
                    raise RunError(f'{name}/{tag}: expected 1 top-level OOF csv, found {len(oofs)}')
                m = re.match(rf'^oof_f(-?\d+)_s(\d+)_{tag}\.csv$', os.path.basename(oofs[0]))
                T.update(oof=oofs[0], fold=int(m.group(1)), seed=int(m.group(2)))
                T['P'] = read_oof(oofs[0])
                T['ckpt'] = ckpts[0] if len(ckpts) == 1 else None
                T['meta'] = None
                if T['ckpt']:
                    try:
                        T['meta'] = ckpt_meta(T['ckpt'])
                    except Exception as e:    # metadata is a cross-check, not required
                        R['notes'].append(f'{tag}: checkpoint metadata unreadable ({type(e).__name__}: {e})')
                elif finished:
                    R['notes'].append(f'{tag}: {len(ckpts)} top-level best checkpoints (expected 1)')
                break
            except (RunError, FileNotFoundError, OSError, pd.errors.EmptyDataError) as e:
                if finished or attempt == 2:
                    if isinstance(e, RunError):
                        raise
                    raise RunError(f'{name}/{tag}: {type(e).__name__}: {e}')
                time.sleep(5)
        ids = list(T['P'].index)
        T['auc'], T['n_labels'] = macro_auc(T['P'].to_numpy(np.float32), ids, ctx['v4'])
        R['tags'][tag] = T
    # best tag
    if finished:
        R['best_tag'] = 'raw' if closing['raw'] >= closing['ema'] else 'ema'
    else:
        R['best_tag'] = 'raw' if R['tags']['raw']['auc'] >= R['tags']['ema']['auc'] else 'ema'
        R['notes'].append('PROVISIONAL: run still training; best tag/epoch can still change')
    bt = R['best_tag']; B = R['tags'][bt]
    R['fold'], R['seed'] = B['fold'], B['seed']
    if R['tags']['raw']['fold'] != R['tags']['ema']['fold']:
        raise RunError(f'{name}: raw/ema OOF folds differ')
    # --- reproduction checks
    rep = {}
    for tag, T in R['tags'].items():
        r = dict(recomputed=T['auc'])
        if finished:
            r['closing'] = closing[tag]; r['diff_closing'] = T['auc'] - closing[tag]
            r['ok_closing'] = bool(abs(r['diff_closing']) < 1e-9)
        if T['meta'] is not None:
            r['ckpt_cv'] = T['meta']['cv']; r['diff_ckpt'] = T['auc'] - T['meta']['cv']
            r['ok_ckpt'] = bool(abs(r['diff_ckpt']) < 1e-9)
        rep[tag] = r
    R['repro'] = rep
    # --- best epoch: ckpt epoch > epochs/ OOF match > log walk
    ep_best, how = None, None
    if B['meta'] is not None:
        ep_best, how = B['meta']['epoch'], 'checkpoint epoch field'
    if ep_best is None:
        target = B['P']
        for p in sorted(glob.glob(os.path.join(run_dir, 'epochs', f'oof_f{R["fold"]}_s{R["seed"]}_{bt}_ep*.csv'))):
            mm = re.search(r'_ep(\d+)\.csv$', p)
            try:
                Q = read_oof(p)
            except Exception:
                continue
            if Q.index.equals(target.index) and np.array_equal(Q.to_numpy(), target.to_numpy()):
                ep_best, how = int(mm.group(1)) - 1, f'epochs/ OOF identical ({os.path.basename(p)})'
                break
    best_val = closing[bt] if finished else B['auc']
    cand = [e for e, rec in sorted(eps[bt].items()) if rec['cv_str'] == f'{best_val:.4f}' and (bt, e) not in nonfin]
    if ep_best is None:
        if len(cand) == 1:
            ep_best, how = cand[0], 'log walk (unique epoch printing the best value)'
        elif cand:
            ep_best, how = cand[0], f'log walk AMBIGUOUS among epochs {cand}; first taken'
            R['notes'].append(f'best epoch ambiguous from the log: {cand}')
    R['best_epoch'], R['best_epoch_how'], R['log_candidates'] = ep_best, how, cand
    rec = eps[bt].get(ep_best) if ep_best is not None else None
    if rec is None:
        R['notes'].append(f'no log line for {bt} ep {ep_best}')
        R['gold58'] = R['gold74'] = float('nan'); R['logline_ok'] = False
    else:
        R['gold58'], R['gold74'] = rec['gold58'], rec['gold74']
        R['logline_ok'] = rec['cv_str'] == f'{best_val:.4f}'
        if not R['logline_ok']:
            R['notes'].append(f'log line ep {ep_best} prints val {rec["cv_str"]} != best {best_val:.4f}')
        if rec['sel_name'] and os.path.basename(ctx['eval_labels']) != rec['sel_name']:
            R['notes'].append(f'run selected on {rec["sel_name"]}, scorer uses {os.path.basename(ctx["eval_labels"])}')
        R['cv_train_best'] = rec['cv_train']
    R['n_epochs_logged'] = len(eps[bt])
    # all-epoch gold58 trajectory of the best tag (info)
    R['gold58_max'] = max([r['gold58'] for r in eps[bt].values() if np.isfinite(r['gold58'])], default=float('nan'))
    # --- val-set identity check
    folds = ctx['folds']; excl = ctx['gold74_ids']
    exp = {s for s in folds.index if folds[s] == R['fold']} - excl - set(folds.index[folds < 0])
    got = set(B['P'].index)
    R['val_check'] = dict(n=len(got), expected=len(exp), ok=got == exp, missing=len(exp - got), extra=len(got - exp))
    if not R['val_check']['ok']:
        R['notes'].append(f'val ids differ from folds5 fold {R["fold"]} minus gold74: '
                          f'{R["val_check"]["missing"]} missing, {R["val_check"]["extra"]} extra')
    meta = B['meta'] or {}
    R['meta'] = {k: meta.get(k) for k in ('train_labels', 'sel_labels', 'model_mod', 'backbone', 'ttr_size',
                                          'epochs', 'lr', 'aug')}
    R['oof_auc'] = B['auc']; R['P'] = B['P']
    R['label_auc'] = label_aucs(B['P'].to_numpy(np.float32), list(B['P'].index), ctx['v4'])
    return R


def score(run_dir, ref, ctx, partial=False):
    R = load_run(run_dir, ctx, partial)
    ids = list(R['P'].index)
    px = ctx['proxy']; pid = [s for s in ids if s in px.index]
    R['n_proxy'] = len(pid)
    R['rho_proxy_l'] = spearman_labels(R['P'], px, pid)
    R['rho_proxy'] = float(np.mean(R['rho_proxy_l']))
    R['ref'] = None
    if ref is not None and ref['dir'] != R['dir']:
        if ref['fold'] != R['fold']:
            R['notes'].append(f'ref {ref["name"]} is fold {ref["fold"]}, run is fold {R["fold"]}: no diversity read')
        else:
            rid = [s for s in ids if s in ref['P'].index]
            R['ref'] = ref['name']; R['n_ref'] = len(rid)
            R['rho_ref_l'] = spearman_labels(R['P'], ref['P'], rid)
            R['rho_ref'] = float(np.mean(R['rho_ref_l']))
            R['ref_oof_auc'] = ref['oof_auc']; R['ref_gold58'] = ref['gold58']
            R['ref_label_auc'] = ref['label_auc']
            if len(rid) != len(ids):
                R['notes'].append(f'ref covers {len(rid)}/{len(ids)} of the run\'s val studies')
    return R


def apply_gate(R, gname):
    G = GATES[gname]; res = []
    if R['name'] != G['run']:
        R['notes'].append(f'gate {gname} is registered for {G["run"]}, applied to {R["name"]}')
    if R.get('ref') != G['ref']:
        R['notes'].append(f'gate {gname} is registered against ref {G["ref"]}, ref here is {R.get("ref")}')
    for metric, op, thr, why in G['checks']:
        v = R.get({'oof': 'oof_auc'}.get(metric, metric), float('nan'))
        v = float('nan') if v is None else float(v)
        ok = bool(np.isfinite(v) and (v >= thr if op == '>=' else v <= thr))
        res.append(dict(metric=metric, op=op, thr=thr, value=v, ok=ok, why=why))
    allok = all(r['ok'] for r in res)
    verdict = ('PASS' if allok else 'FAIL') if R['finished'] else f'PENDING (provisional {"PASS" if allok else "FAIL"})'
    R['gate'] = dict(name=gname, title=G['title'], checks=res, verdict=verdict)
    return R['gate']


# ------------------------------------------------------------------------------------------------ printing ----
def f4(x):
    return 'nan' if x is None or not np.isfinite(x) else f'{x:.4f}'


def f3(x):
    return 'nan' if x is None or not np.isfinite(x) else f'{x:.3f}'


def print_run(R, ctx, brief=False):
    hdr = f'=== {R["name"]}  (fold {R["fold"]}, seed {R["seed"]}' + (')' if R['finished'] else ', PROVISIONAL: still training)')
    print(hdr)
    c = R['closing']
    print(f'closing line   : {json.dumps(c) if c else "(none yet)"}')
    print(f'best tag       : {R["best_tag"]}   OOF macro AUC vs steven_v4 = {R["oof_auc"]:.6f} '
          f'({R["tags"][R["best_tag"]]["n_labels"]} labels, {R["val_check"]["n"]} val studies, '
          f'val-set check {"OK" if R["val_check"]["ok"] else "MISMATCH"})')
    for tag in ('raw', 'ema'):
        r = R['repro'][tag]
        s = f'  repro {tag:3s}    : recomputed {r["recomputed"]:.10f}'
        if 'closing' in r:
            s += f' | closing {r["closing"]:.10f} diff {r["diff_closing"]:+.1e} {"OK" if r["ok_closing"] else "MISMATCH"}'
        if 'ckpt_cv' in r:
            s += f' | ckpt cv diff {r["diff_ckpt"]:+.1e} {"OK" if r["ok_ckpt"] else "MISMATCH"}'
        print(s)
    print(f'best epoch     : ep {R["best_epoch"]} (0-based, = {None if R["best_epoch"] is None else R["best_epoch"] + 1}'
          f'/{R["meta"].get("epochs") or "?"}) via {R["best_epoch_how"]}; log line val prints the best value: '
          f'{"yes" if R["logline_ok"] else "NO"}')
    print(f'gold58 / gold74: {f4(R["gold58"])} / {f4(R["gold74"])}  (best-epoch log line; max gold58 over epochs '
          f'{f4(R["gold58_max"])}, never used to select)')
    print(f'parent-likeness: mean per-label Spearman vs Raptor proxy = {f4(R["rho_proxy"])} (n={R["n_proxy"]})')
    if R.get('ref'):
        print(f'diversity      : mean per-label Spearman vs {R["ref"]} OOF = {f4(R["rho_ref"])} (n={R["n_ref"]}); '
              f'ref OOF {f4(R["ref_oof_auc"])}, dOOF {R["oof_auc"] - R["ref_oof_auc"]:+.4f}')
    m = R['meta']
    if any(v is not None for v in m.values()):
        print(f'ckpt args      : train_labels {m["train_labels"]} | sel {m["sel_labels"]} | {m["model_mod"]} '
              f'{m["backbone"]} ttr_size {m["ttr_size"]} | lr {m["lr"]} aug {m["aug"]}')
    if not brief:
        print(f'  {"label":18s} {"AUC":>7s} {"refAUC":>7s} {"d":>7s} {"rhoPx":>6s} {"rhoRef":>6s}')
        for j, c in enumerate(TARGETS):
            ra = R['ref_label_auc'][j] if R.get('ref') else float('nan')
            d = R['label_auc'][j] - ra if R.get('ref') else float('nan')
            rr = R['rho_ref_l'][j] if R.get('ref') else float('nan')
            print(f'  {c:18s} {f4(R["label_auc"][j]):>7s} {f4(ra):>7s} '
                  f'{("nan" if not np.isfinite(d) else f"{d:+.4f}"):>7s} {f3(R["rho_proxy_l"][j]):>6s} {f3(rr):>6s}')
    for n in R['notes']:
        print(f'  note: {n}')
    if R.get('gate'):
        G = R['gate']
        print(f'--- gate {G["name"]}: {G["title"]}')
        for ch in G['checks']:
            print(f'    [{"PASS" if ch["ok"] else "FAIL"}] {ch["metric"]:9s} {f4(ch["value"])} {ch["op"]} {ch["thr"]:.4f}'
                  f'   ({ch["why"]})')
        print(f'GATE {G["name"].upper()} {R["name"]}: {G["verdict"]}')
    print()


def print_table(rows):
    print(f'{"run":28s} {"tag":3s} {"ep":>3s} {"OOF":>7s} {"dRef":>7s} {"gold58":>7s} {"rhoPx":>6s} {"rhoRef":>6s} '
          f'{"repro":5s} {"train_labels":24s} {"gate":s}')
    for R in rows:
        if 'error' in R:
            print(f'{R["name"]:28s} {R["error"]}')
            continue
        okr = all(r.get('ok_closing', True) and r.get('ok_ckpt', True) for r in R['repro'].values())
        d = R['oof_auc'] - R['ref_oof_auc'] if R.get('ref') else float('nan')
        g = R.get('gate', {}).get('verdict', '-')
        nm = R['name'] + ('' if R['finished'] else ' (prov)')
        ep = '?' if R['best_epoch'] is None else str(R['best_epoch'] + 1)
        print(f'{nm:28s} {R["best_tag"]:3s} {ep:>3s} {R["oof_auc"]:.4f} '
              f'{("   -   " if not np.isfinite(d) else f"{d:+.4f}"):>7s} {f4(R["gold58"]):>7s} '
              f'{f3(R["rho_proxy"]):>6s} {f3(R.get("rho_ref", float("nan"))):>6s} {"OK" if okr else "BAD":5s} '
              f'{str(R["meta"].get("train_labels")):24s} {g}')


def jsonable(R):
    skip = {'P', 'tags'}
    out = {k: v for k, v in R.items() if k not in skip}
    out['tags'] = {t: {k: v for k, v in T.items() if k not in ('P',)} for t, T in R.get('tags', {}).items()}
    return out


def build_ctx(a):
    v4 = read_tab(a.eval_labels)
    folds = read_tab(a.folds)['fold']
    g74 = read_tab(a.gold74)
    ctx = dict(v4=v4, folds=folds, gold74_ids=set(g74.index), eval_labels=a.eval_labels)
    if not os.path.isfile(a.proxy):
        raise SystemExit(f'Raptor proxy {a.proxy} missing (copy teacher_v2_out/raptor_proxy_v5v8.csv there)')
    ctx['proxy'] = read_tab(a.proxy)[TARGETS].astype(np.float64)
    # the trainer's gold58 = gold74 rows with source == original_gold58; check it equals the shared gold58 id list
    if 'source' in g74.columns and os.path.isfile(a.gold58):
        t58 = set(g74.index[g74['source'] == 'original_gold58'])
        ctx['gold58_consistent'] = t58 == read_ids(a.gold58)
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='*', help='run names under --runs_root, or run dirs')
    ap.add_argument('--gate', default='auto', choices=['auto', 'none'] + sorted(GATES),
                    help='auto = the registered gate for a run of that name; with no runs given, --gate X scores X\'s run')
    ap.add_argument('--ref', default=J1, help='reference run for the diversity Spearman (same fold), default J1')
    ap.add_argument('--partial', action='store_true', help='score a still-training run (provisional, gate PENDING)')
    ap.add_argument('--brief', action='store_true', help='no per-label table')
    ap.add_argument('--json', default='', help='write all results to this json file')
    for k, v in DEFAULTS.items():
        ap.add_argument(f'--{k}', default=v)
    a = ap.parse_args(argv)
    runs = list(a.runs)
    if not runs:
        if a.gate in GATES:
            runs = [GATES[a.gate]['run']]
        else:
            ap.error('give run names, or --gate t288|r50')
    ctx = build_ctx(a)
    print(f'scorer: eval labels {a.eval_labels} | proxy {a.proxy} | gold58 list consistent with the trainer\'s: '
          f'{ctx.get("gold58_consistent")}')
    ref = None
    if a.ref and a.ref.lower() != 'none':
        try:
            ref = load_run(resolve(a.ref, a.runs_root), ctx, partial=False)
            print(f'reference: {ref["name"]} best {ref["best_tag"]} OOF {ref["oof_auc"]:.4f} gold58 {f4(ref["gold58"])}')
        except RunError as e:
            print(f'reference unavailable: {e}')
    print()
    rows = []; rc = 0
    for r in runs:
        d = resolve(r, a.runs_root)
        try:
            R = score(d, ref, ctx, a.partial)
        except RunError as e:
            print(f'=== {os.path.basename(d)}: {e}\n')
            rows.append(dict(name=os.path.basename(d), error=str(e))); rc = max(rc, 3)
            continue
        g = a.gate if a.gate in GATES else next((k for k, G in GATES.items() if G['run'] == R['name']), None) \
            if a.gate == 'auto' else None
        if g:
            apply_gate(R, g)
            if R['gate']['verdict'] == 'FAIL':
                rc = max(rc, 1)
        print_run(R, ctx, a.brief)
        rows.append(R)
    if len(rows) > 1 or a.brief:
        print_table(rows)
    if a.json:
        with open(a.json, 'w') as f:
            json.dump([jsonable(R) for R in rows], f, indent=1, default=str)
    return rc


if __name__ == '__main__':
    sys.exit(main())
