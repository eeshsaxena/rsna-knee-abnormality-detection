#!/usr/bin/env python
# v224_check.py (2026-10-14, Claude subagent): commit gate for the V224 copies built by build_v224.py. Reads a fetched kernel
# output dir (kaggle kernels output eeshsaxena/rsna-sub-v224 -p OUT). Prints V224_CHECK PASS/FAIL n/m; exit 0 only on PASS.
#   usage: python v224_check.py OUT [--flip --ref OUT_EXACT] [--control] [--her HER_OUT] [--sample sample_submission.csv]
#   default   V224-exact: clean canary, no flip, r224 used, submission == independent recompute (max diff <= 1e-6).
#   --flip    V224-FLIP: also FLIP_TTA_FORCED (notebook log) + FLIP_TTA_READY mcl_plain_pass_only (worker logs) + receipt flip
#             block; with --ref (the V224-exact commit output): R7 MCL equal within the compile tolerance 1e-3 and the stage-3 MCL
#             column equal within 1e-6, while at least 10 of the other 11 R7 columns move by more than 1e-3 in logit.
#   --control forced r224 failure: R224_STAGE rc!=0, CANARY_FIRED, Fracture == 0.5 exactly for every row, the other 11 columns ==
#             the recompute WITHOUT r224 (max diff <= 1e-6).
#   --her     compare with varshitha's published commit output (same 3 studies): stage csvs within the compile tolerance 1e-3,
#             r224 ranks and submission within 1e-6.
# Recompute (independent of the notebook's pandas rank): p(x) = average-tie 1-based rank / n over the rows a model did not fail on;
# rr = p(r384 raw); clean: rr = .65 rr + .35 p(r224 sub) over Raptor's non-failed rows; out = .5 rr + .5 p(R7) (per-class OA
# weights are .5 too); a Raptor-failed row takes R7's rank. NO re-rank of the mix (that is the V224 recipe).
import os, sys, json, glob, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--flip', action='store_true')
ap.add_argument('--control', action='store_true')
ap.add_argument('--ref', default=None, help='V224-exact commit output dir (for --flip)')
ap.add_argument('--her', default=None, help="varshitha's published commit output dir")
ap.add_argument('--sample', default=os.path.normpath(os.path.join(HERE, '..', '..', '_data', 'sample_submission.csv')))
A = ap.parse_args()
assert not (A.flip and A.control)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
R = []
TOL_BLEND, TOL_COMPILE = 1e-6, 1e-3
A224, W_R = 0.35, 0.5


def chk(name, ok, info=''):
    R.append(bool(ok))
    print(f'  {"OK  " if ok else "FAIL"} {name} {info}', flush=True)


def P(d, name):
    return os.path.join(d, name)


def avg_rank_pct(v):
    v = np.asarray(v, dtype=np.float64)
    order = np.argsort(v, kind='mergesort')
    sv = v[order]
    r = np.empty(len(v), dtype=np.float64)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r / len(v)


def pct_on(v, ok):
    r = np.full(len(v), np.nan)
    if ok.any():
        r[ok] = avg_rank_pct(np.asarray(v, dtype=np.float64)[ok])
    return r


def read_log(d):
    logs = [p for p in glob.glob(P(d, '*.log')) if 'cufile' not in os.path.basename(p)]
    if not logs:
        return '', None
    try:
        return '\n'.join(e.get('data', '') for e in json.load(open(logs[0], encoding='utf-8'))), logs[0]
    except Exception:
        return open(logs[0], encoding='utf-8', errors='replace').read(), logs[0]


def csv_(d, name):
    return pd.read_csv(P(d, name), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')


def find_key(o, key):
    if isinstance(o, dict):
        if key in o:
            return o[key]
        for v in o.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def line_json(text, marker):
    for ln in text.splitlines():
        i = ln.find(marker + ' ')
        if i >= 0:
            try:
                return json.loads(ln[i + len(marker) + 1:].strip())
            except Exception:
                return None
    return None


mode = 'control' if A.control else ('flip' if A.flip else 'exact')
print(f'V224_CHECK {A.out} mode {mode}', flush=True)
text, logp = read_log(A.out)
chk('kernel log present', bool(text), os.path.basename(logp) if logp else '')

# ---- log markers and their order (her cell order: Raptor r384 -> R7 wheels + assets -> R7 run -> r224 -> blend -> canary) ----
WHEEL = 'Successfully installed pydicom-3.0.1 timm-1.0.25'
order = ['RAPTOR_STAGE_DONE', WHEEL, 'ASSETS_READY', 'OURS_STAGE_COMPLETE', 'R224_STAGE', 'NOTEBOOK_COMPLETE',
         'CANARY_FIRED' if A.control else 'CANARY_CLEAN', 'V224_DONE']
if A.flip:
    order.insert(3, 'FLIP_TTA_FORCED')
if A.control:
    order.insert(4, 'CONTROL_R224_REPLACED')
pos = [text.find(m) for m in order]
chk('markers present', all(p >= 0 for p in pos), {m: p for m, p in zip(order, pos) if p < 0})
chk('markers in her cell order (R7 wheels before r224)', all(p >= 0 for p in pos) and pos == sorted(pos), order)
bad = ['R224_SKIPPED', 'R224_TIMEOUT', 'R224_UNUSABLE', 'BLEND_FALLBACK', 'Traceback', 'RuntimeError']
bad += ['CANARY_CLEAN'] if A.control else ['CANARY_FIRED', 'CONTROL_R224']
if not A.flip:
    bad.append('FLIP_TTA_FORCED')
hits = [m for m in bad if m in text]
chk('no fallback / error / foreign marker in the log', not hits, hits)
oc = line_json(text, 'OURS_STAGE_COMPLETE') or {}
chk('R7 stage PASS, 0 fallback studies', oc.get('status') == 'PASS' and oc.get('fallback_studies') == 0 and oc.get('scored') == oc.get('rows'),
    {k: oc.get(k) for k in ('status', 'rows', 'scored', 'fallback_studies', 'seconds')})
r224_line = next((ln for ln in text.splitlines() if 'R224_STAGE rc=' in ln), '')
if A.control:
    chk('control: R224_STAGE failed (rc != 0, ok False)', 'ok=False' in r224_line and 'rc=0 ' not in r224_line, r224_line.strip()[:80])
else:
    chk('R224_STAGE rc=0 ok=True', 'rc=0 ok=True' in r224_line, r224_line.strip()[:80])

# ---- receipts -----------------------------------------------------------------------------------------------------------
br = json.load(open(P(A.out, 'blend_receipt.json'), encoding='utf-8')) if os.path.isfile(P(A.out, 'blend_receipt.json')) else {}
chk('blend receipt: mode blend, 0 failed rows', br.get('mode') == 'blend' and br.get('raptor_failed') == 0 and br.get('ours_failed') == 0
    and br.get('both_failed') == 0, {k: br.get(k) for k in ('mode', 'rows', 'raptor_failed', 'ours_failed', 'both_failed', 'total_seconds')})
chk('blend receipt: weights .5/.5, OA .5', br.get('weights') == {'raptor': 0.5, 'ours': 0.5, 'raptor_per_class': {'Medial OA': 0.5, 'Lateral OA': 0.5, 'PF OA': 0.5}})
sp = br.get('spearman_raptor_raw_vs_sub') or {}
chk('r384 raw vs its sub Spearman >= .99', len(sp) == 12 and min(sp.values()) >= 0.99, min(sp.values()) if sp else None)
cr = json.load(open(P(A.out, 'canary_receipt.json'), encoding='utf-8')) if os.path.isfile(P(A.out, 'canary_receipt.json')) else {}
want_cr = ({'fired': ['r224->Fracture'], 'r224_used': False} if A.control else {'fired': [], 'r224_used': True})
chk('canary receipt', cr and all(cr.get(k) == v for k, v in want_cr.items()) and cr.get('use_f') is False
    and cr.get('flip_tta_forced') is bool(A.flip), cr)
sr = json.load(open(P(A.out, 'submission_receipt.json'), encoding='utf-8')) if os.path.isfile(P(A.out, 'submission_receipt.json')) else {}
wlogs = sorted(glob.glob(P(A.out, 'worker*_log.txt')))
wtext = {os.path.basename(p): open(p, encoding='utf-8', errors='replace').read() for p in wlogs}
ft = find_key(sr, 'flip_tta')
if A.flip:
    ok = bool(wtext) and all('FLIP_TTA_READY' in t and '"mcl_plain_pass_only": true' in t for t in wtext.values())
    chk('FLIP_TTA_READY + mcl_plain_pass_only in every worker log', ok, list(wtext))
    chk('R7 receipt carries the flip_tta block (MCL plain pass only)', isinstance(ft, dict) and ft.get('mcl') == 'plain pass only',
        {k: ft.get(k) for k in ('perm', 'mcl')} if isinstance(ft, dict) else ft)
else:
    chk('no flip in the R7 worker logs / receipt', bool(wtext) and not any('FLIP_TTA_READY' in t for t in wtext.values()) and not ft)

# ---- submission + independent recompute -----------------------------------------------------------------------------------
sample = pd.read_csv(A.sample, dtype={'StudyInstanceUID': str})
cols = list(sample.columns)
labels = cols[1:]
ids = sample['StudyInstanceUID'].tolist()
sub = pd.read_csv(P(A.out, 'submission.csv'), dtype={'StudyInstanceUID': str})
chk('submission columns == sample', list(sub.columns) == cols)
chk('submission ids == sample (order)', sub['StudyInstanceUID'].tolist() == ids, len(sub))
V = sub[labels].values.astype(np.float64)
chk('submission finite in [0,1]', np.isfinite(V).all() and (V >= 0).all() and (V <= 1).all())
ours, raw = csv_(A.out, 'ours_sub.csv').loc[ids, labels], csv_(A.out, 'raptor_raw.csv').loc[ids, labels]
has224 = os.path.isfile(P(A.out, 'raptor224_sub.csv'))
chk('r224 csv present' if not A.control else 'control: no r224 csv (the forced script wrote none)', has224 != A.control)
r224 = csv_(A.out, 'raptor224_sub.csv').loc[ids, labels] if (has224 and not A.control) else None
rap_fail = (raw.values == 0.5).all(axis=1)
our_fail = np.zeros(len(ids), dtype=bool)          # receipts above assert 0 R7 fallback studies
exp = np.zeros((len(ids), len(labels)))
for j, c in enumerate(labels):
    rr, ro = pct_on(raw[c].values, ~rap_fail), pct_on(ours[c].values, ~our_fail)
    if r224 is not None:
        rr = (1 - A224) * rr + A224 * pct_on(r224[c].values, ~rap_fail)
    b = W_R * rr + (1 - W_R) * ro
    b[rap_fail] = ro[rap_fail]
    b[our_fail] = rr[our_fail]
    b[rap_fail & our_fail] = 0.5
    exp[:, j] = b
fi = labels.index('Fracture')
if A.control:
    keep = [j for j in range(len(labels)) if j != fi]
    d = float(np.abs(V[:, keep] - exp[:, keep]).max())
    chk('control: 11 columns == recompute without r224', d <= TOL_BLEND, 'max diff %.3g' % d)
    chk('control: Fracture == 0.5 exactly in every row', bool((sub['Fracture'] == 0.5).all()), sub['Fracture'].tolist())
else:
    d = float(np.abs(V - exp).max())
    chk('submission == independent recompute (V224 recipe, no re-rank)', d <= TOL_BLEND, 'max diff %.3g' % d)
    chk('no blanked column (canary silent)', all(len(set(np.round(V[:, j], 7))) > 1 for j in range(len(labels))),
        [labels[j] for j in range(len(labels)) if len(set(np.round(V[:, j], 7))) <= 1])
    # the re-ranked variant (our HR7-RR rule) must NOT be what was written, when it differs at all
    if r224 is not None:
        exp_rr = np.zeros_like(exp)
        for j, c in enumerate(labels):
            mix = (1 - A224) * pct_on(raw[c].values, ~rap_fail) + A224 * pct_on(r224[c].values, ~rap_fail)
            exp_rr[:, j] = W_R * avg_rank_pct(mix) + (1 - W_R) * pct_on(ours[c].values, ~our_fail)
        drr = float(np.abs(exp_rr - exp).max())
        print(f'  info re-ranked (HR7-RR rule) vs V224 rule on these rows: max diff {drr:.3g}', flush=True)

# ---- flip vs the exact commit ---------------------------------------------------------------------------------------------
if A.flip:
    if A.ref:
        ref_ours = csv_(A.ref, 'ours_sub.csv').loc[ids, labels]
        ref_sub = pd.read_csv(P(A.ref, 'submission.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, labels]
        dm = float(np.abs(ours['MCL'].values - ref_ours['MCL'].values).max())
        chk('flip: R7 MCL == V224-exact R7 MCL within the compile tolerance', dm <= TOL_COMPILE, 'max diff %.3g' % dm)
        dms = float(np.abs(sub.set_index('StudyInstanceUID').loc[ids, 'MCL'].values - ref_sub['MCL'].values).max())
        chk('flip: submission MCL == V224-exact submission MCL', dms <= TOL_BLEND, 'max diff %.3g' % dms)
        def lg(v):
            v = np.clip(np.asarray(v, dtype=np.float64), 1e-12, 1 - 1e-12)
            return np.log(v / (1 - v))
        # logit scale: R7 probabilities are often saturated (1e-6 .. 1 - 1e-6), where a real logit move is < 1e-4 in probability
        dcol = {c: float(np.abs(lg(ours[c].values) - lg(ref_ours[c].values)).max()) for c in labels if c != 'MCL'}
        moved = [c for c, v in dcol.items() if v > 1e-3]
        chk('flip: >= 10 of the other 11 R7 columns move by > 1e-3 in logit', len(moved) >= 10,
            {c: round(v, 4) for c, v in dcol.items()})
        dsub = {c: float(np.abs(sub.set_index('StudyInstanceUID').loc[ids, c].values - ref_sub[c].values).max()) for c in labels}
        print('  info submission vs V224-exact per column max diff', {c: round(v, 4) for c, v in dsub.items()}, flush=True)
        g, gr = (sr.get('per_study_mean') or {}).get('gpu_s'), None
        if os.path.isfile(P(A.ref, 'submission_receipt.json')):
            gr = (json.load(open(P(A.ref, 'submission_receipt.json'), encoding='utf-8')).get('per_study_mean') or {}).get('gpu_s')
        print(f'  info R7 per-study gpu_s flip {g} vs exact {gr} (ratio {g / gr if g and gr else None})', flush=True)
    else:
        chk('flip: --ref V224-exact output given', False)

# ---- vs varshitha's own published commit -----------------------------------------------------------------------------------
if A.her:
    H = A.her
    hs = pd.read_csv(P(H, 'submission.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, labels]
    me = sub.set_index('StudyInstanceUID').loc[ids, labels]
    ds = float(np.abs(me.values - hs.values).max())
    if not A.flip and not A.control:
        chk('her commit: submission.csv equal (3 rows, rank values)', ds <= TOL_BLEND, 'max diff %.3g, cells differing %d of 36'
            % (ds, int((np.abs(me.values - hs.values) > TOL_BLEND).sum())))
    else:
        print(f'  info her commit vs this {mode} submission: max diff {ds:.3g}', flush=True)
    for nm, tol in (('ours_sub.csv', TOL_COMPILE), ('raptor_raw.csv', TOL_COMPILE), ('raptor224_sub.csv', TOL_BLEND)):
        if not os.path.isfile(P(A.out, nm)):
            continue
        a, b = csv_(A.out, nm).loc[ids, labels], csv_(H, nm).loc[ids, labels]
        dd = float(np.abs(a.values - b.values).max())
        if nm == 'ours_sub.csv' and A.flip:
            print(f'  info her commit {nm}: max diff {dd:.3g} (flip run, not gated)', flush=True)
            continue
        chk(f'her commit: {nm} within {tol:g}', dd <= tol, 'max diff %.3g' % dd)

n_ok = sum(R)
print(f'V224_CHECK {"PASS" if all(R) else "FAIL"} {n_ok}/{len(R)} mode {mode}', flush=True)
sys.exit(0 if all(R) else 1)
