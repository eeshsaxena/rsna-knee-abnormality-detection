#!/usr/bin/env python
# pl_check.py (2026-10-14): commit gate for build_pl.py kernels on the V224 / V224-FLIP base (no fleet).
# Reads a fetched kernel output dir (kaggle kernels output eeshsaxena/rsna-sub-pl-<name> -p OUT) and the builder's
# pl_map_resolved.json. Prints PL_CHECK PASS/FAIL n/m; exit 0 only on PASS.
#   usage: python pl_check.py OUT RESOLVED_JSON [--ref OUT_V224_EXACT] [--flip] [--sample sample_submission.csv]
# Checks: her markers in her order + PL_H_OK between NOTEBOOK_COMPLETE and CANARY_CLEAN; no fallback / error marker; R7 PASS with
# 0 fallback studies; r224 rc=0; blend receipt = her weights (PL-H does not touch her stage 3); pl_receipt = this map (sha,
# changed shares, r224 used); submission == independent recompute with the per-label shares (<= 1e-6); every unmapped column
# == the recompute at her .5 (and, with --ref, == the V224-exact commit's column when the component ranks agree); no blanked column.
import os, sys, json, glob, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('resolved')
ap.add_argument('--ref', default=None)
ap.add_argument('--flip', action='store_true')
ap.add_argument('--sample', default=os.path.normpath(os.path.join(HERE, '..', '..', '..', '_data', 'sample_submission.csv')))
A = ap.parse_args()
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
R = []
A224 = 0.35


def chk(name, ok, info=''):
    R.append(bool(ok))
    print(f'  {"OK  " if ok else "FAIL"} {name} {info}', flush=True)


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


def read_log(d):
    logs = [p for p in glob.glob(os.path.join(d, '*.log')) if 'cufile' not in os.path.basename(p)]
    if not logs:
        return '', None
    try:
        return '\n'.join(e.get('data', '') for e in json.load(open(logs[0], encoding='utf-8'))), logs[0]
    except Exception:
        return open(logs[0], encoding='utf-8', errors='replace').read(), logs[0]


def line_json(text, marker):
    for ln in text.splitlines():
        i = ln.find(marker + ' ')
        if i >= 0:
            try:
                return json.loads(ln[i + len(marker) + 1:].strip())
            except Exception:
                return None
    return None


res = json.load(open(A.resolved, encoding='utf-8'))
shares = {c: float(v['s']) for c, v in res['resolved'].items()}
assert not any(float(v['f']) > 0 for v in res['resolved'].values()), 'pl_check covers the no-fleet bases only'
print(f'PL_CHECK {A.out} map {res["name"]} sha {res["map_sha"]} changed {sorted(c for c in shares if shares[c] != 0.5)}', flush=True)
text, logp = read_log(A.out)
chk('kernel log present', bool(text), os.path.basename(logp) if logp else '')
WHEEL = 'Successfully installed pydicom-3.0.1 timm-1.0.25'
order = ['RAPTOR_STAGE_DONE', WHEEL, 'ASSETS_READY', 'OURS_STAGE_COMPLETE', 'R224_STAGE', 'NOTEBOOK_COMPLETE', 'PL_H_OK',
         'CANARY_CLEAN', 'V224_DONE']
if A.flip:
    order.insert(3, 'FLIP_TTA_FORCED')
pos = [text.find(m) for m in order]
chk('markers present', all(p >= 0 for p in pos), {m: p for m, p in zip(order, pos) if p < 0})
chk('markers in order (PL-H between her stage 3 and the canary)', all(p >= 0 for p in pos) and pos == sorted(pos))
bad = ['R224_SKIPPED', 'R224_TIMEOUT', 'R224_UNUSABLE', 'BLEND_FALLBACK', 'Traceback', 'RuntimeError', 'PL_SELF_CHECK_FAIL',
       'CANARY_FIRED', 'CONTROL_R224', 'PL_F_']
if not A.flip:
    bad.append('FLIP_TTA_FORCED')
hits = [m for m in bad if m in text]
chk('no fallback / error / foreign marker', not hits, hits)
oc = line_json(text, 'OURS_STAGE_COMPLETE') or {}
chk('R7 stage PASS, 0 fallback studies', oc.get('status') == 'PASS' and oc.get('fallback_studies') == 0 and oc.get('scored') == oc.get('rows'),
    {k: oc.get(k) for k in ('status', 'rows', 'scored', 'fallback_studies', 'seconds')})
r224_line = next((ln for ln in text.splitlines() if 'R224_STAGE rc=' in ln), '')
chk('R224_STAGE rc=0 ok=True', 'rc=0 ok=True' in r224_line, r224_line.strip()[:80])
def jl(n):
    p_ = os.path.join(A.out, n)
    return json.load(open(p_, encoding='utf-8')) if os.path.isfile(p_) else {}


br = jl('blend_receipt.json')
chk('her blend receipt unchanged: weights .5/.5, OA .5, 0 failed rows',
    br.get('weights') == {'raptor': 0.5, 'ours': 0.5, 'raptor_per_class': {'Medial OA': 0.5, 'Lateral OA': 0.5, 'PF OA': 0.5}}
    and br.get('mode') == 'blend' and br.get('raptor_failed') == 0 and br.get('ours_failed') == 0)
pr = jl('pl_receipt.json')
want_changed = {c: s for c, s in shares.items() if s != 0.5}
chk('pl_receipt: name, map sha, changed shares, r224 used, no F',
    pr.get('name') == res['name'] and pr.get('map_sha') == res['map_sha'] and pr.get('changed_s') == want_changed
    and pr.get('r224_used') is True and pr.get('f_labels') == {} and pr.get('raptor_failed') == 0 and pr.get('ours_failed') == 0,
    {k: pr.get(k) for k in ('name', 'map_sha', 'changed_s', 'r224_used')})
cr = jl('canary_receipt.json')
chk('canary receipt clean', cr.get('fired') == [] and cr.get('r224_used') is True and cr.get('flip_tta_forced') is bool(A.flip), cr)

sample = pd.read_csv(A.sample, dtype={'StudyInstanceUID': str})
cols = list(sample.columns)
labels = cols[1:]
ids = sample['StudyInstanceUID'].tolist()
sub = pd.read_csv(os.path.join(A.out, 'submission.csv'), dtype={'StudyInstanceUID': str})
chk('submission columns / ids == sample', list(sub.columns) == cols and sub['StudyInstanceUID'].tolist() == ids)
V = sub[labels].values.astype(np.float64)
chk('submission finite in [0,1]', np.isfinite(V).all() and (V >= 0).all() and (V <= 1).all())
rd = lambda n: pd.read_csv(os.path.join(A.out, n), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, labels]
raw, ours, r224 = rd('raptor_raw.csv'), rd('ours_sub.csv'), rd('raptor224_sub.csv')
_pp = os.path.join(A.out, 'submission_PLH.csv')
plh = pd.read_csv(_pp, dtype={'StudyInstanceUID': str}) if os.path.isfile(_pp) else None
chk('submission_PLH.csv == submission.csv (clean canary)', plh is not None and plh.equals(sub))
exp, exp_her = np.zeros((len(ids), 12)), np.zeros((len(ids), 12))
for j, c in enumerate(labels):
    rr = (1 - A224) * avg_rank_pct(raw[c].values) + A224 * avg_rank_pct(r224[c].values)
    ro = avg_rank_pct(ours[c].values)
    exp[:, j] = shares[c] * rr + (1 - shares[c]) * ro
    exp_her[:, j] = 0.5 * rr + 0.5 * ro
d = float(np.abs(V - exp).max())
chk('submission == independent recompute (per-label Raptor share)', d <= 1e-6, 'max diff %.3g' % d)
um = [j for j, c in enumerate(labels) if shares[c] == 0.5]
d2 = float(np.abs(V[:, um] - exp_her[:, um]).max()) if um else 0.0
chk('unmapped columns == V224 recipe at her .5', d2 <= 1e-6, 'max diff %.3g over %d columns' % (d2, len(um)))
mv = [labels[j] for j in range(12) if np.abs(V[:, j] - exp_her[:, j]).max() > 1e-6]
chk('moved columns are mapped ones', set(mv) <= set(want_changed), mv)
chk('no blanked column', all(len(set(np.round(V[:, j], 7))) > 1 for j in range(12)))
if A.ref:
    ref = pd.read_csv(os.path.join(A.ref, 'submission.csv'), dtype={'StudyInstanceUID': str})
    rdr = lambda n: pd.read_csv(os.path.join(A.ref, n), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, labels]
    same = [c for c in labels if all(np.array_equal(avg_rank_pct(a[c].values), avg_rank_pct(b[c].values))
                                     for a, b in ((raw, rdr('raptor_raw.csv')), (r224, rdr('raptor224_sub.csv')), (ours, rdr('ours_sub.csv'))))]
    eq = [c for c in same if c not in want_changed and np.array_equal(sub[c].values, ref[c].values)]
    chk('vs V224-exact commit: every unmapped column with the same component ranks is bit-identical',
        set(eq) == {c for c in same if c not in want_changed}, {'same_ranks': len(same), 'equal': len(eq)})
    print('  info moved vs V224-exact commit:', [c for c in labels if not np.array_equal(sub[c].values, ref[c].values)])
print(f'PL_CHECK {"PASS" if all(R) else "FAIL"} {sum(R)}/{len(R)}')
sys.exit(0 if all(R) else 1)
