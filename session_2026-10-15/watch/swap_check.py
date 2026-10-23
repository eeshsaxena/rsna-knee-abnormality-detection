#!/usr/bin/env python
# swap_check.py (2026-10-14, Claude subagent): commit gate for a kernel built by swap_builder.py (any profile). Reads our fetched
# commit output (kaggle kernels output eeshsaxena/<name> -p OUT) and, with --her, the public notebook's own published commit
# output (kaggle kernels output OWNER/SLUG -p HER). Prints SWAP_CHECK PASS/FAIL n/m; exit 0 only on PASS.
#   usage: python swap_check.py OUT [--her HER] [--tol 1e-6] [--strict-csvs] [--expect-fired --canary-col Fracture]
#   default        the copy ran clean: canary marker (SWAP_DONE or V224_DONE) + CANARY_CLEAN, no CANARY_FIRED, no Traceback,
#                  canary_receipt fired == [], submission.csv == sample_submission (columns, ids in order), finite in [0, 1], no
#                  constant column; with --her: submission.csv equal to theirs within --tol (default 1e-6; V224 was identical),
#                  every other csv present in both dirs reported (gated at 1e-3 with --strict-csvs).
#   --expect-fired a forced-fallback control: CANARY_FIRED, receipt fired non-empty, --canary-col == 0.5 exactly in every row.
import os, sys, json, glob, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--her', default=None)
ap.add_argument('--tol', type=float, default=1e-6)
ap.add_argument('--strict-csvs', action='store_true')
ap.add_argument('--expect-fired', action='store_true')
ap.add_argument('--canary-col', default='Fracture')
ap.add_argument('--sample', default=os.path.normpath(os.path.join(HERE, '..', '..', '_data', 'sample_submission.csv')))
A = ap.parse_args()
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
R = []


def chk(name, ok, info=''):
    R.append(bool(ok))
    print('  %s %s %s' % ('OK  ' if ok else 'FAIL', name, info), flush=True)


def read_log(d):
    logs = [p for p in glob.glob(os.path.join(d, '*.log')) if 'cufile' not in os.path.basename(p)]
    if not logs:
        return '', None
    try:
        return '\n'.join(e.get('data', '') for e in json.load(open(logs[0], encoding='utf-8'))), logs[0]
    except Exception:
        return open(logs[0], encoding='utf-8', errors='replace').read(), logs[0]


def load_sub(d):
    return pd.read_csv(os.path.join(d, 'submission.csv'), dtype={'StudyInstanceUID': str})


mode = 'control (expect fired)' if A.expect_fired else 'clean'
print('SWAP_CHECK %s mode %s' % (A.out, mode), flush=True)
text, logp = read_log(A.out)
chk('kernel log present', bool(text), os.path.basename(logp) if logp else '')
chk('canary cell ran (SWAP_DONE or V224_DONE)', 'SWAP_DONE' in text or 'V224_DONE' in text)
if A.expect_fired:
    chk('CANARY_FIRED in the log', 'CANARY_FIRED' in text and 'CANARY_CLEAN' not in text)
else:
    chk('CANARY_CLEAN in the log, no CANARY_FIRED', 'CANARY_CLEAN' in text and 'CANARY_FIRED' not in text)
chk('no Traceback in the log', 'Traceback (most recent call last)' not in text)
for m in ('CANARY_CONSTANT_COLUMNS', 'CANARY_RECEIPT_FALLBACKS', 'CANARY_NO_SAMPLE', 'BLEND_FALLBACK'):
    if m in text:
        print('  info log has %s: %s' % (m, next(ln for ln in text.splitlines() if m in ln).strip()[:200]), flush=True)
crp = os.path.join(A.out, 'canary_receipt.json')
cr = json.load(open(crp, encoding='utf-8')) if os.path.isfile(crp) else None
chk('canary_receipt.json present', cr is not None)
if cr is not None:
    fired = cr.get('fired') or []
    chk('receipt: fired %s' % ('non-empty' if A.expect_fired else '== []'), bool(fired) == A.expect_fired, fired)

sample = pd.read_csv(A.sample, dtype={'StudyInstanceUID': str})
cols, ids = list(sample.columns), sample['StudyInstanceUID'].tolist()
labels = cols[1:]
ok_sub = os.path.isfile(os.path.join(A.out, 'submission.csv'))
chk('submission.csv present', ok_sub)
if ok_sub:
    sub = load_sub(A.out)
    chk('columns == sample_submission', list(sub.columns) == cols)
    chk('ids == sample_submission (same order, unique)', sub['StudyInstanceUID'].tolist() == ids and sub['StudyInstanceUID'].is_unique, len(sub))
    V = sub[labels].values.astype(np.float64) if list(sub.columns) == cols else np.full((1, 1), np.nan)
    chk('values finite in [0, 1]', np.isfinite(V).all() and (V >= 0).all() and (V <= 1).all())
    const = [c for c in labels if c in sub and len(sub) > 1 and sub[c].nunique() <= 1]
    if A.expect_fired:
        chk('control: %s == 0.5 exactly in every row' % A.canary_col, A.canary_col in sub and bool((sub[A.canary_col] == 0.5).all()))
        chk('control: no other constant column', [c for c in const if c != A.canary_col] == [], const)
    else:
        chk('no constant (blanked) column', not const, const)
    if A.her:
        hs = load_sub(A.her).set_index('StudyInstanceUID')
        me = sub.set_index('StudyInstanceUID')
        same_ids = list(hs.index) == list(me.index)
        chk('her submission: same ids, same order', same_ids, (len(hs), len(me)))
        use = [c for c in labels if not (A.expect_fired and c == A.canary_col)]
        if same_ids:
            d = np.abs(me[use].values.astype(np.float64) - hs[use].values.astype(np.float64))
            chk('her submission: equal within %g (%d columns)' % (A.tol, len(use)), float(d.max()) <= A.tol,
                'max diff %.3g, cells differing %d of %d' % (float(d.max()), int((d > A.tol).sum()), d.size))
        for p in sorted(glob.glob(os.path.join(A.her, '*.csv'))):
            nm = os.path.basename(p)
            q = os.path.join(A.out, nm)
            if nm == 'submission.csv' or not os.path.isfile(q):
                continue
            try:
                a = pd.read_csv(q, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
                b = pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
                cc = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c])]
                dd = float(np.abs(a.loc[b.index, cc].values.astype(np.float64) - b[cc].values.astype(np.float64)).max())
            except Exception as e:
                print('  info %s not comparable: %s' % (nm, str(e)[:120]), flush=True)
                continue
            if A.strict_csvs:
                chk('her %s within 1e-3' % nm, dd <= 1e-3, 'max diff %.3g' % dd)
            else:
                print('  info her %s: max diff %.3g' % (nm, dd), flush=True)
print('SWAP_CHECK %s %d/%d mode %s' % ('PASS' if all(R) else 'FAIL', sum(R), len(R), mode), flush=True)
sys.exit(0 if all(R) else 1)
