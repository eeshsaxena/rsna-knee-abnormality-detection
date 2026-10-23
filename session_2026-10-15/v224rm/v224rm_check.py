#!/usr/bin/env python
# v224rm_check.py (2026-10-14, Claude subagent): commit gate for V224-RM kernels built by build_v224rm.py. Reads a fetched kernel
# output dir (kaggle kernels output eeshsaxena/rsna-sub-v224rm -p OUT). Prints V224RM_CHECK PASS/FAIL n/m; exit 0 only on PASS.
#   usage: python v224rm_check.py OUT --wf W [--flip] [--ref-base OUT_BASE_COMMIT] [--ref-r224 DIAG_OUT] [--sample CSV]
# (1) THE SAME GATE THE BASE PASSED, run as a subprocess on this output (its PASS line is required):
#       wf 0            session_2026-10-14/v224/v224_check.py OUT            (V224 recipe recompute 1e-6, canary clean, ...)
#       wf 0 --flip     v224_check.py OUT --flip --ref REF_BASE
#       wf W            session_2026-10-15/v224f/v224f_check.py OUT --wf W    (incl. v224_check on its V224 view + h_check +
#                       final == (1 - w) pct(V224) + w pct(F) recompute 1e-6)
#       wf W --flip     session_2026-10-15/v224ff/v224ff_check.py OUT --wf W  (when that file exists)
#     This is the "blend recompute matches" gate: stage 3 (and the HF blend) recomputed from the stage csvs this run wrote,
#     i.e. with the mirror-TTA r224 ranks.
# (2) RM checks: R224_MIRROR_VERIFIED (perm derived from BOTH implementations == identity, probability-space mean, r384 pieces
#     byte-identical) -> R224_MIRROR_PATCHED (sha of the patched script == the build) -> R224_MIRROR_TTA -> R224_STAGE rc=0 ok=True
#     in cell order; tta receipt (mirror, identity perm, n_pairs == n, n_fail 0); raw avg == float32 0.5 (plain + mirror)
#     exactly; raptor224_sub == her rankpct(avg) (1e-6); THE MIRROR MOVES r224: max |plain - mirror| > 1e-3 and > 1e-4 on at
#     least 10 of 12 labels; plain == her r224 raw from the wheel-order diagnostic commit (same 3 studies, R7-wheel env) within the
#     compile tolerance 1e-3 (so the patch left her plain path alone); canary receipt: r224 used, nothing fired, RM guard
#     mirror_ok; RM_GUARD_CLEAN after the canary. With --ref-base (the base kernel's own commit output): R7 ours_sub and r384
#     raw equal within 1e-3 (untouched stages), and the per-column submission change is printed (info).
import os, sys, json, glob, argparse, subprocess
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
V224_CHECK = os.path.join(ROOT, 'session_2026-10-14', 'v224', 'v224_check.py')
V224F_CHECK = os.path.join(ROOT, 'session_2026-10-15', 'v224f', 'v224f_check.py')
V224FF_CHECK = os.path.join(ROOT, 'session_2026-10-15', 'v224ff', 'v224ff_check.py')
DIAG_OUT = os.path.join(ROOT, 'session_2026-10-14', 'v224', 'out_v224diag_v1')
ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--wf', type=float, required=True)
ap.add_argument('--flip', action='store_true')
ap.add_argument('--ref-base', default=None, help="the base kernel's own commit output dir")
ap.add_argument('--ref-r224', default=DIAG_OUT, help='wheel-order diagnostic commit output (diag_C_raw.csv = her r224 raw)')
ap.add_argument('--base-gate-args', default='', help='extra args for the base gate (e.g. --ref-v224 DIR --ref-fleet DIR)')
ap.add_argument('--sample', default=os.path.join(ROOT, '_data', 'sample_submission.csv'))
A = ap.parse_args()
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
R = []
TOL_BLEND, TOL_COMPILE = 1e-6, 1e-3


def chk(name, ok, info=''):
    R.append(bool(ok))
    print(f'  {"OK  " if ok else "FAIL"} {name} {info}', flush=True)


def P(*a):
    return os.path.join(A.out, *a)


def jload(p):
    try:
        return json.load(open(p, encoding='utf-8'))
    except Exception:
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


def csv_(d, name, ids, labels):
    return pd.read_csv(os.path.join(d, name), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, labels]


mode = ('V224-FLIP-RM' if A.flip else 'V224-RM') if A.wf == 0 else ('V224-FF%02d-RM' if A.flip else 'V224-F%02d-RM') % round(A.wf * 100)
print(f'V224RM_CHECK {A.out} mode {mode}', flush=True)

# ---- (1) the base gate --------------------------------------------------------------------------------------------------
if A.wf == 0:
    cmd = [sys.executable, V224_CHECK, A.out, '--sample', A.sample] + (['--flip', '--ref', A.ref_base] if A.flip else [])
    want = 'V224_CHECK PASS'
elif not A.flip:
    cmd = [sys.executable, V224F_CHECK, A.out, '--wf', repr(A.wf), '--sample', A.sample] + A.base_gate_args.split()
    want = 'V224F_CHECK PASS'
else:
    cmd = [sys.executable, V224FF_CHECK, A.out, '--wf', repr(A.wf)] + A.base_gate_args.split() if os.path.isfile(V224FF_CHECK) else None
    want = 'PASS'
if cmd is None:
    chk('base gate available (v224ff_check.py)', False, V224FF_CHECK)
else:
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    print('\n'.join('    | ' + ln for ln in r.stdout.strip().splitlines()), flush=True)
    chk('base gate %s' % os.path.basename(cmd[1]), r.returncode == 0 and want in r.stdout, r.stdout.strip().splitlines()[-1:] if r.stdout else r.stderr[-300:])

# ---- (2) RM checks ------------------------------------------------------------------------------------------------------
logs = [p for p in glob.glob(P('*.log')) if 'cufile' not in os.path.basename(p)]
text = ''
if logs:
    try:
        text = '\n'.join(e.get('data', '') for e in json.load(open(logs[0], encoding='utf-8')))
    except Exception:
        text = open(logs[0], encoding='utf-8', errors='replace').read()
chk('kernel log present', bool(text), os.path.basename(logs[0]) if logs else '')
order = ['RAPTOR_STAGE_DONE', 'OURS_STAGE_COMPLETE', 'R224_MIRROR_VERIFIED', 'R224_MIRROR_PATCHED', 'R224_MIRROR_TTA',
         'R224_STAGE rc=0 ok=True', 'NOTEBOOK_COMPLETE', 'V224_DONE'] + (['V224F_DONE'] if A.wf > 0 else []) + ['RM_GUARD_CLEAN', 'V224RM_DONE']
pos, p0 = [], 0
for m in order:
    p = text.find(m, p0)
    pos.append(p)
    if p >= 0:
        p0 = p
chk('RM markers present, in cell order', all(p >= 0 for p in pos), {m: p for m, p in zip(order, pos) if p < 0})
bad = ['RM_GUARD_FIRED', 'R224_MIRROR_FAILED', 'Traceback (most recent call last)', 'CANARY_FIRED', 'R224_SKIPPED', 'R224_TIMEOUT',
       'R224_UNUSABLE']
hits = [m for m in bad if m in text]
fb = [ln.strip()[:120] for ln in text.splitlines() if '] study ' in ln and 'FALLBACK (' in ln]
chk('no RM / canary / per-study fallback marker', not hits and not fb, hits + fb[:3])
import build_v224rm as B   # module level only (shas, patches)
vj = line_json(text, 'R224_MIRROR_VERIFIED') or {}
chk('R224_MIRROR_VERIFIED: perm derived from r384 AND r224 == identity, probability-space mean, r384 pieces identical',
    vj.get('perm_r384') == list(range(12)) and vj.get('perm_r224') == list(range(12)) and
    vj.get('combine') == '0.5 * (p + q), probability space (both)' and vj.get('identical_source') and
    all(vj['identical_source'].values()) and vj.get('mirror_tri_functional_equal') is True,
    {k: vj.get(k) for k in ('perm_r224', 'combine')})
pj = line_json(text, 'R224_MIRROR_PATCHED') or {}
chk('R224_MIRROR_PATCHED: her r224 sha + patched sha == this build', pj.get('r224_her_sha') == B.R224_SHA[:16] and
    pj.get('r224_rm_sha') == B.R224_RM_SHA[:16] and pj.get('patches') == 6, pj)
sample = pd.read_csv(A.sample, dtype={'StudyInstanceUID': str})
labels = list(sample.columns)[1:]
ids = sample['StudyInstanceUID'].tolist()
n = len(ids)
tr = jload(P('raptor224_tta_receipt.json')) or {}
chk('tta receipt: mirror, identity perm, labels, n_pairs == n == %d, n_fail 0' % n, tr.get('mirror') is True and
    tr.get('perm') == list(range(12)) and tr.get('labels') == labels and tr.get('n') == n and tr.get('n_pairs') == n and
    tr.get('n_fail') == 0, {k: tr.get(k) for k in ('n', 'n_pairs', 'n_fail', 'devices')})
have = all(os.path.isfile(P(f)) for f in ('raptor224_raw.csv', 'raptor224_raw_plain.csv', 'raptor224_raw_mirror.csv', 'raptor224_sub.csv'))
chk('r224 raw / plain / mirror / sub csvs present', have)
if have:
    avg = csv_(A.out, 'raptor224_raw.csv', ids, labels).values.astype(np.float32)
    pl = csv_(A.out, 'raptor224_raw_plain.csv', ids, labels).values.astype(np.float32)
    mi = csv_(A.out, 'raptor224_raw_mirror.csv', ids, labels).values.astype(np.float32)
    sub224 = csv_(A.out, 'raptor224_sub.csv', ids, labels).values.astype(np.float64)
    d0 = float(np.abs(avg.astype(np.float64) - (np.float32(0.5) * (pl + mi)).astype(np.float64)).max())
    chk('raw avg == float32 0.5 * (plain + mirror)', d0 == 0.0, 'max diff %.3g' % d0)
    order_ = np.clip(avg, 0, 1).argsort(0).argsort(0).astype(np.float64) / max(1, n - 1)
    d1 = float(np.abs(sub224 - order_.astype(np.float32)).max())
    chk("raptor224_sub == her rankpct(mirror-averaged raw)", d1 <= TOL_BLEND, 'max diff %.3g' % d1)
    dpm = np.abs(pl.astype(np.float64) - mi.astype(np.float64))
    moved = [c for j, c in enumerate(labels) if dpm[:, j].max() > 1e-4]
    chk('THE MIRROR MOVES r224 RAW: max |plain - mirror| > 1e-3 and > 1e-4 on >= 10 of 12 labels', float(dpm.max()) > 1e-3 and len(moved) >= 10,
        'max %.4f, mean %.4f, labels moved %d: %s' % (dpm.max(), dpm.mean(), len(moved), {c: round(float(dpm[:, j].max()), 4) for j, c in enumerate(labels)}))
    pl_rank = np.clip(pl, 0, 1).argsort(0).argsort(0)
    av_rank = np.clip(avg, 0, 1).argsort(0).argsort(0)
    print('  info r224 rank cells changed by the mirror (plain vs avg, %d rows): %d of %d' % (n, int((pl_rank != av_rank).sum()), pl_rank.size), flush=True)
    if A.ref_r224 and os.path.isfile(os.path.join(A.ref_r224, 'diag_C_raw.csv')):
        ref = csv_(A.ref_r224, 'diag_C_raw.csv', ids, labels).values.astype(np.float64)
        dr = float(np.abs(pl.astype(np.float64) - ref).max())
        chk('plain pass == her r224 raw (wheel-diag commit, R7-wheel env) within %g' % TOL_COMPILE, dr <= TOL_COMPILE,
            'max diff %.3g (GPU run-to-run floor 2.4e-4)' % dr)
        dm = float(np.abs(mi.astype(np.float64) - ref).max())
        print('  info mirror pass vs her plain r224 raw: max diff %.4f' % dm, flush=True)
    else:
        chk('reference r224 raw (diag_C_raw.csv) available', False, A.ref_r224)
cr = jload(P('canary_receipt.json')) or {}
g = cr.get('rm_guard') or {}
chk('canary receipt: r224 used, nothing fired, RM guard mirror_ok, rm_on', cr.get('r224_used') is True and cr.get('fired') == [] and
    g.get('mirror_ok') is True and cr.get('rm_on') is True, {k: cr.get(k) for k in ('r224_used', 'fired', 'rm_on', 'w_f_global')})
st = next((ln for ln in text.splitlines() if 'R224_STAGE rc=' in ln), '')
try:
    r224_s = float(st.rsplit('seconds', 1)[1].strip())
except Exception:
    r224_s = None
print('  info runtime: R224_STAGE %s s; per study (s) %s; sums %s; script total %s s' % (
    r224_s, tr.get('per_study_s'), tr.get('sum_s'), tr.get('seconds')), flush=True)
print('  info per-study [total, plain, mirror] s:', tr.get('per_study_s_list'), flush=True)

# ---- vs the base commit -----------------------------------------------------------------------------------------------------
if A.ref_base:
    B0 = A.ref_base
    for nm in ('ours_sub.csv', 'raptor_raw.csv'):
        if os.path.isfile(os.path.join(B0, nm)) and os.path.isfile(P(nm)):
            dd = float(np.abs(csv_(A.out, nm, ids, labels).values - csv_(B0, nm, ids, labels).values).max())
            chk('untouched stage %s == base commit within %g' % (nm, TOL_COMPILE), dd <= TOL_COMPILE, 'max diff %.3g' % dd)
    for nm in ('raptor224_sub.csv', 'submission_V224.csv', 'submission.csv'):
        if os.path.isfile(os.path.join(B0, nm)) and os.path.isfile(P(nm)):
            a_, b_ = csv_(A.out, nm, ids, labels).values, csv_(B0, nm, ids, labels).values
            print('  info %s vs base commit: cells differing %d of %d, per column max %s' % (
                nm, int((np.abs(a_ - b_) > 1e-6).sum()), a_.size, {c: round(float(np.abs(a_[:, j] - b_[:, j]).max()), 4) for j, c in enumerate(labels)}), flush=True)
    m0 = ''
    for lp in [q for q in glob.glob(os.path.join(B0, '*.log')) if 'cufile' not in q][:1]:
        try:
            t0 = chr(10).join(e.get('data', '') for e in json.load(open(lp, encoding='utf-8')))
        except Exception:
            t0 = open(lp, encoding='utf-8', errors='replace').read()
        m0 = next((ln for ln in t0.splitlines() if 'R224_STAGE rc=' in ln), '')
    print('  info base commit', m0.strip()[:120], flush=True)

print(f'V224RM_CHECK {"PASS" if all(R) else "FAIL"} {sum(R)}/{len(R)} mode {mode}', flush=True)
sys.exit(0 if all(R) else 1)
