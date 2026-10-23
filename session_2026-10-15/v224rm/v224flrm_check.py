#!/usr/bin/env python
# v224flrm_check.py (2026-10-14, Claude subagent): commit gate for V224-FLIP-RM (eeshsaxena/rsna-sub-v224flrm) = the submitted
# V224-FLIP notebook + the two r224 mirror inserts (RM, RM_GUARD). Reads a fetched kernel output dir. No em dashes.
# Prints V224FLRM_CHECK PASS/FAIL n/m; exit 0 only on PASS.
#   usage: python v224flrm_check.py OUT [--ref-exact DIR] [--ref-flip DIR] [--expected CSV]
# (A) V224-FLIP part = the gate V224-FLIP passed: session_2026-10-14/v224/v224_check.py OUT --flip --ref <V224-exact commit>.
#     Every check must be OK, with ONE by-design exception: "flip: submission MCL == V224-exact submission MCL" may fail only when
#     this run's r224 MCL ranks differ from the V224-exact commit's (RM changes the r224 ranks by design). It is then replaced by
#     (A2): the submitted MCL == the V224 recipe on (this run's r384 raw MCL, this run's r224 MCL, the V224-EXACT commit's R7
#     MCL) within 1e-6, i.e. the R7 flip still leaves MCL alone (MCL from the plain pass).
# (B) RM part = the v224rm gate: v224rm_check.py OUT --wf 0 --flip --ref-base <V224-FLIP commit>. Every top-level check must
#     be OK except its "base gate" line, which runs v224_check against the FLIP commit (where the flip-vs-exact checks cannot
#     hold) and is superseded by (A). Covers: R224_MIRROR_VERIFIED (identity perm derived from both implementations, probability
#     mean), R224_MIRROR_PATCHED sha, tta receipt, avg == 0.5 (plain + mirror) exactly, r224 sub == rankpct(avg), the mirror moves
#     r224, plain pass == her r224 raw (1e-3), RM guard clean, untouched R7 ours_sub / r384 raw == the V224-FLIP commit (1e-3).
# (C) Independent recompute here (scipy rankdata, a third implementation): r224 ranks from raptor224_raw.csv, then the V224
#     recipe (no re-rank) -> submission.csv within 1e-6.
# (D) One change vs the V224-FLIP commit: on every column where the r384 and R7 rank orders equal the FLIP commit's, the
#     submission == the V224 recipe on (FLIP commit r384 raw, FLIP commit R7, THIS run's r224 ranks) within 1e-6; >= 10 of 12
#     columns comparable.
# (E) No fleet, flip on, RM on: canary receipt flip_tta_forced True / use_f False / rm_on True / fired [] / w_f_global None; no
#     fleet files or markers; FLIP_TTA_FORCED before OURS_STAGE_COMPLETE before the RM markers.
import os, sys, json, glob, argparse, subprocess
import numpy as np, pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
V224 = os.path.join(ROOT, 'session_2026-10-14', 'v224')
ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--ref-exact', default=os.path.join(V224, 'out_v224_v1'), help='V224-exact commit output (rsna-sub-v224 v1)')
ap.add_argument('--ref-flip', default=os.path.join(V224, 'out_v224flip_v1'), help='V224-FLIP commit output (rsna-sub-v224flip v1)')
ap.add_argument('--expected', default=os.path.join(HERE, 'sim_flrm_expected_submission.csv'), help='sim_flrm.py prediction (info)')
ap.add_argument('--sample', default=os.path.join(ROOT, '_data', 'sample_submission.csv'))
A = ap.parse_args()
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
R = []
TOL = 1e-6
MCL_EXEMPT = 'flip: submission MCL == V224-exact submission MCL'
RM_SUPERSEDED = 'base gate v224_check.py'


def chk(name, ok, info=''):
    R.append(bool(ok))
    print(f'  {"OK  " if ok else "FAIL"} {name} {info}', flush=True)


def top_checks(stdout):
    # (status, name) of the top-level check lines ('  OK   name info' / '  FAIL name info'), not the nested '    | ' lines
    out = []
    for ln in stdout.splitlines():
        if ln.startswith('  OK   '):
            out.append(('OK', ln[7:]))
        elif ln.startswith('  FAIL '):
            out.append(('FAIL', ln[7:]))
    return out


sample = pd.read_csv(A.sample, dtype={'StudyInstanceUID': str})
labels, ids = list(sample.columns[1:]), sample['StudyInstanceUID'].tolist()
n = len(ids)


def rd(d, f):
    return pd.read_csv(os.path.join(d, f), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, labels].values.astype(np.float64)


pct = lambda v: rankdata(v, method='average') / len(v)


def recipe(raw, r224, ours):
    return np.stack([.5 * (.65 * pct(raw[:, j]) + .35 * pct(r224[:, j])) + .5 * pct(ours[:, j]) for j in range(12)], 1)


print(f'V224FLRM_CHECK {A.out} mode V224-FLIP-RM', flush=True)

# ---- (A) the V224-FLIP gate -----------------------------------------------------------------------------------------------------
cmd = [sys.executable, os.path.join(V224, 'v224_check.py'), A.out, '--flip', '--ref', A.ref_exact, '--sample', A.sample]
r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
print('\n'.join('    | ' + ln for ln in r.stdout.strip().splitlines()), flush=True)
ck = top_checks(r.stdout)
fails = [nm for st, nm in ck if st == 'FAIL']
other_fails = [nm for nm in fails if not nm.startswith(MCL_EXEMPT)]
mcl_failed = any(nm.startswith(MCL_EXEMPT) for nm in fails)
j_mcl = labels.index('MCL')
r224_mcl_moved = None
try:
    r224_mcl_moved = not np.array_equal(rankdata(rd(A.out, 'raptor224_sub.csv')[:, j_mcl]), rankdata(rd(A.ref_exact, 'raptor224_sub.csv')[:, j_mcl]))
except Exception as e:
    print('  info r224 MCL rank comparison failed:', repr(e)[:120], flush=True)
chk('(A) v224_check.py --flip --ref V224-exact: every check OK except at most the by-design MCL-vs-exact one',
    len(ck) >= 20 and not other_fails and 'mode flip' in (r.stdout.strip().splitlines() or [''])[-1],
    {'checks': len(ck), 'fails': fails, 'last': (r.stdout.strip().splitlines() or [''])[-1]})
if mcl_failed:
    chk('(A) the MCL-vs-exact difference is explained by RM: this run\'s r224 MCL ranks differ from the V224-exact commit\'s',
        r224_mcl_moved is True, {'r224_mcl_ranks_moved': r224_mcl_moved})
try:
    raw, r224, ours = rd(A.out, 'raptor_raw.csv'), rd(A.out, 'raptor224_sub.csv'), rd(A.out, 'ours_sub.csv')
    ex_ours = rd(A.ref_exact, 'ours_sub.csv')
    sub = rd(A.out, 'submission.csv')
    m = .5 * (.65 * pct(raw[:, j_mcl]) + .35 * pct(r224[:, j_mcl])) + .5 * pct(ex_ours[:, j_mcl])
    dm = float(np.abs(sub[:, j_mcl] - m).max())
    chk('(A2) submitted MCL == V224 recipe on (this r384 raw, this r224, V224-EXACT R7 MCL): the flip leaves MCL alone', dm <= TOL,
        'max diff %.3g' % dm)
except Exception as e:
    chk('(A2) MCL recompute', False, repr(e)[:200])

# ---- (B) the v224rm gate (RM part) ------------------------------------------------------------------------------------------
cmd = [sys.executable, os.path.join(HERE, 'v224rm_check.py'), A.out, '--wf', '0', '--flip', '--ref-base', A.ref_flip, '--sample', A.sample]
r2 = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
print('\n'.join('    | ' + ln for ln in r2.stdout.strip().splitlines() if not ln.startswith('    | ')), flush=True)
ck2 = top_checks(r2.stdout)
fails2 = [nm for st, nm in ck2 if st == 'FAIL' and not nm.startswith(RM_SUPERSEDED)]
names2 = [nm for st, nm in ck2]
need = ['R224_MIRROR_VERIFIED', 'R224_MIRROR_PATCHED', 'tta receipt', 'raw avg == float32', 'raptor224_sub == her rankpct',
        'THE MIRROR MOVES', 'plain pass == her r224 raw', 'canary receipt: r224 used', 'untouched stage ours_sub.csv',
        'untouched stage raptor_raw.csv', 'RM markers present']
missing = [k for k in need if not any(nm.startswith(k) for nm in names2)]
chk('(B) v224rm_check.py --wf 0 --flip --ref-base V224-FLIP: every RM check OK (its base-gate line superseded by (A))',
    not fails2 and not missing and len(ck2) >= 14, {'checks': len(ck2), 'fails': fails2, 'missing': missing})

# ---- (C) independent recompute from the r224 RAW (third implementation) ---------------------------------------------------------
try:
    avg = rd(A.out, 'raptor224_raw.csv')
    r224_from_raw = np.stack([(rankdata(np.clip(avg[:, j], 0, 1), method='ordinal') - 1) / max(1, n - 1) for j in range(12)], 1)
    d_r = float(np.abs(r224_from_raw - r224).max())
    chk('(C) raptor224_sub.csv == ordinal rank pct of the mirror-averaged raw (scipy)', d_r <= TOL, 'max diff %.3g' % d_r)
    exp = recipe(raw, r224_from_raw, ours)
    d_c = float(np.abs(exp - sub).max())
    chk('(C) submission.csv == V224 recipe (no re-rank) on this run\'s r384 raw, mirror r224 (from raw) and flip R7 (scipy)',
        d_c <= TOL, 'max diff %.3g' % d_c)
except Exception as e:
    chk('(C) independent recompute', False, repr(e)[:200])

# ---- (D) one change vs the V224-FLIP commit -------------------------------------------------------------------------------------
try:
    f_raw, f_ours, f_r224, f_sub = (rd(A.ref_flip, f) for f in ('raptor_raw.csv', 'ours_sub.csv', 'raptor224_sub.csv', 'submission.csv'))
    alt = recipe(f_raw, r224, f_ours)
    comp, bad, info = [], [], {}
    for j, c in enumerate(labels):
        same_order = np.array_equal(rankdata(raw[:, j]), rankdata(f_raw[:, j])) and np.array_equal(rankdata(ours[:, j]), rankdata(f_ours[:, j]))
        if same_order:
            comp.append(c)
            if np.abs(alt[:, j] - sub[:, j]).max() > TOL:
                bad.append(c)
        info[c] = round(float(np.abs(sub[:, j] - f_sub[:, j]).max()), 4)
    chk('(D) vs V224-FLIP commit: submission == recipe(FLIP r384 raw, FLIP R7, THIS r224) on every rank-comparable column (>= 10)',
        len(comp) >= 10 and not bad, {'comparable': len(comp), 'bad': bad})
    r224_moved = [c for j, c in enumerate(labels) if not np.array_equal(rankdata(r224[:, j]), rankdata(f_r224[:, j]))]
    print('  info r224 rank columns changed vs the V224-FLIP commit:', r224_moved, flush=True)
    print('  info submission vs the V224-FLIP commit, per column max |diff|:', info, flush=True)
except Exception as e:
    chk('(D) one change vs the V224-FLIP commit', False, repr(e)[:200])
if os.path.isfile(A.expected):
    e_ = rd(os.path.dirname(A.expected) or '.', os.path.basename(A.expected))
    print('  info vs the offline prediction sim_flrm_expected_submission.csv: max |diff| %.4g, cells differing %d of %d' % (
        float(np.abs(e_ - sub).max()), int((np.abs(e_ - sub) > TOL).sum()), sub.size), flush=True)

# ---- (E) flip on, RM on, no fleet ------------------------------------------------------------------------------------------------
cr = {}
try:
    cr = json.load(open(os.path.join(A.out, 'canary_receipt.json'), encoding='utf-8'))
except Exception:
    pass
chk('(E) canary receipt: flip_tta_forced True, use_f False, rm_on True, r224 used, fired [], w_f_global None',
    cr.get('flip_tta_forced') is True and cr.get('use_f') is False and cr.get('rm_on') is True and cr.get('r224_used') is True
    and cr.get('fired') == [] and cr.get('w_f_global') is None, {k: cr.get(k) for k in ('flip_tta_forced', 'use_f', 'rm_on', 'fired', 'w_f_global')})
logs = [p for p in glob.glob(os.path.join(A.out, '*.log')) if 'cufile' not in os.path.basename(p)]
text = ''
if logs:
    try:
        text = '\n'.join(e.get('data', '') for e in json.load(open(logs[0], encoding='utf-8')))
    except Exception:
        text = open(logs[0], encoding='utf-8', errors='replace').read()
fleet_files = [f for f in ('submission_fleetsolo.csv', 'fleet_receipt.json', 'hf_receipt.json', 'v224f_receipt.json', 'submission_V224.csv')
               if os.path.isfile(os.path.join(A.out, f))]
fleet_marks = [m for m in ('V224F_', 'HF_BLEND', 'FN_IDS', 'K2_STAGE', 'fleet stage') if m in text]
chk('(E) no fleet: no fleet / HF files, no fleet markers in the log', not fleet_files and not fleet_marks, fleet_files + fleet_marks)
order = ['FLIP_TTA_FORCED', 'OURS_STAGE_COMPLETE', 'R224_MIRROR_VERIFIED', 'R224_MIRROR_PATCHED', 'R224_MIRROR_TTA', 'R224_STAGE rc=0 ok=True',
         'NOTEBOOK_COMPLETE', 'CANARY_CLEAN', 'V224_DONE', 'RM_GUARD_CLEAN', 'V224RM_DONE']
pos = [text.find(m) for m in order]
chk('(E) FLIP_TTA_FORCED -> R7 -> RM verify/patch -> r224 mirror stage -> stage 3 -> canary -> RM guard, in this order',
    all(p >= 0 for p in pos) and pos == sorted(pos), {m: p for m, p in zip(order, pos) if p < 0})

# ---- runtime (info) --------------------------------------------------------------------------------------------------------------
try:
    ent = json.load(open(logs[0], encoding='utf-8'))
    t_end = max(float(e.get('time', 0)) for e in ent)
except Exception:
    t_end = None
oc = next((ln for ln in text.splitlines() if 'OURS_STAGE_COMPLETE' in ln), '')
st = next((ln for ln in text.splitlines() if 'R224_STAGE rc=' in ln), '')
try:
    r7s = json.loads(oc[oc.index('OURS_STAGE_COMPLETE') + 20:]).get('seconds')
except Exception:
    r7s = None
tr = {}
try:
    tr = json.load(open(os.path.join(A.out, 'raptor224_tta_receipt.json'), encoding='utf-8'))
except Exception:
    pass
print('  info runtime: kernel log end %s s; R7 stage %s s; %s; r224 script %s s; per study %s' % (
    t_end, r7s, st.strip()[:60], tr.get('seconds'), tr.get('per_study_s')), flush=True)
print(f'V224FLRM_CHECK {"PASS" if all(R) else "FAIL"} {sum(R)}/{len(R)} mode V224-FLIP-RM', flush=True)
sys.exit(0 if all(R) else 1)
