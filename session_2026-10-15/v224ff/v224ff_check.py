#!/usr/bin/env python
# v224ff_check.py (2026-10-15 dir, Claude subagent): commit gate for V224-F-FLIP kernels (build_v224ff.py). Reads a fetched kernel
# output dir. Prints V224FF_CHECK PASS/FAIL n/m; exit 0 only on PASS.
#   usage: python v224ff_check.py OUT --wf 0.20 [--control] --ref-flip OUT_V224FLIP --ref-exact OUT_V224
#                                     [--ref-fleet OUT_HFR7RR] [--flip-budget-h 4.9] [--sample sample_submission.csv]
# Sub-gates (subprocesses, their lines are echoed):
#   (1) session_2026-10-15/v224f/v224f_check.py OUT --wf W: the whole V224-F gate (fleet receipt, fleet solo == npz member mean,
#       final == (1 - w) pct(V224) + w pct(F), h_check hf, markers, receipts). Its ONE exact-mode check ("V224 part ... passes
#       v224_check.py exact") must fail here by design (R7 runs with flip); every other check must pass. It is replaced by (2).
#   (2) session_2026-10-14/v224/v224_check.py --flip --ref OUT_V224 on a VIEW of OUT (log cut after V224_DONE, submission.csv :=
#       submission_V224.csv): her cell order, FLIP_TTA_FORCED, FLIP_TTA_READY + mcl plain pass in every worker log, flip block in the
#       R7 receipt, R7 PASS with 0 fallback studies, r224 used, canary clean, independent V224 recompute (1e-6), MCL == the
#       V224-exact R7 MCL and >= 10 of the other 11 R7 columns moved.
# Own checks: (3) the time-aware flip: FLIP_TTA_FORCED payload (driver, budget, driver sha), FLIP_TA_REPORT between the R7 stage and
#   r224, flip_ta_receipt.json, every worker log FLIP_TA_START / FLIP_TA_DONE without driver errors; clean = every worker kept flip
#   on every study and R7 == the V224-FLIP commit's R7 (1e-3), submission_V224.csv == the V224-FLIP commit submission (1e-6);
#   --control = one worker FLIP_TTA_DISABLED_AT_STUDY 1 (lower-bound rule, child killed -9, plain child rc 0), the 1-study worker
#   unchanged, each R7 row equal to the V224-FLIP commit row if flipped, to the V224-exact commit row if not (1e-3).
#   (4) one END-TO-END independent recompute: H = V224 recipe from raptor_raw / raptor224_sub / ours_sub (numpy ranks, float32 like
#   the notebook), F = member mean of the fleet workers' npz, final = (1 - w) pct(H) + w pct(F) == submission.csv (1e-6).
import os, sys, json, glob, shutil, hashlib, argparse, subprocess, tempfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
V224F_CHECK = os.path.join(ROOT, 'session_2026-10-15', 'v224f', 'v224f_check.py')
V224_CHECK = os.path.join(ROOT, 'session_2026-10-14', 'v224', 'v224_check.py')
DRIVER = os.path.join(HERE, 'compact_infer_ta.py')
ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--wf', type=float, required=True)
ap.add_argument('--control', action='store_true')
ap.add_argument('--ref-flip', required=True, help='V224-FLIP commit output dir (R7 with flip on every study)')
ap.add_argument('--ref-exact', required=True, help='V224-exact commit output dir (R7 without flip)')
ap.add_argument('--ref-fleet', default=None)
ap.add_argument('--flip-budget-h', type=float, default=4.9)
ap.add_argument('--sample', default=os.path.join(ROOT, '_data', 'sample_submission.csv'))
A = ap.parse_args()
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
R = []
TOL_BLEND, TOL_COMPILE = 1e-6, 1e-3
EXPECTED_V224F_FAIL = 'V224 part (log up to V224_DONE, submission_V224.csv) passes v224_check.py exact'


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


def pct(v):
    v = np.asarray(v, dtype=np.float64)
    order = np.argsort(v, kind='mergesort')
    sv = v[order]
    r = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r / len(v)


def lg(v):
    v = np.clip(np.asarray(v, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.log(v / (1 - v))


def csv_(d, name):
    return pd.read_csv(os.path.join(d, name), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')


mode = 'control' if A.control else 'clean'
print(f'V224FF_CHECK {A.out} mode {mode} w_f {A.wf}', flush=True)
logs = [p for p in glob.glob(P('*.log')) if 'cufile' not in os.path.basename(p)]
entries, text = None, ''
if logs:
    try:
        entries = json.load(open(logs[0], encoding='utf-8'))
        text = '\n'.join(e.get('data', '') for e in entries)
    except Exception:
        text = open(logs[0], encoding='utf-8', errors='replace').read()
chk('kernel log present', bool(text), os.path.basename(logs[0]) if logs else '')

# ---- (1) the V224-F gate ---------------------------------------------------------------------------------------------------------
args = [sys.executable, V224F_CHECK, A.out, '--wf', str(A.wf), '--sample', A.sample] + (['--ref-fleet', A.ref_fleet] if A.ref_fleet else [])
r = subprocess.run(args, capture_output=True, text=True, encoding='utf-8')
print('\n'.join('    | ' + ln for ln in r.stdout.strip().splitlines()), flush=True)
fails = [ln.strip()[5:].strip() for ln in r.stdout.splitlines() if ln.startswith('  FAIL ')]
oks = [ln for ln in r.stdout.splitlines() if ln.startswith('  OK   ')]
chk('v224f_check.py: every check passes except its exact-mode V224 sub-gate (replaced by the flip-mode sub-gate below)',
    len(fails) == 1 and fails[0].startswith(EXPECTED_V224F_FAIL) and len(oks) >= 20, {'fails': fails, 'oks': len(oks)})

# ---- (2) v224_check --flip on the V224 view ---------------------------------------------------------------------------------------
view = tempfile.mkdtemp(prefix='v224ff_view_')
ok_view = False
try:
    if entries is not None:
        k = next(i for i, e in enumerate(entries) if 'V224_DONE' in e.get('data', ''))
        json.dump(entries[:k + 1], open(os.path.join(view, os.path.basename(logs[0])), 'w', encoding='utf-8'))
    else:
        cut = text.find('\n', text.find('V224_DONE'))
        open(os.path.join(view, os.path.basename(logs[0])), 'w', encoding='utf-8').write(text[:cut if cut > 0 else None])
    for f in ['blend_receipt.json', 'canary_receipt.json', 'submission_receipt.json', 'ours_sub.csv', 'raptor_raw.csv',
              'raptor224_sub.csv', 'raptor_sub.csv'] + [os.path.basename(p) for p in glob.glob(P('worker*_log.txt'))]:
        if os.path.isfile(P(f)):
            shutil.copyfile(P(f), os.path.join(view, f))
    shutil.copyfile(P('submission_V224.csv'), os.path.join(view, 'submission.csv'))
    r = subprocess.run([sys.executable, V224_CHECK, view, '--flip', '--ref', A.ref_exact, '--sample', A.sample],
                       capture_output=True, text=True, encoding='utf-8')
    print('\n'.join('    | ' + ln for ln in r.stdout.strip().splitlines()), flush=True)
    ok_view = r.returncode == 0 and 'V224_CHECK PASS' in r.stdout
except Exception as e:
    print('    | view error', repr(e), flush=True)
chk('V224 part (log up to V224_DONE, submission_V224.csv) passes v224_check.py --flip --ref V224-exact', ok_view)
shutil.rmtree(view, ignore_errors=True)

# ---- (3) the time-aware flip -------------------------------------------------------------------------------------------------------
ff = line_json(text, 'FLIP_TTA_FORCED') or {}
drv_sha = hashlib.sha256(open(DRIVER, 'rb').read()).hexdigest()
want_budget_src = 'env V224FF_FLIP_BUDGET_H' if A.control else 'built in'
ctl_b = None
if A.control:
    ctl_line = next((ln for ln in text.splitlines() if 'CONTROL_FLIP_BUDGET_H ' in ln), '')
    try:
        ctl_b = float(ctl_line.split('CONTROL_FLIP_BUDGET_H ', 1)[1].split()[0])
    except Exception:
        ctl_b = None
want_b = ctl_b if A.control else A.flip_budget_h
chk('FLIP_TTA_FORCED: driver compact_infer_ta.py (sha), --flip-tta + driver args, budget %s (%s)' % (want_b, want_budget_src),
    ff.get('worker') == 'compact_infer_ta.py' and ff.get('driver_sha256') == drv_sha[:16] and want_b is not None
    and abs(float(ff.get('flip_budget_h', -1)) - want_b) < 1e-9 and ff.get('budget_source') == want_budget_src
    and '--flip-tta' in (ff.get('worker_args') or []) and '--ta-budget-end' in (ff.get('worker_args') or []),
    {k: ff.get(k) for k in ('worker', 'flip_budget_h', 'budget_source', 'driver_sha256', 'elapsed_h')})
order = ['ASSETS_READY', 'FLIP_TTA_FORCED', 'OURS_STAGE_COMPLETE', 'FLIP_TA_REPORT', 'R224_STAGE', 'V224_DONE']
if A.control:
    order.insert(1, 'CONTROL_FLIP_BUDGET_H')
pos = [text.find(m) for m in order]
chk('markers in cell order (' + ' < '.join(order) + ')', all(p >= 0 for p in pos) and pos == sorted(pos), dict(zip(order, pos)))
rep = line_json(text, 'FLIP_TA_REPORT') or {}
fr = jload(P('flip_ta_receipt.json')) or {}
W = fr.get('workers') or {}
chk('flip_ta_receipt.json: 2 R7 workers, no report error', len(W) == 2 and 'error' not in fr and 'FLIP_TA_REPORT_ERROR' not in text,
    {k: fr.get(k) for k in ('budget_h', 'flipped', 'plain', 'decisions', 'all_flipped')})
wl = {os.path.basename(p).replace('_log.txt', ''): open(p, encoding='utf-8', errors='replace').read() for p in sorted(glob.glob(P('worker*_log.txt')))}
chk('every worker log: FLIP_TA_START + FLIP_TA_DONE, no driver error / child failure',
    len(wl) == 2 and all('FLIP_TA_START' in t and 'FLIP_TA_DONE' in t and 'FLIP_TA_DRIVER_ERROR' not in t and 'FLIP_TTA_CHILD_FAILED' not in t
                         and 'FLIP_TA_PLAIN_NO_RECEIPT' not in t for t in wl.values()), list(wl))
sr = jload(P('submission_receipt.json')) or {}
chk('R7 controller ran the driver; every worker receipt has time_aware_flip', sr.get('worker') == 'compact_infer_ta.py'
    and len(sr.get('workers') or []) == 2 and all(isinstance(w.get('time_aware_flip'), dict) for w in sr['workers']))
sample = pd.read_csv(A.sample, dtype={'StudyInstanceUID': str})
cols = list(sample.columns); labels = cols[1:]; ids = sample['StudyInstanceUID'].tolist()
ours = csv_(A.out, 'ours_sub.csv').loc[ids, labels]
o_flip = csv_(A.ref_flip, 'ours_sub.csv').loc[ids, labels]
o_exact = csv_(A.ref_exact, 'ours_sub.csv').loc[ids, labels]
if A.control:
    sw = [t for t in W.values() if t.get('decision') == 'switch']
    nn = [t for t in W.values() if t.get('decision') == 'none']
    s0 = sw[0] if sw else {}
    chk('control: exactly one worker switched at study 1 by the lower-bound rule, killed (-9), plain child rc 0',
        len(sw) == 1 and len(nn) == 1 and s0.get('decided_at_study') == 1 and s0.get('disabled_at_study') == 1 and s0.get('flipped') == 1
        and s0.get('plain') == s0.get('studies', 0) - 1 >= 1 and s0.get('rule') == 'lower bound (k < min_studies)'
        and s0.get('kill_rc') == -9 and s0.get('plain_rc') == 0 and s0.get('projected_end_h', 0) > s0.get('budget_end_h', 1),
        {k: s0.get(k) for k in ('decision', 'decided_at_study', 'disabled_at_study', 'flipped', 'plain', 'studies', 'rule',
                                'kill_rc', 'plain_rc', 'torn_last_line_dropped', 'decided_h', 'budget_end_h', 'end_h')})
    chk('control: the 1-study worker kept flip (nothing left to switch)', len(nn) == 1 and nn[0].get('flipped') == nn[0].get('studies') == 1,
        {k: (nn[0] if nn else {}).get(k) for k in ('decision', 'flipped', 'studies')})
    chk('control: FLIP_TTA_DISABLED_AT_STUDY 1 in the notebook log and in the switching worker log, FLIP_TA_PLAIN_START there',
        'FLIP_TTA_DISABLED_AT_STUDY 1 ' in text and sum('FLIP_TTA_DISABLED_AT_STUDY 1 ' in t and 'FLIP_TA_PLAIN_START' in t for t in wl.values()) == 1)
    # flipped = the switching worker's flipped_uids + every study of a worker that kept flip (its compact_infer per_study rows)
    flip_set = set(s0.get('flipped_uids') or [])
    for w in sr.get('workers') or []:
        if (w.get('time_aware_flip') or {}).get('decision') == 'none':
            flip_set |= set(p[0] for p in (w.get('per_study') or []))
    plain_set = set(ids) - flip_set
    chk('control: 2 studies flipped, 1 plain, partition of the 3 test ids', len(flip_set) == 2 and len(plain_set) == 1 and flip_set | plain_set == set(ids),
        {'flipped': sorted(flip_set), 'plain': sorted(plain_set)})
    df = {u: float(np.abs(ours.loc[u].values - o_flip.loc[u].values).max()) for u in ids}
    de = {u: float(np.abs(ours.loc[u].values - o_exact.loc[u].values).max()) for u in ids}
    chk('control: flipped rows == the V224-FLIP commit R7 rows (%g)' % TOL_COMPILE, all(df[u] <= TOL_COMPILE for u in flip_set),
        {u[-6:]: round(df[u], 6) for u in flip_set})
    chk('control: plain rows == the V224-exact commit R7 rows (%g)' % TOL_COMPILE, all(de[u] <= TOL_COMPILE for u in plain_set),
        {u[-6:]: round(de[u], 6) for u in plain_set})
    dl = {u: float(np.abs(lg(ours.loc[u].values) - lg(o_flip.loc[u].values)).max()) for u in plain_set}
    chk('control: the plain row is NOT the flip row (some R7 column differs by > 1e-3 in logit)', all(v > 1e-3 for v in dl.values()),
        {u[-6:]: round(v, 4) for u, v in dl.items()})
else:
    chk('clean: every worker kept flip on every study (decision none), 3 flipped, 0 plain, no FLIP_TTA_DISABLED',
        len(W) == 2 and all(t.get('decision') == 'none' and t.get('flipped') == t.get('studies') for t in W.values())
        and rep.get('flipped') == 3 and rep.get('plain') == 0 and rep.get('all_flipped') is True
        and 'FLIP_TTA_DISABLED_AT_STUDY' not in text and not any('FLIP_TTA_DISABLED_AT_STUDY' in t for t in wl.values()),
        {k: rep.get(k) for k in ('flipped', 'plain', 'decisions', 'all_flipped')})
    d = float(np.abs(ours.values - o_flip.values).max())
    chk('clean: R7 ours_sub == the V224-FLIP commit R7 (%g)' % TOL_COMPILE, d <= TOL_COMPILE, 'max diff %.3g' % d)
    h = csv_(A.out, 'submission_V224.csv').loc[ids, labels]
    hr = csv_(A.ref_flip, 'submission.csv').loc[ids, labels]
    d = float(np.abs(h.values - hr.values).max())
    chk('clean: submission_V224.csv == the V224-FLIP commit submission (%g)' % TOL_BLEND, d <= TOL_BLEND, 'max diff %.3g, byte-identical %s'
        % (d, open(P('submission_V224.csv'), 'rb').read() == open(os.path.join(A.ref_flip, 'submission.csv'), 'rb').read()))

# ---- (4) end-to-end independent recompute ------------------------------------------------------------------------------------------
raw = csv_(A.out, 'raptor_raw.csv').loc[ids, labels]
r224 = csv_(A.out, 'raptor224_sub.csv').loc[ids, labels]
rap_fail = (raw.values == 0.5).all(axis=1)
chk('recompute inputs: no Raptor-failed row (receipts above assert 0 R7 fallback)', not rap_fail.any())
H64 = np.zeros((len(ids), len(labels)))
for j, c in enumerate(labels):
    rr = 0.65 * pct(raw[c].values) + 0.35 * pct(r224[c].values)
    H64[:, j] = 0.5 * rr + 0.5 * pct(ours[c].values)
H32 = H64.astype(np.float32).astype(np.float64)
hv = csv_(A.out, 'submission_V224.csv').loc[ids, labels].values.astype(np.float64)
d = float(np.abs(hv - H32).max())
chk('recompute: submission_V224.csv == V224 recipe from raptor_raw / raptor224_sub / ours_sub (float32)', d <= TOL_BLEND, 'max diff %.3g' % d)
try:
    parts = [np.load(p, allow_pickle=False) for p in sorted(glob.glob(P('_fleet_*.npz')))]
    Pm = np.concatenate([z['P'] for z in parts], axis=1).astype(np.float64)
    sids = np.concatenate([z['sids'] for z in parts]).astype(str)
    F = pd.DataFrame(Pm.mean(0), index=sids, columns=labels).loc[ids].values
    final = np.stack([(1.0 - A.wf) * pct(H32[:, j]) + A.wf * pct(F[:, j]) for j in range(len(labels))], 1).astype(np.float32)
    S = pd.read_csv(P('submission.csv'), dtype={'StudyInstanceUID': str})
    ok_ids = S['StudyInstanceUID'].tolist() == ids and list(S.columns) == cols
    d = float(np.abs(S[labels].values.astype(np.float64) - final.astype(np.float64)).max())
    chk('END-TO-END: submission.csv == (1 - w) pct(V224 recipe) + w pct(fleet npz member mean), %d members' % Pm.shape[0],
        ok_ids and d <= TOL_BLEND and Pm.shape[0] == 20, 'max diff %.3g' % d)
except Exception as e:
    chk('END-TO-END recompute', False, repr(e))

# ---- runtime info --------------------------------------------------------------------------------------------------------------------
oc = line_json(text, 'OURS_STAGE_COMPLETE') or {}
br = jload(P('blend_receipt.json')) or {}
hf = jload(P('hf_receipt.json')) or {}
info = {'r7_seconds': oc.get('seconds'), 'r7_per_study': oc.get('per_study_mean'), 'v224_total_s': br.get('total_seconds'),
        'hf_total_s': hf.get('total_seconds'), 'workers': {k: {x: v.get(x) for x in ('decision', 'flipped', 'plain', 'start_h', 'end_h',
                                                                                    'phase_f_seconds', 'plain_seconds')} for k, v in W.items()}}
print('  info runtime', json.dumps(info), flush=True)
n_ok = sum(R)
print(f'V224FF_CHECK {"PASS" if all(R) else "FAIL"} {n_ok}/{len(R)} mode {mode}', flush=True)
sys.exit(0 if all(R) else 1)
