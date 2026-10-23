#!/usr/bin/env python
# v224f_check.py (2026-10-15 dir, Claude subagent): commit gate for V224-F kernels built by build_v224f.py. Reads a fetched
# kernel output dir (kaggle kernels output eeshsaxena/rsna-sub-v224f20 -p OUT). Prints V224F_CHECK PASS/FAIL n/m; exit 0 only
# on PASS.
#   usage: python v224f_check.py OUT --wf 0.20 [--control] [--ref-v224 OUT_V224_EXACT] [--ref-fleet OUT_HFR7RR]
#                                    [--start-budget-h 5.5] [--wall-h 8.3] [--sample sample_submission.csv]
# Sub-gates (subprocesses, their own PASS lines are echoed):
#   (1) session_2026-10-14/v224/v224_check.py, mode exact, on a VIEW of OUT: the kernel log cut after V224_DONE and
#       submission.csv := submission_V224.csv (the saved V224-exact output). So the V224 part of this kernel passes the full
#       V224 gate (her cell order, R7 / r224 / blend receipts, canary clean, independent V224 recompute 1e-6).
#   (2) session_2026-10-12/hf/h_check.py: --kind hf --wf W --members 20 (clean) or --kind hv (control).
# Own checks: V224-F markers in cell order after V224_DONE; V224F_SAVED / V224F_CLOCK values; the fleet receipt (clean, 20
# members on every worker, no hard stop, coverage 1, DICOM re-encode parity, TTA AB, the K2_CFG wall / start guard); fleet solo ==
# the member mean recomputed from the workers' npz (1e-6); final submission == (1 - w) pct(V224) + w pct(F) recomputed here
# (average-tie percentile rank, 1e-6); receipts; with --ref-v224: submission_V224.csv == the V224-exact commit's submission
# (1e-6); with --ref-fleet: fleet solo == the HFR7-RR commit's fleet solo within the compile tolerance 1e-3 (same 3 studies, same 20
# members; here the fleet runs after R7's timm 1.0.25 / pydicom 3.0.1 wheels, there before them).
# --control (fleet guard forced to refuse): RESULT FLEET STAGE NOT CLEAN (stage not started), HF_FALLBACK + CANARY_FIRED,
# no fleet solo file, Synovitis == 0.5 in every row, the other 11 columns string-identical to submission_V224.csv.
import os, sys, json, glob, shutil, hashlib, argparse, subprocess, tempfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
V224_CHECK = os.path.join(ROOT, 'session_2026-10-14', 'v224', 'v224_check.py')
H_CHECK = os.path.join(ROOT, 'session_2026-10-12', 'hf', 'h_check.py')
ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--wf', type=float, required=True)
ap.add_argument('--control', action='store_true')
ap.add_argument('--ref-v224', default=None, help='V224-exact commit output dir')
ap.add_argument('--ref-fleet', default=None, help='HFR7-RR commit output dir (its submission_fleetsolo.csv)')
ap.add_argument('--start-budget-h', type=float, default=5.5)
ap.add_argument('--wall-h', type=float, default=8.3)
ap.add_argument('--members', type=int, default=20)
ap.add_argument('--sample', default=os.path.join(ROOT, '_data', 'sample_submission.csv'))
A = ap.parse_args()
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
R = []
TOL_BLEND, TOL_COMPILE = 1e-6, 1e-3
MIN_START = 600.0 if A.control else round((A.wall_h - A.start_budget_h) * 60.0, 1)


def chk(name, ok, info=''):
    R.append(bool(ok))
    print(f'  {"OK  " if ok else "FAIL"} {name} {info}', flush=True)


def P(*a):
    return os.path.join(A.out, *a)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


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
    v = np.asarray(v, dtype=np.float64)          # independent average-tie percentile rank (no pandas rank)
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


mode = 'control' if A.control else 'clean'
print(f'V224F_CHECK {A.out} mode {mode} w_f {A.wf}', flush=True)
logs = [p for p in glob.glob(P('*.log')) if 'cufile' not in os.path.basename(p)]
entries = None
if logs:
    try:
        entries = json.load(open(logs[0], encoding='utf-8'))
        text = '\n'.join(e.get('data', '') for e in entries)
    except Exception:
        text = open(logs[0], encoding='utf-8', errors='replace').read()
else:
    text = ''
chk('kernel log present', bool(text), os.path.basename(logs[0]) if logs else '')

# ---- (1) V224 sub-gate on a view: log cut after V224_DONE, submission.csv := submission_V224.csv ----------------------------
view = tempfile.mkdtemp(prefix='v224f_view_')
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
    r = subprocess.run([sys.executable, V224_CHECK, view, '--sample', A.sample], capture_output=True, text=True, encoding='utf-8')
    print('\n'.join('    | ' + ln for ln in r.stdout.strip().splitlines()), flush=True)
    ok_view = r.returncode == 0 and 'V224_CHECK PASS' in r.stdout
except Exception as e:
    print('    | view error', repr(e), flush=True)
chk('V224 part (log up to V224_DONE, submission_V224.csv) passes v224_check.py exact', ok_view)
shutil.rmtree(view, ignore_errors=True)

# ---- (2) h_check -------------------------------------------------------------------------------------------------------------
hargs = ['--kind', 'hv'] if A.control else ['--kind', 'hf', '--wf', str(A.wf), '--members', str(A.members)]
r = subprocess.run([sys.executable, H_CHECK, A.out] + hargs, capture_output=True, text=True, encoding='utf-8')
print('\n'.join('    | ' + ln for ln in r.stdout.strip().splitlines()), flush=True)
chk('h_check.py ' + ' '.join(hargs), r.returncode == 0 and 'HCHECK PASS' in r.stdout)

# ---- V224-F markers after V224_DONE -------------------------------------------------------------------------------------------
tail = text[text.find('V224_DONE'):] if 'V224_DONE' in text else ''
order = ['V224_DONE', 'V224F_SAVED', '[fn-ids', 'RESULT OK ', '[k12-merge', 'RESULT OK merged fleet dir', 'V224F_CLOCK',
         'STAGE fleet start', 'RESULT FLEET STAGE NOT CLEAN' if A.control else 'RESULT OK fleet stage clean: %d member' % A.members,
         'HF_ENV_RESTORED', 'HF_FALLBACK' if A.control else 'HF_BLEND_OK', 'HF_COMPLETE', 'V224F_DONE']
if A.control:
    order.insert(11, 'CANARY_FIRED fleet not used')
pos, p0 = [], 0
for m in order:
    p = tail.find(m, p0)
    pos.append(p)
    if p >= 0:
        p0 = p
chk('V224-F markers present, in cell order after V224_DONE', all(p >= 0 for p in pos), {m: p for m, p in zip(order, pos) if p < 0})
bad = ['Traceback', 'V224F_H_FROM_MEMORY', 'RESULT FAIL', 'UNEXPECTED', 'PARITY FAIL', 'RESTORE FAILED']
bad += ['HF_BLEND_OK', 'RESULT OK fleet stage clean'] if A.control else ['HF_FALLBACK', 'CANARY_FIRED', 'NOT CLEAN', 'BLEND_FALLBACK']
bad += ['R224_STAGE', 'R224_SKIPPED', 'R224_TIMEOUT', 'R224_UNUSABLE', 'NOTEBOOK_COMPLETE']   # her stages must not run again
after = tail[tail.find('\n'):] if '\n' in tail else ''          # scan after the V224_DONE line (it prints "R224_OK")
hits = [m for m in bad if m in after]
chk('no fallback / error / foreign marker after V224_DONE', not hits, hits)
sv = line_json(tail, 'V224F_SAVED') or {}
want_sv = {'w_f': A.wf, 'wall_h': A.wall_h, 'min_start_min': MIN_START, 'members': A.members, 'control': bool(A.control)}
chk('V224F_SAVED: knobs', all(sv.get(k) == v for k, v in want_sv.items()) and 'save_error' not in sv and len(sv.get('sha256', '')) == 64,
    {k: sv.get(k) for k in list(want_sv) + ['elapsed_h_at_save', 'fleet_may_start']})
chk('V224F_SAVED: fleet_may_start == (not control)', sv.get('fleet_may_start') is (not A.control))
ck = line_json(tail, 'V224F_CLOCK') or {}
chk('V224F_CLOCK: T0 pinned to NB_T0', ck.get('T0') is not None and ck.get('T0') == ck.get('NB_T0'), ck)

# ---- files, receipts ---------------------------------------------------------------------------------------------------------
sample = pd.read_csv(A.sample, dtype={'StudyInstanceUID': str})
cols = list(sample.columns); labels = cols[1:]; ids = sample['StudyInstanceUID'].tolist()
has = {f: os.path.isfile(P(f)) for f in ('submission.csv', 'submission_V224.csv', 'submission_H.csv', 'submission_fleetsolo.csv',
                                          'hf_receipt.json', 'v224f_receipt.json', 'fleet_receipt.json')}
want_has = dict(has, **{'submission.csv': True, 'submission_V224.csv': True, 'hf_receipt.json': True, 'v224f_receipt.json': True,
                        'fleet_receipt.json': True, 'submission_H.csv': not A.control, 'submission_fleetsolo.csv': not A.control})
chk('output files', has == want_has, {f: v for f, v in has.items() if v != want_has[f]})
chk('submission_V224.csv sha == V224F_SAVED sha', has['submission_V224.csv'] and sha(P('submission_V224.csv')) == sv.get('sha256'))
hf = jload(P('hf_receipt.json')) or {}
vr = jload(P('v224f_receipt.json')) or {}
fr = jload(P('fleet_receipt.json')) or {}
cr = jload(P('canary_receipt.json')) or {}
S = pd.read_csv(P('submission.csv'), dtype={'StudyInstanceUID': str})
H = pd.read_csv(P('submission_V224.csv'), dtype={'StudyInstanceUID': str})
chk('submission columns / ids == sample (order)', list(S.columns) == cols and S['StudyInstanceUID'].tolist() == ids, len(S))
V = S[labels].values.astype(np.float64)
chk('submission finite in [0,1]', np.isfinite(V).all() and (V >= 0).all() and (V <= 1).all())
chk('submission sha == hf_receipt submission_sha256', hf.get('submission_sha256') == sha(P('submission.csv')))
chk('hf_receipt: tag / w_f / H source + sha', hf.get('w_f') == A.wf and hf.get('h_source') == 'submission_V224.csv'
    and hf.get('h_sha256') == sv.get('sha256') and hf.get('tag') == ('v224fctl' if A.control else 'v224f%02d' % round(A.wf * 100)),
    {k: hf.get(k) for k in ('tag', 'w_f', 'h_source', 'mode')})
chk('canary receipt (V224 part): r224 used, nothing fired', cr.get('r224_used') is True and cr.get('fired') == [], cr)
if ref := A.ref_v224:
    Ref = pd.read_csv(os.path.join(ref, 'submission.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, labels]
    d = float(np.abs(H.set_index('StudyInstanceUID').loc[ids, labels].values - Ref.values).max())
    chk('submission_V224.csv == the V224-exact commit submission', d <= TOL_BLEND, 'max diff %.3g, byte-identical %s'
        % (d, open(P('submission_V224.csv'), 'rb').read() == open(os.path.join(ref, 'submission.csv'), 'rb').read()))

si = labels.index('Synovitis')
if A.control:
    chk('fleet receipt: not clean, stage not started by the time guard', fr.get('clean') is False and 'stage not started' in str(fr.get('why'))
        and (fr.get('config') or {}).get('MIN_START_MIN') == MIN_START and (fr.get('config') or {}).get('WALL_H') == A.wall_h,
        {'clean': fr.get('clean'), 'why': str(fr.get('why'))[:120]})
    chk('hf_receipt: mode H_only, blanked [Synovitis]', hf.get('mode') == 'H_only' and hf.get('blanked') == ['Synovitis']
        and 'fleet stage not clean' in str(hf.get('why')), {k: hf.get(k) for k in ('mode', 'blanked', 'why')})
    chk('v224f_receipt: fired [fleet->Synovitis], use_f True', vr.get('fired') == ['fleet->Synovitis'] and vr.get('use_f') is True
        and vr.get('mode') == 'H_only', {k: vr.get(k) for k in ('fired', 'mode', 'use_f')})
    chk('control: Synovitis == 0.5 exactly in every row', bool((S['Synovitis'] == 0.5).all()), S['Synovitis'].tolist())
    Ss = pd.read_csv(P('submission.csv'), dtype=str); Hs = pd.read_csv(P('submission_V224.csv'), dtype=str)
    keep = [c for c in cols if c != 'Synovitis']
    chk('control: the other 11 columns string-identical to submission_V224.csv', Ss.shape == Hs.shape and (Ss[keep] == Hs[keep]).all().all())
else:
    cfg = fr.get('config') or {}
    chk('fleet receipt: clean, %d members, n_keep all %d, no hard stop' % (A.members, A.members),
        fr.get('clean') is True and fr.get('models') == A.members and fr.get('n_keep') and all(k == A.members for k in fr['n_keep'])
        and fr.get('hard_stopped') and not any(fr['hard_stopped']), {k: fr.get(k) for k in ('clean', 'models', 'n_keep', 'hard_stopped')})
    chk('fleet receipt: K2_CFG wall %.1f h, MIN_START_MIN %.1f, members as built' % (A.wall_h, MIN_START),
        cfg.get('WALL_H') == A.wall_h and cfg.get('MIN_START_MIN') == MIN_START and len(cfg.get('MEMBERS') or []) == A.members
        and sorted(cfg.get('MEMBERS') or []) == sorted(fr.get('names') or []), {k: cfg.get(k) for k in ('WALL_H', 'MIN_START_MIN')})
    chk('fleet receipt: coverage 1, views/study >= 3', fr.get('coverage') == 1.0 and (fr.get('views_per_study') or 0) >= 3.0,
        {k: fr.get(k) for k in ('coverage', 'views_per_study', 'median_filled')})
    par = fr.get('parity') or []
    p0 = par[0] if par else {}
    chk('fleet DICOM re-encode parity (shard 0) under the R7 wheels: checked, no mismatch, frac_gt8 <= .01',
        p0.get('checked', 0) > 0 and p0.get('mismatch') == [] and p0.get('frac_gt8', 1) <= 0.01
        and all(x.get('decode_mismatch') == 0 for x in par), p0)
    tta = fr.get('tta') or {}
    chk('fleet TTA: every member AB, pass B on every row', tta and set((tta.get('modes') or {}).values()) == {'AB'}
        and all(v.get('A') == tta.get('rows') == v.get('B') for v in (tta.get('fwd') or {}).values()), {'rows': tta.get('rows')})
    # fleet solo == member mean recomputed from the workers' npz
    parts = [np.load(p, allow_pickle=False) for p in sorted(glob.glob(P('_fleet_*.npz')))]
    try:
        Pm = np.concatenate([z['P'] for z in parts], axis=1).astype(np.float64)
        sids = np.concatenate([z['sids'] for z in parts]).astype(str)
        mean = pd.DataFrame(Pm.mean(0), index=sids, columns=labels).loc[ids]
        Fs = pd.read_csv(P('submission_fleetsolo.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, labels]
        d = float(np.abs(mean.values - Fs.values).max())
        chk('fleet solo == member mean from the worker npz (%d parts, %d members)' % (len(parts), Pm.shape[0]),
            d <= TOL_BLEND and Pm.shape[0] == A.members and np.isfinite(Pm).all(), 'max diff %.3g' % d)
    except Exception as e:
        chk('fleet solo == member mean from the worker npz', False, repr(e))
        Fs = None
    if A.ref_fleet and Fs is not None:
        Fr = pd.read_csv(os.path.join(A.ref_fleet, 'submission_fleetsolo.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, labels]
        dd = np.abs(Fs.values - Fr.values)
        chk('fleet solo == HFR7-RR commit fleet solo within %g (R7 wheels vs stock image)' % TOL_COMPILE, float(dd.max()) <= TOL_COMPILE,
            'max diff %.3g, mean %.3g; per label max %s' % (dd.max(), dd.mean(), {c: float('%.2g' % dd[:, j].max()) for j, c in enumerate(labels)}))
    # final blend recompute
    if Fs is not None:
        Hh = H.set_index('StudyInstanceUID').loc[ids, labels]
        exp = np.stack([(1.0 - A.wf) * pct(Hh[c].values) + A.wf * pct(Fs[c].values) for c in labels], 1)
        d = float(np.abs(V - exp).max())
        chk('submission == (1 - w) pct(V224) + w pct(F), recomputed here', d <= TOL_BLEND, 'max diff %.3g' % d)
        same_as_h = float(np.abs(V - Hh.values).max())
        print(f'  info final vs V224 output max |diff| {same_as_h:.4f}; spearman H vs F {hf.get("spearman_H_vs_F")}', flush=True)
    chk('submission_H.csv == submission_V224.csv (bytes)', has['submission_H.csv'] and
        open(P('submission_H.csv'), 'rb').read() == open(P('submission_V224.csv'), 'rb').read())
    chk('hf_receipt: mode HF, blanked []', hf.get('mode') == 'HF' and hf.get('blanked') == [], {k: hf.get(k) for k in ('mode', 'blanked')})
    chk('v224f_receipt: fired [], use_f True, mode HF', vr.get('fired') == [] and vr.get('use_f') is True and vr.get('mode') == 'HF')
    chk('no blanked column', all(len(set(np.round(V[:, j], 7))) > 1 for j in range(len(labels))),
        [labels[j] for j in range(len(labels)) if len(set(np.round(V[:, j], 7))) <= 1])

# ---- runtime info (for the hidden-wall estimate) ------------------------------------------------------------------------------
br = jload(P('blend_receipt.json')) or {}
info = {'v224_total_s': br.get('total_seconds'), 'elapsed_h_at_save': sv.get('elapsed_h_at_save'),
        'fleet_kernel_age_h': fr.get('kernel_age_h'), 'fleet_left_min': fr.get('left_min'), 'fleet_stage_min': fr.get('stage_min'),
        'fleet_workers_s': fr.get('workers_s'), 'worker_min': fr.get('worker_min'), 'worker_loop_min': fr.get('worker_loop_min'),
        'decode_s_per_study_per_thread': fr.get('decode_s_per_study_per_thread'), 'hf_total_s': hf.get('total_seconds')}
b = [x for x in (fr.get('bench') or []) if x]
if b:
    info['microbench_ms_member_study'] = {'gpu%d' % i: {'mean': round(float(np.mean(list(x.values()))), 1),
                                                        'max': round(float(np.max(list(x.values()))), 1)} for i, x in enumerate(b)}
print('  info runtime', json.dumps(info), flush=True)
n_ok = sum(R)
print(f'V224F_CHECK {"PASS" if all(R) else "FAIL"} {n_ok}/{len(R)} mode {mode}', flush=True)
sys.exit(0 if all(R) else 1)
