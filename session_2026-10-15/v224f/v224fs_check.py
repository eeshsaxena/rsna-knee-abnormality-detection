#!/usr/bin/env python
# v224fs_check.py (2026-10-15 dir, 10-14, Claude subagent; R7-teacher line, PLAN_TEACHER FX7/FX8): commit gate for V224-F-S
# kernels built by build_v224fs.py (copy of v224f_check.py; the V224 part, receipts and fleet checks are the same; h_check is
# NOT used because F is the two-group mean, not the fleet solo). Reads a fetched kernel output dir. Prints V224FS_CHECK PASS/FAIL
# n/m; exit 0 only on PASS.
#   usage: python v224fs_check.py OUT --wf 0.20 --build KERNEL_DIR/BUILD_V224FS.json [--ref-v224 OUT_V224_EXACT]
#                                     [--ref-fleet OUT_V224F20 (its fleet solo = the K14 group mean)] [--sample ...]
# FX7: independent recompute of F = (1 - s) mean(K14 rows) + s mean(S rows) from the workers' npz (names from the build json)
#      == submission_fleet2g.csv (1e-6), != the equal mean (unless REPLACE); final == (1 - w) pct(V224) + w pct(F) (1e-6);
#      fleet solo == the equal member mean (1e-6); with --ref-fleet the K14 group mean == the V224-F20 commit fleet solo (1e-3).
# FX8: V224F_SAVED / fleet K2_CFG: WALL_H 8.3, MIN_START_MIN 198 (start budget 5.0 h; fix 10-15: an ff build may use 5.1 h / 192,
#      the start budget defaults to the build json), members 25 (REPLACE 5); a hidden-wall
#      estimate for this member count is printed (info, MODEL).
# The sections below marked 'clean' are the only ones that apply (a V224-F-S kernel has no control variant).
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
ap.add_argument('--start-budget-h', type=float, default=None, help='default: the build json value (fix 10-15: ff may be 5.1)')
ap.add_argument('--build', required=True, help='BUILD_V224FS.json of the pushed kernel')
ap.add_argument('--wall-h', type=float, default=8.3)
ap.add_argument('--members', type=int, default=None)
ap.add_argument('--sample', default=os.path.join(ROOT, '_data', 'sample_submission.csv'))
A = ap.parse_args()
B = json.load(open(A.build, encoding='utf-8'))
A.members = len(B['k14']) + len(B['s'])
if A.start_budget_h is None:
    A.start_budget_h = B['start_budget_h']
assert A.wf == B['w_f'] and A.start_budget_h == B['start_budget_h'] and A.wall_h == B['wall_h'] and not A.control
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
print(f'V224FS_CHECK {A.out} mode {mode} w_f {A.wf} tag {B["tag"]} members {A.members} (K14 {len(B["k14"])} + S {len(B["s"])}) '
      f's_share {B["s_share"]}{" STAND-IN BUILD (offline test only)" if B.get("standin") else ""}', flush=True)
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

# ---- (2) h_check is not used (F = the two-group mean, recomputed below) ---------------------------------------------------------

# ---- V224-F markers after V224_DONE -------------------------------------------------------------------------------------------
tail = text[text.find('V224_DONE'):] if 'V224_DONE' in text else ''
order = ['V224_DONE', 'V224F_SAVED', '[fn-ids', 'RESULT OK ', '[k12-merge', 'RESULT OK merged fleet dir', '[s-merge',
         'RESULT OK merged fleet dir + S: 30 checkpoints', 'V224F_CLOCK', 'STAGE fleet start',
         'RESULT OK fleet stage clean: %d member' % A.members, 'HF_ENV_RESTORED', 'V224FS_F2G', 'HF_BLEND_OK', 'HF_COMPLETE', 'V224F_DONE']
if A.control:
    order.insert(11, 'CANARY_FIRED fleet not used')
pos, p0 = [], 0
for m in order:
    p = tail.find(m, p0)
    pos.append(p)
    if p >= 0:
        p0 = p
chk('V224-F-S markers present, in cell order after V224_DONE', all(p >= 0 for p in pos), {m: p for m, p in zip(order, pos) if p < 0})
bad = ['Traceback', 'V224F_H_FROM_MEMORY', 'RESULT FAIL', 'UNEXPECTED', 'PARITY FAIL', 'RESTORE FAILED']
bad += ['HF_BLEND_OK', 'RESULT OK fleet stage clean'] if A.control else ['HF_FALLBACK', 'CANARY_FIRED', 'NOT CLEAN', 'BLEND_FALLBACK']
bad += ['R224_STAGE', 'R224_SKIPPED', 'R224_TIMEOUT', 'R224_UNUSABLE', 'NOTEBOOK_COMPLETE']   # her stages must not run again
after = tail[tail.find('\n'):] if '\n' in tail else ''          # scan after the V224_DONE line (it prints "R224_OK")
hits = [m for m in bad if m in after]
chk('no fallback / error / foreign marker after V224_DONE', not hits, hits)
sv = line_json(tail, 'V224F_SAVED') or {}
want_sv = {'tag': B['tag'], 'w_f': A.wf, 'wall_h': A.wall_h, 'min_start_min': MIN_START, 'members': A.members, 'control': False,
           'start_budget_h': A.start_budget_h, 'groups': {'K14': len(B['k14']), 'S': len(B['s'])}, 's_share': B['s_share'],
           's_ds': B['s_ds']}
chk('build: MIN_START_MIN %.1f (start budget %.1f h, wall 8.3 h; f 5.0 h = 198, ff 5.0 / 5.1 h = 198 / 192)' % (MIN_START, A.start_budget_h),
    MIN_START == B['min_start_min'] and (B.get('base', 'f'), MIN_START) in (('f', 198.0), ('ff', 198.0), ('ff', 192.0)), MIN_START)
chk('V224F_SAVED: knobs', all(sv.get(k) == v for k, v in want_sv.items()) and 'save_error' not in sv and len(sv.get('sha256', '')) == 64,
    {k: sv.get(k) for k in list(want_sv) + ['elapsed_h_at_save', 'fleet_may_start']})
chk('V224F_SAVED: fleet_may_start == (not control)', sv.get('fleet_may_start') is (not A.control))
ck = line_json(tail, 'V224F_CLOCK') or {}
chk('V224F_CLOCK: T0 pinned to NB_T0', ck.get('T0') is not None and ck.get('T0') == ck.get('NB_T0'), ck)

# ---- files, receipts ---------------------------------------------------------------------------------------------------------
sample = pd.read_csv(A.sample, dtype={'StudyInstanceUID': str})
cols = list(sample.columns); labels = cols[1:]; ids = sample['StudyInstanceUID'].tolist()
has = {f: os.path.isfile(P(f)) for f in ('submission.csv', 'submission_V224.csv', 'submission_H.csv', 'submission_fleetsolo.csv',
                                          'hf_receipt.json', 'v224f_receipt.json', 'fleet_receipt.json', 'submission_fleet2g.csv')}
want_has = dict(has, **{'submission.csv': True, 'submission_V224.csv': True, 'hf_receipt.json': True, 'v224f_receipt.json': True,
                        'fleet_receipt.json': True, 'submission_H.csv': True, 'submission_fleetsolo.csv': True, 'submission_fleet2g.csv': True})
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
    and hf.get('h_sha256') == sv.get('sha256') and hf.get('tag') == B['tag'] and hf.get('s_share') == B['s_share']
    and hf.get('groups') == {'K14': len(B['k14']), 'S': len(B['s'])}, {k: hf.get(k) for k in ('tag', 'w_f', 'h_source', 'mode', 's_share', 'groups')})
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
    chk('fleet receipt: K2_CFG wall %.1f h, MIN_START_MIN %.1f, members as built (K14 + S)' % (A.wall_h, MIN_START),
        cfg.get('WALL_H') == A.wall_h and cfg.get('MIN_START_MIN') == MIN_START and len(cfg.get('MEMBERS') or []) == A.members
        and sorted(cfg.get('MEMBERS') or []) == sorted(fr.get('names') or []) == sorted(B['k14'] + B['s']),
        {k: cfg.get(k) for k in ('WALL_H', 'MIN_START_MIN')})
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
    Pm = nm = mean = None                                     # fix 10-15: mean is set only if the npz pass the names check below
    try:
        Pm = np.concatenate([z['P'] for z in parts], axis=1).astype(np.float64)
        sids = np.concatenate([z['sids'] for z in parts]).astype(str)
        nm = [str(x) for x in parts[0]['names']]
        assert all([str(x) for x in z['names']] == nm for z in parts) and sorted(nm) == sorted(B['k14'] + B['s']), 'npz names'
        TG = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
              "Baker's", 'Contusion', 'Fracture']                  # the fleet cell's _K2_TARGETS order (npz columns)
        Pm = Pm[:, :, [TG.index(c) for c in labels]]
        mean = pd.DataFrame(Pm.mean(0), index=sids, columns=labels).loc[ids]
        Fs = pd.read_csv(P('submission_fleetsolo.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, labels]
        d = float(np.abs(mean.values - Fs.values).max())
        chk('fleet solo == member mean from the worker npz (%d parts, %d members)' % (len(parts), Pm.shape[0]),
            d <= TOL_BLEND and Pm.shape[0] == A.members and np.isfinite(Pm).all(), 'max diff %.3g' % d)
    except Exception as e:
        chk('fleet solo == member mean from the worker npz', False, repr(e))
        Fs = None
    # FX7: the two-group F, recomputed here independently from the npz and the build's group lists
    F2 = None
    try:
        assert mean is not None, 'not run: the worker npz failed the member-mean check above (missing npz or member names != the build)'
        gS = Pm[[nm.index(m) for m in B['s']]].mean(0)
        f2 = gS if not B['k14'] else (1.0 - B['s_share']) * Pm[[nm.index(m) for m in B['k14']]].mean(0) + B['s_share'] * gS
        assert np.isfinite(f2).all(), 'non-finite member predictions (coverage must be 1 here)'
        F2 = pd.DataFrame(f2, index=sids, columns=labels).loc[ids]
        G2 = pd.read_csv(P('submission_fleet2g.csv'), dtype={'StudyInstanceUID': str})
        chk('submission_fleet2g.csv ids / columns == sample', list(G2.columns) == cols and G2['StudyInstanceUID'].tolist() == ids)
        d2 = float(np.abs(F2.values - G2.set_index('StudyInstanceUID').loc[ids, labels].values).max())
        chk('submission_fleet2g.csv == (1 - %.2f) mean(K14 %d) + %.2f mean(S %d), recomputed here' % (B['s_share'], len(B['k14']),
            B['s_share'], len(B['s'])), d2 <= TOL_BLEND, 'max diff %.3g' % d2)
        dq = float(np.abs(F2.values - mean.values).max())
        chk('two-group F differs from the equal member mean' + (' (REPLACE: equal)' if not B['k14'] else ''),
            (dq <= TOL_BLEND) if not B['k14'] else (dq > 1e-5), 'max |F2 - equal mean| %.3g' % dq)
        chk('hf_receipt f2g: eq-mean check <= 1e-6, sha of submission_fleet2g.csv',
            (hf.get('f2g') or {}).get('eq_mean_vs_fleetsolo_max', 1) <= 1e-6 and (hf.get('f2g') or {}).get('sha256') == sha(P('submission_fleet2g.csv')),
            hf.get('f2g'))
        if A.ref_fleet and B['k14']:
            Kg = pd.DataFrame(Pm[[nm.index(m) for m in B['k14']]].mean(0), index=sids, columns=labels).loc[ids]
            Fr = pd.read_csv(os.path.join(A.ref_fleet, 'submission_fleetsolo.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, labels]
            dd = np.abs(Kg.values - Fr.values)
            chk('K14 group mean == the V224-F20 commit fleet solo within %g (same 20 members)' % TOL_COMPILE, float(dd.max()) <= TOL_COMPILE,
                'max diff %.3g, mean %.3g' % (dd.max(), dd.mean()))
    except Exception as e:
        chk('two-group F recompute', False, repr(e))
    # final blend recompute
    if F2 is not None:
        Hh = H.set_index('StudyInstanceUID').loc[ids, labels]
        exp = np.stack([(1.0 - A.wf) * pct(Hh[c].values) + A.wf * pct(F2[c].values) for c in labels], 1)
        d = float(np.abs(V - exp).max())
        chk('submission == (1 - w) pct(V224) + w pct(F two-group), recomputed here', d <= TOL_BLEND, 'max diff %.3g' % d)
        same_as_h = float(np.abs(V - Hh.values).max())
        print(f'  info final vs V224 output max |diff| {same_as_h:.4f}; spearman H vs F {hf.get("spearman_H_vs_F")}', flush=True)
    chk('submission_H.csv == submission_V224.csv (bytes)', has['submission_H.csv'] and
        open(P('submission_H.csv'), 'rb').read() == open(P('submission_V224.csv'), 'rb').read())
    chk('hf_receipt: mode HF, blanked []', hf.get('mode') == 'HF' and hf.get('blanked') == [], {k: hf.get(k) for k in ('mode', 'blanked')})
    chk('v224f_receipt: fired [], use_f True, mode HF, s_share, groups', vr.get('fired') == [] and vr.get('use_f') is True
        and vr.get('mode') == 'HF' and vr.get('s_share') == B['s_share'] and vr.get('groups') == {'K14': len(B['k14']), 'S': len(B['s'])})
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
if b and fr.get('stage_min') is not None:                      # FX8 hidden-wall estimate for this member count (MODEL)
    g = max(sum(x.values()) / 1000.0 for x in b); fixed = fr['stage_min'] - max(fr.get('worker_loop_min') or [0])
    for nh in (1200, 1300):
        cen = (fixed * 60 + int(np.ceil(nh / 2)) * g) / 3600; slow = (fixed * 60 + int(np.ceil(nh / 2)) * g * 1.15) / 3600
        print(f'  info wall MODEL {nh} studies: fleet central {cen:.2f} h, slow {slow:.2f} h, slow session {slow * 1.3:.2f} h | kernel '
              f'central {3.0 + cen:.2f} h, slow end x1.3 {(3.6 + slow) * 1.3:.2f} h | worst start {A.start_budget_h} h + slow-session '
              f'fleet = {A.start_budget_h + slow * 1.3:.2f} h vs the {A.wall_h} h fleet wall', flush=True)
n_ok = sum(R)
print(f'V224FS_CHECK {"PASS" if all(R) else "FAIL"} {n_ok}/{len(R)} tag {B["tag"]}', flush=True)
sys.exit(0 if all(R) else 1)
