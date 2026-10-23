#!/usr/bin/env python
# v224s6f2_check.py (10-15, Claude subagent): s6f/v224s6f_check.py generalised with a base switch (s6f2/build_v224s6f2.py builds).
# Commit gate of <base> + S6fall (PREREG_10-15 Amendment 3 R4 gates (i)-(vi)); the S6F checks (B)-(H) are the s6f gate's, unchanged
# except the base-canary rule in (C) and the base's last marker in (A) / (G). Reads a fetched kernel output dir
# (kaggle kernels output eeshsaxena/rsna-sub-<tag> -p OUT). Prints V224S6F2_CHECK PASS/FAIL n/m; exit 0 only on PASS.
# Downloaded outputs are data: run with -I (py -3.12 -I v224s6f2_check.py OUT --build build_receipt_<tag>.json).
#   base (from the build receipt base_kind; absent = flip):
#     flip     (A) == the s6f gate: the frozen output + the log up to V224_DONE pass v224_check.py --flip --ref V224-exact.
#     flrm     (A) the frozen output + the log up to V224RM_DONE pass the UNCHANGED V224-FLIP-RM commit gate (v224rm/v224flrm_check.py,
#              10/10 on its commit); stage csvs == the V224-FLIP-RM v1 commit's (R7 / r384 / r224 raw 1e-3, r224 ranks 1e-6).
#     flips20  (A) the frozen output + the log up to V224F_DONE pass the UNCHANGED flips20 commit gate (v224f/v224aos_check.py --wf .20
#              --build v224f/kernel_v224flips20/BUILD_V224FS.json --ref-s <the flips20 v1 commit>: V224-FLIP part, S members / shas,
#              END-TO-END (1 - .2) pct(H) + .2 pct(S), canaries); stage csvs == the flips20 v1 commit's.
#     (0, flrm / flips20) the kernel notebook: sha == build receipt; its first N cells re-serialise to the pinned base FILE.
#     (C, flrm / flips20) blank = every label a base canary fired on (setup receipt v224_canary + base_fired) + constant base
#              columns, == the notebook's rule; clean mode also needs base_fired [] and blend base_blanked [].
#   (A) flip base rows: submission_v224flip.csv (the frozen V224-FLIP output) passes the UNCHANGED V224-FLIP gate (v224_check.py --flip
#       --ref V224-exact commit; i.e. == a V224-FLIP-form recompute from this run's stage csvs, 1e-6, every V224 marker / receipt);
#       its sha == the setup receipt's; the V224 stage csvs == the V224-FLIP v1 commit's (R7 / r384 1e-3, r224 1e-6)  [R4 (vi)]
#   (B) S6fall parity: s6f/B/B_probs.csv vs nartaa's public commit k1/B/B_probs.csv on the visible studies, labels BY NAME,
#       max |dp| <= 1e-3                                                                                              [R4 (i)]
#   (C) blend: submission.csv == .8 pct(base) + .2 pct(S6fall) (independent average-tie ranks; unscored -> base rank), 1e-6  [(vi)]
#   (D) coverage 3/3 = 1.0 (driver == blend), no unscored study                                                       [(iv)]
#   (E) receipts: own subprocess + stock site-packages (+ R7's pinned timm / pydicom) + DEC on PYTHONPATH, no R7 dir, versions,
#       timings, microbench s/study; deadline policy == build; driver ok; code manifest / model / compact_infer shas  [(ii) (iii)]
#       [F1 10-15] no S-stage dir (/tmp/k12_fleet, /tmp/k2_members*) on any import path (every base); flrm / flips20 also: the
#       kernel PYTHONPATH the setup saw == the S6F v1 commit's (same docker pin, MEASURED) and S6fall's PYTHONPATH == /tmp/s6f_dec +
#       exactly that, in every receipt copy (setup file, s6f_receipt.setup, driver receipt, s6f_receipt.run.driver). Pre-F1: .pre_f1
#   (F) canary not fired, no S6F failure marker, no constant column; (G) log marker order; (H) submission format
#   --expect-canary  control builds (e.g. --g-start-h 0): S6F canary expected (ACL == 0.5, other 11 == the base), (B)-(D) skipped
#       [rfix S6F-R1] plus the path the control must have taken (build receipt policy.control, or --control):
#       skip: run SKIPPED_GUARD (start age > G), no S6F_START;  kill: run RAN + killed, >= 3 process groups killed incl. the stub's
#       and its 2 child sessions, S6F_KILL_CHECK alive == [] (no stub / grandchild survived), killed within 3 min of the hard deadline
#   [rfix S6F-R2] info RECAL: like-for-like runtime recalibration vs nartaa's commit (warm GPU / loader s per study on the same
#       studies, B stage seconds) -> RECAL_TRIGGER yes/no. The MICROBENCH totals include cold start and are NOT the sim's c.
#   [rfix S6F-R3] info PICK_ELIGIBLE_J: J (a base DEC cell, DEC_OK before V224_DONE) is a SILVER_PLAN sec 4 pick condition
#   usage: py -3.12 -I v224s6f2_check.py OUT --build build_receipt_<tag>.json [--kernel-dir DIR] [--ref-s6 s6f/ref_output]
#          [--ref-flip DIR] [--ref-exact DIR] [--ref-flrm DIR] [--ref-flips20 DIR] [--sample CSV] [--expect-canary [--control skip|kill]]
import os, sys, json, glob, shutil, tempfile, hashlib, argparse, subprocess
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SESS = os.path.dirname(HERE)
ROOT = os.path.dirname(SESS)
S6F = os.path.join(SESS, 's6f')
V224 = os.path.join(ROOT, 'session_2026-10-14', 'v224')
FLRM_CHECK = os.path.join(SESS, 'v224rm', 'v224flrm_check.py')
AOS_CHECK = os.path.join(SESS, 'v224f', 'v224aos_check.py')
FLIPS20_BUILD = os.path.join(SESS, 'v224f', 'kernel_v224flips20', 'BUILD_V224FS.json')
BASES = {'flip': dict(last='V224_DONE', cells=15, sha='73a9d5620d0a637ed171ecec13bcf243bd0af67992b5b4cb8ad7f35bf72b1b1b',
                      nb=os.path.join(V224, 'kernel_v224flip', 'rsna_sub_v224flip.ipynb')),
         'flrm': dict(last='V224RM_DONE', cells=17, sha='f14f6accc7c178536ff8a5d8d70a65da63ca05b2dd6d2a7ce9673100f9178ecf',
                      nb=os.path.join(SESS, 'v224rm', 'kernel_v224flrm', 'rsna_sub_v224flrm.ipynb')),
         'flips20': dict(last='V224F_DONE', cells=24, sha='d2066b8c9c5e3a77a20fa45fc5a0f88db8926566e205a66c49536022df2f17c5',
                         nb=os.path.join(SESS, 'v224f', 'kernel_v224flips20', 'rsna_sub_v224flips20.ipynb'))}
REVIEWED_G = {'flip': 5.75, 'flrm': 6.0, 'flips20': 6.25}            # sim_s6f_runtime (flip) / sim_s6f2_runtime (flrm, flips20)
S6F_FILES = ('s6f', 's6f_setup_receipt.json', 's6f_receipt.json', 'submission_v224flip.csv', 'submission.csv')
ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--build', required=True, help='build_receipt_<tag>.json written by build_v224s6f2.py (or s6f/build_v224s6f.py)')
ap.add_argument('--kernel-dir', default=None, help='dir of the pushed notebook (default: <build dir>/kernel_<tag>)')
ap.add_argument('--ref-s6', default=os.path.join(S6F, 'ref_output'))
ap.add_argument('--ref-flip', default=os.path.join(V224, 'out_v224flip_v1'))
ap.add_argument('--ref-exact', default=os.path.join(V224, 'out_v224_v1'))
ap.add_argument('--ref-flrm', default=os.path.join(SESS, 'v224rm', 'out_v224flrm_v1'))
ap.add_argument('--ref-flips20', default=os.path.join(SESS, 'v224f', 'out_v224flips20_v1'))
ap.add_argument('--sample', default=os.path.join(ROOT, '_data', 'sample_submission.csv'))
ap.add_argument('--expect-canary', action='store_true')
ap.add_argument('--control', choices=['skip', 'kill'], default=None, help='control kind (default: build receipt policy.control)')
A = ap.parse_args()
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
       'Contusion', 'Fracture']
SOURCE_SHA = {'px_cache.py': 'b13e38a7ad3bd496f8027e02209653e224b0926db4f0853eeb0d076a2d6e2632',
              'compact_train.py': '14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80',
              'compact_infer.py': '34f433fe72b732543813165c3397ec80cef01b0d49b25b7b22668714f0b0ef4a',
              'probe_infer.py': 'a6cc31448019c84be5868d0a2ff1d99719b23b0f86110efc03efff3ce6bd6980',
              'knee_anchor_bank_v2.py': 'e564807a129dfcd3c984348917436ee6083066a4d685340587ebd1d3ea14f309'}
ANATOMY_SHA = '5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e'
TOL_PARITY, TOL_BLEND, TOL_COMPILE = 1e-3, 1e-6, 1e-3
# [F1] the image's own PYTHONPATH as the S6F setup saw it on the S6F v1 commit (MEASURED: s6f/out_v224flips6f_v1/s6f_setup_receipt.json
# pythonpath_stock; docker pin 2757e0c7 == both s6f2 kernels), and the S stage's dirs (K12-MERGE / S-MERGE fleet dir, K2 member dirs)
STOCK_PP = ['/kaggle/lib/kagglegym', '/kaggle/lib', '/kaggle/input/competitions/rsna-knee-abnormality-detection']
S_DIRS = ('/tmp/k12_fleet', '/tmp/k2_members')
R = []


def chk(name, ok, info=''):
    R.append(bool(ok))
    print(f'  {"OK  " if ok else "FAIL"} {name} {info}', flush=True)


def P(*a):
    return os.path.join(A.out, *a)


def jload(p, default=None):
    try:
        return json.load(open(p, encoding='utf-8'))
    except Exception:
        return default


def sha_file(p):
    try:
        return hashlib.sha256(open(p, 'rb').read()).hexdigest()
    except Exception:
        return None


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


B = jload(A.build, {}) or {}
POL = B.get('policy') or {}
BASE = B.get('base_kind') or 'flip'
BD = BASES.get(BASE) or BASES['flip']
LAST = BD['last']
CONTROL = (A.control or POL.get('control')) if A.expect_canary else None
mode = ('canary-control' + (' ' + CONTROL if CONTROL else '')) if A.expect_canary else 'S6F'
print(f'V224S6F2_CHECK {A.out} base {BASE} mode {mode} build {os.path.basename(A.build)} ({str(B.get("notebook_sha256"))[:16]})', flush=True)
chk('build receipt present (policy, driver sha, tag), base kind known', bool(POL) and bool(B.get('driver_sha256')) and bool(B.get('tag'))
    and BASE in BASES and B.get('base_sha256') == BD['sha'], (BASE, str(B.get('base_sha256'))[:16]))
if not A.expect_canary and POL.get('control') is None:
    chk('policy == the runtime-reviewed submit policy for this base (G %.2f, budget 3.3, deadline 8.2, stop margin 240, min run 1200, '
        'w .2, coverage .98, canary ACL)' % REVIEWED_G.get(BASE, -1),
        POL.get('g_start_h') == REVIEWED_G.get(BASE) and POL.get('budget_h') == 3.3 and POL.get('deadline_h') == 8.2
        and POL.get('stop_margin_s') == 240.0 and POL.get('min_run_s') == 1200.0 and POL.get('w_s6f') == 0.2 and POL.get('cov_min') == 0.98
        and POL.get('canary_col') == 'ACL', {k: POL.get(k) for k in ('g_start_h', 'budget_h', 'deadline_h', 'w_s6f')})
if BASE != 'flip':                                   # (0) the notebook that ran (the commit script asserts the same before the push)
    kd = A.kernel_dir or os.path.join(os.path.dirname(os.path.abspath(A.build)), 'kernel_' + str(B.get('tag')))
    try:
        nbp = os.path.join(kd, 'rsna_sub_%s.ipynb' % B['tag'])
        raw = open(nbp, 'rb').read()
        nbj = json.loads(raw)
        braw = open(BD['nb'], 'rb').read()
        back = dict(nbj)
        back['cells'] = nbj['cells'][:BD['cells']]
        ok0 = (hashlib.sha256(raw).hexdigest() == B.get('notebook_sha256') and hashlib.sha256(braw).hexdigest() == BD['sha']
               and json.dumps(back, ensure_ascii=False, separators=(',', ':')).encode('utf-8') == braw
               and len(nbj['cells']) == BD['cells'] + 3 and B.get('appended') == [BD['cells'], BD['cells'] + 1, BD['cells'] + 2])
    except Exception as e:
        ok0 = False
    chk('(0) kernel notebook sha == build receipt; minus its 3 appended cells it re-serialises to the pinned base FILE (%s...)' % BD['sha'][:16], ok0,
        os.path.relpath(kd, ROOT) if os.path.isdir(kd) else 'no kernel dir %s' % kd)
if A.expect_canary:
    chk('control kind known (skip / kill, from the build receipt policy.control or --control) and == the receipt when both given',
        CONTROL in ('skip', 'kill') and (POL.get('control') in (None, CONTROL)), (A.control, POL.get('control')))
text, logp = read_log(A.out)
chk('kernel log present and named after the build slug', bool(text) and logp is not None and
    os.path.basename(logp) == 'rsna-sub-%s.log' % B.get('tag'), os.path.basename(logp) if logp else None)
sample = pd.read_csv(A.sample, dtype={'StudyInstanceUID': str})
cols, ids = list(sample.columns), sample['StudyInstanceUID'].tolist()
labels = cols[1:]
assert labels == LAB

# ---- (A) base rows --------------------------------------------------------------------------------------------------------
REF_BASE = {'flip': A.ref_flip, 'flrm': A.ref_flrm, 'flips20': A.ref_flips20}[BASE]
print('(A) %s base rows (frozen output of the cells before S6F, log up to %s)' % (BASE, LAST), flush=True)
base_p = P('submission_v224flip.csv')
setup = jload(P('s6f_setup_receipt.json'), {}) or {}
rec = jload(P('s6f_receipt.json'), {}) or {}
chk('frozen base output present; sha == setup receipt == s6f_receipt.setup', os.path.isfile(base_p) and sha_file(base_p)
    == setup.get('base_sha256') == (rec.get('setup') or {}).get('base_sha256'), str(sha_file(base_p))[:16])
view = tempfile.mkdtemp(prefix='s6f2view_')
try:
    for f in os.listdir(A.out):
        s = P(f)
        if os.path.isfile(s) and f != 'submission.csv' and (BASE == 'flip' or f not in S6F_FILES):
            shutil.copyfile(s, os.path.join(view, f))
    if os.path.isfile(base_p):
        shutil.copyfile(base_p, os.path.join(view, 'submission.csv'))
    if logp:                                       # S6fall removed from the log too: the base gate reads the run up to LAST
        lv = os.path.join(view, os.path.basename(logp))
        try:
            E = json.load(open(logp, encoding='utf-8'))
            k = next(i for i, e in enumerate(E) if LAST in e.get('data', ''))
            json.dump(E[:k + 1], open(lv, 'w', encoding='utf-8'))
        except Exception:
            t_ = open(logp, encoding='utf-8', errors='replace').read()
            open(lv, 'w', encoding='utf-8').write(t_[:t_.find('\n', t_.find(LAST)) + 1] if LAST in t_ else t_)
    if BASE == 'flip':
        cmd, want, gname = ([sys.executable, '-I', os.path.join(V224, 'v224_check.py'), view, '--flip', '--ref', A.ref_exact, '--sample', A.sample],
                            'V224_CHECK ', 'the UNCHANGED V224-FLIP gate (v224_check.py --flip --ref V224-exact)')
    elif BASE == 'flrm':
        cmd, want, gname = ([sys.executable, '-I', FLRM_CHECK, view, '--ref-exact', A.ref_exact, '--ref-flip', A.ref_flip, '--sample', A.sample],
                            'V224FLRM_CHECK ', 'the UNCHANGED V224-FLIP-RM commit gate (v224rm/v224flrm_check.py)')
    else:
        cmd, want, gname = ([sys.executable, '-I', AOS_CHECK, view, '--wf', '0.20', '--build', FLIPS20_BUILD, '--ref-exact', A.ref_exact,
                             '--ref-flip', A.ref_flip, '--ref-s', A.ref_flips20, '--sample', A.sample],
                            'V224AOS_CHECK ', 'the UNCHANGED flips20 commit gate (v224f/v224aos_check.py --wf .20 --ref-s the flips20 v1 commit)')
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, encoding='utf-8', errors='replace')
    last = [l for l in r.stdout.splitlines() if l.startswith(want)][-1:] or ['(no verdict) ' + (r.stderr or '')[-300:]]
    fails = [l.strip() for l in r.stdout.splitlines() if l.strip().startswith('FAIL')]
    chk('the frozen base output + the log up to %s pass %s' % (LAST, gname),
        r.returncode == 0 and last[0].startswith(want + 'PASS'), last[0] + (' | ' + ' | '.join(fails)[:600] if fails else ''))
finally:
    shutil.rmtree(view, ignore_errors=True)
STAGE = [('ours_sub.csv', TOL_COMPILE), ('raptor_raw.csv', TOL_COMPILE), ('raptor224_sub.csv', TOL_BLEND)]
if BASE == 'flrm':
    STAGE.append(('raptor224_raw.csv', TOL_COMPILE))
for nm, tol in STAGE:
    try:
        a = pd.read_csv(P(nm), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, labels].to_numpy(np.float64)
        b = pd.read_csv(os.path.join(REF_BASE, nm), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, labels].to_numpy(np.float64)
        d = float(np.abs(a - b).max())
    except Exception as e:
        d = float('inf')
    chk(f'one change vs the {BASE} v1 commit ({os.path.basename(REF_BASE)}): {nm} within {tol:g}', d <= tol, 'max diff %.3g' % d)
try:
    fb = pd.read_csv(os.path.join(REF_BASE, 'submission.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, labels]
    bb = pd.read_csv(base_p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, labels]
    dd = np.abs(bb.to_numpy(np.float64) - fb.to_numpy(np.float64))
    print('  info frozen base vs the %s v1 commit submission: max diff %.3g, cells differing %d of 36' % (BASE, dd.max(), int((dd > 1e-6).sum())), flush=True)
except Exception as e:
    print('  info frozen base vs the base commit: n/a %r' % (e,), flush=True)
tb = [m for m in ('Traceback', 'RuntimeError') if m in text]
chk('no Traceback / RuntimeError anywhere in the kernel log (the S6F part included)', not tb, tb)
pos = {m: text.find(m) for m in (LAST, 'S6F_SETUP_OK')}
chk('the freeze ran after the base\'s last cell (%s before S6F_SETUP_OK)' % LAST, 0 <= pos[LAST] < pos['S6F_SETUP_OK']
    and text.find('S6F_', 0, max(pos[LAST], 0)) < 0, pos)
# [rfix S6F-R3] SILVER_PLAN sec 4 makes J (a DEC cell for the base stages) a final-pick condition. The base DEC cell prints DEC_OK /
# DEC_FALLBACK before stage 1 (V224-J form); S6F's own S6F_DEC_OK only serves the S6fall subprocess. Info only (R4 does not need J).
import re as _re
_pre = text[:pos[LAST]] if pos[LAST] >= 0 else ''
_jb = 'DEC_OK' if _re.search(r'(^|\n)DEC_OK ', _pre) else ('DEC_FALLBACK' if _re.search(r'(^|\n)DEC_FALLBACK ', _pre) else None)
print('  info PICK_ELIGIBLE_J %s: base-stage decoders %s -> %s' % (
    'yes' if _jb == 'DEC_OK' else 'no', _jb or 'absent (cells 0-14 = V224-FLIP, no DEC cell; only the S6fall subprocess decodes JPEG Lossless)',
    'J met (SILVER_PLAN sec 4)' if _jb == 'DEC_OK' else 'a READ only under PREREG Amendment 3 R4, NOT pick-eligible (SILVER_PLAN sec 4 needs J: '
    'the same 3 cells on a FLIP-J base)'), flush=True)

# ---- (E) receipts (before B / C so a missing receipt is named) -------------------------------------------------------------
print('(E) receipts, environment, policy', flush=True)
run = rec.get('run') or {}
drv_rec = jload(P('s6f', 's6f_driver_receipt.json'), {}) or {}
blend = rec.get('blend') or {}
chk('s6f_receipt tag == build tag', rec.get('tag') == B.get('tag'), rec.get('tag'))
chk('setup: ok, driver sha == build, DEC ok (3 decoders from /tmp/s6f_dec, self-test exact), PYTHONPATH = /tmp/s6f_dec first',
    setup.get('ok') is True and setup.get('driver_sha256') == B.get('driver_sha256') and setup.get('dec_ok') is True
    and str(setup.get('pythonpath_s6f') or '').split(':')[0] == '/tmp/s6f_dec'
    and ((setup.get('dec') or {}).get('selftest') or {}).get('versions') == {'pylibjpeg': '2.1.0', 'libjpeg': '2.4.0', 'openjpeg': '2.6.0'},
    {k: setup.get(k) for k in ('ok', 'dec_ok', 'pythonpath_s6f', 'why')})
pol_run = {k: run.get(k) for k in ('g_start_h', 'budget_h', 'deadline_h', 'stop_margin_s')}
pol_b = {k: POL.get(k) for k in ('g_start_h', 'budget_h', 'deadline_h', 'stop_margin_s')}
chk('run: deadline policy == build (G, budget, deadline NB_T0 + h, stop margin), hard deadline (a number) <= NB_T0 + 8.2 h',
    pol_run == pol_b and isinstance(run.get('hard_deadline_age_h'), (int, float)) and run.get('hard_deadline_age_h') <= 8.2 + 1e-9,
    (pol_run, run.get('hard_deadline_age_h')))
if A.expect_canary and CONTROL == 'skip':               # [rfix S6F-R1] the skip control must have skipped at the start guard
    chk('control skip: run SKIPPED_GUARD (start age > G), S6F_SKIPPED start guard, the driver never started (no S6F_START)',
        run.get('status') == 'SKIPPED_GUARD' and isinstance(run.get('start_age_h'), (int, float)) and run['start_age_h'] > run.get('g_start_h', 99)
        and 'S6F_SKIPPED start guard' in text and 'S6F_START' not in text, {k: run.get(k) for k in ('status', 'start_age_h', 'g_start_h')})
if A.expect_canary and CONTROL == 'kill':               # [rfix S6F-R1] the kill control must have run the POSIX process-group kill
    kc = run.get('kill_check') or {}
    sp = kc.get('stub_pids') or {}
    grp = run.get('killed_groups') or []
    want_g = [sp.get('stub')] + list(sp.get('children') or [])
    gch = [g_ for g_ in (sp.get('grandchildren') or []) if isinstance(g_, int)]
    chk('control kill: run RAN + killed at the hard deadline, S6F_KILLED; >= 3 groups killed incl. the stub and its 2 child sessions; '
        '2 grandchildren recorded; S6F_KILL_CHECK alive == [] (POSIX probe); kill within 3 min of the hard deadline',
        POL.get('control') == 'kill' and run.get('status') == 'RAN' and run.get('killed') is True and 'S6F_KILLED' in text
        and 'S6F_KILL_CHECK' in text and len(grp) >= 3 and len(want_g) == 3 and all(isinstance(g_, int) and g_ in grp for g_ in want_g)
        and len(gch) == 2 and kc.get('alive') == [] and isinstance(run.get('end_age_h'), (int, float))
        and isinstance(run.get('hard_deadline_age_h'), (int, float)) and run['end_age_h'] - run['hard_deadline_age_h'] <= 0.05,
        {'status': run.get('status'), 'killed': run.get('killed'), 'groups': grp, 'stub': want_g, 'grandchildren': gch,
         'alive': kc.get('alive'), 'end-hard h': (round(run['end_age_h'] - run['hard_deadline_age_h'], 4)
                                                  if isinstance(run.get('end_age_h'), (int, float)) and isinstance(run.get('hard_deadline_age_h'), (int, float)) else None)})
if not A.expect_canary:
    chk('run: RAN in its own subprocess, rc 0, not killed, start age <= G', run.get('status') == 'RAN' and run.get('rc') == 0 and
        run.get('killed') is False and isinstance(run.get('start_age_h'), (int, float)) and run['start_age_h'] <= POL.get('g_start_h', -1),
        {k: run.get(k) for k in ('status', 'rc', 'killed', 'start_age_h', 'seconds')})
    rd = run.get('driver') or {}
    chk('driver receipt ok (file == the copy in s6f_receipt)', drv_rec.get('ok') is True and rd.get('ok') is True and
        rd.get('total_seconds') == drv_rec.get('total_seconds'), drv_rec.get('error'))
    chk('driver: code manifest == the scored run (b66c574b), model sha 0c7297d5', drv_rec.get('code_manifest_pin') ==
        B.get('code_manifest_sha256') == ((drv_rec.get('inputs') or {}).get('manifest_sha256')) and
        ((drv_rec.get('inputs') or {}).get('model') or {}).get('sha256') == B.get('model_sha256'),
        str((drv_rec.get('inputs') or {}).get('manifest_sha256'))[:16])
    env = drv_rec.get('env') or {}
    iso = drv_rec.get('env_isolation') or {}
    mods = iso.get('mods') or {}
    need = ('python', 'torch', 'cuda', 'timm', 'pydicom', 'numpy', 'pandas', 'cv2', 'scipy')
    chk('versions in the receipt (env probe: %s)' % ', '.join(need), all(env.get(k) not in (None, '') and not str(env.get(k)).startswith('MISSING')
                                                                         for k in need) and env.get('n_gpu', 0) >= 1,
        {k: env.get(k) for k in need})
    pp = str(iso.get('PYTHONPATH') or '')
    bad = [p for p in (iso.get('sys_path') or []) + pp.split(':') + (iso.get('driver_sys_path') or [])
           + str(setup.get('pythonpath_s6f') or '').split(':') + list(setup.get('pythonpath_stock') or [])
           if str(p).startswith(('/tmp/probe', '/kaggle/temp/probe', '/kaggle/working') + S_DIRS)]
    chk('own subprocess env: PYTHONPATH starts /tmp/s6f_dec, no R7 dir / working dir / S-stage dir (%s*) on any import path'
        % '*, '.join(S_DIRS), pp.split(':')[0] == '/tmp/s6f_dec' and not bad and iso.get('r7_or_working_on_path') == [], (pp, bad))
    if BASE != 'flip':                             # [F1] the S stage / RM run before S6F here: S6fall's import path must be the image's
        want_pp = ':'.join(['/tmp/s6f_dec'] + STOCK_PP)
        rs_ = rec.get('setup') or {}
        rdi = ((run.get('driver') or {}).get('env_isolation') or {})
        got_pp = {'setup.stock': setup.get('pythonpath_stock'), 's6f_receipt.setup.stock': rs_.get('pythonpath_stock'),
                  'setup.pythonpath_s6f': setup.get('pythonpath_s6f'), 's6f_receipt.setup.pythonpath_s6f': rs_.get('pythonpath_s6f'),
                  'driver env PYTHONPATH': iso.get('PYTHONPATH'), 's6f_receipt.run.driver env PYTHONPATH': rdi.get('PYTHONPATH')}
        chk('(E, %s) S6fall import path == the S6F v1 commit\'s (same docker pin): kernel PYTHONPATH seen by the setup == %s and S6fall '
            'PYTHONPATH == /tmp/s6f_dec + exactly that, in every receipt copy' % (BASE, STOCK_PP),
            setup.get('pythonpath_stock') == STOCK_PP == rs_.get('pythonpath_stock')
            and setup.get('pythonpath_s6f') == rs_.get('pythonpath_s6f') == iso.get('PYTHONPATH') == rdi.get('PYTHONPATH') == want_pp,
            {k: v for k, v in got_pp.items() if v != (STOCK_PP if k.endswith('stock') else want_pp)} or 'all == %s' % want_pp)
    site_dirs = {m: (mods.get(m) or [None, None])[1] for m in ('torch', 'timm', 'pydicom', 'numpy', 'pandas')}
    chk('site-packages: torch / timm / pydicom / numpy / pandas from the image dist-packages; decoders from /tmp/s6f_dec',
        all(str(v).endswith('-packages') for v in site_dirs.values()) and all((mods.get(m) or [None, None])[1] == '/tmp/s6f_dec'
                                                                              for m in ('pylibjpeg', 'libjpeg', 'openjpeg')),
        {m: (mods.get(m) or [None])[0] for m in ('timm', 'pydicom', 'pylibjpeg')})
    print('  info site-packages versions seen by S6fall: timm %s, pydicom %s (stock image 1.0.29 / 3.0.2; R7 cell 6 pip-installs 1.0.25 / '
          '3.0.1 globally); nartaa scored with %s / %s' % ((mods.get('timm') or [None])[0], (mods.get('pydicom') or [None])[0],
                                                          (jload(os.path.join(A.ref_s6, 'k1', 'k1_receipt.json'), {}) or {}).get('env', {}).get('timm'),
                                                          (jload(os.path.join(A.ref_s6, 'k1', 'k1_receipt.json'), {}) or {}).get('env', {}).get('pydicom')),
          flush=True)
    bp = drv_rec.get('B_probe') or {}
    chk('B stack probe: MONAI 0.6.0 vendored from the code tree, spatial model, 50,018,284 params (== nartaa commit)',
        (bp.get('monai') or [None])[0] == '0.6.0' and str((bp.get('monai') or [None, ''])[1]).startswith('training/vendor_runtime/monai060/')
        and bp.get('spatial') is True and bp.get('params') == 50018284, {k: bp.get(k) for k in ('monai', 'spatial', 'params', 'timm')})
    st = drv_rec.get('stages') or {}
    mb = (drv_rec.get('B') or {}).get('microbench') or {}
    chk('timings: driver stages env / inputs / B_probe / B with seconds; microbench s/study present',
        set(st) == {'env', 'inputs', 'B_probe', 'B'} and all(isinstance((st.get(k) or {}).get('seconds'), (int, float)) for k in st)
        and all(isinstance(mb.get(k), (int, float)) for k in ('s_per_study_per_gpu_max', 'gpu_s_per_study', 'decode_s_per_study_per_worker')),
        {k: (st.get(k) or {}).get('seconds') for k in st})
    # [rfix S6F-R2] the driver's microbench totals include worker spawn, model load, CUDA warm-up and the first loader item of each
    # part (3 studies over parts [2, 1]): NOT the sim's c. Compare them only with the SAME fields on nartaa's commit, and evaluate the
    # recalibration rule like for like: per study, same studies, warm = not the first study of its part (RECAL line).
    def _parts(bdir):
        per, rows, gpu, dec, per_gpu = {}, 0, 0.0, 0.0, []
        for pth in sorted(glob.glob(os.path.join(bdir, 'part*', 'receipt.json'))):
            r_ = jload(pth, {}) or {}
            s_ = r_.get('stats') or {}
            rows += int(s_.get('rows') or 0)
            gpu += float(s_.get('gpu_s') or 0)
            dec += float(s_.get('decode_s') or 0)
            if r_.get('seconds') and r_.get('scored'):
                per_gpu.append(r_['seconds'] / max(1, r_['scored']))
            for i_, x in enumerate(r_.get('per_study') or []):
                try:                                       # [uid, windows, decode s, render s, gpu s]
                    per[str(x[0])] = {'warm': i_ > 0, 'loader': float(x[2]) + float(x[3]), 'gpu': float(x[4])}
                except Exception:
                    pass
        tot = {'s_per_study_per_gpu_max': round(max(per_gpu), 3) if per_gpu else None,
               'gpu_s_per_study': round(gpu / rows, 3) if rows else None, 'decode_s_per_study_per_worker': round(dec / rows, 3) if rows else None}
        return per, tot
    ours_ps, _ = _parts(P('s6f', 'B'))
    ref_ps, ref_tot = _parts(os.path.join(A.ref_s6, 'k1', 'B'))
    print('  info MICROBENCH driver totals over the 3 cold-start studies (NOT the sim c; compare only with nartaa\'s commit, same fields '
          '%s): %s | run seconds %s | start age %.3f h, end age %s h' % (json.dumps(ref_tot), json.dumps(mb), run.get('seconds'),
                                                                        run.get('start_age_h') or -1, run.get('end_age_h')), flush=True)
    warm = sorted(u for u in ours_ps if ours_ps[u]['warm'] and u in ref_ps and ref_ps[u]['warm'])
    b_ours = (st.get('B') or {}).get('seconds')
    b_ref = (((jload(os.path.join(A.ref_s6, 'k1', 'k1_receipt.json'), {}) or {}).get('stages') or {}).get('B') or {}).get('seconds')
    if warm and isinstance(b_ours, (int, float)) and isinstance(b_ref, (int, float)) and b_ref > 0:
        g_o, g_r = sum(ours_ps[u]['gpu'] for u in warm), sum(ref_ps[u]['gpu'] for u in warm)
        l_o, l_r = sum(ours_ps[u]['loader'] for u in warm), sum(ref_ps[u]['loader'] for u in warm)
        rg, rl, rb_ = g_o / g_r, l_o / l_r, b_ours / b_ref
        trig = rg > 1.5 or rl > 1.5 or rb_ > 2.0
        print('  info RECAL like for like vs nartaa\'s commit (%d warm study, same uid; rules: warm GPU > 1.5x, warm loader decode+render '
              '> 1.5x, B stage > 2.0x): warm GPU %.2f vs %.2f s (x%.2f) | warm loader %.2f vs %.2f s (x%.2f) | B stage %.1f vs %.1f s (x%.2f)'
              % (len(warm), g_o, g_r, rg, l_o, l_r, rl, b_ours, b_ref, rb_), flush=True)
        print('  info RECAL_TRIGGER %s%s' % ('yes' if trig else 'no', (': scale sim_s6f_runtime.py C_LO / C_HI by x%.2f, re-run it, rebuild '
                                                                       'with --g-start-h' % max(rg, rl)) if trig else ''), flush=True)
    else:
        print('  info RECAL_TRIGGER n/a: no warm study matched nartaa\'s parts or no B stage seconds (%s, %s, %s)' % (warm, b_ours, b_ref),
              flush=True)
    parts = sorted(glob.glob(P('s6f', 'B', 'part*', 'receipt.json')))
    pr = [jload(p, {}) for p in parts]
    chk('worker receipts: PASS, fp16, model E_prior_s42_fall.pt (spatial), compact_infer 34f433fe + 4 source shas, anchor sha, '
        'timed_out False', len(pr) >= 1 and all(r_.get('status') == 'PASS' and r_.get('amp') == 'fp16' and r_.get('models') == ['E_prior_s42_fall.pt']
                                                and r_.get('spatial_models') == 1 and all((r_.get('sha256') or {}).get(k) == v for k, v in SOURCE_SHA.items())
                                                and r_.get('anchor_weights_sha256') == ANATOMY_SHA and r_.get('timed_out') is False for r_ in pr),
        [(r_.get('status'), (r_.get('sha256') or {}).get('compact_infer.py', '')[:8], r_.get('timed_out')) for r_ in pr])

# ---- (B) parity, (C) blend, (D) coverage ------------------------------------------------------------------------------------
sub = pd.read_csv(P('submission.csv'), dtype={'StudyInstanceUID': str}) if os.path.isfile(P('submission.csv')) else None
base = pd.read_csv(base_p, dtype={'StudyInstanceUID': str}) if os.path.isfile(base_p) else None
if not A.expect_canary:
    print('(B) S6fall parity vs nartaa public commit (labels by name)', flush=True)
    try:
        mine = pd.read_csv(P('s6f', 'B', 'B_probs.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
        ref = pd.read_csv(os.path.join(A.ref_s6, 'k1', 'B', 'B_probs.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
        same_ids = sorted(mine.index) == sorted(ref.index) == sorted(ids)
        dp = np.abs(mine.loc[ids, LAB].to_numpy(np.float64) - ref.loc[ids, LAB].to_numpy(np.float64))
        dz = np.abs(mine.loc[ids, ['z|' + l for l in LAB]].to_numpy(np.float64) - ref.loc[ids, ['z|' + l for l in LAB]].to_numpy(np.float64))
        per = {l: float(dp[:, j].max()) for j, l in enumerate(LAB)}
        chk('S6fall B_probs == nartaa k1/B/B_probs.csv on the %d visible studies, labels by name, max |dp| <= 1e-3' % len(ids),
            same_ids and np.isfinite(dp).all() and float(dp.max()) <= TOL_PARITY, 'max |dp| %.2e (label %s), max |dz| %.2e' % (
                float(dp.max()), max(per, key=per.get), float(dz.max())))
        rk = sum(int((avg_rank_pct(mine.loc[ids, l]) != avg_rank_pct(ref.loc[ids, l])).any()) for l in LAB)
        print('  info labels whose 3-study S6fall rank order differs from nartaa: %d of 12' % rk, flush=True)
        chk('B_probs.csv sha == driver receipt == blend receipt', sha_file(P('s6f', 'B', 'B_probs.csv')) == (drv_rec.get('B') or {}).get('probs_sha256')
            == blend.get('b_probs_sha256'))
    except Exception as e:
        chk('S6fall parity computable', False, repr(e)[:300])
    print('(C) blend recompute', flush=True)
    try:
        Sx = mine.loc[ids, LAB].to_numpy(np.float64)
        ok = np.isfinite(Sx).all(axis=1)
        bb = base.set_index('StudyInstanceUID').loc[ids, LAB]
        if BASE == 'flip':
            blank = ['Fracture'] if 'r224->Fracture' in ((setup.get('v224_canary') or {}).get('fired') or []) else []
        else:                                      # [s6f2] the notebook's base-canary rule, re-derived from the receipts + the frozen base
            fired_ = list((setup.get('v224_canary') or {}).get('fired') or []) + list(setup.get('base_fired') or [])
            blank = [c for c in LAB if any(str(x).endswith('->' + c) for x in fired_)]
            blank += [c for c in LAB if c not in blank and len(ids) >= 2 and bb[c].nunique() == 1]
            chk('(C) base canaries clean: setup base_fired == [] (V224 CANARY / RM_GUARD / HF), v224_canary fired [], blend base_blanked == []',
                setup.get('base_fired') == [] and (setup.get('v224_canary') or {}).get('fired') == [] and blend.get('base_blanked') == []
                and blank == [], {'base_fired': setup.get('base_fired'), 'base_blanked': blend.get('base_blanked'), 'blank': blank})
        exp = np.zeros((len(ids), len(LAB)))
        for j, c in enumerate(LAB):
            rb = avg_rank_pct(bb[c].to_numpy(np.float64))
            rs = rb.copy()
            rs[ok] = avg_rank_pct(Sx[ok, j])
            exp[:, j] = 0.8 * rb + 0.2 * rs
            if c in blank:
                exp[:, j] = 0.5
        got = sub.set_index('StudyInstanceUID').loc[ids, LAB].to_numpy(np.float64)
        d = float(np.abs(got - exp).max())
        chk('submission.csv == .8 pct(frozen base) + .2 pct(S6fall) (independent ranks, unscored -> base rank, blanked -> 0.5) within 1e-6', d <= TOL_BLEND,
            'max diff %.3g' % d)
        chk('blend receipt: mode S6F, w .2, coverage floor .98, canary col ACL', blend.get('mode') == 'S6F' and blend.get('w_s6f') == 0.2 ==
            POL.get('w_s6f') and blend.get('cov_min') == 0.98 and blend.get('canary_col') == 'ACL', {k: blend.get(k) for k in ('mode', 'w_s6f')})
        nd = int((np.abs(got - bb.to_numpy(np.float64)) > 1e-6).sum())
        print('  info S6fall moved %d of 36 cells vs the frozen base' % nd, flush=True)
    except Exception as e:
        chk('blend recompute computable', False, repr(e)[:300])
    print('(D) coverage', flush=True)
    bcov = (drv_rec.get('B') or {}).get('coverage')
    chk('coverage 1.0 on the visible studies (driver 3/3 == blend), no unscored study', bcov == 1.0 == blend.get('coverage') and
        (drv_rec.get('B') or {}).get('n_unscored') == 0 == blend.get('n_unscored') and (drv_rec.get('B') or {}).get('n') == len(ids),
        (bcov, blend.get('coverage')))
else:
    print('(B-D) skipped: control build, S6F canary expected', flush=True)

# ---- (F) canary, (G) markers, (H) format -----------------------------------------------------------------------------------
print('(F) canary / failure markers', flush=True)
fail_markers = ['S6F_SKIPPED', 'S6F_SETUP_FAILED', 'S6F_DEC_FALLBACK', 'S6F_KILLED', 'S6F_RUN_CELL_ERROR', 'S6F_BLEND_CELL_ERROR',
                'S6F_DRIVER_FAILED', 'S6F_NO_BASE', 'S6F_CANARY_WRITE_ERROR', 'S6F_RECEIPT_ERROR']
if A.expect_canary:
    others = [c for c in LAB if c != 'ACL']
    same = sub is not None and base is not None and all(pd.read_csv(P('submission.csv'), dtype=str)[c].tolist() ==
                                                        pd.read_csv(base_p, dtype=str)[c].tolist() for c in others)
    chk('control: S6F canary fired (mode V224FLIP_CANARY, fired [s6f->ACL]), ACL == 0.5 every row, other 11 columns == the base (text)',
        rec.get('mode') == 'V224FLIP_CANARY' and rec.get('fired') == ['s6f->ACL'] and 'S6F_CANARY_FIRED' in text and sub is not None
        and bool((sub['ACL'] == 0.5).all()) and same, (rec.get('mode'), blend.get('why')))
else:
    hits = [m for m in fail_markers + ['S6F_CANARY_FIRED'] if m in text]
    chk('canary not fired: mode S6F, fired [], no S6F failure marker in the log', rec.get('mode') == 'S6F' and rec.get('fired') == [] and not hits,
        (rec.get('mode'), hits))
    const = [c for c in LAB if sub is not None and sub[c].nunique() <= 1]
    chk('no constant column in submission.csv (ACL not blanked)', not const, const)
print('(G) log marker order', flush=True)
order = [LAST, 'S6F_DEC_OK', 'S6F_SETUP_OK'] + (['S6F_START', 'S6F_DRIVER_DONE', 'S6F_STAGE_DONE', 'S6F_BLEND_OK'] if not A.expect_canary
                                                      else {'skip': ['S6F_SKIPPED', 'S6F_CANARY_FIRED'],
                                                            'kill': ['S6F_START', 'S6F_KILLED', 'S6F_KILL_CHECK', 'S6F_STAGE_DONE',
                                                                     'S6F_CANARY_FIRED']}.get(CONTROL, ['S6F_CANARY_FIRED'])) + ['S6F_DONE']
pp_ = [text.find(m) for m in order]
chk('markers present and in cell order', all(p >= 0 for p in pp_) and pp_ == sorted(pp_), dict(zip(order, pp_)))
print('(H) submission format', flush=True)
chk('submission columns == sample, ids == sample order, finite in [0, 1]', sub is not None and list(sub.columns) == cols and
    sub['StudyInstanceUID'].tolist() == ids and np.isfinite(sub[labels].to_numpy(np.float64)).all() and
    ((sub[labels].to_numpy(np.float64) >= 0) & (sub[labels].to_numpy(np.float64) <= 1)).all())
chk('submission sha == s6f_receipt.blend', sub is not None and sha_file(P('submission.csv')) == blend.get('submission_sha256'))
n_ok = sum(R)
print(f'V224S6F2_CHECK {"PASS" if all(R) else "FAIL"} {n_ok}/{len(R)} base {BASE} mode {mode}', flush=True)
sys.exit(0 if all(R) else 1)
