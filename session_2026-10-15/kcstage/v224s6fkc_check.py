#!/usr/bin/env python
# v224s6fkc_check.py (10-15, Claude subagent, kcstage): GPU-commit gate of the S6F+KC kernel built by build_v224s6fkc.py
# (PREREG_10-15 Amendment 6 G3). Reads a fetched kernel output dir (kaggle kernels output eeshsaxena/rsna-sub-TAG -p OUT) and the box
# CPU parity JSON (kc_box_parity.py on OUT's kc/dump). Prints V224S6FKC_CHECK PASS/FAIL n/m; exit 0 only on PASS. Run with -I.
#   (A) base rows identical with KC removed: the frozen base submission_s6f.csv (sha == setup receipt) + the output with every KC file
#       removed + the log cut at S6F_DONE pass the UNCHANGED base gate (s6f: v224s6f_check.py, J: j_check.py vs the S6F commit) and
#       equal the base kernel's v1 commit submission within 1e-6; the freeze ran after S6F_DONE; no Traceback / RuntimeError
#   (B) KC parity: kc_box_parity.json PASS (box CPU fp32 run of the SAME member files on the dumped visible studies: max |dp| <= 1e-3,
#       labels by name; side / rule / flips / serve / vol sha / views re-derived from kc_common); its member, manifest, kc_probs and dump
#       shas == this output's; the side table == the side kernel's own Kaggle output on the visible studies (all columns); kc_probs
#       kc_side / kc_rule / kc_serve == the side table
#   (B2) C9 (A6 G3, KCS2 fix): box/c9_served.py JSON PASS = kc_prob_parity.py (ca9cd0ca) VERBATIM on the SERVED path (kc_worker.study_inputs
#       <- kc_serve.make_study_inputs, serve sha == build; side rows from the side kernel's own study() == the store side table) on >= 20
#       box train studies incl. >= 1 weak-zone (served on both canons) and >= 1 SagPD-absent: store vs served DICOM max |dp| <= 1e-3, the
#       SAME member files (names in manifest order + sha256 == build, manifest sha == build). REQUIRED for a real KC1 build (manifest
#       kind kc1); for a stand-in build it is checked only when supplied (default OUT/c9_served.json, written by gpu_commit_kc.sh)
#   (C) blend: submission.csv == .8 pct(base) + .2 pct(KC) (independent average-tie ranks; unscored -> base rank; constant base
#       columns unchanged) within 1e-6; receipt w .2 / cov .98 / canary Baker's
#   (D) coverage 1.0 (driver == blend), no unscored study
#   (E) receipts: setup (manifest sha, member shas, driver / serve sha, KC PYTHONPATH /tmp/kc/code first, JPEG decoders on it), policy
#       == build (G_KC, budget, deadline <= NB_T0 + 8.2 h), run RAN rc 0 not killed start age <= G_KC, driver ok + no test overrides +
#       code pins, env versions / GPUs / import dirs, one serve process per T4 (cuda:0 shard 0/2, cuda:1 shard 1/2) with the members'
#       model_sha f141aeed; info MICROBENCH + RECAL (sim_s6fkc_runtime.py inputs) and PICK_ELIGIBLE_J
#   (F) KC canary not fired, no KC failure marker, Baker's not constant; (G) log marker order; (H) submission format + sha
#   --expect-canary (control build --g-kc-h 0): KC canary expected (Baker's == 0.5, other 11 columns == the base text), run SKIPPED_GUARD
#   usage: py -3.12 -I v224s6fkc_check.py OUT [--build build_receipt_TAG.json] [--box-parity OUT/kc_box_parity.json] [--c9 OUT/c9_served.json]
#          [--side-ref ../kcrop/out_kc_side_v1/kc_side_test.csv] [--sample sample_submission.csv] [--expect-canary]
import os, re, sys, json, glob, shutil, tempfile, hashlib, argparse, subprocess
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SESS = os.path.dirname(HERE)
ROOT = os.path.dirname(SESS)
ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--build', default=None, help='default: build_receipt_<tag>.json next to this script, tag from the log name')
ap.add_argument('--box-parity', default=None, help='default: OUT/kc_box_parity.json')
ap.add_argument('--c9', default=None, help='default: OUT/c9_served.json (box/c9_served.py; required for a real KC1 build)')
ap.add_argument('--side-ref', default=os.path.join(SESS, 'kcrop', 'out_kc_side_v1', 'kc_side_test.csv'))
ap.add_argument('--sample', default=os.path.join(ROOT, '_data', 'sample_submission.csv'))
ap.add_argument('--s6f-gate', default=os.path.join(SESS, 's6f', 'v224s6f_check.py'))
ap.add_argument('--j-gate', default=os.path.join(SESS, 'jpeg', 'overlay', 'j_check.py'))
ap.add_argument('--s6f-out', default=os.path.join(SESS, 's6f', 'out_v224flips6f_v1'))
ap.add_argument('--s6fj-out', default=os.path.join(SESS, 'jpeg', 'overlay', 'r2', 'out_v224flips6fj_v1'))
ap.add_argument('--expect-canary', action='store_true')
ap.add_argument('--skip-base-gate', action='store_true', help='mock tests only: the base gate needs the full real commit output')
A = ap.parse_args()
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
       'Contusion', 'Fracture']
MODEL_SHA = 'f141aeed136dfc01'
TOL_PARITY, TOL_BLEND = 1e-3, 1e-6
KC_FILES = ('kc', 'kc_setup_receipt.json', 'kc_receipt.json', 'submission_s6f.csv', 'kc_box_parity.json', 'c9_served.json')
KPP_MD5 = 'ca9cd0ca7310cf5d6853a50a6fabdd70'                          # kc_prob_parity.py, the A6-frozen C9 tool
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
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(1 << 22), b''):
                h.update(b)
        return h.hexdigest()
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


def find_log(d):
    logs = [p for p in glob.glob(os.path.join(d, '*.log')) if 'cufile' not in os.path.basename(p)]
    return logs[0] if logs else None


def read_log(p):
    if not p:
        return '', None
    try:
        return '\n'.join(e.get('data', '') for e in json.load(open(p, encoding='utf-8'))), 'json'
    except Exception:
        return open(p, encoding='utf-8', errors='replace').read(), 'text'


logp = find_log(A.out)
text, logfmt = read_log(logp)
m = re.fullmatch(r'rsna-sub-(\w+)\.log', os.path.basename(logp or ''))
TAG = m.group(1) if m else None
B = jload(A.build or os.path.join(HERE, 'build_receipt_%s.json' % TAG), {}) or {}
POL = B.get('policy') or {}
J = B.get('base') == 's6fj'
mode = 'canary-control' if A.expect_canary else 'KC'
print(f'V224S6FKC_CHECK {A.out} mode {mode} tag {TAG} build {str(B.get("notebook_sha256"))[:16]} base {B.get("base")}', flush=True)
chk('build receipt present (tag == log name, policy, driver / serve sha, KC dataset manifest)', B.get('tag') == TAG and bool(POL)
    and bool(B.get('driver_sha256')) and bool(B.get('serve_sha256')) and bool((B.get('kc_dataset') or {}).get('manifest_sha256')),
    (B.get('tag'), TAG))
if A.expect_canary:
    chk('control build (policy.control skip), never submit', POL.get('control') == 'skip' and B.get('never_submit') is True, POL.get('control'))
sample = pd.read_csv(A.sample, dtype={'StudyInstanceUID': str})
cols, ids = list(sample.columns), sample['StudyInstanceUID'].tolist()
assert cols[1:] == LAB
setup = jload(P('kc_setup_receipt.json'), {}) or {}
rec = jload(P('kc_receipt.json'), {}) or {}
run = rec.get('run') or {}
blend = rec.get('blend') or {}
drv = jload(P('kc', 'kc_driver_receipt.json'), {}) or {}
sub = pd.read_csv(P('submission.csv'), dtype={'StudyInstanceUID': str}) if os.path.isfile(P('submission.csv')) else None
base_p = P('submission_s6f.csv')
base = pd.read_csv(base_p, dtype={'StudyInstanceUID': str}) if os.path.isfile(base_p) else None

# ---- (A) base rows identical with KC removed -------------------------------------------------------------------------------
print('(A) base rows (KC removed)', flush=True)
chk('frozen base submission_s6f.csv present; sha == KC setup receipt == kc_receipt.setup', base is not None and sha_file(base_p)
    == setup.get('base_sha256') == (rec.get('setup') or {}).get('base_sha256'), str(sha_file(base_p))[:16])
pos = {k: text.find(k) for k in ('S6F_DONE', 'KC_SETUP_OK', 'KC_SETUP_FAILED')}
chk('the freeze ran after the S6F stage (S6F_DONE before KC_SETUP_OK / KC_SETUP_FAILED)',
    0 <= pos['S6F_DONE'] < max(pos['KC_SETUP_OK'], pos['KC_SETUP_FAILED']), pos)
if not A.skip_base_gate:
    view = tempfile.mkdtemp(prefix='kcview_')
    try:
        for f in os.listdir(A.out):
            s = P(f)
            if f in KC_FILES or f == 'submission.csv' or s == logp:
                continue
            if os.path.isdir(s):
                shutil.copytree(s, os.path.join(view, f))
            else:
                shutil.copyfile(s, os.path.join(view, f))
        if base is not None:
            shutil.copyfile(base_p, os.path.join(view, 'submission.csv'))
        base_tag = 'v224flips6fj' if J else 'v224flips6f'
        lv = os.path.join(view, 'rsna-sub-%s.log' % base_tag)
        if logfmt == 'json':
            E = json.load(open(logp, encoding='utf-8'))
            k = next(i for i, e in enumerate(E) if 'S6F_DONE' in e.get('data', ''))
            json.dump(E[:k + 1], open(lv, 'w', encoding='utf-8'))
        else:
            open(lv, 'w', encoding='utf-8').write(text[:text.find('\n', text.find('S6F_DONE')) + 1])
        if J:
            cmd = [sys.executable, '-I', A.j_gate, view, '--base-out', A.s6f_out, '--tag', base_tag]
            want = 'J_CHECK PASS'
        else:
            cmd = [sys.executable, '-I', A.s6f_gate, view, '--sample', A.sample]
            want = 'V224S6F_CHECK PASS'
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        last = [l for l in r.stdout.splitlines() if l.startswith(want.split()[0] + ' ')][-1:] or ['(no verdict)']
        fails = [l.strip() for l in r.stdout.splitlines() if l.strip().startswith('FAIL')]
        chk('the output with every KC file removed (submission = the frozen base, log cut at S6F_DONE) passes the UNCHANGED base gate (%s)'
            % os.path.basename(A.j_gate if J else A.s6f_gate), r.returncode == 0 and last[0].startswith(want),
            last[0] + (' | ' + ' | '.join(fails)[:600] if fails else ''))
    finally:
        shutil.rmtree(view, ignore_errors=True)
else:
    print('  info base gate SKIPPED (--skip-base-gate: mock tests only)', flush=True)
try:
    ref_sub = pd.read_csv(os.path.join(A.s6fj_out if J else A.s6f_out, 'submission.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    d = float(np.abs(base.set_index('StudyInstanceUID').loc[ids, LAB].to_numpy(np.float64) - ref_sub.loc[ids, LAB].to_numpy(np.float64)).max())
except Exception as e:
    d = float('inf')
chk('frozen base == the base kernel v1 commit submission (%s) within 1e-6' % ('S6F-J' if J else 'S6F'), d <= TOL_BLEND, 'max diff %.3g' % d)
tb = [k for k in ('Traceback', 'RuntimeError') if k in text]
chk('no Traceback / RuntimeError anywhere in the kernel log (KC part included)', not tb, tb)
s6mode = (setup.get('s6f') or {}).get('mode')
print('  info base S6F mode %s (fired %s); V224 canary fired %s' % (s6mode, (setup.get('s6f') or {}).get('fired'),
                                                                  (setup.get('v224_canary') or {}).get('fired')), flush=True)
_pre = text[:text.find('V224_DONE')] if 'V224_DONE' in text else ''
_jb = bool(re.search(r'(^|\n)DEC_OK ', _pre))
print('  info PICK_ELIGIBLE_J %s (base-stage DEC_OK before V224_DONE: %s)' % ('yes' if _jb else 'no', _jb), flush=True)

# ---- (E) receipts -------------------------------------------------------------------------------------------------------------
print('(E) receipts, environment, policy', flush=True)
KD = B.get('kc_dataset') or {}
chk('kc_receipt tag == build tag', rec.get('tag') == B.get('tag'), rec.get('tag'))
pp = str(setup.get('pythonpath_kc') or '').split(':')
dec_dirs = [p for p in pp if p in ('/tmp/s6f_dec', '/tmp/_dec')]
chk('setup: ok, manifest sha == build, member shas == build, driver / serve sha == build, PYTHONPATH /tmp/kc/code first, JPEG decoders '
    'on it (%s)' % ('/tmp/_dec inherited (J)' if J else '/tmp/s6f_dec'), setup.get('ok') is True and setup.get('manifest_sha256') ==
    KD.get('manifest_sha256') and setup.get('member_sha256') == KD.get('member_sha256') and setup.get('driver_sha256') ==
    B.get('driver_sha256') and setup.get('serve_sha256') == B.get('serve_sha256') and pp[0] == '/tmp/kc/code' and
    (('/tmp/_dec' in dec_dirs) if J else ('/tmp/s6f_dec' in dec_dirs)), {k: setup.get(k) for k in ('ok', 'why', 'pythonpath_kc', 'dataset_dir')})
pol_run = {k: run.get(k) for k in ('g_kc_h', 'budget_h', 'deadline_h', 'stop_margin_s', 'min_run_s')}
pol_b = {'g_kc_h': POL.get('g_kc_h'), 'budget_h': POL.get('budget_h'), 'deadline_h': POL.get('deadline_h'),
         'stop_margin_s': POL.get('stop_margin_s'), 'min_run_s': POL.get('min_run_s')}
chk('run: policy == build (G_KC, budget, deadline, stop margin, min run), hard deadline (a number) <= NB_T0 + 8.2 h', pol_run == pol_b and
    isinstance(run.get('hard_deadline_age_h'), (int, float)) and run['hard_deadline_age_h'] <= 8.2 + 1e-9, (pol_run, run.get('hard_deadline_age_h')))
env = drv.get('env') or {}
mods = env.get('mods') or {}
if A.expect_canary:
    chk('control skip: run SKIPPED_GUARD (start age > G_KC), KC_SKIPPED start guard, the driver never started (no KC_START)',
        run.get('status') == 'SKIPPED_GUARD' and isinstance(run.get('start_age_h'), (int, float)) and run['start_age_h'] > run.get('g_kc_h', 99)
        and 'KC_SKIPPED start guard' in text and 'KC_START' not in text, {k: run.get(k) for k in ('status', 'start_age_h', 'g_kc_h')})
else:
    chk('run: RAN in its own subprocess, rc 0, not killed, start age <= G_KC', run.get('status') == 'RAN' and run.get('rc') == 0 and
        run.get('killed') is False and isinstance(run.get('start_age_h'), (int, float)) and run['start_age_h'] <= POL.get('g_kc_h', -1),
        {k: run.get(k) for k in ('status', 'rc', 'killed', 'start_age_h', 'seconds')})
    chk('driver receipt ok (file == the copy in kc_receipt), no test override, code pins (kc_worker 4af60f6f, kc_common 45255ae0, fleet '
        'models, d96 reader 30aa72fd, side kernel 2979c191)', drv.get('ok') is True and (run.get('driver') or {}).get('ok') is True and
        (run.get('driver') or {}).get('total_seconds') == drv.get('total_seconds') and drv.get('overrides') == {} and
        (drv.get('inputs') or {}).get('code_pins') == {**B.get('code_md5', {}), **B.get('code_sha256', {})},
        (drv.get('error'), drv.get('overrides')))
    need = ('torch', 'timm', 'pydicom', 'numpy', 'pandas', 'cv2')
    chk('env probe: python / torch / timm / pydicom / numpy / pandas / cv2 versions, 2 GPUs, no R7 / S6F / working dir on any import path',
        bool(env.get('python')) and all(not str((mods.get(k) or ['MISSING'])[0]).startswith('MISSING') for k in need) and
        env.get('n_gpu') == 2 and env.get('r7_s6f_or_working_on_path') == [], {k: (mods.get(k) or [None])[0] for k in need})
    ddir = {k: (mods.get(k) or [None, None])[1] for k in ('pylibjpeg', 'libjpeg', 'openjpeg')}
    chk('import dirs: kc_common / kc_worker from /tmp/kc/code; JPEG decoders from a DEC dir (/tmp/s6f_dec or /tmp/_dec); torch / timm / '
        'pydicom from the image dist-packages', (mods.get('kc_common') or [None, None])[1] == '/tmp/kc/code' ==
        (mods.get('kc_worker') or [None, None])[1] and all(v in ('/tmp/s6f_dec', '/tmp/_dec') for v in ddir.values()) and
        all(str((mods.get(k) or [None, ''])[1]).endswith('-packages') for k in ('torch', 'timm', 'pydicom')), {**ddir, 'timm': (mods.get('timm') or [None])[0]})
    W_ = drv.get('workers') or []
    nm = len(KD.get('members') or [])
    okw = len(W_) == 2 and all(w.get('rc') == 0 and w.get('device') == 'cuda:%d' % k and w.get('shard') == '%d/2' % k and w.get('procs') == 2
                               and isinstance(w.get('closing'), str) and (' %d members' % nm) in w['closing'] and 'coverage 1.0000' in w['closing']
                               and ('model_sha %s' % MODEL_SHA) in w['closing'] and ('device cuda:%d' % k) in w['closing'] for k, w in enumerate(W_))
    chk('serve: one process per T4 (cuda:0 shard 0/2, cuda:1 shard 1/2, --procs 2), rc 0, %d members, model_sha %s, coverage 1.0000 each'
        % (nm, MODEL_SHA), okw, [w.get('closing') for w in W_])
    mb = (drv.get('B') or {}).get('microbench') or {}
    print('  info MICROBENCH %s | stages %s | gpus %s' % (json.dumps(mb), json.dumps({k: v.get('seconds') for k, v in (drv.get('stages') or {}).items()}),
                                                      [w.get('gpu_name') for w in W_]), flush=True)
    # RECAL (MODEL, sim_s6fkc_runtime.py): effective c_kc ~ 0.4 x decode s per study per decode process (4 decoders on 4 vCPUs);
    # the sim assumed c_kc U(1.0, 2.2) and GPU 0.27-0.31 s per member per study. Rebuild with a new G_KC only on RECAL_TRIGGER yes.
    dec, gw = mb.get('decode_s_per_study_per_proc'), mb.get('gpu_s_per_study_warm')
    if isinstance(dec, (int, float)) and isinstance(gw, (int, float)) and nm:
        ck = 0.4 * dec
        gpm = gw / nm / (1.0 + float(mb.get('both_frac') or 0))
        trig = ck > 2.2 or gpm > 0.45
        print('  info RECAL implied c_kc %.2f s/study (sim U(1.0, 2.2)) | warm GPU %.3f s per member per study (sim 0.27-0.31, trigger > 0.45)'
              % (ck, gpm), flush=True)
        print('  info RECAL_TRIGGER %s%s' % ('yes' if trig else 'no', ': rescale KC_C in sim_s6fkc_runtime.py, re-run, rebuild with --g-kc-h'
                                                                      if trig else ''), flush=True)
    else:
        print('  info RECAL_TRIGGER n/a (no decode / warm GPU microbench: %s, %s)' % (dec, gw), flush=True)

# ---- (B) parity, (C) blend, (D) coverage --------------------------------------------------------------------------------------
if not A.expect_canary:
    print('(B) KC parity: box CPU run of the same members on the visible studies; side table vs the side kernel', flush=True)
    bp = jload(A.box_parity or P('kc_box_parity.json'), {}) or {}
    dumps = sorted(glob.glob(P('kc', 'dump', '*.npz')))
    dsha = {os.path.basename(p)[:-4]: sha_file(p) for p in dumps}
    chk('box parity PASS: max |dp| <= 1e-3 over the %d visible studies, labels by name, all re-derivation checks (side / rule / flips / '
        'serve / vol sha / views)' % len(ids), bp.get('PASS') is True and bp.get('n') == len(ids) and isinstance(bp.get('max_dp'), (int, float))
        and bp['max_dp'] <= TOL_PARITY and bp.get('all_checks') is True and bp.get('labels') == list(LAB),
        'max |dp| %s by label %s' % (bp.get('max_dp'), {k: '%.1e' % v for k, v in (bp.get('max_dp_by_label') or {}).items()}))
    chk('box parity inputs == this run: member shas == build, manifest sha == build, kc_probs sha == this output, dump npz shas == this output',
        bp.get('members') == KD.get('member_sha256') and bp.get('manifest_sha256') == KD.get('manifest_sha256') and
        bp.get('probs_sha256') == sha_file(P('kc', 'kc_probs.csv')) and {r.get('sid'): r.get('npz_sha256') for r in bp.get('rows') or []} == dsha
        and len(dsha) == len(ids), (len(dsha), str(bp.get('probs_sha256'))[:16]))
    try:
        mine = pd.read_csv(P('kc', 'kc_side_test.csv'), dtype={'StudyInstanceUID': str, 'tag': str}).set_index('StudyInstanceUID').sort_index()
        ref = pd.read_csv(A.side_ref, dtype={'StudyInstanceUID': str, 'tag': str}).set_index('StudyInstanceUID').loc[mine.index]
        diff = {}
        for c in set(mine.columns) | set(ref.columns):
            if c not in mine.columns or c not in ref.columns:
                diff[c] = 'missing'
                continue
            a, b = mine[c], ref[c]
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                x, y = a.to_numpy(np.float64), b.to_numpy(np.float64)
                bad = ~((np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= 1e-9 * np.maximum(1, np.abs(y))))
            else:
                bad = (a.fillna('<NA>').astype(str) != b.fillna('<NA>').astype(str)).to_numpy()
            if bad.any():
                diff[c] = int(bad.sum())
        chk('side table (kc_side_test.csv) == the side kernel\'s own Kaggle output on the visible studies, every column (A6 G3 "the side '
            'matches the header rule")', sorted(mine.index) == sorted(ids) and not diff, diff)
        kp = pd.read_csv(P('kc', 'kc_probs.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
        same = all(str(kp.at[s, 'kc_side']) == str(mine.at[s, 'side']) and str(kp.at[s, 'kc_rule']) == str(mine.at[s, 'rule'])
                   and (str(kp.at[s, 'kc_serve']) == 'both' if str(mine.at[s, 'serve']) == 'both' else True) for s in ids)
        chk('kc_probs kc_side / kc_rule == the side table; a study the table serves on both canons is served on both', same,
            {s[-8:]: (kp.at[s, 'kc_side'], kp.at[s, 'kc_rule'], kp.at[s, 'kc_serve']) for s in ids})
    except Exception as e:
        chk('side table comparable', False, repr(e)[:300])
    print('(B2) C9 (A6 G3): store vs SERVED-path probabilities of the same member files on >= 20 box train studies', flush=True)
    c9 = jload(A.c9 or P('c9_served.json'), None)
    REAL = KD.get('standin') is False
    if c9 is None and not REAL:
        print('  info C9 not supplied: stand-in build, not required (a real KC1 build requires it)', flush=True)
    else:
        c9 = c9 if isinstance(c9, dict) else {}
        kpp = c9.get('kpp') if isinstance(c9.get('kpp'), dict) else {}
        kr = [r for r in (kpp.get('rows') or []) if isinstance(r, dict)]
        n9 = len(kr)
        try:
            mx9 = max(float(r['max_dp']) for r in kr) if kr else float('inf')
        except Exception:
            mx9 = float('inf')
        wk = [r for r in kr if str(r.get('weak')) == '1']
        ab = [r for r in kr if 'sag_pd' in (r.get('absent') or [])]
        sids9 = [r.get('sid') for r in kr]
        ok9 = (c9.get('PASS') is True and c9.get('error') is None and c9.get('path') == 'served' and c9.get('substitution_live') is True
               and c9.get('kpp_md5') == KPP_MD5 and c9.get('serve_sha256') == B.get('serve_sha256') and c9.get('code_md5') == B.get('code_md5')
               and c9.get('side_source') == 'fresh' and c9.get('side_equal_ref') is True and kpp.get('PASS') is True and n9 >= 20
               and len(set(sids9)) == n9 and sorted(sids9) == sorted(c9.get('picks') or []) and mx9 <= TOL_PARITY
               and len(wk) >= 1 and all(r.get('serve') == 'both' for r in wk) and len(ab) >= 1)
        chk('C9 PASS on the SERVED path: kc_prob_parity.py ca9cd0ca + the kc_serve substitution (serve sha == build), side rows fresh from '
            'the side kernel == the store side table, n %d >= 20, weak-zone served on both canons %d >= 1, SagPD-absent %d >= 1, max |dp| '
            '%.2e <= 1e-3' % (n9, len(wk), len(ab), mx9), ok9, ({k: v for k, v in (c9.get('requirements') or {}).items() if not v} or
                                                               c9.get('error') or ('missing' if not c9 else '')))
        mem9 = c9.get('members') or []
        chk("C9 member files == this build's members (names in manifest order, sha256), manifest sha == build",
            [m.get('file') for m in mem9] == list(KD.get('members') or []) and len(mem9) > 0 and
            {m.get('file'): m.get('sha256') for m in mem9} == KD.get('member_sha256') and c9.get('manifest_sha256') == KD.get('manifest_sha256'),
            ([m.get('file') for m in mem9], str(c9.get('manifest_sha256'))[:16]))
    dmp = drv.get('dump') or {}
    chk('dump: every visible study, vol sha == the served kc_vol_sha', isinstance(dmp, dict) and sorted(dmp) == sorted(ids) and
        all(v.get('equal') is True for v in dmp.values()), {k[-8:]: v.get('equal', v.get('error')) for k, v in (dmp.items() if isinstance(dmp, dict) else [])})
    chk('kc_probs.csv sha == driver receipt == blend receipt', sha_file(P('kc', 'kc_probs.csv')) == (drv.get('B') or {}).get('probs_sha256')
        == blend.get('kc_probs_sha256'))
    print('(C) blend recompute', flush=True)
    try:
        kp = pd.read_csv(P('kc', 'kc_probs.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
        Sx = kp.loc[ids, LAB].to_numpy(np.float64)
        ok = np.isfinite(Sx).all(axis=1)
        bb = base.set_index('StudyInstanceUID').loc[ids, LAB]
        exp = np.zeros((len(ids), len(LAB)))
        const = [c for c in LAB if bb[c].nunique() <= 1]
        for j, c in enumerate(LAB):
            if c in const:
                exp[:, j] = bb[c].to_numpy(np.float64)
                continue
            rb = avg_rank_pct(bb[c].to_numpy(np.float64))
            rs = rb.copy()
            rs[ok] = avg_rank_pct(Sx[ok, j])
            exp[:, j] = 0.8 * rb + 0.2 * rs
        got = sub.set_index('StudyInstanceUID').loc[ids, LAB].to_numpy(np.float64)
        d = float(np.abs(got - exp).max())
        chk('submission.csv == .8 pct(base) + .2 pct(KC) (independent ranks, unscored -> base rank, constant base columns %s kept) within 1e-6'
            % const, d <= TOL_BLEND, 'max diff %.3g' % d)
        chk('blend receipt: mode KC, w .2, coverage floor .98, canary col Baker\'s, on fail canary', blend.get('mode') == 'KC' and
            blend.get('w_kc') == 0.2 == POL.get('w_kc') and blend.get('cov_min') == 0.98 and blend.get('canary_col') == "Baker's" and
            blend.get('on_fail') == 'canary', {k: blend.get(k) for k in ('mode', 'w_kc', 'on_fail')})
        print('  info KC moved %d of %d cells vs the frozen base' % (int((np.abs(got - bb.to_numpy(np.float64)) > 1e-6).sum()), got.size), flush=True)
    except Exception as e:
        chk('blend recompute computable', False, repr(e)[:300])
    print('(D) coverage', flush=True)
    bc = (drv.get('B') or {}).get('coverage')
    chk('coverage 1.0 on the visible studies (driver == blend), no unscored study', bc == 1.0 == blend.get('coverage') and
        (drv.get('B') or {}).get('n_unscored') == 0 == blend.get('n_unscored') and (drv.get('B') or {}).get('n') == len(ids), (bc, blend.get('coverage')))
else:
    print('(B-D) skipped: control build, KC canary expected', flush=True)

# ---- (F) canary, (G) markers, (H) format --------------------------------------------------------------------------------------
print('(F) canary / failure markers', flush=True)
fail_markers = ['KC_SKIPPED', 'KC_SETUP_FAILED', 'KC_KILLED', 'KC_RUN_CELL_ERROR', 'KC_BLEND_CELL_ERROR', 'KC_NO_BASE', 'KC_CANARY_WRITE_ERROR',
                'KC_RECEIPT_ERROR', 'KC_NOT_USED']
if A.expect_canary:
    others = [c for c in LAB if c != "Baker's"]
    same = sub is not None and base is not None and all(pd.read_csv(P('submission.csv'), dtype=str)[c].tolist() ==
                                                        pd.read_csv(base_p, dtype=str)[c].tolist() for c in others)
    chk("control: KC canary fired (mode KC_CANARY, fired [kc->Baker's]), Baker's == 0.5 every row, other 11 columns == the base (text)",
        rec.get('mode') == 'KC_CANARY' and rec.get('fired') == ["kc->Baker's"] and 'KC_CANARY_FIRED' in text and sub is not None
        and bool((sub["Baker's"] == 0.5).all()) and same, (rec.get('mode'), blend.get('why')))
else:
    hits = [k for k in fail_markers + ['KC_CANARY_FIRED'] if k in text]
    chk('KC canary not fired: mode KC, fired [], no KC failure marker in the log', rec.get('mode') == 'KC' and rec.get('fired') == [] and not hits,
        (rec.get('mode'), hits))
    bconst = [c for c in LAB if base is not None and base[c].nunique() <= 1]
    const = [c for c in LAB if sub is not None and sub[c].nunique() <= 1]
    chk("no constant column in submission.csv beyond the base's own (Baker's not blanked)", const == bconst and "Baker's" not in const,
        (const, bconst))
print('(G) log marker order', flush=True)
order = ['S6F_DONE', 'KC_SETUP_OK'] + (['KC_START', 'KC_SIDE_DONE', 'KC_SERVE_START', 'KC_DRIVER_DONE', 'KC_STAGE_DONE', 'KC_BLEND_OK']
                                       if not A.expect_canary else ['KC_SKIPPED', 'KC_CANARY_FIRED']) + ['KC_DONE']
pp_ = [text.find(k) for k in order]
chk('markers present and in cell order', all(p >= 0 for p in pp_) and pp_ == sorted(pp_), dict(zip(order, pp_)))
print('(H) submission format', flush=True)
chk('submission columns == sample, ids == sample order, finite in [0, 1]', sub is not None and list(sub.columns) == cols and
    sub['StudyInstanceUID'].tolist() == ids and np.isfinite(sub[LAB].to_numpy(np.float64)).all() and
    ((sub[LAB].to_numpy(np.float64) >= 0) & (sub[LAB].to_numpy(np.float64) <= 1)).all())
chk('submission sha == kc_receipt.blend', sub is not None and sha_file(P('submission.csv')) == blend.get('submission_sha256'))
print(f'V224S6FKC_CHECK {"PASS" if all(R) else "FAIL"} {sum(R)}/{len(R)} mode {mode} tag {TAG}', flush=True)
sys.exit(0 if all(R) else 1)
