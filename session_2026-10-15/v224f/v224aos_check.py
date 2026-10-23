#!/usr/bin/env python
# v224aos_check.py (session_2026-10-15/v224f, 10-15, Claude subagent): commit gate for the ALWAYS-ON-FLIP student kernels built by
# build_v224fs.py --base flip | flrm (V224-FLIP-AO + S = PREREG_10-15 Amendment 3 R2, branch b; V224-FLRM-AO + S = R3, branch c).
# Reads a fetched kernel output dir + the kernel dir of the push (BUILD_V224FS.json + the notebook). No em dashes.
# Prints V224AOS_CHECK PASS/FAIL n/m; exit 0 only on PASS.
#   usage: python v224aos_check.py OUT --wf 0.20 --build KERNEL_DIR/BUILD_V224FS.json --ref-exact OUT_V224 --ref-flip OUT_V224FLIP
#                                  [--ref-flrm OUT_V224FLRM (required for flrm)] [--ref-s OUT (same 5 S members)] [--sample CSV]
# (0) BUILD: base flip / flrm, REPLACE (k14 [], s_share 1, 5 S), w_f .20, start guard G in the runtime-reviewed set and
#     MIN_START_MIN == (8.3 - G) x 60; the notebook == the build sha; its first N cells re-serialise to the pinned base FILE (V224-FLIP
#     73a9d562 / V224-FLRM f14f6acc, == the Kaggle pulls) byte for byte; 9 appended cells in order (SAVE-AO, ENV-SNAP, FN-IDS,
#     K12-MERGE, S-MERGE, CLOCK, FLEET, ENV-RESTORE, HF-AO); no time-aware driver anywhere; K2_CFG MEMBERS == S, MIN_START_MIN, wall;
#     S_CFG manifest == the build (sha256 of its sorted json), every S sha256[:16] == its sha256_full.
# (1) the V224-F-S gate v224fs_check.py (subprocess): every check passes except exactly two by-design lines, each replaced here:
#     its exact-mode V224 sub-gate (R7 runs with flip; replaced by (2)) and its "build: MIN_START_MIN" line (it knows only the f / ff
#     budgets 198 / 192; replaced by (0) + (6)).
# (2) the base part on a VIEW of OUT (log cut after the base's last marker, submission.csv := submission_V224.csv): flip base ->
#     session_2026-10-14/v224/v224_check.py --flip --ref OUT_V224 (the gate V224-FLIP passed: FLIP_TTA_FORCED, FLIP_TTA_READY +
#     mcl plain pass in every worker log, R7 receipt flip block, R7 MCL == exact, >= 10 other R7 columns moved, canary clean,
#     independent recompute); flrm base -> v224rm/v224flrm_check.py (the gate V224-FLIP-RM passed, 10/10 on its commit).
# (3) flip ON FOR EVERY STUDY, no time-aware driver: FLIP_TTA_FORCED payload is the V224-FLIP form (worker_args only, --flip-tta once,
#     no --ta-* args); no driver marker / file anywhere (FLIP_TA_*, FLIP_TTA_DISABLED_AT_STUDY, compact_infer_ta, flip_ta_receipt.json);
#     R7 controller + every worker = pinned compact_infer.py (sha 8401812c) with the flip_tta block (M/L perm, MCL plain pass only);
#     every worker log FLIP_TTA_READY + WORKER_DONE studies == scored; the workers' per-study rows partition the test ids (every
#     study was scored by a --flip-tta worker); R7 ours_sub == the V224-FLIP commit R7 (1e-3); submission_V224.csv == the base commit
#     submission (1e-6).
# (4) S members and shas: S-MERGE "RESULT OK ... + S: 30 checkpoints (5 S from .../<s_ds>, sha256 verified"; fleet receipt names /
#     members_filter / K2 MEMBERS == the 5 S of the build; manifest_sha16 of each S == the build's S manifest; TTA AB; S microbench
#     <= 400 ms / member / study on every GPU (the runtime review's 1.62 s / study for 5); optional --ref-s: S group mean == the S
#     group mean of another output with the same 5 S (1e-3).
# (5) END-TO-END recompute (numpy ranks, independent of the notebook): H = V224 recipe from raptor_raw / raptor224_sub / ours_sub
#     (float32) == submission_V224.csv (1e-6); F = mean of the 5 S rows of the worker npz; submission.csv == (1 - w) pct(H) + w pct(F).
# (6) canary not fired + start guard honoured: no CANARY_FIRED / HF_FALLBACK / RM_GUARD_FIRED / V224F_H_FROM_MEMORY; canary receipt
#     fired [] (flrm: rm_guard mirror_ok); hf_receipt mode HF, blanked [], fired_before_hf [] (the HF-AO key: proves the AO HF cell
#     ran); v224f_receipt fired []; no constant column; fleet started at kernel age <= G with >= MIN_START_MIN left.
import os, re, sys, json, glob, shutil, hashlib, argparse, subprocess, tempfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
V224FS_CHECK = os.path.join(HERE, 'v224fs_check.py')
V224_CHECK = os.path.join(ROOT, 'session_2026-10-14', 'v224', 'v224_check.py')
FLRM_CHECK = os.path.join(ROOT, 'session_2026-10-15', 'v224rm', 'v224flrm_check.py')
BASES = {'flip': dict(sha='73a9d5620d0a637ed171ecec13bcf243bd0af67992b5b4cb8ad7f35bf72b1b1b', cells=15, last='V224_DONE',
                      nb='session_2026-10-14/v224/kernel_v224flip/rsna_sub_v224flip.ipynb',
                      pulled='session_2026-10-15/v224rm/pulled_v224flip/rsna-sub-v224flip.ipynb'),
         'flrm': dict(sha='f14f6accc7c178536ff8a5d8d70a65da63ca05b2dd6d2a7ce9673100f9178ecf', cells=17, last='V224RM_DONE',
                      nb='session_2026-10-15/v224rm/kernel_v224flrm/rsna_sub_v224flrm.ipynb',
                      pulled='session_2026-10-15/v224rm/pulled_v224flrm/rsna-sub-v224flrm.ipynb')}
AO_GUARDS = (6.0, 6.5, 7.0)
CI_SHA = '8401812cd63737ec99839d317c4037c4efb445d1baec892fa7481bb30399cc2d'
S_RE = re.compile(r'^f([0-4])_s1047r7tb_raw_ttr224-320x24_cv\d{4}_best\.pt$')
FLIP_PERM = [0, 1, 3, 2, 5, 4, 6, 7, 8, 9, 10, 11]
TG = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
      'Contusion', 'Fracture']                                    # the fleet cell's _K2_TARGETS order (npz columns)
EXPECTED_FS_FAILS = ('V224 part (log up to V224_DONE, submission_V224.csv) passes v224_check.py exact', 'build: MIN_START_MIN')
DRIVER_MARKS = ('FLIP_TA_', 'FLIP_TTA_DISABLED_AT_STUDY', 'compact_infer_ta', 'CONTROL_FLIP_BUDGET_H', 'V224FF_FLIP_BUDGET_H', '--ta-')
SAVE_HDR = '# [eeshsaxena V224-AO-S SAVE (V224-F SAVE: header + knob dict only)'
HF_HDR = '# [eeshsaxena V224-AO-S HF BLEND;'
AO_RULE = "_hf_blank = ['Fracture'] if any(str(_x).endswith('->Fracture') for _x in _hf_fired) else []\n"

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--wf', type=float, required=True)
ap.add_argument('--build', required=True, help='BUILD_V224FS.json of the pushed kernel (its notebook must sit next to it)')
ap.add_argument('--ref-exact', required=True, help='V224-exact commit output dir (rsna-sub-v224 v1)')
ap.add_argument('--ref-flip', required=True, help='V224-FLIP commit output dir (rsna-sub-v224flip v1)')
ap.add_argument('--ref-flrm', default=None, help='V224-FLIP-RM commit output dir (rsna-sub-v224flrm v1); required for a flrm build')
ap.add_argument('--ref-s', default=None, help='an output dir whose worker npz hold the same 5 S members (e.g. the other AO commit)')
ap.add_argument('--sample', default=os.path.join(ROOT, '_data', 'sample_submission.csv'))
A = ap.parse_args()
B = json.load(open(A.build, encoding='utf-8'))
BASE = B.get('base')
assert BASE in BASES, 'v224aos_check is for build_v224fs.py --base flip | flrm builds (this build: base %r)' % BASE
assert BASE != 'flrm' or A.ref_flrm, '--ref-flrm (the V224-FLIP-RM commit output) is required for a flrm build'
BD = BASES[BASE]
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


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def jload(p):
    try:
        return json.load(open(p, encoding='utf-8'))
    except Exception:
        return None


def src(c):
    s = c['source']
    return ''.join(s) if isinstance(s, list) else s


def dumps(nb):
    return json.dumps(nb, ensure_ascii=False, separators=(',', ':'))


def lit(text, name):
    line = text.split('\n', 1)[0]
    return eval(line[len(name + ' = '):line.index('   # literal')], {})


def line_json(text, marker):
    for ln in text.splitlines():
        i = ln.find(marker + ' ')
        if i >= 0:
            try:
                return json.loads(ln[i + len(marker) + 1:].strip())
            except Exception:
                return None
    return None


def line_json_head(text, marker):                                 # JSON object right after the marker, trailing text allowed
    for ln in text.splitlines():
        i = ln.find(marker + ' ')
        if i >= 0:
            try:
                return json.JSONDecoder().raw_decode(ln[i + len(marker) + 1:].strip())[0]
            except Exception:
                return None
    return None


def pct(v):
    v = np.asarray(v, dtype=np.float64)                         # average-tie percentile rank, own implementation
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


def csv_(d, name):
    return pd.read_csv(os.path.join(d, name), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')


def top_checks(stdout):
    return [('OK', ln[7:]) if ln.startswith('  OK   ') else ('FAIL', ln[7:]) for ln in stdout.splitlines()
            if ln.startswith('  OK   ') or ln.startswith('  FAIL ')]


G = B.get('start_budget_h')
MIN_START = round((8.3 - G) * 60.0, 1) if isinstance(G, (int, float)) else None
print(f'V224AOS_CHECK {A.out} base {BASE} tag {B.get("tag")} w_f {A.wf} guard {G} h (MIN_START_MIN {MIN_START})'
      f'{" STAND-IN BUILD (offline test only)" if B.get("standin") else ""}', flush=True)
logs = [p for p in glob.glob(P('*.log')) if 'cufile' not in os.path.basename(p)]
entries, text = None, ''
if logs:
    try:
        entries = json.load(open(logs[0], encoding='utf-8'))
        text = '\n'.join(e.get('data', '') for e in entries)
    except Exception:
        text = open(logs[0], encoding='utf-8', errors='replace').read()
chk('kernel log present', bool(text), os.path.basename(logs[0]) if logs else '')
sample = pd.read_csv(A.sample, dtype={'StudyInstanceUID': str})
cols = list(sample.columns); labels = cols[1:]; ids = sample['StudyInstanceUID'].tolist()

# ---- (0) the build ---------------------------------------------------------------------------------------------------------------
S = list(B.get('s') or [])
chk('(0) build: AO REPLACE (k14 [], 5 S named f0-f4 s1047r7tb, s_share 1.0, members 5), w_f %.2f == .20, no flip budget' % A.wf,
    B.get('replace') is True and B.get('k14') == [] and B.get('s_share') == 1.0 and B.get('members') == 5 and len(S) == 5
    and sorted(int(S_RE.match(n).group(1)) if S_RE.match(n) else -1 for n in S) == [0, 1, 2, 3, 4]
    and B.get('w_f') == A.wf and abs(A.wf - 0.20) < 1e-12 and B.get('flip_budget_h') is None and B.get('patched_base_cells') == [],
    {k: B.get(k) for k in ('replace', 'k14', 's_share', 'members', 'w_f', 'flip_budget_h')})
chk('(0) build: start guard G in the runtime-reviewed %r, MIN_START_MIN == (8.3 - G) x 60, wall 8.3, ao_guard_h == G' % (AO_GUARDS,),
    G in AO_GUARDS and B.get('min_start_min') == MIN_START and B.get('wall_h') == 8.3 and B.get('ao_guard_h') == G,
    {k: B.get(k) for k in ('start_budget_h', 'min_start_min', 'wall_h', 'ao_guard_h')})
KD = os.path.dirname(os.path.abspath(A.build))
nbs = [p for p in glob.glob(os.path.join(KD, '*.ipynb'))]
NB = None
try:
    assert len(nbs) == 1, 'kernel dir must hold exactly one notebook: %r' % nbs
    assert sha(nbs[0]) == B['notebook_sha256'], 'notebook sha != BUILD_V224FS.json'
    NB = json.load(open(nbs[0], encoding='utf-8'))
    chk('(0) notebook sha256 == the build (%s...)' % B['notebook_sha256'][:16], True, os.path.basename(nbs[0]))
except Exception as e:
    chk('(0) notebook sha256 == the build', False, repr(e)[:200])
try:
    braw = open(os.path.join(ROOT, BD['nb']), 'rb').read()
    assert hashlib.sha256(braw).hexdigest() == BD['sha'] == B.get('base_sha256'), 'pinned base sha / build base_sha256'
    pl = os.path.join(ROOT, BD['pulled'])
    assert not os.path.isfile(pl) or open(pl, 'rb').read() == braw, 'base != its Kaggle pull'
    bnb = json.loads(braw.decode('utf-8'))
    assert len(bnb['cells']) == BD['cells'] == B.get('base_cells')
    back = dict(NB); back['cells'] = NB['cells'][:BD['cells']]
    chk('(0) base cells 0-%d re-serialise to the %s FILE byte for byte (sha %s, == Kaggle pull %s)' % (BD['cells'] - 1, BASE,
        BD['sha'][:16], os.path.isfile(pl)), dumps(back).encode('utf-8') == braw)
except Exception as e:
    chk('(0) base cells byte-identical to the pinned base', False, repr(e)[:200])
C = [src(c) for c in (NB or {}).get('cells', [])]
N0 = BD['cells']
heads = ['# [eeshsaxena V224-AO-S SAVE', '# === HF ENV-SNAP', '# === FN-IDS', 'K12M_CFG = {', 'S_CFG = {', '# [eeshsaxena V224-F CLOCK',
         'K2_CFG = {', '# === HF ENV-RESTORE', HF_HDR]
app = C[N0:]
chk('(0) 9 appended cells in order: SAVE-AO, ENV-SNAP, FN-IDS, K12-MERGE, S-MERGE, CLOCK, FLEET, ENV-RESTORE, HF-AO',
    len(app) == 9 and all(a.startswith(h) for a, h in zip(app, heads)) and app[0].startswith(SAVE_HDR) and AO_RULE in app[8]
    and "'ao_base': %r" % BASE in app[0], [a[:40] for a in app])
dr = [m for m in DRIVER_MARKS if any(m in c for c in C)]
chk('(0) no time-aware flip driver anywhere in the notebook (FLIP_TA_*, compact_infer_ta, --ta-*)', C and not dr, dr)
try:
    k2 = lit(app[6], 'K2_CFG'); sc = lit(app[4], 'S_CFG')
    sm = sc['s_manifest']
    ok_s = (sm['checkpoints'] == S and hashlib.sha256(json.dumps(sm, sort_keys=True).encode()).hexdigest() == B['s_manifest_sha256']
            and all(sm['sources'][n]['sha256'] == sm['sources'][n]['sha256_full'][:16] and len(sm['sources'][n]['sha256_full']) == 64 for n in S)
            and sc['s_ds'] == B['s_ds'] and sc['merged_manifest']['checkpoints'][-5:] == S)
    chk('(0) K2_CFG: MEMBERS == the 5 S (all AB), MIN_START_MIN %s, WALL_H 8.3; S_CFG manifest == the build (sorted-json sha256), '
        'S sha256[:16] == sha256_full' % MIN_START,
        k2['MEMBERS'] == S and k2['TTA'] == {m: 'AB' for m in S} and k2['MIN_START_MIN'] == MIN_START and k2['WALL_H'] == 8.3 and ok_s,
        {'MIN_START_MIN': k2.get('MIN_START_MIN'), 's_manifest_ok': ok_s})
    SMAN = sm
except Exception as e:
    chk('(0) K2_CFG / S_CFG literals', False, repr(e)[:200])
    SMAN = None

# ---- (1) the V224-F-S gate (nested) ----------------------------------------------------------------------------------------------
r = subprocess.run([sys.executable, V224FS_CHECK, A.out, '--wf', str(A.wf), '--build', A.build, '--sample', A.sample],
                   capture_output=True, text=True, encoding='utf-8')
print('\n'.join('    | ' + ln for ln in r.stdout.strip().splitlines()), flush=True)
ck = top_checks(r.stdout)
fails = [nm for st, nm in ck if st == 'FAIL']
unexpected = [nm for nm in fails if not nm.startswith(EXPECTED_FS_FAILS)]
chk('(1) v224fs_check.py: every check passes except exactly its exact-mode V224 sub-gate and its f / ff-only MIN_START_MIN line '
    '(replaced by (2) and (0) / (6))', len(fails) == 2 and not unexpected and sum(1 for st, _ in ck if st == 'OK') >= 20
    and all(any(nm.startswith(e) for nm in fails) for e in EXPECTED_FS_FAILS), {'fails': fails, 'oks': sum(1 for st, _ in ck if st == 'OK')})

# ---- (2) the base gate on a view ------------------------------------------------------------------------------------------------
view = tempfile.mkdtemp(prefix='v224aos_view_')
ok_view, vlast = False, ''
try:
    if entries is not None:
        k = max(i for i, e in enumerate(entries) if BD['last'] in e.get('data', ''))
        json.dump(entries[:k + 1], open(os.path.join(view, os.path.basename(logs[0])), 'w', encoding='utf-8'))
    else:
        cut = text.find('\n', text.rfind(BD['last']))
        open(os.path.join(view, os.path.basename(logs[0])), 'w', encoding='utf-8').write(text[:cut if cut > 0 else None])
    for f in ['blend_receipt.json', 'canary_receipt.json', 'submission_receipt.json', 'ours_sub.csv', 'raptor_raw.csv', 'raptor_sub.csv',
              'raptor224_sub.csv', 'raptor224_raw.csv', 'raptor224_raw_plain.csv', 'raptor224_raw_mirror.csv', 'raptor224_tta_receipt.json'] \
            + [os.path.basename(p) for p in glob.glob(P('worker*_log.txt'))]:
        if os.path.isfile(P(f)):
            shutil.copyfile(P(f), os.path.join(view, f))
    shutil.copyfile(P('submission_V224.csv'), os.path.join(view, 'submission.csv'))
    if BASE == 'flip':
        cmd = [sys.executable, V224_CHECK, view, '--flip', '--ref', A.ref_exact, '--sample', A.sample]; want = 'V224_CHECK PASS'
    else:
        cmd = [sys.executable, FLRM_CHECK, view, '--ref-exact', A.ref_exact, '--ref-flip', A.ref_flip, '--sample', A.sample]
        want = 'V224FLRM_CHECK PASS'
    r2 = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    print('\n'.join('    | ' + ln for ln in r2.stdout.strip().splitlines() if not ln.startswith('    |     | ')), flush=True)
    vlast = (r2.stdout.strip().splitlines() or [''])[-1]
    ok_view = r2.returncode == 0 and want in vlast
except Exception as e:
    print('    | view error', repr(e), flush=True)
chk('(2) base part (log up to %s, submission_V224.csv) passes %s' % (BD['last'], 'v224_check.py --flip --ref V224-exact' if BASE == 'flip'
    else 'v224flrm_check.py (the V224-FLIP-RM commit gate)'), ok_view, vlast)
shutil.rmtree(view, ignore_errors=True)

# ---- (3) flip on for every study, no time-aware driver ----------------------------------------------------------------------------
ff = line_json(text, 'FLIP_TTA_FORCED')
wa = (ff or {}).get('worker_args') or []
chk('(3) FLIP_TTA_FORCED is the V224-FLIP form: payload {worker_args} only, --flip-tta exactly once, no driver / --ta-* args',
    isinstance(ff, dict) and list(ff) == ['worker_args'] and wa.count('--flip-tta') == 1 and not any(str(a).startswith('--ta-') for a in wa)
    and 'compact_infer_ta.py' not in wa, {k: (v if k != 'worker_args' else len(v)) for k, v in (ff or {}).items()})
wl = {os.path.basename(p): open(p, encoding='utf-8', errors='replace').read() for p in sorted(glob.glob(P('worker*_log.txt')))}
hits = sorted(set(m for m in DRIVER_MARKS if m in text or any(m in t for t in wl.values()))) + (
    ['flip_ta_receipt.json'] if os.path.isfile(P('flip_ta_receipt.json')) else [])
chk('(3) no time-aware driver marker / file in the kernel log, the worker logs or the output dir', not hits, hits)
sr = jload(P('submission_receipt.json')) or {}
ws = sr.get('workers') or []
per = [p[0] for w in ws for p in (w.get('per_study') or [])]
okw = (sr.get('worker') == 'compact_infer.py' and sr.get('status') == 'PASS' and sr.get('fallback_studies') == 0 and len(ws) == 2
       and all(w.get('worker_file') == 'compact_infer.py' and (w.get('sha256') or {}).get('compact_infer.py') == CI_SHA
               and isinstance(w.get('flip_tta'), dict) and w['flip_tta'].get('perm') == FLIP_PERM and w['flip_tta'].get('mcl') == 'plain pass only'
               and 'time_aware_flip' not in w and w.get('status') == 'PASS' and w.get('studies') == w.get('scored') == len(w.get('per_study') or [])
               and not w.get('fallback') for w in ws))
chk('(3) R7 receipt: controller + both workers ran the pinned compact_infer.py (sha 8401812c) WITH the flip_tta block (M/L perm, '
    'MCL plain pass only), no time_aware_flip block, studies == scored, 0 fallback', okw,
    {'worker': sr.get('worker'), 'workers': [(w.get('worker_file'), bool(w.get('flip_tta')), w.get('studies'), w.get('scored')) for w in ws]})
chk('(3) every test study was scored by a --flip-tta worker: the workers\' per-study rows partition the %d test ids' % len(ids),
    sorted(per) == sorted(ids) and len(set(per)) == len(per), {'rows': len(per)})
okl = len(wl) == 2
for t in wl.values():
    wd = line_json_head(t, 'WORKER_DONE') or {}
    okl = okl and 'FLIP_TTA_READY' in t and '"mcl_plain_pass_only": true' in t and wd.get('studies') == wd.get('scored') and wd.get('timed_out') is False
chk('(3) every worker log: FLIP_TTA_READY + mcl_plain_pass_only, WORKER_DONE studies == scored, not timed out', okl, list(wl))
o_flip = csv_(A.ref_flip, 'ours_sub.csv').loc[ids, labels]
try:
    ours = csv_(A.out, 'ours_sub.csv').loc[ids, labels]
    d = float(np.abs(ours.values - o_flip.values).max())
    chk('(3) R7 ours_sub == the V224-FLIP commit R7 (%g): the flip is numerically on for all %d studies' % (TOL_COMPILE, len(ids)),
        d <= TOL_COMPILE, 'max diff %.3g' % d)
except Exception as e:
    ours = None
    chk('(3) R7 ours_sub == the V224-FLIP commit R7', False, repr(e)[:200])
ref_base = A.ref_flip if BASE == 'flip' else A.ref_flrm
try:
    h = csv_(A.out, 'submission_V224.csv').loc[ids, labels]
    hr = csv_(ref_base, 'submission.csv').loc[ids, labels]
    d = float(np.abs(h.values - hr.values).max())
    chk('(3) submission_V224.csv == the %s commit submission (%g)' % ('V224-FLIP' if BASE == 'flip' else 'V224-FLIP-RM', TOL_BLEND),
        d <= TOL_BLEND, 'max diff %.3g, byte-identical %s' % (d, open(P('submission_V224.csv'), 'rb').read() ==
                                                          open(os.path.join(ref_base, 'submission.csv'), 'rb').read()))
except Exception as e:
    chk('(3) submission_V224.csv == the base commit submission', False, repr(e)[:200])

# ---- (4) S members and shas -----------------------------------------------------------------------------------------------------
sm_line = next((ln for ln in text.splitlines() if '[s-merge' in ln and 'RESULT' in ln), '')
chk('(4) S-MERGE: RESULT OK, K12 25 + S 5 = 30 checkpoints from eeshsaxena/%s, sha256 verified' % B.get('s_ds'),
    'RESULT OK merged fleet dir + S: 30 checkpoints (5 S from ' in sm_line and B.get('s_ds', '?') in sm_line and 'sha256 verified' in sm_line,
    sm_line.strip()[:160])
fr = jload(P('fleet_receipt.json')) or {}
cfg = fr.get('config') or {}
chk('(4) fleet receipt: names / members_filter / K2_CFG MEMBERS == the 5 S of the build, TTA AB, models 5, n_keep [5, 5]',
    sorted(fr.get('names') or []) == sorted(S) and list(fr.get('members_filter') or []) == S and list(cfg.get('MEMBERS') or []) == S
    and (fr.get('tta') or {}).get('modes') == {m: 'AB' for m in S} and fr.get('models') == 5 and fr.get('n_keep') == [5, 5],
    {k: fr.get(k) for k in ('models', 'n_keep')})
ms16 = fr.get('manifest_sha16') or {}
okm = SMAN is not None and all(ms16.get(n) == SMAN['sources'][n]['sha256'] for n in S) and set(S) <= set(fr.get('manifest') or [])
chk('(4) fleet receipt manifest_sha16 of each S == the build\'s S manifest sha256[:16] (the S files the fleet loaded are the built ones)',
    okm, {n[:20]: (ms16.get(n), (SMAN or {}).get('sources', {}).get(n, {}).get('sha256')) for n in S if ms16.get(n) != (SMAN or {}).get('sources', {}).get(n, {}).get('sha256')})
bench = [b for b in fr.get('bench') or [] if b]
bs = [b.get(n) for b in bench for n in S]
chk('(4) S microbench <= 400 ms / member / study on every GPU (runtime review: 5 x <= 325 ms MEASURED; 1.62 s / study)',
    len(bench) == 2 and bs and all(isinstance(x, (int, float)) and x <= 400 for x in bs), {'max_ms': max(bs) if bs and all(bs) else None})


def s_mean(d):
    zs = [np.load(p, allow_pickle=False) for p in sorted(glob.glob(os.path.join(d, '_fleet_*.npz')))]
    assert zs, 'no _fleet_*.npz in %s' % d
    nm_ = [str(x) for x in zs[0]['names']]
    assert all([str(x) for x in z['names']] == nm_ for z in zs) and set(S) <= set(nm_), 'npz members in %s' % d
    Pm_ = np.concatenate([z['P'] for z in zs], axis=1).astype(np.float64)[:, :, [TG.index(c) for c in labels]]
    sid = np.concatenate([z['sids'] for z in zs]).astype(str)
    return nm_, pd.DataFrame(Pm_[[nm_.index(m) for m in S]].mean(0), index=sid, columns=labels)


if A.ref_s:
    try:
        assert os.path.realpath(A.ref_s) != os.path.realpath(A.out), '--ref-s is this output itself'
        nm_o, so = s_mean(A.out); nm_r, srf = s_mean(A.ref_s)
        assert sorted(nm_o) == sorted(S), 'this npz members != the 5 S'
        d = float(np.abs(so.loc[ids].values - srf.loc[ids].values).max())
        chk('(4) --ref-s: S group mean == the S group mean of %s (%g)' % (os.path.basename(os.path.normpath(A.ref_s)), TOL_COMPILE),
            d <= TOL_COMPILE, 'max diff %.3g' % d)
    except Exception as e:
        chk('(4) --ref-s S anchor', False, repr(e)[:200])
else:
    print('  WARN no --ref-s: the S numerics are checked for self-consistency only (fleet solo == npz mean, F, blend); pass --ref-s '
          '<an output with the same 5 S> once one exists (e.g. the other AO commit)', flush=True)

# ---- (5) END-TO-END recompute ---------------------------------------------------------------------------------------------------
try:
    raw = csv_(A.out, 'raptor_raw.csv').loc[ids, labels]
    r224 = csv_(A.out, 'raptor224_sub.csv').loc[ids, labels]
    assert ours is not None and not (raw.values == 0.5).all(axis=1).any(), 'Raptor-failed rows (all 0.5) or no R7 csv'
    H64 = np.zeros((len(ids), len(labels)))
    for j, c in enumerate(labels):
        H64[:, j] = 0.5 * (0.65 * pct(raw[c].values) + 0.35 * pct(r224[c].values)) + 0.5 * pct(ours[c].values)
    H32 = H64.astype(np.float32).astype(np.float64)
    hv = csv_(A.out, 'submission_V224.csv').loc[ids, labels].values.astype(np.float64)
    d = float(np.abs(hv - H32).max())
    chk('(5) submission_V224.csv == V224 recipe from raptor_raw / raptor224_sub / ours_sub (float32, no re-rank)', d <= TOL_BLEND, 'max diff %.3g' % d)
    nm, Fm = s_mean(A.out)
    assert sorted(nm) == sorted(S), 'npz members %r != the 5 S' % nm
    F = Fm.loc[ids].values
    final = np.stack([(1.0 - A.wf) * pct(H32[:, j]) + A.wf * pct(F[:, j]) for j in range(len(labels))], 1).astype(np.float32)
    Sb = pd.read_csv(P('submission.csv'), dtype={'StudyInstanceUID': str})
    d = float(np.abs(Sb[labels].values.astype(np.float64) - final.astype(np.float64)).max())
    chk('(5) END-TO-END: submission.csv == (1 - w) pct(V224 recipe) + w pct(mean of the 5 S npz rows)',
        Sb['StudyInstanceUID'].tolist() == ids and list(Sb.columns) == cols and d <= TOL_BLEND, 'max diff %.3g' % d)
except Exception as e:
    chk('(5) END-TO-END recompute', False, repr(e)[:200])

# ---- (6) canaries + the start guard ---------------------------------------------------------------------------------------------
bad = [m for m in ('CANARY_FIRED', 'HF_FALLBACK', 'RM_GUARD_FIRED', 'V224F_H_FROM_MEMORY', 'R224_MIRROR_FAILED', 'NOT CLEAN') if m in text]
chk('(6) no canary / fallback marker anywhere in the log', not bad, bad)
cr = jload(P('canary_receipt.json')) or {}
chk('(6) canary receipt: r224 used, fired [], flip_tta_forced True' + (', rm_on + RM guard mirror_ok' if BASE == 'flrm' else ''),
    cr.get('r224_used') is True and cr.get('fired') == [] and cr.get('flip_tta_forced') is True and cr.get('use_f') is False
    and (BASE == 'flip' or (cr.get('rm_on') is True and (cr.get('rm_guard') or {}).get('mirror_ok') is True)),
    {k: cr.get(k) for k in ('r224_used', 'fired', 'flip_tta_forced', 'rm_on')})
hf = jload(P('hf_receipt.json')) or {}
vr = jload(P('v224f_receipt.json')) or {}
chk('(6) hf_receipt: mode HF, blanked [], fired_before_hf [] (the HF-AO key), tag; v224f_receipt fired [], mode HF',
    hf.get('mode') == 'HF' and hf.get('blanked') == [] and hf.get('fired_before_hf') == [] and hf.get('tag') == B.get('tag')
    and vr.get('fired') == [] and vr.get('mode') == 'HF', {k: hf.get(k) for k in ('mode', 'blanked', 'fired_before_hf', 'tag')})
try:
    V = pd.read_csv(P('submission.csv'), dtype={'StudyInstanceUID': str})[labels].values.astype(np.float64)
    const = [labels[j] for j in range(len(labels)) if len(set(np.round(V[:, j], 7))) <= 1]
    chk('(6) no constant (blanked) column in submission.csv', not const, const)
except Exception as e:
    chk('(6) no constant column', False, repr(e)[:200])
sv = line_json(text, 'V224F_SAVED') or {}
chk('(6) start guard honoured: V224F_SAVED start_budget_h G / min_start_min / ao_base; fleet started at kernel age <= G with '
    '>= MIN_START_MIN left; K2 receipt config MIN_START_MIN == build',
    sv.get('start_budget_h') == G and sv.get('min_start_min') == MIN_START and sv.get('ao_base') == BASE and sv.get('fleet_may_start') is True
    and isinstance(fr.get('kernel_age_h'), (int, float)) and fr['kernel_age_h'] <= G and (fr.get('left_min') or 0) >= MIN_START
    and cfg.get('MIN_START_MIN') == MIN_START and cfg.get('WALL_H') == 8.3,
    {'saved': {k: sv.get(k) for k in ('start_budget_h', 'min_start_min', 'ao_base', 'elapsed_h_at_save', 'fleet_may_start')},
     'fleet': {k: fr.get(k) for k in ('kernel_age_h', 'left_min', 'stage_min')}})

# ---- runtime info (MODEL) -------------------------------------------------------------------------------------------------------
oc = line_json(text, 'OURS_STAGE_COMPLETE') or {}
info = {'r7_seconds': oc.get('seconds'), 'r7_gpu_s_per_study': (oc.get('per_study_mean') or {}).get('gpu_s'),
        'elapsed_h_at_save': sv.get('elapsed_h_at_save'), 'fleet_stage_min': fr.get('stage_min'), 'fleet_workers_s': fr.get('workers_s'),
        'hf_total_s': hf.get('total_seconds')}
print('  info runtime', json.dumps(info), flush=True)
if bench and fr.get('stage_min') is not None:
    g = max(sum(b.get(n, 0) for n in S) / 1000.0 for b in bench)
    fixed = fr['stage_min'] - max(fr.get('worker_loop_min') or [0])
    for nh in (1300, 1500):
        cen = (fixed * 60 + int(np.ceil(nh / 2)) * g) / 3600
        print(f'  info wall MODEL {nh} studies: S fleet central {cen:.2f} h, slow x1.15 x 1.3 {cen * 1.15 * 1.3:.2f} h | worst clean end = '
              f'guard {G} h + slow S fleet = {G + cen * 1.15 * 1.3:.2f} h (fleet wall 8.3 h); base: V224-FLIP MEASURED 4.57-4.65 h '
              f'submit-to-score (sim_ffs_runtime.py section D)', flush=True)
n_ok = sum(R)
print(f'V224AOS_CHECK {"PASS" if all(R) else "FAIL"} {n_ok}/{len(R)} base {BASE} tag {B.get("tag")}', flush=True)
sys.exit(0 if all(R) else 1)
