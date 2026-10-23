#!/usr/bin/env python
# build_v224fs.py (2026-10-15 dir, written 10-14 by a Claude subagent; R7-teacher line, PLAN_TEACHER FX7 + FX8). NOT pushed.
# V224-F-S = V224-F (build_v224f.py, run here as a subprocess with --start-budget-h 5.0) with the F slot of the HF blend =
#     F = (1 - S_SHARE) * mean(K14 R members, 20) + S_SHARE * mean(S members, 5)        S_SHARE = .5 (PREREG_TEACHER sec 7)
# i.e. an explicit TWO-GROUP probability mean, not the fleet cell's equal raw mean over all 25 members (which would give S
# .2 of F, .04 of the blend, instead of the pre-registered .5 of F, .1 of the blend at w_f .2). w_f unchanged.
# S = the 5 student checkpoints (T_b, seed 1047, PREREG_TEACHER + A1) in a private dataset eeshsaxena/<--s-ds> made by the
# main session (make_s_pack.py stages the upload folder; this builder only reads its manifest).
#
# Cells (0-13 V224-exact byte-identical; 15-17, 19, 21 and the fleet body = the V224-F build byte-identical):
#   14 SAVE      V224-F SAVE; only the _V224F knob dict changes (tag v224fsNN, members 25, groups, s_share, s_ds)
#   15 ENV-SNAP / 16 FN-IDS / 17 K12-MERGE   verbatim
#   18 S-MERGE   NEW: adds the S checkpoints to /tmp/k12_fleet (symlinks) + the merged manifest (K12 merged + S); full sha256 of
#                every S file vs the build; ANY failure removes /tmp/k12_fleet -> fleet not clean -> HF fallback (Synovitis canary)
#   19 CLOCK     verbatim
#   20 FLEET     V224-F fleet cell; only its K2_CFG literal differs: MEMBERS = K14 20 + S 5 (REPLACE: S 5 only), TTA AB for each,
#                WALL_H 8.3, MIN_START_MIN = (8.3 - 5.0) x 60 = 198 (FX8: fleet start budget 5.0 h; V224-F used 5.5 h / 168)
#   21 ENV-RESTORE verbatim
#   22 HF-S      NEW (V224-F HF semantics + FX7): reads the fleet workers' npz (the P the fleet averaged), recomputes the fleet's
#                equal mean and requires it == submission_fleetsolo.csv (1e-6), then F = the two-group mean (failed studies ->
#                column median, the fleet's own rule), writes submission_fleet2g.csv, and blends (1 - W_F) pct(V224) + W_F pct(F).
#                Any problem -> HF_FALLBACK = V224 output with Synovitis 0.5 (the V224-F canary). Receipts hf_receipt.json +
#                v224f_receipt.json (+ groups, s_share, eq-mean check); markers HF_BLEND_OK / HF_COMPLETE / V224FS_F2G / V224F_DONE.
# Proofs asserted at build time: cells 0-13 == V224-exact (via the V224-F build's own asserts + equality here); verbatim cells
# equal; the fleet body (all but its first line) equal; putting back the V224-F SAVE / K2 line / HF cell and dropping S-MERGE
# re-serialises to the V224-F (5.0 h) notebook byte for byte.
#
#   usage: python build_v224fs.py --wf 0.20 --base ff --s-manifest UPLOAD_DIR/r7tb_manifest.json [--s-ds rsna-knee-cnxt-fleet-r7tb]
#                                 [--replace] [--start-budget-h 5.1 (ff) | 5.0 (f)] [--flip-budget-h 4.9] [--out DIR]
#          python build_v224fs.py --wf 0.20 --base ff --standin   offline proof only: S = the 5 k14w (DT) checkpoints relabelled
#                                                           f{F}_s1047r7tb_...; writes tests/standin_v224fs20/ WITHOUT
#                                                           kernel-metadata.json (cannot be pushed)
#   slug:  eeshsaxena/rsna-sub-v224fs{WF*100}; REPLACE (F = S alone, only after a positive V224-F-S read): rsna-sub-v224fsr{WF*100}
# FIX 10-15 (v224f/RESULT.md "Fixes 10-15"): --base is required (F3; no silent non-flip build); ff keeps the V224-FF20 time-aware
#   flip budget 4.9 h (F1; cell 7 left byte-identical, so the V224 part == V224-FF20) and defaults the fleet start budget to
#   5.1 h (MIN_START_MIN 192; RUNTIME_FFS_1015.md); --start-budget-h 5.0 stays allowed for ff, f stays at 5.0. The S dataset does not exist yet (needs the trained T_b
#   checkpoints, the student gate PASS, an OAI-id scan, make_s_pack.py, then `kaggle datasets create` by the main session).
# AO 10-15 (PREREG_10-15 Amendment 3 R2 / R3; RESULT.md "AO 10-15"): two always-on-flip bases. f / ff builds are byte-identical
#   to the pre_ao_1015 builder (their code path is unchanged; proven by rebuilding the stand-ins + the mock gates).
#   --base flip  V224-FLIP = session_2026-10-14/v224/kernel_v224flip (sha 73a9d562, == the Kaggle pull of rsna-sub-v224flip v1, the
#                scored team best 57174021): R7 flip FORCED on every study by the V224-FLIP cell (pinned compact_infer.py
#                --flip-tta), NO time-aware driver.                                                  -> tag v224flips{WF*100}
#   --base flrm  V224-FLIP-RM = session_2026-10-15/v224rm/kernel_v224flrm (sha f14f6acc, == the Kaggle pull of rsna-sub-v224flrm
#                v1, built by v224rm/build_v224rm.py --wf 0 --flip --slug rsna-sub-v224flrm): V224-FLIP + RM + RM_GUARD.
#                                                                                                    -> tag v224flrms{WF*100}
#   Every base cell is kept byte-identical and in order (revert proof: dropping the appended cells re-serialises to the base FILE).
#   Appended after the base (9 cells): the V224-F fleet cells built by build_v224f.py at the AO start guard (SAVE, ENV-SNAP,
#   FN-IDS, K12-MERGE, CLOCK, FLEET, ENV-RESTORE; ENV-SNAP / FN-IDS / K12-MERGE / CLOCK / ENV-RESTORE and the fleet body verbatim)
#   + S-MERGE (the f / ff text) + HF-AO. REPLACE semantics are implied: MEMBERS = the 5 S only, F = mean(S), s_share 1.0, w_f .20.
#   Start guard G = --start-budget-h (default 7.0 h, MIN_START_MIN 78 since fix aofix 10-15 F1; 6.0 / 6.5 allowed, all three
#   runtime-reviewed in sim_ffs_runtime.py section D; guard_paired_1015.py: a higher G never adds a canary, so 7.0 dominates):
#   the fleet starts only if the kernel is <= G h old, else "stage not started" -> HF_FALLBACK ->
#   the VISIBLE Synovitis canary (V224-F semantics). Fleet wall 8.3 h (+300 s kill) unchanged.
#   SAVE-AO = the V224-F SAVE with its header naming the base + the f / ff knob dict + 'ao_base'. HF-AO = the f / ff HF-S text with
#   ONE functional change: the Fracture canary is re-applied after the blend when ANY earlier canary fired on Fracture, i.e.
#   the V224 CANARY ('r224->Fracture' in _canary) OR, on the flrm base, RM_GUARD ('r224 mirror->Fracture' in its receipt dict
#   _gcr). The f / ff HF-S only looks at _canary, which would let the S blend un-blank a Fracture column RM_GUARD set to 0.5.
#   usage: python build_v224fs.py --wf 0.20 --base flip|flrm --start-budget-h 7.0 --s-manifest UPLOAD_DIR/r7tb_manifest.json
#          python build_v224fs.py --wf 0.20 --base flip|flrm --standin     (tests/standin_v224flips20 / standin_v224flrms20)
#   slug suggestion: eeshsaxena/rsna-sub-v224flips20 (branch b), eeshsaxena/rsna-sub-v224flrms20 (branch c)
import os, re, sys, json, copy, hashlib, argparse, subprocess, tempfile, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
BUILD_F = os.path.join(HERE, 'build_v224f.py')
FF_DIR = os.path.join(ROOT, 'session_2026-10-15', 'v224ff')
BUILD_FF = os.path.join(FF_DIR, 'build_v224ff.py')
V224_NB = os.path.join(ROOT, 'session_2026-10-14', 'v224', 'kernel_v224', 'rsna_sub_v224.ipynb')
V224_SHA = '4259ec9774167bb280b794574b73e1ba73203e8a1995e97116388011c4226877'
S_RE = re.compile(r'^f([0-4])_s1047r7tb_raw_ttr224-320x24_cv\d{4}_best\.pt$')
MODEL_SHA = 'f141aeed136dfc01'
# AO bases (10-15): the commit-tested, Kaggle-pulled always-on-flip notebooks
AO_BASES = {
    'flip': dict(name='V224-FLIP', tag='v224flip', slug='eeshsaxena/rsna-sub-v224flip',
                 nb=os.path.join(ROOT, 'session_2026-10-14', 'v224', 'kernel_v224flip', 'rsna_sub_v224flip.ipynb'),
                 sha='73a9d5620d0a637ed171ecec13bcf243bd0af67992b5b4cb8ad7f35bf72b1b1b',
                 pulled=os.path.join(ROOT, 'session_2026-10-15', 'v224rm', 'pulled_v224flip', 'rsna-sub-v224flip.ipynb'), cells=15),
    'flrm': dict(name='V224-FLIP-RM', tag='v224flrm', slug='eeshsaxena/rsna-sub-v224flrm',
                 nb=os.path.join(ROOT, 'session_2026-10-15', 'v224rm', 'kernel_v224flrm', 'rsna_sub_v224flrm.ipynb'),
                 sha='f14f6accc7c178536ff8a5d8d70a65da63ca05b2dd6d2a7ce9673100f9178ecf',
                 pulled=os.path.join(ROOT, 'session_2026-10-15', 'v224rm', 'pulled_v224flrm', 'rsna-sub-v224flrm.ipynb'), cells=17),
}
AO_GUARD_DEFAULT = 7.0                       # fix aofix 10-15 F1 (RESULT.md "AO fix 10-15"): was 6.5; section D + guard_paired_1015
AO_GUARDS = (6.0, 6.5, 7.0)                  # the start guards section D reviewed (MIN_START_MIN 138 / 108 / 78)

ap = argparse.ArgumentParser()
ap.add_argument('--wf', type=float, required=True)
ap.add_argument('--s-manifest', default='', help='the S dataset manifest (make_s_pack.py output, r7tb_manifest.json)')
ap.add_argument('--s-ds', default='rsna-knee-cnxt-fleet-r7tb', help='slug of the private S weights dataset (eeshsaxena/<slug>)')
ap.add_argument('--s-manifest-name', default='r7tb_manifest.json')
ap.add_argument('--s-share', type=float, default=0.5)
ap.add_argument('--replace', action='store_true', help='F = S alone (s_share 1, MEMBERS = S only); only after a positive read; '
                'implied by --base flip / flrm')
ap.add_argument('--start-budget-h', type=float, default=None, help='fleet start budget: f 5.0 (FX8); ff 5.1 (default) or 5.0; '
                'flip / flrm: the AO start guard, 7.0 (default) or 6.0 / 6.5')
ap.add_argument('--wall-h', type=float, default=8.3)
ap.add_argument('--standin', action='store_true')
ap.add_argument('--base', choices=['f', 'ff', 'flip', 'flrm'], required=True,
                help='f = V224-F (no flip); ff = V224-FF (time-aware R7 flip, the FF-S entry); flip = V224-FLIP (flip forced, AO); '
                     'flrm = V224-FLIP-RM (flip forced + r224 mirror, AO)')
ap.add_argument('--flip-budget-h', type=float, default=None, help='ff only: the time-aware flip budget (default 4.9 = V224-FF20, '
                'cell 7 not patched); not allowed for flip / flrm (the flip is forced, there is no budget)')
ap.add_argument('--out', default=HERE)
A = ap.parse_args()
AO = A.base in AO_BASES
if AO:
    assert A.flip_budget_h is None, '--base %s: the V224-FLIP cell forces the flip on every study; there is no flip budget' % A.base
    A.replace = True                                               # Amendment 3 R2 / R3: F = S alone behind the start guard
elif A.flip_budget_h is None:
    A.flip_budget_h = 4.9
if A.start_budget_h is None:
    A.start_budget_h = AO_GUARD_DEFAULT if AO else (5.0 if A.base == 'f' else 5.1)
if A.replace:
    A.s_share = 1.0
assert A.s_share == 1.0 if A.replace else A.s_share == 0.5, 'PREREG_TEACHER sec 7: .5 split (no tuning); REPLACE = 1.0'
if AO:
    assert A.wall_h == 8.3 and A.start_budget_h in AO_GUARDS, 'AO start guard %r h not in the runtime-reviewed %r (wall 8.3 h)' % (
        A.start_budget_h, AO_GUARDS)
    assert abs(A.wf - 0.20) < 1e-12, 'Amendment 3: w_f .20 (no tuning)'
else:
    assert A.wall_h == 8.3 and A.start_budget_h in ((5.0,) if A.base == 'f' else (5.0, 5.1)), 'start budget f 5.0 h / ff 5.0 or 5.1 h, wall 8.3 h'
assert bool(A.standin) != bool(A.s_manifest), 'give exactly one of --s-manifest / --standin'
assert A.base in ('f', 'flip', 'flrm') or (1.0 < A.flip_budget_h < A.start_budget_h), 'ff: the flip budget must end before the fleet start guard'


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def dumps(nb):
    return json.dumps(nb, ensure_ascii=False, separators=(',', ':'))


def src(cell):
    s = cell['source']
    return ''.join(s) if isinstance(s, list) else s


def code_cell(text):
    return {'cell_type': 'code', 'metadata': {}, 'source': text, 'outputs': [], 'execution_count': None}


def lit(line, name):
    return eval(line[len(name + ' = '):line.index('   # literal')], {})


# ---- 1. the base. f: V224-F built at the FX8 start budget by its own builder (all its asserts). ff (added 10-14 18:35 after
# V224-FLIP became the team best): the commit-tested V224-FF kernel built by build_v224ff.py at its defaults (asserted
# byte-identical to v224ff/kernel_v224ffNN), then the start budget patched in place: fleet start budget 5.5 -> 5.1 h (or 5.0) (SAVE
# knobs + K2_CFG MIN_START_MIN 168 -> 192 (198)). The time-aware flip budget stays V224-FF20's 4.9 h (cell 7 untouched; fix F1
# 10-15); only a non-default --flip-budget-h patches cell 7. The revert proof below restores every patched cell and must give the
# base bytes back.
assert sha(V224_NB) == V224_SHA
tmp = tempfile.mkdtemp(prefix='v224fs_base_')
WT = '%02d' % int(round(A.wf * 100))
if A.base in ('f', 'flip', 'flrm'):            # flip / flrm (AO): the V224-F build at the AO start guard supplies the fleet cells
    r = subprocess.run([sys.executable, BUILD_F, '--wf', str(A.wf), '--start-budget-h', str(A.start_budget_h), '--wall-h', str(A.wall_h),
                        '--out', tmp], capture_output=True, text=True)
    FTAG = 'v224f' + WT; OFF = 0
else:
    r = subprocess.run([sys.executable, BUILD_FF, '--wf', repr(A.wf), '--out', tmp], capture_output=True, text=True)
    FTAG = 'v224ff' + WT; OFF = 2
assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
print('base:', r.stdout.strip().splitlines()[-1][:220])
F_NB_PATH = os.path.join(tmp, 'kernel_' + FTAG, 'rsna_sub_%s.ipynb' % FTAG)
F_RAW = open(F_NB_PATH, 'rb').read()
FNB = json.loads(F_RAW.decode('utf-8')); FMETA = json.load(open(os.path.join(tmp, 'kernel_' + FTAG, 'kernel-metadata.json')))
assert dumps(FNB).encode('utf-8') == F_RAW
FC = FNB['cells']; assert len(FC) == 22 + OFF
V224 = json.load(open(V224_NB, encoding='utf-8'))
ORIG = {}                                                        # patched base cells: index -> original text (revert proof)
if A.base in ('f', 'flip', 'flrm'):
    assert [src(c) for c in FC[:14]] == [src(c) for c in V224['cells']]
else:
    committed = os.path.join(FF_DIR, 'kernel_' + FTAG, 'rsna_sub_%s.ipynb' % FTAG)
    assert os.path.isfile(committed) and open(committed, 'rb').read() == F_RAW, 'the FF base is not the commit-tested ' + FTAG
    assert [src(c) for c in FC[:7]] == [src(c) for c in V224['cells'][:7]]
    FL = src(FC[7]); OLD_B = 'FLIP_TA_BUDGET_H = float(_fta_env) if _fta_env else 4.9\n'
    assert FL.startswith('# [eeshsaxena V224-F-FLIP;') and FL.count(OLD_B) == 1
    if A.flip_budget_h != 4.9:                                     # default 4.9 = V224-FF20's own budget: cell 7 stays byte-identical
        ORIG[7] = FL
        FC[7] = dict(FC[7], source=FL.replace(OLD_B, 'FLIP_TA_BUDGET_H = float(_fta_env) if _fta_env else %r   # V224-F-S (PLAN_TEACHER FX8): '
                                              'flip budget 4.9 -> %r h with the fleet start budget 5.5 -> %r h\n' % (A.flip_budget_h, A.flip_budget_h, A.start_budget_h)))
    sv = src(FC[16]); assert sv.count("'start_budget_h': 5.5,") == 1 and sv.count("'min_start_min': 168.0,") == 1
    ORIG[16] = sv
    FC[16] = dict(FC[16], source=sv.replace("'start_budget_h': 5.5,", "'start_budget_h': %r," % A.start_budget_h)
                  .replace("'min_start_min': 168.0,", "'min_start_min': %r," % round((A.wall_h - A.start_budget_h) * 60.0, 1)))
    fl = src(FC[21]); l0 = fl[:fl.index('\n')]; c0 = lit(l0, 'K2_CFG')
    assert c0['WALL_H'] == 8.3 and c0['MIN_START_MIN'] == 168.0
    ORIG[21] = fl
    c1 = dict(c0, MIN_START_MIN=round((A.wall_h - A.start_budget_h) * 60.0, 1))
    FC[21] = dict(FC[21], source=f'K2_CFG = {c1!r}' + l0[l0.index('   # literal'):] + fl[fl.index('\n'):])
SAVE_F, ENV_SNAP, FN_IDS, K12_MERGE, CLOCK, FLEET_F, ENV_RESTORE, HF_F = (src(FC[i + OFF]) for i in range(14, 22))
assert SAVE_F.startswith('# [eeshsaxena V224-F SAVE') and CLOCK.startswith('# [eeshsaxena V224-F CLOCK')
assert K12_MERGE.startswith('K12M_CFG = {') and FLEET_F.startswith('K2_CFG = {') and HF_F.startswith('# [eeshsaxena V224-F HF BLEND')
K12M = lit(K12_MERGE.split('\n', 1)[0], 'K12M_CFG')
nl = FLEET_F.index('\n'); K2_LINE_F, FLEET_BODY = FLEET_F[:nl], FLEET_F[nl:]
cfg_f = lit(K2_LINE_F, 'K2_CFG')
MIN_START = round((A.wall_h - A.start_budget_h) * 60.0, 1)
assert cfg_f['WALL_H'] == A.wall_h and cfg_f['MIN_START_MIN'] == MIN_START in (
    (198.0, 192.0) if not AO else tuple(round((8.3 - g) * 60.0, 1) for g in AO_GUARDS)), (cfg_f['WALL_H'], cfg_f['MIN_START_MIN'])
K14 = list(cfg_f['MEMBERS']); assert len(K14) == 20 and set(cfg_f['TTA'].values()) == {'AB'}
assert "'_fleet_{g}.npz'" in FLEET_BODY.replace("f'{_K2_WORK}/_fleet_{g}.npz'", "'_fleet_{g}.npz'")
assert 'mean = P.astype(np.float64).mean(0)                                 # raw-prob mean over members' in FLEET_BODY
assert "rec.update(models=M, names=names, n_keep=keeps, hard_stopped=stops," in FLEET_BODY
assert "study_ok = np.isfinite(P).all(axis=(0, 2)) & (NV > 0)" in FLEET_BODY

# ---- 1b. AO (flip / flrm): the always-on-flip base notebook, kept byte-identical; the V224-F (G) build above only supplies the
# fleet cells (its cells 0-13 = V224-exact are NOT used). BTAG / BASE_RAW = what BUILD_V224FS.json records as the base.
BTAG, BASE_RAW = FTAG, F_RAW
if AO:
    AB = AO_BASES[A.base]
    BASE_RAW = open(AB['nb'], 'rb').read()
    assert hashlib.sha256(BASE_RAW).hexdigest() == AB['sha'], ('the %s notebook changed' % AB['name'], AB['nb'])
    PULLED_OK = os.path.isfile(AB['pulled'])
    assert not PULLED_OK or open(AB['pulled'], 'rb').read() == BASE_RAW, 'the %s notebook != its Kaggle pull (v1)' % AB['name']
    BNB = json.loads(BASE_RAW.decode('utf-8'))
    assert dumps(BNB).encode('utf-8') == BASE_RAW, 'base is not in her serialisation'
    BC = BNB['cells']; NB0 = len(BC); assert NB0 == AB['cells'], (NB0, AB['cells'])
    BMETA = json.load(open(os.path.join(os.path.dirname(AB['nb']), 'kernel-metadata.json'), encoding='utf-8'))
    BTAG = AB['tag']
    HER13 = V224['cells'][:13]                                     # V224-exact = her 13 cells + our CANARY
    hidx, j = [], 0
    for i, c in enumerate(BC):
        if j < 13 and c['cell_type'] == HER13[j]['cell_type'] and src(c) == src(HER13[j]):
            hidx.append(i); j += 1
    assert j == 13, 'the AO base does not hold her 13 cells verbatim and in order'
    FLIPC = src(BC[hidx[6] + 1])
    assert FLIPC.startswith('# [eeshsaxena V224-FLIP; the ONE change vs V224-exact] R7 anatomical flip TTA forced on.') and hidx[7] == hidx[6] + 2
    assert "WORKER_ARGS += ['--flip-tta']" in FLIPC and 'FLIP_TTA_FORCED_ON = True' in FLIPC and "print('FLIP_TTA_FORCED'" in FLIPC
    assert src(BC[hidx[12] + 1]) == src(V224['cells'][13]), 'the V224 CANARY must follow her stage 3 verbatim'
    ALLB = '\n'.join(src(c) for c in BC)
    assert 'compact_infer_ta' not in ALLB and 'FLIP_TA_BUDGET_H' not in ALLB, 'an AO base must not carry the time-aware flip driver'
    assert not any(k in ALLB for k in ('K2_CFG', 'K12M_CFG', 'S_CFG', '_V224F', 'HF BLEND')), 'an AO base must not carry fleet cells'
    if A.base == 'flrm':
        assert src(BC[hidx[9] + 1]).startswith('# [eeshsaxena V224-RM; the ONE change vs its base]') and hidx[10] == hidx[9] + 2
        assert src(BC[-1]).startswith('# [eeshsaxena V224-RM GUARD;') and NB0 == hidx[12] + 3
        assert "_gcr.setdefault('fired', []).append('r224 mirror->Fracture')" in src(BC[-1])
    else:
        assert NB0 == hidx[12] + 2
    assert sum('_gcr' in src(c) for c in BC) == (1 if A.base == 'flrm' else 0), 'only RM_GUARD may bind _gcr (HF-AO reads it)'
    NBM = len(BMETA['dataset_sources'])
    FLEET_DS = FMETA['dataset_sources'][NBM:]
    assert FMETA['dataset_sources'][:NBM] == BMETA['dataset_sources'] and FLEET_DS == [
        'eeshsaxena/rsna-knee-cnxt-fleet-k11w', 'eeshsaxena/rsna-knee-cnxt-fleet-k14w'], FMETA['dataset_sources']
    for _k in ('docker_image', 'machine_shape', 'competition_sources', 'kernel_sources', 'model_sources', 'enable_internet',
               'enable_gpu', 'language', 'kernel_type'):
        assert BMETA.get(_k) == FMETA.get(_k), _k
    assert BMETA['id'] == AB['slug']
    for t in (ENV_SNAP, FN_IDS, K12_MERGE, CLOCK, FLEET_F, ENV_RESTORE):
        assert '_gcr' not in t and '_canary' not in t, 'a fleet cell touches a name HF-AO reads'

# ---- 2. the S manifest ---------------------------------------------------------------------------------------------------------
if A.standin:
    add = K12M['add_manifest']                                         # the 5 k14w D-OAI (DT) checkpoints, relabelled
    smap = {n: re.sub(r'^f(\d)_s1043f75oai_', r'f\1_s1047r7tb_', n) for n in add['checkpoints']}
    SMAN = {'checkpoints': [smap[n] for n in add['checkpoints']],
            'sources': {smap[n]: dict(add['sources'][n], original=smap[n], standin_of=n) for n in add['checkpoints']}}
else:
    SMAN = json.load(open(A.s_manifest, encoding='utf-8'))
SN = list(SMAN['checkpoints'])
assert len(SN) == 5 and sorted(int(S_RE.match(n).group(1)) for n in SN) == [0, 1, 2, 3, 4], SN
assert not set(SN) & set(K12M['merged_manifest']['checkpoints']), 'S names collide with the K12 fleet'
for n in SN:
    s = SMAN['sources'][n]
    assert s['model_mod'] == 'fleet_model_ttr' and s['model_sha'] == MODEL_SHA and s['tta'] == 'AB', (n, s)
    assert len(s['sha256_full']) == 64 and s['sha256'] == s['sha256_full'][:16] and s['original'] == n, (n, s)
MERGED = {'checkpoints': K12M['merged_manifest']['checkpoints'] + SN,
          'sources': dict(K12M['merged_manifest']['sources'],
                          **{n: {k: v for k, v in SMAN['sources'][n].items() if k not in ('sha256_full', 'standin_of')} for n in SN})}
MEMBERS = SN if A.replace else K14 + SN
G_K14 = [] if A.replace else K14

# ---- 3. new / changed cells ------------------------------------------------------------------------------------------------------
TAG = ('v224f' if A.base == 'f' else 'v224ff') + ('sr' if A.replace else 's') + WT
if AO:
    TAG = 'v224%ss%s' % (A.base, WT)                               # v224flips20 / v224flrms20 (REPLACE implied, no 'r')
S_CFG = {'s_ds': A.s_ds, 'manifest_file': A.s_manifest_name, 's_manifest': SMAN, 'merged_manifest': MERGED}
S_MERGE = ('S_CFG = %r   # literal, written by session_2026-10-15/v224f/build_v224fs.py (S = %d student checkpoints%s)\n'
           % (S_CFG, len(SN), ', STAND-IN, never push' if A.standin else '')) + r'''# [eeshsaxena V224-F-S S-MERGE] Runs after K12-MERGE (k11w + k14w linked into /tmp/k12_fleet) and BEFORE the fleet cell.
# Adds the S checkpoints of eeshsaxena/<s_ds> to /tmp/k12_fleet (symlinks, nothing copied) and writes the merged manifest
# (K12 merged + S) the build recorded. Fail loud: the fleet dir must hold K12-MERGE's manifest (== K12M_CFG['merged_manifest']),
# the S manifest must equal the build's, every S file's full sha256 must equal it, no name may collide and the S dataset may hold
# no other file. On ANY failure /tmp/k12_fleet is REMOVED, so the fleet cell finds no fleet_worker.py, the fleet stage is NOT
# clean, and the HF cell falls back (V224 output, Synovitis canary). Writes nothing under /kaggle/working.
import os, json, hashlib, shutil, time

_SM_OUT = '/tmp/k12_fleet'
_sm_rec = {'ok': False, 'out': _SM_OUT, 's_ds': S_CFG['s_ds']}


def _sm_log(msg):
    print(f'[s-merge {time.strftime("%H:%M:%S")}] {msg}', flush=True)


def _sm_find(slug, need):
    for c in (f'/kaggle/input/datasets/eeshsaxena/{slug}', f'/kaggle/input/{slug}'):
        if os.path.isfile(os.path.join(c, need)):
            return c
    return None


def _sm_sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def _sm_merge():
    t0 = time.time()
    mp = os.path.join(_SM_OUT, 'fleet_manifest.json')
    if not (os.path.isfile(mp) and os.path.isfile(os.path.join(_SM_OUT, 'fleet_worker.py'))):
        raise RuntimeError(f'no K12 fleet dir at {_SM_OUT} (K12-MERGE failed, see its RESULT line)')
    if json.load(open(mp)) != K12M_CFG['merged_manifest']:
        raise RuntimeError(f'{mp} != the K12 merged manifest')
    sd = _sm_find(S_CFG['s_ds'], S_CFG['manifest_file'])
    if not sd:
        raise RuntimeError(f'S weights dataset eeshsaxena/{S_CFG["s_ds"]} not attached (no {S_CFG["manifest_file"]})')
    ms = json.load(open(os.path.join(sd, S_CFG['manifest_file'])))
    if ms != S_CFG['s_manifest']:
        raise RuntimeError(f'{sd}/{S_CFG["manifest_file"]} != the build (another dataset version attached?)')
    merged = S_CFG['merged_manifest']
    if merged['checkpoints'] != K12M_CFG['merged_manifest']['checkpoints'] + ms['checkpoints']:
        raise RuntimeError('merged manifest checkpoints != K12 merged + S (build error)')
    extra = sorted(set(os.listdir(sd)) - set(ms['checkpoints']) - {S_CFG['manifest_file'], 'dataset-metadata.json'})
    if extra:
        raise RuntimeError(f'unexpected files in the S weights dataset: {extra}')
    have = set(os.listdir(_SM_OUT))
    for n in ms['checkpoints']:
        if n in have:
            raise RuntimeError(f'name collision: {n} is already in {_SM_OUT}')
        s = _sm_sha(os.path.join(sd, n))
        if s != ms['sources'][n]['sha256_full']:
            raise RuntimeError(f'{n}: sha256 {s[:16]} != S manifest {ms["sources"][n]["sha256_full"][:16]}')
        if merged['sources'][n]['sha256'] != s[:16]:
            raise RuntimeError(f'{n}: merged manifest sha256[:16] != the file')
    tmp, old = _SM_OUT + '.s.tmp', _SM_OUT + '.k12only'
    for d in (tmp, old):
        if os.path.lexists(d):
            shutil.rmtree(d)
    os.makedirs(tmp)
    for f in sorted(have):
        if f == 'fleet_manifest.json':
            continue
        os.symlink(os.path.realpath(os.path.join(_SM_OUT, f)), os.path.join(tmp, f))   # the same targets K12-MERGE linked
    for n in ms['checkpoints']:
        os.symlink(os.path.join(sd, n), os.path.join(tmp, n))
    with open(os.path.join(tmp, 'fleet_manifest.json'), 'w') as fh:
        json.dump(merged, fh, indent=1)
    os.rename(_SM_OUT, old)
    os.rename(tmp, _SM_OUT)
    shutil.rmtree(old)                                          # symlinks only: rmtree unlinks them, the targets stay
    _sm_rec.update(ok=True, s_dir=sd, n_s=len(ms['checkpoints']), n_total=len(merged['checkpoints']), s=round(time.time() - t0, 1))


try:
    _sm_merge()
    _sm_log(f'RESULT OK merged fleet dir + S: {_sm_rec["n_total"]} checkpoints ({_sm_rec["n_s"]} S from {_sm_rec["s_dir"]}, '
            f'sha256 verified, {_sm_rec["s"]} s)')
except Exception as _e:
    for _d in (_SM_OUT, _SM_OUT + '.s.tmp', _SM_OUT + '.k12only'):
        if os.path.lexists(_d):
            shutil.rmtree(_d, ignore_errors=True)
    _sm_rec['error'] = repr(_e)
    _sm_log(f'RESULT FAIL {_e!r} -> {_SM_OUT} removed: the fleet stage will not be clean (HF fallback, Synovitis canary)')
'''

cfg_s = dict(cfg_f, MEMBERS=list(MEMBERS), TTA={m: 'AB' for m in MEMBERS})
assert [k for k in cfg_f if cfg_f[k] != cfg_s[k]] == ['MEMBERS', 'TTA'] and list(cfg_s) == list(cfg_f)
K2_LINE_S = (f'K2_CFG = {cfg_s!r}   # literal, written by session_2026-10-15/v224f/build_v224fs.py from the V224-F (start budget '
             f'{A.start_budget_h} h) fleet cell: only MEMBERS / TTA differ ({len(G_K14)} K14 R + {len(SN)} S, all AB); WALL_H '
             f'{A.wall_h}, MIN_START_MIN {MIN_START}: the fleet starts only if the kernel is <= {A.start_budget_h} h old')
assert lit(K2_LINE_S, 'K2_CFG') == cfg_s

KN_OLD = "'members': 20, 'control': False}"
STAG = 'v224f' + WT                                              # the SAVE cell's own tag (V224-F's, also inside V224-FF)
assert SAVE_F.count(KN_OLD) == 1 and SAVE_F.count("'tag': '%s'" % STAG) == 1
SAVE_S = (SAVE_F.replace('# [eeshsaxena V224-F SAVE;', '# [eeshsaxena V224-F-S SAVE (V224-F SAVE, knob dict only);', 1)
          .replace("'tag': '%s'" % STAG, "'tag': '%s'" % TAG, 1)
          .replace(KN_OLD, "'members': %d, 'control': False,\n          'groups': {'K14': %d, 'S': %d}, 's_share': %r, 's_ds': %r}"
                   % (len(MEMBERS), len(G_K14), len(SN), float(A.s_share), A.s_ds), 1))

HF_S = r'''# [eeshsaxena V224-F-S HF BLEND; V224-F HF cell semantics + the two-group F of PREREG_TEACHER sec 7 (PLAN_TEACHER FX7)]
# F = (1 - S_SHARE) * mean(K14 R members) + S_SHARE * mean(S members): raw-prob means per study and label from the fleet workers'
# npz (_fleet_<g>.npz in /kaggle/working = the P the fleet stage averaged). The fleet's equal mean over all members is recomputed
# from the same npz and must equal submission_fleetsolo.csv (1e-6), else the fleet counts as not used. Failed studies (any
# non-finite P or no views) -> column median, the fleet stage's own rule. Written to submission_fleet2g.csv. Then exactly the
# V224-F HF semantics: submission.csv = per-label average-tie pct rank (1 - W_F) pct(V224) + W_F pct(F), float32, not re-ranked;
# a column the V224 canary blanked (r224 not used -> Fracture) is set back to 0.5 after the blend; ANY fleet problem ->
# HF_FALLBACK = the V224 output with Synovitis = 0.5 for every study (the fleet canary).
import os, json, time, hashlib, shutil
import numpy as np, pandas as pd
W_F = __WF__
HF_TAG = '__TAG__'
S_SHARE = __SS__
G_K14 = __GK__
G_S = __GS__
_W = '/kaggle/working'
_hf = {'tag': HF_TAG, 'w_f': W_F, 'members_expected': len(G_K14) + len(G_S), 's_share': S_SHARE,
       'groups': {'K14': len(G_K14), 'S': len(G_S)}}
_hf_lab = list(labels)
_hf_cols = ['StudyInstanceUID'] + _hf_lab
assert 'Synovitis' in _hf_lab and 'Fracture' in _hf_lab
_hf_pct = lambda v: pd.Series(np.asarray(v, np.float64)).rank(method='average', pct=True).values
_hf_blank = ['Fracture'] if 'r224->Fracture' in (globals().get('_canary') or {}).get('fired', []) else []
# ---- H = the V224 output -------------------------------------------------------------------------------------------------
_H = None
try:
    if _V224F.get('save_error'):
        raise RuntimeError('save failed: ' + _V224F['save_error'])
    if hashlib.sha256(open(f'{_W}/submission_V224.csv', 'rb').read()).hexdigest() != _V224F['sha256']:
        raise RuntimeError('submission_V224.csv changed since the save')
    _H = pd.read_csv(f'{_W}/submission_V224.csv', dtype={'StudyInstanceUID': str})
    _hf['h_source'] = 'submission_V224.csv'
except Exception as _e:
    _H = sub.copy()
    _hf['h_source'] = 'in-memory sub (%r)' % (_e,)
    print('V224F_H_FROM_MEMORY', repr(_e), flush=True)
assert _H.columns.tolist() == _hf_cols and _H['StudyInstanceUID'].tolist() == test_ids
assert np.isfinite(_H[_hf_lab].values.astype(np.float64)).all()
_H[_hf_lab] = _H[_hf_lab].astype(np.float32)
_hf['h_sha256'] = _V224F.get('sha256')
# ---- F (two groups) and the blend ----------------------------------------------------------------------------------------
try:
    _rec = globals().get('_k2_receipt')
    if not (isinstance(_rec, dict) and _rec.get('clean') is True):
        raise RuntimeError('fleet stage not clean: %s' % ((_rec or {}).get('why') if isinstance(_rec, dict) else 'no receipt'))
    if _rec.get('models') != len(G_K14) + len(G_S):
        raise RuntimeError('fleet ran %r members, expected %d' % (_rec.get('models'), len(G_K14) + len(G_S)))
    if sorted(_rec.get('names') or []) != sorted(G_K14 + G_S):
        raise RuntimeError('fleet member names != K14 + S of the build')
    if not os.path.isfile(f'{_W}/submission_fleetsolo.csv'):
        raise RuntimeError('no submission_fleetsolo.csv')
    _tg = list(_K2_TARGETS)
    if sorted(_tg) != sorted(_hf_lab):
        raise RuntimeError('fleet targets != the submission labels')
    _parts = [np.load(f'{_W}/_fleet_{_g}.npz', allow_pickle=False) for _g in range(len(_rec.get('n_keep') or []))]
    if not _parts:
        raise RuntimeError('no fleet worker npz')
    _names = [str(x) for x in _parts[0]['names']]
    if any([str(x) for x in _z['names']] != _names for _z in _parts[1:]) or sorted(_names) != sorted(G_K14 + G_S):
        raise RuntimeError('worker npz member names differ from each other or from K14 + S')
    _P = np.concatenate([_z['P'] for _z in _parts], axis=1).astype(np.float64)[:, :, [_tg.index(c) for c in _hf_lab]]
    _NV = np.concatenate([_z['NV'] for _z in _parts])
    _sids = np.concatenate([_z['sids'] for _z in _parts]).astype(str)
    if _P.shape != (len(_names), len(test_ids), len(_hf_lab)) or sorted(_sids.tolist()) != sorted(test_ids) or len(set(_sids)) != len(_sids):
        raise RuntimeError('worker npz shape / ids %r differ from the V224 output' % (_P.shape,))
    _ok = np.isfinite(_P).all(axis=(0, 2)) & (_NV > 0)


    def _hf_fill(m):
        m = np.array(m, np.float64)
        m[~_ok] = np.nan
        med = np.nanmedian(m, axis=0)
        return np.where(np.isfinite(m), m, med[None])


    _Fs = pd.read_csv(f'{_W}/submission_fleetsolo.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    _d_eq = float(np.abs(_hf_fill(_P.mean(0)) - _Fs.loc[_sids, _hf_lab].to_numpy(np.float64)).max())
    if not _d_eq <= 1e-6:
        raise RuntimeError('the equal member mean of the npz != submission_fleetsolo.csv (max diff %.3g)' % _d_eq)
    _iS = [_names.index(m) for m in G_S]
    _f2 = _P[_iS].mean(0)
    if G_K14:
        _f2 = (1.0 - S_SHARE) * _P[[_names.index(m) for m in G_K14]].mean(0) + S_SHARE * _f2
    _F = pd.DataFrame(_hf_fill(_f2), index=_sids, columns=_hf_lab).loc[test_ids]
    if not (np.isfinite(_F.values).all() and (_F.values >= 0).all() and (_F.values <= 1).all()):
        raise RuntimeError('two-group F not finite / outside [0, 1]')
    _F2 = _F.reset_index().rename(columns={'index': 'StudyInstanceUID'})
    _F2.columns = _hf_cols
    _tmp = f'{_W}/submission_fleet2g.csv.tmp'; _F2.to_csv(_tmp, index=False); os.replace(_tmp, f'{_W}/submission_fleet2g.csv')
    _hf['f2g'] = {'eq_mean_vs_fleetsolo_max': _d_eq, 'n_failed_studies': int((~_ok).sum()),
                  'max_abs_f2g_minus_eq': float(np.abs(_F.values - _Fs.loc[test_ids, _hf_lab].to_numpy(np.float64)).max()),
                  'sha256': hashlib.sha256(open(f'{_W}/submission_fleet2g.csv', 'rb').read()).hexdigest()}
    print('V224FS_F2G', json.dumps(_hf['f2g']), flush=True)
    _out = _H.copy()
    _rho = {}
    for _c in _hf_lab:
        _out[_c] = ((1.0 - W_F) * _hf_pct(_H[_c].values) + W_F * _hf_pct(_F[_c].values)).astype(np.float32)
        if len(_H) >= 3:
            _rho[_c] = round(float(pd.Series(_H[_c].values).corr(pd.Series(_F[_c].values), method='spearman')), 4)
    for _c in _hf_blank:
        _out[_c] = np.float32(0.5)                           # keep the V224 canary visible through the blend
    _v = _out[_hf_lab].values
    assert _out['StudyInstanceUID'].tolist() == test_ids and np.isfinite(_v).all() and ((_v >= 0) & (_v <= 1)).all()
    if _hf['h_source'] == 'submission_V224.csv':           # submission_H.csv = the H this blend used
        shutil.copyfile(f'{_W}/submission_V224.csv', f'{_W}/submission_H.csv')
    else:
        _H.to_csv(f'{_W}/submission_H.csv', index=False)
    _tmp = f'{_W}/submission.csv.tmp'; _out.to_csv(_tmp, index=False); os.replace(_tmp, f'{_W}/submission.csv')
    _hf.update(mode='HF', rows=len(_out), spearman_H_vs_F=_rho, blanked=list(_hf_blank),
               fleet={k: _rec.get(k) for k in ('models', 'n_keep', 'hard_stopped', 'stage_min', 'workers_s', 'kernel_age_h', 'coverage')})
    print('HF_BLEND_OK', json.dumps(_hf), flush=True)
except Exception as _e:
    _hf_blank = sorted(set(_hf_blank) | {'Synovitis'})
    _out = _H.copy()
    for _c in _hf_blank:
        _out[_c] = np.float32(0.5)
    _tmp = f'{_W}/submission.csv.tmp'; _out.to_csv(_tmp, index=False); os.replace(_tmp, f'{_W}/submission.csv')
    _hf.update(mode='H_only', why=repr(_e), rows=len(_out), blanked=_hf_blank)
    print('HF_FALLBACK', json.dumps(_hf), flush=True)
    print('CANARY_FIRED fleet not used: Synovitis set to 0.5 for all %d rows (%r)' % (len(_out), _e), flush=True)
_hf['submission_sha256'] = hashlib.sha256(open(f'{_W}/submission.csv', 'rb').read()).hexdigest()
_hf['total_seconds'] = round(time.time() - NB_T0, 1)
json.dump(_hf, open(f'{_W}/hf_receipt.json', 'w'), indent=1)
print('HF_COMPLETE', json.dumps(_hf), flush=True)
_v224f = {'use_f': True, 'w_f': W_F, 'tag': HF_TAG, 'mode': _hf['mode'], 'blanked': _hf['blanked'], 's_share': S_SHARE,
          'groups': _hf['groups'], 'fired': (['fleet->Synovitis'] if _hf['mode'] != 'HF' else []) + (['r224->Fracture'] if 'Fracture' in _hf['blanked'] else []),
          'r224_used': bool((globals().get('_canary') or {}).get('r224_used')), 'save': _V224F, 'total_seconds': _hf['total_seconds']}
json.dump(_v224f, open(f'{_W}/v224f_receipt.json', 'w'), indent=1)
print('V224F_DONE', json.dumps(_v224f), flush=True)
'''.replace('__WF__', repr(float(A.wf))).replace('__TAG__', TAG).replace('__SS__', repr(float(A.s_share))) \
   .replace('__GK__', repr(G_K14)).replace('__GS__', repr(SN))

if AO:                                                            # SAVE-AO / HF-AO = the f / ff texts + the AO edits (reverted below)
    SAVE_S_FF, HF_S_FF = SAVE_S, HF_S
    SAVE_HDR_OLD = ('# [eeshsaxena V224-F-S SAVE (V224-F SAVE, knob dict only); not in the public notebook] Everything above is '
                    'V224-exact (her 13 cells verbatim + our canary). Keep\n')
    SAVE_HDR_NEW = ('# [eeshsaxena V224-AO-S SAVE (V224-F SAVE: header + knob dict only); not in the public notebook] Everything above is\n'
                    '# %s (%s v1, byte-identical: R7 flip forced on every study, no time-aware driver). Keep\n' % (AB['name'], AB['slug']))
    KN_S = "'s_ds': %r}" % A.s_ds
    assert SAVE_S.startswith(SAVE_HDR_OLD) and SAVE_S.count(SAVE_HDR_OLD) == 1 and SAVE_S.count(KN_S) == 1
    SAVE_S = SAVE_S.replace(SAVE_HDR_OLD, SAVE_HDR_NEW, 1).replace(KN_S, "'s_ds': %r, 'ao_base': %r}" % (A.s_ds, A.base), 1)
    HF_HDR_OLD = '# [eeshsaxena V224-F-S HF BLEND; V224-F HF cell semantics + the two-group F of PREREG_TEACHER sec 7 (PLAN_TEACHER FX7)]\n'
    HF_HDR_NEW = ('# [eeshsaxena V224-AO-S HF BLEND; the V224-F-S HF cell (REPLACE: F = S alone) + the AO Fracture canary rule, '
                  'PREREG_10-15 A3]\n')
    BLANK_OLD = "_hf_blank = ['Fracture'] if 'r224->Fracture' in (globals().get('_canary') or {}).get('fired', []) else []\n"
    BLANK_NEW = ("# AO: Fracture goes back to 0.5 after the blend when ANY earlier canary fired on Fracture: the V224 CANARY (_canary: r224\n"
                 "# not used) or, on the V224-FLIP-RM base, RM_GUARD (_gcr: r224 used without a valid mirror receipt). RM_GUARD writes\n"
                 "# Fracture 0.5 into submission.csv (= H here) but not into _canary, so the f / ff rule would let the S blend un-blank it.\n"
                 "_hf_fired = list((globals().get('_canary') or {}).get('fired', [])) + list((globals().get('_gcr') or {}).get('fired', []))\n"
                 "_hf['fired_before_hf'] = sorted(set(str(_x) for _x in _hf_fired))\n"
                 "_hf_blank = ['Fracture'] if any(str(_x).endswith('->Fracture') for _x in _hf_fired) else []\n")
    assert HF_S.startswith(HF_HDR_OLD) and HF_S.count(HF_HDR_OLD) == 1 and HF_S.count(BLANK_OLD) == 1
    HF_S = HF_S.replace(HF_HDR_OLD, HF_HDR_NEW, 1).replace(BLANK_OLD, BLANK_NEW, 1)
    assert SAVE_S.replace(SAVE_HDR_NEW, SAVE_HDR_OLD, 1).replace("'s_ds': %r, 'ao_base': %r}" % (A.s_ds, A.base), KN_S, 1) == SAVE_S_FF
    assert HF_S.replace(HF_HDR_NEW, HF_HDR_OLD, 1).replace(BLANK_NEW, BLANK_OLD, 1) == HF_S_FF
    assert G_K14 == [] and MEMBERS == SN and A.s_share == 1.0 and A.replace

for t in (SAVE_S, S_MERGE, K2_LINE_S + FLEET_BODY, HF_S):
    compile(t, '<cell>', 'exec')

# ---- 4. assemble + proofs ------------------------------------------------------------------------------------------------------
if not AO:
  nb = copy.deepcopy(FNB)
  O = OFF
  nb['cells'] = ([copy.deepcopy(c) for c in FC[:14 + O]] + [code_cell(SAVE_S)] + [copy.deepcopy(c) for c in FC[15 + O:18 + O]]
                 + [code_cell(S_MERGE), copy.deepcopy(FC[18 + O]), code_cell(K2_LINE_S + FLEET_BODY), copy.deepcopy(FC[20 + O]), code_cell(HF_S)])
  assert len(nb['cells']) == 23 + O
  C = [src(c) for c in nb['cells']]
  assert C[15 + O:18 + O] == [ENV_SNAP, FN_IDS, K12_MERGE] and C[19 + O] == CLOCK and C[21 + O] == ENV_RESTORE
  assert C[20 + O].split('\n', 1)[1] == FLEET_F.split('\n', 1)[1]
  back = copy.deepcopy(nb)
  back['cells'] = back['cells'][:18 + O] + back['cells'][19 + O:]
  back['cells'][14 + O]['source'] = SAVE_F; back['cells'][19 + O]['source'] = FLEET_F; back['cells'][21 + O]['source'] = HF_F
  for i, t in ORIG.items():                                  # ff: undo the FX8 budget patches too
      back['cells'][i]['source'] = t
  assert dumps(back).encode('utf-8') == F_RAW, 'V224-F-S minus its changes is not the base notebook (%s)' % FTAG
else:                                                            # AO: base cells byte-identical + 9 appended cells
  APP = [code_cell(SAVE_S), copy.deepcopy(FC[15]), copy.deepcopy(FC[16]), copy.deepcopy(FC[17]), code_cell(S_MERGE),
         copy.deepcopy(FC[18]), code_cell(K2_LINE_S + FLEET_BODY), copy.deepcopy(FC[20]), code_cell(HF_S)]
  nb = copy.deepcopy(BNB)
  nb['cells'] = [copy.deepcopy(c) for c in BC] + APP
  assert len(nb['cells']) == NB0 + 9
  C = [src(c) for c in nb['cells']]
  assert C[NB0 + 1:NB0 + 4] == [ENV_SNAP, FN_IDS, K12_MERGE] and C[NB0 + 5] == CLOCK and C[NB0 + 7] == ENV_RESTORE
  assert C[NB0 + 6].split('\n', 1)[1] == FLEET_F.split('\n', 1)[1]
  back = copy.deepcopy(nb)                                       # proof 1: minus the appended cells == the base FILE
  back['cells'] = back['cells'][:NB0]
  assert dumps(back).encode('utf-8') == BASE_RAW, 'AO: the base cells are not byte-identical to %s' % AB['name']
  app = copy.deepcopy(nb['cells'][NB0:])                         # proof 2: SAVE / K2 line / HF reverted, S-MERGE dropped ==
  del app[4]                                                     # the V224-F (G) fleet cells 14-21, byte for byte
  app[0]['source'] = SAVE_F; app[5]['source'] = FLEET_F; app[7]['source'] = HF_F
  assert dumps({'c': app}) == dumps({'c': FC[14:22]}), 'AO: the appended cells are not the V224-F (G) fleet cells + the S changes'
name = 'rsna-sub-' + TAG
code_file = 'rsna_sub_%s.ipynb' % TAG
out_dir = os.path.join(HERE, 'tests', 'standin_' + TAG) if A.standin else os.path.join(A.out, 'kernel_' + TAG)
if os.path.isdir(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir)
nb_path = os.path.join(out_dir, code_file)
with open(nb_path, 'w', encoding='utf-8', newline='') as f:
    f.write(dumps(nb))
meta = copy.deepcopy(FMETA if not AO else BMETA)
meta.update(id='eeshsaxena/' + name, title=name.replace('-', ' '), code_file=code_file, is_private=True, enable_gpu=True,
            enable_tpu=False, enable_internet=False)
meta['dataset_sources'] = FMETA['dataset_sources'] + ['eeshsaxena/' + A.s_ds]
if AO:
    meta['dataset_sources'] = BMETA['dataset_sources'] + FLEET_DS + ['eeshsaxena/' + A.s_ds]
    assert meta['dataset_sources'] == FMETA['dataset_sources'] + ['eeshsaxena/' + A.s_ds]
if A.standin:
    open(os.path.join(out_dir, 'DO_NOT_PUSH.txt'), 'w').write('stand-in S members (k14w DT relabelled): offline proof only\n')
    json.dump(meta, open(os.path.join(out_dir, 'kernel-metadata.STANDIN_NOT_PUSHABLE.json'), 'w'), indent=1)
else:
    json.dump(meta, open(os.path.join(out_dir, 'kernel-metadata.json'), 'w', encoding='utf-8'), indent=1)
rec = dict(tag=TAG, notebook_sha256=sha(nb_path), base=A.base, base_tag=BTAG, base_sha256=hashlib.sha256(BASE_RAW).hexdigest(),
           flip_budget_h=(A.flip_budget_h if A.base == 'ff' else None), patched_base_cells=sorted(ORIG), w_f=A.wf, s_share=A.s_share,
           replace=A.replace, start_budget_h=A.start_budget_h, wall_h=A.wall_h, min_start_min=MIN_START, members=len(MEMBERS),
           k14=G_K14, s=SN, s_ds=A.s_ds, s_manifest=(A.s_manifest or 'STANDIN'), s_manifest_sha256=hashlib.sha256(
               json.dumps(SMAN, sort_keys=True).encode()).hexdigest(), standin=A.standin, datasets=meta['dataset_sources'])
if AO:                                                            # extra keys only for AO (f / ff json unchanged)
    rec.update(base_name=AB['name'], base_slug=AB['slug'], base_notebook=os.path.relpath(AB['nb'], ROOT).replace(os.sep, '/'),
               base_pulled_identical=PULLED_OK, base_cells=NB0, appended_cells=list(range(NB0, NB0 + 9)),
               fleet_cells_from='%s = build_v224f.py --wf %r --start-budget-h %r --wall-h %r' % (FTAG, A.wf, A.start_budget_h, A.wall_h),
               fleet_cells_sha256=hashlib.sha256(F_RAW).hexdigest(),
               flip='forced on every study (V224-FLIP cell: pinned compact_infer.py --flip-tta; no time-aware driver)',
               ao_guard_h=A.start_budget_h, hf_cell='HF-AO (f / ff HF-S + Fracture re-blank on _canary or RM_GUARD _gcr)')
json.dump(rec, open(os.path.join(out_dir, 'BUILD_V224FS.json'), 'w'), indent=1)
if AO:
    with open(os.path.join(out_dir, 'celldiff_%s.txt' % TAG), 'w', encoding='utf-8', newline='\n') as f:
        f.write(f'# {name}: base {AB["name"]} ({AB["slug"]} v1, sha {AB["sha"][:16]}, == Kaggle pull: {PULLED_OK}); its {NB0} cells '
                f'byte-identical (cells 0-{NB0 - 1}); appended cells {NB0}-{NB0 + 8} = the V224-F fleet cells of {FTAG} at start budget '
                f'{A.start_budget_h} h (sha {hashlib.sha256(F_RAW).hexdigest()[:16]}) with SAVE-AO, S-MERGE (new), K2 line, HF-AO; '
                f'ENV-SNAP / FN-IDS / K12-MERGE / CLOCK / ENV-RESTORE / fleet body verbatim\n')
        f.write(f'\n#### cell {NB0} SAVE-AO (V224-F SAVE: header + knob dict)\n' + SAVE_S + f'\n#### cell {NB0 + 4} S-MERGE (NEW; '
                f'S_CFG literal elided; = the f / ff text)\n' + S_MERGE.split('\n', 1)[1] + f'\n#### cell {NB0 + 6} FLEET line 1 (K2_CFG '
                f'literal; the rest verbatim)\n' + K2_LINE_S.replace(repr(cfg_s['MEMBERS']), '[%d members]' % len(MEMBERS))
                .replace(repr(cfg_s['TTA']), '{...AB...}') + f'\n\n#### cell {NB0 + 8} HF-AO (NEW)\n' + HF_S)
else:
    with open(os.path.join(out_dir, 'celldiff_%s.txt' % TAG), 'w', encoding='utf-8', newline='\n') as f:
        f.write(f'# {name}: base {FTAG} (sha {hashlib.sha256(F_RAW).hexdigest()[:16]}); base cells verbatim except the 4 S changes'
                f'{" and the FX8 budget patches of cells " + str(sorted(ORIG)) + " (flip budget " + str(A.flip_budget_h) + " h, start budget " + str(A.start_budget_h) + " h)" if ORIG else ""}; '
                f'cell indices below are for the f base (ff: +2).\n')
        for i in sorted(ORIG):
            f.write(f'\n#### base cell {i} PATCHED (FX8)\n' + src(nb['cells'][i])[:600] + '\n...\n')
        f.write('\n#### cell 14 SAVE (knob dict only)\n' + SAVE_S + '\n#### cell 18 S-MERGE (NEW; S_CFG literal elided)\n'
                + S_MERGE.split('\n', 1)[1] + '\n#### cell 20 FLEET line 1 (K2_CFG literal; the rest verbatim)\n'
                + K2_LINE_S.replace(repr(cfg_s['MEMBERS']), '[%d members]' % len(MEMBERS)).replace(repr(cfg_s['TTA']), '{...AB...}')
                + '\n\n#### cell 22 HF-S (NEW)\n' + HF_S)
shutil.rmtree(tmp, ignore_errors=True)
print('BUILT', TAG, nb_path, 'sha256', rec['notebook_sha256'], 'cells', len(nb['cells']), 'members', len(MEMBERS),
      '(K14 %d + S %d)' % (len(G_K14), len(SN)), 's_share', A.s_share, 'MIN_START_MIN', MIN_START, 'standin', A.standin,
      'datasets', meta['dataset_sources'])
