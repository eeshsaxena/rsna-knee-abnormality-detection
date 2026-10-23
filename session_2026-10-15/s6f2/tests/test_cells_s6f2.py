#!/usr/bin/env python
# test_cells_s6f2.py (10-15, Claude subagent): the three APPENDED cells of both s6f2 notebooks, executed as written (paths redirected
# to a temp tree; s6f2_harness), against fake drivers, on the REAL base commit outputs. Offline, laptop CPU, -I. No GPU, no Kaggle.
# DEC: the manylinux cp313 wheels cannot install here -> every case runs the DEC_FALLBACK branch (as s6f/tests/test_cells_local.py).
# Every case: os.environ / sys.path unchanged, s6f_receipt.json written, submission.csv in base row / column order.
#  FLRM-AO + S6fall (base = the V224-FLIP-RM v1 commit submission):
#   R1 clean (driver port on nartaa's parts) -> used, == independent recompute; run policy G 6.0 / 3.3 / 8.2
#   R2 RM_GUARD fired (Fracture 0.5; _gcr fired [r224 mirror->Fracture], _canary fired []) -> used, Fracture stays 0.5, rest == recompute;
#      CONTROL: the s6f (FLIP) cells on the same input un-blank Fracture (why the splice exists)
#   R3 V224 CANARY fired (r224 not used: _canary + _gcr fired [r224->Fracture]) -> Fracture stays 0.5
#   R4 kernel age 5.9 h (> the FLIP G 5.75, <= 6.0) -> S6F runs (the FLRM guard)   R5 age 6.1 h -> skip, ACL canary, other 11 == base
#   R6 RM_GUARD fired AND start-guard skip -> ACL and Fracture 0.5, other 10 == base   R7 hung driver -> killed -> canary
#  FLIP-AO + S6fall + S (base = the flips20 v1 commit submission, i.e. the HF output):
#   S1 clean -> used == recompute; G 6.25        S2 HF fallback (Synovitis 0.5; _v224f fired [fleet->Synovitis]) -> Synovitis stays 0.5;
#      CONTROL: the s6f cells un-blank Synovitis    S3 HF fallback + V224 CANARY (Synovitis + Fracture 0.5) -> both stay
#   S4 age 6.2 h -> runs     S5 age 6.3 h -> skip, ACL canary     S6 HF fallback + skip -> ACL and Synovitis 0.5, other 10 == base
#   S7 constant base column with no fired entry (belt and braces) -> stays 0.5, receipt base_blanked names it   S8 driver fails -> canary
#   usage: py -3.12 -I tests/test_cells_s6f2.py  -> TEST_CELLS_S6F2 PASS/FAIL n/m
import os, sys, json, time, shutil
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s6f2_harness as H

R = []
TMPB = os.path.join(HERE, '_tmp_cells_s6f2')
shutil.rmtree(TMPB, ignore_errors=True)
os.makedirs(TMPB)


def chk(name, ok, info=''):
    R.append(bool(ok))
    print('  %s %s %s' % ('OK  ' if ok else 'FAIL', name, info), flush=True)


def case(name, kind, base, **kw):
    env0, path0 = dict(os.environ), list(sys.path)
    t = H.Tree(TMPB, name)
    t0 = time.time()
    ns, text = H.run_cells(t, kind, base, **kw)
    sub = pd.read_csv(os.path.join(t.work, 'submission.csv'), dtype={'StudyInstanceUID': str})
    rec = json.load(open(os.path.join(t.work, 's6f_receipt.json')))
    common = (dict(os.environ) == env0 and sys.path == path0 and sub['StudyInstanceUID'].tolist() == base['StudyInstanceUID'].tolist()
              and list(sub.columns) == list(base.columns))
    chk('%s: os.environ / sys.path untouched, receipt written, submission in base row / column order' % name, common, rec.get('mode'))
    return t, ns, text, sub, rec, time.time() - t0


def col_text(path, col):
    return pd.read_csv(path, dtype=str)[col].tolist()


def canary_ok(name, t, sub, rec, text, base_path, also=()):
    others = [c for c in H.LAB if c != 'ACL' and c not in also]
    same = all(col_text(os.path.join(t.work, 'submission.csv'), c) == col_text(base_path, c) for c in others)
    chk('%s: S6F CANARY: mode V224FLIP_CANARY, fired [s6f->ACL], ACL == 0.5 every row, %s other columns text == the base%s' % (
        name, len(others), (', ' + '/'.join(also) + ' == 0.5 kept') if also else ''),
        rec.get('mode') == 'V224FLIP_CANARY' and rec.get('fired') == ['s6f->ACL'] and bool((sub['ACL'] == 0.5).all()) and same
        and all(bool((sub[c] == 0.5).all()) for c in also) and 'S6F_CANARY_FIRED' in text and 'S6F_BLEND_OK' not in text,
        (rec.get('mode'), str((rec.get('blend') or {}).get('why'))[:120]))


def used_ok(name, sub, rec, text, base, blank, refb):
    exp = H.recompute(base, refb, blank=blank)
    d = float(np.abs(sub[H.LAB].to_numpy(np.float64) - exp[H.LAB].to_numpy(np.float64)).max())
    bl = rec.get('blend') or {}
    chk('%s: used (mode S6F, fired [], S6F_BLEND_OK), == independent recompute with %s blanked (1e-6), base_blanked == %s' % (name, list(blank), list(blank)),
        rec.get('mode') == 'S6F' and rec.get('fired') == [] and 'S6F_BLEND_OK' in text and d <= 1e-6 and bl.get('base_blanked') == list(blank)
        and all(bool((sub[c] == 0.5).all()) for c in blank), 'max diff %.2e, base_blanked %s' % (d, bl.get('base_blanked')))
    return d


def write(df, name):
    p = os.path.join(TMPB, name)
    df.to_csv(p, index=False, lineterminator='\n')
    return p


refb = H.ref_bprobs()
chk('the driver embedded in both s6f2 notebooks == the s6f driver the fake port runs (sha a2acdd1d)', H.driver_same())
# ================================================== FLRM =========================================================================
K = 'flrm'
bR, bR_path = H.base_df(K), H.base_file(K)
print('R1 FLRM clean', flush=True)
t, ns, text, sub, rec, dt = case('r1', K, bR, base_file=bR_path, age_h=4.7, mode='port', gcr_fired=[])
used_ok('R1', sub, rec, text, bR, [], refb)
run = rec.get('run') or {}
su = rec.get('setup') or {}
chk('R1: run RAN rc 0 not killed, policy G 6.0 / budget 3.3 / deadline 8.2; setup tag v224flrms6f, base_fired [], base sha == the FLRM commit file',
    run.get('status') == 'RAN' and run.get('rc') == 0 and run.get('killed') is False and run.get('g_start_h') == 6.0 and run.get('budget_h') == 3.3
    and run.get('deadline_h') == 8.2 and su.get('tag') == 'v224flrms6f' and su.get('base_fired') == [] and su.get('base_sha256') == H.sha_file(bR_path),
    (run.get('status'), run.get('g_start_h'), su.get('base_fired')))
print('R2 RM_GUARD fired', flush=True)
bG = bR.copy(); bG['Fracture'] = np.float32(0.5)
pG = write(bG, 'flrm_rmguard.csv')
t, ns, text, sub, rec, dt = case('r2', K, bG, base_file=pG, age_h=4.7, mode='port', gcr_fired=['r224 mirror->Fracture'])
used_ok('R2', sub, rec, text, bG, ['Fracture'], refb)
chk('R2: setup base_fired == [r224 mirror->Fracture] (from _gcr), v224_canary fired []', (rec.get('setup') or {}).get('base_fired') == ['r224 mirror->Fracture']
    and ((rec.get('setup') or {}).get('v224_canary') or {}).get('fired') == [], (rec.get('setup') or {}).get('base_fired'))
t, ns, text, sub, rec, dt = case('r2_ctl', K, bG, base_file=pG, age_h=4.7, mode='port', gcr_fired=['r224 mirror->Fracture'],
                                 cell_texts=H.cells('flip'))
chk('R2 CONTROL: the s6f (FLIP) cells on the same input DROP the RM_GUARD canary (Fracture not constant): the splice is needed',
    rec.get('mode') == 'S6F' and sub['Fracture'].nunique() > 1 and (rec.get('blend') or {}).get('base_blanked') == [], sub['Fracture'].tolist())
print('R3 V224 CANARY fired (r224 not used)', flush=True)
t, ns, text, sub, rec, dt = case('r3', K, bG, base_file=pG, age_h=4.7, mode='port', fired=('r224->Fracture',), gcr_fired=['r224->Fracture'])
used_ok('R3', sub, rec, text, bG, ['Fracture'], refb)
print('R4 / R5 start guard G 6.0', flush=True)
t, ns, text, sub, rec, dt = case('r4', K, bR, base_file=bR_path, age_h=5.9, mode='port', gcr_fired=[])
chk('R4: kernel age 5.9 h (> FLIP G 5.75) -> S6F starts and is used', 'S6F_START' in text and rec.get('mode') == 'S6F', (rec.get('run') or {}).get('start_age_h'))
t, ns, text, sub, rec, dt = case('r5', K, bR, base_file=bR_path, age_h=6.1, mode='port', gcr_fired=[])
canary_ok('R5', t, sub, rec, text, bR_path)
chk('R5: S6F_SKIPPED start guard (6.1 > 6.0), driver never started', 'S6F_SKIPPED start guard' in text and 'S6F_START' not in text
    and (rec.get('run') or {}).get('status') == 'SKIPPED_GUARD')
print('R6 RM_GUARD fired + skip', flush=True)
t, ns, text, sub, rec, dt = case('r6', K, bG, base_file=pG, age_h=6.1, mode='port', gcr_fired=['r224 mirror->Fracture'])
canary_ok('R6', t, sub, rec, text, pG, also=('Fracture',))
print('R7 hung driver', flush=True)
ov = {'G_START_H, BUDGET_H, DEADLINE_H, STOP_MARGIN_S, MIN_RUN_S = 6.0, 3.3, 8.2, 240.0, 1200.0':
      'G_START_H, BUDGET_H, DEADLINE_H, STOP_MARGIN_S, MIN_RUN_S = 6.0, 0.003, 8.2, 1.0, 0.0'}
t, ns, text, sub, rec, dt = case('r7', K, bR, base_file=bR_path, age_h=4.7, mode='sleep', overrides=ov, gcr_fired=[])
canary_ok('R7', t, sub, rec, text, bR_path)
chk('R7: killed at the hard deadline (~11 s)', (rec.get('run') or {}).get('killed') is True and 'S6F_KILLED' in text and dt < 60, '%.0f s' % dt)
# ================================================== FLIP-AO + S6fall + S ======================================================
K = 'flips20'
bS, bS_path = H.base_df(K), H.base_file(K)
print('S1 composite clean', flush=True)
t, ns, text, sub, rec, dt = case('s1', K, bS, base_file=bS_path, age_h=4.9, mode='port', v224f_fired=[])
used_ok('S1', sub, rec, text, bS, [], refb)
run = rec.get('run') or {}
su = rec.get('setup') or {}
chk('S1: run RAN, policy G 6.25 / 3.3 / 8.2; setup tag v224flips6fs20, base_fired [], frozen base == the flips20 commit submission (the HF output)',
    run.get('status') == 'RAN' and run.get('g_start_h') == 6.25 and run.get('budget_h') == 3.3 and run.get('deadline_h') == 8.2
    and su.get('tag') == 'v224flips6fs20' and su.get('base_fired') == [] and su.get('base_sha256') == H.sha_file(bS_path), (run.get('g_start_h'), su.get('base_fired')))
print('S2 HF fallback (Synovitis canary)', flush=True)
H_only = pd.read_csv(os.path.join(H.KIND[K]['base_out'], 'submission_V224.csv'), dtype={'StudyInstanceUID': str})
bF = H_only.copy(); bF['Synovitis'] = np.float32(0.5)                 # what HF_FALLBACK writes: H with Synovitis 0.5
pF = write(bF, 'flips20_hf_fallback.csv')
t, ns, text, sub, rec, dt = case('s2', K, bF, base_file=pF, age_h=4.9, mode='port', v224f_fired=['fleet->Synovitis'])
used_ok('S2', sub, rec, text, bF, ['Synovitis'], refb)
t, ns, text, sub, rec, dt = case('s2_ctl', K, bF, base_file=pF, age_h=4.9, mode='port', v224f_fired=['fleet->Synovitis'], cell_texts=H.cells('flip'))
chk('S2 CONTROL: the s6f (FLIP) cells on the same input DROP the Synovitis (S) canary: the splice is needed',
    rec.get('mode') == 'S6F' and sub['Synovitis'].nunique() > 1, sub['Synovitis'].tolist())
print('S3 HF fallback + V224 CANARY', flush=True)
bFF = bF.copy(); bFF['Fracture'] = np.float32(0.5)
pFF = write(bFF, 'flips20_hf_fallback_frac.csv')
t, ns, text, sub, rec, dt = case('s3', K, bFF, base_file=pFF, age_h=4.9, mode='port', fired=('r224->Fracture',),
                                 v224f_fired=['fleet->Synovitis', 'r224->Fracture'])
used_ok('S3', sub, rec, text, bFF, ['Synovitis', 'Fracture'], refb)
print('S4 / S5 start guard G 6.25', flush=True)
t, ns, text, sub, rec, dt = case('s4', K, bS, base_file=bS_path, age_h=6.2, mode='port', v224f_fired=[])
chk('S4: kernel age 6.2 h (> 5.75, <= 6.25) -> S6F starts and is used', 'S6F_START' in text and rec.get('mode') == 'S6F')
t, ns, text, sub, rec, dt = case('s5', K, bS, base_file=bS_path, age_h=6.3, mode='port', v224f_fired=[])
canary_ok('S5', t, sub, rec, text, bS_path)
chk('S5: S6F_SKIPPED start guard (6.3 > 6.25), S part kept (other 11 columns == the HF output text)', 'S6F_SKIPPED start guard' in text
    and (rec.get('run') or {}).get('status') == 'SKIPPED_GUARD')
print('S6 HF fallback + skip: both canaries', flush=True)
t, ns, text, sub, rec, dt = case('s6', K, bF, base_file=pF, age_h=6.3, mode='port', v224f_fired=['fleet->Synovitis'])
canary_ok('S6', t, sub, rec, text, pF, also=('Synovitis',))
print('S7 constant base column, nothing fired (belt and braces)', flush=True)
bC = bS.copy(); bC['Baker\'s'] = np.float32(0.5)
pC = write(bC, 'flips20_const.csv')
t, ns, text, sub, rec, dt = case('s7', K, bC, base_file=pC, age_h=4.9, mode='port', v224f_fired=[])
used_ok('S7', sub, rec, text, bC, ["Baker's"], refb)
print('S8 driver fails', flush=True)
t, ns, text, sub, rec, dt = case('s8', K, bS, base_file=bS_path, age_h=4.9, mode='fail', v224f_fired=[])
canary_ok('S8', t, sub, rec, text, bS_path)
shutil.rmtree(TMPB, ignore_errors=True)
print('TEST_CELLS_S6F2 %s %d/%d' % ('PASS' if all(R) else 'FAIL', sum(R), len(R)), flush=True)
sys.exit(0 if all(R) else 1)
