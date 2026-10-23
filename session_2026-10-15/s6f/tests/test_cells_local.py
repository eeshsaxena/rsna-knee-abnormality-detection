#!/usr/bin/env python
# test_cells_local.py (10-15, Claude subagent): the three APPENDED cells of the built notebook, executed as written (paths
# redirected to a temp tree; s6f_harness), against fake drivers. Offline, laptop CPU, -I. No GPU, no Kaggle.
# On this Windows laptop the DEC wheels cannot install (manylinux cp313; no dataset here) -> every case runs the DEC_FALLBACK
# branch (S6F still runs, receipt dec_ok False); the DEC success branch is the reviewed V224-J logic proven on Kaggle CPU.
# Cases (each: the notebook's os.environ / sys.path unchanged; s6f_receipt.json written; submission.csv in base row order):
#   C1 clean (real driver port on nartaa's parts, the real V224-FLIP commit output as base) -> S6F used, == independent recompute
#   C2 permuted B_probs columns (label order by name) -> identical submission to C1
#   C3 start guard (kernel age 6.0 h > G 5.75) -> S6F_SKIPPED, canary: ACL == 0.5, the other 11 columns' csv text == base's
#   C4 driver never ends (budget forced to 11 s) -> killed at the hard deadline, canary
#   C5 driver fails (exit 3, ok False) -> canary
#   C6 B_probs lacks a label -> canary            C7 B_probs changed after the driver receipt (sha) -> canary
#   C8 N 100: 2 unscored (98 %) -> used, unscored rows == their V224-FLIP rank; 3 unscored (97 %) -> canary
#   C9 N 200 sanity floor: S6 unrelated noise -> canary; S6 = base + small noise -> used
#   C10 V224 canary already fired (Fracture 0.5) -> blend used, Fracture stays 0.5
#   C11 setup fails (no submission.csv when it runs) -> run skipped, blend writes the canary from the in-memory `sub`
#   usage: py -3.12 -I tests/test_cells_local.py  -> TEST_CELLS PASS/FAIL n/m
import os, sys, io, json, time, shutil
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s6f_harness as H

R = []
TMPB = os.path.join(HERE, '_tmp_cells')
shutil.rmtree(TMPB, ignore_errors=True)
os.makedirs(TMPB)


def chk(name, ok, info=''):
    R.append(bool(ok))
    print('  %s %s %s' % ('OK  ' if ok else 'FAIL', name, info), flush=True)


def case(name, base, **kw):
    env0, path0 = dict(os.environ), list(sys.path)
    t = H.Tree(TMPB, name)
    t0 = time.time()
    ns, text = H.run_cells(t, base, **kw)
    sub = pd.read_csv(os.path.join(t.work, 'submission.csv'), dtype={'StudyInstanceUID': str})
    rec = json.load(open(os.path.join(t.work, 's6f_receipt.json')))
    common = (dict(os.environ) == env0 and sys.path == path0 and sub['StudyInstanceUID'].tolist() == base['StudyInstanceUID'].tolist()
              and list(sub.columns) == list(base.columns))
    chk('%s: os.environ / sys.path untouched, receipt written, submission in base row / column order' % name, common, rec.get('mode'))
    return t, ns, text, sub, rec, time.time() - t0


def col_text(path, col):
    df = pd.read_csv(path, dtype=str)
    return df[col].tolist()


def canary_ok(name, t, sub, rec, text, base_path):
    others = [c for c in H.LAB if c != 'ACL']
    same = all(col_text(os.path.join(t.work, 'submission.csv'), c) == col_text(base_path, c) for c in others)
    chk('%s: CANARY: mode V224FLIP_CANARY, fired [s6f->ACL], ACL == 0.5 every row, other 11 columns text == V224-FLIP' % name,
        rec.get('mode') == 'V224FLIP_CANARY' and rec.get('fired') == ['s6f->ACL'] and bool((sub['ACL'] == 0.5).all()) and same
        and 'S6F_CANARY_FIRED' in text and 'S6F_BLEND_OK' not in text, (rec.get('mode'), (rec.get('blend') or {}).get('why')))


base3 = H.flip_base()
base3_path = os.path.join(H.FLIP_OUT, 'submission.csv')
refb = H.ref_bprobs()
# ---- C1
print('C1 clean (driver port on nartaa parts)', flush=True)
t, ns, text, sub, rec, dt = case('c1', base3, base_file=base3_path, mode='port')
exp = H.recompute(base3, refb)
d = float(np.abs(sub[H.LAB].to_numpy(np.float64) - exp[H.LAB].to_numpy(np.float64)).max())
chk('C1: mode S6F, fired [], S6F_BLEND_OK, == independent recompute (1e-6)', rec.get('mode') == 'S6F' and rec.get('fired') == [] and
    'S6F_BLEND_OK' in text and d <= 1e-6, 'max diff %.2e, %.0f s' % (d, dt))
run = rec.get('run') or {}
chk('C1: run RAN rc 0 not killed, start guard / deadline recorded, driver ok coverage 1.0', run.get('status') == 'RAN' and run.get('rc') == 0
    and run.get('killed') is False and run.get('g_start_h') == 5.75 and run.get('budget_h') == 3.3 and run.get('deadline_h') == 8.2 and
    (run.get('driver') or {}).get('ok') is True and ((run.get('driver') or {}).get('B') or {}).get('coverage') == 1.0, run.get('status'))
su = rec.get('setup') or {}
chk('C1: setup ok, base sha == the V224-FLIP commit file, driver sha, DEC fallback on this laptop (no wheels)', su.get('ok') is True and
    su.get('base_sha256') == H.sha_file(base3_path) == H.sha_file(os.path.join(t.work, 'submission_v224flip.csv')) and
    su.get('dec_ok') is False and 'S6F_DEC_FALLBACK' in text, (su.get('ok'), su.get('dec_ok')))
order = [text.find(m) for m in ('S6F_DEC_FALLBACK', 'S6F_SETUP_OK', 'S6F_START', 'S6F_STAGE_DONE', 'S6F_BLEND_OK', 'S6F_DONE')]
chk('C1: printed markers in order', all(p >= 0 for p in order) and order == sorted(order), order)
sub1 = sub.copy()
# ---- C2
print('C2 B_probs columns permuted', flush=True)
perm = refb[['StudyInstanceUID'] + H.LAB[::-1] + [c for c in refb.columns if c.startswith('z|')]]
pp = os.path.join(TMPB, 'perm.csv'); perm.to_csv(pp, index=False)
t, ns, text, sub, rec, dt = case('c2', base3, base_file=base3_path, mode='bprobs', bprobs=pp)
chk('C2: used; submission == C1 exactly (label order by name)', rec.get('mode') == 'S6F' and sub[H.LAB].equals(sub1[H.LAB]))
# ---- C3
print('C3 start guard', flush=True)
t, ns, text, sub, rec, dt = case('c3', base3, base_file=base3_path, age_h=6.0, mode='port')
canary_ok('C3', t, sub, rec, text, base3_path)
chk('C3: S6F_SKIPPED start guard, driver never started', 'S6F_SKIPPED start guard' in text and (rec.get('run') or {}).get('status') == 'SKIPPED_GUARD'
    and 'S6F_START' not in text)
# ---- C4
print('C4 hung driver, hard deadline', flush=True)
ov = {'G_START_H, BUDGET_H, DEADLINE_H, STOP_MARGIN_S, MIN_RUN_S = 5.75, 3.3, 8.2, 240.0, 1200.0':
      'G_START_H, BUDGET_H, DEADLINE_H, STOP_MARGIN_S, MIN_RUN_S = 5.75, 0.003, 8.2, 1.0, 0.0'}
t, ns, text, sub, rec, dt = case('c4', base3, base_file=base3_path, mode='sleep', overrides=ov)
canary_ok('C4', t, sub, rec, text, base3_path)
chk('C4: killed at the hard deadline (~11 s), run status RAN killed True', (rec.get('run') or {}).get('killed') is True and 'S6F_KILLED' in text
    and dt < 60, '%.0f s' % dt)
# ---- C5
print('C5 driver fails', flush=True)
t, ns, text, sub, rec, dt = case('c5', base3, base_file=base3_path, mode='fail')
canary_ok('C5', t, sub, rec, text, base3_path)
chk('C5: reason = driver failed', 'S6F driver failed' in str((rec.get('blend') or {}).get('why')))
# ---- C6 / C7
print('C6 B_probs without Effusion; C7 B_probs changed after the receipt', flush=True)
mp = os.path.join(TMPB, 'miss.csv'); refb.drop(columns=['Effusion']).to_csv(mp, index=False)
t, ns, text, sub, rec, dt = case('c6', base3, base_file=base3_path, mode='bprobs', bprobs=mp)
canary_ok('C6', t, sub, rec, text, base3_path)
rp = os.path.join(TMPB, 'ref.csv'); refb.to_csv(rp, index=False)
t, ns, text, sub, rec, dt = case('c7', base3, base_file=base3_path, mode='bprobs', bprobs=rp, tamper=True)
canary_ok('C7', t, sub, rec, text, base3_path)
chk('C7: reason = sha mismatch', 'differs from the driver receipt' in str((rec.get('blend') or {}).get('why')))
# ---- C8
print('C8 coverage 98 % / 97 % (N 100)', flush=True)
rng = np.random.default_rng(0)
ids = ['1.2.826.0.1.%04d' % i for i in range(100)]
bN = pd.DataFrame({'StudyInstanceUID': ids})
for c in H.LAB:
    bN[c] = pd.Series(rng.random(100)).rank(pct=True).astype(np.float32).values
bN_path = os.path.join(TMPB, 'baseN.csv'); bN.to_csv(bN_path, index=False)
for k, want in ((2, 'S6F'), (3, 'V224FLIP_CANARY')):
    S = pd.DataFrame({'StudyInstanceUID': ids[::-1]})                    # rows in another order (ids matched by value)
    for c in H.LAB:
        S[c] = np.clip(bN.set_index('StudyInstanceUID').loc[ids[::-1], c].values + rng.normal(0, .1, 100), 0, 1)
    S.loc[S.index[:k], H.LAB] = np.nan
    spth = os.path.join(TMPB, 's%d.csv' % k); S.to_csv(spth, index=False)
    t, ns, text, sub, rec, dt = case('c8_%d' % k, bN, mode='bprobs', bprobs=spth)
    bl = rec.get('blend') or {}
    chk('C8: %d unscored of 100 -> %s (coverage %.2f)' % (k, want, bl.get('coverage') or 0), rec.get('mode') == want and bl.get('n_unscored') == k)
    if want == 'S6F':
        uns = S['StudyInstanceUID'].iloc[:k].tolist()
        rb = {c: pd.Series(bN[c].values.astype(np.float64)).rank(pct=True).values for c in H.LAB}
        pos = [ids.index(u) for u in uns]
        du = max(abs(float(sub[c].iloc[p]) - float(np.float32(rb[c][p]))) for c in H.LAB for p in pos)
        exp = H.recompute(bN, S)
        d = float(np.abs(sub[H.LAB].to_numpy(np.float64) - exp[H.LAB].to_numpy(np.float64)).max())
        chk('C8: unscored rows == their V224-FLIP rank; whole output == independent recompute', du <= 1e-7 and d <= 1e-6, (du, d))
    else:
        canary_ok('C8_3', t, sub, rec, text, bN_path)
# ---- C9
print('C9 sanity floor (N 200)', flush=True)
ids2 = ['1.2.826.0.2.%04d' % i for i in range(200)]
b2 = pd.DataFrame({'StudyInstanceUID': ids2})
for c in H.LAB:
    b2[c] = pd.Series(rng.random(200)).rank(pct=True).astype(np.float32).values
b2_path = os.path.join(TMPB, 'base200.csv'); b2.to_csv(b2_path, index=False)
for nm, noise_only in (('c9_noise', True), ('c9_close', False)):
    S = pd.DataFrame({'StudyInstanceUID': ids2})
    for c in H.LAB:
        S[c] = rng.random(200) if noise_only else np.clip(b2[c].values + rng.normal(0, .15, 200), 0, 1)
    spth = os.path.join(TMPB, nm + '.csv'); S.to_csv(spth, index=False)
    t, ns, text, sub, rec, dt = case(nm, b2, mode='bprobs', bprobs=spth)
    bl = rec.get('blend') or {}
    if noise_only:
        canary_ok('C9 noise', t, sub, rec, text, b2_path)
        chk('C9 noise: reason = sanity floor', 'sanity floor' in str(bl.get('why')), bl.get('rho_mean'))
    else:
        chk('C9 close: used (rho mean %.2f)' % (bl.get('rho_mean') or 0), rec.get('mode') == 'S6F')
# ---- C10
print('C10 the V224 canary already fired (Fracture 0.5)', flush=True)
bF = base3.copy(); bF['Fracture'] = np.float32(0.5)
t, ns, text, sub, rec, dt = case('c10', bF, mode='port', fired=('r224->Fracture',))
exp = H.recompute(bF, refb, blank=('Fracture',))
d = float(np.abs(sub[H.LAB].to_numpy(np.float64) - exp[H.LAB].to_numpy(np.float64)).max())
chk('C10: used, Fracture == 0.5 kept, rest == recompute', rec.get('mode') == 'S6F' and bool((sub['Fracture'] == 0.5).all()) and d <= 1e-6
    and (rec.get('blend') or {}).get('base_blanked') == ['Fracture'], d)
# ---- C11
print('C11 setup fails (no submission.csv at setup)', flush=True)


def drop_sub(tree, ns):
    pass


t = H.Tree(TMPB, 'c11')
env0 = dict(os.environ)
setup_t, run_t, blend_t = H.cells()
nsx = {'__name__': '__main__', 'NB_T0': time.time() - 4.6 * 3600, 'sub': base3.copy(), '_canary': {'fired': []}}
buf = io.StringIO()
import contextlib
with contextlib.redirect_stdout(buf):
    for cell in (setup_t, run_t, blend_t):
        exec(compile(t.redirect(cell), '<c11>', 'exec'), nsx)
text = buf.getvalue()
sub = pd.read_csv(os.path.join(t.work, 'submission.csv'), dtype={'StudyInstanceUID': str})
rec = json.load(open(os.path.join(t.work, 's6f_receipt.json')))
chk('C11: setup failed, run skipped, canary from the in-memory sub (base_source), other 11 == sub',
    'S6F_SETUP_FAILED' in text and 'S6F_SKIPPED setup failed' in text and rec.get('mode') == 'V224FLIP_CANARY'
    and (rec.get('blend') or {}).get('base_source') == 'in-memory sub' and bool((sub['ACL'] == 0.5).all())
    and np.allclose(sub[[c for c in H.LAB if c != 'ACL']].values, base3[[c for c in H.LAB if c != 'ACL']].values, atol=0)
    and dict(os.environ) == env0, (rec.get('mode'), (rec.get('blend') or {}).get('base_source')))
shutil.rmtree(TMPB, ignore_errors=True)
print('TEST_CELLS %s %d/%d' % ('PASS' if all(R) else 'FAIL', sum(R), len(R)), flush=True)
sys.exit(0 if all(R) else 1)
