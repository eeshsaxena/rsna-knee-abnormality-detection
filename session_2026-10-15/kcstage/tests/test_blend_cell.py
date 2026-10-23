# test_blend_cell.py (10-15, Claude subagent, kcstage): the KC BLEND cell of the BUILT stand-in kernel, executed as written (only
# '/kaggle/working' redirected; B13 also flips ON_FAIL to 'base'), on synthetic N=200 outputs: the use / canary rules and the blend math.
#   py -3.12 -I tests/test_blend_cell.py  -> TEST_BLEND PASS n/m
import os, io, sys, json, shutil, hashlib, tempfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KS = os.path.dirname(HERE)
NB = os.path.join(KS, 'kernel_v224s6fkcstandin', 'rsna_sub_v224s6fkcstandin.ipynb')
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
       'Contusion', 'Fracture']
cells = [''.join(c['source']) if isinstance(c['source'], list) else c['source'] for c in json.load(open(NB, encoding='utf-8'))['cells']]
BLEND = cells[-1]
assert 'KC 3/3 BLEND' in BLEND.splitlines()[0]
R = []


def chk(name, ok, info=''):
    R.append(bool(ok))
    print('  %s %s %s' % ('OK  ' if ok else 'FAIL', name, info), flush=True)


rng = np.random.default_rng(3)
N = 200
ids = ['1.2.3.%d' % i for i in range(N)]
pct = lambda v: pd.Series(np.asarray(v, np.float64)).rank(method='average', pct=True).values


def case(name, nan_rows=0, mut=None, run_mut=None, base_mut=None, on_fail=None, cols=None):
    W = tempfile.mkdtemp(prefix='kcblend_').replace(chr(92), '/')
    os.makedirs(W + '/kc')
    base = pd.DataFrame({'StudyInstanceUID': ids})
    for c in LAB:
        base[c] = pct(rng.random(N)).astype(np.float32)
    if base_mut:
        base_mut(base)
    base.to_csv(W + '/submission_s6f.csv', index=False)
    shutil.copyfile(W + '/submission_s6f.csv', W + '/submission.csv')
    kp = pd.DataFrame({'StudyInstanceUID': ids})
    for c in LAB:
        kp[c] = rng.random(N)
    if nan_rows:
        kp.loc[:nan_rows - 1, LAB] = np.nan
    if mut:
        kp = mut(kp)
    if cols:
        kp = kp[cols]
    kp.to_csv(W + '/kc/kc_probs.csv', index=False)
    h = hashlib.sha256(open(W + '/kc/kc_probs.csv', 'rb').read()).hexdigest()
    setup = {'ok': True, 'base_sha256': hashlib.sha256(open(W + '/submission_s6f.csv', 'rb').read()).hexdigest()}
    run = {'status': 'RAN', 'killed': False, 'driver': {'ok': True, 'B': {'probs_sha256': h}}}
    if run_mut:
        run_mut(run, setup)
    cell = BLEND.replace('/kaggle/working', W)
    if on_fail:
        cell = cell.replace("\"Baker's\", 'canary'", "\"Baker's\", '%s'" % on_fail)
    g = {'__name__': '__main__', 'NB_T0': 0.0, '_KC_SETUP': setup, '_KC_RUN': run}
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        exec(compile(cell, '<blend>', 'exec'), g)
        log = sys.stdout.getvalue()
    finally:
        sys.stdout = old
    out = pd.read_csv(W + '/submission.csv', dtype={'StudyInstanceUID': str})
    b = pd.read_csv(W + '/submission_s6f.csv', dtype={'StudyInstanceUID': str})
    rec = g['_KC_BLEND']
    return out, b, kp, rec, log, W


def expect(b, kp):
    S = kp.set_index('StudyInstanceUID').loc[ids, LAB].to_numpy(np.float64)
    ok = np.isfinite(S).all(axis=1)
    e = b.copy()
    for j, c in enumerate(LAB):
        if b[c].nunique() <= 1:
            continue
        rb = pct(b[c].values)
        rs = rb.copy()
        rs[ok] = pct(S[ok, j])
        e[c] = (0.8 * rb + 0.2 * rs).astype(np.float32)
    return e


def canary_ok(out, b, rec, log):
    t_out = pd.read_csv(io.StringIO(out.to_csv(index=False)), dtype=str)
    t_b = pd.read_csv(io.StringIO(b.to_csv(index=False)), dtype=str)
    return rec['mode'] == 'KC_CANARY' and rec['fired'] == ["kc->Baker's"] and bool((out["Baker's"] == 0.5).all()) and \
        all(t_out[c].tolist() == t_b[c].tolist() for c in LAB if c != "Baker's") and 'KC_CANARY_FIRED' in log


out, b, kp, rec, log, W = case('B1')
d = float(np.abs(out[LAB].to_numpy(np.float64) - expect(b, kp)[LAB].to_numpy(np.float64)).max())
chk('B1 clean: mode KC, == independent recompute (1e-6)', rec['mode'] == 'KC' and d <= 1e-6 and 'KC_BLEND_OK' in log, (rec['mode'], d))
out, b, kp, rec, log, W = case('B2', nan_rows=4)
e = expect(b, kp)
d = float(np.abs(out[LAB].to_numpy(np.float64) - e[LAB].to_numpy(np.float64)).max())
rb_rows = all(np.allclose(out.loc[:3, c].to_numpy(np.float64), pct(b[c].values)[:4], atol=1e-6) for c in LAB)
chk('B2 coverage .98 (4 of 200 unscored): used; unscored rows keep the base rank exactly', rec['mode'] == 'KC' and rec['coverage'] == 0.98
    and d <= 1e-6 and rb_rows, (rec.get('coverage'), d))
out, b, kp, rec, log, W = case('B3', nan_rows=5)
chk('B3 coverage .975 -> KC canary (Baker\'s 0.5, other 11 columns == base text)', canary_ok(out, b, rec, log), rec.get('why'))
out, b, kp, rec, log, W = case('B4', run_mut=lambda r, s: r['driver']['B'].update(probs_sha256='0' * 64))
chk('B4 kc_probs sha != driver receipt -> canary', canary_ok(out, b, rec, log), rec.get('why'))
out, b, kp, rec, log, W = case('B5', cols=['StudyInstanceUID'] + [c for c in LAB if c != 'Contusion'])
chk('B5 a label missing -> canary', canary_ok(out, b, rec, log), rec.get('why'))
out, b, kp, rec, log, W = case('B6', mut=lambda k: k.assign(StudyInstanceUID=['x%d' % i for i in range(N)]))
chk('B6 ids differ -> canary', canary_ok(out, b, rec, log), rec.get('why'))
rng = np.random.default_rng(3)
o1, b1, k1, r1, _, _ = case('B7a')
rng = np.random.default_rng(3)
o2, b2, k2, r2, _, _ = case('B7b', cols=['StudyInstanceUID'] + LAB[::-1])
chk('B7 kc_probs columns permuted -> identical submission (labels by name)', r1['mode'] == r2['mode'] == 'KC' and o1.equals(o2))
out, b, kp, rec, log, W = case('B8', base_mut=lambda bb: bb.__setitem__('ACL', np.float32(0.5)) or bb.__setitem__('Fracture', np.float32(0.5)))
chk('B8 constant base columns (S6F ACL, V224 Fracture canaries) kept exactly; others blended', rec['mode'] == 'KC' and
    bool((out['ACL'] == 0.5).all() and (out['Fracture'] == 0.5).all()) and rec['base_blanked'] == ['ACL', 'Fracture'])
out, b, kp, rec, log, W = case('B9', run_mut=lambda r, s: r.update(killed=True))
chk('B9 run killed -> canary', canary_ok(out, b, rec, log), rec.get('why'))
out, b, kp, rec, log, W = case('B10', run_mut=lambda r, s: r.update(status='SKIPPED_GUARD'))
chk('B10 start guard skip -> canary', canary_ok(out, b, rec, log), rec.get('why'))
out, b, kp, rec, log, W = case('B11', run_mut=lambda r, s: s.update(base_sha256='0' * 64))
chk('B11 frozen base sha mismatch -> canary on the unfrozen submission.csv', canary_ok(out, b, rec, log) and
    rec.get('base_source') == 'submission.csv (not frozen)', rec.get('base_source'))
out, b, kp, rec, log, W = case('B12', mut=lambda k: k.assign(MCL=k['MCL'] + 1.5))
chk('B12 probability > 1 -> canary', canary_ok(out, b, rec, log), rec.get('why'))
out, b, kp, rec, log, W = case('B13', nan_rows=5, on_fail='base')
t_out = pd.read_csv(io.StringIO(out.to_csv(index=False)), dtype=str)
t_b = pd.read_csv(io.StringIO(b.to_csv(index=False)), dtype=str)
chk('B13 final-pick form (--on-fail base): coverage .975 -> base rows unchanged, no canary', rec['mode'] == 'KC_BASE_ROWS' and
    rec['fired'] == [] and all(t_out[c].tolist() == t_b[c].tolist() for c in LAB) and 'KC_NOT_USED' in log, rec.get('mode'))
print('TEST_BLEND %s %d/%d' % ('PASS' if all(R) else 'FAIL', sum(R), len(R)), flush=True)
sys.exit(0 if all(R) else 1)
