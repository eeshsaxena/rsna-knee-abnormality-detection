#!/usr/bin/env python
# sim_flrm.py (2026-10-14, Claude subagent): offline CPU simulation of the V224-FLIP-RM tail IN THE FLIP CONTEXT, before its one
# commit. Executes the pushed notebook's own cells (read from kernel_v224flrm, not from the builder): her stage 3 (cell 14), the
# V224 CANARY (cell 15) and RM_GUARD (cell 16), with FLIP_TTA_FORCED_ON = True as the FLIP cell sets it, on the V224-FLIP commit
# stage csvs (out_v224flip_v1: flip R7 ours_sub, r384 raw / sub) with the r224 part replaced by the V224-F20-RM commit's
# mirror-TTA r224 files (out_v224rm_v1: raptor224_raw / _plain / _mirror / _sub + tta receipt; same 3 studies).
# Prints SIM_FLRM PASS/FAIL n/m; writes sim_flrm_expected_submission.csv (the predicted commit submission on the 3 rows).
#   The RM cell (cell 11) itself runs first in the same namespace (its /tmp paths -> a temp dir holding her cell 3 and cell 10
#   bodies, taken from this notebook), so RM_ON and the patched-script install come from the real cell, not from a fixture.
#   F1 clean: CANARY_CLEAN, RM_GUARD_CLEAN, V224RM_DONE with flip_tta_forced True / rm_on True / fired []; submission == an
#      independent scipy recompute (V224 recipe, no re-rank); vs the V224-FLIP commit submission only the columns whose r224 ranks
#      changed move.
#   F2 mirror receipt missing / wrong perm / raw inconsistent -> RM_GUARD_FIRED, Fracture 0.5, other 11 columns byte-equal to F1,
#      canary receipt keeps flip_tta_forced True and lists 'r224 mirror->Fracture'.
#   F3 r224 failed (R224_OK False, no csv) -> CANARY_FIRED (Fracture 0.5, 11 columns == recompute without r224), RM_GUARD no-op.
import os, sys, json, shutil, io, contextlib, time, tempfile, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
NB = os.path.join(HERE, 'kernel_v224flrm', 'rsna_sub_v224flrm.ipynb')
FLIP_OUT = os.path.join(ROOT, 'session_2026-10-14', 'v224', 'out_v224flip_v1')
RM_OUT = os.path.join(HERE, 'out_v224rm_v1')
R = []


def chk(name, ok, info=''):
    R.append(bool(ok))
    print(f'  {"OK  " if ok else "FAIL"} {name} {info}', flush=True)


nb = json.load(open(NB, encoding='utf-8'))
S = [c['source'] if isinstance(c['source'], str) else ''.join(c['source']) for c in nb['cells']]
ST3, CAN, GRD, RMC = S[14], S[15], S[16], S[11]
assert RMC.startswith('# [eeshsaxena V224-RM; the ONE change') and S[3].startswith('%%writefile /tmp/raptor_run.py\n') \
    and S[10].startswith('%%writefile /tmp/raptor224_run.py\n')
assert ST3.startswith('# Stage 3: per-finding rank blend') and CAN.startswith('# [eeshsaxena CANARY') and GRD.startswith('# [eeshsaxena V224-RM GUARD')
assert S[7].startswith('# [eeshsaxena V224-FLIP') and 'FLIP_TTA_FORCED_ON = True' in S[7], 'the FLIP cell sets FLIP_TTA_FORCED_ON'
sample = pd.read_csv(os.path.join(ROOT, '_data', 'sample_submission.csv'), dtype={'StudyInstanceUID': str})
labels, ids = list(sample.columns[1:]), sample['StudyInstanceUID'].tolist()
n = len(ids)


def setup(T, r224='rm', receipt='ok', raw='ok'):
    T = Path(T)
    if T.exists():
        shutil.rmtree(T)
    W, D, PR = T / 'working', T / 'data', T / 'pred'
    for d in (W, D, PR / 'worker0'):
        d.mkdir(parents=True)
    for f in ('ours_sub.csv', 'raptor_raw.csv', 'raptor_sub.csv'):
        shutil.copy(os.path.join(FLIP_OUT, f), W / f)
    shutil.copy(os.path.join(ROOT, '_data', 'test.csv'), D / 'test.csv')
    shutil.copy(os.path.join(ROOT, '_data', 'sample_submission.csv'), D / 'sample_submission.csv')
    shutil.copy(W / 'ours_sub.csv', PR / 'worker0' / 'predictions.csv')
    if r224 == 'rm':
        for f in ('raptor224_sub.csv', 'raptor224_raw.csv', 'raptor224_raw_plain.csv', 'raptor224_raw_mirror.csv', 'raptor224_tta_receipt.json'):
            shutil.copy(os.path.join(RM_OUT, f), W / f)
        if receipt == 'missing':
            os.remove(W / 'raptor224_tta_receipt.json')
        elif receipt == 'perm':
            rc = json.load(open(W / 'raptor224_tta_receipt.json')); rc['perm'] = [0, 1, 3, 2] + list(range(4, 12))
            json.dump(rc, open(W / 'raptor224_tta_receipt.json', 'w'))
        if raw == 'reversed':
            df = pd.read_csv(W / 'raptor224_raw.csv', dtype={'StudyInstanceUID': str})
            df[labels] = 1 - df[labels]
            df.to_csv(W / 'raptor224_raw.csv', index=False, float_format='%.9g')
    return T, W, D, PR


def run(T, W, D, PR, r224_ok=True):
    g = {'DATA': D, 'WORK': W, 'pred': PR, 'R224_OK': r224_ok, 'RAPTOR_SECONDS': 54.0, 'NB_T0': time.time(), 'json': json,
         'time': time, 'display': lambda *a, **k: None, 'FLIP_TTA_FORCED_ON': True}
    tmp = (T / 'tmp'); tmp.mkdir()
    for k, nm in ((3, 'raptor_run.py'), (10, 'raptor224_run.py')):      # what IPython's %%writefile writes: the body after line 1
        open(tmp / nm, 'w', encoding='utf-8', newline='').write(S[k].split('\n', 1)[1])
    rmc = RMC.replace("'/tmp/", "'" + str(tmp).replace('\\', '/') + '/')
    assert "'/tmp/" not in rmc
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(rmc, 'rm_cell', 'exec'), g)
        exec(compile(ST3.replace("'/kaggle/working", "'" + str(W).replace('\\', '/')), 'stage3', 'exec'), g)
        exec(compile(CAN, 'canary', 'exec'), g)
        exec(compile(GRD.replace("'/kaggle/working'", repr(str(W).replace('\\', '/'))), 'rm_guard', 'exec'), g)
    return buf.getvalue(), open(W / 'submission.csv', 'rb').read(), json.load(open(W / 'canary_receipt.json'))


def rd(d, f):
    return pd.read_csv(os.path.join(d, f), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, labels].values.astype(np.float64)


pct = lambda v: rankdata(v, method='average') / len(v)


def recompute(d, use224=True):
    raw, ours = rd(d, 'raptor_raw.csv'), rd(d, 'ours_sub.csv')
    r224 = rd(d, 'raptor224_sub.csv') if use224 else None
    out = np.zeros((n, 12))
    for j in range(12):
        rr = pct(raw[:, j])
        if use224:
            rr = .65 * rr + .35 * pct(r224[:, j])
        out[:, j] = .5 * rr + .5 * pct(ours[:, j])
    return out


TD = tempfile.mkdtemp(prefix='flrm_sim_')
print('F1 clean (FLIP context)', flush=True)
T, W, D, PR = setup(os.path.join(TD, 'clean'))
o, s1, cr = run(T, W, D, PR)
chk('RM cell ran in the FLIP namespace: R224_MIRROR_VERIFIED + R224_MIRROR_PATCHED, RM_ON True, patched script installed',
    'R224_MIRROR_VERIFIED' in o and 'R224_MIRROR_PATCHED' in o and hashlib.sha256(open(T / 'tmp' / 'raptor224_run.py', 'rb').read()).hexdigest()
    == '90b2a4eb29f22d63afd82bff822af3cc1c47167b2be6a0de7af57117073b0a9d', [ln[:90] for ln in o.splitlines() if 'PATCHED' in ln])
chk('CANARY_CLEAN then RM_GUARD_CLEAN then V224RM_DONE', 0 <= o.find('CANARY_CLEAN') < o.find('RM_GUARD_CLEAN') < o.find('V224RM_DONE'),
    [ln[:100] for ln in o.splitlines() if 'DONE' in ln])
chk('canary receipt: r224 used, fired [], flip_tta_forced True, use_f False, rm_on True, rm_guard mirror_ok',
    cr.get('r224_used') is True and cr.get('fired') == [] and cr.get('flip_tta_forced') is True and cr.get('use_f') is False
    and cr.get('rm_on') is True and (cr.get('rm_guard') or {}).get('mirror_ok') is True, {k: cr.get(k) for k in ('fired', 'flip_tta_forced', 'rm_on')})
sub = pd.read_csv(io.BytesIO(s1), dtype={'StudyInstanceUID': str})
V = sub.set_index('StudyInstanceUID').loc[ids, labels].values.astype(np.float64)
dd = float(np.abs(V - recompute(W)).max())
chk('submission == independent scipy recompute (V224 recipe, flip R7 + mirror r224)', dd <= 1e-6, 'max diff %.3g' % dd)
chk('submission ids / columns == sample', sub['StudyInstanceUID'].tolist() == ids and list(sub.columns) == list(sample.columns))
fsub = rd(FLIP_OUT, 'submission.csv')
r224_moved = [c for j, c in enumerate(labels) if not np.array_equal(rd(FLIP_OUT, 'raptor224_sub.csv')[:, j], rd(W, 'raptor224_sub.csv')[:, j])]
sub_moved = [c for j, c in enumerate(labels) if np.abs(V[:, j] - fsub[:, j]).max() > 1e-6]
chk('vs the V224-FLIP commit submission: only columns whose r224 ranks changed move', set(sub_moved) <= set(r224_moved),
    {'r224 rank columns changed': r224_moved, 'submission columns changed': sub_moved,
     'max': {c: round(float(np.abs(V[:, j] - fsub[:, j]).max()), 4) for j, c in enumerate(labels) if c in sub_moved}})
exp = sub.copy()
exp.to_csv(os.path.join(HERE, 'sim_flrm_expected_submission.csv'), index=False)

print('F2 RM_GUARD fires in the FLIP context', flush=True)
for case, kw in (('mirror receipt missing', dict(receipt='missing')), ('wrong perm', dict(receipt='perm')),
                 ('raw not the r224 csv ranks', dict(raw='reversed'))):
    T, W, D, PR = setup(os.path.join(TD, case.replace(' ', '_')), **kw)
    o, s, cr = run(T, W, D, PR)
    a = pd.read_csv(io.BytesIO(s1), dtype=str, keep_default_na=False)
    b = pd.read_csv(io.BytesIO(s), dtype=str, keep_default_na=False)
    keep = [c for c in a.columns if c != 'Fracture']
    chk(f'{case}: RM_GUARD_FIRED, Fracture 0.5, other 11 columns byte-equal to F1, receipt fired + flip kept',
        'RM_GUARD_FIRED' in o and 'CANARY_CLEAN' in o and a[keep].equals(b[keep]) and (b['Fracture'] == '0.5').all()
        and cr.get('fired') == ['r224 mirror->Fracture'] and cr.get('flip_tta_forced') is True,
        (cr.get('rm_guard') or {}).get('why', '')[:60])

print('F3 r224 failed in the FLIP context (her canary path)', flush=True)
T, W, D, PR = setup(os.path.join(TD, 'r224fail'), r224='none')
o, s, cr = run(T, W, D, PR, r224_ok=False)
sub3 = pd.read_csv(io.BytesIO(s), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, labels]
fi = labels.index('Fracture')
keep = [j for j in range(12) if j != fi]
d3 = float(np.abs(sub3.values[:, keep] - recompute(W, use224=False)[:, keep]).max())
chk('CANARY_FIRED, Fracture 0.5, 11 columns == recompute without r224, RM_GUARD no-op (clean, r224 unused)',
    'CANARY_FIRED' in o and 'RM_GUARD_CLEAN' in o and bool((sub3['Fracture'] == 0.5).all()) and d3 <= 1e-6
    and cr.get('fired') == ['r224->Fracture'] and (cr.get('rm_guard') or {}).get('r224_used') is False, 'max diff %.3g' % d3)
shutil.rmtree(TD, ignore_errors=True)
print('  info expected commit submission (3 rows) sha', hashlib.sha256(s1).hexdigest()[:16], '-> sim_flrm_expected_submission.csv', flush=True)
print(f'SIM_FLRM {"PASS" if all(R) else "FAIL"} {sum(R)}/{len(R)}', flush=True)
sys.exit(0 if all(R) else 1)
