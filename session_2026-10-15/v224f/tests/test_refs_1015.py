# test_refs_1015.py (2026-10-15 dir, Claude subagent): tests for the v224ffs_check.py --ref-s fix (REPLACE S anchor) and the
# v224fs_check.py clean-FAIL fix (no NameError when the worker npz fail the names check). Offline, laptop CPU, no Kaggle.
# Needs the mock outputs of `python tests/mock_fs_output.py ff` and `... ffsr` (tests/mock_out_v224ffs20, tests/mock_out_v224ffsr20).
# Runs every case on the fixed gates and on the *.pre_refs_1015 copies (the pre v224ffs_check nests the current v224fs_check.py).
#   usage: python tests/test_refs_1015.py
import os, sys, json, time, shutil, io, contextlib, subprocess, re, tempfile
import numpy as np, pandas as pd

T = os.path.dirname(os.path.abspath(__file__)); V = os.path.dirname(T); ROOT = os.path.normpath(os.path.join(V, '..', '..'))
REF = os.path.join(ROOT, 'session_2026-10-15', 'v224ff', 'out_v224ff20_v1')
FFSR, FFS = os.path.join(T, 'mock_out_v224ffsr20'), os.path.join(T, 'mock_out_v224ffs20')
B_R, B_S = os.path.join(T, 'standin_v224ffsr20', 'BUILD_V224FS.json'), os.path.join(T, 'standin_v224ffs20', 'BUILD_V224FS.json')
assert os.path.isdir(FFSR) and os.path.isdir(FFS), 'run tests/mock_fs_output.py ff and ffsr first'
BR = json.load(open(B_R))
NB = json.load(open(os.path.join(T, 'standin_v224ffsr20', 'rsna_sub_v224ffsr20.ipynb'), encoding='utf-8'))
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
HF = [src(c) for c in NB['cells'] if src(c).startswith('# [eeshsaxena V224-F-S HF')][0]
sample = pd.read_csv(os.path.join(ROOT, '_data', 'sample_submission.csv'), dtype={'StudyInstanceUID': str})
labels = list(sample.columns)[1:]; test_ids = sample['StudyInstanceUID'].tolist()
TG = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
      'Contusion', 'Fracture']
W = tempfile.mkdtemp(prefix='refs1015_')


def gap_out():                                       # FFSR mock, worker 0's 2 studies swapped, re-made by the kernel's own HF-S cell
    M = os.path.join(W, 'ffsr_gap'); shutil.copytree(FFSR, M)
    zs = []
    for g in (0, 1):
        z = dict(np.load(os.path.join(M, '_fleet_%d.npz' % g), allow_pickle=False))
        if g == 0:
            z['P'] = z['P'][:, ::-1, :].copy()
        zs.append(z); np.savez(os.path.join(M, '_fleet_%d.npz' % g), **z)
    PP = np.concatenate([z['P'] for z in zs], axis=1).astype(np.float64); ii = np.concatenate([z['sids'] for z in zs]).astype(str)
    so = pd.DataFrame(PP.mean(0), columns=TG, index=ii).reindex(test_ids); so.insert(0, 'StudyInstanceUID', test_ids)
    so.to_csv(os.path.join(M, 'submission_fleetsolo.csv'), index=False)
    logp = os.path.join(M, 'rsna-sub-v224ffsr20.log'); E = json.load(open(logp, encoding='utf-8'))
    sv = json.loads(re.search(r'V224F_SAVED (\{.*\})', next(e['data'] for e in E if 'V224F_SAVED' in e['data'])).group(1))
    fr = json.load(open(os.path.join(M, 'fleet_receipt.json')))
    H = pd.read_csv(os.path.join(M, 'submission_V224.csv'), dtype={'StudyInstanceUID': str})
    g = {'NB_T0': time.time() - 600, 'labels': labels, 'test_ids': test_ids, 'sub': H.copy(), '_K2_TARGETS': TG, '_V224F': sv,
         '_canary': {'fired': [], 'r224_used': True}, '_k2_receipt': dict(fr, clean=True)}
    b = io.StringIO()
    with contextlib.redirect_stdout(b):
        exec(compile(HF.replace('/kaggle/working', M.replace('\\', '/')), 'HF-S', 'exec'), g)
    for z_ in g.get('_parts') or []:
        z_.close()
    g.clear()
    new = b.getvalue().strip().splitlines()
    assert [ln.split(' ', 1)[0] for ln in new] == ['V224FS_F2G', 'HF_BLEND_OK', 'HF_COMPLETE', 'V224F_DONE'], new
    k = [i for i, e in enumerate(E) if any(x in e['data'] for x in ('V224FS_F2G', 'HF_BLEND_OK', 'HF_COMPLETE', 'V224F_DONE'))]
    E = E[:k[0]] + [dict(E[k[0]], data=ln + '\n') for ln in new] + E[k[-1] + 1:]
    json.dump(E, open(logp, 'w', encoding='utf-8'))
    d = float(np.abs(pd.read_csv(os.path.join(M, 'submission.csv'))[labels].values - pd.read_csv(os.path.join(FFSR, 'submission.csv'))[labels].values).max())
    print(f'setup: gap output, submission max |diff| vs the clean FFSR mock {d:.4f}')
    return M


def ffs_s_row_half():                                # FF-S mock with its last S row x0.5 in both worker npz (a wrong reference)
    M = os.path.join(W, 'ffs_srow'); shutil.copytree(FFS, M)
    for g in (0, 1):
        z = dict(np.load(os.path.join(M, '_fleet_%d.npz' % g), allow_pickle=False)); nm = [str(x) for x in z['names']]
        z['P'] = z['P'].copy(); z['P'][nm.index(BR['s'][-1])] *= 0.5
        np.savez(os.path.join(M, '_fleet_%d.npz' % g), **z)
    return M


def ffsr_no_npz():                                   # FFSR mock without its worker npz (the nested 'Pm' NameError case)
    M = os.path.join(W, 'ffsr_nonpz'); shutil.copytree(FFSR, M)
    for g in (0, 1):
        os.remove(os.path.join(M, '_fleet_%d.npz' % g))
    return M


def ffs_gate(gate, out, build, refs=None):
    a = [sys.executable, os.path.join(V, gate), out, '--wf', '0.2', '--build', build,
         '--ref-flip', os.path.join(ROOT, 'session_2026-10-14', 'v224', 'out_v224flip_v1'),
         '--ref-exact', os.path.join(ROOT, 'session_2026-10-14', 'v224', 'out_v224_v1'), '--ref-fleet', REF] + (['--ref-s', refs] if refs else [])
    r = subprocess.run(a, capture_output=True, text=True, encoding='utf-8')
    lines = r.stdout.strip().splitlines()
    last = lines[-1] if lines else (r.stderr.strip().splitlines() or [''])[-1]
    own = [ln.strip()[:240] for ln in lines if ln.startswith('  FAIL ') and 'REPLACE' in ln]
    return r.returncode, last[:200], any(ln.startswith('  WARN REPLACE build without --ref-s') for ln in lines), own


def fs_gate(gate, out, build):
    r = subprocess.run([sys.executable, os.path.join(V, gate), out, '--wf', '0.2', '--build', build, '--ref-fleet', REF],
                       capture_output=True, text=True, encoding='utf-8')
    lines = r.stdout.strip().splitlines()
    two = [ln.strip()[:200] for ln in lines if 'two-group F recompute' in ln]
    return r.returncode, lines[-1][:120], 'NameError' in r.stdout, two


GAP, SROW, NONPZ = gap_out(), ffs_s_row_half(), ffsr_no_npz()
# name, gate args, expected post: (rc == 0, last-line prefix, WARN expected)
CASES = [
    ('FFSR mock, no --ref-s', (FFSR, B_R, None), (True, 'V224FFS_CHECK PASS 14/14', True)),
    ('FFSR mock, --ref-s FF-S mock', (FFSR, B_R, FFS), (True, 'V224FFS_CHECK PASS 15/15', False)),
    ('GAP study misalignment, no --ref-s', (GAP, B_R, None), (True, 'V224FFS_CHECK PASS 14/14', True)),
    ('GAP study misalignment, --ref-s FF-S mock', (GAP, B_R, FFS), (False, 'V224FFS_CHECK FAIL 14/15', False)),
    ('--ref-s = the FFSR output itself', (FFSR, B_R, FFSR), (False, 'V224FFS_CHECK FAIL 14/15', False)),
    ('--ref-s = FF20 commit (no S members)', (FFSR, B_R, REF), (False, 'V224FFS_CHECK FAIL 14/15', False)),
    ('--ref-s = FF-S mock with an S row x0.5', (FFSR, B_R, SROW), (False, 'V224FFS_CHECK FAIL 14/15', False)),
    ('FF-S build + --ref-s (refused)', (FFS, B_S, FFS), (False, 'AssertionError: --ref-s is for REPLACE builds only', False)),
    ('FF-S mock, FF-S build, no --ref-s', (FFS, B_S, None), (True, 'V224FFS_CHECK PASS 14/14', False)),
]
res = []
print('== v224ffs_check.py --ref-s (post = fixed, pre = v224ffs_check.py.pre_refs_1015)')
for name, (out, build, refs), (ok0, pre_last, warn) in CASES:
    rc, last, w, own = ffs_gate('v224ffs_check.py', out, build, refs)
    rcp, lastp, wp, _ = ffs_gate('v224ffs_check.py.pre_refs_1015', out, build, refs)
    ok = (rc == 0) == ok0 and last.startswith(pre_last) and w == warn
    res.append(ok)
    print(f'{"OK  " if ok else "BAD "} {name:<44} post rc {rc} WARN {w} | {last} | pre rc {rcp} | {lastp}')
    for ln in own:
        print('       post ' + ln)
print('== v224fs_check.py clean FAIL instead of NameError (post = fixed, pre = v224fs_check.py.pre_refs_1015)')
for name, out, build in [('FFSR mock without worker npz (Pm)', NONPZ, B_R), ('FF-S mock under the FFSR build (mean)', FFS, B_R)]:
    rc, last, ne, two = fs_gate('v224fs_check.py', out, build)
    rcp, lastp, nep, twop = fs_gate('v224fs_check.py.pre_refs_1015', out, build)
    ok = rc != 0 and rcp != 0 and not ne and any('not run: the worker npz failed the member-mean check' in t for t in two)
    res.append(ok)
    print(f'{"OK  " if ok else "BAD "} {name:<44} post rc {rc} NameError {ne} | {last} | pre rc {rcp} NameError {nep} | {lastp}')
    print('       post ' + ' ; '.join(two)); print('       pre  ' + ' ; '.join(twop))
shutil.rmtree(W, ignore_errors=True)
print(f'TEST_REFS_1015 {"PASS" if all(res) else "FAIL"} {sum(res)}/{len(res)}')
