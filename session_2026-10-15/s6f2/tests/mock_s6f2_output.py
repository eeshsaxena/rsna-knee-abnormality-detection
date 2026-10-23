#!/usr/bin/env python
# mock_s6f2_output.py (10-15, Claude subagent): a MOCK commit output of an s6f2 kernel, then the gate (v224s6f2_check.py) on it and
# on mutated copies. Offline, laptop CPU, -I. Built from REAL outputs only (as s6f/tests/mock_s6f_output.py):
#   - the base's v1 commit output (flrm: v224rm/out_v224flrm_v1; flips20: v224f/out_v224flips20_v1; stage csvs, npz, receipts, logs),
#   - the 3 appended cells of the built s6f2 notebook EXECUTED as written (s6f2_harness: paths redirected) with the real driver port on
#     nartaa's REAL commit part outputs -> submission_v224flip.csv (= the base commit's submission bytes), s6f/, receipts, submission.csv;
#   - the kernel log = the base log + the lines our cells and the driver print, in run order.
# Mock-only patches (stated, as in s6f): the setup receipt / its log line take the Kaggle DEC_OK form (the manylinux wheels cannot
# install here); the env-isolation probe is the harness's Kaggle-like stand-in.
# [F1 10-15] the import path in every receipt copy takes the REAL Kaggle form of the S6F v1 commit (s6f/out_v224flips6f_v1, same docker
# pin): pythonpath_stock = the image PYTHONPATH (3 entries), pythonpath_s6f = env PYTHONPATH = /tmp/s6f_dec + it, sys_path /
# driver_sys_path = that commit's (the laptop has no image PYTHONPATH, so the harness gave [] / '/tmp/s6f_dec'). New negatives N18-N20:
# an S-stage dir / S dataset dir on S6fall's import path, each consistent across every receipt copy (pre-change copy: .pre_f1).
#   usage: py -3.12 -I tests/mock_s6f2_output.py flrm|flips20  -> MOCK_S6F2 PASS/FAIL n/m
import os, sys, json, shutil, hashlib, subprocess
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s6f2_harness as H

KIND = sys.argv[1] if len(sys.argv) > 1 else 'flrm'
assert KIND in ('flrm', 'flips20')
KD = H.KIND[KIND]
TAG, LAST, BASE_OUT = KD['tag'], KD['last'], KD['base_out']
GATE = os.path.join(H.S6F2, 'v224s6f2_check.py')
BUILD = os.path.join(H.S6F2, 'build_receipt_%s.json' % TAG)
MOCK = os.path.join(HERE, 'mock_out_' + TAG)
MOCK_C = os.path.join(HERE, 'mock_out_%s_canary' % TAG)
NEG = os.path.join(HERE, '_neg_' + KIND)
AGE = {'flrm': 4.7, 'flips20': 4.9}[KIND]
R = []
# [F1] the REAL Kaggle import path of the S6F v1 commit (MEASURED: s6f_setup_receipt.json + s6f/s6f_driver_receipt.json env_isolation)
_V1 = os.path.join(H.SESS, 's6f', 'out_v224flips6f_v1')
_V1_ISO = json.load(open(os.path.join(_V1, 's6f', 's6f_driver_receipt.json'), encoding='utf-8'))['env_isolation']
STOCK = json.load(open(os.path.join(_V1, 's6f_setup_receipt.json'), encoding='utf-8'))['pythonpath_stock']
PP_S6F = ':'.join(['/tmp/s6f_dec'] + STOCK)
assert STOCK == ['/kaggle/lib/kagglegym', '/kaggle/lib', '/kaggle/input/competitions/rsna-knee-abnormality-detection'], STOCK
assert _V1_ISO['PYTHONPATH'] == PP_S6F, _V1_ISO['PYTHONPATH']


def chk(name, ok, info=''):
    R.append(bool(ok))
    print('  %s %s %s' % ('OK  ' if ok else 'FAIL', name, info), flush=True)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def build_mock(dst, age_h):
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(BASE_OUT, dst, ignore=shutil.ignore_patterns('__pycache__'))
    base_log = [f for f in os.listdir(dst) if f.endswith('.log') and 'cufile' not in f]
    assert len(base_log) == 1, base_log
    os.replace(os.path.join(dst, base_log[0]), os.path.join(dst, 'rsna-sub-%s.log' % TAG))
    t = H.Tree(os.path.join(HERE, '_tmp_mock_' + KIND), os.path.basename(dst))
    base = H.base_df(KIND)
    ns, text = H.run_cells(t, KIND, base, age_h=age_h, mode='port', base_file=H.base_file(KIND),
                           **({'gcr_fired': []} if KIND == 'flrm' else {'v224f_fired': []}))
    dec = {'target': '/tmp/s6f_dec', 'pip_rc': 0, 'selftest': {'pydicom': '3.0.1', 'versions': {'pylibjpeg': '2.1.0', 'libjpeg': '2.4.0', 'openjpeg': '2.6.0'},
           'decode': {'JLL57_12U': {'exact': True}, 'JLL70_16S': {'exact': True}, 'U_16S': {'exact': True}}}}
    p = os.path.join(t.work, 's6f_setup_receipt.json')
    r = json.load(open(p))
    r.update(dec_ok=True, dec=dec, pythonpath_s6f=PP_S6F, pythonpath_stock=list(STOCK))              # [F1] Kaggle form
    json.dump(r, open(p, 'w'), indent=1)
    rr = json.load(open(os.path.join(t.work, 's6f_receipt.json')))
    rr['setup'].update(dec_ok=True, dec=dec, pythonpath_s6f=PP_S6F, pythonpath_stock=list(STOCK))
    iso = ((rr.get('run') or {}).get('driver') or {}).get('env_isolation')       # [F1] the s6f_receipt copy of the driver probe
    if isinstance(iso, dict):
        iso.update(PYTHONPATH=PP_S6F, sys_path=list(_V1_ISO['sys_path']), driver_sys_path=list(_V1_ISO['driver_sys_path']))
    json.dump(rr, open(os.path.join(t.work, 's6f_receipt.json'), 'w'), indent=1)
    pd_ = os.path.join(t.work, 's6f', 's6f_driver_receipt.json')
    if os.path.isfile(pd_):                                           # [F1] the driver receipt file (the gate's env_isolation source)
        dj = json.load(open(pd_, encoding='utf-8'))
        dj['env_isolation'].update(PYTHONPATH=PP_S6F, sys_path=list(_V1_ISO['sys_path']), driver_sys_path=list(_V1_ISO['driver_sys_path']))
        json.dump(dj, open(pd_, 'w', encoding='utf-8'), indent=1)
    ppq = '"PYTHONPATH": "/tmp/s6f_dec"'
    lines = []
    for ln in text.splitlines():
        if ln.startswith('S6F_DEC_FALLBACK '):
            ln = 'S6F_DEC_OK ' + json.dumps({'why': None, 'pip_rc': 0, 'pydicom': '3.0.1', 'versions': dec['selftest']['versions'],
                                             'decode': {k: True for k in dec['selftest']['decode']}})
        if ln.startswith('S6F_SETUP_OK '):
            ln = ln.replace('"dec_ok": false', '"dec_ok": true').replace('"pythonpath_s6f": null', '"pythonpath_s6f": "%s"' % PP_S6F)
        lines.append(ln)
        if ln.startswith('S6F_START '):
            dl = os.path.join(t.work, 's6f', 's6f_driver_log.txt')
            if os.path.isfile(dl):
                lines += [x.replace(ppq, '"PYTHONPATH": "%s"' % PP_S6F) for x in open(dl, encoding='utf-8').read().splitlines()]
    for f in ('submission.csv', 'submission_v224flip.csv', 's6f_setup_receipt.json', 's6f_receipt.json'):
        shutil.copyfile(os.path.join(t.work, f), os.path.join(dst, f))
    if os.path.isdir(os.path.join(t.work, 's6f')):
        shutil.copytree(os.path.join(t.work, 's6f'), os.path.join(dst, 's6f'))
    lp = os.path.join(dst, 'rsna-sub-%s.log' % TAG)
    E = json.load(open(lp, encoding='utf-8'))
    t_end = E[-1]['time'] if E else 0.0
    E += [{'stream_name': 'stdout', 'time': t_end + 1 + 0.01 * i, 'data': ln + '\n'} for i, ln in enumerate(lines)]
    json.dump(E, open(lp, 'w', encoding='utf-8'))
    shutil.rmtree(t.t, ignore_errors=True)
    return text


def gate(d, *extra):
    r = subprocess.run([sys.executable, '-I', GATE, d, '--build', BUILD] + list(extra), capture_output=True, text=True, timeout=1800,
                       encoding='utf-8', errors='replace')
    v = [l for l in r.stdout.splitlines() if l.startswith('V224S6F2_CHECK ')][-1:] or ['(no verdict) ' + r.stderr[-300:]]
    return r.returncode, v[-1], r.stdout


print('MOCK_S6F2 %s: building the mock commit output from %s' % (KIND, os.path.relpath(BASE_OUT, H.ROOT)), flush=True)
build_mock(MOCK, AGE)
rc, v, out = gate(MOCK)
print(out, flush=True)
open(os.path.join(HERE, 'mock_s6f2_gate_positive_%s.txt' % KIND), 'w', encoding='utf-8').write(out)
chk('POSITIVE: the gate PASSES the mock clean output', rc == 0 and 'PASS' in v, v)


def mutate(name, fn, expect_fail=True, must=(), extra=()):
    d = os.path.join(NEG, name)
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(MOCK, d)
    fn(d)
    rc, v, out = gate(d, *extra)
    failed = [l.strip()[5:] for l in out.splitlines() if l.strip().startswith('FAIL')]
    hit = all(any(m in f for f in failed) for m in must)
    ok = (rc != 0 and 'FAIL' in v and hit) if expect_fail else (rc == 0 and 'PASS' in v)
    chk('%s %s: %s' % ('NEGATIVE' if expect_fail else 'POSITIVE', name, v), ok,
        ('fails: ' + ' || '.join(f[:70] for f in failed[:4])) if expect_fail else '')


def jedit(d, rel, fn):
    p = os.path.join(d, *rel.split('/'))
    j = json.load(open(p, encoding='utf-8'))
    fn(j)
    json.dump(j, open(p, 'w', encoding='utf-8'), indent=1)


def logedit(d, fn):
    p = os.path.join(d, 'rsna-sub-%s.log' % TAG)
    E = json.load(open(p, encoding='utf-8'))
    E = fn(E)
    json.dump(E, open(p, 'w', encoding='utf-8'))


def bprobs_edit(d, fn, resha=True):
    p = os.path.join(d, 's6f', 'B', 'B_probs.csv')
    df = pd.read_csv(p, dtype={'StudyInstanceUID': str})
    df = fn(df)
    df.to_csv(p, index=False, float_format='%.17g', lineterminator='\n')
    if resha:
        h = sha(p)
        jedit(d, 's6f/s6f_driver_receipt.json', lambda j: j['B'].__setitem__('probs_sha256', h))
        jedit(d, 's6f_receipt.json', lambda j: (j['run']['driver']['B'].__setitem__('probs_sha256', h), j['blend'].__setitem__('b_probs_sha256', h)))


def sub_write(d, df):
    p = os.path.join(d, 'submission.csv')
    df.to_csv(p, index=False, lineterminator='\n')
    h = sha(p)
    jedit(d, 's6f_receipt.json', lambda j: j['blend'].__setitem__('submission_sha256', h))


def frozen_write(d, df, also_setup=True):
    p = os.path.join(d, 'submission_v224flip.csv')
    df.to_csv(p, index=False, lineterminator='\n')
    if also_setup:                                                    # a consistent run: the setup froze exactly these bytes
        h = sha(p)
        jedit(d, 's6f_setup_receipt.json', lambda j: j.__setitem__('base_sha256', h))
        jedit(d, 's6f_receipt.json', lambda j: j['setup'].__setitem__('base_sha256', h))


def read(d, f):
    return pd.read_csv(os.path.join(d, f), dtype={'StudyInstanceUID': str})


def reblend(d, blank):
    base = read(d, 'submission_v224flip.csv')
    B = read(d, 's6f/B/B_probs.csv')
    sub_write(d, H.recompute(base, B, blank=blank))


# ---- generic negatives (the s6f set) ------------------------------------------------------------------------------------------
def n1(d):
    def f(df):
        df.loc[0, 'Effusion'] = df.loc[0, 'Effusion'] + 2e-3
        return df
    bprobs_edit(d, f)


def n3(d):
    s = read(d, 'submission.csv')
    s.loc[1, 'PF OA'] = np.float32(s.loc[1, 'PF OA'] + 1e-5)
    sub_write(d, s)


def n5(d):
    s = read(d, 'submission_v224flip.csv')
    s['ACL'] = np.float32(0.5)
    sub_write(d, s)
    jedit(d, 's6f_receipt.json', lambda j: (j.update(mode='V224FLIP_CANARY', fired=['s6f->ACL']), j['blend'].update(mode='V224FLIP_CANARY')))
    logedit(d, lambda E: E + [{'stream_name': 'stdout', 'time': E[-1]['time'] + 1, 'data': 'S6F_CANARY_FIRED S6fall not used (test)\n'}])


def n6(d):
    jedit(d, 's6f_receipt.json', lambda j: j['run'].update(status='SKIPPED_GUARD'))
    logedit(d, lambda E: [dict(e, data=e['data'].replace('S6F_START ', 'S6F_SKIPPED start guard; S6F_START ')) for e in E])


def n7(d):
    p = os.path.join(d, 'submission_v224flip.csv')
    s = pd.read_csv(p, dtype={'StudyInstanceUID': str})
    s.loc[0, 'Effusion'] = np.float32(0.77777)
    s.to_csv(p, index=False, lineterminator='\n')


def n9(d):
    jedit(d, 's6f/s6f_driver_receipt.json', lambda j: j['env_isolation'].update(PYTHONPATH='/tmp/s6f_dec:/tmp/probe/mainline_code'))


def n11(d):
    def f(E):
        i = next(k for k, e in enumerate(E) if e['data'].startswith('S6F_SETUP_OK'))
        j = next(k for k, e in enumerate(E) if LAST in e['data'])
        e = E.pop(i)
        E.insert(j, e)
        return E
    logedit(d, f)


def n13(d):
    jedit(d, 's6f/B/part0/receipt.json', lambda j: j['sha256'].update({'compact_infer.py': '8401812cd63737ec99839d317c4037c4efb445d1baec892fa7481bb30399cc2d'}))


def n14(d):
    jedit(d, 's6f_receipt.json', lambda j: j['run'].update(g_start_h=5.75))


def n15(d):
    base = read(d, 'submission_v224flip.csv')
    B = read(d, 's6f/B/B_probs.csv')
    sub_write(d, H.recompute(base, B, w=0.3))


def n16(d):
    jedit(d, 's6f_setup_receipt.json', lambda j: j.update(dec_ok=False))
    jedit(d, 's6f_receipt.json', lambda j: j['setup'].update(dec_ok=False))
    logedit(d, lambda E: [dict(e, data=e['data'].replace('S6F_DEC_OK ', 'S6F_DEC_FALLBACK ')) for e in E])


def p1(d):
    bprobs_edit(d, lambda df: df[['StudyInstanceUID'] + H.LAB[::-1] + [c for c in df.columns if c.startswith('z|')]])


def env_leak(d, stock, sys_extra=()):
    """[F1] a CONSISTENT run whose S6fall import path carries an extra dir: the kernel PYTHONPATH the setup saw (stock) and / or the
    subprocess sys.path; every receipt copy agrees (setup file, s6f_receipt.setup, driver receipt, s6f_receipt.run.driver)."""
    pp = ':'.join(['/tmp/s6f_dec'] + list(stock))
    add = [x for x in stock if x not in STOCK] + list(sys_extra)

    def setup_(j):
        j.update(pythonpath_stock=list(stock), pythonpath_s6f=pp)

    def iso_(e):
        e['PYTHONPATH'] = pp
        e['sys_path'] = e['sys_path'][:2] + add + e['sys_path'][2:]                  # after '' / '/tmp/s6f' and '/tmp/s6f_dec'
        e['driver_sys_path'] = e['driver_sys_path'][:2] + add + e['driver_sys_path'][2:]
    jedit(d, 's6f_setup_receipt.json', setup_)
    jedit(d, 's6f_receipt.json', lambda j: (setup_(j['setup']), iso_(j['run']['driver']['env_isolation'])))
    jedit(d, 's6f/s6f_driver_receipt.json', lambda j: iso_(j['env_isolation']))


# ---- base-specific negatives -------------------------------------------------------------------------------------------------
def rm_guard_fired(d, keep_blank):
    """the base ran with RM_GUARD fired (Fracture 0.5 written by RM_GUARD) and the S6F blend either kept it (notebook rule) or not"""
    b = read(d, 'submission_v224flip.csv')
    b['Fracture'] = np.float32(0.5)
    frozen_write(d, b)
    jedit(d, 's6f_setup_receipt.json', lambda j: j.update(base_fired=['r224 mirror->Fracture']))
    jedit(d, 's6f_receipt.json', lambda j: j['setup'].update(base_fired=['r224 mirror->Fracture']))
    jedit(d, 'canary_receipt.json', lambda j: (j.setdefault('fired', []).append('r224 mirror->Fracture'), j.setdefault('rm_guard', {}).update(mirror_ok=False)))
    logedit(d, lambda E: [dict(e, data=e['data'].replace('RM_GUARD_CLEAN', 'RM_GUARD_FIRED')) for e in E])
    reblend(d, ['Fracture'] if keep_blank else [])
    jedit(d, 's6f_receipt.json', lambda j: j['blend'].update(base_blanked=['Fracture'] if keep_blank else []))


def hf_fallback(d, keep_blank):
    """the S stage fell back (HF_FALLBACK: H with Synovitis 0.5) and the S6F blend either kept the canary or not"""
    h = read(d, 'submission_V224.csv')
    h['Synovitis'] = np.float32(0.5)
    frozen_write(d, h)
    jedit(d, 's6f_setup_receipt.json', lambda j: j.update(base_fired=['fleet->Synovitis']))
    jedit(d, 's6f_receipt.json', lambda j: j['setup'].update(base_fired=['fleet->Synovitis']))
    jedit(d, 'hf_receipt.json', lambda j: j.update(mode='H_only', blanked=['Synovitis'], why='test'))
    jedit(d, 'v224f_receipt.json', lambda j: j.update(mode='H_only', fired=['fleet->Synovitis']))
    logedit(d, lambda E: [dict(e, data=e['data'].replace('HF_BLEND_OK', 'HF_FALLBACK')) for e in E])
    reblend(d, ['Synovitis'] if keep_blank else [])
    jedit(d, 's6f_receipt.json', lambda j: j['blend'].update(base_blanked=['Synovitis'] if keep_blank else []))


def s_dropped(d):
    """S6F blended on H (submission_V224.csv) instead of the HF output: the S part is missing from the frozen base"""
    frozen_write(d, read(d, 'submission_V224.csv'))
    reblend(d, [])


def s_npz_changed(d):
    p = os.path.join(d, '_fleet_0.npz')
    z = dict(np.load(p, allow_pickle=False))
    z['P'] = (z['P'] * 0.5).astype(z['P'].dtype)
    np.savez(p, **z)


def r224_raw_changed(d):
    p = os.path.join(d, 'raptor224_raw.csv')
    s = pd.read_csv(p, dtype={'StudyInstanceUID': str})
    s.loc[0, 'ACL'] = s.loc[0, 'ACL'] + 2e-3
    s.to_csv(p, index=False, lineterminator='\n')


shutil.rmtree(NEG, ignore_errors=True)
print('MOCK_S6F2 %s negatives' % KIND, flush=True)
mutate('N1_s6_prob_plus_2e-3', n1, must=('nartaa k1/B/B_probs.csv',))
mutate('N3_submission_cell_plus_1e-5', n3, must=('submission.csv == .8 pct',))
mutate('N5_canary_fired', n5, must=('canary not fired',))
mutate('N6_start_guard_skipped', n6, must=('canary not fired', 'run: RAN'))
mutate('N7_frozen_base_changed', n7, must=('frozen base output present', 'pass the UNCHANGED'))
mutate('N9_r7_dir_on_pythonpath', n9, must=('no R7 dir',))
mutate('N11_freeze_before_base_end', n11, must=('the freeze ran after', 'markers present and in cell order'))
mutate('N13_compact_infer_8401812c', n13, must=('worker receipts',))
mutate('N14_run_policy_G_5.75', n14, must=('deadline policy == build',))
mutate('N15_weight_0.3', n15, must=('submission.csv == .8 pct',))
mutate('N16_dec_fallback', n16, must=('DEC ok',))
# [F1] S-stage leaks into the S6fall subprocess (each passed the pre-F1 gate: v224s6f2_check.py.pre_f1)
mutate('N18_k12_fleet_leaked_into_kernel_PYTHONPATH', lambda d: env_leak(d, ['/tmp/k12_fleet'] + STOCK),
       must=('S6fall import path', 'S-stage dir'))
mutate('N19_k2_member_dir_on_subprocess_sys_path', lambda d: env_leak(d, STOCK, ['/tmp/k2_members_x7q1']), must=('S-stage dir',))
mutate('N20_S_dataset_dir_on_kernel_PYTHONPATH', lambda d: env_leak(d, STOCK + ['/kaggle/input/rsna-knee-cnxt-fleet-r7tb']),
       must=('S6fall import path',))
rec_g =json.load(open(BUILD, encoding='utf-8'))                    # N17: a receipt (same notebook) whose G is not the reviewed one
rec_g['policy']['g_start_h'] = 5.75
os.makedirs(NEG, exist_ok=True)
pg = os.path.join(NEG, '_tmp_rec_g575_%s.json' % KIND)
json.dump(rec_g, open(pg, 'w', encoding='utf-8'), indent=1)
d17 = os.path.join(NEG, 'N17_build_receipt_G_5.75')
shutil.rmtree(d17, ignore_errors=True)
shutil.copytree(MOCK, d17)
r = subprocess.run([sys.executable, '-I', GATE, d17, '--build', pg, '--kernel-dir', os.path.join(H.S6F2, 'kernel_' + TAG)],
                   capture_output=True, text=True, timeout=1800, encoding='utf-8', errors='replace')
fl = [l.strip()[5:] for l in r.stdout.splitlines() if l.strip().startswith('FAIL')]
v17 = ([l for l in r.stdout.splitlines() if l.startswith('V224S6F2_CHECK ')] or ['(none)'])[-1]
chk('NEGATIVE N17_build_receipt_G_5.75 (not the reviewed G for this base): %s' % v17, r.returncode != 0 and any('runtime-reviewed' in f for f in fl),
    ' || '.join(f[:70] for f in fl[:3]))
os.remove(pg)
if KIND == 'flrm':
    mutate('B1_rm_guard_fired_kept', lambda d: rm_guard_fired(d, True), must=('pass the UNCHANGED', 'base canaries clean'))
    mutate('B2_rm_guard_fired_unblanked', lambda d: rm_guard_fired(d, False), must=('submission.csv == .8 pct', 'base canaries clean'))
    mutate('B3_r224_raw_plus_2e-3', r224_raw_changed, must=('raptor224_raw.csv within',))
else:
    mutate('C1_hf_fallback_kept', lambda d: hf_fallback(d, True), must=('pass the UNCHANGED', 'base canaries clean'))
    mutate('C2_hf_fallback_unblanked', lambda d: hf_fallback(d, False), must=('submission.csv == .8 pct', 'base canaries clean'))
    mutate('C3_s6f_on_H_not_HF_output', s_dropped, must=('pass the UNCHANGED',))
    mutate('C4_S_npz_halved', s_npz_changed, must=('pass the UNCHANGED',))
mutate('P1_bprobs_columns_permuted', p1, expect_fail=False)
mutate('X1_clean_mock_with_expect_canary', lambda d: None, must=('control: S6F canary fired', 'control skip'), extra=('--expect-canary', '--control', 'skip'))
print('MOCK_S6F2 %s control mock (kernel age > G: start guard skip -> canary)' % KIND, flush=True)
build_mock(MOCK_C, KD['G'] + 0.1)
rc, v, out = gate(MOCK_C, '--expect-canary', '--control', 'skip')
open(os.path.join(HERE, 'mock_s6f2_gate_control_%s.txt' % KIND), 'w', encoding='utf-8').write(out)
chk('POSITIVE control: the canary mock (age G + .1 h) PASSES with --expect-canary --control skip', rc == 0 and 'PASS' in v, v)
rc, v, out = gate(MOCK_C)
chk('NEGATIVE control: the canary mock FAILS the normal gate', rc != 0 and 'FAIL' in v, v)
shutil.rmtree(NEG, ignore_errors=True)
shutil.rmtree(os.path.join(HERE, '_tmp_mock_' + KIND), ignore_errors=True)
print('MOCK_S6F2 %s %d/%d' % ('PASS' if all(R) else 'FAIL', sum(R), len(R)), flush=True)
sys.exit(0 if all(R) else 1)
