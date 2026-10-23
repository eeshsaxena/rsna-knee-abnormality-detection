#!/usr/bin/env python
# mock_s6f_output.py (10-15, Claude subagent): a MOCK V224-FLIP-S6F commit output, then the gate on it and on mutated copies.
# Offline, laptop CPU, -I. Built from REAL outputs only:
#   - the V224-FLIP v1 commit output (session_2026-10-14/v224/out_v224flip_v1: stage csvs, receipts, worker logs, kernel log),
#   - the appended cells of the built notebook EXECUTED as written (s6f_harness: paths redirected) with the real driver port on
#     nartaa's REAL commit part outputs (ref_output/k1/B/part0, part1) -> submission_v224flip.csv, s6f/, receipts, submission.csv;
#   - the kernel log = the V224-FLIP log + the lines our cells and the driver print, in run order.
# Two mock-only patches (stated): the DEC branch cannot run on this Windows laptop (manylinux cp313 wheels), so the setup receipt /
# its log line are set to the Kaggle DEC_OK form (DEC proven on Kaggle CPU by the V224-J kernel); the env-isolation probe is the
# harness's Kaggle-like stand-in. Then: positive PASS, 16 negatives (each must FAIL, on the intended check), 2 positives
# (permuted B_probs columns; control mock with --expect-canary --control skip), 1 cross negative (clean mock with --expect-canary).
# [rfix S6F-R1] the canary gate now also checks the control's path (skip: SKIPPED_GUARD); kill-control cases: test_controls.py.
#   usage: py -3.12 -I tests/mock_s6f_output.py  -> MOCK_S6F PASS/FAIL n/m (prints every gate run's verdict line)
import os, sys, io, json, time, shutil, hashlib, subprocess, contextlib
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s6f_harness as H

GATE = os.path.join(H.S6F, 'v224s6f_check.py')
MOCK = os.path.join(HERE, 'mock_out_v224flips6f')
MOCK_C = os.path.join(HERE, 'mock_out_v224flips6f_canary')
NEG = os.path.join(HERE, '_neg')
R = []


def chk(name, ok, info=''):
    R.append(bool(ok))
    print('  %s %s %s' % ('OK  ' if ok else 'FAIL', name, info), flush=True)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def build_mock(dst, age_h):
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(H.FLIP_OUT, dst, ignore=shutil.ignore_patterns('__pycache__'))
    os.replace(os.path.join(dst, 'rsna-sub-v224flip.log'), os.path.join(dst, 'rsna-sub-%s.log' % H.TAG))
    t = H.Tree(os.path.join(HERE, '_tmp_mock'), os.path.basename(dst))
    base = H.flip_base()
    ns, text = H.run_cells(t, base, age_h=age_h, mode='port', base_file=os.path.join(H.FLIP_OUT, 'submission.csv'))
    # mock-only patch 1: the Kaggle DEC_OK form of the setup receipt + line
    dec = {'target': '/tmp/s6f_dec', 'pip_rc': 0, 'selftest': {'pydicom': '3.0.1', 'versions': {'pylibjpeg': '2.1.0', 'libjpeg': '2.4.0', 'openjpeg': '2.6.0'},
           'decode': {'JLL57_12U': {'exact': True}, 'JLL70_16S': {'exact': True}, 'U_16S': {'exact': True}}}}
    for p in (os.path.join(t.work, 's6f_setup_receipt.json'),):
        r = json.load(open(p))
        r.update(dec_ok=True, dec=dec, pythonpath_s6f='/tmp/s6f_dec')
        json.dump(r, open(p, 'w'), indent=1)
    rr = json.load(open(os.path.join(t.work, 's6f_receipt.json')))
    rr['setup'].update(dec_ok=True, dec=dec, pythonpath_s6f='/tmp/s6f_dec')
    json.dump(rr, open(os.path.join(t.work, 's6f_receipt.json'), 'w'), indent=1)
    lines = []
    for ln in text.splitlines():
        if ln.startswith('S6F_DEC_FALLBACK '):
            ln = 'S6F_DEC_OK ' + json.dumps({'why': None, 'pip_rc': 0, 'pydicom': '3.0.1', 'versions': dec['selftest']['versions'],
                                             'decode': {k: True for k in dec['selftest']['decode']}})
        if ln.startswith('S6F_SETUP_OK '):
            ln = ln.replace('"dec_ok": false', '"dec_ok": true').replace('"pythonpath_s6f": null', '"pythonpath_s6f": "/tmp/s6f_dec"')
        lines.append(ln)
        if ln.startswith('S6F_START '):
            dl = os.path.join(t.work, 's6f', 's6f_driver_log.txt')
            if os.path.isfile(dl):
                lines += open(dl, encoding='utf-8').read().splitlines()
    for f in ('submission.csv', 'submission_v224flip.csv', 's6f_setup_receipt.json', 's6f_receipt.json'):
        shutil.copyfile(os.path.join(t.work, f), os.path.join(dst, f))
    if os.path.isdir(os.path.join(t.work, 's6f')):
        shutil.copytree(os.path.join(t.work, 's6f'), os.path.join(dst, 's6f'))
    lp = os.path.join(dst, 'rsna-sub-%s.log' % H.TAG)
    E = json.load(open(lp, encoding='utf-8'))
    t_end = E[-1]['time'] if E else 0.0
    E += [{'stream_name': 'stdout', 'time': t_end + 1 + 0.01 * i, 'data': ln + '\n'} for i, ln in enumerate(lines)]
    json.dump(E, open(lp, 'w', encoding='utf-8'))
    shutil.rmtree(t.t, ignore_errors=True)
    return text


def gate(d, *extra):
    r = subprocess.run([sys.executable, '-I', GATE, d] + list(extra), capture_output=True, text=True, timeout=900)
    v = [l for l in r.stdout.splitlines() if l.startswith('V224S6F_CHECK ')][-1:] or ['(no verdict) ' + r.stderr[-300:]]
    return r.returncode, v[-1], r.stdout


print('MOCK_S6F building the mock commit output', flush=True)
build_mock(MOCK, 4.6)
rc, v, out = gate(MOCK)
print(out, flush=True)
open(os.path.join(HERE, 'mock_s6f_gate_positive.txt'), 'w', encoding='utf-8').write(out)
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
    p = os.path.join(d, 'rsna-sub-%s.log' % H.TAG)
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


def n1(d):
    def f(df):
        df.loc[0, 'Effusion'] = df.loc[0, 'Effusion'] + 2e-3
        return df
    bprobs_edit(d, f)


def n2(d):
    def f(df):
        a, b = df['Medial Meniscus'].copy(), df['Lateral Meniscus'].copy()
        df['Medial Meniscus'], df['Lateral Meniscus'] = b, a
        return df
    bprobs_edit(d, f)


def n3(d):
    s = pd.read_csv(os.path.join(d, 'submission.csv'), dtype={'StudyInstanceUID': str})
    s.loc[1, 'PF OA'] = np.float32(s.loc[1, 'PF OA'] + 1e-5)
    sub_write(d, s)


def n4(d):
    def f(df):
        df.loc[2, [c for c in df.columns if c != 'StudyInstanceUID']] = np.nan
        return df
    bprobs_edit(d, f)
    jedit(d, 's6f/s6f_driver_receipt.json', lambda j: j['B'].update(coverage=2 / 3, n_unscored=1, scored=2))
    jedit(d, 's6f_receipt.json', lambda j: j['blend'].update(coverage=2 / 3, n_unscored=1))


def n5(d):
    s = pd.read_csv(os.path.join(d, 'submission_v224flip.csv'), dtype={'StudyInstanceUID': str})
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
    s.loc[0, 'Synovitis'] = np.float32(0.77777)
    s.to_csv(p, index=False, lineterminator='\n')


def n8(d):
    jedit(d, 's6f/s6f_driver_receipt.json', lambda j: j['env'].pop('timm'))


def n9(d):
    jedit(d, 's6f/s6f_driver_receipt.json', lambda j: j['env_isolation'].update(PYTHONPATH='/tmp/s6f_dec:/tmp/probe/mainline_code'))


def n10(d):
    jedit(d, 's6f/s6f_driver_receipt.json', lambda j: j.update(ok=False, error='K1Error(test)'))
    jedit(d, 's6f_receipt.json', lambda j: j['run']['driver'].update(ok=False))


def n11(d):
    def f(E):
        i = next(k for k, e in enumerate(E) if e['data'].startswith('S6F_SETUP_OK'))
        j = next(k for k, e in enumerate(E) if 'V224_DONE' in e['data'])
        e = E.pop(i)
        E.insert(j, e)
        return E
    logedit(d, f)


def n12(d):
    jedit(d, 's6f/s6f_driver_receipt.json', lambda j: j.update(code_manifest_pin='0' * 64))


def n13(d):
    jedit(d, 's6f/B/part0/receipt.json', lambda j: j['sha256'].update({'compact_infer.py': '8401812cd63737ec99839d317c4037c4efb445d1baec892fa7481bb30399cc2d'}))


def n14(d):
    jedit(d, 's6f_receipt.json', lambda j: j['run'].update(g_start_h=6.0))


def n15(d):
    base = pd.read_csv(os.path.join(d, 'submission_v224flip.csv'), dtype={'StudyInstanceUID': str})
    B = pd.read_csv(os.path.join(d, 's6f', 'B', 'B_probs.csv'), dtype={'StudyInstanceUID': str})
    sub_write(d, H.recompute(base, B, w=0.3))


def n16(d):
    jedit(d, 's6f_setup_receipt.json', lambda j: j.update(dec_ok=False))
    jedit(d, 's6f_receipt.json', lambda j: j['setup'].update(dec_ok=False))
    logedit(d, lambda E: [dict(e, data=e['data'].replace('S6F_DEC_OK ', 'S6F_DEC_FALLBACK ')) for e in E])


def p1(d):
    bprobs_edit(d, lambda df: df[['StudyInstanceUID'] + H.LAB[::-1] + [c for c in df.columns if c.startswith('z|')]])


shutil.rmtree(NEG, ignore_errors=True)
print('MOCK_S6F negatives', flush=True)
mutate('N1_s6_prob_plus_2e-3', n1, must=('nartaa k1/B/B_probs.csv',))
mutate('N2_meniscus_names_swapped', n2, must=('nartaa k1/B/B_probs.csv',))
mutate('N3_submission_cell_plus_1e-5', n3, must=('submission.csv == .8 pct',))
mutate('N4_coverage_2_of_3', n4, must=('coverage 1.0',))
mutate('N5_canary_fired', n5, must=('canary not fired',))
mutate('N6_start_guard_skipped', n6, must=('canary not fired', 'run: RAN'))
mutate('N7_frozen_base_changed', n7, must=('frozen V224-FLIP output present', 'UNCHANGED V224-FLIP gate'))
mutate('N8_receipt_without_timm', n8, must=('versions in the receipt',))
mutate('N9_r7_dir_on_pythonpath', n9, must=('no R7 dir',))
mutate('N10_driver_not_ok', n10, must=('driver receipt ok',))
mutate('N11_freeze_before_v224_done', n11, must=('freeze ran after the V224 canary', 'markers present and in cell order'))
mutate('N12_code_manifest_pin', n12, must=('code manifest',))
mutate('N13_compact_infer_8401812c', n13, must=('worker receipts',))
mutate('N14_policy_G_6.0', n14, must=('deadline policy == build',))
mutate('N15_weight_0.3', n15, must=('submission.csv == .8 pct',))
mutate('N16_dec_fallback', n16, must=('DEC ok',))
mutate('P1_bprobs_columns_permuted', p1, expect_fail=False)
mutate('X1_clean_mock_with_expect_canary', lambda d: None, must=('control: S6F canary fired', 'control skip'), extra=('--expect-canary', '--control', 'skip'))
print('MOCK_S6F control mock (kernel age 6.0 h > G: start guard skip -> canary)', flush=True)
build_mock(MOCK_C, 6.0)
rc, v, out = gate(MOCK_C, '--expect-canary', '--control', 'skip')        # [rfix S6F-R1] the receipt of the submit tag names no control
open(os.path.join(HERE, 'mock_s6f_gate_control.txt'), 'w', encoding='utf-8').write(out)
chk('POSITIVE control: the canary mock PASSES with --expect-canary', rc == 0 and 'PASS' in v, v)
rc, v, out = gate(MOCK_C)
chk('NEGATIVE control: the canary mock FAILS the normal gate', rc != 0 and 'FAIL' in v, v)
shutil.rmtree(NEG, ignore_errors=True)
shutil.rmtree(os.path.join(HERE, '_tmp_mock'), ignore_errors=True)
print('MOCK_S6F %s %d/%d' % ('PASS' if all(R) else 'FAIL', sum(R), len(R)), flush=True)
sys.exit(0 if all(R) else 1)
