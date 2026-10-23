# mock_gate.py (10-15, Claude subagent, kcstage): tests of v224s6fkc_check.py on MOCK commit outputs made from the BOX CPU end-to-end runs
# of the built stand-in kernel's KC cells (box/e2e_cells.py: real driver, real serve wrapper, real side pass, real stand-in members, 5 RSNA
# train studies; pulled to tests/pulled_e2e/). Mock-only edits (documented, the CPU run cannot produce them): the driver receipt's test
# overrides -> {}, env n_gpu 2 and Kaggle-like import dirs, worker devices / closing lines cpu -> cuda:K, KC PYTHONPATH with /tmp/s6f_dec,
# and a JSON log (V224_DONE + S6F_DONE + the KC lines of the e2e log). The side reference = the side kernel's KAGGLE output for the same 5
# train studies (box ~/kc_store/side_train.csv rows), the box parity JSON = kc_box_parity.py on the e2e output (CPU vs CPU here).
# KCS2 (C9 on the served path, A6 G3): K0-K13 below; the C9 JSON is the box run of box/c9_served.py with the 2 stand-in members on 20
# box train studies (tests/pulled_e2e/c9_served_standin_n20.json); 'real build' = the stand-in build receipt with kc_dataset standin False
# / kind kc1 (the gate's only switch for the C9 requirement). K0 runs the PRE-FIX gate (bak_kcs2/v224s6fkc_check.py): fail-before.
# Base gate: --skip-base-gate here (needs the real S6F commit output); tested separately in M2 on the real S6F v1 output.
#   py -3.12 -I tests/mock_gate.py   -> MOCK_GATE PASS n/m
import os, sys, json, glob, shutil, tempfile, subprocess, hashlib
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KS = os.path.dirname(HERE)
SESS = os.path.dirname(KS)
PULL = os.path.join(HERE, 'pulled_e2e')
GATE = os.path.join(KS, 'v224s6fkc_check.py')
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
       'Contusion', 'Fracture']
TMP = tempfile.mkdtemp(prefix='kcmock_')
R = []


def chk(name, ok, info=''):
    R.append(bool(ok))
    print('  %s %s %s' % ('OK  ' if ok else 'FAIL', name, info), flush=True)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def jl(p):
    return json.load(open(p, encoding='utf-8'))


def jd(o, p):
    json.dump(o, open(p, 'w', encoding='utf-8'), indent=1)


def make_mock(case_dir, tag, name, kc_log):
    out = os.path.join(TMP, name, 'out')
    shutil.copytree(os.path.join(case_dir, 'working'), out)
    lines = ['V224_DONE {"fired": []}', 'S6F_DONE {"mode": "S6F", "fired": [], "total_age_h": 5.99}'] + \
        [l for l in open(kc_log, encoding='utf-8').read().splitlines() if l.startswith('KC_')]
    jd([{'stream_name': 'stdout', 'time': float(i), 'data': l + '\n'} for i, l in enumerate(lines)], os.path.join(out, 'rsna-sub-%s.log' % tag))
    st = jl(os.path.join(out, 'kc_setup_receipt.json'))
    if st.get('ok'):
        st['pythonpath_kc'] = '/tmp/kc/code:/tmp/s6f_dec:/kaggle/lib/kagglegym:/kaggle/lib:/kaggle/input/competitions/rsna-knee-abnormality-detection'
    jd(st, os.path.join(out, 'kc_setup_receipt.json'))
    rec = jl(os.path.join(out, 'kc_receipt.json'))
    rec['setup'] = st
    dp = os.path.join(out, 'kc', 'kc_driver_receipt.json')
    if os.path.isfile(dp):
        d = jl(dp)
        d['overrides'] = {}
        e = d['env']
        e['n_gpu'] = 2
        e['gpus'] = ['Tesla T4', 'Tesla T4']
        for m in ('pylibjpeg', 'libjpeg', 'openjpeg'):
            e['mods'][m] = ['2.x', '/tmp/s6f_dec']
        for m in ('kc_common', 'kc_worker'):
            e['mods'][m] = ['', '/tmp/kc/code']
        for m in ('torch', 'timm', 'pydicom', 'numpy', 'pandas', 'cv2'):
            e['mods'][m] = [e['mods'][m][0], '/usr/local/lib/python3.13/dist-packages']
        for k, w in enumerate(d['workers']):
            w['device'] = 'cuda:%d' % k
            w['closing'] = w['closing'].replace('device cpu', 'device cuda:%d' % k)
        jd(d, dp)
        rec['run']['driver'] = {k: d.get(k) for k in rec['run']['driver']}
    jd(rec, os.path.join(out, 'kc_receipt.json'))
    return out


def gate(out, extra=(), sample=None, s6f_out=None, build=None, parity=None, gate_py=GATE):
    cmd = [sys.executable, '-I', gate_py, out, '--skip-base-gate', '--sample', sample, '--side-ref', SIDE_REF, '--s6f-out', s6f_out,
           '--build', build] + (['--box-parity', parity] if parity else []) + list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    last = [l for l in r.stdout.splitlines() if l.startswith('V224S6FKC_CHECK ')][-1:] or ['(none)']
    fails = [l.strip()[5:80] for l in r.stdout.splitlines() if l.strip().startswith('FAIL')]
    return r.returncode, last[0], fails, r.stdout


clean = os.path.join(PULL, 'clean')
ids = pd.read_csv(os.path.join(clean, 'working', 'submission.csv'), dtype=str)['StudyInstanceUID'].tolist()
SAMPLE = os.path.join(TMP, 'sample_submission.csv')
pd.DataFrame({'StudyInstanceUID': ids, **{c: 0.5 for c in LAB}}).to_csv(SAMPLE, index=False)
SIDE_REF = os.path.join(PULL, 'side_ref_kaggle_n5.csv')
S6 = os.path.join(TMP, 's6f_out')
os.makedirs(S6)
shutil.copyfile(os.path.join(clean, 'working', 'submission_s6f.csv'), os.path.join(S6, 'submission.csv'))
BUILD = os.path.join(KS, 'build_receipt_v224s6fkcstandin.json')
BUILD_CTL = os.path.join(KS, 'build_receipt_v224s6fkcstandinskip.json')
PAR = os.path.join(PULL, 'clean_box_parity.json')
M = make_mock(clean, 'v224s6fkcstandin', 'M', os.path.join(PULL, 'clean.log'))
rc, last, fails, txt = gate(M, sample=SAMPLE, s6f_out=S6, build=BUILD, parity=PAR)
open(os.path.join(HERE, 'mock_gate_positive.txt'), 'w', encoding='utf-8').write(txt)
chk('P1 mock commit output (KC clean, box e2e) PASS', rc == 0 and ' PASS ' in last, (last, fails))


def neg(name, mutate, expect_word):
    d = os.path.join(TMP, name.split()[0])
    out = os.path.join(d, 'out')
    shutil.copytree(M, out)
    par = os.path.join(d, 'parity.json')
    shutil.copyfile(PAR, par)
    mutate(out, par)
    rc, last, fails, _ = gate(out, sample=SAMPLE, s6f_out=S6, build=BUILD, parity=par)
    hit = any(expect_word.lower() in f.lower() for f in fails)
    chk('%s -> FAIL on "%s"' % (name, expect_word), rc != 0 and ' FAIL ' in last and hit, (last, fails[:4]))


def m_probs(out, par):
    p = os.path.join(out, 'kc', 'kc_probs.csv'); k = pd.read_csv(p, dtype={'StudyInstanceUID': str}); k.loc[0, 'MCL'] += 2e-3; k.to_csv(p, index=False)


def m_sub(out, par):
    p = os.path.join(out, 'submission.csv'); s = pd.read_csv(p, dtype={'StudyInstanceUID': str}); s.loc[1, 'MCL'] += 1e-5; s.to_csv(p, index=False)


def m_cov(out, par):
    d = jl(os.path.join(out, 'kc', 'kc_driver_receipt.json')); d['B']['coverage'] = 0.8; d['B']['n_unscored'] = 1
    jd(d, os.path.join(out, 'kc', 'kc_driver_receipt.json'))


def m_canary(out, par):
    r = jl(os.path.join(out, 'kc_receipt.json')); r['mode'] = 'KC_CANARY'; r['fired'] = ["kc->Baker's"]; jd(r, os.path.join(out, 'kc_receipt.json'))


def m_override(out, par):
    d = jl(os.path.join(out, 'kc', 'kc_driver_receipt.json')); d['overrides'] = {'KC_DEVICE': 'cpu'}; jd(d, os.path.join(out, 'kc', 'kc_driver_receipt.json'))


def m_policy(out, par):
    r = jl(os.path.join(out, 'kc_receipt.json')); r['run']['g_kc_h'] = 7.2; jd(r, os.path.join(out, 'kc_receipt.json'))


def m_side(out, par):
    p = os.path.join(out, 'kc', 'kc_side_test.csv'); t = pd.read_csv(p, dtype={'StudyInstanceUID': str, 'tag': str})
    t.loc[0, 'side'] = 'R' if t.loc[0, 'side'] == 'L' else 'L'; t.to_csv(p, index=False)


def m_parity(out, par):
    j = jl(par); j['max_dp'] = 2e-3; j['PASS'] = False; jd(j, par)


def m_dump(out, par):
    os.remove(sorted(glob.glob(os.path.join(out, 'kc', 'dump', '*.npz')))[0])


def m_order(out, par):
    lp = glob.glob(os.path.join(out, '*.log'))[0]; E = jl(lp)
    i = next(k for k, e in enumerate(E) if e['data'].startswith('S6F_DONE')); e = E.pop(i); E.append(e); jd(E, lp)


def m_baker(out, par):
    p = os.path.join(out, 'submission.csv'); s = pd.read_csv(p, dtype={'StudyInstanceUID': str}); s["Baker's"] = 0.5; s.to_csv(p, index=False)


def m_device(out, par):
    d = jl(os.path.join(out, 'kc', 'kc_driver_receipt.json')); d['workers'][1]['device'] = 'cuda:0'; jd(d, os.path.join(out, 'kc', 'kc_driver_receipt.json'))


def m_trace(out, par):
    lp = glob.glob(os.path.join(out, '*.log'))[0]; E = jl(lp); E.append({'stream_name': 'stderr', 'time': 99.0, 'data': 'Traceback (most recent call last):\n'}); jd(E, lp)


def m_base(out, par):
    p = os.path.join(out, 'submission_s6f.csv'); s = pd.read_csv(p, dtype={'StudyInstanceUID': str}); s.loc[0, 'ACL'] = 0.123; s.to_csv(p, index=False)


def m_member(out, par):
    j = jl(par); k = sorted(j['members'])[0]; j['members'][k] = '0' * 64; jd(j, par)


for nm, fn, word in (('N1 kc_probs +2e-3', m_probs, 'box parity inputs'), ('N2 submission +1e-5', m_sub, 'submission.csv == .8'),
                     ('N3 coverage .8', m_cov, 'coverage 1.0'), ('N4 KC canary fired', m_canary, 'canary not fired'),
                     ('N5 test override in the receipt', m_override, 'driver receipt ok'), ('N6 G_KC 7.2 in the run', m_policy, 'policy == build'),
                     ('N7 side table L/R swapped', m_side, 'side table'), ('N8 box parity 2e-3', m_parity, 'box parity PASS'),
                     ('N9 one dump missing', m_dump, 'box parity inputs'), ('N10 freeze before S6F_DONE', m_order, 'freeze ran after'),
                     ("N11 Baker's blanked", m_baker, 'constant column'), ('N12 both workers on cuda:0', m_device, 'one process per T4'),
                     ('N13 Traceback in the log', m_trace, 'Traceback'), ('N14 frozen base changed', m_base, 'frozen base'),
                     ('N15 box parity on another member file', m_member, 'box parity inputs')):
    neg(nm, fn, word)
# control (skip): the e2e run of the --g-kc-h 0 build
C = make_mock(os.path.join(PULL, 'ctl_guard'), 'v224s6fkcstandinskip', 'C', os.path.join(PULL, 'ctl_guard.log'))
rc, last, fails, txt = gate(C, extra=['--expect-canary'], sample=SAMPLE, s6f_out=S6, build=BUILD_CTL)
open(os.path.join(HERE, 'mock_gate_control.txt'), 'w', encoding='utf-8').write(txt)
chk('C1 control skip mock PASS with --expect-canary', rc == 0 and ' PASS ' in last, (last, fails))
rc, last, fails, _ = gate(C, sample=SAMPLE, s6f_out=S6, build=BUILD_CTL)
chk('C2 control mock FAILS the normal gate', rc != 0 and ' FAIL ' in last, last)
rc, last, fails, _ = gate(M, extra=['--expect-canary'], sample=SAMPLE, s6f_out=S6, build=BUILD)
chk('C3 clean mock FAILS --expect-canary', rc != 0 and ' FAIL ' in last, last)
# ---- KCS2: C9 on the served path (A6 G3); required for a real KC1 build, checked for a stand-in build only when supplied ----
C9J = os.path.join(PULL, 'c9_served_standin_n20.json')
BUILD_REAL = os.path.join(TMP, 'build_receipt_real.json')
_b = jl(BUILD)
_b['kc_dataset']['standin'] = False
_b['kc_dataset']['kind'] = 'kc1'
jd(_b, BUILD_REAL)
OLD_GATE = os.path.join(KS, 'bak_kcs2', 'v224s6fkc_check.py')


def c9case(name, mutate, expect_pass, word=None, build=BUILD_REAL, gate_py=GATE):
    d = os.path.join(TMP, 'c9_' + name.split()[0])
    out = os.path.join(d, 'out')
    shutil.copytree(M, out)
    if mutate is not None:
        j = jl(C9J)
        mutate(j)
        jd(j, os.path.join(out, 'c9_served.json'))
    rc, last, fails, _ = gate(out, sample=SAMPLE, s6f_out=S6, build=build, parity=PAR, gate_py=gate_py)
    if expect_pass:
        ok = rc == 0 and ' PASS ' in last
    else:
        ok = rc != 0 and ' FAIL ' in last and any(word.lower() in f.lower() for f in fails)
    chk('%s -> %s' % (name, 'PASS' if expect_pass else 'FAIL on "%s"' % word), ok, (last, fails[:3]))


def _rows(j):
    return j['kpp']['rows']


def k_hdr(j):
    j['path'] = 'header'; j['substitution'] = None; j['substitution_live'] = False


def k_dp(j):
    _rows(j)[3]['max_dp'] = 2e-3


def k_n19(j):
    j['kpp']['rows'] = _rows(j)[:19]


def k_noweak(j):
    for r in _rows(j):
        r['weak'] = 0


def k_noabs(j):
    for r in _rows(j):
        r['absent'] = []


def k_mem(j):
    j['members'][0]['sha256'] = '0' * 64


def k_side(j):
    j['side_equal_ref'] = False


def k_serve(j):
    j['serve_sha256'] = '1' * 64


def k_man(j):
    j['manifest_sha256'] = '2' * 64


c9case('K0 PRE-FIX gate, real build, no C9 (the KCS2 gap: it passes)', None, True, gate_py=OLD_GATE)
c9case('K1 real build, no C9', None, False, 'C9 PASS')
c9case('K2 real build + the C9 JSON of the box run', lambda j: None, True)
for nm, fn, word in (('K3 C9 run on the header path (the pre-fix tool)', k_hdr, 'C9 PASS'), ('K4 one C9 row |dp| 2e-3', k_dp, 'C9 PASS'),
                     ('K5 C9 on 19 studies', k_n19, 'C9 PASS'), ('K6 C9 without a weak-zone study', k_noweak, 'C9 PASS'),
                     ('K7 C9 without a SagPD-absent study', k_noabs, 'C9 PASS'), ('K8 C9 on another member file', k_mem, 'C9 member files'),
                     ('K9 C9 side rows != the store side table', k_side, 'C9 PASS'), ('K10 C9 with another kc_serve.py', k_serve, 'C9 PASS'),
                     ('K11 C9 against another manifest', k_man, 'C9 member files')):
    c9case(nm, fn, False, word)
c9case('K12 stand-in build + a valid C9 (checked when supplied)', lambda j: None, True, build=BUILD)
c9case('K13 stand-in build + a failing C9 (|dp| 2e-3)', k_dp, False, 'C9 PASS', build=BUILD)
print('MOCK_GATE %s %d/%d' % ('PASS' if all(R) else 'FAIL', sum(R), len(R)), flush=True)
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(0 if all(R) else 1)
