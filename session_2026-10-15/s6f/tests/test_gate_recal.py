#!/usr/bin/env python
# test_gate_recal.py (10-15 rfix, Claude subagent; findings S6F-R2, S6F-R3): the gate's runtime-recalibration and J lines.
# Offline, laptop CPU, -I. Inputs: the mock commit output (tests/mock_out_v224flips6f = nartaa's REAL part outputs through our
# driver, written by mock_s6f_output.py) and, when present, the real GPU commit output out_v224flips6f_v1 (data only; copied without
# *.py / __pycache__). Every mutated copy goes to a temp dir.
#   R2  G1 mock (identical pipeline): RECAL warm GPU x1.00, warm loader x1.00, RECAL_TRIGGER no; the MICROBENCH line no longer says
#          "feed sim ... C_LO / C_HI" and prints nartaa's same-field totals (10.433 / 26.867 / 50.5)
#       G2 real commit: RECAL_TRIGGER no (while its cold-start gpu_s_per_study / 7.3 s reads ~1.45x, the old rule's near-false trigger)
#       G3 warm study GPU x1.6 -> trigger yes      G4 warm study decode x1.6 (GPU unchanged; the old rule watched GPU only) -> yes
#       G5 cold first study GPU x3 -> trigger no (cold start is not c)      G6 B stage 2.1x nartaa's 64.1 s -> yes
#       every case: gate verdict PASS 31/31 (RECAL is info, never a gate failure)
#   R3  G7 PICK_ELIGIBLE_J no on the mock and the real output (cells 0-14 = V224-FLIP, no base DEC cell)
#       G8 a base DEC_OK line before V224_DONE -> yes;  DEC_FALLBACK -> no
#   usage: py -3.12 -I tests/test_gate_recal.py [--gate PATH]  -> TEST_GATE_RECAL PASS/FAIL n/m
import os, sys, json, shutil, argparse, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
S6F = os.path.dirname(HERE)
ap = argparse.ArgumentParser()
ap.add_argument('--gate', default=os.path.join(S6F, 'v224s6f_check.py'))
A = ap.parse_args()
MOCK = os.path.join(HERE, 'mock_out_v224flips6f')
REAL = os.path.join(S6F, 'out_v224flips6f_v1')
T = os.path.join(HERE, '_tmp_recal')
shutil.rmtree(T, ignore_errors=True)
os.makedirs(T)
R = []


def chk(name, ok, info=''):
    R.append(bool(ok))
    print('  %s %s %s' % ('OK  ' if ok else 'FAIL', name, info), flush=True)


def gate(d):
    r = subprocess.run([sys.executable, '-I', A.gate, d], capture_output=True, text=True, timeout=900)
    lines = r.stdout.splitlines()
    v = [l for l in lines if l.startswith('V224S6F_CHECK ')][-1:] or ['(no verdict)']
    get = lambda key: ([l.strip() for l in lines if l.strip().startswith(key)] or [''])[-1]
    return {'rc': r.returncode, 'v': v[-1], 'recal': get('info RECAL like'), 'trig': get('info RECAL_TRIGGER'),
            'mb': get('info MICROBENCH'), 'j': get('info PICK_ELIGIBLE_J')}


def copy(src, name):
    d = os.path.join(T, name)
    shutil.copytree(src, d, ignore=shutil.ignore_patterns('*.py', '__pycache__'))
    return d


def jedit(p, fn):
    j = json.load(open(p, encoding='utf-8'))
    fn(j)
    json.dump(j, open(p, 'w', encoding='utf-8'), indent=1)


def per_study_edit(d, part, idx, field, factor):              # per_study item = [uid, windows, decode s, render s, gpu s]
    k = {'decode': 2, 'gpu': 4}[field]
    jedit(os.path.join(d, 's6f', 'B', part, 'receipt.json'), lambda j: j['per_study'][idx].__setitem__(k, round(j['per_study'][idx][k] * factor, 3)))


def passed(g):
    return g['rc'] == 0 and g['v'].startswith('V224S6F_CHECK PASS 31/31')


if not os.path.isdir(MOCK):
    chk('mock output present (run mock_s6f_output.py first)', False, MOCK)
else:
    print('R2 RECAL', flush=True)
    g = gate(MOCK)
    chk('G1 mock: PASS 31/31; RECAL warm GPU x1.00, warm loader x1.00, RECAL_TRIGGER no', passed(g) and 'warm GPU 7.32 vs 7.32 s (x1.00)' in g['recal']
        and '(x1.00) | B stage' in g['recal'] and g['trig'] == 'info RECAL_TRIGGER no', g['recal'][-170:])
    chk('G1 mock: MICROBENCH not labelled as sim input, prints nartaa\'s same-field totals', 'C_LO' not in g['mb'] and 'NOT the sim c' in g['mb']
        and '"gpu_s_per_study": 10.433' in g['mb'] and '"s_per_study_per_gpu_max": 50.5' in g['mb'], g['mb'][:150])
    for name, edits, want in (('G3 warm study GPU x1.6', [('part0', 1, 'gpu', 1.6)], 'yes'),
                              ('G4 warm study decode x1.6 (GPU unchanged)', [('part0', 1, 'decode', 1.6)], 'yes'),
                              ('G5 cold first studies GPU x3', [('part0', 0, 'gpu', 3.0), ('part1', 0, 'gpu', 3.0)], 'no')):
        d = copy(MOCK, name.split()[0])
        for e in edits:
            per_study_edit(d, *e)
        g = gate(d)
        chk('%s -> RECAL_TRIGGER %s; gate still PASS 31/31' % (name, want), passed(g) and g['trig'].startswith('info RECAL_TRIGGER ' + want),
            g['recal'][-120:] + ' | ' + g['trig'][:60])
    d = copy(MOCK, 'G6')
    jedit(os.path.join(d, 's6f', 's6f_driver_receipt.json'), lambda j: j['stages']['B'].__setitem__('seconds', round(64.1 * 2.1, 1)))
    g = gate(d)
    chk('G6 B stage 2.1x nartaa -> RECAL_TRIGGER yes; gate still PASS 31/31', passed(g) and g['trig'].startswith('info RECAL_TRIGGER yes'),
        g['recal'][-60:])
    print('R3 PICK_ELIGIBLE_J', flush=True)
    g = gate(MOCK)
    chk('G7 mock: PICK_ELIGIBLE_J no (no base DEC cell), READ only', g['j'].startswith('info PICK_ELIGIBLE_J no') and 'NOT pick-eligible' in g['j'],
        g['j'][:90])
    for name, line, want in (('G8 base DEC_OK before V224_DONE', 'DEC_OK {"versions": {"pylibjpeg": "2.1.0"}}\n', 'yes'),
                             ('G8 base DEC_FALLBACK before V224_DONE', 'DEC_FALLBACK {"why": "test"}\n', 'no')):
        d = copy(MOCK, 'G8_' + line.split()[0])
        lp = os.path.join(d, 'rsna-sub-v224flips6f.log')
        E = json.load(open(lp, encoding='utf-8'))
        E.insert(1, {'stream_name': 'stdout', 'time': E[0]['time'] + 0.001, 'data': line})
        json.dump(E, open(lp, 'w', encoding='utf-8'))
        g = gate(d)
        chk('%s -> PICK_ELIGIBLE_J %s; gate still PASS 31/31' % (name, want), passed(g) and g['j'].startswith('info PICK_ELIGIBLE_J ' + want), g['j'][:70])
if os.path.isdir(REAL):
    d = copy(REAL, 'real')
    g = gate(d)
    mb = json.load(open(os.path.join(d, 's6f', 's6f_driver_receipt.json')))['B']['microbench']
    chk('G2 real commit: PASS 31/31, RECAL_TRIGGER no; PICK_ELIGIBLE_J no (old-rule reading gpu_s_per_study / 7.3 s = %.2f)' % (mb['gpu_s_per_study'] / 7.3),
        passed(g) and g['trig'] == 'info RECAL_TRIGGER no' and g['j'].startswith('info PICK_ELIGIBLE_J no'), g['recal'][-150:])
else:
    print('  info no real commit output at %s (G2 skipped)' % REAL, flush=True)
shutil.rmtree(T, ignore_errors=True)
print('TEST_GATE_RECAL %s %d/%d' % ('PASS' if all(R) else 'FAIL', sum(R), len(R)), flush=True)
sys.exit(0 if all(R) else 1)
