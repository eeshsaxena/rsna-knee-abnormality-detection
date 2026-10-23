# test_fix1015.py (session_2026-10-15/v224f/tests, 10-15, Claude fix agent): tests for the 10-15 fixes F1 / F2 / F3
# (v224f/RESULT.md "Fixes 10-15"). Laptop CPU, offline, no Kaggle, no box, no OAI data. Each case runs the script as a
# subprocess, in a temp dir.
#   python tests/test_fix1015.py          the fixed scripts (expected: TEST_FIX1015 PASS)
#   python tests/test_fix1015.py pre      the *.pre_fix_1015 copies (expected: FAIL on every fixed case)
# F1  build_v224fs.py --base ff keeps the V224-FF20 flip budget 4.9 h: cells 0-15 (everything before SAVE, incl. the cell 7 flip
#     driver) == the commit-tested V224-FF20 kernel; fleet start budget default 5.1 h (MIN_START_MIN 192), 5.0 h still allowed.
# F2  make_s_pack.py accepts only a REAL all-pass gate file of the student --tpl on the table basename(--table).
# F3  build_v224fs.py without --base is refused; make_s_pack.py's NEXT text is the FLIP-base plan.
import os, sys, json, glob, shutil, subprocess, tempfile, re

HERE = os.path.dirname(os.path.abspath(__file__)); V = os.path.dirname(HERE); ROOT = os.path.normpath(os.path.join(V, '..', '..'))
PRE = len(sys.argv) > 1 and sys.argv[1] == 'pre'
SUF = '.pre_fix_1015' if PRE else ''
BUILD = os.path.join(V, 'build_v224fs.py' + SUF)
PACK = os.path.join(V, 'make_s_pack.py' + SUF)
FF20 = os.path.join(ROOT, 'session_2026-10-15', 'v224ff', 'kernel_v224ff20', 'rsna_sub_v224ff20.ipynb')
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
R = []


def chk(name, ok, info=''):
    R.append(bool(ok))
    print(f'  {"OK  " if ok else "FAIL"} {name} {info}', flush=True)


def run(args, cwd=None):
    r = subprocess.run([sys.executable] + args, capture_output=True, text=True, encoding='utf-8', cwd=cwd)
    return r.returncode, r.stdout, r.stderr


TMP = tempfile.mkdtemp(prefix='fix1015_')
FNB = json.load(open(FF20, encoding='utf-8')); FC = [src(c) for c in FNB['cells']]
# a manifest in the make_s_pack format: the 5 k14w DT checkpoints relabelled as S (exactly the builder's stand-in, as a file)
k = [c for c in FC if c.startswith('K12M_CFG = {')][0].split('\n', 1)[0]
K12M = eval(k[len('K12M_CFG = '):k.index('   # literal')], {})
add = K12M['add_manifest']
smap = {n: re.sub(r'^f(\d)_s1043f75oai_', r'f\1_s1047r7tb_', n) for n in add['checkpoints']}
MAN = os.path.join(TMP, 'r7tb_manifest.json')
json.dump({'checkpoints': [smap[n] for n in add['checkpoints']],
           'sources': {smap[n]: dict(add['sources'][n], original=smap[n]) for n in add['checkpoints']}}, open(MAN, 'w'), indent=1)
print(f'TEST_FIX1015 {"PRE-FIX copies" if PRE else "fixed scripts"} | tmp {TMP}', flush=True)


def build(tag, extra):
    out = os.path.join(TMP, tag)
    os.makedirs(out)
    rc, so, se = run([BUILD, '--wf', '0.20', '--s-manifest', MAN, '--out', out] + extra)
    kd = glob.glob(os.path.join(out, 'kernel_*'))
    B = json.load(open(os.path.join(kd[0], 'BUILD_V224FS.json'))) if rc == 0 and len(kd) == 1 else {}
    nb = json.load(open(glob.glob(os.path.join(kd[0], '*.ipynb'))[0], encoding='utf-8')) if B else None
    return rc, so, se, B, ([src(c) for c in nb['cells']] if nb else []), [os.path.basename(d) for d in kd]


# ---- F1 -----------------------------------------------------------------------------------------------------------------------
rc, so, se, B, C, kd = build('ff_default', ['--base', 'ff'])
chk('F1 default ff build (the main-session command) succeeds', rc == 0 and kd == ['kernel_v224ffs20'], (rc, kd, se.strip()[-200:]))
chk('F1 flip budget 4.9 (V224-FF20), cell 7 not patched', B.get('flip_budget_h') == 4.9 and 7 not in (B.get('patched_base_cells') or [7]),
    {k: B.get(k) for k in ('flip_budget_h', 'patched_base_cells')})
chk('F1 cells 0-15 (all before SAVE, incl. the flip driver cell 7) == V224-FF20 commit-tested kernel, byte for byte',
    len(C) == 25 and C[:16] == FC[:16], [i for i in range(16) if C[i:i + 1] != FC[i:i + 1]] if C else 'no build')
chk('F1 cell 7 carries the FF20 literal FLIP_TA_BUDGET_H ... else 4.9', bool(C) and 'FLIP_TA_BUDGET_H = float(_fta_env) if _fta_env else 4.9\n' in C[7])
k2 = [c for c in C if c.startswith('K2_CFG = {')]
chk('F1 default fleet start budget 5.1 h: build json + SAVE knobs + K2 MIN_START_MIN 192.0',
    B.get('start_budget_h') == 5.1 and B.get('min_start_min') == 192.0 and len(k2) == 1 and "'MIN_START_MIN': 192.0" in k2[0].split('\n', 1)[0]
    and "'start_budget_h': 5.1," in C[16] and "'min_start_min': 192.0," in C[16], {k: B.get(k) for k in ('start_budget_h', 'min_start_min')})
rc, so, se, B, C, kd = build('ff_start50', ['--base', 'ff', '--start-budget-h', '5.0'])
chk('F1 --start-budget-h 5.0 builds with the flip budget 4.9 (cells 0-15 == FF20, MIN_START_MIN 198)',
    rc == 0 and B.get('flip_budget_h') == 4.9 and C[:16] == FC[:16] and B.get('min_start_min') == 198.0, (rc, se.strip()[-160:]))
rc, so, se, B, C, kd = build('ff_old44', ['--base', 'ff', '--flip-budget-h', '4.4', '--start-budget-h', '5.0'])
chk('F1 explicit old variant --flip-budget-h 4.4 still builds (cell 7 patched, revert proof inside the builder)',
    rc == 0 and B.get('flip_budget_h') == 4.4 and 7 in B.get('patched_base_cells', []) and C[7] != FC[7], (rc, se.strip()[-160:]))
for tag, extra in (('neg_flip_ge_start', ['--base', 'ff', '--flip-budget-h', '5.1', '--start-budget-h', '5.1']),
                   ('neg_start55', ['--base', 'ff', '--start-budget-h', '5.5']),
                   ('neg_f_start51', ['--base', 'f', '--start-budget-h', '5.1'])):
    rc, so, se, B, C, kd = build(tag, extra)
    chk(f'F1 refused: {" ".join(extra)}', rc != 0 and 'AssertionError' in se and not kd, (rc, se.strip()[-120:]))

# ---- F3 -----------------------------------------------------------------------------------------------------------------------
rc, so, se, B, C, kd = build('no_base', [])
chk('F3 build_v224fs.py without --base is refused (argparse, nothing built)', rc == 2 and '--base' in se and not kd, (rc, kd, se.strip()[-160:]))
t = open(PACK, encoding='utf-8').read()
nx = t[t.index("NEXT (main session"):]
chk('F3 make_s_pack NEXT text: build with --wf 0.20 --base ff, gate v224ffs_check.py (no v224fs_check.py, no kernel_v224fs<WF>)',
    'build_v224fs.py --wf 0.20 --base ff --s-manifest' in nx and 'v224ffs_check.py OUT --wf 0.20' in nx and '--ref-flip' in nx
    and 'v224fs_check.py' not in nx and 'kernel_v224fs<WF>' not in nx)

# ---- F2 -----------------------------------------------------------------------------------------------------------------------
HDR = 'student_gate 2026-10-15 03:10:00 | student {s} | table {t} | teacher real/R7OOF.csv | {kind}'
OKV = "R7T_STUDENT_GATE REAL A1 | G0 P G1 P G2 P G3 P G3' P G4 P | VERDICT PASS"


def gate_file(name, s, t, verdicts, kind='REAL'):
    p = os.path.join(TMP, 'gate', name + '.txt')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    body = [HDR.format(s=s, t=t, kind=kind), 'G0 integrity: PASS', 'G1 gold58: S 0.92 | D 0.9214 | D2 0.9192 -> PASS'] + verdicts
    open(p, 'w').write('\n'.join(body) + '\n' + json.dumps({'verdict': 'PASS' if verdicts[-1].endswith('VERDICT PASS') else 'FAIL',
                                                             'sim': kind != 'REAL', 'amendment': 'A1'}, indent=1) + '\n')
    return p


TB, TA = 'f{F}_s1047_ttr_r7tb', 'f{F}_s1048_ttr_r7ta'
REF = os.path.join(TMP, 'empty_ref'); os.makedirs(REF)
cases = [  # name, gate file, expect past-the-gate (True) or refused at the gate (False)
    ('T_b REAL all-pass (the right file)', gate_file('tb_pass', TB, 'real/labels_r7t_b.csv', [OKV]), True),
    ('T_a REAL all-pass (another student)', gate_file('ta_pass', TA, 'real/labels_r7t_a.csv', [OKV]), False),
    ('T_b student but table labels_r7t_a.csv', gate_file('tb_table_a', TB, 'real/labels_r7t_a.csv', [OKV]), False),
    ('T_b FAIL (G3\' F)', gate_file('tb_fail', TB, 'real/labels_r7t_b.csv',
                                    ["R7T_STUDENT_GATE REAL A1 | G0 P G1 P G2 P G3 P G3' F G4 P | VERDICT FAIL"]), False),
    ('T_b FAIL file with an all-pass line appended (two verdict lines)', gate_file(
        'tb_two', TB, 'real/labels_r7t_b.csv', ["R7T_STUDENT_GATE REAL A1 | G0 P G1 F G2 P G3 P G3' P G4 P | VERDICT FAIL", OKV]), False),
    ('SIMULATION gate', gate_file('sim', TB, 'real/labels_r7t_b.csv',
                                  ["R7T_STUDENT_GATE SIMULATION (TEST ONLY) A1 | G0 P G1 P G2 P G3 P G3' P G4 P | VERDICT PASS (not a real verdict)"],
                                  kind='SIMULATION (TEST ONLY)'), False),
]
for name, p, past in cases:
    rc, so, se = run([PACK, '--gate', p, '--ref', REF, '--out', os.path.join(TMP, 'pack_out')])
    last = se.strip().splitlines()[-1] if se.strip() else ''
    got_past = last == 'AssertionError: []'                    # the next assert after the gate: no reference checkpoint in --ref
    refused = rc != 0 and last.startswith('AssertionError: ') and ('student gate' in last or 'REAL A1' in last)
    chk(f'F2 {name}: {"passes the gate check" if past else "refused at the gate check"}', got_past if past else refused, last[:150])
# the T_a gate with --tpl T_a passes the gate check (then make_s_pack's seed-1047 checks refuse the T_a runs, as before)
rc, so, se = run([PACK, '--gate', cases[1][1], '--tpl', TA, '--table', '/x/real/labels_r7t_a.csv', '--ref', REF])
chk('F2 T_a gate with --tpl T_a --table labels_r7t_a.csv passes the gate check', (se.strip().splitlines() or [''])[-1] == 'AssertionError: []')

n = sum(R)
print(f'TEST_FIX1015 {"PASS" if all(R) else "FAIL"} {n}/{len(R)} ({"pre-fix copies" if PRE else "fixed scripts"})', flush=True)
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(0 if all(R) else 1)
