# test_aofix_1015.py (session_2026-10-15/v224f/tests, 10-15, Claude fix agent): tests for the fixes F1 / F2 of the AO review
# (v224f/RESULT.md "AO fix 10-15"). Laptop CPU, offline, no Kaggle, no box, no GPU (torch CPU only for synthetic checkpoints).
#   python tests/test_aofix_1015.py          the fixed files (expected: TEST_AOFIX_1015 PASS)
#   python tests/test_aofix_1015.py pre      the *.pre_aofix_1015 copies (expected: every [FIX] case FAILs; [INV] cases hold in both)
# Needs tests/mock_out_v224flips20 / mock_out_v224flrms20 made from the CURRENT stand-ins (run tests/mock_ao_output.py flips / flrms
# first), because the branch b / c gate lines printed by make_s_pack.py are run on them.
# F1 [FIX] build_v224fs.py with --base flip / flrm and NO --start-budget-h builds the 7.0 h guard: BUILD json start_budget_h 7.0,
#    min_start_min 78, ao_guard_h 7.0; K2_CFG MIN_START_MIN 78; SAVE-AO knobs 7.0 / 78.
#    [INV] --start-budget-h 6.5 still builds MIN_START_MIN 108 (the reviewed set is unchanged); guard_paired_1015.py (MODEL, paired
#    sessions): a higher guard never adds a canary and the base part is identical across guards.
# F2 [FIX] make_s_pack.py, run end to end on a synthetic 5-fold T_b student (temp dir), prints the PREREG_10-15 A2 + A3 branch table:
#    its branch b / c build lines, run verbatim (manifest = the one make_s_pack just wrote, + --out TMP), build kernel_v224flips20 /
#    kernel_v224flrms20 at MIN_START_MIN 78; its branch b / c gate lines, run verbatim on the AO mocks (stand-in build json), PASS 28/28;
#    the c line carries --ref-flrm. [INV] its branch a build line builds kernel_v224ffs20 (MIN_START_MIN 192).
#    [FIX] RUNBOOK_STUDENT_1015.md: Step 7 carries the branch table and the b / c commands (--base flip / flrm --start-budget-h 7.0,
#    v224aos_check.py, --ref-flrm); the unqualified "Only --wf 0.20 --base ff is pre-registered" / "Do NOT build --replace now" lines
#    are gone; Step 13 has the b / c note. [INV] the Step 4 md5 line == the md5 of the make_s_pack.py it ships with.
import os, sys, json, glob, shutil, subprocess, tempfile, re, hashlib

T = os.path.dirname(os.path.abspath(__file__)); V = os.path.dirname(T); ROOT = os.path.normpath(os.path.join(V, '..', '..'))
PRE = len(sys.argv) > 1 and sys.argv[1] == 'pre'
SUF = '.pre_aofix_1015' if PRE else ''
BUILD = os.path.join(V, 'build_v224fs.py' + SUF)
PACK = os.path.join(V, 'make_s_pack.py' + SUF)
RUNBOOK = os.path.join(ROOT, 'session_2026-10-15', 'r7teacher', 'RUNBOOK_STUDENT_1015.md' + SUF)
F20 = os.path.join(V, 'kernel_v224f20', 'rsna_sub_v224f20.ipynb')
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
R = []   # (kind, ok)


def chk(kind, name, ok, info=''):
    R.append((kind, bool(ok)))
    print(f'  {"OK  " if ok else "FAIL"} [{kind}] {name} {info}', flush=True)


def run(args, cwd=None):
    r = subprocess.run([sys.executable] + args, capture_output=True, text=True, encoding='utf-8', cwd=cwd)
    return r.returncode, r.stdout, r.stderr


def lit(cell, name):
    ln = cell.split('\n', 1)[0]
    return eval(ln[len(name + ' = '):ln.index('   # literal')], {})


def ao_build_facts(kdir):
    """BUILD json guard keys + the K2_CFG MIN_START_MIN + the SAVE-AO knobs of an AO kernel dir."""
    B = json.load(open(os.path.join(kdir, 'BUILD_V224FS.json')))
    nb = json.load(open(glob.glob(os.path.join(kdir, '*.ipynb'))[0], encoding='utf-8')); C = [src(c) for c in nb['cells']]
    k2 = [c for c in C if c.startswith('K2_CFG = {')]; sv = [c for c in C if c.startswith('# [eeshsaxena V224-AO-S SAVE')]
    m = re.search(r"'start_budget_h': ([0-9.]+), 'wall_h': 8.3, 'min_start_min': ([0-9.]+),", sv[0]) if len(sv) == 1 else None
    return dict(G=B.get('start_budget_h'), M=B.get('min_start_min'), ao=B.get('ao_guard_h'),
                k2=lit(k2[0], 'K2_CFG')['MIN_START_MIN'] if len(k2) == 1 else None,
                save=(float(m.group(1)), float(m.group(2))) if m else None, tag=B.get('tag'))


TMP = tempfile.mkdtemp(prefix='aofix1015_')
print(f'TEST_AOFIX_1015 ({"pre-change copies" if PRE else "fixed files"}) | tmp {TMP}', flush=True)
# a manifest in the make_s_pack format: the 5 k14w DT checkpoints relabelled as S (exactly test_ao_1015's / test_fix1015's)
FC = [src(c) for c in json.load(open(F20, encoding='utf-8'))['cells']]
add = lit([c for c in FC if c.startswith('K12M_CFG = {')][0], 'K12M_CFG')['add_manifest']
smap = {n: re.sub(r'^f(\d)_s1043f75oai_', r'f\1_s1047r7tb_', n) for n in add['checkpoints']}
MAN = os.path.join(TMP, 'r7tb_manifest.json')
json.dump({'checkpoints': [smap[n] for n in add['checkpoints']],
           'sources': {smap[n]: dict(add['sources'][n], original=smap[n]) for n in add['checkpoints']}}, open(MAN, 'w'), indent=1)

# ---- F1 -------------------------------------------------------------------------------------------------------------------------
for base in ('flip', 'flrm'):
    out = os.path.join(TMP, 'f1_' + base); os.makedirs(out)
    rc, so, se = run([BUILD, '--wf', '0.20', '--base', base, '--s-manifest', MAN, '--out', out])
    kd = glob.glob(os.path.join(out, 'kernel_*'))
    f = ao_build_facts(kd[0]) if rc == 0 and len(kd) == 1 else {}
    chk('FIX', 'F1 --base %s without --start-budget-h builds the 7.0 h guard (BUILD json 7.0 / 78 / ao_guard_h 7.0, K2 78, SAVE 7.0 / 78)' % base,
        f.get('G') == 7.0 and f.get('M') == 78.0 and f.get('ao') == 7.0 and f.get('k2') == 78.0 and f.get('save') == (7.0, 78.0),
        {k: f.get(k) for k in ('G', 'M', 'k2', 'save')} if f else (rc, se.strip()[-200:]))
out = os.path.join(TMP, 'f1_flip_65'); os.makedirs(out)
rc, so, se = run([BUILD, '--wf', '0.20', '--base', 'flip', '--start-budget-h', '6.5', '--s-manifest', MAN, '--out', out])
kd = glob.glob(os.path.join(out, 'kernel_*')); f = ao_build_facts(kd[0]) if rc == 0 and len(kd) == 1 else {}
chk('INV', 'F1 --base flip --start-budget-h 6.5 still builds MIN_START_MIN 108 (reviewed set unchanged)',
    f.get('G') == 6.5 and f.get('M') == 108.0 and f.get('k2') == 108.0 and f.get('save') == (6.5, 108.0), {k: f.get(k) for k in ('G', 'M', 'k2')})
rc, so, se = run([os.path.join(V, 'guard_paired_1015.py'), '--n', '3000'])
hdr = [ln for ln in so.splitlines() if ln.startswith(('FLIP-AO', 'FLRM-AO'))]
chk('INV', 'F1 guard_paired_1015 (MODEL, 3000 paired sessions x 8 rows): base part identical across guards, 0 sessions canary at 7.0 '
    'but not at 6.5 (nor at 6.5 but not at 6.0)', rc == 0 and len(hdr) == 8 and all(
        'identical across guards: True' in h and h.endswith('canary at 6.5 not at 6.0: 0 | canary at 7.0 not at 6.5: 0') for h in hdr),
    (rc, len(hdr), se.strip()[-200:]))

# ---- F2 make_s_pack.py end to end on a synthetic T_b student -------------------------------------------------------------------
import torch  # noqa: E402  (CPU; tiny tensors only)
TB = 'labels_r7t_b.csv'
UIDS = ['1.2.826.0.1.3680043.8.498.%d' % i for i in range(1, 5)]
KEYV = dict(contract='c1', views=['sag', 'cor', 'ax'], backbone='convnext_small', dim=768, n_slices=24, n_views=3, n_contrast=1,
            store='jpeg', crop_mm=160, src_slices=32, jpeg_fp='fp', select_sha='abc123')
REF = os.path.join(TMP, 'ref'); RUNS = os.path.join(TMP, 'runs'); os.makedirs(REF); os.makedirs(RUNS)
torch.save({'model': {'w': torch.zeros(2)}, 'args': dict(KEYV), 'cv': 0.87, 'epoch': 20},
           os.path.join(REF, 'f0_s1043f75_raw_ttr224-320x24_cv8700_best.pt'))
for u in UIDS:
    open(os.path.join(REF, 'parity_%s.npz' % u), 'wb').write(b'x')
for F in range(5):
    d = os.path.join(RUNS, 'f%d_s1047_ttr_r7tb' % F); os.makedirs(d)
    a = dict(KEYV, fold=F, seed=1047, epochs=21, ttr_size=224, model_mod='fleet_model_ttr', model_sha='f141aeed136dfc01', amp='fp16',
             train_labels=TB, sel_labels=TB, decode_probe={UIDS[F % 4]: 1})
    torch.save({'model': {'w': torch.full((2,), float(F))}, 'args': a, 'cv': 0.88 + F / 1000, 'epoch': 20},
               os.path.join(d, 'f%d_s1047_raw_ttr224-320x24_cv88%d0_best.pt' % (F, F)))
GATE = os.path.join(TMP, 'student_gate_REAL.txt')
open(GATE, 'w').write('student_gate 2026-10-15 03:10:00 | student f{F}_s1047_ttr_r7tb | table real/labels_r7t_b.csv | teacher real/R7OOF.csv '
                      "| REAL\nG0 integrity: PASS\nR7T_STUDENT_GATE REAL A1 | G0 P G1 P G2 P G3 P G3' P G4 P | VERDICT PASS\n")
PK = os.path.join(TMP, 'pack')
rc, so, se = run([PACK, '--gate', GATE, '--runs', RUNS, '--table', os.path.join(TMP, 'real', TB), '--ref', REF, '--out', PK])
up = os.path.join(PK, 'upload', 'rsna-knee-cnxt-fleet-r7tb')
chk('INV', 'F2 make_s_pack.py on the synthetic 5-fold T_b student: S_PACK OK + r7tb_manifest.json + NEXT text',
    rc == 0 and 'S_PACK OK' in so and os.path.isfile(os.path.join(up, 'r7tb_manifest.json')) and 'NEXT (main session' in so,
    (rc, se.strip()[-300:]))
nx = so[so.index('NEXT (main session'):] if 'NEXT (main session' in so else ''
L = nx.splitlines()
MANP = '<LAPTOP>/session_2026-10-15/v224f/s_upload/rsna-knee-cnxt-fleet-r7tb/r7tb_manifest.json'


def cmd_line(key):
    """The printed command line holding `key`, from its 'python ' on (one line per command in the NEXT text)."""
    h = [ln for ln in L if key in ln]
    return h[0][h[0].index('python '):].strip() if len(h) == 1 and 'python ' in h[0] else None


def build_from(line, tag):
    out = os.path.join(TMP, 'nx_' + tag); os.makedirs(out)
    a = line.replace(MANP, os.path.join(up, 'r7tb_manifest.json')).split()[1:]
    rc_, so_, se_ = run(a + ['--out', out], cwd=ROOT)
    kd_ = glob.glob(os.path.join(out, 'kernel_*'))
    return rc_, kd_, se_


for br, base, tag in (('b', 'flip', 'v224flips20'), ('c', 'flrm', 'v224flrms20')):
    bl = cmd_line('build_v224fs.py --wf 0.20 --base %s ' % base)
    ok = bl is not None and '--start-budget-h 7.0' in bl and MANP in bl
    f = {}
    if ok:
        rc, kd, se = build_from(bl, br)
        ok = rc == 0 and [os.path.basename(k) for k in kd] == ['kernel_' + tag]
        f = ao_build_facts(kd[0]) if ok else {}
        ok = ok and f.get('M') == 78.0 and f.get('k2') == 78.0 and f.get('G') == 7.0
    chk('FIX', 'F2 NEXT branch %s build line (--base %s --start-budget-h 7.0), run verbatim: kernel_%s at MIN_START_MIN 78' % (br, base, tag),
        ok, bl and bl[:160])
    gl = cmd_line('v224aos_check.py OUT --wf 0.20 --build session_2026-10-15/v224f/kernel_%s/BUILD_V224FS.json' % tag)
    ok = gl is not None and '--ref-exact session_2026-10-14/v224/out_v224_v1' in gl and '--ref-flip session_2026-10-14/v224/out_v224flip_v1' in gl \
        and (('--ref-flrm session_2026-10-15/v224rm/out_v224flrm_v1' in gl) == (base == 'flrm'))
    last = ''
    if ok:
        g = re.sub(r'\s*\[--ref-s [^\]]*\]', '', gl)
        g = g.replace(' OUT ', ' session_2026-10-15/v224f/tests/mock_out_%s ' % tag, 1)
        g = g.replace('session_2026-10-15/v224f/kernel_%s/BUILD_V224FS.json' % tag, 'session_2026-10-15/v224f/tests/standin_%s/BUILD_V224FS.json' % tag)
        rc, so_, se = run(g.split()[1:], cwd=ROOT)
        last = (so_.strip().splitlines() or [''])[-1]
        ok = rc == 0 and last.startswith('V224AOS_CHECK PASS 28/28')
    chk('FIX', 'F2 NEXT branch %s gate line (v224aos_check.py%s), run verbatim on the AO mock + stand-in build json: PASS 28/28'
        % (br, ' --ref-flrm' if base == 'flrm' else ' --ref-exact --ref-flip'), ok, last[:120] or (gl and gl[:160]))
bl = cmd_line('build_v224fs.py --wf 0.20 --base ff --s-manifest')
ok = bl is not None
if ok:
    rc, kd, se = build_from(bl, 'a')
    ok = rc == 0 and [os.path.basename(k) for k in kd] == ['kernel_v224ffs20'] and json.load(
        open(os.path.join(kd[0], 'BUILD_V224FS.json')))['min_start_min'] == 192.0
chk('INV', 'F2 NEXT branch a build line (--base ff), run verbatim: kernel_v224ffs20 at MIN_START_MIN 192', ok, bl and bl[:120])
chk('FIX', 'F2 NEXT names the A3 branches and supersessions (b FLIP-AO + S never --base ff --replace; c FLRM-AO + S; d / e)',
    'FLIP-AO + S (A3 R2' in nx and 'never --base ff --replace' in nx and 'FLRM-AO + S (A3 R3)' in nx and 'no FF-S / FFSR (A2 d)' in nx)

# ---- F2 RUNBOOK -----------------------------------------------------------------------------------------------------------------
rb = open(RUNBOOK, encoding='utf-8').read()
s7 = rb[rb.index('## Step 7.'):rb.index('## Step 8.')]
s13 = rb[rb.index('## Step 13.'):rb.index('## Appendix A.')]
chk('FIX', 'F2 RUNBOOK Step 7: branch table + b / c build (--base flip / flrm --start-budget-h 7.0) and v224aos_check (--ref-flrm for c)',
    '| b | V224-FLIP stays the team best' in s7 and '| c | V224-FLIP-RM became the team best' in s7
    and 'build_v224fs.py --wf 0.20 --base flip --start-budget-h 7.0' in s7 and 'c: --base flrm' in s7 and 'v224aos_check.py' in s7
    and '--ref-flrm session_2026-10-15/v224rm/out_v224flrm_v1 (required)' in s7)
chk('FIX', 'F2 RUNBOOK: the unqualified "Only --wf 0.20 --base ff is pre-registered" and "Do NOT build --replace now" lines are gone',
    'Only `--wf 0.20 --base ff` is pre-registered.' not in rb and 'Do NOT build `--replace` now' not in rb)
chk('FIX', 'F2 RUNBOOK Step 13: branches b / c note (no REPLACE step)', 'Branches b / c' in s13 and 'no REPLACE step' in s13)
m = re.search(r'md5sum make_s_pack\.py\s+#\s*([0-9a-f]{32})', rb)
chk('INV', 'F2 RUNBOOK Step 4 md5 line == md5 of the make_s_pack.py it ships with',
    m is not None and m.group(1) == hashlib.md5(open(PACK, 'rb').read()).hexdigest(), m and m.group(1))

shutil.rmtree(TMP, ignore_errors=True)
fx = [ok for k, ok in R if k == 'FIX']; iv = [ok for k, ok in R if k == 'INV']
if PRE:
    good = not any(fx) and all(iv)
    print(f'TEST_AOFIX_1015 pre-change copies: [FIX] {sum(fx)}/{len(fx)} pass (expected 0), [INV] {sum(iv)}/{len(iv)} '
          f'-> {"AS EXPECTED (every fixed case FAILs before the fix)" if good else "UNEXPECTED"}', flush=True)
    sys.exit(0 if good else 1)
print(f'TEST_AOFIX_1015 {"PASS" if all(fx + iv) else "FAIL"} {sum(fx + iv)}/{len(fx + iv)} ([FIX] {sum(fx)}/{len(fx)}, [INV] {sum(iv)}/{len(iv)})',
      flush=True)
sys.exit(0 if all(fx + iv) else 1)
