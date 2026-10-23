# sim_verdict.py (Claude, 2026-10-14): offline test of driver_jj.verdict() on synthetic phase results (no Kaggle).
# A consistent "fix works" RESULT must PASS; each single defect must FAIL exactly the expected check(s).
import os, sys, json, copy, tempfile, io, contextlib
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, '..', 'kernel_src', 'driver_jj.py'), encoding='utf-8').read()
assert src.rstrip().endswith('main()')
src = src.rstrip()[:-len('main()')]
tmp = tempfile.mkdtemp()
src = src.replace("WORK = Path('/kaggle/working')", 'WORK = Path(%r)' % tmp).replace("JJ = Path('/tmp/jj')", 'JJ = Path(%r)' % os.path.join(tmp, 'jj'))
for k in ('@@DEC_SRC@@', '@@JLL_SRC@@', '@@PROBE_SRC@@', '@@FASTREAD_B64@@', '@@R7_SRC_Z@@', '@@DEC_SHA@@', '@@NB_SHA@@', '@@FASTREAD_SHA@@'):
    src = src.replace(k, "''")
for k in ('@@R7_SOURCE_SHA@@', '@@R7_CODE_SHA@@'):
    src = src.replace(k, '{}')
V = ['U', 'IMPL', 'JLL70', 'JLL57', 'J2KR', 'J2KI', 'RLE', 'MIX']
REAL = ['t1', 't2', 't3', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6']
SEP = os.pathsep
PP0 = '/kaggle/lib/kagglegym' + SEP + '/kaggle/lib'


def rec_for(reader, sha):
    if reader == 'r7':
        return {'sha': sha, 'windows': 510, 'item_error': None, 'native_series': 5}
    if reader == 'raptor':
        return {'sha': sha, 'mask_sha': 'm', 'mask_sum': 84}
    if reader == 'fleet':
        return {'sha': sha, 'views': ['Axi1', 'Sag0', 'Cor1', 'Sag1'], 'views_present': 4}
    return {'sha': sha, 'ok': 135, 'files': 135}


def phase(on, readers, pyd):
    env = {'pydicom': pyd + ' @ x', 'PYTHONPATH': 'p',
           'pylibjpeg': ('2.1.0 @ /tmp/_dec/pylibjpeg' if on else 'NO (ModuleNotFoundError)'),
           'pydicom_decoders': {'1.2.840.10008.1.2.4.70': ['pylibjpeg'] if on else []}}
    p = {'env': env}
    for rd in readers:
        fake = {}
        for v in V:
            if v == 'J2KI':
                s = 'lossy'
            elif v in ('JLL70', 'JLL57', 'MIX') and not on:
                s = 'dropped_' + v
            else:
                s = 'U'
            fake['f' + v] = rec_for(rd, s)
        p[rd + '_fake'] = fake
        p[rd + '_real'] = {u: rec_for(rd, 'real_' + u) for u in REAL}
    if 'plain' in readers:
        p['synthetic'] = {'f%02d.dcm' % i: ({'sha': 's%d' % i} if i > 2 else {'error': 'E%d' % i}) for i in range(48)}
    return p


base = {
    'dec_main': {'ok_line': True, 'pp0': PP0, 'ppj': '/tmp/_dec' + SEP + PP0, 'names_left': ['_DEC_RECEIPT'],
                 'receipt': {'ok': True, 'selftest': {'versions': {'pylibjpeg': '2.1.0', 'libjpeg': '2.4.0', 'openjpeg': '2.6.0'},
                                                      'dirs': {'pylibjpeg': '/tmp/_dec', 'libjpeg': '/tmp/_dec', 'openjpeg': '/tmp/_dec'},
                                                      'plugins': {'1.2.840.10008.1.2.4.57': ['pylibjpeg'], '1.2.840.10008.1.2.4.70': ['pylibjpeg']},
                                                      'decode': {k: {'exact': True} for k in ('JLL57_12U', 'JLL70_16S', 'U_16S')}}}},
    'dec_negatives': {k: {'rc': 0, 'dec_ok': k == 'positive', 'dec_fallback': k != 'positive',
                          'pp_after': ('/tmp/_dec_pos' + SEP + PP0) if k == 'positive' else PP0}
                      for k in ('missing', 'tampered', 'selftest', 'positive')},
    'r7_install': {'pip_rc': 0, 'success_line': True, 'pythonpath_at_pip': '/tmp/_dec' + SEP + PP0},
    'fixtures': {'variants': {v: {'study': 'f' + v, 'encode': {'files': 135, 'errors': 0, 'exact': 135 if v != 'J2KI' else 0}} for v in V}},
    'real_root': {'test': REAL[:3], 'train': REAL[3:]},
    'probe_S': phase(False, ['plain', 'raptor', 'fleet'], '3.0.2'), 'probe_SJ': phase(True, ['plain', 'raptor', 'fleet'], '3.0.2'),
    'probe_R': phase(False, ['plain', 'raptor', 'fleet'], '3.0.1'), 'probe_RJ': phase(True, ['plain', 'raptor', 'fleet'], '3.0.1'),
    'probe_R7': phase(False, ['r7'], '3.0.1'), 'probe_R7J': phase(True, ['r7'], '3.0.1')}


def run(result):
    g = {'__name__': 'drv'}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(src, 'driver_jj', 'exec'), g)
        g['RESULT'].update(copy.deepcopy(result))
        g['verdict']()
    return g['RESULT']['verdict'], [c['name'] for c in g['CHECKS'] if not c['ok']]


res = []
v, fails = run(base)
print('clean:', v, fails[:3])
res.append(v.startswith('V224F20J_CPU PASS'))
n_total = int(v.split('/')[-1])


def mut(name, f, expect_sub):
    r = copy.deepcopy(base); f(r)
    v, fails = run(r)
    ok = v.startswith('V224F20J_CPU FAIL') and bool(fails) and all(any(s in x for s in expect_sub) for x in fails)
    res.append(ok)
    print('%s %-55s -> %s | failed: %s' % ('OK  ' if ok else 'BAD ', name, v, fails))


mut('real study differs ON vs OFF (Raptor r384)', lambda r: r['probe_SJ']['raptor_real']['r2'].update(sha='x'), ['Raptor r384 (stock image): 8/9'])
mut('U-path differs ON vs OFF (fleet V224-F20)', lambda r: [r['probe_RJ']['fleet_fake'][k].update(sha='x') for k in ('fU', 'fIMPL', 'fJLL70', 'fJLL57', 'fJ2KR', 'fRLE', 'fMIX')],
    ['fleet in V224-F20 (after R7 pip): U fixture'])
mut('JLL57 not recovered in R7J', lambda r: r['probe_R7J']['r7_fake']['fJLL57'].update(sha='dropped', item_error='no series'), ['R7 worker (CODE + PYTHONPATH): JLL57'])
mut('J2K lossy changed by the fix (r224)', lambda r: r['probe_RJ']['raptor_fake']['fJ2KI'].update(sha='lossy2'), ['Raptor r224 (after R7 pip): J2K lossy'])
mut('synthetic refusal message differs', lambda r: r['probe_SJ']['synthetic']['f01.dcm'].update(error='E99'), ['synthetic DICOM fixtures'])
mut('DEC fallback in the main run', lambda r: (r['dec_main'].update(ok_line=False, ppj=PP0), r['dec_main']['receipt'].update(ok=False)),
    ['DEC cell (verbatim)', 'PYTHONPATH after', 'R7 pip install'])
mut('tampered wheel accepted', lambda r: r['dec_negatives']['tampered'].update(dec_ok=True, dec_fallback=False, pp_after='/x'), ['"tampered"'])
mut('R7 pip success line missing', lambda r: r['r7_install'].update(success_line=False), ['R7 pip install'])
mut('pylibjpeg missing in phase RJ', lambda r: r['probe_RJ']['env'].update(pylibjpeg='NO (x)'), ['phase RJ environment'])
mut('real study errors in both phases (R7)', lambda r: [r[p]['r7_real'].update(t1={'item_exc': 'boom'}) for p in ('probe_R7', 'probe_R7J')], ['R7 worker (CODE + PYTHONPATH): 8/9'])
mut('a probe step failed', lambda r: r['probe_S'].update(fleet_real={'step_error': 'x'}), ['fleet in the stock image (HFR7-RR): 0/9', 'phase S: every probe step ran'])
mut('real study item_error in both phases (R7, equal keys)', lambda r: [r[p]['r7_real']['r3'].update(item_error='no series file present', windows=0) for p in ('probe_R7', 'probe_R7J')], ['R7 worker (CODE + PYTHONPATH): 8/9'])
mut('R7 real probe process failed to load', lambda r: r['probe_R7J'].update(error_real='x'), ['phase R7J: every probe step ran'])
print('SIM_VERDICT %s %d/%d (clean run has %d checks)' % ('PASS' if all(res) else 'FAIL', sum(res), len(res), n_total))
