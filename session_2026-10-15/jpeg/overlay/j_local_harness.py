#!/usr/bin/env python
# j_local_harness.py (2026-10-15, Claude subagent): local CPU harness for J kernels built by make_j.py (laptop, Windows, Python 3.12,
# no network, nothing downloaded). For each J kernel it takes the DEC cell VERBATIM from the built notebook and changes only:
#   - the path line (INPUT / TARGET / WORK -> temp dirs),
#   - in the cases that need a working decoder, the WHEELS / TOP constants -> the local stand-in wheels of tests/stub_wheels.py
#     (same distribution / import names and versions; the real cp313 manylinux wheels cannot be installed on win32),
#   - in the timeout case, the timeout constants.
# It then runs the cell in a simulated notebook process (tests/j_sim.py --nb) and launches every stage chain of THAT notebook the
# way the notebook launches it: the env expression of each launch is extracted from the J notebook text (R7 controller env, R7
# probe_dual worker env, fleet worker env, S6F driver env from the real S6F SETUP cell, S6F b_env) and evaluated in the launching
# process; Raptor r384 / r224 inherit (no env=); fork pools / DataLoader workers are simulated with multiprocessing spawn (stricter
# than fork: the child rebuilds sys.path from the parent's). Every chain ends in pydicom.dcmread(path).pixel_array on synthetic
# DICOM files (JPEG Lossless .70 16-bit signed, .57 12-bit unsigned, a new .70 16-bit unsigned one, an uncompressed control).
# Review fixes (J1 / J2 / J3): every case now has a fake competition test root (2 studies, 4 series dirs, 7 files: uncompressed,
# JPEG Lossless .70 / .57, and a 37-row JPEG Lossless TRAP file) for the DEC preflight, and a fake /proc that the simulated
# processes fill (tests/j_sim.py) for the DEC cell's envwatch thread, which runs VERBATIM except for its PRE_TIMEOUT_S /
# WATCH_EVERY_S / PROC line. New cases: P checks the preflight (5 flagged files decoded by live processes) and the watcher output
# (judged by j_envwatch.verdict, the gate's own rule); N9 stand-in decoder dies on the trap (os._exit(139), a native fault):
# fail closed, and a Raptor-like spawn pool over the test files then completes; N10 decoder hangs on the trap: preflight timeout,
# fail closed; N11 decoder raises a Python exception on the trap: DEC_OK (pydicom / the stages handle it as in BASE); N12 one
# stage hop launched with -E (a J3 form): that chain loses JPEG Lossless and the watcher flags it, verdict FAIL; N13 no test root:
# fail closed. Run on a pre-fix J kernel (old DEC cell, no PRE line) the new checks FAIL (fail-before evidence).
#   usage: python j_local_harness.py J_KERNEL_DIR [J_KERNEL_DIR ...]      -> J_HARNESS PASS n/n per kernel, exit 0 if all pass
import os, re, sys, ast, json, base64, shutil, hashlib, tempfile, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True                                   # no __pycache__ next to the scripts
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tests'))
import make_j          # noqa: E402  (units_of / DEC header only)
import stub_wheels     # noqa: E402
import j_envwatch      # noqa: E402  (the gate's envwatch rule)

JPEG = os.path.normpath(os.path.join(HERE, '..'))
FIXTURES = os.path.join(JPEG, 'f20j', 'selftest_fixtures.json')
SIM = os.path.join(HERE, 'tests', 'j_sim.py')
PATH_LINE = "    INPUT, TARGET, WORK, SLUG = '/kaggle/input', '/tmp/_dec', '/kaggle/working', 'rsna-knee-offline-codecs-r09'"
TIME_LINE = '    PIP_TIMEOUT_S, TEST_TIMEOUT_S = 300, 300'
PRE_LINE = "    PRE_TIMEOUT_S, WATCH_EVERY_S, PROC = 600, 1.0, '/proc'"
COMP = 'rsna-knee-abnormality-detection'
S6F_PATH_LINE = ("    W, ROOT, DEC, INPUT, SLUG = '/kaggle/working', '/tmp/s6f', '/tmp/s6f_dec', '/kaggle/input', "
                 "'rsna-knee-offline-codecs-r09'")
SLUG = 'rsna-knee-offline-codecs-r09'


def norm(p):
    return os.path.normcase(os.path.normpath(p)) if p else p


def one_line_sub(text, prefix, new_line):
    lines = text.split('\n')
    hits = [i for i, ln in enumerate(lines) if ln.startswith(prefix)]
    assert len(hits) == 1, (prefix, len(hits))
    lines[hits[0]] = new_line
    return '\n'.join(lines)


def write_fixtures(d):
    """the 3 embedded self-test DICOMs + one NEW JPEG Lossless file (made here with ../jll.py) on disk, + expected.json."""
    import io, importlib.util
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.encaps import encapsulate
    from pydicom.uid import JPEGLosslessSV1, MRImageStorage, UID
    os.makedirs(d, exist_ok=True)
    exp = {}
    for name, (b64, dtype, shape, digest) in json.load(open(FIXTURES)).items():
        open(os.path.join(d, name + '.dcm'), 'wb').write(base64.b64decode(b64))
        exp[name] = [dtype, shape, digest]
    spec = importlib.util.spec_from_file_location('jll', os.path.join(JPEG, 'jll.py'))
    jll = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jll)
    rng = np.random.default_rng(20261015)
    yy, xx = np.mgrid[0:48, 0:64]
    a = np.clip(np.round(30000 + 20000 * np.sin(xx / 5.0) * np.cos(yy / 9.0) + rng.normal(0, 400, (48, 64))), 0, 65535).astype(np.int64)
    a[0, 0], a[-1, -1] = 0, 65535
    stream = jll.encode_sv1(a, 16)
    assert np.array_equal(jll.decode_sv1_reference(stream), a)
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = MRImageStorage
    sop = UID('1.2.826.0.1.3680043.8.498.1015.77.1')
    fm.MediaStorageSOPInstanceUID = sop
    fm.TransferSyntaxUID = JPEGLosslessSV1
    ds = Dataset()
    ds.file_meta = fm
    ds.SOPClassUID, ds.SOPInstanceUID, ds.Modality = MRImageStorage, sop, 'MR'
    ds.StudyInstanceUID, ds.SeriesInstanceUID = sop + '.1', sop + '.2'
    ds.Rows, ds.Columns = a.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 'MONOCHROME2'
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.PixelData = encapsulate([stream])
    ds['PixelData'].VR = 'OB'
    ds['PixelData'].is_undefined_length = True
    ds.save_as(os.path.join(d, 'JLL70_16U_X.dcm'), enforce_file_format=True)
    want = np.ascontiguousarray(a.astype('<u2'))
    exp['JLL70_16U_X'] = ['uint16', list(want.shape), hashlib.sha256(want.tobytes()).hexdigest()]
    json.dump(exp, open(os.path.join(d, 'expected.json'), 'w'), indent=1)
    # the TRAP (review fix J1): a valid 37-row JPEG Lossless image; the crash / hang / raise stand-ins misbehave on 37 rows only
    t = np.clip(np.round(1000 + 900 * np.cos(np.mgrid[0:37, 0:41][1] / 4.0)), 0, 65535).astype(np.int64)
    ds.Rows, ds.Columns = t.shape
    ds.PixelData = encapsulate([jll.encode_sv1(t, 16)])
    ds['PixelData'].VR = 'OB'
    ds['PixelData'].is_undefined_length = True
    os.makedirs(os.path.join(d, 'trap'), exist_ok=True)
    ds.save_as(os.path.join(d, 'trap', 'TRAP.dcm'), enforce_file_format=True)
    return exp


LAYOUT = [('1.2.826.0.1.3680043.8.498.1', 'A', [('0001.dcm', 'U_16S'), ('0002.dcm', 'U_16S')]),
          ('1.2.826.0.1.3680043.8.498.1', 'B', [('0001.dcm', 'JLL70_16S'), ('0002.dcm', 'JLL57_12U')]),
          ('1.2.826.0.1.3680043.8.498.1', 'C', [('0001.dcm', 'TRAP'), ('0002.dcm', 'U_16S')]),
          ('1.2.826.0.1.3680043.8.498.2', 'D', [('0001.dcm', 'JLL70_16U_X')])]


def make_test_root(inp, fx_dir):
    """the competition layout the Raptor scripts search: INPUT/competitions/<comp>/{test.csv, test_series/<study>/<series>/*.dcm}."""
    root = os.path.join(inp, 'competitions', COMP)
    files = []
    for study, series, fs in LAYOUT:
        sd = os.path.join(root, 'test_series', study, series)
        os.makedirs(sd, exist_ok=True)
        for fn, src in fs:
            shutil.copyfile(os.path.join(fx_dir, 'trap', 'TRAP.dcm') if src == 'TRAP' else os.path.join(fx_dir, src + '.dcm'),
                            os.path.join(sd, fn))
            files.append(os.path.join(sd, fn))
    with open(os.path.join(root, 'test.csv'), 'w') as f:
        f.write('StudyInstanceUID\n' + '\n'.join(sorted(set(x[0] for x in LAYOUT))) + '\n')
    return files


def chains_of(nb, tmp):
    """the stage launches of this notebook, as the hop chains j_sim.py replays (env expressions taken from the notebook text)."""
    cells = nb['cells']
    units = make_j.units_of(cells)
    by = {n: t for n, k, ci, t in units}
    out, used = [], {}

    def assign_value(text, name, pred, func=None):
        tree = ast.parse(text)
        scope = tree
        if func:
            scope = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func)
        for node in ast.walk(scope):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                v = ast.get_source_segment(text, node.value)
                if pred(v):
                    return v
        raise KeyError(name)

    def free_names(expr):
        import builtins
        return sorted({n.id for n in ast.walk(ast.parse(expr, mode='eval')) if isinstance(n, ast.Name)} -
                      {'os', 'json', 'sys'} - set(dir(builtins)))

    def names_for(expr, tmp):
        return {n: (0 if n == 'gi' else '0' if n == 'gid' else None if n == 'px_dir' else tmp) for n in free_names(expr)}

    cell_texts = [(n, t) for n, k, ci, t in units if k == 'cell']
    for nm, script in (('Raptor r384', '/tmp/raptor_run.py'), ('Raptor r224', '/tmp/raptor224_run.py')):
        if any(("subprocess.run([sys.executable, '-u', '%s']" % script) in t for n, t in cell_texts):
            out.append({'name': nm, 'hops': [{'name': nm, 'via': 'subprocess', 'env_expr': None},
                                             {'name': nm + ' fork pool child', 'via': 'spawn'}]})
    r7 = [t for n, t in cell_texts if "str(SRC / 'probe_dual.py')" in t]
    if r7:
        e1 = assign_value(r7[0], 'env', lambda v: 'PYTHONPATH=str(CODE)' in v)
        pd_src = next(t for n, t in by.items() if n.endswith('>probe_dual.py'))
        e2 = assign_value(pd_src, 'env', lambda v: v.startswith('dict(os.environ, CUDA_VISIBLE_DEVICES'))
        out.append({'name': 'R7', 'hops': [{'name': 'R7 controller', 'via': 'subprocess', 'env_expr': e1, 'names': names_for(e1, tmp)},
                                           {'name': 'R7 worker', 'via': 'subprocess', 'env_expr': e2, 'names': names_for(e2, tmp)},
                                           {'name': 'R7 DataLoader child', 'via': 'spawn'}]})
        used['R7 controller'], used['R7 worker'] = e1, e2
    fl = [t for n, t in cell_texts if "f'{fleet_dir}/fleet_worker.py'" in t]
    if fl:
        e = assign_value(fl[0], 'env', lambda v: v.startswith('{**os.environ'))
        out.append({'name': 'fleet', 'hops': [{'name': 'fleet worker', 'via': 'subprocess', 'env_expr': e, 'names': names_for(e, tmp)}]})
        used['fleet worker'] = e
    rm = [t for n, t in cell_texts if "'/tmp/rm_verify.py'" in t]
    if rm:
        e = "dict(os.environ, CUDA_VISIBLE_DEVICES='')"
        assert e in rm[0]
        out.append({'name': 'RM verify', 'hops': [{'name': 'RM verify', 'via': 'subprocess', 'env_expr': e, 'names': {}}]})
    s6 = [t for n, t in cell_texts if '_S6F_SETUP, _S6F_ENV = _s6f_setup()' in t]
    if s6:
        drv = next(t for n, t in by.items() if n.endswith('>DRIVER_B64'))
        e = assign_value(drv, 'env', lambda v: v.startswith('dict(os.environ, CUDA_VISIBLE_DEVICES=str(gi)'), func='b_env')
        out.append({'name': 'S6fall', 'env_from': 's6f', 'hops': [{'name': 'S6F driver', 'via': 'subprocess', 'env_expr': '(S6F SETUP env)'},
                                                                  {'name': 'S6F B worker', 'via': 'subprocess', 'env_expr': e, 'names': names_for(e, tmp)},
                                                                  {'name': 'S6F DataLoader child', 'via': 'spawn'}]})
        used['S6F B worker'] = e
    return out, used, (s6[0] if s6 else None)


def flatten(res):
    """chain result -> (list of hop views, leaf dict)."""
    views, r = [], res
    while isinstance(r, dict):
        if 'error' in r and 'result' not in r:
            return views, {'error': r['error']}
        if r.get('via') == 'spawn':
            return views, r.get('result') or {}
        if 'result' in r:
            r = r['result']
            continue
        if 'view' in r:
            views.append(r['view'])
            r = r['next']
            continue
        return views, r
    return views, {}


def main(kdirs):
    py = sys.executable
    root = tempfile.mkdtemp(prefix='jharness_')
    stub = {}
    for v in ('good', 'broken', 'slow', 'shadow', 'crash', 'hang', 'raise'):
        stub[v] = stub_wheels.build(os.path.join(root, 'stub_' + v), v)
    assert stub['good'] == stub_wheels.build(os.path.join(root, 'stub_again'), 'good'), 'stand-in wheels not deterministic'
    fx_dir = os.path.join(root, 'fixtures')
    exp = write_fixtures(fx_dir)
    all_ok = True
    for kd in kdirs:
        meta = json.load(open(os.path.join(kd, 'kernel-metadata.json')))
        nbp = os.path.join(kd, meta['code_file'])
        nb = json.load(open(nbp, encoding='utf-8'))
        decs = [c for c in nb['cells'] if c['cell_type'] == 'code' and make_j.src(c).startswith(make_j.DEC_HEADER)]
        assert len(decs) == 1
        DEC = make_j.src(decs[0])
        tag = re.search(r"'tag': '([a-z0-9]+)'", DEC).group(1)
        kroot = os.path.join(root, tag)
        os.makedirs(os.path.join(kroot, 'code_dir'))
        chains, used, s6f_setup = chains_of(nb, os.path.join(kroot, 'code_dir'))
        print('J_HARNESS %s (%s, sha %s) chains %s' % (tag, nbp, hashlib.sha256(open(nbp, 'rb').read()).hexdigest()[:16],
                                                       [c['name'] for c in chains]), flush=True)
        for k, v in used.items():
            print('  env expression used for %-14s (from the notebook): %s' % (k, v[:150]), flush=True)
        R = []

        def chk(name, ok, info=''):
            R.append(bool(ok))
            print('  %s %s %s' % ('OK  ' if ok else 'FAIL', name, info), flush=True)

        base_pp = os.pathsep.join([os.path.join(kroot, 'kagglegym'), os.path.join(kroot, 'kaggle_lib')])
        for d in base_pp.split(os.pathsep):
            os.makedirs(d, exist_ok=True)

        def case(cname, wheels_variant=None, constants=None, timeouts=None, pre_import=False, unset_pp=False, no_dec=False,
                 s6f_mode=None, chains_sel=None, corrupt=False, real_names=False, test_root=True, pre_timeout=120, extra_chains=(),
                 hop_flags=None):
            cd = os.path.join(kroot, cname)
            inp, tgt, work = os.path.join(cd, 'input'), os.path.join(cd, 'tmp', '_dec'), os.path.join(cd, 'working')
            fproc = os.path.join(cd, 'proc')
            os.makedirs(work)
            os.makedirs(fproc)
            os.makedirs(os.path.dirname(tgt))
            tfiles = make_test_root(inp, fx_dir) if test_root else []
            ds_dir = os.path.join(inp, 'datasets', 'glorysing', SLUG)
            if wheels_variant:
                os.makedirs(ds_dir)
                for n in stub[wheels_variant]:
                    src = os.path.join(root, 'stub_' + wheels_variant, n)
                    dst_name = n
                    if real_names:                                     # stand-ins under the REAL wheel names (= tampered)
                        dst_name = {'pylibjpeg': list(make_j.PYPI_SHA)[0], 'pylibjpeg_libjpeg': list(make_j.PYPI_SHA)[1],
                                    'pylibjpeg_openjpeg': list(make_j.PYPI_SHA)[2]}[n.split('-')[0]]
                    shutil.copyfile(src, os.path.join(ds_dir, dst_name))
                if corrupt:
                    p = os.path.join(ds_dir, stub_wheels.NAMES['libjpeg'])
                    open(p, 'wb').write(b'PK\x03\x04 this is not a wheel')
            dec = DEC.replace(PATH_LINE, "    INPUT, TARGET, WORK, SLUG = %r, %r, %r, %r" % (inp, tgt, work, SLUG))
            assert dec != DEC
            wh = dict(stub[wheels_variant or 'good'])
            if corrupt:
                wh[stub_wheels.NAMES['libjpeg']] = hashlib.sha256(b'PK\x03\x04 this is not a wheel').hexdigest()
            if constants == 'stub':
                dec = one_line_sub(dec, '    WHEELS = {', '    WHEELS = %r' % wh)
                dec = one_line_sub(dec, '    TOP = [', '    TOP = %r' % (stub_wheels.TOP_STUB,))
            if timeouts:
                assert dec.count(TIME_LINE) == 1
                dec = dec.replace(TIME_LINE, '    PIP_TIMEOUT_S, TEST_TIMEOUT_S = %d, %d' % timeouts)
            if PRE_LINE in dec:                                        # absent in a pre-fix DEC cell (fail-before runs)
                dec = dec.replace(PRE_LINE, '    PRE_TIMEOUT_S, WATCH_EVERY_S, PROC = %d, 0.2, %r' % (pre_timeout, fproc))
            s6spec = None
            if s6f_mode and s6f_setup:
                s6root = os.path.join(cd, 's6f')
                st = s6f_setup
                assert st.count(S6F_PATH_LINE) == 1
                st = st.replace(S6F_PATH_LINE, "    W, ROOT, DEC, INPUT, SLUG = %r, %r, %r, %r, %r" % (
                    work, os.path.join(s6root, 'root'), os.path.join(s6root, 's6f_dec'), inp, SLUG))
                if s6f_mode == 'stub':
                    st = one_line_sub(st, '    WHEELS = {', '    WHEELS = %r' % dict(stub['good']))
                    st = one_line_sub(st, '    TOP = [', '    TOP = %r' % (stub_wheels.TOP_STUB,))
                s6spec = {'setup_text': st, 'work': work, 's6f_dec': os.path.join(s6root, 's6f_dec')}
            sel = [c for c in chains if chains_sel is None or c['name'] in chains_sel]
            if not s6spec:
                sel = [c for c in sel if c.get('env_from') != 's6f']
            if hop_flags:
                sel = [dict(c, hops=[dict(h, py_flags=hop_flags) if i == 0 else h for i, h in enumerate(c['hops'])]) for c in sel]
            sel = sel + [dict(c, pool_files=tfiles) for c in extra_chains]
            spec = {'dec_text': '' if no_dec else dec, 'pre_import_pydicom': pre_import, 'chains': sel, 'fixtures_dir': fx_dir,
                    'target': tgt, 'work': work, 'inproc': True, 's6f': s6spec, 'fake_proc': fproc, 'watch_wait_s': 1.5}
            sp = os.path.join(cd, 'spec.json')
            json.dump(spec, open(sp, 'w'))
            env = dict(os.environ)
            env.pop('PYTHONPATH', None)
            if not unset_pp:
                env['PYTHONPATH'] = base_pp
            env['PYTHONDONTWRITEBYTECODE'] = '1'
            r = subprocess.run([py, '-u', SIM, '--nb', sp], env=env, cwd=cd, capture_output=True, text=True, timeout=1800)
            line = next((ln for ln in r.stdout.splitlines() if ln.startswith('J_SIM_NB ')), None)
            if r.returncode != 0 or line is None:
                return {'sim_error': 'rc %s %s' % (r.returncode, r.stderr[-1500:])}, tgt, s6spec
            return json.loads(line[len('J_SIM_NB '):]), tgt, s6spec

        def leaf_ok(res, where):
            views, lf = flatten(res)
            d = lf.get('decode') or {}
            return (all(d.get(k) is True for k in exp) and norm(lf.get('libjpeg_dir')) == norm(where)), views, lf

        def leaf_base(res):
            views, lf = flatten(res)
            d = lf.get('decode') or {}
            jl = [k for k in exp if k.startswith('JLL')]
            return (d.get('U_16S') is True and all(isinstance(d.get(k), str) and 'RuntimeError' in d.get(k) and 'pylibjpeg' in d.get(k)
                                                    for k in jl) and lf.get('libjpeg_dir') is None), views, lf

        def dec_lines(out):
            s = out.get('dec_stdout') or ''
            return ([ln for ln in s.splitlines() if ln.startswith('DEC_OK ')], [ln for ln in s.splitlines() if ln.startswith('DEC_FALLBACK ')],
                    [ln for ln in s.splitlines() if ln.startswith('J_UNAVAILABLE ')], 'Traceback' in s)

        # ---- BASE: no DEC cell at all (what the base notebook does) -------------------------------------------------------------
        base, _, _ = case('base_nodec', no_dec=True, wheels_variant='good', constants='stub')
        base_leaf_pp = {}
        okb = 'sim_error' not in base
        for c in chains:
            if c.get('env_from') == 's6f':
                continue
            ok, views, lf = leaf_base(base['chains'].get(c['name'], {})) if okb else (False, [], {})
            base_leaf_pp[c['name']] = (views and [v.get('PYTHONPATH') for v in views], lf.get('PYTHONPATH'))
            okb = okb and ok
        chk('BASE (no DEC cell): every chain loses JPEG Lossless (no pylibjpeg plugin), the uncompressed control decodes',
            okb, base.get('sim_error', ''))

        # ---- P: positive (stand-in wheels, DEC constants pointed at them) -----------------------------------------------------
        P, tgt, s6 = case('P_dec_ok', wheels_variant='good', constants='stub', s6f_mode='stub')
        if 'sim_error' in P:
            chk('P runs', False, P['sim_error'])
        else:
            okl, fb, un, tb = dec_lines(P)
            rc = P.get('receipt') or {}
            chk('P: one DEC_OK line, no DEC_FALLBACK / J_UNAVAILABLE / Traceback, cell raised nothing',
                len(okl) == 1 and not fb and not un and not tb and P['dec_exception'] is None, (len(okl), P['dec_exception']))
            d = json.loads(okl[0][len('DEC_OK '):]) if okl else {}
            chk('P: DEC_OK reports versions 2.1.0 / 2.4.0 / 2.6.0, pydicom + numpy + python, plugin pylibjpeg for .57 / .70, 3/3 exact',
                d.get('versions') == {'pylibjpeg': '2.1.0', 'libjpeg': '2.4.0', 'openjpeg': '2.6.0'} and d.get('pydicom') and d.get('numpy')
                and d.get('python') and all('pylibjpeg' in v for v in (d.get('plugins') or {'x': []}).values()) and len(d.get('plugins') or {}) == 2
                and d.get('decode') == {'JLL57_12U': True, 'JLL70_16S': True, 'U_16S': True}, {k: d.get(k) for k in ('versions', 'pydicom', 'python', 'seconds')})
            chk('P: os.environ PYTHONPATH == TARGET + pathsep + before; sys.path[0] == TARGET; only _DEC_RECEIPT left in the namespace',
                P['pythonpath_after'] == os.pathsep.join([tgt, base_pp]) and norm(P['sys_path_after'][0]) == norm(tgt) and
                P['sys_path_after'][1:] == P['sys_path_before'] and P['namespace'] == ['_DEC_RECEIPT'],
                (P['pythonpath_after'], P['namespace']))
            chk('P: dec_receipt.json ok, wheels found by sha, target == the expected entries, decoders imported from TARGET, sys_path_inserted',
                rc.get('ok') is True and len(rc.get('wheels') or {}) == 3 and rc.get('target_files') == stub_wheels.TOP_STUB and
                all(norm(v) == norm(tgt) for v in ((rc.get('selftest') or {}).get('dirs') or {'x': None}).values()) and
                rc.get('sys_path_inserted') is True and rc.get('pip_rc') == 0, rc.get('why'))
            for c in chains:
                where = tgt
                if c.get('env_from') == 's6f':
                    where = s6['s6f_dec']
                ok, views, lf = leaf_ok(P['chains'].get(c['name'], {}), where)
                vpp = [v.get('PYTHONPATH') for v in views]
                chk('P chain %-11s: every hop process has TARGET on its import path and the leaf (%s) decodes 4/4 exactly with libjpeg from %s' % (
                    c['name'], c['hops'][-1]['name'], 'S6F own dir' if c.get('env_from') else 'TARGET'),
                    ok and all(v.get('target_on_sys_path') for v in views), {'leaf': lf.get('decode'), 'pp': [p and p[:60] for p in vpp[:3]]})
            il = P.get('inproc') or {}
            chk('P in-process (notebook process, sys.path): decodes 4/4 exactly, libjpeg from TARGET',
                all((il.get('decode') or {}).get(k) is True for k in exp) and norm(il.get('libjpeg_dir')) == norm(tgt), il.get('decode'))
            pf = rc.get('preflight') or {}
            dpf = (json.loads(okl[0][len('DEC_OK '):]) if okl else {}).get('preflight') or {}
            chk('P preflight (J1): fake test root found and scanned (2 studies, 4 series dirs, 7 files; probes = first + last per dir), '
                'flagged = the 5 files of the 3 dirs with a compressed probe, all 5 decoded by live processes, same in DEC_OK',
                pf.get('ok') is True and pf.get('n_studies') == 2 and pf.get('n_groups') == 4 and pf.get('n_files') == 7 and
                pf.get('n_flagged') == 5 and pf.get('decoded_ok') == 5 and pf.get('decoded_err') == 0 and
                norm(pf.get('test_root')) == norm(os.path.join(os.path.dirname(os.path.dirname(tgt)), 'input', 'competitions', COMP)) and
                dpf.get('n_flagged') == 5 and dpf.get('ok') is True,
                {k: pf.get(k) for k in ('n_studies', 'n_groups', 'n_files', 'n_flagged', 'decoded_ok', 'decoded_err', 'probe_ts', 'seconds')})
            ew = P.get('envwatch') or {}
            n_hops = sum(1 for c in chains if not c.get('env_from') for h in c['hops'])
            ok_v, pr_v, info_v = j_envwatch.verdict(ew, ['j_sim.py'], target=tgt, sep=os.pathsep)
            recs = [r for r in ew.get('records') or [] if r.get('scope')]
            chk('P envwatch (J2): the DEC watcher thread runs (dec_envwatch thread alive, receipt "started") and records every simulated '
                'stage process (>= %d hop / spawn processes) as a descendant with TARGET on PYTHONPATH; j_envwatch.verdict PASS' % n_hops,
                ok_v and rc.get('envwatch') == 'started' and 'dec_envwatch' in (P.get('threads') or []) and len(recs) >= n_hops and
                all(r.get('dec_on_path') and not r.get('flags') for r in recs) and
                {r.get('script') for r in recs} >= {'j_sim.py', '-c'},
                {'verdict': pr_v[:2] or info_v.get('scripts'), 'n_scope': len(recs)})
            if s6:
                s = P.get('s6f') or {}
                chk('P S6F SETUP after J: S6F_DEC_OK (its pip --target installs although /tmp/_dec is on PYTHONPATH), S6F env PYTHONPATH '
                    '== s6f_dec + TARGET + base', s.get('ok') is True and s.get('dec_ok') is True and
                    s.get('pythonpath_s6f') == os.pathsep.join([s6['s6f_dec'], tgt, base_pp]) and 'S6F_DEC_OK' in (s.get('stdout') or ''),
                    {'pythonpath_s6f': s.get('pythonpath_s6f'), 'why': s.get('dec_why')})
        if s6f_setup:
            # S6F's own decoder install fails (its REAL constants vs the stand-ins) but ours works: S6fall still decodes via TARGET
            Q, tgt2, s62 = case('Q_s6f_own_dec_fails', wheels_variant='good', constants='stub', s6f_mode='real', chains_sel=['S6fall'])
            s = Q.get('s6f') or {}
            ok, views, lf = leaf_ok(Q.get('chains', {}).get('S6fall', {}), tgt2) if 'sim_error' not in Q else (False, [], {})
            chk('Q: S6F own DEC falls back (S6F_DEC_FALLBACK, real sha vs stand-ins) but J holds: S6fall chain decodes 4/4 via TARGET',
                ok and s.get('dec_ok') is False and 'S6F_DEC_FALLBACK' in (s.get('stdout') or '') and
                s.get('pythonpath_s6f') == os.pathsep.join([tgt2, base_pp]), {'leaf': lf.get('decode'), 'pp': s.get('pythonpath_s6f')})

        # ---- fail-closed cases: DEC must change nothing, print DEC_FALLBACK + J_UNAVAILABLE, never raise ------------------------
        neg = [('N1_no_dataset', dict(wheels_variant=None, constants=None), 'PyPI-identical wheels not found', None),
               ('N2_tampered_wheels_real_names', dict(wheels_variant='good', constants=None, real_names=True), 'PyPI-identical wheels not found', None),
               ('N3_selftest_not_exact', dict(wheels_variant='broken', constants='stub'), 'self-test decode not exact', None),
               ('N4_pip_fails', dict(wheels_variant='good', constants='stub', corrupt=True), 'pip exit', None),
               ('N5_shadow_package_in_target', dict(wheels_variant='shadow', constants='stub'), 'target dir content differs', None),
               ('N6_selftest_timeout', dict(wheels_variant='slow', constants='stub', timeouts=(120, 3)), 'TimeoutExpired', None)]
        for cname, kw, why, _ in neg:
            sel = None if cname == 'N1_no_dataset' else ['Raptor r384']
            N, tgtn, _ = case(cname, chains_sel=sel, s6f_mode=None, **kw)
            if 'sim_error' in N:
                chk('%s runs' % cname, False, N['sim_error'])
                continue
            okl, fb, un, tb = dec_lines(N)
            rc = N.get('receipt') or {}
            leafs = []
            for c in [c for c in chains if (sel is None or c['name'] in sel) and not c.get('env_from')]:
                okb2, views, lf = leaf_base(N['chains'].get(c['name'], {}))
                same_pp = [v.get('PYTHONPATH') for v in views] == base_leaf_pp.get(c['name'], (None,))[0]
                leafs.append(okb2 and same_pp)
            il = N.get('inproc') or {}
            chk('%s: DEC_FALLBACK + J_UNAVAILABLE, no DEC_OK, nothing raised / no Traceback, why "%s", PYTHONPATH + sys.path unchanged, '
                'every chain == BASE (%d chains)' % (cname, why, len(leafs)),
                not okl and len(fb) == 1 and len(un) == 1 and not tb and N['dec_exception'] is None and rc.get('ok') is False and
                why in (rc.get('why') or '') and N['pythonpath_after'] == N['pythonpath_before'] and N['sys_path_after'] == N['sys_path_before']
                and N['namespace'] == ['_DEC_RECEIPT'] and all(leafs) and leafs and (il.get('decode') or {}).get('U_16S') is True and
                il.get('libjpeg_dir') is None, {'why': (rc.get('why') or '')[:90], 'seconds': rc.get('seconds')})

        # ---- N7: pydicom imported before the DEC cell (in-process plugin table already built) -----------------------------------
        N7, tgt7, _ = case('N7_pydicom_preimported', wheels_variant='good', constants='stub', chains_sel=['Raptor r384'], pre_import=True)
        okl, fb, un, tb = dec_lines(N7)
        rc = N7.get('receipt') or {}
        ok, views, lf = leaf_ok((N7.get('chains') or {}).get('Raptor r384', {}), tgt7)
        chk('N7 pydicom pre-imported: DEC_OK, subprocesses decode via PYTHONPATH, in-process sys.path left as it was (noted)',
            len(okl) == 1 and ok and rc.get('sys_path_inserted') is False and 'already imported' in (rc.get('sys_path_note') or '') and
            N7.get('sys_path_after') == N7.get('sys_path_before'),
            {'note': rc.get('sys_path_note'), 'ins': rc.get('sys_path_inserted'), 'leaf': lf.get('decode'), 'okl': len(okl),
             'err': N7.get('sim_error', '')[-600:], 'sp_same': N7.get('sys_path_after') == N7.get('sys_path_before')})
        # ---- N8: no PYTHONPATH in the notebook environment -----------------------------------------------------------------------
        N8, tgt8, _ = case('N8_no_pythonpath', wheels_variant='good', constants='stub', unset_pp=True, chains_sel=['Raptor r384', 'R7'])
        okl, fb, un, tb = dec_lines(N8)
        oks = [leaf_ok((N8.get('chains') or {}).get(c, {}), tgt8)[0] for c in ['Raptor r384', 'R7'] if any(x['name'] == c for x in chains)]
        chk('N8 PYTHONPATH unset before: DEC_OK, PYTHONPATH == TARGET exactly (no empty entry), chains decode',
            len(okl) == 1 and N8.get('pythonpath_after') == tgt8 and N8.get('had_pythonpath') is False and all(oks) and oks,
            N8.get('pythonpath_after'))

        # ---- review fix J1: the preflight on the test files ------------------------------------------------------------------
        pool = [{'name': 'Raptor-like pool over the test files',
                 'hops': [{'name': 'Raptor r384 (pool parent)', 'via': 'subprocess', 'env_expr': None}]}]

        def closed(N, why):
            okl, fb, un, tb = dec_lines(N)
            rc = N.get('receipt') or {}
            return (not okl and len(fb) == 1 and len(un) == 1 and not tb and N.get('dec_exception') is None and rc.get('ok') is False
                    and why in (rc.get('why') or '') and N['pythonpath_after'] == N['pythonpath_before'] and
                    N['sys_path_after'] == N['sys_path_before'] and N['namespace'] == ['_DEC_RECEIPT'] and
                    'dec_envwatch' not in (N.get('threads') or []) and (N.get('envwatch') or {}).get('missing') is not None), rc

        N9, _, _ = case('N9_decoder_crash_on_a_test_file', wheels_variant='crash', constants='stub', chains_sel=['Raptor r384'],
                        extra_chains=pool)
        ok9, rc9 = closed(N9, 'preflight: decode process exit') if 'sim_error' not in N9 else (False, {})
        okb9, _, _ = leaf_base((N9.get('chains') or {}).get('Raptor r384', {}))
        _, pl9 = flatten((N9.get('chains') or {}).get(pool[0]['name'], {}))
        pr9 = pl9.get('pool_result') or {}
        chk('N9 native decoder fault on a hidden test file (stand-in os._exit(139) on the trap): fail closed (preflight decode process '
            'exit), nothing changed, no watcher; Raptor r384 chain == BASE; a Raptor-like spawn pool over the 7 test files COMPLETES '
            '(J off: JPEG Lossless lost as in BASE, no hang)',
            ok9 and okb9 and pr9.get('pool') == 'DONE' and len(pr9.get('got') or {}) == 7 and pr9['got'].get('A/0001.dcm') == 'ok' and
            str(pr9['got'].get('B/0001.dcm')).startswith('ERR') and str(pr9['got'].get('C/0001.dcm')).startswith('ERR'),
            {'why': (rc9.get('why') or N9.get('sim_error') or '')[:110], 'pool': pr9.get('pool'), 'got': len(pr9.get('got') or {})})
        N10, _, _ = case('N10_decoder_hang_on_a_test_file', wheels_variant='hang', constants='stub', chains_sel=['Raptor r384'],
                         pre_timeout=8)
        ok10, rc10 = closed(N10, 'preflight: decode timeout') if 'sim_error' not in N10 else (False, {})
        okb10, _, _ = leaf_base((N10.get('chains') or {}).get('Raptor r384', {}))
        chk('N10 decoder hang on a hidden test file: preflight decode timeout (cap 8 s here) -> fail closed, nothing changed, chain == BASE',
            ok10 and okb10 and (rc10.get('seconds') or 999) < 60, {'why': (rc10.get('why') or N10.get('sim_error') or '')[:90],
                                                                      'seconds': rc10.get('seconds')})
        N11, tgt11, _ = case('N11_decoder_raises_on_a_test_file', wheels_variant='raise', constants='stub', chains_sel=['Raptor r384'])
        okl, fb, un, tb = dec_lines(N11)
        pf11 = (N11.get('receipt') or {}).get('preflight') or {}
        ok11, _, lf11 = leaf_ok((N11.get('chains') or {}).get('Raptor r384', {}), tgt11)
        chk('N11 decoder raises a Python exception on a test file: DEC_OK (pydicom tries its other plugins, the stages catch the rest '
            'as in BASE), preflight counts it (4 ok, 1 err), chain decodes the fixtures 4/4',
            len(okl) == 1 and not fb and not un and pf11.get('decoded_ok') == 4 and pf11.get('decoded_err') == 1 and ok11,
            {'pf': {k: pf11.get(k) for k in ('decoded_ok', 'decoded_err', 'errs')}, 'leaf': lf11.get('decode')})
        # ---- review fixes J2 / J3: a stage launched with -E (ignores PYTHONPATH) is caught by the watcher ----------------------
        N12, tgt12, _ = case('N12_stage_launched_with_-E', wheels_variant='good', constants='stub', chains_sel=['Raptor r384'],
                             hop_flags=['-E'])
        okl, fb, un, tb = dec_lines(N12)
        okb12, views12, _ = leaf_base((N12.get('chains') or {}).get('Raptor r384', {}))
        ok_v, pr_v, _ = j_envwatch.verdict(N12.get('envwatch'), ['j_sim.py'], target=tgt12, sep=os.pathsep)
        chk('N12 a stage hop launched as python -E (J3 form): DEC_OK, but that chain loses JPEG Lossless like BASE and the watcher '
            'records the -E process; the gate rule (j_envwatch.verdict) FAILS naming -E',
            len(okl) == 1 and okb12 and not ok_v and any("'-E'" in x for x in pr_v),
            {'verdict': [x[:110] for x in pr_v[:2]]})
        N13, _, _ = case('N13_no_test_root', wheels_variant='good', constants='stub', chains_sel=['Raptor r384'], test_root=False)
        ok13, rc13 = closed(N13, 'preflight: no test root') if 'sim_error' not in N13 else (False, {})
        chk('N13 no competition test root under INPUT: fail closed (the preflight cannot vouch for the decoders), nothing changed',
            ok13, (rc13.get('why') or N13.get('sim_error') or '')[:100])
        ok = all(R)
        all_ok = all_ok and ok
        print('J_HARNESS %s %s %d/%d' % ('PASS' if ok else 'FAIL', tag, sum(R), len(R)), flush=True)
    shutil.rmtree(root, ignore_errors=True)
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
