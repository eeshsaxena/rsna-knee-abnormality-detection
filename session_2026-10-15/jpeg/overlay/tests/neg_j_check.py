#!/usr/bin/env python
# neg_j_check.py (2026-10-15, Claude subagent): offline tests of j_check.py. A simulated clean J commit output = the REAL base commit
# output (copied, every file and subdir) + one DEC_OK log entry before stage 1 + dec_receipt.json, log renamed to the J slug. The
# DEC receipt is the REAL one the DEC cell wrote on Kaggle in rsna-v224f20j-check v1 (f20j/out_jj_v1/dec_receipt.json, same docker
# pin) with the fields the generalised cell adds (python, sys_path_inserted / sys_path_head, selftest.python / pydicom_dir: MOCK
# values) and this variant's tag. The clean copy must PASS; each mutation must FAIL; the positive controls must PASS.
# Review fixes (J1 / J2): the mock also carries the DEC preflight fields (MOCK values for the 3 public studies: 15 series, 557
# files, all explicit VR LE as MEASURED), envwatch 'started', a MOCK dec_envwatch.json in the DEC cell's schema (every expected
# stage script of j_envwatch.EXPECT seen with /tmp/_dec, plus '-c' spawn children, pip, and one NON-descendant process without
# /tmp/_dec that the rule must ignore) and, for S6F, the base S6F receipts with /tmp/_dec added where the J run puts it.
#   usage: python tests/neg_j_check.py VARIANT_TAG BASE_OUT [--gate PATH_TO_j_check.py]   (--gate: run against another gate copy,
#   e.g. the pre-change one, to show which negatives it misses)
import os, sys, json, shutil, tempfile, subprocess
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OV = os.path.normpath(os.path.join(HERE, '..'))
sys.path.insert(0, OV)
import j_envwatch                                                # noqa: E402
args = [a for a in sys.argv[1:]]
GATE = os.path.join(OV, 'j_check.py')
if '--gate' in args:
    i = args.index('--gate')
    GATE = os.path.abspath(args[i + 1])
    del args[i:i + 2]
REAL_RECEIPT = os.path.join(OV, '..', 'f20j', 'out_jj_v1', 'dec_receipt.json')
TAG, BASE = args[0], args[1]
base_log = next(f for f in os.listdir(BASE) if f.endswith('.log') and 'cufile' not in f)
S6F = os.path.isfile(os.path.join(BASE, 's6f_setup_receipt.json'))


def receipt():
    r = json.load(open(REAL_RECEIPT, encoding='utf-8'))
    r['tag'] = TAG
    r['python'] = '3.13.15'                                    # the pin's python (MEASURED in rsna-jpeg-check); field is new
    r['sys_path_inserted'] = True                              # MOCK: the generalised cell's new fields
    r['sys_path_head'] = ['/tmp/_dec', '/kaggle/lib/kagglegym', '/kaggle/lib']
    r['selftest']['python'] = '3.13.15'
    r['selftest']['pydicom_dir'] = '/usr/local/lib/python3.13/dist-packages'
    r['envwatch'] = 'started'                                  # MOCK: review-fix fields
    r['preflight'] = {'ok': True, 'timeout_s': 600, 'test_root': '/kaggle/input/competitions/rsna-knee-abnormality-detection',
                      'series_dir': '/kaggle/input/competitions/rsna-knee-abnormality-detection/test_series', 'n_groups': 15,
                      'n_files': 557, 'probe_ts': {'1.2.840.10008.1.2.1': 30}, 'errors': [], 'n_studies': 3, 'n_flagged': 0,
                      'flagged_sample': [], 'decoded_ok': 0, 'decoded_err': 0, 'errs': [], 'seconds': 1.4}
    return r


def envwatch(r, drop=(), mut=None):
    """MOCK dec_envwatch.json in the DEC cell watcher's schema (counters summed from the records, as the watcher does)."""
    pp = r['pythonpath_after']
    code = '/tmp/probe/code'
    exe = '/usr/bin/python3'
    rows = [('raptor_run.py', [exe, '-u', '/tmp/raptor_run.py'], pp), ('raptor_run.py', [exe, '-u', '/tmp/raptor_run.py'], pp),
            ('-m pip', [exe, '-m', 'pip', 'install', '-q'], pp),
            ('probe_dual.py', [exe, '-u', '/tmp/probe/src/probe_dual.py', '--data'], code + ':' + pp),
            ('compact_infer.py', [exe, '-u', '/tmp/probe/src/compact_infer.py', '--data'], code + ':' + pp),
            ('-c', [exe, '-c', 'from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=5, pipe_handle=7)'], code + ':' + pp),
            ('raptor224_run.py', [exe, '-u', '/tmp/raptor224_run.py'], pp)]
    for s_ in j_envwatch.EXPECT.get(TAG, []):
        if s_ not in [x[0] for x in rows]:
            pre = '/tmp/s6f_dec:' if s_ in ('s6f_driver.py', 'k1_b_infer.py') else '/tmp/k12_fleet:' if s_ == 'fleet_worker.py' else ''
            rows.append((s_, [exe, '-u', '/tmp/x/' + s_], pre + pp))
    st = {'ok': True, 'kernel_pid': 100, 'started_unix': 1760500000.0, 'every_s': 1.0, 'target': '/tmp/_dec', 'sweeps': 480,
          'n_python': 0, 'n_scope_python': 0, 'n_scope_bad': 0, 'scripts': {}, 'bad': [], 'records': [], 'errors': 0}
    recs = [{'pid': 200 + i, 'ppid': 100, 't': 5.0 + i, 'script': sc, 'flags': [], 'argv': argv, 'environ': True,
             'pythonpath': p_, 'dec_on_path': '/tmp/_dec' in p_.split(':'), 'scope': True}
            for i, (sc, argv, p_) in enumerate(rows) if sc not in drop]
    recs.append({'pid': 7, 'ppid': 1, 't': 3.0, 'script': 'jupyter-lab', 'flags': [], 'argv': [exe, '/usr/local/bin/jupyter-lab'],
                 'environ': True, 'pythonpath': '/kaggle/lib', 'dec_on_path': False, 'scope': False})   # not ours: ignored
    if mut:
        mut(recs)
    for x in recs:
        x['dec_on_path'] = x['pythonpath'] is not None and '/tmp/_dec' in x['pythonpath'].split(':')
        bad = bool(x['flags']) or (x['environ'] and not x['dec_on_path'])
        c = st['scripts'].setdefault(x['script'], {'n': 0, 'scope': 0, 'scope_ok': 0, 'scope_bad': 0})
        c['n'] += 1
        st['n_python'] += 1
        if x['scope'] is not False:
            st['n_scope_python'] += 1
            c['scope'] += 1
            if bad:
                st['n_scope_bad'] += 1
                c['scope_bad'] += 1
                st['bad'].append(x)
            elif x['environ']:
                c['scope_ok'] += 1
    st['records'] = recs
    return st


def dec_line(r):
    st = r['selftest']
    short = {k: r.get(k) for k in ('tag', 'ok', 'why', 'seconds', 'pythonpath_after', 'sys_path_inserted', 'pip_rc', 'python',
                                   'envwatch')}
    pf = r.get('preflight') or {}
    short['preflight'] = {k: pf.get(k) for k in ('ok', 'n_studies', 'n_groups', 'n_files', 'n_flagged', 'decoded_ok', 'decoded_err',
                                                 'seconds')}
    short.update(pydicom=st['pydicom'], numpy=st['numpy'], versions=st['versions'], plugins=st['plugins'],
                 decode={k: v.get('exact') for k, v in st['decode'].items()})
    return 'DEC_OK ' + json.dumps(short)


def s6f_j(d, r):
    """the base S6F receipts as the J run writes them: /tmp/_dec kept after /tmp/s6f_dec (MOCK edit of the REAL base files)."""
    want = '/tmp/s6f_dec:' + r['pythonpath_after']
    p1, p2 = os.path.join(d, 's6f_setup_receipt.json'), os.path.join(d, 's6f', 's6f_driver_receipt.json')
    a, b = json.load(open(p1, encoding='utf-8')), json.load(open(p2, encoding='utf-8'))
    a['pythonpath_s6f'] = want
    b['env_isolation']['PYTHONPATH'] = want
    sp = b['env_isolation']['driver_sys_path']
    sp.insert(sp.index('/tmp/s6f_dec') + 1, '/tmp/_dec')
    json.dump(a, open(p1, 'w', encoding='utf-8'), indent=1)
    json.dump(b, open(p2, 'w', encoding='utf-8'), indent=1)


def make(mut=None, ew=None):
    d = tempfile.mkdtemp(prefix='jcheck_sim_')
    shutil.rmtree(d)
    shutil.copytree(BASE, d)
    entries = json.load(open(os.path.join(d, base_log), encoding='utf-8'))
    k = next(i for i, e in enumerate(entries) if e.get('stream_name') == 'stdout')
    r = receipt()
    lines = [dec_line(r)]
    name = 'rsna-sub-%s.log' % TAG
    if S6F:
        s6f_j(d, r)
    w = envwatch(r)
    if mut:
        lines, r, k, entries, name = mut(lines, r, k, entries, d, name)
    for j, ln in enumerate(lines):
        entries.insert(k + j, {'stream_name': 'stdout', 'time': 1.0, 'data': ln + '\n'})
    os.remove(os.path.join(d, base_log))
    json.dump(entries, open(os.path.join(d, name), 'w', encoding='utf-8'))
    if r is not None:
        json.dump(r, open(os.path.join(d, 'dec_receipt.json'), 'w'), indent=1)
    w = ew(r) if ew else w
    if w is not None:
        json.dump(w, open(os.path.join(d, 'dec_envwatch.json'), 'w'), indent=1)
    return d


RES = []


def run(name, mut, want_pass, extra=(), ew=None):
    d = make(mut, ew)
    p = subprocess.run([sys.executable, GATE, d, '--base-out', BASE] + list(extra), capture_output=True, text=True, encoding='utf-8')
    last = (p.stdout.strip().splitlines() or [''])[-1]
    got = p.returncode == 0 and 'J_CHECK PASS' in last
    ok = got == want_pass
    RES.append(ok)
    fails = [ln.strip()[:150] for ln in p.stdout.splitlines() if ln.strip().startswith('FAIL')]
    print('%s %-52s want %-4s got %-4s | %s' % ('OK  ' if ok else 'BAD ', name, 'PASS' if want_pass else 'FAIL', 'PASS' if got else 'FAIL',
                                              last if want_pass else (fails[:2] or [last])), flush=True)
    shutil.rmtree(d, ignore_errors=True)


def edit_csv(f, fn):
    def m(lines, r, k, entries, d, name):
        p = os.path.join(d, f)
        df = pd.read_csv(p, dtype={'StudyInstanceUID': str})
        df = fn(df)
        df.to_csv(p, index=False)
        return lines, r, k, entries, name
    return m


def num_col(df):
    return [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])][0]


def bump(eps):
    def f(df):
        c = num_col(df)
        df.loc[0, c] = df.loc[0, c] + eps
        return df
    return f


def rec_mut(fn):
    def m(lines, r, k, entries, d, name):
        fn(r)
        return [dec_line(r)], r, k, entries, name
    return m


def dec_mut(fn):
    def m(lines, r, k, entries, d, name):
        dd = json.loads(lines[0][len('DEC_OK '):])
        fn(dd)
        return ['DEC_OK ' + json.dumps(dd)], r, k, entries, name
    return m


def after_stage(lines, r, k, entries, d, name):
    k2 = next(i for i, e in enumerate(entries) if 'RAPTOR_STAGE_DONE' in e.get('data', ''))
    return lines, r, k2 + 1, entries, name


def drop_marker(tok, add=None):
    def m(lines, r, k, entries, d, name):
        for e in entries:
            if e.get('data', '').startswith(tok):
                e['data'] = (add + '\n') if add else 'x\n'
                break
        return lines, r, k, entries, name
    return m


print('NEG_J_CHECK %s base %s' % (TAG, BASE), flush=True)
run('clean simulated J output', None, True)
run('clean, --strict-bytes', None, True, ['--strict-bytes'])
run('raptor_raw.csv +4e-5 (GPU noise level)', edit_csv('raptor_raw.csv', bump(4e-5)), True)
if os.path.isfile(os.path.join(BASE, 'submission_fleetsolo.csv')):
    run('submission_fleetsolo.csv +2e-6 (raw fleet output, stage tolerance)', edit_csv('submission_fleetsolo.csv', bump(2e-6)), True)
    run('submission_H.csv +1e-5 (rank blend)', edit_csv('submission_H.csv', bump(1e-5)), False)
if os.path.isfile(os.path.join(BASE, 's6f', 'B', 'part0', 'predictions.csv')):
    run('S6F part predictions rows permuted (compared by id)', edit_csv('s6f/B/part0/predictions.csv', lambda df: df.iloc[::-1]), True)
run('DEC_FALLBACK + J_UNAVAILABLE instead of DEC_OK',
    lambda l, r, k, e, d, n: (['DEC_FALLBACK {"tag": "%s", "ok": false}' % TAG, 'J_UNAVAILABLE {"why": "x"}'], dict(r, ok=False), k, e, n), False)
run('DEC_OK + a J_UNAVAILABLE line', lambda l, r, k, e, d, n: (l + ['J_UNAVAILABLE {"why": "x"}'], r, k, e, n), False)
run('DEC_OK after RAPTOR_STAGE_DONE', after_stage, False)
run('two DEC_OK lines', lambda l, r, k, e, d, n: (l + l, r, k, e, n), False)
run('submission.csv +1e-5 on one cell', edit_csv('submission.csv', bump(1e-5)), False)
run('submission.csv rows reordered', edit_csv('submission.csv', lambda df: df.iloc[::-1]), False)
run('raptor_raw.csv +0.01', edit_csv('raptor_raw.csv', bump(0.01)), False)
run('ours_sub.csv missing', lambda l, r, k, e, d, n: (os.remove(os.path.join(d, 'ours_sub.csv')), (l, r, k, e, n))[1], False)
run('DEC_OK libjpeg 2.3.0', dec_mut(lambda dd: dd['versions'].update(libjpeg='2.3.0')), False)
run('DEC_OK self-test JLL70 not exact', dec_mut(lambda dd: dd['decode'].update(JLL70_16S=False)), False)
run('DEC_OK sys_path_inserted false (info only by default)', dec_mut(lambda dd: dd.update(sys_path_inserted=False)), True)
run('DEC_OK sys_path_inserted false, --require-inproc', dec_mut(lambda dd: dd.update(sys_path_inserted=False)), False, ['--require-inproc'])
run('clean, --require-inproc', None, True, ['--require-inproc'])
run('DEC_OK python 3.12', dec_mut(lambda dd: dd.update(python='3.12.3')), False)
run('DEC_OK tag of another kernel', dec_mut(lambda dd: dd.update(tag='v224f20j')), False)
run('receipt PYTHONPATH not extended', rec_mut(lambda r: r.update(pythonpath_after=r['pythonpath_before'])), False)
run('receipt target with an extra entry', rec_mut(lambda r: r['target_files'].append('numpy')), False)
run('receipt decoders not from /tmp/_dec', rec_mut(lambda r: r['selftest']['dirs'].update(libjpeg='/usr/lib')), False)
run('receipt 2 of 3 wheels', rec_mut(lambda r: r['wheels'].pop(sorted(r['wheels'])[0])), False)
run('no dec_receipt.json', lambda l, r, k, e, d, n: (l, None, k, e, n), False)
run('log named after the base slug', lambda l, r, k, e, d, n: (l, r, k, e, base_log), False)
run('R224_STAGE marker replaced by R224_SKIPPED', drop_marker('R224_STAGE', 'R224_SKIPPED only 5 min left'), False)
run('extra Traceback in the J log', lambda l, r, k, e, d, n: (l + ['Traceback (most recent call last):'], r, k, e, n), False)
# ---- review fixes J1 / J2 -----------------------------------------------------------------------------------------------------
run('J1 DEC_OK preflight scanned no file (n_files 0)', dec_mut(lambda dd: dd['preflight'].update(n_files=0)), False)
run('J1 DEC_OK without preflight fields (pre-fix DEC cell)', dec_mut(lambda dd: dd.pop('preflight')), False)
run('J1 receipt preflight series dir not a test dir', rec_mut(lambda r: r['preflight'].update(series_dir='/kaggle/input/x')), False)
run('J2 receipt envwatch not started', rec_mut(lambda r: r.update(envwatch='not started: RuntimeError')), False)
run('J2 no dec_envwatch.json', None, False, ew=lambda r: None)
run('J2 envwatch: watcher not ok (no /proc)', None, False, ew=lambda r: dict(envwatch(r), ok=False, why='no /proc'))
run('J2 envwatch: compact_infer.py never seen', None, False, ew=lambda r: envwatch(r, drop=('compact_infer.py',)))
run('J2 envwatch: raptor_run.py without /tmp/_dec', None, False,
    ew=lambda r: envwatch(r, mut=lambda R: R[0].update(pythonpath='/kaggle/lib/kagglegym:/kaggle/lib')))
run('J2 envwatch: probe_dual.py launched with -I', None, False,
    ew=lambda r: envwatch(r, mut=lambda R: [x.update(flags=['-I']) for x in R if x['script'] == 'probe_dual.py']))
run('J2 envwatch: an extra descendant -c process without /tmp/_dec', None, False,
    ew=lambda r: envwatch(r, mut=lambda R: R.append(dict(R[0], pid=999, script='-c', pythonpath='/kaggle/lib'))))
run('J2 same, --envwatch-waive=-c (main-session waiver)', None, True, ['--envwatch-waive=-c'],
    ew=lambda r: envwatch(r, mut=lambda R: R.append(dict(R[0], pid=999, script='-c', pythonpath='/kaggle/lib'))))
run('J2 envwatch: counters disagree with the records', None, False,
    ew=lambda r: (lambda w: (w['scripts']['raptor_run.py'].update(n=1), w)[1])(envwatch(r)))
if S6F:
    def s6_mut(fn):
        def m(lines, r, k, entries, d, name):
            fn(d, r)
            return lines, r, k, entries, name
        return m

    def setup_base_pp(d, r):
        p1 = os.path.join(d, 's6f_setup_receipt.json')
        a = json.load(open(p1, encoding='utf-8'))
        a['pythonpath_s6f'] = json.load(open(os.path.join(BASE, 's6f_setup_receipt.json'), encoding='utf-8'))['pythonpath_s6f']
        json.dump(a, open(p1, 'w', encoding='utf-8'))

    def drv(fn):
        def f(d, r):
            p2 = os.path.join(d, 's6f', 's6f_driver_receipt.json')
            b = json.load(open(p2, encoding='utf-8'))
            fn(b['env_isolation'])
            json.dump(b, open(p2, 'w', encoding='utf-8'))
        return f
    run('J2 S6F setup PYTHONPATH without /tmp/_dec (the base value)', s6_mut(setup_base_pp), False)
    run('J2 S6F driver env PYTHONPATH without /tmp/_dec', s6_mut(drv(lambda e: e.update(PYTHONPATH=e['PYTHONPATH'].replace(
        '/tmp/_dec:', '')))), False)
    run('J2 S6F driver libjpeg from /tmp/_dec (S6F own decoders not used)', s6_mut(drv(lambda e: e['mods'].update(
        libjpeg=['2.4.0', '/tmp/_dec']))), False)
    run('J2 S6F driver sys.path without /tmp/_dec', s6_mut(drv(lambda e: e['driver_sys_path'].remove('/tmp/_dec'))), False)
ok = all(RES)
print('NEG_J_CHECK %s %s %d/%d' % ('PASS' if ok else 'FAIL', TAG, sum(RES), len(RES)), flush=True)
sys.exit(0 if ok else 1)
