#!/usr/bin/env python
# neg_j_check.py (2026-10-15, Claude subagent): offline tests of j_check.py. A simulated clean J commit output = the REAL base commit
# output (copied, every file and subdir) + one DEC_OK log entry before stage 1 + dec_receipt.json, log renamed to the J slug. The
# DEC receipt is the REAL one the DEC cell wrote on Kaggle in rsna-v224f20j-check v1 (f20j/out_jj_v1/dec_receipt.json, same docker
# pin) with the fields the generalised cell adds (python, sys_path_inserted / sys_path_head, selftest.python / pydicom_dir: MOCK
# values) and this variant's tag. The clean copy must PASS; each mutation must FAIL; the positive controls must PASS.
#   usage: python tests/neg_j_check.py VARIANT_TAG BASE_OUT
import os, sys, json, shutil, tempfile, subprocess
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OV = os.path.normpath(os.path.join(HERE, '..'))
GATE = os.path.join(OV, 'j_check.py')
REAL_RECEIPT = os.path.join(OV, '..', 'f20j', 'out_jj_v1', 'dec_receipt.json')
TAG, BASE = sys.argv[1], sys.argv[2]
base_log = next(f for f in os.listdir(BASE) if f.endswith('.log') and 'cufile' not in f)


def receipt():
    r = json.load(open(REAL_RECEIPT, encoding='utf-8'))
    r['tag'] = TAG
    r['python'] = '3.13.15'                                    # the pin's python (MEASURED in rsna-jpeg-check); field is new
    r['sys_path_inserted'] = True                              # MOCK: the generalised cell's new fields
    r['sys_path_head'] = ['/tmp/_dec', '/kaggle/lib/kagglegym', '/kaggle/lib']
    r['selftest']['python'] = '3.13.15'
    r['selftest']['pydicom_dir'] = '/usr/local/lib/python3.13/dist-packages'
    return r


def dec_line(r):
    st = r['selftest']
    short = {k: r.get(k) for k in ('tag', 'ok', 'why', 'seconds', 'pythonpath_after', 'sys_path_inserted', 'pip_rc', 'python')}
    short.update(pydicom=st['pydicom'], numpy=st['numpy'], versions=st['versions'], plugins=st['plugins'],
                 decode={k: v.get('exact') for k, v in st['decode'].items()})
    return 'DEC_OK ' + json.dumps(short)


def make(mut=None):
    d = tempfile.mkdtemp(prefix='jcheck_sim_')
    shutil.rmtree(d)
    shutil.copytree(BASE, d)
    entries = json.load(open(os.path.join(d, base_log), encoding='utf-8'))
    k = next(i for i, e in enumerate(entries) if e.get('stream_name') == 'stdout')
    r = receipt()
    lines = [dec_line(r)]
    name = 'rsna-sub-%s.log' % TAG
    if mut:
        lines, r, k, entries, name = mut(lines, r, k, entries, d, name)
    for j, ln in enumerate(lines):
        entries.insert(k + j, {'stream_name': 'stdout', 'time': 1.0, 'data': ln + '\n'})
    os.remove(os.path.join(d, base_log))
    json.dump(entries, open(os.path.join(d, name), 'w', encoding='utf-8'))
    if r is not None:
        json.dump(r, open(os.path.join(d, 'dec_receipt.json'), 'w'), indent=1)
    return d


RES = []


def run(name, mut, want_pass, extra=()):
    d = make(mut)
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
ok = all(RES)
print('NEG_J_CHECK %s %s %d/%d' % ('PASS' if ok else 'FAIL', TAG, sum(RES), len(RES)), flush=True)
sys.exit(0 if ok else 1)
