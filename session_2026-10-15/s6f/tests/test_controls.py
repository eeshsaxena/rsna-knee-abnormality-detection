#!/usr/bin/env python
# test_controls.py (10-15 rfix, Claude subagent; finding S6F-R1): the control builds exercise what they claim, and the canary gate
# tells them apart. Offline, laptop CPU, -I. Control builds go to a temp dir (never the project kernel dir).
#   K1 the builder REFUSES the old documented cut control (--budget-h 0.03 --min-run-s 0): its RUN cell had stop_at = hard - 240 s
#      < start and returned SKIPPED_NO_TIME before Popen, so neither kill path ever ran (shown on its cells in K5)
#   K2 --kill-control builds (tag v224flips6fkill): receipt policy.control kill; cells 0-14 == the submit kernel; SETUP / BLEND ==
#      the submit kernel's except the tag; RUN = the submit RUN + only the kill-control splices; refused on the submit tag
#   K3 the default build is byte-identical to the project's submit-kernel files (notebook 9c0e2558, metadata, receipt, driver, diff)
#   K4 the kill-control cells executed as written (harness, paths redirected; this laptop has no process groups, so the RUN cell
#      takes its non-POSIX fallback kill): stub + 2 child sessions + 2 grandchildren started, RAN, killed at the hard deadline,
#      S6F_KILLED, S6F_KILL_CHECK, canary (ACL 0.5, the other 11 columns == V224-FLIP)
#   K5 the canary gate on mock outputs (the V224-FLIP v1 output + the executed cells, as mock_s6f_output.py; mock-only patches: the
#      Kaggle DEC_OK form, and for the kill mock the POSIX kill-check form with groups = the stub's real pids):
#      kill mock PASS (control from its receipt); old-cut mock (SKIPPED_NO_TIME, receipt without control) FAIL, also with
#      --control kill; skip mock (age 6.0 h) --control skip PASS, --control kill FAIL; kill mock --control skip FAIL; kill mock
#      with a survivor / with 1 killed group / without a hard deadline number FAIL
#   usage: py -3.12 -I tests/test_controls.py [--builder PATH] [--gate PATH]  -> TEST_CONTROLS PASS/FAIL n/m
import os, sys, io, json, time, shutil, signal, argparse, subprocess, contextlib
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s6f_harness as H

ap = argparse.ArgumentParser()
ap.add_argument('--builder', default=os.path.join(H.S6F, 'build_v224s6f.py'))
ap.add_argument('--gate', default=os.path.join(H.S6F, 'v224s6f_check.py'))
A = ap.parse_args()
T = os.path.join(HERE, '_tmp_controls')
shutil.rmtree(T, ignore_errors=True)
os.makedirs(T)
R = []
KTAG = 'v224flips6fkill'


def chk(name, ok, info=''):
    R.append(bool(ok))
    print('  %s %s %s' % ('OK  ' if ok else 'FAIL', name, info), flush=True)


def build(out, *args):
    os.makedirs(out, exist_ok=True)
    r = subprocess.run([sys.executable, '-I', A.builder, '--out', out] + list(args), capture_output=True, text=True, timeout=300)
    return r.returncode, (r.stdout + r.stderr).strip().splitlines()[-1:] or ['']


def nb_cells(path):
    return [H.src(c) for c in json.load(open(path, encoding='utf-8'))['cells']]


# ---- K1 -----------------------------------------------------------------------------------------------------------------
print('K1 old cut-control flags', flush=True)
rc, last = build(os.path.join(T, 'b_cut'), '--tag', 'v224flips6fcut', '--budget-h', '0.03', '--min-run-s', '0')
chk('K1: builder refuses --budget-h 0.03 --min-run-s 0 (RUN cell could never start the driver)', rc != 0 and 'never start the driver' in last[0],
    'rc %s: %s' % (rc, last[0][:160]))
# ---- K2 -----------------------------------------------------------------------------------------------------------------
print('K2 kill-control build', flush=True)
BK = os.path.join(T, 'b_kill')
rc, last = build(BK, '--tag', KTAG, '--kill-control', '--budget-h', '0.003', '--min-run-s', '-1000')
knb = os.path.join(BK, 'kernel_' + KTAG, 'rsna_sub_%s.ipynb' % KTAG)
krec = os.path.join(BK, 'build_receipt_%s.json' % KTAG)
dc = nb_cells(H.NB_PATH)
kc = nb_cells(knb) if rc == 0 and os.path.isfile(knb) else None
chk('K2: --kill-control builds; receipt policy.control == kill', rc == 0 and kc is not None and
    (json.load(open(krec)).get('policy') or {}).get('control') == 'kill', 'rc %s: %s' % (rc, last[0][:120]))
if kc:
    run_d, run_k = dc[16], kc[16]
    spl = {}
    exec(compile(open(os.path.join(H.S6F, 'cell_run_killctl_tpl.py'), encoding='utf-8').read(), 'killctl', 'exec'), spl)
    back = run_k.replace(spl['KILLCTL_HEADER'], '', 1).replace(spl['KILLCTL_DEFS'], '', 1).replace(spl['KILLCTL_CHECK'], '', 1) \
        .replace(spl['KILLCTL_LAUNCH'], "        p = subprocess.Popen([sys.executable, '-u', ROOT + '/s6f_driver.py', ", 1)
    pol_d = 'G_START_H, BUDGET_H, DEADLINE_H, STOP_MARGIN_S, MIN_RUN_S = 5.75, 3.3, 8.2, 240.0, 1200.0'
    pol_k = 'G_START_H, BUDGET_H, DEADLINE_H, STOP_MARGIN_S, MIN_RUN_S = 5.75, 0.003, 8.2, 240.0, -1000.0'
    chk('K2: cells 0-14 == the submit kernel; SETUP / BLEND == the submit kernel but the tag; RUN == submit RUN + the splices only',
        len(kc) == 18 and kc[:15] == dc[:15] and kc[15].replace(KTAG, H.TAG) == dc[15] and kc[17].replace(KTAG, H.TAG) == dc[17]
        and back.replace(pol_k, pol_d) == run_d and "ROOT + '/kill_stub.py'" in run_k and "'/s6f_driver.py'" not in run_k)
rc2, last2 = build(os.path.join(T, 'b_kill_submit'), '--kill-control', '--budget-h', '0.003', '--min-run-s', '-1000')
chk('K2: --kill-control refused on the submit tag', rc2 != 0, last2[0][:120])
# ---- K3 -----------------------------------------------------------------------------------------------------------------
print('K3 default build byte-identical to the submit kernel', flush=True)
BD = os.path.join(T, 'b_def')
rc, last = build(BD)
same = []
for rel in ('kernel_v224flips6f/rsna_sub_v224flips6f.ipynb', 'kernel_v224flips6f/kernel-metadata.json', 'build_receipt_v224flips6f.json',
            's6f_driver_built_v224flips6f.py', 'celldiff_v224flips6f.txt'):
    a, b = os.path.join(H.S6F, *rel.split('/')), os.path.join(BD, *rel.split('/'))
    same.append(os.path.isfile(b) and open(a, 'rb').read() == open(b, 'rb').read())
chk('K3: default build == the project files byte for byte (notebook sha 9c0e2558...)', rc == 0 and all(same) and
    H.sha_file(H.NB_PATH).startswith('9c0e2558d51e4adb'), same)


# ---- helpers for K4 / K5 ------------------------------------------------------------------------------------------------
def exec_cells(tree, texts, age_h, run_over=None, fake=None):
    setup_t, run_t, blend_t = texts
    for a, b in (run_over or {}).items():
        assert run_t.count(a) == 1, a
        run_t = run_t.replace(a, b)
    base = H.flip_base()
    shutil.copyfile(os.path.join(H.FLIP_OUT, 'submission.csv'), os.path.join(tree.work, 'submission.csv'))
    ns = {'__name__': '__main__', 'NB_T0': time.time() - age_h * 3600.0, 'sub': base.copy(), '_canary': {'r224_used': True, 'fired': []}}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(tree.redirect(setup_t), '<s6f setup>', 'exec'), ns)
    if fake:
        H.install_fake(tree, fake)
    with contextlib.redirect_stdout(buf):
        exec(compile(tree.redirect(run_t), '<s6f run>', 'exec'), ns)
        exec(compile(tree.redirect(blend_t), '<s6f blend>', 'exec'), ns)
    return ns, buf.getvalue()


def make_mock(dst, tree, text, tag, extra_lines=None, run_patch=None):
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(H.FLIP_OUT, dst, ignore=shutil.ignore_patterns('__pycache__'))
    os.replace(os.path.join(dst, 'rsna-sub-v224flip.log'), os.path.join(dst, 'rsna-sub-%s.log' % tag))
    dec = {'target': '/tmp/s6f_dec', 'pip_rc': 0, 'selftest': {'pydicom': '3.0.1', 'versions': {'pylibjpeg': '2.1.0', 'libjpeg': '2.4.0',
           'openjpeg': '2.6.0'}, 'decode': {'JLL57_12U': {'exact': True}, 'JLL70_16S': {'exact': True}, 'U_16S': {'exact': True}}}}
    p = os.path.join(tree.work, 's6f_setup_receipt.json')
    r = json.load(open(p))
    r.update(dec_ok=True, dec=dec, pythonpath_s6f='/tmp/s6f_dec')
    json.dump(r, open(p, 'w'), indent=1)
    rr = json.load(open(os.path.join(tree.work, 's6f_receipt.json')))
    rr['setup'].update(dec_ok=True, dec=dec, pythonpath_s6f='/tmp/s6f_dec')
    if run_patch:
        run_patch(rr['run'])
    json.dump(rr, open(os.path.join(tree.work, 's6f_receipt.json'), 'w'), indent=1)
    lines = []
    for ln in text.splitlines():
        if ln.startswith('S6F_DEC_FALLBACK '):
            ln = 'S6F_DEC_OK ' + json.dumps({'why': None, 'pip_rc': 0, 'pydicom': '3.0.1', 'versions': dec['selftest']['versions']})
        if ln.startswith('S6F_SETUP_OK '):
            ln = ln.replace('"dec_ok": false', '"dec_ok": true').replace('"pythonpath_s6f": null', '"pythonpath_s6f": "/tmp/s6f_dec"')
        if ln.startswith('S6F_KILL_CHECK ') and run_patch:
            ln = 'S6F_KILL_CHECK ' + json.dumps(rr['run'].get('kill_check'))
        lines.append(ln)
        if ln.startswith('S6F_START ') and extra_lines:
            lines += extra_lines
    for f in ('submission.csv', 'submission_v224flip.csv', 's6f_setup_receipt.json', 's6f_receipt.json'):
        shutil.copyfile(os.path.join(tree.work, f), os.path.join(dst, f))
    lp = os.path.join(dst, 'rsna-sub-%s.log' % tag)
    E = json.load(open(lp, encoding='utf-8'))
    t_end = E[-1]['time'] if E else 0.0
    E += [{'stream_name': 'stdout', 'time': t_end + 1 + 0.01 * i, 'data': ln + '\n'} for i, ln in enumerate(lines)]
    json.dump(E, open(lp, 'w', encoding='utf-8'))
    return rr


def gate(d, *extra):
    r = subprocess.run([sys.executable, '-I', A.gate, d] + list(extra), capture_output=True, text=True, timeout=900)
    v = [l for l in r.stdout.splitlines() if l.startswith('V224S6F_CHECK ')][-1:] or ['(no verdict) ' + r.stderr.strip()[-160:]]
    fails = [l.strip()[5:] for l in r.stdout.splitlines() if l.strip().startswith('FAIL')]
    return r.returncode, v[-1], fails


def gate_case(name, d, expect_pass, must=(), extra=()):
    rc, v, fails = gate(d, *extra)
    if expect_pass:
        ok = rc == 0 and 'PASS' in v
    else:
        ok = rc != 0 and 'FAIL' in v and all(any(m in f for f in fails) for m in must)
    chk('K5 %s %s: %s' % ('POSITIVE' if expect_pass else 'NEGATIVE', name, v), ok, '' if expect_pass else 'fails: ' + ' || '.join(f[:60] for f in fails[:4]))


def kill_leftovers(tree):
    """this laptop: the non-POSIX fallback kills only the stub; end its children / grandchildren here (pids from the stub files)."""
    pids = []
    try:
        s = json.load(open(os.path.join(tree.root, 'work', 'kill_stub_pids.json')))
        pids += list(s.get('children') or [])
    except Exception:
        pass
    for i in range(2):
        try:
            pids.append(json.load(open(os.path.join(tree.root, 'work', 'kill_child%d.json' % i))).get('grandchild'))
        except Exception:
            pass
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except Exception:
            pass


# ---- K4 -----------------------------------------------------------------------------------------------------------------
print('K4 kill-control cells executed (non-POSIX fallback here)', flush=True)
k_tree, k_text, k_rec = None, '', {}
if kc:
    k_tree = H.Tree(T, 'kill')
    t0 = time.time()
    try:
        ns, k_text = exec_cells(k_tree, kc[15:18], 4.6, run_over={'KILL_STUB_SLEEP_S = 900': 'KILL_STUB_SLEEP_S = 40'})
    finally:
        time.sleep(1)
        kill_leftovers(k_tree)
    dt = time.time() - t0
    k_rec = json.load(open(os.path.join(k_tree.work, 's6f_receipt.json')))
    run = k_rec.get('run') or {}
    sp = (run.get('kill_check') or {}).get('stub_pids') or {}
    chk('K4: stub tree started (stub pid, 2 child sessions, 2 grandchildren recorded)', isinstance(sp.get('stub'), int) and
        len(sp.get('children') or []) == 2 and len([g for g in sp.get('grandchildren') or [] if isinstance(g, int)]) == 2, sp)
    chk('K4: RAN, killed at the hard deadline (~11 s), S6F_START < S6F_KILLED < S6F_KILL_CHECK < S6F_STAGE_DONE', run.get('status') == 'RAN'
        and run.get('killed') is True and dt < 60 and all(k_text.find(m) >= 0 for m in ('S6F_START', 'S6F_KILLED', 'S6F_KILL_CHECK', 'S6F_STAGE_DONE'))
        and k_text.find('S6F_START') < k_text.find('S6F_KILLED') < k_text.find('S6F_KILL_CHECK') < k_text.find('S6F_STAGE_DONE'),
        '%.0f s, %s' % (dt, run.get('status')))
    sub = pd.read_csv(os.path.join(k_tree.work, 'submission.csv'), dtype=str)
    bas = pd.read_csv(os.path.join(H.FLIP_OUT, 'submission.csv'), dtype=str)
    chk('K4: canary (killed): mode V224FLIP_CANARY, ACL 0.5, the other 11 columns text == V224-FLIP', k_rec.get('mode') == 'V224FLIP_CANARY'
        and 'killed' in str((k_rec.get('blend') or {}).get('why')) and bool((sub['ACL'].astype(float) == 0.5).all())
        and all(sub[c].tolist() == bas[c].tolist() for c in H.LAB if c != 'ACL'), (k_rec.get('blend') or {}).get('why'))
else:
    chk('K4: kill-control kernel available', False)
# ---- K5 -----------------------------------------------------------------------------------------------------------------
print('K5 canary gate per control', flush=True)
MK = os.path.join(T, 'mock_kill')
if k_tree and k_rec:
    sp = ((k_rec.get('run') or {}).get('kill_check') or {}).get('stub_pids') or {}

    def posix_form(run):                                      # mock-only: what the POSIX kill path records on Kaggle
        g = [sp.get('stub')] + list(sp.get('children') or [])
        run['killed_groups'] = g
        run['kill_check'] = {'stub_pids': sp, 'alive': [], 'groups': g}
    stub_line = ['S6F_KILL_STUB started: stub %s, child sessions %s' % (sp.get('stub'), sp.get('children'))]
    make_mock(MK, k_tree, k_text, KTAG, extra_lines=stub_line, run_patch=posix_form)
    gate_case('kill mock (control kill from its receipt)', MK, True, extra=('--expect-canary', '--build', krec))
    gate_case('kill mock with --control skip', MK, False, must=('control skip',), extra=('--expect-canary', '--build', krec, '--control', 'skip'))
    for name, patch, must in (('kill mock, a grandchild survived', lambda r: r['kill_check'].update(alive=[[4242, 'S', 4243, 'python -c']]), 'control kill'),
                              ('kill mock, only 1 group killed', lambda r: r.update(killed_groups=r['killed_groups'][:1]), 'control kill'),
                              ('kill mock, no hard deadline number', lambda r: r.update(hard_deadline_age_h=None), 'hard deadline (a number)')):
        d = MK + '_' + name.split(', ')[1].replace(' ', '_')
        shutil.copytree(MK, d)
        p = os.path.join(d, 's6f_receipt.json')
        rr = json.load(open(p))
        patch(rr['run'])
        json.dump(rr, open(p, 'w'), indent=1)
        gate_case(name, d, False, must=(must,), extra=('--expect-canary', '--build', krec))
else:
    chk('K5: kill mock available', False)
# old cut control: the submit kernel's cells with the old cut policy (what build --budget-h 0.03 --min-run-s 0 produced)
cut_tree = H.Tree(T, 'cut')
pol_d = 'G_START_H, BUDGET_H, DEADLINE_H, STOP_MARGIN_S, MIN_RUN_S = 5.75, 3.3, 8.2, 240.0, 1200.0'
pol_c = 'G_START_H, BUDGET_H, DEADLINE_H, STOP_MARGIN_S, MIN_RUN_S = 5.75, 0.03, 8.2, 240.0, 0.0'
ns, c_text = exec_cells(cut_tree, dc[15:18], 4.6, run_over={pol_d: pol_c}, fake='fail')
c_run = json.load(open(os.path.join(cut_tree.work, 's6f_receipt.json'))).get('run') or {}
chk('K5: the old cut control never launched: SKIPPED_NO_TIME before Popen, killed / killed_groups never set',
    c_run.get('status') == 'SKIPPED_NO_TIME' and 'S6F_START' not in c_text and c_run.get('killed') is None, (c_run.get('status'), c_run.get('killed')))
MC = os.path.join(T, 'mock_cut')
make_mock(MC, cut_tree, c_text, H.TAG)
cut_rec = os.path.join(T, 'build_receipt_v224flips6fcut_old.json')
rb = json.load(open(os.path.join(H.S6F, 'build_receipt_v224flips6f.json')))
rb['policy'].update(budget_h=0.03, min_run_s=0.0)
json.dump(rb, open(cut_rec, 'w'), indent=1)
gate_case('old cut mock (receipt without control)', MC, False, must=('control kind known',), extra=('--expect-canary', '--build', cut_rec))
gate_case('old cut mock --control kill', MC, False, must=('control kill',), extra=('--expect-canary', '--build', cut_rec, '--control', 'kill'))
# skip control: kernel age 6.0 h > G
skip_tree = H.Tree(T, 'skip')
ns, s_text = exec_cells(skip_tree, dc[15:18], 6.0, fake='fail')
MS = os.path.join(T, 'mock_skip')
make_mock(MS, skip_tree, s_text, H.TAG)
gate_case('skip mock --control skip', MS, True, extra=('--expect-canary', '--control', 'skip'))
gate_case('skip mock --control kill', MS, False, must=('control kill',), extra=('--expect-canary', '--control', 'kill'))
shutil.rmtree(T, ignore_errors=True)
print('TEST_CONTROLS %s %d/%d' % ('PASS' if all(R) else 'FAIL', sum(R), len(R)), flush=True)
sys.exit(0 if all(R) else 1)
