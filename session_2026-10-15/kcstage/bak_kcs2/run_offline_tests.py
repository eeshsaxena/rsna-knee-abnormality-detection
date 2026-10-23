#!/usr/bin/env python
# run_offline_tests.py (10-15, Claude subagent, kcstage): re-runs every laptop-side test of the S6F+KC build and accepts only exact
# verdict lines (<NAME> PASS n/n, rc 0). Laptop CPU, light. The box-side tests (e2e cells, Linux kill, driver stop, box parity) are
# recorded in box_results/ (box/*.py; re-run with box/run_e2e_all.sh etc.) and checked here by their saved verdict lines.
#   py -3.12 -I run_offline_tests.py [--tags v224s6fkcstandin,v224s6fkcstandinj,v224s6fkcstandinskip]  -> OFFLINE_TESTS PASS n/n
import os, re, sys, json, glob, shutil, tempfile, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--tags', default='v224s6fkcstandin,v224s6fkcstandinj,v224s6fkcstandinskip')
A = ap.parse_args()
PY = [sys.executable, '-I']
R = []


def verdict(name, txt, rc):
    m = re.search(r'^%s (PASS|FAIL) (\d+)/(\d+)' % re.escape(name), txt.replace('\r', ''), re.M)
    ok = rc == 0 and m is not None and m.group(1) == 'PASS' and m.group(2) == m.group(3)
    R.append(ok)
    print('  %s %s %s' % ('OK  ' if ok else 'FAIL', name, m.group(0) if m else '(no verdict) rc %s' % rc), flush=True)


tmp = tempfile.mkdtemp(prefix='kcoff_')
for tag in A.tags.split(','):
    rcp = json.load(open(os.path.join(HERE, 'build_receipt_%s.json' % tag), encoding='utf-8'))
    args = ['--kc-dir', rcp['kc_dataset']['local_dir'], '--base', rcp['base'], '--tag', tag, '--out', tmp]
    pol = rcp['policy']
    args += ['--g-kc-h', repr(pol['g_kc_h']), '--budget-h', repr(pol['budget_h']), '--deadline-h', repr(pol['deadline_h']),
             '--stop-margin-s', repr(pol['stop_margin_s']), '--min-run-s', repr(pol['min_run_s'])]
    r = subprocess.run(PY + [os.path.join(HERE, 'build_v224s6fkc.py')] + args, capture_output=True, text=True, timeout=600)
    re_ = json.load(open(os.path.join(tmp, 'build_receipt_%s.json' % tag), encoding='utf-8')) if r.returncode == 0 else {}
    same = r.returncode == 0 and re_.get('notebook_sha256') == rcp['notebook_sha256'] and \
        open(os.path.join(tmp, 'kernel_' + tag, 'kernel-metadata.json'), 'rb').read() == \
        open(os.path.join(HERE, 'kernel_' + tag, 'kernel-metadata.json'), 'rb').read()
    R.append(same)
    print('  %s REBUILD %s byte-identical (notebook %s, metadata)' % ('OK  ' if same else 'FAIL', tag, str(re_.get('notebook_sha256'))[:16]),
          flush=True)
    r = subprocess.run(PY + [os.path.join(HERE, 'celldiff_v224s6fkc_proof.py'), os.path.join(HERE, 'kernel_' + tag)], capture_output=True,
                       text=True, timeout=300)
    verdict('KC_CELLDIFF', r.stdout, r.returncode)
shutil.rmtree(tmp, ignore_errors=True)
for script, name in (('tests/neg_celldiff.py', 'NEG_CELLDIFF'), ('tests/test_blend_cell.py', 'TEST_BLEND'), ('tests/mock_gate.py', 'MOCK_GATE'),
                     ('tests/base_view_check.py', 'BASE_VIEW')):
    r = subprocess.run(PY + [os.path.join(HERE, script)], capture_output=True, text=True, timeout=3600)
    open(os.path.join(HERE, script[:-3] + '.txt'), 'w', encoding='utf-8').write(r.stdout + r.stderr[-3000:])
    verdict(name, r.stdout, r.returncode)
r = subprocess.run(PY + [os.path.join(HERE, 'sim_s6fkc_runtime.py'), '--reps', '3000'], capture_output=True, text=True, timeout=1800)
same = r.returncode == 0 and r.stdout == open(os.path.join(HERE, 'sim_s6fkc_runtime.txt'), encoding='utf-8').read()
R.append(same)
print('  %s SIM reproduces sim_s6fkc_runtime.txt (seed 0, 3000 reps) and chooses G_KC 6.90 / budget 1.5' % ('OK  ' if same else 'FAIL'), flush=True)
for f, name in (('box_results/e2e_s6f_guard.log', 'E2E_CELLS'), ('box_results/e2e_s6f_clean.log', 'E2E_CELLS'),
                ('box_results/e2e_s6f_s6fcanary.log', 'E2E_CELLS'), ('box_results/e2e_s6fj_clean.log', 'E2E_CELLS'),
                ('box_results/e2e_ctl_guard.log', 'E2E_CELLS'), ('box_results/test_kill_linux.txt', 'KC_KILL_LINUX'),
                ('box_results/test_driver_stop.txt', 'KC_DRIVER_STOP')):
    p = os.path.join(HERE, f)
    verdict(name, open(p, encoding='utf-8', errors='replace').read() if os.path.isfile(p) else '', 0 if os.path.isfile(p) else 1)
p = os.path.join(HERE, 'box_results', 'e2e_s6f_parity.log')
t = open(p, encoding='utf-8', errors='replace').read() if os.path.isfile(p) else ''
ok = bool(re.search(r'^KC_BOX_PARITY PASS ', t, re.M))
R.append(ok)
print('  %s KC_BOX_PARITY (box CPU, e2e clean output) %s' % ('OK  ' if ok else 'FAIL', (re.findall(r'^KC_BOX_PARITY .*$', t, re.M) or ['(none)'])[0]))
print('OFFLINE_TESTS %s %d/%d' % ('PASS' if all(R) else 'FAIL', sum(R), len(R)), flush=True)
sys.exit(0 if all(R) else 1)
