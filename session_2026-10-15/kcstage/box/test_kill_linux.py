# test_kill_linux.py (10-15, Claude subagent, kcstage): the KC RUN cell, AS BUILT, on Linux (box CPU, no GPU), with a HUNG stub in place
# of /tmp/kc/kc_driver.py (the cell text unchanged except: '/tmp/kc' -> a unique test root, the policy line -> BUDGET 0.01 h (36 s),
# STOP_MARGIN 5 s, MIN_RUN -1e9, G 99). The stub (own session) starts 2 children, each in its OWN session (like kc_serve processes), each
# with a grandchild in the child's group whose cmdline does NOT name the root (like a fork-pool decode worker after exec of `sleep`), and a
# control process that names nothing and is in no KC group. Expect: status RAN, killed True at the hard deadline, >= 3 groups killed
# (stub + 2 children), every stub / child / grandchild dead, the control alive (then this test stops it), the notebook group untouched.
#   python test_kill_linux.py --nb KERNEL.ipynb
import os, re, sys, json, time, signal, argparse, subprocess, tempfile

ap = argparse.ArgumentParser()
ap.add_argument('--nb', required=True)
A = ap.parse_args()
nb = json.load(open(A.nb, encoding='utf-8'))
RUN = [''.join(c['source']) if isinstance(c['source'], list) else c['source'] for c in nb['cells']][-2]
assert 'KC 2/3 RUN' in RUN.splitlines()[0]
root = tempfile.mkdtemp(prefix='kckill_') + '/kc'
os.makedirs(root + '/work')
pids_file = root + '/pids.json'
stub = r'''
import os, sys, json, time, subprocess
kids, grand = [], []
for k in range(2):
    c = subprocess.Popen([sys.executable, '-c', "import subprocess, time, sys; g = subprocess.Popen(['sleep', '600']); "
                          "open(sys.argv[1], 'w').write(str(g.pid)); time.sleep(600)", '%s/gpid%d' % (os.path.dirname(__file__), k),
                          os.path.dirname(__file__) + '/kc_serve.py'], start_new_session=True)
    kids.append(c.pid)
time.sleep(2)
grand = [int(open('%s/gpid%d' % (os.path.dirname(__file__), k)).read()) for k in range(2)]
json.dump({'stub': os.getpid(), 'children': kids, 'grandchildren': grand}, open(os.path.dirname(__file__) + '/pids.json', 'w'))
time.sleep(600)
'''
open(root + '/kc_driver.py', 'w').write(stub)
ctl = subprocess.Popen(['sleep', '600'])                                     # unrelated control process
cell = RUN.replace('/tmp/kc', root)
cell, n = re.subn(r'G_KC_H, BUDGET_H, DEADLINE_H, STOP_MARGIN_S, MIN_RUN_S = [^\n]+',
                  'G_KC_H, BUDGET_H, DEADLINE_H, STOP_MARGIN_S, MIN_RUN_S = 99.0, 0.01, 8.2, 5.0, -1e9', cell)
assert n == 1
g = {'__name__': '__main__', 'NB_T0': time.time() - 1.0 * 3600, '_KC_SETUP': {'ok': True, 'members': ['m1.pt']}, '_KC_ENV': dict(os.environ)}
t0 = time.time()
exec(compile(cell, '<kc run cell>', 'exec'), g)
r = g['_KC_RUN']
time.sleep(2)
pids = json.load(open(pids_file))


def alive(p):
    try:
        os.kill(p, 0)
    except ProcessLookupError:
        return False
    try:
        st = open('/proc/%d/stat' % p).read()
        return st[st.rindex(')') + 2] != 'Z'
    except Exception:
        return False


every = [pids['stub']] + pids['children'] + pids['grandchildren']
left = [p for p in every if alive(p)]
R = []


def chk(name, ok, info=''):
    R.append(bool(ok))
    print('  %s %s %s' % ('OK  ' if ok else 'FAIL', name, info), flush=True)


chk('RUN cell: status RAN, killed True, rc set (SIGKILL), seconds ~ the 36 s budget', r.get('status') == 'RAN' and r.get('killed') is True
    and 30 <= r.get('seconds', 0) <= 60, {k: r.get(k) for k in ('status', 'killed', 'rc', 'seconds')})
chk('>= 3 process groups killed incl. the stub and its 2 child sessions', len(r.get('killed_groups') or []) >= 3 and
    all(p in r['killed_groups'] for p in [pids['stub']] + pids['children']), (r.get('killed_groups'), pids))
chk('no stub / child / grandchild survived (grandchildren: cmdline sleep 600, in the children groups)', not left, left)
chk('the control process (no KC cmdline, own group) and this process survived', alive(ctl.pid) and os.getpgrp() not in (r.get('killed_groups') or []))
ctl.kill()
print('KC_KILL_LINUX %s %d/%d (%.0f s)' % ('PASS' if all(R) else 'FAIL', sum(R), len(R), time.time() - t0), flush=True)
sys.exit(0 if all(R) else 1)
