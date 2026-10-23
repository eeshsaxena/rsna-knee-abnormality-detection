# cell_run_killctl_tpl.py (10-15 rfix, Claude subagent; S6F-R1): the KILL CONTROL pieces build_v224s6f.py --kill-control splices into
# the RUN cell (cell_run_tpl.py). Only control builds get them (never the submit tag; the default build stays byte-identical).
# Read by the builder with ast.literal_eval (never executed by the builder). Each constant is inserted at one anchor:
#   KILLCTL_HEADER  prepended to the cell
#   KILLCTL_DEFS    inserted before "def _s6f_run(": the stub program + _s6f_alive (post-kill liveness probe)
#   KILLCTL_LAUNCH  replaces the driver launch: the RUN cell writes /tmp/s6f/kill_stub.py and starts IT (same Popen, own session)
#   KILLCTL_CHECK   inserted after the after-exit sweep: 2 s later every stub process must be gone (zombies count as gone)
# The stub: its own session (start_new_session, as the driver), 2 children each in their OWN session (as compact_infer workers),
# each child with a grandchild in the child's group whose cmdline does NOT name /tmp/s6f (as a spawned DataLoader worker).
# So the hard-deadline kill must find 3 groups (driver pgid + /proc cmdline scan) and kill the grandchildren by group only.
KILLCTL_HEADER = (
    "# [KILL CONTROL BUILD (build_v224s6f.py --kill-control): NEVER SUBMIT. The RUN cell starts a stub process tree under /tmp/s6f\n"
    "# instead of the S6F driver, and checks after the hard-deadline kill that no stub process survived (S6F_KILL_CHECK). The\n"
    "# blend then fires the S6F canary. Gate: v224s6f_check.py --expect-canary --build build_receipt_<tag>.json (control kill).]\n")

KILLCTL_DEFS = r'''_S6F_KILL_STUB = r"""
import os, sys, json, time, subprocess
SLEEP_S = @SLEEP@
work = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'work')
if len(sys.argv) > 2 and sys.argv[1] == '--child':
    g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(%d)' % SLEEP_S])    # child's group; no S6F path in argv
    json.dump({'child': os.getpid(), 'grandchild': g.pid, 'pgid': os.getpgrp() if os.name == 'posix' else None},
              open(sys.argv[2] + '.tmp', 'w'))
    os.replace(sys.argv[2] + '.tmp', sys.argv[2])
    time.sleep(SLEEP_S)
    sys.exit(0)
kids = [subprocess.Popen([sys.executable, '-u', os.path.abspath(__file__), '--child', os.path.join(work, 'kill_child%d.json' % i)],
                         start_new_session=True) for i in range(2)]
json.dump({'stub': os.getpid(), 'pgid': os.getpgrp() if os.name == 'posix' else None, 'children': [k.pid for k in kids]},
          open(os.path.join(work, 'kill_stub_pids.json.tmp'), 'w'))
os.replace(os.path.join(work, 'kill_stub_pids.json.tmp'), os.path.join(work, 'kill_stub_pids.json'))
print('S6F_KILL_STUB started: stub %d, child sessions %s' % (os.getpid(), [k.pid for k in kids]), flush=True)
time.sleep(SLEEP_S)
"""
KILL_STUB_SLEEP_S = 900


def _s6f_alive(root, groups):
    """[kill control] the stub tree's pids (stub file + each child's file) and every live process that is one of them, sits in a
    killed group, or names the S6F root. Zombies / dead count as gone. alive None on a non-POSIX host (local tests only)."""
    import os, json, glob
    pids = {}
    try:
        pids = dict(json.load(open(root + '/work/kill_stub_pids.json')))
    except Exception as e:
        pids = {'error': type(e).__name__}
    kids = []
    for p in sorted(glob.glob(root + '/work/kill_child[0-9].json')):
        try:
            kids.append(dict(json.load(open(p))))
        except Exception as e:
            kids.append({'error': type(e).__name__})
    pids['child_files'] = kids
    pids['grandchildren'] = [k.get('grandchild') for k in kids]
    if os.name != 'posix':
        return {'stub_pids': pids, 'alive': None, 'groups': sorted(groups)}
    want = set()
    for x in [pids.get('stub')] + list(pids.get('children') or []) + pids['grandchildren']:
        try:
            want.add(int(x))
        except Exception:
            pass
    me, alive = os.getpid(), []
    for d in os.listdir('/proc'):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            cmd = open('/proc/%s/cmdline' % d, 'rb').read().replace(b'\0', b' ')
            st = open('/proc/%s/stat' % d).read()
            f = st[st.rindex(')') + 2:].split()
            state, pg = f[0], int(f[2])
        except Exception:
            continue
        if state in ('Z', 'X'):
            continue
        if int(d) in want or pg in groups or (root + '/').encode() in cmd:
            alive.append([int(d), state, pg, cmd[:160].decode('utf-8', 'replace')])
    return {'stub_pids': pids, 'alive': alive, 'groups': sorted(groups)}


'''

KILLCTL_LAUNCH = (
    "        open(ROOT + '/kill_stub.py', 'w').write(_S6F_KILL_STUB.replace('@SLEEP@', str(int(KILL_STUB_SLEEP_S))))   # [kill control]\n"
    "        p = subprocess.Popen([sys.executable, '-u', ROOT + '/kill_stub.py', ")

KILLCTL_CHECK = (
    "        time.sleep(2)                                                # [kill control] SIGKILL delivered; then nothing may live\n"
    "        rec['kill_check'] = _s6f_alive(ROOT, sorted(set(groups) | set(rec['after_exit_sweep'])))\n"
    "        print('S6F_KILL_CHECK ' + json.dumps(rec['kill_check']), flush=True)\n")
