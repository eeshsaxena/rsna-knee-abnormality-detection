# [eeshsaxena V224-FLIP-S6F 2/3 RUN; not in any public notebook] ONE subprocess (/tmp/s6f/s6f_driver.py, its own session /
# process group, the S6F environment of the setup cell) runs the S6fall B path: nartaa's prepare_inputs + probe_b_stack + run_b,
# one compact_infer worker per T4. Time policy (fixed at build time from sim_s6f_runtime.py, PREREG_10-15 Amendment 3 R4):
#   start guard G: S6F starts only if the kernel age (from NB_T0) is <= G_START_H; else S6F_SKIPPED (the blend fires the canary);
#   hard deadline = min(start + BUDGET_H, NB_T0 + DEADLINE_H): the driver is told to stop by (hard - STOP_MARGIN_S) (compact_infer
#   takes no new study 20 min before that), and at the hard deadline this cell SIGKILLs every S6F process group (the driver's,
#   each worker's, their DataLoader children) found by pgid / cmdline. The 9 h limit keeps >= 0.75 h after the hard deadline.
# Never raises; the V224-FLIP submission.csv is not touched here.
def _s6f_kill(driver, root):
    import os, signal
    if os.name != 'posix':                                           # local tests only (no process groups): kill the driver
        try:
            driver.kill()
        except Exception:
            pass
        return []
    me, groups = os.getpgrp(), set()
    try:
        groups.add(os.getpgid(driver.pid))
    except Exception:
        pass
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        try:
            cmd = open('/proc/%s/cmdline' % d, 'rb').read().replace(b'\0', b' ')
            st = open('/proc/%s/stat' % d).read()
            pg = int(st[st.rindex(')') + 2:].split()[2])
        except Exception:
            continue
        if (root + '/').encode() in cmd:
            groups.add(pg)
    groups.discard(me)
    for g in sorted(groups):
        try:
            os.killpg(g, signal.SIGKILL)
        except Exception:
            pass
    return sorted(groups)


def _s6f_run(setup, env):
    import sys, json, time, subprocess
    G_START_H, BUDGET_H, DEADLINE_H, STOP_MARGIN_S, MIN_RUN_S = __G_START_H__, __BUDGET_H__, __DEADLINE_H__, __STOP_MARGIN_S__, __MIN_RUN_S__
    W, ROOT = '/kaggle/working', '/tmp/s6f'
    rec = {'g_start_h': G_START_H, 'budget_h': BUDGET_H, 'deadline_h': DEADLINE_H, 'stop_margin_s': STOP_MARGIN_S, 'status': None}
    t_start = time.time()
    nb_t0 = globals().get('NB_T0')
    if not isinstance(nb_t0, float):
        rec.update(status='NO_NB_T0')
        print('S6F_SKIPPED no NB_T0 in the notebook namespace', flush=True)
        return rec
    age = (t_start - nb_t0) / 3600.0
    hard = min(t_start + BUDGET_H * 3600.0, nb_t0 + DEADLINE_H * 3600.0)
    stop_at = hard - STOP_MARGIN_S
    rec.update(start_age_h=round(age, 4), hard_deadline_age_h=round((hard - nb_t0) / 3600.0, 4),
               stop_at_age_h=round((stop_at - nb_t0) / 3600.0, 4))
    if not (isinstance(setup, dict) and setup.get('ok') and env is not None):
        rec.update(status='SETUP_FAILED')
        print('S6F_SKIPPED setup failed: %s' % (setup or {}).get('why'), flush=True)
        return rec
    if age > G_START_H:
        rec.update(status='SKIPPED_GUARD')
        print('S6F_SKIPPED start guard: kernel age %.2f h > G %.2f h' % (age, G_START_H), flush=True)
        return rec
    if stop_at - t_start < MIN_RUN_S:
        rec.update(status='SKIPPED_NO_TIME')
        print('S6F_SKIPPED only %.0f s before the stop time' % (stop_at - t_start), flush=True)
        return rec
    print('S6F_START ' + json.dumps({k: rec[k] for k in ('start_age_h', 'g_start_h', 'budget_h', 'deadline_h', 'hard_deadline_age_h',
                                                          'stop_at_age_h')}), flush=True)
    rc, killed, groups = None, False, []
    try:
        p = subprocess.Popen([sys.executable, '-u', ROOT + '/s6f_driver.py', '--stop-at', '%.3f' % stop_at, '--nb-t0', '%.3f' % nb_t0],
                             env=env, cwd=ROOT + '/work', stdin=subprocess.DEVNULL, start_new_session=True)
        rec['driver_pid'] = p.pid
        while True:
            rc = p.poll()
            if rc is not None:
                break
            if time.time() >= hard:
                killed = True
                groups = _s6f_kill(p, ROOT)
                try:
                    rc = p.wait(timeout=120)
                except Exception:
                    rc = None
                print('S6F_KILLED at the hard deadline (kernel age %.2f h): process groups %s' % ((time.time() - nb_t0) / 3600.0, groups),
                      flush=True)
                break
            time.sleep(5)
        rec['after_exit_sweep'] = _s6f_kill(p, ROOT)                 # an orphan of a finished driver must not hold a GPU
        rec['status'] = 'RAN'
    except Exception as e:
        rec.update(status='LAUNCH_ERROR', why='%s: %s' % (type(e).__name__, str(e)[:400]))
    rec.update(rc=rc, killed=killed, killed_groups=groups, seconds=round(time.time() - t_start, 1),
               end_age_h=round((time.time() - nb_t0) / 3600.0, 4))
    try:
        d = json.load(open(W + '/s6f/s6f_driver_receipt.json'))
        rec['driver'] = {'ok': d.get('ok'), 'error': d.get('error'), 'stages': d.get('stages'), 'env': d.get('env'),
                         'env_isolation': d.get('env_isolation'), 'B_probe': d.get('B_probe'), 'B': d.get('B'),
                         'inputs': d.get('inputs'), 'start_age_h': d.get('start_age_h'), 'end_age_h': d.get('end_age_h'),
                         'total_seconds': d.get('total_seconds'), 'code_manifest_pin': d.get('code_manifest_pin')}
    except Exception as e:
        rec['driver'] = {'ok': False, 'error': 'no driver receipt: %s' % type(e).__name__}
    b = rec['driver'].get('B') or {}
    print('S6F_STAGE_DONE ' + json.dumps({'status': rec['status'], 'rc': rc, 'killed': killed, 'seconds': rec['seconds'],
                                          'end_age_h': rec['end_age_h'], 'driver_ok': rec['driver'].get('ok'),
                                          'coverage': b.get('coverage'), 'scored': b.get('scored'), 'n': b.get('n'),
                                          'microbench': b.get('microbench')}), flush=True)
    return rec


try:
    _S6F_RUN = _s6f_run(globals().get('_S6F_SETUP'), globals().get('_S6F_ENV'))
except Exception as _e:
    _S6F_RUN = {'status': 'RUN_CELL_ERROR', 'why': type(_e).__name__}
    print('S6F_RUN_CELL_ERROR ' + str(_S6F_RUN), flush=True)
