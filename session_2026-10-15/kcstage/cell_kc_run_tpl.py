# [eeshsaxena __TAG__ KC 2/3 RUN; not in any public notebook] ONE subprocess (/tmp/kc/kc_driver.py, its own session / process
# group, the KC environment of the setup cell) runs the KC stage: the side kernel's study() on every test study (header-only),
# then one serve process per T4 (/tmp/kc/kc_serve.py = kc_worker.py's own __main__ with the side table as the side / geometry
# source; --device cuda:K --shard K/N --procs 2; members fp16 autocast, ONE pass, mean of member sigmoids; weak-zone /
# not-GEOM_OK studies = the mean of both canons). Time policy (fixed at build time from sim_s6fkc_runtime.py, A6 G3):
#   start guard G_KC: KC starts only if the kernel age (from NB_T0) is <= G_KC_H; else KC_SKIPPED (the blend fires the KC canary);
#   hard deadline = min(start + BUDGET_H, NB_T0 + DEADLINE_H): the driver is told to stop by (hard - STOP_MARGIN_S) (it kills its
#   serve processes then), and at the hard deadline this cell SIGKILLs every KC process group (the driver's, each serve process's,
#   their decode pool children: every /proc process whose cmdline names /tmp/kc/, never the notebook's own group).
# Never raises; submission.csv is not touched here.
def _kc_kill(driver, root):
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


def _kc_run(setup, env):
    import sys, json, time, subprocess
    G_KC_H, BUDGET_H, DEADLINE_H, STOP_MARGIN_S, MIN_RUN_S = __G_KC_H__, __BUDGET_H__, __DEADLINE_H__, __STOP_MARGIN_S__, __MIN_RUN_S__
    W, ROOT = '/kaggle/working', '/tmp/kc'
    rec = {'g_kc_h': G_KC_H, 'budget_h': BUDGET_H, 'deadline_h': DEADLINE_H, 'stop_margin_s': STOP_MARGIN_S, 'min_run_s': MIN_RUN_S,
           'status': None}
    t_start = time.time()
    nb_t0 = globals().get('NB_T0')
    if not isinstance(nb_t0, float):
        rec.update(status='NO_NB_T0')
        print('KC_SKIPPED no NB_T0 in the notebook namespace', flush=True)
        return rec
    age = (t_start - nb_t0) / 3600.0
    hard = min(t_start + BUDGET_H * 3600.0, nb_t0 + DEADLINE_H * 3600.0)
    stop_at = hard - STOP_MARGIN_S
    rec.update(start_age_h=round(age, 4), hard_deadline_age_h=round((hard - nb_t0) / 3600.0, 4),
               stop_at_age_h=round((stop_at - nb_t0) / 3600.0, 4))
    if not (isinstance(setup, dict) and setup.get('ok') and env is not None):
        rec.update(status='SETUP_FAILED')
        print('KC_SKIPPED setup failed: %s' % (setup or {}).get('why'), flush=True)
        return rec
    if age > G_KC_H:
        rec.update(status='SKIPPED_GUARD')
        print('KC_SKIPPED start guard: kernel age %.2f h > G_KC %.2f h' % (age, G_KC_H), flush=True)
        return rec
    if stop_at - t_start < MIN_RUN_S:
        rec.update(status='SKIPPED_NO_TIME')
        print('KC_SKIPPED only %.0f s before the stop time' % (stop_at - t_start), flush=True)
        return rec
    print('KC_START ' + json.dumps({k: rec[k] for k in ('start_age_h', 'g_kc_h', 'budget_h', 'deadline_h', 'hard_deadline_age_h',
                                                         'stop_at_age_h')}), flush=True)
    rc, killed, groups = None, False, []
    try:
        p = subprocess.Popen([sys.executable, '-u', ROOT + '/kc_driver.py', '--stop-at', '%.3f' % stop_at, '--nb-t0', '%.3f' % nb_t0,
                              '--members', *setup['members']], env=env, cwd=ROOT + '/work', stdin=subprocess.DEVNULL,
                             start_new_session=True)
        rec['driver_pid'] = p.pid
        while True:
            rc = p.poll()
            if rc is not None:
                break
            if time.time() >= hard:
                killed = True
                groups = _kc_kill(p, ROOT)
                try:
                    rc = p.wait(timeout=120)
                except Exception:
                    rc = None
                print('KC_KILLED at the hard deadline (kernel age %.2f h): process groups %s' % ((time.time() - nb_t0) / 3600.0, groups),
                      flush=True)
                break
            time.sleep(5)
        rec['after_exit_sweep'] = _kc_kill(p, ROOT)                  # an orphan of a finished driver must not hold a GPU
        rec['status'] = 'RAN'
    except Exception as e:
        rec.update(status='LAUNCH_ERROR', why='%s: %s' % (type(e).__name__, str(e)[:400]))
    rec.update(rc=rc, killed=killed, killed_groups=groups, seconds=round(time.time() - t_start, 1),
               end_age_h=round((time.time() - nb_t0) / 3600.0, 4))
    try:
        d = json.load(open(W + '/kc/kc_driver_receipt.json'))
        rec['driver'] = {k: d.get(k) for k in ('ok', 'error', 'stages', 'env', 'inputs', 'side', 'workers', 'B', 'dump', 'start_age_h',
                                               'end_age_h', 'total_seconds', 'serve_timed_out', 'killed_worker_groups')}
    except Exception as e:
        rec['driver'] = {'ok': False, 'error': 'no driver receipt: %s' % type(e).__name__}
    b = rec['driver'].get('B') or {}
    print('KC_STAGE_DONE ' + json.dumps({'status': rec['status'], 'rc': rc, 'killed': killed, 'seconds': rec['seconds'],
                                         'end_age_h': rec['end_age_h'], 'driver_ok': rec['driver'].get('ok'),
                                         'coverage': b.get('coverage'), 'scored': b.get('scored'), 'n': b.get('n'),
                                         'microbench': b.get('microbench')}), flush=True)
    return rec


try:
    _KC_RUN = _kc_run(globals().get('_KC_SETUP'), globals().get('_KC_ENV'))
except Exception as _e:
    _KC_RUN = {'status': 'RUN_CELL_ERROR', 'why': type(_e).__name__}
    print('KC_RUN_CELL_ERROR ' + str(_KC_RUN), flush=True)
