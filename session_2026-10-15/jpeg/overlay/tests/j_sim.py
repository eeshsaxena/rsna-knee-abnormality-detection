#!/usr/bin/env python
# j_sim.py (2026-10-15, Claude subagent): process simulator for j_local_harness.py (laptop CPU, Python 3.12, Windows).
#   --nb SPEC   = the notebook process: runs the DEC cell text (as exec'd source, like a notebook cell), optionally the S6F SETUP
#                 cell, then launches every stage chain exactly as the notebook does (env = the notebook's own env expression,
#                 evaluated in this process) and decodes in-process too. Prints one line: J_SIM_NB {json}.
#   --hop SPEC  = one stage process: evaluates the NEXT hop's env expression in ITS OWN environment (as the real stage process
#                 does), launches it (subprocess) or a spawn child (multiprocessing, like a fork / spawn pool or DataLoader worker),
#                 or, at the end of the chain, decodes the fixture files with pydicom.dcmread(path).pixel_array (the primitive all
#                 four stage readers call). Prints one line: J_SIM_HOP {json}.
# Review fixes (J1 / J2):
#   - fake /proc: when the spec names one, every simulated process (notebook, hop, spawn child) writes <fake_proc>/<pid>/{environ,
#     cmdline,stat} (its real os.environ, sys.orig_argv and parent pid; stat last) so the DEC cell's envwatch thread, pointed at
#     the fake dir, records them as it would from Linux /proc (the entries stay after exit: the polling miss-rate is not tested);
#   - a hop may carry 'py_flags' (e.g. ['-E']): it is then started as [python, *py_flags, -u, j_sim.py, ...], the J3 launch forms;
#   - a chain may end in 'pool_files': a Raptor-like spawn Pool(2).imap_unordered(chunksize=2) over those files, each decode
#     wrapped in try / except like raptor_run.py _pre_one; a 25 s watchdog reports POOL_HANG (a worker that died took its chunk).
import os, sys, io, json, time, hashlib, tempfile, threading, contextlib, subprocess, multiprocessing

HERE = os.path.abspath(__file__)
SAFE = {'os': os, 'json': json, 'sys': sys}                      # + builtins (eval adds them)


def _norm(p):
    return os.path.normcase(os.path.normpath(p)) if p else p


def fake_proc_entry(fake_proc):
    if not fake_proc:
        return
    d = os.path.join(fake_proc, str(os.getpid()))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'environ'), 'wb') as f:
        f.write(b''.join(('%s=%s' % kv).encode('utf-8', 'replace') + b'\0' for kv in os.environ.items()))
    with open(os.path.join(d, 'cmdline'), 'wb') as f:
        f.write(b''.join(a.encode('utf-8', 'replace') + b'\0' for a in getattr(sys, 'orig_argv', sys.argv)))
    with open(os.path.join(d, 'stat'), 'w') as f:                # written last: the watcher reads stat first
        f.write('%d (python) S %d %s %d 0 0\n' % (os.getpid(), os.getppid(), ' '.join(['0'] * 17), int(time.time() * 100)))


def leaf(fixtures_dir, target, fake_proc=None):
    fake_proc_entry(fake_proc)
    r = {'pid': os.getpid(), 'PYTHONPATH': os.environ.get('PYTHONPATH'),
         'target_on_sys_path': any(_norm(p) == _norm(target) for p in sys.path)}
    try:
        import numpy as np, pydicom
        exp = json.load(open(os.path.join(fixtures_dir, 'expected.json')))
        dec = {}
        for name, (dtype, shape, digest) in sorted(exp.items()):
            try:
                a = pydicom.dcmread(os.path.join(fixtures_dir, name + '.dcm')).pixel_array
                dec[name] = bool(str(a.dtype) == dtype and list(a.shape) == shape and
                                 hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() == digest)
            except Exception as e:
                dec[name] = 'ERR %s: %s' % (type(e).__name__, str(e)[:400])
        r['decode'] = dec
        m = sys.modules.get('libjpeg')
        r['libjpeg_dir'] = os.path.dirname(os.path.dirname(os.path.abspath(m.__file__))) if m else None
        r['pydicom'] = pydicom.__version__
        r['pydicom_dir'] = os.path.dirname(os.path.dirname(os.path.abspath(pydicom.__file__)))
    except Exception as e:
        r['error'] = '%s: %s' % (type(e).__name__, str(e)[:200])
    return r


def _spawn_target(q, fixtures_dir, target, fake_proc=None):
    q.put(leaf(fixtures_dir, target, fake_proc))


def _pool_init(fake_proc):
    fake_proc_entry(fake_proc)


def _pool_one(path):                                             # = raptor_run.py _pre_one: any exception -> that study zero-filled
    try:
        import pydicom
        pydicom.dcmread(path).pixel_array
        return os.path.basename(os.path.dirname(path)) + '/' + os.path.basename(path), 'ok'
    except Exception as e:
        return os.path.basename(os.path.dirname(path)) + '/' + os.path.basename(path), 'ERR %s' % type(e).__name__


def pool_leaf(files, target, fake_proc):
    r = {'pid': os.getpid(), 'PYTHONPATH': os.environ.get('PYTHONPATH'),
         'target_on_sys_path': any(_norm(p) == _norm(target) for p in sys.path)}
    got = {}

    def watchdog(t0):
        time.sleep(25)
        r.update(pool='HANG', got=dict(got), seconds=round(time.time() - t0, 1))
        print('J_SIM_HOP ' + json.dumps({'view': r, 'next': {'pool_result': r}}), flush=True)
        os._exit(0)
    threading.Thread(target=watchdog, args=(time.time(),), daemon=True).start()
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(2, initializer=_pool_init, initargs=(fake_proc,)) as pool:
        for name, res in pool.imap_unordered(_pool_one, files, chunksize=2):
            got[name] = res
    r.update(pool='DONE', got=got)
    return r


def run_hops(hops, fixtures_dir, target, env_override=None, fake_proc=None, pool_files=None):
    """launch hops[0] from THIS process; the child handles hops[1:]."""
    if not hops:
        if pool_files:
            return {'pool_result': pool_leaf(pool_files, target, fake_proc)}
        return leaf(fixtures_dir, target, fake_proc)
    h, rest = hops[0], hops[1:]
    if h['via'] == 'spawn':
        ctx = multiprocessing.get_context('spawn')
        q = ctx.Queue()
        p = ctx.Process(target=_spawn_target, args=(q, fixtures_dir, target, fake_proc))
        p.start()
        try:
            res = q.get(timeout=120)
        except Exception as e:
            res = {'error': 'spawn child: %s' % type(e).__name__}
        p.join(30)
        return {'via': 'spawn', 'result': res}
    if env_override is not None:
        env = env_override
    elif h.get('env_expr') is None:
        env = None
    else:
        env = eval(h['env_expr'], dict(SAFE), dict(h.get('names') or {}))
    fd, sp = tempfile.mkstemp(suffix='.json', prefix='jsim_hop_')
    with os.fdopen(fd, 'w') as f:
        json.dump({'hops': rest, 'fixtures_dir': fixtures_dir, 'target': target, 'name': h.get('name'), 'fake_proc': fake_proc,
                   'pool_files': pool_files}, f)
    r = subprocess.run([sys.executable] + list(h.get('py_flags') or []) + ['-u', HERE, '--hop', sp], env=env, capture_output=True,
                       text=True, timeout=300)
    os.remove(sp)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith('J_SIM_HOP ')), None)
    if r.returncode != 0 or line is None:
        return {'via': 'subprocess', 'error': 'rc %s %s' % (r.returncode, (r.stderr or '')[-400:])}
    return {'via': 'subprocess', 'env_expr': h.get('env_expr'), 'result': json.loads(line[len('J_SIM_HOP '):])}


def hop_main(sp):
    spec = json.load(open(sp))
    fake_proc_entry(spec.get('fake_proc'))
    view = {'pid': os.getpid(), 'name': spec.get('name'), 'PYTHONPATH': os.environ.get('PYTHONPATH'),
            'target_on_sys_path': any(_norm(p) == _norm(spec['target']) for p in sys.path)}
    nxt = run_hops(spec['hops'], spec['fixtures_dir'], spec['target'], fake_proc=spec.get('fake_proc'),
                   pool_files=spec.get('pool_files'))
    print('J_SIM_HOP ' + json.dumps({'view': view, 'next': nxt}), flush=True)


def nb_main(sp):
    spec = json.load(open(sp))
    fake_proc_entry(spec.get('fake_proc'))
    out = {'pythonpath_before': os.environ.get('PYTHONPATH'), 'had_pythonpath': 'PYTHONPATH' in os.environ,
           'sys_path_before': list(sys.path), 'pid': os.getpid()}
    if spec.get('pre_import_pydicom'):
        import pydicom  # noqa: F401  (a notebook that imported pydicom before the DEC cell)
    g = {'__name__': '__main__'}
    buf = io.StringIO()
    exc = None
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(spec['dec_text'], '<dec cell>', 'exec'), g)
    except BaseException as e:
        exc = '%s: %s' % (type(e).__name__, e)
    out.update(dec_seconds=round(time.time() - t0, 2), dec_stdout=buf.getvalue(), dec_exception=exc,
               namespace=sorted(k for k in g if not k.startswith('__')),
               pythonpath_after=os.environ.get('PYTHONPATH'), has_pythonpath_after='PYTHONPATH' in os.environ,
               sys_path_after=list(sys.path), threads=sorted(t.name for t in threading.enumerate()))
    try:
        out['receipt'] = json.load(open(os.path.join(spec['work'], 'dec_receipt.json')))
    except Exception as e:
        out['receipt'] = {'missing': type(e).__name__}
    s6f_env = None
    if spec.get('s6f'):
        s = spec['s6f']
        g['NB_T0'] = time.time()
        os.makedirs(s['work'], exist_ok=True)
        with open(os.path.join(s['work'], 'submission.csv'), 'w') as f:
            f.write('StudyInstanceUID,ACL\n1.2.3,0.5\n')
        b2 = io.StringIO()
        e2 = None
        try:
            with contextlib.redirect_stdout(b2):
                exec(compile(s['setup_text'], '<s6f setup cell>', 'exec'), g)
        except BaseException as e:
            e2 = '%s: %s' % (type(e).__name__, e)
        st = g.get('_S6F_SETUP') or {}
        s6f_env = g.get('_S6F_ENV')
        out['s6f'] = {'stdout': b2.getvalue(), 'exception': e2, 'ok': st.get('ok'), 'dec_ok': st.get('dec_ok'),
                      'why': st.get('why'), 'dec_why': (st.get('dec') or {}).get('why'), 'pythonpath_s6f': st.get('pythonpath_s6f'),
                      'pythonpath_stock': st.get('pythonpath_stock'), 'env_none': s6f_env is None}
    chains = {}
    for ch in spec['chains']:
        try:
            eo = s6f_env if ch.get('env_from') == 's6f' else None
            if ch.get('env_from') == 's6f' and s6f_env is None:
                chains[ch['name']] = {'error': 'no S6F env (setup failed)'}
                continue
            chains[ch['name']] = run_hops(ch['hops'], spec['fixtures_dir'], spec['target'], env_override=eo,
                                          fake_proc=spec.get('fake_proc'), pool_files=ch.get('pool_files'))
        except Exception as e:
            chains[ch['name']] = {'error': '%s: %s' % (type(e).__name__, str(e)[:300])}
    out['chains'] = chains
    if spec.get('inproc'):
        out['inproc'] = leaf(spec['fixtures_dir'], spec['target'])
    if spec.get('fake_proc'):
        time.sleep(spec.get('watch_wait_s', 2.0))                 # let the watcher take its last sweep
        try:
            out['envwatch'] = json.load(open(os.path.join(spec['work'], 'dec_envwatch.json')))
        except Exception as e:
            out['envwatch'] = {'missing': type(e).__name__}
    print('J_SIM_NB ' + json.dumps(out), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == '--hop':
        hop_main(sys.argv[2])
    else:
        nb_main(sys.argv[2])
