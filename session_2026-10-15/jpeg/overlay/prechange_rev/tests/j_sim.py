#!/usr/bin/env python
# j_sim.py (2026-10-15, Claude subagent): process simulator for j_local_harness.py (laptop CPU, Python 3.12, Windows).
#   --nb SPEC   = the notebook process: runs the DEC cell text (as exec'd source, like a notebook cell), optionally the S6F SETUP
#                 cell, then launches every stage chain exactly as the notebook does (env = the notebook's own env expression,
#                 evaluated in this process) and decodes in-process too. Prints one line: J_SIM_NB {json}.
#   --hop SPEC  = one stage process: evaluates the NEXT hop's env expression in ITS OWN environment (as the real stage process
#                 does), launches it (subprocess) or a spawn child (multiprocessing, like a fork / spawn pool or DataLoader worker),
#                 or, at the end of the chain, decodes the fixture files with pydicom.dcmread(path).pixel_array (the primitive all
#                 four stage readers call). Prints one line: J_SIM_HOP {json}.
import os, sys, io, json, time, hashlib, tempfile, contextlib, subprocess, multiprocessing

HERE = os.path.abspath(__file__)
SAFE = {'os': os, 'json': json, 'sys': sys}                      # + builtins (eval adds them)


def _norm(p):
    return os.path.normcase(os.path.normpath(p)) if p else p


def leaf(fixtures_dir, target):
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


def _spawn_target(q, fixtures_dir, target):
    q.put(leaf(fixtures_dir, target))


def run_hops(hops, fixtures_dir, target, env_override=None):
    """launch hops[0] from THIS process; the child handles hops[1:]."""
    if not hops:
        return leaf(fixtures_dir, target)
    h, rest = hops[0], hops[1:]
    if h['via'] == 'spawn':
        ctx = multiprocessing.get_context('spawn')
        q = ctx.Queue()
        p = ctx.Process(target=_spawn_target, args=(q, fixtures_dir, target))
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
        json.dump({'hops': rest, 'fixtures_dir': fixtures_dir, 'target': target, 'name': h.get('name')}, f)
    r = subprocess.run([sys.executable, '-u', HERE, '--hop', sp], env=env, capture_output=True, text=True, timeout=300)
    os.remove(sp)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith('J_SIM_HOP ')), None)
    if r.returncode != 0 or line is None:
        return {'via': 'subprocess', 'error': 'rc %s %s' % (r.returncode, (r.stderr or '')[-400:])}
    return {'via': 'subprocess', 'env_expr': h.get('env_expr'), 'result': json.loads(line[len('J_SIM_HOP '):])}


def hop_main(sp):
    spec = json.load(open(sp))
    view = {'pid': os.getpid(), 'name': spec.get('name'), 'PYTHONPATH': os.environ.get('PYTHONPATH'),
            'target_on_sys_path': any(_norm(p) == _norm(spec['target']) for p in sys.path)}
    nxt = run_hops(spec['hops'], spec['fixtures_dir'], spec['target'])
    print('J_SIM_HOP ' + json.dumps({'view': view, 'next': nxt}), flush=True)


def nb_main(sp):
    spec = json.load(open(sp))
    out = {'pythonpath_before': os.environ.get('PYTHONPATH'), 'had_pythonpath': 'PYTHONPATH' in os.environ,
           'sys_path_before': list(sys.path)}
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
               sys_path_after=list(sys.path))
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
            chains[ch['name']] = run_hops(ch['hops'], spec['fixtures_dir'], spec['target'], env_override=eo)
        except Exception as e:
            chains[ch['name']] = {'error': '%s: %s' % (type(e).__name__, str(e)[:300])}
    out['chains'] = chains
    if spec.get('inproc'):
        out['inproc'] = leaf(spec['fixtures_dir'], spec['target'])
    print('J_SIM_NB ' + json.dumps(out), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == '--hop':
        hop_main(sys.argv[2])
    else:
        nb_main(sys.argv[2])
