# s6f2_harness.py (10-15, Claude subagent): shared helpers for the s6f2 local tests. Offline, laptop CPU only, no GPU, no Kaggle.
# Reuses s6f/tests/s6f_harness.py (imported read-only, no bytecode written) for the temp tree, the path redirection and the FAKE
# drivers (the port runs the REAL built driver, sha a2acdd1d, == the bytes embedded in both s6f2 notebooks, on nartaa's REAL commit
# part outputs). What is new here: the appended cells are read from the s6f2 notebooks (cells N..N+2 after the base's N cells), and
# the notebook namespace can carry the base canary dicts of the other bases (_gcr = RM_GUARD's receipt dict on V224-FLIP-RM, _v224f =
# the HF cell's receipt dict on flips20) exactly as the base cells leave them.
import os, sys, io, json, time, shutil, hashlib, contextlib
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
S6F2 = os.path.dirname(HERE)
SESS = os.path.dirname(S6F2)
ROOT = os.path.dirname(SESS)
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(SESS, 's6f', 'tests'))
import s6f_harness as H0                                              # noqa: E402  (read only)

LAB = H0.LAB
KIND = {'flrm': dict(tag='v224flrms6f', start=17, G=6.0, base_out=os.path.join(SESS, 'v224rm', 'out_v224flrm_v1'), last='V224RM_DONE'),
        'flips20': dict(tag='v224flips6fs20', start=24, G=6.25, base_out=os.path.join(SESS, 'v224f', 'out_v224flips20_v1'), last='V224F_DONE'),
        'flip': dict(tag='v224flips6f', start=15, G=5.75, base_out=H0.FLIP_OUT, last='V224_DONE')}


def nb_path(kind):
    t = KIND[kind]['tag']
    if kind == 'flip':
        return H0.NB_PATH                                             # the pushed s6f kernel (read only)
    return os.path.join(S6F2, 'kernel_' + t, 'rsna_sub_%s.ipynb' % t)


def cells(kind, path=None):
    nb = json.load(open(path or nb_path(kind), encoding='utf-8'))
    k = KIND[kind]['start']
    assert len(nb['cells']) == k + 3
    return [H0.src(c) for c in nb['cells'][k:k + 3]]


def driver_same():
    """the driver built here == the s6f one the FAKE port runs (same sha)."""
    a = H0.sha_file(H0.DRIVER_BUILT)
    return all(H0.sha_file(os.path.join(S6F2, 's6f_driver_built_%s.py' % KIND[k]['tag'])) == a for k in ('flrm', 'flips20'))


def base_df(kind):
    return pd.read_csv(os.path.join(KIND[kind]['base_out'], 'submission.csv'), dtype={'StudyInstanceUID': str})


def base_file(kind):
    return os.path.join(KIND[kind]['base_out'], 'submission.csv')


def run_cells(tree, kind, base, age_h=4.6, fired=(), gcr_fired=None, v224f_fired=None, mode='port', overrides=None, setup_hook=None,
              base_file=None, cell_texts=None, extra_ns=None, **kw):
    """write the base submission, execute SETUP, swap in a fake driver, execute RUN and BLEND (as s6f_harness.run_cells) with the
    base's canary dicts in the namespace: _canary (always), _gcr (flrm: RM_GUARD), _v224f (flips20: HF). Returns (ns, text)."""
    setup_t, run_t, blend_t = cell_texts or cells(kind)
    for a, b in (overrides or {}).items():
        assert run_t.count(a) == 1 or blend_t.count(a) == 1, a
        run_t, blend_t = run_t.replace(a, b), blend_t.replace(a, b)
    if base_file:
        shutil.copyfile(base_file, os.path.join(tree.work, 'submission.csv'))
    else:
        base.to_csv(os.path.join(tree.work, 'submission.csv'), index=False, lineterminator='\n')
    ns = {'__name__': '__main__', 'NB_T0': time.time() - age_h * 3600.0, 'sub': base.copy(),
          '_canary': {'r224_used': 'r224->Fracture' not in fired, 'fired': list(fired)}}
    if gcr_fired is not None:
        ns['_gcr'] = {'fired': list(gcr_fired), 'rm_guard': {'mirror_ok': 'r224 mirror->Fracture' not in gcr_fired}}
    if v224f_fired is not None:
        ns['_v224f'] = {'use_f': True, 'w_f': 0.2, 'mode': 'HF' if 'fleet->Synovitis' not in v224f_fired else 'H_only', 'fired': list(v224f_fired)}
    ns.update(extra_ns or {})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(tree.redirect(setup_t), '<s6f setup>', 'exec'), ns)
    if setup_hook:
        setup_hook(tree, ns)
    if mode is not None and os.path.isfile(os.path.join(tree.root, 's6f_driver.py')):
        H0.install_fake(tree, mode, **kw)
    with contextlib.redirect_stdout(buf):
        exec(compile(tree.redirect(run_t), '<s6f run>', 'exec'), ns)
        exec(compile(tree.redirect(blend_t), '<s6f blend>', 'exec'), ns)
    return ns, buf.getvalue()


Tree = H0.Tree
recompute = H0.recompute
avg_rank_pct = H0.avg_rank_pct
ref_bprobs = H0.ref_bprobs
sha_file = H0.sha_file
REF = H0.REF
