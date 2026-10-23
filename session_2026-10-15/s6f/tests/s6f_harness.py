# s6f_harness.py (10-15, Claude subagent): shared helpers for the S6F local tests. Offline, laptop CPU only, no GPU, no Kaggle.
# Executes the APPENDED cells of the built notebook (our own code) with /kaggle/working, /kaggle/input, /tmp/s6f, /tmp/s6f_dec
# redirected to a temp tree, and replaces the driver the setup cell wrote with a FAKE driver per scenario:
#   port   the REAL built driver (s6f_driver_built_<tag>.py == the notebook's embedded bytes) with its GPU / dataset steps
#          monkeypatched: probe_env / probe_b_stack return nartaa's MEASURED commit values, prepare_inputs returns the 3 public
#          studies (+ series counts from _data/test_series.csv), run_group copies nartaa's real part outputs (ref_output/k1/B/partN)
#          into our part dirs; everything else (run_b_s6f, collect_part, B_probs.csv, receipts, main) runs as on Kaggle.
#   bprobs write a given B_probs.csv + a driver receipt with its sha (synthetic coverage / sanity scenarios)
#   sleep  never ends (hard-deadline kill)          fail   exit 3 with {"ok": false}
# Downloaded data (ref_output csv / json) is only parsed; every Python process that reads it runs with -I.
import os, sys, io, re, json, time, shutil, hashlib, base64, contextlib, subprocess
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
S6F = os.path.dirname(HERE)
SESS = os.path.dirname(S6F)
ROOT = os.path.dirname(SESS)
TAG = 'v224flips6f'
NB_PATH = os.path.join(S6F, 'kernel_' + TAG, 'rsna_sub_%s.ipynb' % TAG)
DRIVER_BUILT = os.path.join(S6F, 's6f_driver_built_%s.py' % TAG)
REF = os.path.join(S6F, 'ref_output')
FLIP_OUT = os.path.join(ROOT, 'session_2026-10-14', 'v224', 'out_v224flip_v1')
DATA = os.path.join(ROOT, '_data')
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
       'Contusion', 'Fracture']


def src(c):
    return ''.join(c['source']) if isinstance(c['source'], list) else c['source']


def cells(nb_path=NB_PATH):
    nb = json.load(open(nb_path, encoding='utf-8'))
    return [src(c) for c in nb['cells'][15:18]]


def sha_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


class Tree:
    def __init__(self, base, name):
        self.t = os.path.join(base, name)
        shutil.rmtree(self.t, ignore_errors=True)
        self.work = os.path.join(self.t, 'working')
        self.inp = os.path.join(self.t, 'input')
        self.root = os.path.join(self.t, 's6froot')
        self.dec = os.path.join(self.t, 's6fdec')
        for d in (self.work, self.inp):
            os.makedirs(d)

    def fw(self, p):
        return p.replace('\\', '/')

    def redirect(self, text):
        out = (text.replace('/tmp/s6f_dec', self.fw(self.dec)).replace('/tmp/s6f', self.fw(self.root))
               .replace('/kaggle/working', self.fw(self.work)).replace('/kaggle/input', self.fw(self.inp)))
        assert '/kaggle/' not in out.replace("'/kaggle/temp/probe'", '') and '/tmp/s6f' not in out.replace(self.fw(self.root), '') \
            .replace(self.fw(self.dec), ''), 'a path was not redirected'
        return out


FAKE_PORT = r'''
import sys, os, json, shutil, importlib.util
import numpy as np, pandas as pd
DRIVER, REF, DATA, OUT, TMP = %(driver)r, %(ref)r, %(data)r, %(out)r, %(tmp)r
MUT = %(mut)r
spec = importlib.util.spec_from_file_location('s6f_driver', DRIVER)
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)
kr = json.load(open(os.path.join(REF, 'k1', 'k1_receipt.json')))
D.probe_env = lambda: dict(kr['env'])
def env_iso():
    e = {'executable': '/usr/bin/python3', 'sys_path': ['', '/tmp/s6f_dec', '/usr/lib/python313.zip', '/usr/lib/python3.13',
         '/usr/lib/python3.13/lib-dynload', '/usr/local/lib/python3.13/dist-packages', '/usr/lib/python3/dist-packages'],
         'PYTHONPATH': '/tmp/s6f_dec', 'PYTHONNOUSERSITE': None, 'MPLBACKEND': 'Agg',
         'mods': {'torch': ['2.11.0+cu128', '/usr/local/lib/python3.13/dist-packages'], 'timm': ['1.0.25', '/usr/local/lib/python3.13/dist-packages'],
                  'pydicom': ['3.0.1', '/usr/local/lib/python3.13/dist-packages'], 'numpy': ['2.1.3', '/usr/local/lib/python3.13/dist-packages'],
                  'pandas': ['2.3.3', '/usr/local/lib/python3.13/dist-packages'], 'pylibjpeg': ['2.1.0', '/tmp/s6f_dec'],
                  'libjpeg': ['2.4.0', '/tmp/s6f_dec'], 'openjpeg': ['2.6.0', '/tmp/s6f_dec']},
         'r7_or_working_on_path': [], 'driver_sys_path': ['/tmp/s6f', '/usr/local/lib/python3.13/dist-packages']}
    if MUT == 'r7path':
        e['PYTHONPATH'] = '/tmp/probe/mainline_code:/tmp/s6f_dec'
    return e
D.env_isolation = env_iso
def prep():
    I = D.Inputs()
    I.comp, I.ds, I.helios = D.Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'), D.Path('/kaggle/input/rsna-knee-s6fall-weights'), D.Path('/kaggle/input/rsna-orthofoundation-fold0-epoch7-20261004')
    I.manifest_sha256 = kr['inputs']['manifest_sha256'] if MUT != 'manifest' else '0' * 64
    I.code_root = D.TMP / 'k1src'; I.src = I.code_root / 'heliosli'; I.code = I.code_root / 'cloud_code'
    I.models = D.TMP / 'k1models'; I.anatomy = I.helios / D.ANATOMY_FILE; I.launcher = D.TMP / 'k1launch' / 'k1_b_infer.py'
    I.test_ids = pd.read_csv(os.path.join(DATA, 'test.csv'), dtype={'StudyInstanceUID': str})['StudyInstanceUID'].tolist()
    I.sub_cols = ['StudyInstanceUID'] + D.LAB
    sc = pd.read_csv(os.path.join(DATA, 'test_series.csv'), dtype=str, usecols=['StudyInstanceUID'])
    I.cost = sc['StudyInstanceUID'].value_counts().to_dict()
    D._RECEIPT['inputs'] = {'manifest_sha256': I.manifest_sha256, 'model': D.B_MODEL, 'n_test': len(I.test_ids)}
    return I
D.prepare_inputs = prep
D.probe_b_stack = lambda I: dict(kr['B_probe'])
def fake_run_group(specs, timeout_s, what):
    runs = []
    for s in specs:
        pdir = os.path.dirname(str(s['log']))
        name = os.path.basename(pdir)
        src_dir = os.path.join(REF, 'k1', 'B', name)
        mine = json.load(open(os.path.join(pdir, 'ids.json')))
        theirs = json.load(open(os.path.join(src_dir, 'ids.json')))
        assert mine == theirs, ('split_parts differs from nartaa', mine, theirs)
        for f in ('receipt.json', 'logits.json', 'predictions.csv', 'log.txt'):
            if MUT == 'kill_part1' and name == 'part1' and f != 'predictions.csv':
                continue
            shutil.copyfile(os.path.join(src_dir, f), os.path.join(pdir, f))
        if MUT == 'timed_out_part0' and name == 'part0':
            r = json.load(open(os.path.join(pdir, 'receipt.json')))
            lg = json.load(open(os.path.join(pdir, 'logits.json')))
            keep = mine[0]
            r.update(timed_out=True, scored=1, fallback={mine[1]: 'not reached before the deadline'}, anchor_methods={'slab_prior': 1})
            json.dump(r, open(os.path.join(pdir, 'receipt.json'), 'w'))
            json.dump({keep: lg[keep]}, open(os.path.join(pdir, 'logits.json'), 'w'))
        if MUT == 'ci_sha' and name == 'part0':
            r = json.load(open(os.path.join(pdir, 'receipt.json')))
            r['sha256']['compact_infer.py'] = '8401812cd63737ec99839d317c4037c4efb445d1baec892fa7481bb30399cc2d'
            json.dump(r, open(os.path.join(pdir, 'receipt.json'), 'w'))
        runs.append((s['name'], 0, 1.0, ''))
    if MUT == 'kill_part1':
        raise D.K1Timeout('B: timeout (test)')
    return runs
D.run_group = fake_run_group
argv = sys.argv[1:] + ['--out', OUT, '--tmp', TMP]
sys.exit(D.main(argv))
'''

FAKE_BPROBS = r'''
import sys, os, json, hashlib, shutil
OUT, SRC = %(out)r, %(bprobs)r
os.makedirs(os.path.join(OUT, 'B'), exist_ok=True)
dst = os.path.join(OUT, 'B', 'B_probs.csv')
shutil.copyfile(SRC, dst)
h = hashlib.sha256(open(dst, 'rb').read()).hexdigest()
if %(tamper)r:
    open(dst, 'a').write('\n')
json.dump({'ok': True, 'B': {'coverage': None, 'probs_sha256': h}, 'stages': {}}, open(os.path.join(OUT, 's6f_driver_receipt.json'), 'w'))
print('FAKE_DRIVER bprobs', flush=True)
'''

FAKE_SLEEP = r'''
import time
print('FAKE_DRIVER sleeping', flush=True)
time.sleep(600)
'''

FAKE_FAIL = r'''
import os, json, sys
os.makedirs(%(out)r, exist_ok=True)
json.dump({'ok': False, 'error': "K1Error('component-B stack probe failed (test)')"}, open(os.path.join(%(out)r, 's6f_driver_receipt.json'), 'w'))
sys.exit(3)
'''

WRAP = r'''
import sys, subprocess
sys.exit(subprocess.call([sys.executable, '-I', %(inner)r] + sys.argv[1:]))
'''


def install_fake(tree, mode, **kw):
    """replace the driver the setup cell wrote (it must exist: written + sha-checked by the real setup cell)."""
    drv = os.path.join(tree.root, 's6f_driver.py')
    assert os.path.isfile(drv), 'setup cell did not write the driver'
    out = tree.fw(os.path.join(tree.work, 's6f'))
    tmp = tree.fw(os.path.join(tree.root, 'work'))
    if mode == 'port':
        inner = os.path.join(tree.t, 'fake_port.py')
        open(inner, 'w').write(FAKE_PORT % dict(driver=DRIVER_BUILT, ref=REF, data=DATA, out=out, tmp=tmp, mut=kw.get('mut')))
        open(drv, 'w').write(WRAP % dict(inner=inner))
    elif mode == 'bprobs':
        inner = os.path.join(tree.t, 'fake_bprobs.py')
        open(inner, 'w').write(FAKE_BPROBS % dict(out=out, bprobs=kw['bprobs'], tamper=bool(kw.get('tamper'))))
        open(drv, 'w').write(WRAP % dict(inner=inner))
    elif mode == 'sleep':
        open(drv, 'w').write(FAKE_SLEEP)
    elif mode == 'fail':
        open(drv, 'w').write(FAKE_FAIL % dict(out=out))
    else:
        raise ValueError(mode)


def run_cells(tree, base_df, age_h=4.6, fired=(), mode='port', overrides=None, setup_hook=None, base_file=None, **kw):
    """write the base submission (base_file: copy those bytes, e.g. the real V224-FLIP commit file), execute SETUP, swap in the
    fake driver, execute RUN and BLEND. Returns (ns, printed text)."""
    setup_t, run_t, blend_t = cells()
    for a, b in (overrides or {}).items():
        assert run_t.count(a) == 1 or blend_t.count(a) == 1, a
        run_t, blend_t = run_t.replace(a, b), blend_t.replace(a, b)
    if base_file:
        shutil.copyfile(base_file, os.path.join(tree.work, 'submission.csv'))
    else:
        base_df.to_csv(os.path.join(tree.work, 'submission.csv'), index=False, lineterminator='\n')
    ns = {'__name__': '__main__', 'NB_T0': time.time() - age_h * 3600.0, 'sub': base_df.copy(),
          '_canary': {'r224_used': 'r224->Fracture' not in fired, 'fired': list(fired)}}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(tree.redirect(setup_t), '<s6f setup>', 'exec'), ns)
    if setup_hook:
        setup_hook(tree, ns)
    if mode is not None and os.path.isfile(os.path.join(tree.root, 's6f_driver.py')):
        install_fake(tree, mode, **kw)
    with contextlib.redirect_stdout(buf):
        exec(compile(tree.redirect(run_t), '<s6f run>', 'exec'), ns)
        exec(compile(tree.redirect(blend_t), '<s6f blend>', 'exec'), ns)
    return ns, buf.getvalue()


def avg_rank_pct(v):
    """independent average-tie rank / n (as v224_check.py)."""
    v = np.asarray(v, dtype=np.float64)
    order = np.argsort(v, kind='mergesort')
    sv = v[order]
    r = np.empty(len(v), dtype=np.float64)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r / len(v)


def recompute(base_df, bprobs_df, w=0.2, blank=()):
    ids = base_df['StudyInstanceUID'].astype(str).tolist()
    S = bprobs_df.set_index('StudyInstanceUID').loc[ids, LAB].to_numpy(np.float64)
    ok = np.isfinite(S).all(axis=1)
    out = base_df.copy()
    for j, c in enumerate(LAB):
        rb = avg_rank_pct(base_df[c].to_numpy(np.float64))
        rs = rb.copy()
        rs[ok] = avg_rank_pct(S[ok, j])
        out[c] = ((1 - w) * rb + w * rs).astype(np.float32)
    for c in blank:
        out[c] = np.float32(0.5)
    return out


def flip_base():
    return pd.read_csv(os.path.join(FLIP_OUT, 'submission.csv'), dtype={'StudyInstanceUID': str})


def ref_bprobs():
    return pd.read_csv(os.path.join(REF, 'k1', 'B', 'B_probs.csv'), dtype={'StudyInstanceUID': str})
